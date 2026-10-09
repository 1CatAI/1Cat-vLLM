# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research only: fixed original K reduction layout and sequential delta.

Head-local completion handshakes require all CTAs to be resident. This is not a
production dispatch and must be launched cooperatively by the microbenchmark.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from vllm.triton_utils import triton


@gluon.jit
def cta_barrier():
    gl.inline_asm_elementwise(
        "bar.sync 0; mov.u32 $0, 0;",
        constraints="=r",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def sigmoid(x):
    return 1 / (1 + gl.exp(-x))


@gluon.jit
def conv_panel(
    x, history, w, features, accepted, WIDTH: gl.constexpr, Layout: gl.constexpr
):
    token = gl.arange(0, 8, layout=gl.SliceLayout(1, Layout))[:, None]
    features = features[None, :]
    acc = gl.full((8, WIDTH), 0, gl.float32, layout=Layout)
    for j in gl.static_range(4):
        prior = token + j - 3
        val = gl.load(x + gl.maximum(prior, 0) * 4096 + features, prior >= 0, 0)
        old = gl.load(
            history + features + 2560 * (accepted - 1 + gl.maximum(prior + 3, 0)),
            prior < 0,
            0,
        )
        weight = gl.load(w + features * 4 + j)
        # Match the trained runtime's half product, then FP32 accumulation.
        # Its PTX explicitly uses mul.f16 followed by cvt.f32.f16/add.f32.
        acc += (gl.where(prior < 0, old, val) * weight).to(gl.float32)
    return (acc / (1 + gl.exp(-acc))).to(gl.float16).to(gl.float32)


@gluon.jit
def write_history(x, history, features, accepted):
    # Each physical channel has one owner. Its own initial reads finished.
    for j in gl.static_range(10):
        prior = j - 2
        old = gl.load(history + features + 2560 * (accepted + j), prior < 0, 0)
        value = gl.load(x + gl.maximum(prior, 0) * 4096 + features, prior >= 0, 0)
        gl.store(history + features + 2560 * j, gl.where(prior < 0, old, value))


@gluon.jit
def await_epoch(pointer, epoch):
    return gl.inline_asm_elementwise(
        """{ .reg .u64 seen; .reg .pred pending;
        WAIT: ld.relaxed.gpu.global.u64 seen, [$1];
        setp.lt.u64 pending, seen, $2;
        @pending bra WAIT;
        fence.acq_rel.gpu;
        mov.u32 $0, 0;
        }""",
        constraints="=r,l,l,~{memory}",
        args=[pointer, epoch],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def release_epoch(pointer, epoch):
    return gl.inline_asm_elementwise(
        """{ .reg .u32 tid; .reg .pred active;
        mov.u32 tid, %tid.x;
        setp.eq.u32 active, tid, 0;
        @active st.release.gpu.global.u64 [$1], $2;
        mov.u32 $0, 0; }""",
        constraints="=r,l,l,~{memory}",
        args=[pointer, epoch],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def await_scalar_epoch(pointer, epoch):
    # One CTA thread polls. All threads rendezvous before their acquire fence.
    return gl.inline_asm_elementwise(
        """{ .reg .u64 seen; .reg .u32 tid; .reg .pred pending, inactive;
        mov.u32 tid, %tid.x;
        setp.ne.u32 inactive, tid, 0;
        @inactive bra READY;
        WAIT: ld.relaxed.gpu.global.u64 seen, [$1];
        setp.lt.u64 pending, seen, $2;
        @pending bra WAIT;
        READY: bar.sync 0;
        fence.acq_rel.gpu;
        mov.u32 $0, 0;
        }""",
        constraints="=r,l,l,~{memory}",
        args=[pointer, epoch],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def fixed_layout_batched_preprocess_kernel(
    x,
    ba,
    history,
    w,
    a_log,
    bias,
    norm,
    state,
    indices,
    raw,
    out,
    done,
    accepted_ptr,
    EPS: gl.constexpr,
    BV: gl.constexpr,
    Warps: gl.constexpr,
):
    tile = gl.program_id(0)
    h = gl.program_id(1)
    count: gl.constexpr = 128 // BV
    pid = h * count + tile
    accepted = gl.load(accepted_ptr).to(gl.int64)
    epoch = gl.load(done + pid, cache_modifier=".cg") + 1
    StateLayout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [Warps, 1], [1, 0])
    KLayout: gl.constexpr = gl.SliceLayout(0, StateLayout)
    VLayout: gl.constexpr = gl.SliceLayout(1, StateLayout)
    QLayout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, Warps], [1, 0])
    VPLayout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, Warps], [0, 1])
    kdim = gl.arange(0, 128, layout=KLayout)
    vdim = tile * BV + gl.arange(0, BV, layout=VLayout)
    qfeature = h // 3 * 128 + kdim
    kfeature = 512 + h // 3 * 128 + kdim
    vfeature = 1024 + h * 128 + vdim
    index = gl.load(indices + accepted - 1).to(gl.int64)
    matrix = gl.load(
        state + index * 196608 + h * 16384 + vdim[:, None] * 128 + kdim[None, :]
    )
    qs = conv_panel(
        x,
        history,
        w,
        gl.convert_layout(qfeature, gl.SliceLayout(0, QLayout)),
        accepted,
        128,
        QLayout,
    )
    ks = conv_panel(
        x,
        history,
        w,
        gl.convert_layout(kfeature, gl.SliceLayout(0, QLayout)),
        accepted,
        128,
        QLayout,
    )
    vs = conv_panel(
        x,
        history,
        w,
        gl.convert_layout(vfeature, gl.SliceLayout(0, VPLayout)),
        accepted,
        BV,
        VPLayout,
    )
    qs = qs / gl.sqrt(gl.sum(qs * qs, 1)[:, None] + 1e-6)
    qs = qs * 0.08838834764831845
    ks = ks / gl.sqrt(gl.sum(ks * ks, 1)[:, None] + 1e-6)
    for t in gl.static_range(8):
        qi = gl.full((1, 128), t, gl.int32, layout=QLayout)
        vi = gl.full((1, BV), t, gl.int32, layout=VPLayout)
        q = gl.convert_layout(
            gl.gather(qs, qi, 0).reshape((128,)), KLayout, assert_trivial=True
        )
        k = gl.convert_layout(
            gl.gather(ks, qi, 0).reshape((128,)), KLayout, assert_trivial=True
        )
        v = gl.convert_layout(
            gl.gather(vs, vi, 0).reshape((BV,)), VLayout, assert_trivial=True
        )
        a = gl.load(ba + t * 24 + 12 + h).to(gl.float32) + gl.load(bias + h).to(
            gl.float32
        )
        soft = gl.where(a <= 20, gl.log(1 + gl.exp(a)), a)
        g = -gl.exp(gl.load(a_log + h).to(gl.float32)) * soft
        beta = sigmoid(gl.load(ba + t * 24 + h).to(gl.float32))
        matrix *= gl.exp(g)
        v = (v - gl.sum(matrix * k[None, :], 1)) * beta
        matrix += v[:, None] * k[None, :]
        y = gl.sum(matrix * q[None, :], 1)
        gl.store(raw + t * 1536 + h * 128 + vdim, y)
        slot = gl.load(indices + t).to(gl.int64)
        gl.store(
            state + slot * 196608 + h * 16384 + vdim[:, None] * 128 + kdim[None, :],
            matrix,
        )
    write_history(x, history, vfeature, accepted)
    # All snapshots, raw outputs and this CTA's initial q/k reads precede done.
    # Publish only after every CTA thread has completed its stores.
    cta_barrier()
    release_epoch(done + pid, epoch)
    if tile == 0:
        headtiles = h * count + gl.arange(
            0, count, layout=gl.BlockedLayout([1], [32], [Warps], [0])
        )
        await_epoch(done + headtiles, epoch)
        cta_barrier()
        d = gl.arange(0, 128, layout=KLayout)
        for norm_token in gl.static_range(8):
            head_y = gl.load(
                raw + norm_token * 1536 + h * 128 + d, cache_modifier=".cg"
            ).to(gl.float32)
            head_z = gl.load(x + norm_token * 4096 + 2560 + h * 128 + d).to(gl.float32)
            variance = gl.sum(head_y * head_y) / 128
            value = head_y * gl.rsqrt(variance + EPS) * gl.load(norm + d).to(gl.float32)
            value *= head_z * sigmoid(head_z)
            gl.store(out + norm_token * 1536 + h * 128 + d, value)
        if h % 3 == 0:
            # q/k history is shared by three value heads. Do not overwrite it
            # until every tile in those heads has completed its initial reads.
            group = h * count + gl.arange(
                0,
                triton.next_power_of_2(3 * count),
                layout=gl.BlockedLayout([1], [32], [Warps], [0]),
            )
            valid = (
                gl.arange(
                    0,
                    triton.next_power_of_2(3 * count),
                    layout=gl.BlockedLayout([1], [32], [Warps], [0]),
                )
                < 3 * count
            )
            await_epoch(gl.where(valid, done + group, done + pid), epoch)
            cta_barrier()
            write_history(x, history, qfeature, accepted)
            write_history(x, history, kfeature, accepted)


@gluon.jit
def head_prepared_fixed_layout_kernel(
    x,
    ba,
    history,
    w,
    a_log,
    bias,
    norm,
    state,
    indices,
    raw,
    out,
    done,
    accepted_ptr,
    prepared,
    EPS: gl.constexpr,
    BV: gl.constexpr,
    Warps: gl.constexpr,
):
    tile = gl.program_id(0)
    h = gl.program_id(1)
    count: gl.constexpr = 128 // BV
    pid = h * count + tile
    accepted = gl.load(accepted_ptr).to(gl.int64)
    epoch = gl.load(done + pid, cache_modifier=".cg") + 1
    StateLayout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [Warps, 1], [1, 0])
    KLayout: gl.constexpr = gl.SliceLayout(0, StateLayout)
    VLayout: gl.constexpr = gl.SliceLayout(1, StateLayout)
    QLayout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, Warps], [1, 0])
    kdim = gl.arange(0, 128, layout=KLayout)
    vdim = tile * BV + gl.arange(0, BV, layout=VLayout)
    vfeature = 1024 + h * 128 + vdim
    index = gl.load(indices + accepted - 1).to(gl.int64)
    prepared_head = prepared + h * 8 * 386
    ready = done + 12 * count + h
    if tile == 0:
        qcols = gl.arange(0, 128, layout=gl.SliceLayout(0, QLayout))
        qs = conv_panel(x, history, w, h // 3 * 128 + qcols, accepted, 128, QLayout)
        ks = conv_panel(
            x, history, w, 512 + h // 3 * 128 + qcols, accepted, 128, QLayout
        )
        vs = conv_panel(x, history, w, 1024 + h * 128 + qcols, accepted, 128, QLayout)
        qs = qs / gl.sqrt(gl.sum(qs * qs, 1)[:, None] + 1e-6)
        qs = qs * 0.08838834764831845
        ks = ks / gl.sqrt(gl.sum(ks * ks, 1)[:, None] + 1e-6)
        tokens = gl.arange(0, 8, layout=gl.SliceLayout(1, QLayout))
        address = tokens[:, None] * 386 + qcols[None, :]
        gl.store(prepared_head + address, qs)
        gl.store(prepared_head + address + 128, ks)
        gl.store(prepared_head + address + 256, vs)
        a = gl.load(ba + tokens * 24 + 12 + h).to(gl.float32)
        a += gl.load(bias + h).to(gl.float32)
        soft = gl.where(a <= 20, gl.log(1 + gl.exp(a)), a)
        g = -gl.exp(gl.load(a_log + h).to(gl.float32)) * soft
        beta = sigmoid(gl.load(ba + tokens * 24 + h).to(gl.float32))
        gl.store(prepared_head + tokens * 386 + 384, g)
        gl.store(prepared_head + tokens * 386 + 385, beta)
        cta_barrier()
        release_epoch(ready, epoch)
    await_scalar_epoch(ready, epoch)
    matrix = gl.load(
        state + index * 196608 + h * 16384 + vdim[:, None] * 128 + kdim[None, :]
    )
    for t in gl.static_range(8):
        q = gl.load(prepared_head + t * 386 + kdim, cache_modifier=".cg")
        k = gl.load(prepared_head + t * 386 + 128 + kdim, cache_modifier=".cg")
        v = gl.load(prepared_head + t * 386 + 256 + vdim, cache_modifier=".cg")
        g = gl.load(prepared_head + t * 386 + 384, cache_modifier=".cg")
        beta = gl.load(prepared_head + t * 386 + 385, cache_modifier=".cg")
        matrix *= gl.exp(g)
        v = (v - gl.sum(matrix * k[None, :], 1)) * beta
        matrix += v[:, None] * k[None, :]
        y = gl.sum(matrix * q[None, :], 1)
        gl.store(raw + t * 1536 + h * 128 + vdim, y)
        slot = gl.load(indices + t).to(gl.int64)
        gl.store(
            state + slot * 196608 + h * 16384 + vdim[:, None] * 128 + kdim[None, :],
            matrix,
        )
    write_history(x, history, vfeature, accepted)
    # All snapshots, raw outputs and this CTA's initial q/k reads precede done.
    # Publish only after every CTA thread has completed its stores.
    cta_barrier()
    release_epoch(done + pid, epoch)
    if tile == 0:
        headtiles = h * count + gl.arange(
            0, count, layout=gl.BlockedLayout([1], [32], [Warps], [0])
        )
        await_epoch(done + headtiles, epoch)
        cta_barrier()
        d = gl.arange(0, 128, layout=KLayout)
        for norm_token in gl.static_range(8):
            head_y = gl.load(
                raw + norm_token * 1536 + h * 128 + d, cache_modifier=".cg"
            ).to(gl.float32)
            head_z = gl.load(x + norm_token * 4096 + 2560 + h * 128 + d).to(gl.float32)
            variance = gl.sum(head_y * head_y) / 128
            value = head_y * gl.rsqrt(variance + EPS) * gl.load(norm + d).to(gl.float32)
            value *= head_z * sigmoid(head_z)
            gl.store(out + norm_token * 1536 + h * 128 + d, value)
        if h % 3 == 0:
            # q/k history is shared by three value heads. Do not overwrite it
            # until every tile in those heads has completed its initial reads.
            group = h * count + gl.arange(
                0,
                triton.next_power_of_2(3 * count),
                layout=gl.BlockedLayout([1], [32], [Warps], [0]),
            )
            valid = (
                gl.arange(
                    0,
                    triton.next_power_of_2(3 * count),
                    layout=gl.BlockedLayout([1], [32], [Warps], [0]),
                )
                < 3 * count
            )
            await_epoch(gl.where(valid, done + group, done + pid), epoch)
            cta_barrier()
            write_history(x, history, h // 3 * 128 + kdim, accepted)
            write_history(x, history, 512 + h // 3 * 128 + kdim, accepted)
