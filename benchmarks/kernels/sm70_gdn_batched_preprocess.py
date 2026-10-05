# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research only: batch convolution preparation, preserve sequential delta.

Head-local completion handshakes require all CTAs to be resident. This is not a
production dispatch and must be launched cooperatively by the microbenchmark.
"""

from vllm.model_executor.layers.fla.ops.op import exp
from vllm.triton_utils import tl, triton


@triton.jit
def conv_panel(x, history, w, features, accepted, WIDTH: tl.constexpr):
    token = tl.arange(0, 8)[:, None]
    features = features[None, :]
    acc = tl.full((8, WIDTH), 0, tl.float32)
    for j in tl.static_range(4):
        prior = token + j - 3
        val = tl.load(x + tl.maximum(prior, 0) * 4096 + features, prior >= 0, 0)
        old = tl.load(
            history + features + 2560 * (accepted - 1 + tl.maximum(prior + 3, 0)),
            prior < 0,
            0,
        )
        weight = tl.load(w + features * 4 + j)
        # Match the trained runtime's half product, then FP32 accumulation.
        # Its PTX explicitly uses mul.f16 followed by cvt.f32.f16/add.f32.
        acc += (tl.where(prior < 0, old, val) * weight).to(tl.float32)
    return (acc / (1 + tl.exp(-acc))).to(tl.float16).to(tl.float32)


@triton.jit
def write_history(x, history, features, accepted):
    # Each physical channel has one owner. Its own initial reads finished.
    for j in tl.range(0, 10, loop_unroll_factor=1, disable_licm=True):
        prior = j - 2
        old = tl.load(history + features + 2560 * (accepted + j), prior < 0, 0)
        value = tl.load(x + tl.maximum(prior, 0) * 4096 + features, prior >= 0, 0)
        tl.store(history + features + 2560 * j, tl.where(prior < 0, old, value))


@triton.jit
def await_epoch(pointer, epoch):
    return tl.inline_asm_elementwise(
        """{ .reg .u64 seen; .reg .pred pending;
        WAIT: ld.relaxed.gpu.global.u64 seen, [$1];
        setp.lt.u64 pending, seen, $2;
        @pending bra WAIT;
        fence.acq_rel.gpu;
        mov.u32 $0, 0;
        }""",
        constraints="=r,l,l,~{memory}",
        args=[pointer, epoch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def release_epoch(pointer, epoch):
    return tl.inline_asm_elementwise(
        """{ .reg .u32 tid; .reg .pred active;
        mov.u32 tid, %tid.x;
        setp.eq.u32 active, tid, 0;
        @active st.release.gpu.global.u64 [$1], $2;
        mov.u32 $0, 0; }""",
        constraints="=r,l,l,~{memory}",
        args=[pointer, epoch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def await_scalar_epoch(pointer, epoch):
    # One CTA thread polls. All threads rendezvous before their acquire fence.
    return tl.inline_asm_elementwise(
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
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def batched_preprocess_kernel(
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
    EPS: tl.constexpr,
    BV: tl.constexpr,
):
    tile = tl.program_id(0)
    h = tl.program_id(1)
    count: tl.constexpr = 128 // BV
    pid = h * count + tile
    accepted = tl.load(accepted_ptr).to(tl.int64)
    epoch = tl.load(done + pid, cache_modifier=".cg") + 1
    kdim = tl.arange(0, 128)
    vdim = tile * BV + tl.arange(0, BV)
    qfeature = h // 3 * 128 + kdim
    kfeature = 512 + h // 3 * 128 + kdim
    vfeature = 1024 + h * 128 + vdim
    index = tl.load(indices + accepted - 1).to(tl.int64)
    matrix = tl.load(
        state + index * 196608 + h * 16384 + vdim[:, None] * 128 + kdim[None, :]
    )
    qs = conv_panel(x, history, w, qfeature, accepted, 128)
    ks = conv_panel(x, history, w, kfeature, accepted, 128)
    vs = conv_panel(x, history, w, vfeature, accepted, BV)
    qs = qs / tl.sqrt(tl.sum(qs * qs, 1)[:, None] + 1e-6)
    qs = qs * 0.08838834764831845
    ks = ks / tl.sqrt(tl.sum(ks * ks, 1)[:, None] + 1e-6)
    for t in range(8):
        qi = tl.full((1, 128), t, tl.int32)
        vi = tl.full((1, BV), t, tl.int32)
        q = tl.gather(qs, qi, 0).reshape((128,))
        k = tl.gather(ks, qi, 0).reshape((128,))
        v = tl.gather(vs, vi, 0).reshape((BV,))
        a = tl.load(ba + t * 24 + 12 + h).to(tl.float32) + tl.load(bias + h).to(
            tl.float32
        )
        soft = tl.where(a <= 20, tl.log(1 + tl.exp(a)), a)
        g = -tl.exp(tl.load(a_log + h).to(tl.float32)) * soft
        beta = tl.sigmoid(tl.load(ba + t * 24 + h).to(tl.float32))
        matrix *= exp(g)
        v = (v - tl.sum(matrix * k[None, :], 1)) * beta
        matrix += v[:, None] * k[None, :]
        y = tl.sum(matrix * q[None, :], 1)
        tl.store(raw + t * 1536 + h * 128 + vdim, y)
        slot = tl.load(indices + t).to(tl.int64)
        tl.store(
            state + slot * 196608 + h * 16384 + vdim[:, None] * 128 + kdim[None, :],
            matrix,
        )
    write_history(x, history, vfeature, accepted)
    # All snapshots, raw outputs and this CTA's initial q/k reads precede done.
    # Publish only after every CTA thread has completed its stores.
    tl.debug_barrier()
    release_epoch(done + pid, epoch)
    if tile == 0:
        headtiles = h * count + tl.arange(0, count)
        await_epoch(done + headtiles, epoch)
        tl.debug_barrier()
        d = tl.arange(0, 128)
        for norm_token in tl.static_range(8):
            head_y = tl.load(
                raw + norm_token * 1536 + h * 128 + d, cache_modifier=".cg"
            ).to(tl.float32)
            head_z = tl.load(x + norm_token * 4096 + 2560 + h * 128 + d).to(tl.float32)
            variance = tl.sum(head_y * head_y) / 128
            value = head_y * tl.rsqrt(variance + EPS) * tl.load(norm + d).to(tl.float32)
            value *= head_z * tl.sigmoid(head_z)
            tl.store(out + norm_token * 1536 + h * 128 + d, value)
        if h % 3 == 0:
            # q/k history is shared by three value heads. Do not overwrite it
            # until every tile in those heads has completed its initial reads.
            group = h * count + tl.arange(0, triton.next_power_of_2(3 * count))
            valid = tl.arange(0, triton.next_power_of_2(3 * count)) < 3 * count
            await_epoch(tl.where(valid, done + group, done + pid), epoch)
            tl.debug_barrier()
            write_history(x, history, qfeature, accepted)
            write_history(x, history, kfeature, accepted)
