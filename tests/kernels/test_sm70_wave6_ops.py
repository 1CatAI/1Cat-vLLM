# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 Wave 6 concurrency & single-request kernel parity and dispatch tests."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
from vllm.models.qwen4_exp.nvidia import sm70_hcx as hcx
from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention


@contextmanager
def _matmul_flags(reduced_precision: bool, fp16_accum: bool):
    backend = torch.backends.cuda.matmul
    old_red = backend.allow_fp16_reduced_precision_reduction
    old_acc = backend.allow_fp16_accumulation
    backend.allow_fp16_reduced_precision_reduction = reduced_precision
    backend.allow_fp16_accumulation = fp16_accum
    try:
        yield
    finally:
        backend.allow_fp16_reduced_precision_reduction = old_red
        backend.allow_fp16_accumulation = old_acc


@pytest.mark.parametrize("n", (640, 3584))
def test_qsa_input_packing_preserves_half_bits(n: int):
    raw = torch.randint(-(2**15), 2**15, (n, 2560), dtype=torch.int16)
    packed = gemv._pack_qsa_input_weight(raw.view(torch.float16))
    restored = packed.permute(0, 3, 1, 2, 4).contiguous().view(n, 2560)
    assert torch.equal(restored.view(torch.int16), raw)


def test_qsa_input_bad_pack_rejected():
    with pytest.raises(ValueError):
        gemv._pack_qsa_input_weight(torch.empty(641, 2560, dtype=torch.float16))
    with pytest.raises(ValueError):
        gemv._pack_qsa_input_weight(torch.empty(640, 2560, dtype=torch.float32))


def test_mtp_active_context_admission(monkeypatch: pytest.MonkeyPatch):
    import vllm.forward_context as fc

    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_QWEN38_BATCH_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(
        gemv.current_platform,
        "is_device_capability",
        lambda cap, device_id=None: cap == 70,
    )
    with _matmul_flags(reduced_precision=True, fp16_accum=False):
        desc_x5 = SimpleNamespace(
            ndim=2,
            shape=(5, 1536),
            dtype=torch.float16,
            is_cuda=True,
            device=torch.device("cuda:0"),
            stride=lambda: (1536, 1),
            data_ptr=lambda: 16,
        )
        desc_x16 = SimpleNamespace(
            ndim=2,
            shape=(16, 1536),
            dtype=torch.float16,
            is_cuda=True,
            device=torch.device("cuda:0"),
            stride=lambda: (1536, 1),
            data_ptr=lambda: 16,
        )
        desc_x40 = SimpleNamespace(
            ndim=2,
            shape=(40, 1536),
            dtype=torch.float16,
            is_cuda=True,
            device=torch.device("cuda:0"),
            stride=lambda: (1536, 1),
            data_ptr=lambda: 16,
        )
        desc_w = SimpleNamespace(
            ndim=2,
            shape=(2560, 1536),
            dtype=torch.float16,
            is_cuda=True,
            device=torch.device("cuda:0"),
            stride=lambda: (1536, 1),
            data_ptr=lambda: 16,
        )
        assert not gemv._mtp_active()
        assert not gemv._can_use_dense_batch(
            desc_x5, desc_w, "layers.0.linear_attn.out_proj"
        )
        assert not gemv._can_use_dense_batch(
            desc_x40, desc_w, "layers.0.linear_attn.out_proj"
        )

        fake_ctx = SimpleNamespace(
            runtime_resources={
                "execution_policies": {"speculative_config.sampling_policy": object()}
            }
        )
        monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(fc, "get_forward_context", lambda: fake_ctx)
        assert gemv._mtp_active()
        assert gemv._can_use_dense_batch(
            desc_x5, desc_w, "layers.0.linear_attn.out_proj"
        )
        assert gemv._can_use_dense_batch(
            desc_x16, desc_w, "layers.0.linear_attn.out_proj"
        )
        assert not gemv._can_use_dense_batch(
            desc_x40, desc_w, "layers.0.linear_attn.out_proj"
        )
    with _matmul_flags(reduced_precision=False, fp16_accum=False):
        assert gemv._can_use_dense_batch(
            desc_x16, desc_w, "layers.0.linear_attn.out_proj"
        )
        assert not gemv._can_use_dense_batch(
            desc_x40, desc_w, "layers.0.linear_attn.out_proj"
        )
    envs.disable_envs_cache()


def _require_sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")


@pytest.mark.parametrize("rows", (1, 2, 5, 10, 20, 40, 64))
@pytest.mark.parametrize("row_major", (False, True))
def test_qsa_input_batch_and_m1_cudagraph_parity(
    rows: int, row_major: bool, monkeypatch: pytest.MonkeyPatch
):
    _require_sm70()
    if not hasattr(torch.ops._C, "qwen38_qsa_input_batch_sm70_out"):
        pytest.skip("qwen38_qsa_input_batch_sm70_out unavailable")
    monkeypatch.setenv("VLLM_SM70_QWEN38_BATCH_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(gemv, "_mtp_active", lambda: True)
    envs.disable_envs_cache()

    torch.manual_seed(6100 + rows * 3 + int(row_major))
    dev = torch.device("cuda")
    qw = torch.randn(3584, 2560, device=dev, dtype=torch.float16) * 0.02
    iw = torch.randn(640, 2560, device=dev, dtype=torch.float16) * 0.02
    pq = gemv._pack_qsa_input_weight(qw)
    pi = gemv._pack_qsa_input_weight(iw)
    x = torch.randn(rows, 2560, device=dev, dtype=torch.float16)

    if rows == 1:
        q_ref = gemv._qwen38_sm70_fp16_gemv(x, qw, "layers.0.self_attn.qkv_proj")
        i_ref = gemv._qwen38_sm70_fp16_gemv(
            x, iw, "layers.0.self_attn.indexer.index_qk_proj"
        )
        q_act, i_act = torch.ops.vllm.qwen38_sm70_fp16_qsa_input(x, qw, iw, pq, pi)
        assert torch.equal(q_act.view(torch.int16), q_ref.view(torch.int16))
        assert torch.equal(i_act.view(torch.int16), i_ref.view(torch.int16))
        return

    s_q = torch.full((rows * 3584 + 16,), 19, device=dev, dtype=torch.float16)
    s_i = torch.full((rows * 640 + 16,), 23, device=dev, dtype=torch.float16)
    out_q = s_q[8:-8].view(rows, 3584)
    out_i = s_i[8:-8].view(rows, 640)
    w_q, w_i = (qw, iw) if row_major else (pq, pi)

    with _matmul_flags(reduced_precision=(rows <= 16), fp16_accum=False):
        torch.ops._C.qwen38_qsa_input_batch_sm70_out(out_q, out_i, x, w_q, w_i)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.ops._C.qwen38_qsa_input_batch_sm70_out(out_q, out_i, x, w_q, w_i)
            op_q, op_i = torch.ops.vllm.qwen38_sm70_fp16_qsa_input(x, qw, iw, pq, pi)

        x.normal_(0, 0.7)
        out_q.fill_(float("nan"))
        out_i.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert torch.all(s_q[:8] == 19) and torch.all(s_q[-8:] == 19)
        assert torch.all(s_i[:8] == 23) and torch.all(s_i[-8:] == 23)
        if rows <= 16:
            assert torch.equal(out_q.view(torch.int16), op_q.view(torch.int16))
            assert torch.equal(out_i.view(torch.int16), op_i.view(torch.int16))
        else:
            q_lin = F.linear(x, qw)
            i_lin = F.linear(x, iw)
            assert torch.equal(op_q.view(torch.int16), q_lin.view(torch.int16))
            assert torch.equal(op_i.view(torch.int16), i_lin.view(torch.int16))
    q_ref = (x.float() @ qw.float().T).half()
    i_ref = (x.float() @ iw.float().T).half()
    torch.testing.assert_close(out_q, q_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_i, i_ref, rtol=1e-3, atol=1e-3)
    envs.disable_envs_cache()


@pytest.mark.parametrize("rows", (5, 10, 20, 40, 64))
def test_multigroup_gdn_dense_router_shared_down_and_hcx_out_cudagraph(
    rows: int, monkeypatch: pytest.MonkeyPatch
):
    _require_sm70()
    monkeypatch.setattr(gemv, "_mtp_active", lambda: True)
    monkeypatch.setenv("VLLM_SM70_QWEN38_BATCH_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()

    with _matmul_flags(reduced_precision=False, fp16_accum=False):
        torch.manual_seed(6200 + rows)
        dev = torch.device("cuda")
        x = torch.randn(rows, 2560, device=dev, dtype=torch.float16)
        qw = torch.randn(4096, 2560, device=dev, dtype=torch.float16) * 0.02
        bw = torch.randn(24, 2560, device=dev, dtype=torch.float16) * 0.02
        pq = gemv._pack_gdn_input_weight(qw)
        pb = gemv._pack_gdn_input_weight(bw)

        qkv, z, b, a = (x.new_empty((rows, n)) for n in (2560, 1536, 12, 12))
        torch.ops._C.qwen38_gdn_input_batch_sm70_out(qkv, z, b, a, x, pq, pb)
        qkvz_ref = F.linear(x, qw)
        ba_ref = torch.cat(
            [F.linear(x[s : s + 8], bw) for s in range(0, rows, 8)], dim=0
        )
        assert torch.equal(
            qkv.view(torch.int16), qkvz_ref[:, :2560].contiguous().view(torch.int16)
        )
        assert torch.equal(
            z.view(torch.int16), qkvz_ref[:, 2560:].contiguous().view(torch.int16)
        )
        assert torch.equal(
            b.view(torch.int16), ba_ref[:, :12].contiguous().view(torch.int16)
        )
        assert torch.equal(
            a.view(torch.int16), ba_ref[:, 12:].contiguous().view(torch.int16)
        )

        x_out = torch.randn(rows, 1536, device=dev, dtype=torch.float16)
        w_out = torch.randn(2560, 1536, device=dev, dtype=torch.float16) * 0.02
        out_buf = torch.empty(rows, 2560, device=dev, dtype=torch.float16)
        torch.ops._C.qwen38_dense_batch_sm70_out(out_buf, x_out, w_out)
        out_ref = torch.cat(
            [F.linear(x_out[s : s + 8], w_out) for s in range(0, rows, 8)], dim=0
        )
        assert torch.equal(out_buf.view(torch.int16), out_ref.view(torch.int16))

        fake_layer = SimpleNamespace(
            weight=w_out,
            bias=None,
            reduce_results=False,
            output_size=2560,
            prefix="model.layers.0.linear_attn.out_proj",
            quant_method=gemv.Qwen38SM70FP16LinearMethod(),
            _sm70_qwen38_dense_batch=True,
        )
        hcx.register_output_projection("_test_w6_out", fake_layer, defer=False)
        monkeypatch.setattr(hcx, "can_defer_ll_allreduce", lambda r: True)
        import vllm.compilation.sm70_decode_graph as dg

        monkeypatch.setattr(dg, "use_sm70_decode_graph_semantics", lambda: True)
        dst = torch.full_like(out_buf, -7.0)
        torch.ops.vllm.qwen38_sm70_hcx_output_projection_out(dst, x_out, "_test_w6_out")
        expected_dst = out_ref if rows <= 16 else F.linear(x_out, w_out)
        assert torch.equal(dst.view(torch.int16), expected_dst.view(torch.int16))

        rw = torch.randn(512, 2560, device=dev, dtype=torch.float16) * 0.02
        rpack = gemv._pack_router_batch_weight(rw)
        rout = torch.empty(rows, 512, device=dev, dtype=torch.float16)
        torch.ops._C.qwen38_router_batch_sm70_out(rout, x, rpack)
        rref = torch.cat([F.linear(x[s : s + 8], rw) for s in range(0, rows, 8)], dim=0)
        assert torch.equal(rout.view(torch.int16), rref.view(torch.int16))

        inter = torch.randn(rows, 160, device=dev, dtype=torch.float16)
        dw = torch.randn(2560, 160, device=dev, dtype=torch.float16) * 0.02
        gw = torch.randn(1, 2560, device=dev, dtype=torch.float16) * 0.02
        sd_act = torch.ops.vllm.qwen38_sm70_shared_down_gated(x, inter, dw, gw)
        sd_ref = torch.sigmoid(x.float() @ gw.float().T) * (
            inter.float() @ dw.float().T
        )
        torch.testing.assert_close(sd_act, sd_ref.half(), rtol=2e-3, atol=2e-3)
    envs.disable_envs_cache()


@pytest.mark.parametrize("encoded", (False, True))
def test_qsa_hot_concurrent_high_occupancy_and_sibling_rows(encoded: bool):
    _require_sm70()
    torch.manual_seed(6300 + int(encoded))
    dev = torch.device("cuda")
    num_kv_tokens = 512
    hot_tokens = 256  # 64 4-token page slots -> ~90% occupancy under 58 active pages
    rows = 40
    kv_hot = HostQSAKV(
        8,
        64,
        256,
        dev,
        width=64,
        rows=rows,
        resolve_rows=rows,
        hot_tokens=hot_tokens,
        direct_device=True,
        hot_fp8=encoded,
    )
    kv_ref = HostQSAKV(
        8,
        64,
        256,
        dev,
        width=64,
        rows=rows,
        resolve_rows=rows,
        hot_tokens=64,
        device_reference=True,
        direct_device=True,
    )
    assert kv_hot.device_history_hot is True

    k = torch.randn(num_kv_tokens, 1, 256, device=dev, dtype=torch.float16) * 0.25
    v = torch.randn(num_kv_tokens, 1, 256, device=dev, dtype=torch.float16) * 0.25
    slots = torch.arange(num_kv_tokens, device=dev)
    kv_hot.write(k, v, slots)
    kv_ref.write(k, v, slots)

    # Concurrent c=8 MTP verify (M=40: 8 requests x 5 sibling rows sharing pages)
    q = torch.randn(rows, 6, 256, device=dev, dtype=torch.float16) * 0.3
    gate = torch.randn(rows, 6, 256, device=dev, dtype=torch.float16)
    table = torch.arange(8, device=dev, dtype=torch.int32).repeat(8, 1)
    reqs = torch.arange(8, device=dev, dtype=torch.int32).repeat_interleave(5)
    pos = torch.tensor([251, 252, 253, 254, 255], device=dev, dtype=torch.int64).repeat(
        8
    )
    lens = torch.full((8,), 256, device=dev, dtype=torch.int32)
    req_indices = []
    shared = torch.arange(192, 232, device=dev, dtype=torch.int32)  # 10 shared pages
    for r in range(8):
        uniq = torch.arange(r * 24, r * 24 + 24, device=dev, dtype=torch.int32)
        req_indices.append(torch.cat([uniq, shared], dim=0))
    indices = torch.stack(req_indices, dim=0).repeat_interleave(5, dim=0).contiguous()
    assert indices.shape == (rows, 64)

    out_hot = torch.empty_like(q)
    out_ref = torch.empty_like(q)
    for _ in range(3):
        host_qsa_attention(q, kv_hot, indices, table, reqs, pos, lens, out_hot, gate)
    host_qsa_attention(q, kv_ref, indices, table, reqs, pos, lens, out_ref, gate)
    assert not (kv_hot.page_slots == -2).any()
    torch.testing.assert_close(out_hot, out_ref, rtol=3e-3, atol=1e-3)


def test_kv_block_zeroer_circular_buffer_and_host_qsa_kv_copy_invalidation():
    _require_sm70()
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

    torch.manual_seed(6400)
    dev = torch.device("cuda")
    kv_hot = HostQSAKV(
        4,
        64,
        256,
        dev,
        width=64,
        rows=5,
        resolve_rows=5,
        hot_tokens=128,
        direct_device=True,
        hot_fp8=True,
    )
    # Populate block 0 with real prefix K/V and block 1 with stale K/V + hot cache.
    k0 = torch.randn(64, 1, 256, device=dev, dtype=torch.float16) * 0.3
    v0 = torch.randn(64, 1, 256, device=dev, dtype=torch.float16) * 0.3
    k_stale = torch.randn(64, 1, 256, device=dev, dtype=torch.float16) * 0.9
    v_stale = torch.randn(64, 1, 256, device=dev, dtype=torch.float16) * 0.9
    kv_hot.write(k0, v0, torch.arange(0, 64, device=dev))
    kv_hot.write(k_stale, v_stale, torch.arange(64, 128, device=dev))

    q = torch.randn(5, 6, 256, device=dev, dtype=torch.float16) * 0.3
    gate = torch.randn(5, 6, 256, device=dev, dtype=torch.float16)
    reqs = torch.zeros(5, device=dev, dtype=torch.int32)
    pos = torch.tensor([59, 60, 61, 62, 63], device=dev, dtype=torch.int64)
    lens = torch.tensor([64], device=dev, dtype=torch.int32)
    indices = torch.arange(64, device=dev, dtype=torch.int32).unsqueeze(0).repeat(5, 1)

    # Warm block 1's pages into kv_hot's device hot-page cache!
    out_stale = torch.empty_like(q)
    host_qsa_attention(
        q,
        kv_hot,
        indices,
        torch.tensor([[1]], device=dev, dtype=torch.int32),
        reqs,
        pos,
        lens,
        out_stale,
        gate,
    )
    assert (kv_hot.page_slots[16:32] >= 0).all()

    ring_buf = torch.randn((4, 8, 1, 128), device=dev, dtype=torch.float16)
    swa_buf = torch.randn((2, 4, 16, 2, 64), device=dev, dtype=torch.float16)

    class _Dim1Backend:
        @staticmethod
        def get_kv_cache_block_dim(*_args, **_kwargs) -> int:
            return 1

    static_ctx = {
        "qsa0": SimpleNamespace(kv_cache=kv_hot.history, host_kv=kv_hot),
        "ring0": SimpleNamespace(kv_cache=ring_buf),
        "swa0": SimpleNamespace(kv_cache=swa_buf),
    }
    groups = [
        AttentionGroup(
            backend=QSAStateBackend,
            layer_names=["qsa0"],
            kv_cache_spec=FullAttentionSpec(
                block_size=64,
                num_kv_heads=1,
                head_size=256,
                dtype=torch.uint8,
                host_backed=True,
            ),
            kv_cache_group_id=0,
        ),
        AttentionGroup(
            backend=QSAStateBackend,
            layer_names=["ring0"],
            kv_cache_spec=CircularBufferSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=128,
                head_size_v=0,
                dtype=torch.float16,
            ),
            kv_cache_group_id=1,
        ),
        AttentionGroup(
            backend=_Dim1Backend,
            layer_names=["swa0"],
            kv_cache_spec=SlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=64,
                dtype=torch.float16,
                sliding_window=32,
            ),
            kv_cache_group_id=2,
        ),
    ]
    zeroer = KVBlockZeroer(dev, pin_memory=True)
    zeroer.init_meta(
        groups,
        kernel_block_sizes=[64, 8, 16],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=static_ctx,
        host_backed_layers={"qsa0"},
    )
    assert zeroer.warmup_kernel() is True

    # 1) zero_block_ids zeroes CircularBufferSpec + SlidingWindowSpec, NOT HostQSAKV.
    hist_b2_before = kv_hot.history[2].clone()
    zeroer.zero_block_ids([2])
    assert torch.count_nonzero(ring_buf[2]).item() == 0
    assert torch.count_nonzero(swa_buf[:, 2]).item() == 0
    assert torch.equal(kv_hot.history[2], hist_b2_before)

    # 2) Group-filtered copy for group 1 (ring) touches ONLY ring_buf, NOT HostQSAKV!
    ring_src = torch.randn_like(ring_buf[1])
    ring_buf[1].copy_(ring_src)
    zeroer.copy_block_ids([(1, 3)], [1])
    assert torch.equal(ring_buf[3], ring_src)
    assert (kv_hot.page_slots[16:32] >= 0).all()

    # 3) Group 0 copy copies HostQSAKV history + scales AND invalidates stale hot pages!
    zeroer.copy_block_ids([(0, 1)], [0])
    assert (kv_hot.page_slots[16:32] == -1).all()
    out_blk0 = torch.empty_like(q)
    out_blk1 = torch.empty_like(q)
    host_qsa_attention(
        q,
        kv_hot,
        indices,
        torch.tensor([[0]], device=dev, dtype=torch.int32),
        reqs,
        pos,
        lens,
        out_blk0,
        gate,
    )
    host_qsa_attention(
        q,
        kv_hot,
        indices,
        torch.tensor([[1]], device=dev, dtype=torch.int32),
        reqs,
        pos,
        lens,
        out_blk1,
        gate,
    )
    assert torch.equal(out_blk0, out_blk1)
    assert not torch.allclose(out_blk1, out_stale, atol=1e-2)


def test_kv_block_zeroer_host_qsa_kv_storage_kernel_ratio() -> None:
    """HostQSAKV uses 256-token kernel blocks while scheduler blocks are 2048."""
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

    torch.manual_seed(6401)
    dev = torch.device("cuda")
    # 6 logical scheduler blocks * ratio 8 = 48 kernel blocks of 256 tokens.
    kv_hot = HostQSAKV(
        48,
        256,
        256,
        dev,
        width=64,
        rows=2,
        resolve_rows=2,
        hot_tokens=2048,
        direct_device=True,
        hot_fp8=True,
    )
    # Populate logical block 2 (kernel blocks 16..23, slots 4096..6143) with
    # true prefix K/V, and logical block 5 (kernel blocks 40..47, slots
    # 10240..12287) with stale K/V + resident hot-cache pages.
    k_src = torch.randn(2048, 1, 256, device=dev, dtype=torch.float16) * 0.3
    v_src = torch.randn(2048, 1, 256, device=dev, dtype=torch.float16) * 0.3
    k_stale = torch.randn(2048, 1, 256, device=dev, dtype=torch.float16) * 0.9
    v_stale = torch.randn(2048, 1, 256, device=dev, dtype=torch.float16) * 0.9
    kv_hot.write(k_src, v_src, torch.arange(4096, 6144, device=dev))
    kv_hot.write(k_stale, v_stale, torch.arange(10240, 12288, device=dev))

    q = torch.randn(2, 6, 256, device=dev, dtype=torch.float16) * 0.3
    gate = torch.randn(2, 6, 256, device=dev, dtype=torch.float16)
    reqs = torch.zeros(2, device=dev, dtype=torch.int32)
    pos = torch.tensor([991, 992], device=dev, dtype=torch.int64)
    lens = torch.tensor([993], device=dev, dtype=torch.int32)
    # Select 16 aligned 4-token pages across 256-token kernel sub-blocks (0..3)!
    page_starts = (torch.arange(16, device=dev, dtype=torch.int32) * 60).unsqueeze(1)
    sel = (page_starts + torch.arange(4, device=dev, dtype=torch.int32)).reshape(-1)
    indices = sel.unsqueeze(0).repeat(2, 1)

    tbl_src = torch.arange(16, 24, device=dev, dtype=torch.int32).unsqueeze(0)
    tbl_dst = torch.arange(40, 48, device=dev, dtype=torch.int32).unsqueeze(0)
    out_stale = torch.empty_like(q)
    host_qsa_attention(q, kv_hot, indices, tbl_dst, reqs, pos, lens, out_stale, gate)
    assert (kv_hot.page_slots[5 * 512 : 6 * 512] >= 0).any()

    zeroer = KVBlockZeroer(dev, pin_memory=True)
    zeroer.init_meta(
        [
            AttentionGroup(
                backend=QSAStateBackend,
                layer_names=["qsa0"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=2048,
                    num_kv_heads=1,
                    head_size=256,
                    dtype=torch.uint8,
                    host_backed=True,
                ),
                kv_cache_group_id=0,
            )
        ],
        kernel_block_sizes=[256],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={
            "qsa0": SimpleNamespace(kv_cache=kv_hot.history, host_kv=kv_hot)
        },
        host_backed_layers={"qsa0"},
    )
    assert zeroer.warmup_kernel() is True

    # Copy logical scheduler block 2 -> 5: all 8 kernel sub-blocks, 4096 scales,
    # and 512 hot-page slots must match!
    zeroer.copy_block_ids([(2, 5)], [0])
    assert torch.equal(kv_hot.history[16:24], kv_hot.history[40:48])
    assert torch.equal(kv_hot.scales[4096:6144], kv_hot.scales[10240:12288])
    assert (kv_hot.page_slots[5 * 512 : 6 * 512] == -1).all()

    out_src = torch.empty_like(q)
    out_dst = torch.empty_like(q)
    host_qsa_attention(q, kv_hot, indices, tbl_src, reqs, pos, lens, out_src, gate)
    host_qsa_attention(q, kv_hot, indices, tbl_dst, reqs, pos, lens, out_dst, gate)
    assert torch.equal(out_src, out_dst)
    assert not torch.allclose(out_dst, out_stale, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_select_paged_tokens_bounds_columns_and_multi_req(monkeypatch) -> None:
    """Eager warm-tail and multi-request QSA selection bounds columns to max_seq_len."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    dev = torch.device("cuda", 0)
    torch.manual_seed(19)
    # Emulate max_model_len=262144: 128 pages * 512 compressed keys = 65536 columns.
    k_cache = torch.randn((24, 512, 1, 128), device=dev, dtype=torch.float16) * 0.2
    page_table = torch.full((2, 128), -1, device=dev, dtype=torch.int32)
    page_table[0, :6] = torch.arange(0, 6, device=dev, dtype=torch.int32)
    page_table[1, :4] = torch.arange(6, 10, device=dev, dtype=torch.int32)

    # Multi-request batch: req 0 has 560 rows at seq_len=9600, req 1 has 48 rows
    # at seq_len=3600 (short warm tail).
    rows0, rows1 = 560, 48
    total_rows = rows0 + rows1
    q = torch.randn((total_rows, 4, 128), device=dev, dtype=torch.float16) * 0.3
    token_to_req = torch.cat(
        [
            torch.zeros(rows0, device=dev, dtype=torch.int32),
            torch.ones(rows1, device=dev, dtype=torch.int32),
        ]
    )
    pos = torch.cat(
        [
            torch.arange(9600 - rows0, 9600, device=dev, dtype=torch.int64),
            torch.arange(3600 - rows1, 3600, device=dev, dtype=torch.int64),
        ]
    )
    seq_lens = torch.tensor([9600, 3600], device=dev, dtype=torch.int32)
    qsl_cpu = torch.tensor([0, rows0, total_rows], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([9600, 3600], dtype=torch.int32)

    seen_columns: list[int | None] = []
    orig_paged = qsa_ops.qsa_mqa_paged

    def wrapped_paged(*args, **kwargs):
        seen_columns.append(kwargs.get("num_columns"))
        return orig_paged(*args, **kwargs)

    monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", wrapped_paged)

    ref = qsa_ops.qsa_select_paged_tokens(
        q,
        k_cache,
        page_table,
        token_to_req,
        pos,
        seq_lens,
        token_topk=2048,
        compress_ratio=4,
    )
    seen_columns.clear()
    opt = qsa_ops.qsa_select_paged_tokens(
        q,
        k_cache,
        page_table,
        token_to_req,
        pos,
        seq_lens,
        token_topk=2048,
        compress_ratio=4,
        query_start_loc_cpu=qsl_cpu,
        max_seq_len=9600,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
    )
    assert torch.equal(ref, opt)
    # Short warm tail (req 1, seq_len=3600 -> 900 compressed keys) must NOT
    # allocate/scan all 65536 capacity columns.
    assert seen_columns and all(c is not None and c <= 2400 for c in seen_columns)
    assert min(c for c in seen_columns if c is not None) == 900
