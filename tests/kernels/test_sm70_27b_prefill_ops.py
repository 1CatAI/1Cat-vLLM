# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SM70 27B dense prefill custom operators and V3 Q2048 dispatch."""

from __future__ import annotations

import pytest
import torch


def _require_sm70() -> None:
    if not torch.accelerator.is_available() or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("SM70 CUDA test")


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        # TP=4 Qwen3.8-27B shapes
        (4096, 5120, 8704),
        (4096, 4352, 5120),
        (4096, 5120, 4096),
        (4096, 1536, 5120),
        # TP=2 Qwen3.8-27B shapes
        (2048, 5120, 17408),
        (2048, 8704, 5120),
        (2048, 5120, 8192),
        (2048, 3072, 5120),
        # Odd M tail
        (769, 5120, 3584),
    ],
)
@torch.inference_mode()
def test_cublaslt_fp32_mm_out_accuracy(m: int, k: int, n: int) -> None:
    _require_sm70()
    from vllm._sm70_27b_prefill_ops import has_sm70_27b_prefill_ops

    if not has_sm70_27b_prefill_ops():
        pytest.skip("_sm70_27b_prefill_C extension not built")

    torch.manual_seed(42 + m + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn(k, n, device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    torch.ops.vllm_sm70_27b_prefill.cublaslt_fp32_mm_out(out, x, w)
    ref = torch.matmul(x.float(), w.float()).half()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize(("m", "d"), [(4096, 4352), (2048, 8704), (513, 4352)])
@torch.inference_mode()
def test_silu_and_mul_out_accuracy(m: int, d: int) -> None:
    _require_sm70()
    from vllm._sm70_27b_prefill_ops import has_sm70_27b_prefill_ops

    if not has_sm70_27b_prefill_ops():
        pytest.skip("_sm70_27b_prefill_C extension not built")

    torch.manual_seed(99 + m)
    gate_up = torch.randn(m, 2 * d, device="cuda", dtype=torch.float16)
    out = torch.empty(m, d, device="cuda", dtype=torch.float16)
    torch.ops.vllm_sm70_27b_prefill.silu_and_mul_out(out, gate_up)
    gate = gate_up[:, :d].float()
    up = gate_up[:, d:].float()
    ref = (torch.nn.functional.silu(gate) * up).half()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("m", "num_heads", "weight_dtype"),
    [
        (4096, 12, torch.float16),
        (4096, 12, torch.float32),
        (2048, 24, torch.float16),
        (513, 24, torch.float32),
    ],
)
@torch.inference_mode()
def test_gdn_rmsnorm_gated_d128_strided_z(
    m: int, num_heads: int, weight_dtype: torch.dtype
) -> None:
    _require_sm70()
    from vllm._sm70_27b_prefill_ops import has_sm70_27b_prefill_ops

    if not has_sm70_27b_prefill_ops():
        pytest.skip("_sm70_27b_prefill_C extension not built")

    torch.manual_seed(123 + num_heads)
    x = torch.randn(m, num_heads, 128, device="cuda", dtype=torch.float16)
    # Simulate non-contiguous z slice from fused QKVZ projection.
    qkv_dim = num_heads * 128 + 1024
    packed = torch.randn(
        m, qkv_dim + num_heads * 128, device="cuda", dtype=torch.float16
    )
    z = packed[:, qkv_dim:].view(m, num_heads, 128)
    assert not z.is_contiguous()
    w = torch.randn(128, device="cuda", dtype=weight_dtype)
    eps = 1e-6

    out = torch.empty(m, num_heads * 128, device="cuda", dtype=torch.float16)
    torch.ops.vllm_sm70_27b_prefill.gdn_rmsnorm_gated_d128_out(out, x, z, w, eps)

    x_f = x.float()
    z_f = z.float()
    rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    ref = (
        (x_f * rms * w.float() * torch.nn.functional.silu(z_f))
        .reshape(m, num_heads * 128)
        .half()
    )
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=2e-2)


def test_peer_runtime_returns_none_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.envs as envs
    from vllm._sm70_27b_prefill_ops import Sm7027BPeerRuntime

    monkeypatch.setattr(envs, "VLLM_SM70_27B_DISABLE_P2P_NORM", True)
    rt = Sm7027BPeerRuntime.try_create(
        cpu_group=None,
        device=torch.device("cpu"),
        rank=0,
        world_size=4,
        ca_comm=None,
    )
    assert rt is None


@pytest.mark.parametrize(
    ("query_len", "kv_len"),
    [
        (2048, 16384),
        (1536, 16384),
        (1777, 16395),
        (3781, 12301),
    ],
)
@torch.inference_mode()
def test_v3_q2048_and_odd_kv_causal_alignment(query_len: int, kv_len: int) -> None:
    _require_sm70()
    from vllm.v1.attention.backends.flash_v100 import dense_prefill, ops
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))
    q2048_op = ops.get_sm70_d256_gqa_architecture_q2048_op()
    q4096_op = ops.get_sm70_d256_gqa_architecture_q4096_op()
    if q2048_op is None or q4096_op is None:
        pytest.skip("SM70 Q2048/Q4096 architecture operators were not built")

    torch.manual_seed(2048 + query_len + kv_len)
    q = torch.randn(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    q_target = 2048 if query_len <= 2048 else 4096
    op = q2048_op if q_target == 2048 else q4096_op
    dense_prefill._run_sm70_27b_v3_padded(op, q, k, v, out, 0.0625, q_target)
    assert torch.isfinite(out).all()

    rows = torch.tensor([0, 63, 64, query_len // 2, query_len - 1], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].float(), k[0, :, 0].float()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - query_len + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].float()).permute(1, 0, 2)
    torch.testing.assert_close(out[0, rows].float(), reference, rtol=0.01, atol=0.03)
