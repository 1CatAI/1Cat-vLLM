# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-like scores must not overflow the Q8000 FP16 PV path."""

import pytest
import torch


@pytest.mark.parametrize("kv_len", [16000, 128000])
@torch.inference_mode()
def test_large_scores_and_biased_values(kv_len):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    if not hasattr(torch.ops._vllm_fa2_C, "sm70_d256_gqa_architecture_fwd"):
        pytest.skip("SM70 architecture operator was not built")
    torch.manual_seed(173)
    q = torch.randn(1, 8000, 6, 256, device="cuda", dtype=torch.float16) * 4
    k = torch.randn(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k) + 8
    output = torch.empty_like(q)
    torch.ops._vllm_fa2_C.sm70_d256_gqa_architecture_fwd(q, k, v, output, 0.0625, True)
    assert torch.isfinite(output).all()
    rows = torch.tensor([0, 63, 64, 3999, 7999], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].float(), k[0, :, 0].float()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - 8000 + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].float()).permute(1, 0, 2)
    torch.testing.assert_close(output[0, rows].float(), reference, rtol=0.01, atol=0.03)


@torch.inference_mode()
def test_periodic_score_spikes_do_not_overflow():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    if not hasattr(torch.ops._vllm_fa2_C, "sm70_d256_gqa_architecture_fwd"):
        pytest.skip("SM70 architecture operator was not built")
    kv_len = 16000
    q = torch.zeros(1, 8000, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)
    q[..., 0] = 16
    # Put every large score at the same nonzero residue. This reproduces the
    # sparse, correlated numerator growth that overflowed long model requests.
    k[:, 3::128, :, 0] = 16
    v[:, 3::128, :, 0] = 1
    output = torch.empty_like(q)
    torch.ops._vllm_fa2_C.sm70_d256_gqa_architecture_fwd(q, k, v, output, 0.0625, True)
    assert torch.isfinite(output).all()
    rows = torch.tensor([0, 63, 64, 3999, 7999], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].float(), k[0, :, 0].float()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - 8000 + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].float()).permute(1, 0, 2)
    torch.testing.assert_close(output[0, rows].float(), reference, rtol=0.01, atol=0.01)
