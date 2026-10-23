# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default (untagged) portable prefill numerical and shape boundaries."""

import pytest


@pytest.fixture
def prefill_environment(monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() not in ((7, 0), (7, 2), (8, 6), (8, 9)):
        pytest.skip("requires qualified portable sparse hardware")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.sm70_sparse import sparse_policy
    from vllm.models.deepseek_v4.sm70.indexer import sm70_indexer_prefill_logits
    from vllm.utils.torch_utils import set_high_precision_cuda_matmul_defaults

    matmul = torch.backends.cuda.matmul
    names = (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    )
    previous = {name: getattr(matmul, name) for name in names if hasattr(matmul, name)}
    try:
        set_high_precision_cuda_matmul_defaults()
        with set_current_vllm_config(VllmConfig()):
            policy = sparse_policy()
            monkeypatch.setattr(policy, "indexer_relu", True)
            monkeypatch.setattr(policy, "indexer_prefill_cublas", True)
            monkeypatch.setattr(policy, "indexer_prefill_tile_mb", 192)
            yield torch, policy, sm70_indexer_prefill_logits
    finally:
        for name, value in previous.items():
            setattr(matmul, name, value)


def finite_case(torch, rows, heads, keys):
    generator = torch.Generator().manual_seed(20261023)
    q = (torch.randn((rows, heads, 128), generator=generator) * 0.25).half()
    weights = torch.randn((rows, heads), generator=generator)
    bits_palette = torch.tensor([0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=torch.uint8)
    values_palette = torch.tensor([0.5, 1, 2, -0.5, -1, -2], dtype=torch.float64)
    picks = torch.randint(6, (keys, 128), generator=generator)
    bits, values = bits_palette[picks], values_palette[picks]
    scales = torch.linspace(0.5, 2, keys)
    return q, weights, bits, values, scales


def reference(torch, q, weights, values, scales, relu=True):
    if relu:
        dots = q.double() @ values.T
        return (dots.relu() * weights.double().unsqueeze(-1)).sum(1) * scales.double()
    weighted = (q.double() * weights.double().unsqueeze(-1)).sum(1)
    return weighted @ (values * scales.double().unsqueeze(-1)).T


@pytest.mark.parametrize("heads", [1, 7, 8, 9, 16, 32, 64])
def test_finite_scores_for_partial_and_complete_head_groups(prefill_environment, heads):
    torch, _, call = prefill_environment
    q, weights, bits, values, scales = finite_case(torch, 3, heads, 65)
    # Exercise the production reshape/contiguous path rather than only dense Q.
    q_gpu = q.cuda().repeat_interleave(2, dim=-1)[..., ::2]
    assert not q_gpu.is_contiguous()
    actual = call(
        q_gpu, bits.cuda().view(torch.float8_e4m3fn), scales.cuda(), weights.cuda()
    ).cpu()
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.double(),
        reference(torch, q, weights, values, scales),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("signed_cancellation", [False, True])
def test_large_finite_products_remain_finite_without_request_tags(
    prefill_environment, signed_cancellation
):
    torch, _, call = prefill_environment
    q = torch.zeros((2, 32, 128), dtype=torch.float16)
    weights = torch.zeros((2, 32), dtype=torch.float32)
    q[:, 0], weights[:, 0] = 16, 1
    if signed_cancellation:
        q[:, 1], weights[:, 1] = 8, -2
    bits = torch.full((17, 128), 0x78, dtype=torch.uint8)
    values = torch.full((17, 128), 256, dtype=torch.float64)
    scales = torch.tensor([0.5, 1, 2] * 5 + [0.5, 1], dtype=torch.float32)
    actual = call(
        q.cuda(), bits.cuda().view(torch.float8_e4m3fn), scales.cuda(), weights.cuda()
    ).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.double(), reference(torch, q, weights, values, scales), rtol=0, atol=0
    )


def test_sub64_tile_and_partial_final_tile(prefill_environment, monkeypatch):
    torch, policy, call = prefill_environment
    monkeypatch.setattr(policy, "indexer_prefill_tile_mb", 1)
    q, weights, bits, values, scales = finite_case(torch, 512, 9, 65)
    # A 1 MiB C budget fits56 keys here:56+9, not a rounded-up64-key tile.
    actual = call(
        q.cuda(), bits.cuda().view(torch.float8_e4m3fn), scales.cuda(), weights.cuda()
    ).cpu()
    torch.testing.assert_close(
        actual.double(),
        reference(torch, q, weights, values, scales),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("branch", ["cublas_off", "relu_off"])
def test_existing_opt_out_semantics(prefill_environment, monkeypatch, branch):
    torch, policy, call = prefill_environment
    relu = branch != "relu_off"
    monkeypatch.setattr(policy, "indexer_relu", relu)
    monkeypatch.setattr(policy, "indexer_prefill_cublas", branch != "cublas_off")
    q, weights, bits, values, scales = finite_case(torch, 4, 8, 23)
    actual = call(
        q.cuda(), bits.cuda().view(torch.float8_e4m3fn), scales.cuda(), weights.cuda()
    ).cpu()
    expected = reference(torch, q, weights, values, scales, relu=relu)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-2, atol=1e-2)
