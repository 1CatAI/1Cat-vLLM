# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-like scores must not overflow the Q8000/Q8192 FP32 MMA path."""

import pytest
import torch


@pytest.mark.parametrize("kv_len", [16000, 128000])
@pytest.mark.parametrize(
    ("query_len", "op_name"),
    [
        (8000, "sm70_d256_gqa_architecture_fwd"),
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
    ],
)
@torch.inference_mode()
def test_large_scores_and_biased_values(kv_len, query_len, op_name):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    if not hasattr(torch.ops._vllm_fa2_C, op_name):
        pytest.skip("SM70 architecture operator was not built")
    torch.manual_seed(173)
    q = torch.randn(1, query_len, 6, 256, device="cuda", dtype=torch.float16) * 4
    k = torch.randn(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k) + 8
    output = torch.empty_like(q)
    getattr(torch.ops._vllm_fa2_C, op_name)(q, k, v, output, 0.0625, True)
    assert torch.isfinite(output).all()
    rows = torch.tensor([0, 63, 64, query_len // 2 - 1, query_len - 1], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].float(), k[0, :, 0].float()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - query_len + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].float()).permute(1, 0, 2)
    torch.testing.assert_close(output[0, rows].float(), reference, rtol=0.01, atol=0.03)


@pytest.mark.parametrize(
    ("query_len", "op_name"),
    [
        (8000, "sm70_d256_gqa_architecture_fwd"),
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
    ],
)
@torch.inference_mode()
def test_periodic_score_spikes_do_not_overflow(query_len, op_name):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    if not hasattr(torch.ops._vllm_fa2_C, op_name):
        pytest.skip("SM70 architecture operator was not built")
    kv_len = 16000
    q = torch.zeros(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)
    q[..., 0] = 16
    # Put every large score at the same nonzero residue. This reproduces the
    # sparse, correlated numerator growth that overflowed long model requests.
    k[:, 3::128, :, 0] = 16
    v[:, 3::128, :, 0] = 1
    output = torch.empty_like(q)
    getattr(torch.ops._vllm_fa2_C, op_name)(q, k, v, output, 0.0625, True)
    assert torch.isfinite(output).all()
    rows = torch.tensor([0, 63, 64, query_len // 2 - 1, query_len - 1], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].float(), k[0, :, 0].float()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - query_len + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].float()).permute(1, 0, 2)
    torch.testing.assert_close(output[0, rows].float(), reference, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    ("query_len", "op_name"),
    [
        (8000, "sm70_d256_gqa_architecture_fwd"),
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
    ],
)
@torch.inference_mode()
def test_rejects_partial_prefix_pv_tile(query_len, op_name):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    if not hasattr(torch.ops._vllm_fa2_C, op_name):
        pytest.skip("SM70 architecture operator was not built")
    q = torch.empty(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.empty(1, query_len + 8, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.empty_like(k)
    output = torch.empty_like(q)
    with pytest.raises(RuntimeError, match="32-token alignment"):
        getattr(torch.ops._vllm_fa2_C, op_name)(q, k, v, output, 0.0625, True)


@torch.inference_mode()
def test_q8000_q8192_share_scores_and_preserve_graph_replay():
    if not torch.accelerator.is_available() or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires SM70")
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    torch.manual_seed(732)
    cases = []
    for length, name in [
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
        (8000, "sm70_d256_gqa_architecture_fwd"),
    ]:
        q = torch.randn(1, length, 6, 256, device="cuda", dtype=torch.float16)
        k = torch.randn(1, length + 8192, 1, 256, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        op = getattr(torch.ops._vllm_fa2_C, name)
        before = torch.accelerator.memory_allocated()
        op(q, k, v, out, 0.0625, True)
        if length == 8000:
            # The second family must not allocate another ~2.2 GiB scores buffer.
            assert torch.accelerator.memory_allocated() - before < 1024**3
        reference = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(q, k, v, out, 0.0625, True)
        cases.append((graph, out, reference, q, k, v, op))
    combined = torch.cuda.CUDAGraph()
    with torch.cuda.graph(combined):
        for _, out, _, q, k, v, op in cases:
            op(q, k, v, out, 0.0625, True)
    for _ in range(3):
        for graph, out, reference, *_ in cases:
            graph.replay()
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
        combined.replay()
        for _, out, reference, *_ in cases:
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
