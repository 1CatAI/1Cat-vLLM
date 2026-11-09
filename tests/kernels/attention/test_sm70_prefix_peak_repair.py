# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repair clipped prefix weights without changing the normal score path."""

import pytest
import torch


def _capture_attention(op, q, k, v, output):
    # Initialize handles/workspaces outside capture, then validate only replay.
    op(q, k, v, output, 0.0625, True)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(q, k, v, output, 0.0625, True)
    graph.replay()
    return graph


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    ("query_len", "op_name"),
    [
        (8000, "sm70_d256_gqa_architecture_fwd"),
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
    ],
)
@torch.inference_mode()
def test_unsampled_prefix_peaks_preserve_weights(query_len, op_name, sparse):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))

    if not hasattr(torch.ops._vllm_fa2_C, op_name):
        pytest.skip("SM70 architecture operator was not built")
    prefix = 8192
    kv_len = prefix + query_len
    q = torch.zeros(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)
    if sparse:
        q[:, 3653, 2, 0] = 16
    else:
        q[..., 0] = 16
    start = 0
    # Both peaks evade the former stride-8 sample. Clipping their distinct
    # logits to the same value gives roughly zero instead of almost +/-1.
    v[:, start + 3, :, 0] = 1
    v[:, start + 5, :, 0] = -1
    output = torch.empty_like(q)
    k[:, start + 3, :, 0] = 16
    k[:, start + 5, :, 0] = 24
    graph = _capture_attention(getattr(torch.ops._vllm_fa2_C, op_name), q, k, v, output)
    rows = torch.tensor([255, 256, 3653, 4095, query_len - 1], device="cuda")
    keys = torch.arange(kv_len, device="cuda")
    for first, second in [(16, 24), (28, 20), (2, 3)]:
        # Replay must recompute maxima when tensor values change in place.
        k[:, start + 3, :, 0] = first
        k[:, start + 5, :, 0] = second
        graph.replay()
        assert torch.isfinite(output).all()
        scores = (
            torch.einsum("rhd,kd->hrk", q[0, rows].double(), k[0, :, 0].double()) / 16
        )
        scores.masked_fill_(
            keys[None, None, :] > (prefix + rows)[None, :, None], -torch.inf
        )
        reference = (scores.softmax(-1) @ v[0, :, 0].double()).permute(1, 0, 2)
        torch.testing.assert_close(
            output[0, rows].double(), reference, rtol=0.003, atol=0.001
        )
