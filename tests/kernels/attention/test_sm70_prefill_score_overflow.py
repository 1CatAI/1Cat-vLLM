# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recover overflowing compact score tiles while preserving finite rounding."""

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


@pytest.mark.parametrize("query_len", [8000, 8192])
@pytest.mark.parametrize(
    ("location", "total_kv"),
    [
        ("prefix", 0),
        ("last_prefix_block", 0),
        ("tail", 0),
        ("last_prefix_block", 256000),
    ],
)
@torch.inference_mode()
def test_compact_score_overflow_recovery_with_changing_graph_inputs(
    query_len, location, total_kv
):
    """Finite Q/K can overflow FP16 score storage; graph flags must reset."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))

    name = (
        "sm70_d256_gqa_architecture_fwd"
        if query_len == 8000
        else "sm70_d256_gqa_architecture_q8192_fwd"
    )
    op = getattr(torch.ops._vllm_fa2_C, name, None)
    if op is None:
        pytest.skip("SM70 architecture operator was not built")
    kv_len = total_kv or 8224 + query_len
    prefix = kv_len - query_len
    q = torch.zeros(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)
    out = torch.empty_like(q)
    start = {
        "prefix": 0,
        "last_prefix_block": (prefix - 1) // 8192 * 8192,
        "tail": prefix,
    }[location]
    graph = _capture_attention(op, q, k, v, out)
    rows = torch.tensor(
        [0, 1, 63, 64, 255, 256, 403, 1023, 1024, 3653, query_len - 1],
        device="cuda",
    )
    keys = torch.arange(kv_len, device="cuda")
    for case in ["overflow", "negative_overflow", "ordinary"]:
        q.zero_()
        k.zero_()
        v.zero_()
        v[:, start + 3, :, 0] = 1
        v[:, start + 5, :, 0] = -1
        # Only one head in scattered query rows is exceptional. Other heads
        # and rows still need their ordinary answer after selective recovery.
        q[:, rows, 2, 0] = 16
        if case == "overflow":
            q[:, rows, 2, 0] = 256
            k[:, start + 3, :, 0] = 4096
            k[:, start + 5, :, 0] = 4100
        elif case == "negative_overflow":
            q[:, rows, 2, 0] = 256
            q[:, rows, 2, 1] = 16
            k[..., 0] = -4096
            k[:, start + 3, :, 1] = 1
            k[:, start + 5, :, 1] = 2
            v[:, start + 3, :, 0] = 1000
            v[:, start + 5, :, 0] = -1000
        else:
            # Clear previous flags as well as maxima and denominators.
            k[:, start + 3, :, 0] = 2
            k[:, start + 5, :, 0] = 3
        graph.replay()
        scores = (
            torch.einsum("rhd,kd->hrk", q[0, rows].double(), k[0, :, 0].double()) / 16
        )
        scores.masked_fill_(
            keys[None, None, :] > (prefix + rows)[None, :, None], -torch.inf
        )
        reference = (scores.softmax(-1) @ v[0, :, 0].double()).permute(1, 0, 2)
        assert torch.isfinite(out).all(), case
        torch.testing.assert_close(
            out[0, rows].double(), reference, rtol=0.003, atol=0.001, msg=case
        )
