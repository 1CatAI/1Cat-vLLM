# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.sm70_compact_topk import compact_half_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 required",
)


@pytest.mark.parametrize(
    "rows,width",
    [
        (8, 62080),
        (32, 65535),
        (40, 62080),
        (64, 65535),
        (8, 1025),
        (8, 65536),
        (8, 124160),
        (16, 124160),
        (8, 248320),
        (5, 524288),
    ],
)
@torch.inference_mode()
def test_half_source_values_and_live_graph(rows, width):
    torch.manual_seed(1530)
    logits = torch.randn(rows, width, dtype=torch.float16, device="cuda")
    assert compact_half_topk(logits) is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = compact_half_topk(logits)
    for case in ("random", "ties", "zeros", "nonfinite"):
        logits.normal_()
        if case == "ties":
            logits.round_()
        elif case == "zeros":
            logits.zero_()
            logits[:, ::2] = -0.0
        elif case == "nonfinite":
            logits[:, 0] = float("nan")
            logits[:, 1] = float("inf")
            logits[:, 2] = -float("inf")
        graph.replay()
        values, ids = selected
        expected = logits.float().topk(64, dim=-1).values
        torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
        original = logits.gather(1, ids).float()
        torch.testing.assert_close(values, original, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(values)
        assert torch.equal(
            values.view(torch.int32)[finite], original.view(torch.int32)[finite]
        )
        assert all(len(set(row)) == 64 for row in ids.cpu().tolist())


@torch.inference_mode()
def test_fp32_source_never_qualifies():
    logits = torch.randn(8, 62080, device="cuda", dtype=torch.float32)
    assert compact_half_topk(logits) is None


@pytest.mark.parametrize("rows", [8, 32, 40])
@torch.inference_mode()
def test_target_probe_preserves_dense_token_mask(rows):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
    from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
        _compact_target_reference_rows,
    )

    torch.manual_seed(1530)
    logits = torch.randn(rows, 248320, dtype=torch.float16, device="cuda") * 1.5
    logits[0].fill_(1)
    logits[1].fill_(-20)
    logits[1, :160] = 3
    logits[2].fill_(-20)
    logits[2, :24] = 1
    logits[2, :2] = 2
    logits[3].fill_(-float("inf"))
    logits[3, 7] = 1

    def probe(custom):
        shard_values, shard_ids = [], []
        for rank in range(4):
            shard = logits[:, rank * 62080 : (rank + 1) * 62080]
            if custom:
                values, ids = compact_half_topk(shard)
            else:
                values, ids = shard.float().topk(64, dim=-1)
            shard_values.append(values)
            shard_ids.append(ids + rank * 62080)
        values, order = torch.cat(shard_values, 1).topk(64, dim=-1)
        ids = torch.cat(shard_ids, 1).gather(1, order)
        return sort_topk_with_vocab_ties(
            values, ids, vocab_size=248320, descending=True
        )

    original_values, _ = probe(False)
    values, ids = probe(True)
    for temperature in (0.7, 1.0):
        for top_p in (0.6, 0.9, 1.0):
            reference = _compact_target_reference_rows(
                values, temperature, top_p, vocab_ordered=True
            )
            old_reference = _compact_target_reference_rows(
                original_values, temperature, top_p, vocab_ordered=True
            )
            assert np.array_equal(reference, old_reference)
            assert reference[0] and reference[1] and reference[3]
            for row in np.flatnonzero(~reference):
                scaled = values[row] / temperature
                scaled.masked_fill_(values[row] < values[row, 19], -float("inf"))
                probabilities = scaled.softmax(-1)
                keep = probabilities.cumsum(-1) - probabilities < top_p
                keep &= torch.isfinite(scaled)
                actual = torch.full((1, 248320), -float("inf"), device="cuda")
                actual[0, ids[row, keep]] = scaled[keep]
                expected = apply_top_k_top_p_pytorch(
                    logits[row : row + 1].float() / temperature,
                    torch.tensor([20], device="cuda"),
                    torch.tensor([top_p], device="cuda"),
                )
                assert torch.equal(actual, expected)


@pytest.mark.parametrize("vocab_size", (124160, 248320))
def test_stream_captured_topk_topp_compact_avoids_full_sort_workspace(vocab_size):
    from vllm.config.execution_policy import layer_policy
    from vllm.v1.sample.ops.topk_topp_runtime import TopKTopPRuntime
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

    torch.manual_seed(20261115)
    base = torch.randn((8, vocab_size), device="cuda", dtype=torch.float32)
    base[2].fill_(-10.0)
    base[2, :24] = 2.5
    k = torch.full((8,), 20, device="cuda", dtype=torch.int32)
    p = torch.full((8,), 0.85, device="cuda", dtype=torch.float32)

    eager_logits = base.clone()
    eager_rt = TopKTopPRuntime(layer_policy(), no_host_sync=True)
    eager_out = apply_top_k_top_p_triton(eager_logits, k, p, runtime=eager_rt)

    cap_in = base.clone()
    cap_rt = TopKTopPRuntime(layer_policy())
    # Warm up compact workspace outside capture so CUDA graph capture has no mallocs.
    _ = apply_top_k_top_p_triton(cap_in.clone(), k, p, runtime=cap_rt)
    assert len(cap_rt.buffers) == 0

    g = torch.cuda.CUDAGraph()
    torch.accelerator.synchronize()
    with torch.cuda.graph(g):
        cap_out = apply_top_k_top_p_triton(cap_in, k, p, runtime=cap_rt)

    cap_in.copy_(base)
    g.replay()
    torch.accelerator.synchronize()
    assert len(cap_rt.buffers) == 0
    assert torch.equal(cap_out, eager_out)
