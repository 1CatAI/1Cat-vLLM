# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _sm70_ops  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_unsupported_child_keeps_parent_replayable():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 graph branch gate")
    values = torch.ones(1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(values)
    flags = torch.ones(1, dtype=torch.bool, device="cuda")
    prefix = torch.cuda.CUDAGraph(keep_graph=True)
    reference = torch.cuda.CUDAGraph(keep_graph=True)
    compact = torch.cuda.CUDAGraph(keep_graph=True)
    event = torch.cuda.Event(external=True)
    with torch.cuda.graph(prefix):
        output.copy_(values)
    with torch.cuda.graph(reference):
        event.record()
        output.fill_(17)
    with torch.cuda.graph(compact):
        output.fill_(13)
    status = torch.ops._C.sm70_sampler_graph_prepare_collective(
        flags, reference.raw_cuda_graph()
    )
    assert status == 801
    status = torch.ops._C.sm70_sampler_graph_attach_branch(
        flags,
        prefix.raw_cuda_graph(),
        reference.raw_cuda_graph(),
        compact.raw_cuda_graph(),
    )
    assert status == 801
    prefix.instantiate()
    values.fill_(23)
    prefix.replay()
    assert output.item() == 23


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_sampler_branches_follow_live_device_flags(rows):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 graph branch gate")
    if not hasattr(torch.ops._C, "sm70_sampler_graph_attach_branch"):
        pytest.fail("Build the complete SM70 sampler module")
    values = torch.zeros(rows, dtype=torch.int32, device="cuda")
    flags = torch.zeros(rows, dtype=torch.bool, device="cuda")
    output = torch.empty_like(values)
    counters = torch.zeros(2, dtype=torch.int64, device="cuda")
    prefix = torch.cuda.CUDAGraph(keep_graph=True)
    reference = torch.cuda.CUDAGraph(keep_graph=True)
    compact = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(prefix):
        flags.copy_(values > 0)
    with torch.cuda.graph(reference):
        output.copy_(values + 17)
    with torch.cuda.graph(compact):
        output.copy_(values - 13)
    status = torch.ops._C.sm70_sampler_graph_attach_branch(
        flags,
        prefix.raw_cuda_graph(),
        reference.raw_cuda_graph(),
        compact.raw_cuda_graph(),
        counters,
    )
    if status == 801:  # Older CUDA runtime or unsupported driver.
        pytest.skip("CUDA conditional graph support is unavailable")
    assert status == 0
    prefix.instantiate()
    for step in range(8):
        host_values = torch.full((rows,), -step, dtype=torch.int32)
        if step % 2:
            # The final verifier row must participate in the request decision.
            host_values[-1] = step
        values.copy_(host_values)
        prefix.replay()
        expected = host_values + (17 if (host_values > 0).any() else -13)
        assert torch.equal(output.cpu(), expected)

    assert counters.cpu().tolist() == [4, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_fp64", [False, True])
@torch.inference_mode()
def test_live_rejection_branches_match_dense_distribution(use_fp64):
    """Changing cutoff ties must select exact rejection without a host fence."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
        _compact_target_reference_rows_kernel,
    )
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        dflash2_sparse_topk_rejection_sample,
        rejection_sample,
    )

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 graph branch gate")
    torch.manual_seed(20261004)
    rows, vocab, steps, slots = 8, 248320, 7, 8
    target = torch.full((rows, vocab), -20.0, device="cuda")
    target[:, :21] = torch.arange(21, 0, -1, device="cuda") / 8
    mapping = torch.tensor([7], dtype=torch.int32, device="cuda")
    expanded = mapping.expand(rows).contiguous()
    local = torch.arange(rows, dtype=torch.int32, device="cuda")
    positions = local.long() + 8192
    inputs = local.clone()
    cu = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    temperatures = torch.full((slots,), 0.7, device="cuda")
    top_ps = torch.full((slots,), 0.9, device="cuda")
    seeds = torch.arange(slots, dtype=torch.int64, device="cuda") + 20261004
    draft_ids = torch.arange(16, device="cuda").expand(slots, steps, 16).contiguous()
    draft_scores = torch.randn(slots, steps, 16, device="cuda") * 0.5
    draft_dense = torch.full((slots, steps, vocab), -float("inf"), device="cuda")
    draft_dense.scatter_(2, draft_ids, draft_scores)
    flags = torch.empty(rows, dtype=torch.bool, device="cuda")
    output = torch.empty(1, rows, dtype=torch.int64, device="cuda")
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    branch_taken = torch.empty(1, dtype=torch.int32, device="cuda")

    def probe():
        values, ids = target.topk(64, dim=-1)
        values, ids = sort_topk_with_vocab_ties(
            values, ids, vocab_size=vocab, descending=True
        )
        _compact_target_reference_rows_kernel[(rows,)](
            values,
            temperatures,
            top_ps,
            expanded,
            flags,
            values.stride(0),
            64,
            20,
            64,
            num_warps=1,
        )
        return values, ids

    def dense():
        processed = target.clone()
        apply_temperature(processed, expanded, temperatures)
        processed = apply_top_k_top_p(
            processed,
            torch.full((rows,), 20, dtype=torch.int32, device="cuda"),
            top_ps[expanded],
        )
        return rejection_sample(
            processed,
            draft_dense,
            inputs,
            cu,
            positions,
            mapping,
            expanded,
            local,
            temperatures,
            seeds,
            steps,
            use_fp64=use_fp64,
        )

    def compact(values, ids):
        return dflash2_sparse_topk_rejection_sample(
            ids,
            values,
            draft_ids,
            draft_scores,
            inputs,
            cu,
            positions,
            mapping,
            temperatures,
            top_ps,
            seeds,
            steps,
            use_fp64=use_fp64,
            target_top_k=20,
        )

    values, ids = probe()
    dense()
    compact(values, ids)
    torch.accelerator.synchronize()
    prefix = torch.cuda.CUDAGraph(keep_graph=True)
    reference = torch.cuda.CUDAGraph(keep_graph=True)
    shortlist = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(prefix):
        values, ids = probe()
    with torch.cuda.graph(reference):
        tokens, n = dense()
        output.copy_(tokens)
        count.copy_(n)
        branch_taken.fill_(1)
    with torch.cuda.graph(shortlist):
        tokens, n = compact(values, ids)
        output.copy_(tokens)
        count.copy_(n)
        branch_taken.fill_(0)
    status = torch.ops._C.sm70_sampler_graph_attach_branch(
        flags,
        prefix.raw_cuda_graph(),
        reference.raw_cuda_graph(),
        shortlist.raw_cuda_graph(),
    )
    if status == 801:
        pytest.skip("CUDA conditional graph support is unavailable")
    assert status == 0
    prefix.instantiate()
    for iteration in range(20):
        tie_width = [21, 24, 63, 64, 80][iteration % 5]
        target.fill_(-20)
        target[:, :21] = torch.arange(21, 0, -1, device="cuda") / 8
        # A fallback in the bonus row must switch the entire request.
        target[-1, :tie_width] = 2.0
        seeds.add_(17)
        positions.add_(11)
        top_ps.fill_(1.0 if iteration % 2 else 0.9)
        prefix.replay()
        expected, expected_count = dense()
        assert branch_taken.item() == int(tie_width >= 64)
        assert torch.equal(count, expected_count)
        valid = torch.arange(rows, device="cuda")[None] < count[:, None]
        assert torch.equal(output[valid], expected[valid])
