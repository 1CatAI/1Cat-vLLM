# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packing exact fallback requests must preserve request RNG and rejection."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.spec_decode.dflash2 import sparse_rejection
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


def test_reference_mask_keeps_safe_rows():
    probe = torch.arange(21, 0, -1, dtype=torch.float32)[None].repeat(4, 1) / 8
    probe[2, -1] = probe[2, -2]
    mask = sparse_rejection._compact_target_reference_rows(probe, 0.7, 0.8)
    np.testing.assert_array_equal(mask, [False, False, True, False])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("vocab", [32768, 248320])
@torch.inference_mode()
def test_partial_fallback_matches_dense_with_request_slot_permutation(
    monkeypatch, batch_size, ragged, use_fp64, vocab
):
    torch.manual_seed(260926)
    device = "cuda"
    # At 248320 tokens, packing reference requests can cross the SM70
    # top-k/top-p kernel's batch-dependent warp-count boundary. Cover that
    # real vocabulary as well as the smaller generic dispatch case.
    steps, slots = 7, 16
    counts = np.resize([1, 8, 5, 3], batch_size) if ragged else np.full(batch_size, 8)
    cu_np = np.concatenate(([0], counts.cumsum())).astype(np.int32)
    rows = int(cu_np[-1])
    mapping_np = np.array([7, 2, 11, 0, 6, 3, 9, 5][:batch_size])
    mapping = torch.tensor(mapping_np, dtype=torch.int32, device=device)
    cu = torch.tensor(cu_np, device=device)
    expanded = mapping.repeat_interleave(torch.tensor(counts, device=device))
    local = torch.cat([torch.arange(int(n), device=device) for n in counts]).int()
    positions = 8192 + expanded.long() * 32 + local
    target = torch.full((rows, vocab), -20.0, device=device)
    # Use unique, separated supports for safe rows, with a top-20 tie in
    # exactly one row of each fallback request.
    target[:, :21] = torch.arange(21, 0, -1, device=device).float() / 8
    ambiguous = np.array([1] if batch_size < 8 else [1, 5])
    for request in ambiguous:
        target[int(cu_np[request]), :24] = 2.0
    values, ids = target.topk(21, dim=-1)
    draft_ids = torch.arange(16, device=device).expand(slots, steps, 16).contiguous()
    draft_scores = torch.randn(slots, steps, 16, device=device) * 0.5
    draft_dense = torch.full((slots, steps, vocab), -float("inf"), device=device)
    draft_dense.scatter_(2, draft_ids, draft_scores)
    inputs = (local.long() % 16).int()
    # Exercise nonidentity logits_indices as well as ragged q1--q8 tails.
    input_buffer = torch.full((rows * 2,), -1, device=device, dtype=torch.int32)
    position_buffer = torch.zeros(rows * 2, device=device, dtype=torch.int64)
    indices = torch.arange(rows, device=device) * 2
    input_buffer[indices] = inputs
    position_buffer[indices] = positions
    temp_np = np.resize(np.array([0.7, 1.0, 1.3], dtype=np.float32), slots)
    p_np = np.resize(np.array([0.8, 0.95, 1.0], dtype=np.float32), slots)
    temperatures = torch.tensor(temp_np, device=device)
    top_ps = torch.tensor(p_np, device=device)
    seeds = torch.arange(slots, device=device, dtype=torch.int64) + 20260926
    calls = []

    def process(logits, expanded_idx, req_idx_np, pos, draft, local_pos):
        calls.append((logits.shape[0], req_idx_np.copy()))
        logits = logits.float().clone()
        apply_temperature(logits, expanded_idx, temperatures)
        return apply_top_k_top_p(
            logits,
            torch.full((logits.shape[0],), 20, device=device, dtype=torch.int32),
            top_ps[expanded_idx],
        )

    states = SimpleNamespace(
        temperature=SimpleNamespace(np=temp_np, gpu=temperatures),
        top_p=SimpleNamespace(np=p_np, gpu=top_ps),
        seeds=SimpleNamespace(gpu=seeds),
    )
    sampler = SimpleNamespace(
        sampling_states=states, use_fp64_gumbel=use_fp64, apply_sampling_params=process
    )
    rejection = SimpleNamespace(sampler=sampler, num_speculative_steps=steps)
    batch = SimpleNamespace(
        has_structured_output_reqs=False,
        idx_mapping_np=mapping_np,
        idx_mapping=mapping,
        cu_num_logits_np=cu_np,
        cu_num_logits=cu,
        expanded_idx_mapping=expanded,
        expanded_local_pos=local,
        logits_indices=indices,
        input_ids=input_buffer,
        positions=position_buffer,
        num_tokens=rows * 2,
    )

    class Speculator:
        draft_logits = draft_dense

        def get_sparse_draft_logits(self):
            return draft_ids, draft_scores

    monkeypatch.setattr(sparse_rejection, "DFlash2Speculator", Speculator)
    monkeypatch.setattr(
        sparse_rejection.envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", True
    )
    monkeypatch.setattr(sparse_rejection.envs, "VLLM_SPEC_DUMP_ALIGNMENT", False)
    monkeypatch.setattr(
        sparse_rejection, "_supports_sparse_sampling_contract", lambda *args: True
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (7, 0))
    fallback = Mock(return_value=target)
    model = SimpleNamespace(
        get_topk_tokens_and_logits=lambda *args: (ids, values),
        get_topk_tokens_and_logits_with_fallback=lambda *args: (ids, values, fallback),
    )
    for iteration in range(8):
        seeds.add_(17)
        processed = process(target, expanded, mapping_np, positions, inputs, local)
        expected, num_expected = rejection_sample(
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
        calls.clear()
        actual = sparse_rejection.try_dflash2_sparse_target_rejection(
            model,
            Speculator(),
            rejection,
            torch.empty(rows, 4, device=device),
            batch,
            None,
        )
        assert calls[0][0] == int(counts[ambiguous].sum())
        np.testing.assert_array_equal(calls[0][1], mapping_np[ambiguous])
        assert torch.equal(actual.num_sampled, num_expected)
        valid = torch.arange(8, device=device)[None] < num_expected[:, None]
        assert torch.equal(actual.sampled_token_ids[valid], expected[valid])
    assert fallback.call_count == 8
