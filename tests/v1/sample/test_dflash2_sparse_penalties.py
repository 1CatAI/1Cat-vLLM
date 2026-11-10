# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact DFlash2 rejection with penalties must match the dense sampler."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.penalties import apply_penalties
from vllm.v1.worker.gpu.spec_decode.dflash2 import sparse_rejection
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

ROWS = 8
VOCAB = 32768
SLOTS = 6

# (repetition, frequency, presence) per request slot used by the tests.
PENALTIES = {
    "presence": (1.0, 0.0, 1.5),
    "all": (1.3, 0.5, 0.7),
    "none": (1.0, 0.0, 0.0),
}


def _uva(values: np.ndarray, device: str) -> SimpleNamespace:
    return SimpleNamespace(np=values, gpu=torch.tensor(values, device=device))


def _penalties_state(slot_penalties: dict[int, tuple], device: str):
    rep = np.ones(SLOTS, dtype=np.float32)
    freq = np.zeros(SLOTS, dtype=np.float32)
    pres = np.zeros(SLOTS, dtype=np.float32)
    for slot, (r, f, p) in slot_penalties.items():
        rep[slot], freq[slot], pres[slot] = r, f, p
    return SimpleNamespace(
        use_penalty=(rep != 1.0) | (freq != 0.0) | (pres != 0.0),
        repetition_penalty=_uva(rep, device),
        frequency_penalty=_uva(freq, device),
        presence_penalty=_uva(pres, device),
        prompt_bin_mask=torch.zeros(
            SLOTS, cdiv(VOCAB, 32), dtype=torch.int32, device=device
        ),
        output_bin_counts=torch.zeros(SLOTS, VOCAB, dtype=torch.int32, device=device),
    )


def _mark(state, slot: int, tokens: torch.Tensor, generator: torch.Generator):
    """Give some candidate tokens prompt bits and earlier output counts."""
    tokens = tokens.unique()
    picked = tokens[torch.rand(tokens.numel(), generator=generator) < 0.2]
    counts = torch.randint(1, 4, (picked.numel(),), generator=generator)
    state.output_bin_counts[slot, picked] = counts.to(state.output_bin_counts)
    for token in tokens[torch.rand(tokens.numel(), generator=generator) < 0.1]:
        token = int(token)
        state.prompt_bin_mask[slot, token // 32] |= 1 << (token % 32)


def _dense_penalized(logits, state, expanded, token_ids, local):
    result = logits.float().clone()
    apply_penalties(
        result,
        expanded,
        token_ids,
        local,
        state.repetition_penalty.gpu,
        state.frequency_penalty.gpu,
        state.presence_penalty.gpu,
        state.prompt_bin_mask,
        state.output_bin_counts,
    )
    return result


def test_contract_admits_only_monotone_penalties_on_the_wide_probe():
    def fixture(rows: int, vocab: int, penalties: tuple):
        states = SimpleNamespace(
            vocab_size=vocab,
            temperature=SimpleNamespace(np=np.ones(1, dtype=np.float32)),
            top_k=SimpleNamespace(np=np.full(1, 20, dtype=np.int32)),
            top_p=SimpleNamespace(np=np.full(1, 0.95, dtype=np.float32)),
            min_p=SimpleNamespace(np=np.zeros(1, dtype=np.float32)),
            max_num_logprobs=Mock(return_value=-1),
        )
        rep, freq, pres = (np.array([v], dtype=np.float32) for v in penalties)
        sampler = SimpleNamespace(
            sampling_states=states,
            penalties_state=SimpleNamespace(
                use_penalty=np.array([True]),
                repetition_penalty=SimpleNamespace(np=rep),
                frequency_penalty=SimpleNamespace(np=freq),
                presence_penalty=SimpleNamespace(np=pres),
            ),
            logit_bias_state=SimpleNamespace(use_logit_bias=np.zeros(1, bool)),
            bad_words_state=SimpleNamespace(
                num_bad_words=SimpleNamespace(np=np.zeros(1, dtype=np.int32))
            ),
            logprob_token_ids_state=SimpleNamespace(
                max_num_token_ids=Mock(return_value=0)
            ),
            compute_nans=False,
        )
        rejection = SimpleNamespace(rejection_sample_method="standard", sampler=sampler)
        batch = SimpleNamespace(
            num_reqs=1,
            is_prefilling_np=np.zeros(1, dtype=bool),
            idx_mapping_np=np.zeros(1, dtype=np.int32),
            cu_num_logits_np=np.array([0, rows]),
        )
        return rejection, batch

    contract = sparse_rejection._supports_sparse_sampling_contract
    for penalties in [(1.0, 0.0, 1.5), (1.3, 0.5, 0.7)]:
        assert contract(*fixture(8, 248320, penalties), allow_penalties=True)
        # Callers that cannot penalize the probe keep the dense sampler.
        assert not contract(*fixture(8, 248320, penalties))
    # Negative penalties can raise an unprobed token above the cutoff.
    for penalties in [(0.9, 0.0, 0.0), (1.0, -0.5, 0.0), (1.0, 0.0, -1.5)]:
        assert not contract(*fixture(8, 248320, penalties), allow_penalties=True)
    # The floor guard needs the wide, tie-retaining probe.
    assert not contract(*fixture(1, 248320, (1.0, 0.0, 1.5)), allow_penalties=True)
    assert not contract(*fixture(8, 1024, (1.0, 0.0, 1.5)), allow_penalties=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_penalized_probe_matches_dense_top_k_or_flags_reference():
    device = "cuda"
    generator = torch.Generator().manual_seed(20261106)
    slots = {1: "presence", 3: "all", 4: "none", 5: "presence"}
    mapping = torch.tensor(list(slots), dtype=torch.int32, device=device)
    expanded = mapping.repeat_interleave(ROWS)
    local = torch.arange(ROWS, dtype=torch.int32, device=device).repeat(len(slots))
    rows = expanded.numel()
    logits = (torch.randn(rows, VOCAB, generator=generator) * 3).to(device)
    top = logits.topk(64, dim=-1).indices.cpu()
    # Draft tokens drawn from the candidates, so earlier rows penalize later ones.
    pick = torch.randint(0, 64, (rows,), generator=generator)
    token_ids = top[torch.arange(rows), pick].to(device)
    state = _penalties_state({s: PENALTIES[n] for s, n in slots.items()}, device)
    for index, slot in enumerate(slots):
        _mark(state, slot, top[index * ROWS : (index + 1) * ROWS].flatten(), generator)
    # Slot 5 penalized every candidate heavily: unprobed tokens now rank inside
    # the top 20, which the floor guard must detect.
    state.presence_penalty.gpu[5] = 8.0
    state.output_bin_counts[5, top[3 * ROWS :].flatten().to(device)] = 1

    dense = _dense_penalized(logits, state, expanded, token_ids, local)
    dense_values, dense_ids = sort_topk_with_vocab_ties(
        *dense.topk(64, dim=-1), vocab_size=VOCAB, descending=True
    )

    values, ids = logits.topk(64, dim=-1)
    floors = sparse_rejection._penalize_probe(
        values, ids, state, expanded, token_ids, local
    )
    values, ids = sort_topk_with_vocab_ties(
        values, ids, vocab_size=VOCAB, descending=True
    )
    temperatures = torch.ones(SLOTS, device=device)
    top_ps = torch.ones(SLOTS, device=device)
    flags = sparse_rejection._compact_target_reference_flags(
        values, temperatures, top_ps, expanded, floors
    ).cpu()

    assert torch.equal(floors, logits.topk(64, dim=-1).values[:, -1])
    assert flags[3 * ROWS :].all()
    exact = ~flags
    assert exact[: 3 * ROWS].sum() >= ROWS
    k = sparse_rejection._TARGET_TOP_K
    exact = exact.to(device)
    assert torch.equal(ids[exact, :k], dense_ids[exact, :k])
    assert torch.equal(values[exact, :k], dense_values[exact, :k])
    # Rows without penalties are untouched.
    unpenalized = slice(2 * ROWS, 3 * ROWS)
    assert torch.equal(values[unpenalized], logits.topk(64, dim=-1).values[unpenalized])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("penalty", ["presence", "all", "heavy"])
@pytest.mark.parametrize("use_fp64", [False, True])
@torch.inference_mode()
def test_compact_rejection_with_penalties_samples_dense_tokens(
    monkeypatch, penalty, use_fp64
):
    """Same seeds, same tokens as dense penalties + top-k/top-p rejection."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Compact target rejection is SM70-only")
    device = "cuda"
    generator = torch.Generator().manual_seed(20261107)
    steps = ROWS - 1
    mapping_np = np.array([4, 2])
    mapping = torch.tensor(mapping_np, dtype=torch.int32, device=device)
    expanded = mapping.repeat_interleave(ROWS)
    local = torch.arange(ROWS, dtype=torch.int32, device=device).repeat(2)
    rows = expanded.numel()
    target = (torch.randn(rows, VOCAB, generator=generator) * 2).to(device)
    top = target.topk(64, dim=-1).indices
    # Draft step j proposes the next row's input from target row j's leading
    # candidates, so the verifier accepts some steps.
    draft_ids = torch.zeros(SLOTS, steps, 16, dtype=torch.int64, device=device)
    for index, slot in enumerate(mapping_np):
        draft_ids[slot] = top[index * ROWS : (index + 1) * ROWS - 1, :16]
    draft_scores = torch.randn(SLOTS, steps, 16, generator=generator).to(device)
    draft_dense = torch.full((SLOTS, steps, VOCAB), -float("inf"), device=device)
    draft_dense.scatter_(2, draft_ids, draft_scores)
    pick = torch.randint(0, 2, (rows,), generator=generator)
    inputs = top[torch.arange(rows), pick.to(device)].roll(1)
    positions = local.long() + 4096
    cu_np = np.array([0, ROWS, 2 * ROWS], dtype=np.int32)
    cu = torch.tensor(cu_np, device=device)
    temp_np = np.full(SLOTS, 0.7, dtype=np.float32)
    p_np = np.full(SLOTS, 0.8, dtype=np.float32)
    temperatures = torch.tensor(temp_np, device=device)
    top_ps = torch.tensor(p_np, device=device)
    seeds = torch.arange(SLOTS, device=device, dtype=torch.int64) + 20261107

    name = "presence" if penalty == "heavy" else penalty
    state = _penalties_state({4: PENALTIES[name], 2: PENALTIES["presence"]}, device)
    for index, slot in enumerate(mapping_np):
        candidates = top[index * ROWS : (index + 1) * ROWS].flatten().cpu()
        _mark(state, slot, candidates, generator)
    if penalty == "heavy":
        state.presence_penalty.gpu[4] = 8.0
        state.output_bin_counts[4, top[:ROWS].flatten()] = 1

    def process(logits, expanded_idx, req_idx_np, pos, draft, local_pos):
        result = _dense_penalized(logits, state, expanded_idx, draft, local_pos)
        apply_temperature(result, expanded_idx, temperatures)
        return apply_top_k_top_p(
            result,
            torch.full((result.shape[0],), 20, dtype=torch.int32, device=device),
            top_ps[expanded_idx],
        )

    states = SimpleNamespace(
        vocab_size=VOCAB,
        temperature=SimpleNamespace(np=temp_np, gpu=temperatures),
        top_p=SimpleNamespace(np=p_np, gpu=top_ps),
        seeds=SimpleNamespace(gpu=seeds),
    )
    sampler = SimpleNamespace(
        sampling_states=states,
        use_fp64_gumbel=use_fp64,
        apply_sampling_params=process,
        penalties_state=state,
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
        logits_indices=torch.arange(rows, device=device),
        input_ids=inputs,
        positions=positions,
        num_tokens=rows,
    )

    class Speculator:
        draft_logits = draft_dense
        _trace = SimpleNamespace(value=lambda name: False)

        def get_sparse_draft_logits(self):
            return draft_ids, draft_scores

    monkeypatch.setattr(sparse_rejection, "DFlash2Speculator", Speculator)
    monkeypatch.setattr(sparse_rejection, "sm70_dflash2_enabled", lambda *a: True)
    monkeypatch.setattr(
        sparse_rejection, "_supports_sparse_sampling_contract", lambda *a, **k: True
    )
    fallback = Mock(return_value=target)

    def project(hidden, k):
        values, ids = target.topk(k, dim=-1)
        return ids, values, fallback

    model = SimpleNamespace(
        get_topk_tokens_and_logits=lambda hidden, k: project(hidden, k)[:2],
        get_topk_tokens_and_logits_with_fallback=project,
    )
    accepted = 0
    for _ in range(16):
        seeds.add_(17)
        actual = sparse_rejection.try_dflash2_sparse_target_rejection(
            model,
            Speculator(),
            rejection,
            torch.empty(rows, 4, device=device),
            batch,
            None,
            allow_graph=False,
        )
        expected, count = rejection_sample(
            process(target, expanded, mapping_np, positions, inputs, local),
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
        # Request 2 (presence 1.5) stays compact, so the round never hands the
        # whole batch back; a flagged request is sampled densely inside.
        assert not isinstance(actual, sparse_rejection.DFlash2LogitsFallback)
        assert torch.equal(actual.num_sampled, count)
        valid = torch.arange(ROWS, device=device)[None] < count[:, None]
        assert torch.equal(actual.sampled_token_ids[valid], expected[valid])
        accepted += int(count.sum())
    assert accepted > 2 * 16
    if penalty == "heavy":
        # Request 4 penalized every probed token: its cutoff needs the reference.
        assert fallback.call_count == 16
