# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import _sm70_ops  # noqa: F401
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.dflash2 import sampler_graph, sparse_rejection
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.states import RequestState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("requests", [1, 3])
@pytest.mark.parametrize("use_fp64", [False, True])
@torch.inference_mode()
def test_graph_matches_existing_rejection_with_live_slots(
    default_vllm_config, monkeypatch, use_fp64, requests
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 graph sampling")
    torch.manual_seed(20261004)
    width, vocab, slots, steps = 8, 248320, 8, 7
    rows = requests * width
    target = torch.full((rows, vocab), -20.0, device="cuda")
    hidden = torch.empty(rows, 4, device="cuda", dtype=torch.float16)
    inputs = torch.arange(rows, device="cuda", dtype=torch.int32)
    positions = inputs.long() + 8192
    draft_ids = torch.arange(16, device="cuda").expand(slots, steps, 16).contiguous()
    draft_values = torch.randn(slots, steps, 16, device="cuda") * 0.5
    draft_dense = torch.full((slots, steps, vocab), -float("inf"), device="cuda")
    draft_dense.scatter_(2, draft_ids, draft_values)

    req_states = RequestState(slots, 16384, rows, steps, vocab, torch.device("cuda"))
    sampler = Sampler(
        slots,
        vocab,
        torch.device("cuda"),
        req_states,
        num_speculative_tokens=steps,
        use_fp64_gumbel=use_fp64,
    )
    for slot in range(slots):
        sampler.add_request(
            slot,
            0,
            SamplingParams(temperature=0.7, top_k=20, top_p=0.9, seed=20261004 + slot),
        )
    sampler.apply_staged_writes()
    states = sampler.sampling_states
    rejection = SimpleNamespace(
        num_speculative_steps=steps,
        sampler=sampler,
    )

    class Speculator:
        draft_logits = draft_dense
        _trace = SimpleNamespace(value=lambda name: False)

        def get_sparse_draft_logits(self):
            return draft_ids, draft_values

    def project(hidden, k, *, local_logits_transform=None):
        logits = target.clone()
        if local_logits_transform is not None:
            local_logits_transform(logits, 0)
        values, ids = logits.topk(k, dim=-1)
        return ids, values, lambda: target

    model = SimpleNamespace(
        get_topk_tokens_and_logits=lambda h, k: project(h, k)[:2],
        get_topk_tokens_and_logits_with_fallback=project,
    )
    speculator = Speculator()
    monkeypatch.setattr(
        sampler_graph, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(sampler_graph, "graph_capture", lambda **kw: nullcontext())
    monkeypatch.setattr(sparse_rejection, "DFlash2Speculator", Speculator)
    monkeypatch.setattr(sparse_rejection, "sm70_dflash2_enabled", lambda *a: True)
    monkeypatch.setattr(
        sparse_rejection, "_supports_sparse_sampling_contract", lambda *a, **k: True
    )
    monkeypatch.setattr(sampler_graph, "_all_ranks_support", lambda status: status == 0)
    original_graph_try = sampler_graph.try_graph_rejection
    previous = None
    for iteration in range(20):
        # Replacing all input/proposal tensors must reuse the same graph and
        # preserve the current proposal distribution, not a captured address.
        hidden = torch.randn_like(hidden)
        inputs = inputs.clone()
        positions = positions.clone()
        draft_ids = draft_ids.clone()
        draft_values = draft_values.clone() + 0.05
        draft_dense = torch.full_like(draft_dense, -float("inf"))
        draft_dense.scatter_(2, draft_ids, draft_values)
        speculator.draft_logits = draft_dense
        mapping_np = np.roll(np.array([7, 2, 5], dtype=np.int32), iteration)[:requests]
        mapping = torch.tensor(mapping_np, device="cuda")
        local = torch.arange(width, dtype=torch.int32, device="cuda").repeat(requests)
        cu_num_logits_np = np.arange(requests + 1, dtype=np.int32) * width
        batch = SimpleNamespace(
            num_reqs=requests,
            num_tokens=rows,
            req_ids=[f"request-{iteration // 10}-{r}" for r in range(requests)],
            has_structured_output_reqs=False,
            idx_mapping_np=mapping_np,
            idx_mapping=mapping,
            expanded_idx_mapping=mapping.repeat_interleave(width),
            expanded_local_pos=local,
            input_ids=inputs,
            positions=positions,
            logits_indices=torch.arange(rows, device="cuda"),
            cu_num_logits_np=cu_num_logits_np,
            cu_num_logits=torch.tensor(cu_num_logits_np, device="cuda"),
        )
        target.fill_(-20)
        target[:, :21] = torch.arange(21, 0, -1, device="cuda") / 8
        target[-1, : [21, 24, 63, 64, 80][iteration % 5]] = 2.0
        positions.add_(11)
        # Real UVA parameter pools rotate addresses on each update. A graph
        # must see the current seed/parameters, including high request slots.
        states.seeds.np += 17
        sampler.apply_staged_writes()
        actual = original_graph_try(
            model, speculator, rejection, hidden, batch, (draft_ids, draft_values)
        )
        assert actual is not None
        # Compare with the existing host-decision route, using exactly the
        # same compact and dense operators and the same request random stream.
        monkeypatch.setattr(sampler_graph, "try_graph_rejection", lambda *a: None)
        expected = sparse_rejection.try_dflash2_sparse_target_rejection(
            model,
            speculator,
            rejection,
            hidden,
            batch,
            None,
        )
        if isinstance(expected, sparse_rejection.DFlash2LogitsFallback):
            tokens, count = rejection_sample(
                sampler.apply_sampling_params(
                    target.clone(),
                    batch.expanded_idx_mapping,
                    batch.idx_mapping_np,
                    positions,
                    inputs,
                    local,
                ),
                draft_dense,
                inputs,
                batch.cu_num_logits,
                positions,
                mapping,
                batch.expanded_idx_mapping,
                local,
                states.temperature.gpu,
                states.seeds.gpu,
                steps,
                use_fp64=use_fp64,
            )
        else:
            tokens, count = expected.sampled_token_ids, expected.num_sampled
        assert torch.equal(actual.num_sampled, count)
        valid = torch.arange(width, device="cuda")[None] < count[:, None]
        assert torch.equal(actual.sampled_token_ids[valid], tokens[valid])
        if previous is not None:
            output, saved = previous
            assert torch.equal(output, saved), "Replay overwrote outstanding output"
        previous = (actual.sampled_token_ids, actual.sampled_token_ids.clone())
    # Fresh request metadata allocations and changing slots do not recapture.
    assert len(rejection._sm70_dflash2_rejection_graphs) == 1
    assert sum(rejection.sm70_dflash2_reference_counts.cpu().tolist()) == 10
