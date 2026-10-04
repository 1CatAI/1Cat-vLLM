# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import _sm70_ops  # noqa: F401
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.spec_decode.dflash2 import sampler_graph, sparse_rejection
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_fp64", [False, True])
@torch.inference_mode()
def test_graph_matches_existing_rejection_with_live_slots(monkeypatch, use_fp64):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 graph sampling")
    torch.manual_seed(20261004)
    rows, vocab, slots, steps = 8, 248320, 8, 7
    target = torch.full((rows, vocab), -20.0, device="cuda")
    hidden = torch.empty(rows, 4, device="cuda", dtype=torch.float16)
    inputs = torch.arange(rows, device="cuda", dtype=torch.int32)
    positions = inputs.long() + 8192
    temperature_np = np.full(slots, 0.7, dtype=np.float32)
    top_p_np = np.full(slots, 0.9, dtype=np.float32)
    temperatures = torch.tensor(temperature_np, device="cuda")
    top_ps = torch.tensor(top_p_np, device="cuda")
    seeds = torch.arange(slots, device="cuda", dtype=torch.int64) + 20261004
    draft_ids = torch.arange(16, device="cuda").expand(slots, steps, 16).contiguous()
    draft_values = torch.randn(slots, steps, 16, device="cuda") * 0.5
    draft_dense = torch.full((slots, steps, vocab), -float("inf"), device="cuda")
    draft_dense.scatter_(2, draft_ids, draft_values)

    penalty = SimpleNamespace(use_penalty=np.zeros(slots, dtype=bool))

    def process(logits, mapping, *params):
        result = logits.clone()
        apply_temperature(result, mapping, temperatures)
        return apply_top_k_top_p(
            result,
            torch.full((rows,), 20, device="cuda", dtype=torch.int32),
            top_ps[mapping],
        )

    states = SimpleNamespace(
        vocab_size=vocab,
        temperature=SimpleNamespace(np=temperature_np, gpu=temperatures),
        top_p=SimpleNamespace(np=top_p_np, gpu=top_ps),
        seeds=SimpleNamespace(gpu=seeds),
    )
    rejection = SimpleNamespace(
        num_speculative_steps=steps,
        sampler=SimpleNamespace(
            sampling_states=states,
            use_fp64_gumbel=use_fp64,
            apply_sampling_params=process,
            penalties_state=penalty,
        ),
    )

    class Speculator:
        draft_logits = draft_dense

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
        sparse_rejection, "_supports_sparse_sampling_contract", lambda *a: True
    )
    monkeypatch.setattr(sparse_rejection.envs, "VLLM_SPEC_DUMP_ALIGNMENT", False)
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
        slot = [7, 2, 5][iteration % 3]
        mapping_np = np.array([slot], dtype=np.int32)
        mapping = torch.tensor(mapping_np, device="cuda")
        local = torch.arange(rows, dtype=torch.int32, device="cuda")
        batch = SimpleNamespace(
            num_reqs=1,
            num_tokens=rows,
            req_ids=[f"request-{iteration // 10}"],
            has_structured_output_reqs=False,
            idx_mapping_np=mapping_np,
            idx_mapping=mapping,
            expanded_idx_mapping=mapping.expand(rows).contiguous(),
            expanded_local_pos=local,
            input_ids=inputs,
            positions=positions,
            cu_num_logits_np=np.array([0, rows], dtype=np.int32),
            cu_num_logits=torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
        )
        target.fill_(-20)
        target[:, :21] = torch.arange(21, 0, -1, device="cuda") / 8
        target[-1, : [21, 24, 63, 64, 80][iteration % 5]] = 2.0
        positions.add_(11)
        seeds.add_(17)
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
                process(target, batch.expanded_idx_mapping),
                draft_dense,
                inputs,
                batch.cu_num_logits,
                positions,
                mapping,
                batch.expanded_idx_mapping,
                local,
                temperatures,
                seeds,
                steps,
                use_fp64=use_fp64,
            )
        else:
            tokens, count = expected.sampled_token_ids, expected.num_sampled
        assert torch.equal(actual.num_sampled, count)
        valid = torch.arange(rows, device="cuda")[None] < count[:, None]
        assert torch.equal(actual.sampled_token_ids[valid], tokens[valid])
        if previous is not None:
            output, saved = previous
            assert torch.equal(output, saved), "Replay overwrote outstanding output"
        previous = (actual.sampled_token_ids, actual.sampled_token_ids.clone())
    # Fresh request metadata allocations and changing slots do not recapture.
    assert len(rejection._sm70_dflash2_rejection_graphs) == 1
    assert sum(rejection.sm70_dflash2_reference_counts.cpu().tolist()) == 10
