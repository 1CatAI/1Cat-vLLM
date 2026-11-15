# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import _sm70_ops  # noqa: F401
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode import sm70_sparse_target
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.target_sampling import ComputedTargetLogits
from vllm.v1.worker.gpu.states import RequestState


def _ensure_non_inference_cudagraph_rng_state() -> None:
    buf = torch.empty(1, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        buf.zero_()
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("requests", [1, 3, 8])
@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("with_penalties", [False, True])
def test_greedy_draft_graph_matches_host_rejection_with_live_slots(
    default_vllm_config, monkeypatch, requests, use_fp64, with_penalties
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 graph sampling")
    if with_penalties and (requests != 3 or use_fp64):
        pytest.skip("Cover monotone penalty branch on representative batch")

    # Initialize PyTorch's singleton CUDA generator graph state outside
    # InferenceMode so subsequent non-inference-mode CUDAGraph tests in the
    # same pytest worker can still capture graphs cleanly.
    _ensure_non_inference_cudagraph_rng_state()
    try:
        with torch.inference_mode():
            _run_greedy_draft_graph_test(
                monkeypatch,
                requests=requests,
                use_fp64=use_fp64,
                with_penalties=with_penalties,
            )
    finally:
        gc.collect()


def _run_greedy_draft_graph_test(
    monkeypatch, *, requests: int, use_fp64: bool, with_penalties: bool
) -> None:
    torch.manual_seed(20261115)
    steps = 4
    width = steps + 1
    vocab = 248320
    slots = 10
    rows = requests * width
    padded_tokens = rows + 3

    target = torch.full((rows, vocab), -20.0, device="cuda")
    hidden = torch.empty(rows, 16, device="cuda", dtype=torch.float16)
    inputs_buf = torch.arange(padded_tokens, device="cuda", dtype=torch.int32)
    positions_buf = inputs_buf.long() + 4096

    req_states = RequestState(
        slots, 16384, padded_tokens, steps, vocab, torch.device("cuda")
    )
    req_states.prompt_len.np[:] = 8
    req_states.prefill_len.np[:] = 16
    req_states.prompt_len.copy_to_uva()
    req_states.prefill_len.copy_to_uva()
    req_states.all_token_ids.gpu[:, :16] = torch.arange(16, device="cuda")

    sampler = Sampler(
        slots,
        vocab,
        torch.device("cuda"),
        req_states,
        num_speculative_tokens=steps,
        use_fp64_gumbel=use_fp64,
    )
    for slot in range(slots):
        params = SamplingParams(
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            repetition_penalty=1.1 if with_penalties else 1.0,
            frequency_penalty=0.2 if with_penalties else 0.0,
            presence_penalty=0.1 if with_penalties else 0.0,
            seed=20261115 + slot,
        )
        sampler.add_request(slot, 8, params)
    sampler.apply_staged_writes()
    states = sampler.sampling_states
    rejection = SimpleNamespace(
        rejection_sample_method="standard",
        num_speculative_steps=steps,
        sampler=sampler,
    )

    def project(h, k):
        cur_rows = h.shape[0]
        values, ids = target[:cur_rows].topk(k, dim=-1)
        return ids, values, lambda: target[:cur_rows]

    model = SimpleNamespace(
        get_topk_tokens_and_logits=lambda h, k: project(h, k)[:2],
        get_topk_tokens_and_logits_with_fallback=project,
        get_compact_target_probe_with_fallback=project,
    )
    monkeypatch.setattr(
        sm70_sparse_target, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(sm70_sparse_target, "graph_capture", lambda **kw: nullcontext())
    monkeypatch.setattr(
        sm70_sparse_target, "_all_ranks_support", lambda status: status == 0
    )

    if requests == 3 and not with_penalties and not use_fp64:
        # Verify startup pre-capture populates shared-pool graphs for B=1..3
        # without mutating live request state.
        speculator = SimpleNamespace(
            draft_logits=None,
            device=torch.device("cuda"),
            max_num_reqs=3,
            input_buffers=SimpleNamespace(
                input_ids=inputs_buf,
                positions=positions_buf,
            ),
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(
                    sm70_sparse_target_sampling=True,
                    sm70_sparse_target_reference_check=True,
                ),
                model_config=SimpleNamespace(
                    get_hidden_size=lambda: 16,
                    dtype=torch.float16,
                ),
            ),
        )
        sm70_sparse_target.maybe_capture_sparse_target_graphs(
            speculator, model, rejection
        )
        assert len(rejection._sm70_greedy_target_rejection_graphs) == 3

    slot_order = np.arange(slots, dtype=np.int32)
    previous = None
    for iteration in range(16):
        states.temperature.np.fill([0.6, 1.0, 0.7, 0.9][iteration % 4])
        states.top_p.np.fill([0.95, 1.0, 0.9, 1.0][iteration % 4])
        hidden = torch.randn_like(hidden)
        inputs_buf = (inputs_buf + 3) % 32
        positions_buf = positions_buf.clone() + 7
        mapping_np = np.roll(slot_order, iteration)[:requests].copy()
        mapping = torch.tensor(mapping_np, device="cuda", dtype=torch.int32)
        local = torch.arange(width, dtype=torch.int32, device="cuda").repeat(requests)
        cu_np = np.arange(requests + 1, dtype=np.int32) * width
        batch = SimpleNamespace(
            num_reqs=requests,
            num_tokens=rows,
            req_ids=[f"mtp-{iteration // 8}-{r}" for r in range(requests)],
            is_prefilling_np=np.zeros(requests, dtype=bool),
            has_structured_output_reqs=False,
            idx_mapping_np=mapping_np,
            idx_mapping=mapping,
            expanded_idx_mapping=mapping.repeat_interleave(width),
            expanded_local_pos=local,
            input_ids=inputs_buf,
            positions=positions_buf,
            logits_indices=torch.arange(rows, device="cuda", dtype=torch.int32),
            cu_num_logits_np=cu_np,
            cu_num_logits=torch.tensor(cu_np, device="cuda", dtype=torch.int32),
        )
        target.fill_(-20.0)
        target[:, :25] = torch.arange(25, 0, -1, device="cuda", dtype=torch.float32) / 6
        target[-1, : [21, 24, 63, 64, 80][iteration % 5]] = 3.5
        states.seeds.np += 19
        sampler.apply_staged_writes()

        actual = sm70_sparse_target.try_greedy_draft_sparse_rejection(
            model,
            rejection,
            hidden,
            batch,
            None,
            reference_check=True,
            allow_graph=True,
        )
        expected = sm70_sparse_target.try_greedy_draft_sparse_rejection(
            model,
            rejection,
            hidden,
            batch,
            None,
            reference_check=True,
            allow_graph=False,
        )

        def materialize(res, cur_batch, cur_in, cur_pos, cur_map, cur_local):
            if isinstance(res, ComputedTargetLogits):
                return rejection_sample(
                    sampler.apply_sampling_params(
                        res.logits.clone(),
                        cur_batch.expanded_idx_mapping,
                        cur_batch.idx_mapping_np,
                        cur_pos[:rows],
                        cur_in[:rows],
                        cur_local,
                    ),
                    None,
                    cur_in[:rows],
                    cur_batch.cu_num_logits,
                    cur_pos[:rows],
                    cur_map,
                    cur_batch.expanded_idx_mapping,
                    cur_local,
                    states.temperature.gpu,
                    states.seeds.gpu,
                    steps,
                    use_fp64=use_fp64,
                )
            assert res is not None
            return res.sampled_token_ids, res.num_sampled

        actual_tokens, actual_count = materialize(
            actual, batch, inputs_buf, positions_buf, mapping, local
        )
        tokens, count = materialize(
            expected, batch, inputs_buf, positions_buf, mapping, local
        )

        assert torch.equal(actual_count, count)
        valid = torch.arange(width, device="cuda")[None] < count[:, None]
        assert torch.equal(actual_tokens[valid], tokens[valid])
        if previous is not None and not isinstance(actual, ComputedTargetLogits):
            out_buf, out_copy = previous
            assert torch.equal(out_buf, out_copy)
        if not isinstance(actual, ComputedTargetLogits):
            previous = (actual.sampled_token_ids, actual.sampled_token_ids.clone())

    if requests <= sm70_sparse_target._MAX_GRAPH_REQUESTS:
        counters = (
            torch.stack(
                [
                    g.counters
                    for g in rejection._sm70_greedy_target_rejection_graphs.values()
                ]
            )
            .sum(0)
            .cpu()
            .tolist()
        )
        assert sum(counters) == 8
        assert all(c > 0 for c in counters)
        rejection.__dict__.clear()
    else:
        assert not hasattr(rejection, "_sm70_greedy_target_rejection_graphs")
