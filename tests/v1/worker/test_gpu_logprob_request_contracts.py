# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded prompt/speculative logprob regressions; no model initialization."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample import prompt_logprob as prompt_module
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    PromptLogprobsWorker,
    get_prompt_logprobs_token_ids,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA runtime operators"
)


@pytest.mark.parametrize(
    "width,start,query_len,mixed",
    [(8, 0, 4, False), (4, 0, 4, False), (8, 2, 2, True), (8, 0, 2, False)],
)
def test_prompt_logprobs_do_not_read_unknown_next_tokens(
    width, start, query_len, mixed, monkeypatch
):
    worker = PromptLogprobsWorker(2)
    worker.add_request("prompt", 1, SamplingParams(prompt_logprobs=1))
    worker.add_request("decode", 0, SamplingParams())
    # The extra storage row makes the old out-of-view read safe to observe.
    # Never send poisoned IDs to the logprob gather on the unfixed source.
    storage = torch.full((3, width), -91, dtype=torch.int32, device="cuda")
    storage[1, :4] = torch.tensor([1, 2, 3, 4], device="cuda")
    storage[0, :4] = torch.tensor([5, 6, 7, 8], device="cuda")
    slots = [1, 0] if mixed else [1]
    lengths = [query_len, 1] if mixed else [query_len]
    cu = np.array([0, *np.cumsum(lengths)], dtype=np.int32)
    batch = SimpleNamespace(
        req_ids=["prompt", "decode"] if mixed else ["prompt"],
        idx_mapping_np=np.array(slots, dtype=np.int32),
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        num_tokens=sum(lengths),
        num_scheduled_tokens=np.array(lengths, dtype=np.int32),
        query_start_loc=torch.tensor(cu, device="cuda"),
        query_start_loc_np=cu,
    )
    expected_ids = [
        p + 1 if p < 4 else 0 for p in range(start + 1, start + query_len + 1)
    ]
    if mixed:
        expected_ids.append(0)
    real_compute = prompt_module.compute_prompt_logprobs_with_chunking

    def checked_compute(token_ids, *args):
        assert token_ids.tolist() == expected_ids
        return real_compute(token_ids, *args)

    monkeypatch.setattr(
        prompt_module, "compute_prompt_logprobs_with_chunking", checked_compute
    )
    logits = torch.arange(128, device="cuda", dtype=torch.float32).repeat(
        sum(lengths), 1
    )
    output = worker.compute_prompt_logprobs(
        lambda x: x,
        logits,
        batch,
        storage[:2],
        torch.tensor([3, start], dtype=torch.int32, device="cuda"),
        np.array([2, 4], dtype=np.int32),
        np.array([4, 4], dtype=np.int32),
        np.array([3, start], dtype=np.int32),
        torch.tensor([2, 4], dtype=torch.int32, device="cuda"),
    )
    if start + query_len < 4:
        assert output == {}
    else:
        assert list(output) == ["prompt"]
        result = output["prompt"]
        assert result.logprob_token_ids[:, 0].tolist() == expected_ids[: query_len - 1]
        expected = torch.log_softmax(logits[: query_len - 1], dim=-1).gather(
            1, result.logprob_token_ids.long()
        )
        torch.testing.assert_close(result.logprobs, expected, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("explicit_count", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("num_drafts,reject_first", [(0, False), (2, False), (2, True)])
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_rejection_sampler_preserves_custom_logprob_ids(
    explicit_count, mixed, num_drafts, reject_first, mode
):
    device = torch.device("cuda")
    reqs = RequestState(2, 16, 8, 2, 128, device)
    sampler = Sampler(2, 128, device, reqs, logprobs_mode=mode)
    custom_ids = [13, 17, 19]
    spec_config = SimpleNamespace(
        num_speculative_tokens=2, rejection_sample_method="standard"
    )
    slots = []
    for i in range(2 if mixed else 1):
        name = str(i)
        reqs.add_request(name, 2, [1, 5], 2, 8)
        slot = reqs.req_id_to_index[name]
        slots.append(slot)
        params = (
            SamplingParams(
                temperature=0,
                logprobs=3 if explicit_count else None,
                logprob_token_ids=custom_ids,
                allowed_token_ids=[7, 9, 11, *custom_ids],
            )
            if i == 0
            else SamplingParams(temperature=0, logprobs=2)
        )
        params._validate_logprobs(
            SimpleNamespace(max_logprobs=20, get_vocab_size=lambda: 128)
        )
        params._validate_spec_decode(spec_config)
        sampler.add_request(slot, 2, params)
    reqs.apply_staged_writes()
    sampler.apply_staged_writes()
    first_rows = num_drafts + 1
    inputs = [5, 7, 9][:first_rows] + ([5] if mixed else [])
    if reject_first:
        inputs[1] = 9
    winners = [7, 9, 11][:first_rows] + ([23] if mixed else [])
    expanded = [slots[0]] * first_rows + ([slots[1]] if mixed else [])
    cu = np.array(
        [0, first_rows, first_rows + 1] if mixed else [0, first_rows], dtype=np.int32
    )
    batch = SimpleNamespace(
        idx_mapping_np=np.array(slots, dtype=np.int32),
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.tensor(expanded, dtype=torch.int32, device=device),
        expanded_local_pos=torch.tensor(
            list(range(first_rows)) + ([0] if mixed else []),
            dtype=torch.int32,
            device=device,
        ),
        input_ids=torch.tensor(inputs, dtype=torch.int32, device=device),
        positions=torch.tensor(
            list(range(1, first_rows + 1)) + ([1] if mixed else []), device=device
        ),
        logits_indices=torch.arange(len(inputs), device=device),
        cu_num_logits=torch.tensor(cu, device=device),
        cu_num_logits_np=cu,
    )
    logits = torch.linspace(-3, 3, 128, device=device).repeat(len(inputs), 1)
    logits[
        torch.arange(len(inputs), device=device), torch.tensor(winners, device=device)
    ] = 8
    result = RejectionSampler(sampler, spec_config, device)(logits, batch)
    emitted = 1 if reject_first else first_rows
    assert result.num_sampled.tolist() == ([emitted, 1] if mixed else [emitted])
    assert result.sampled_token_ids[0, :emitted].tolist() == winners[:emitted]
    output = result.logprobs_tensors
    assert output is not None
    assert (
        output.logprob_token_ids[:first_rows, 1:4].tolist() == [custom_ids] * first_rows
    )
    selected = winners.copy()
    selected[emitted:first_rows] = [0] * (first_rows - emitted)
    assert output.logprob_token_ids[:, 0].tolist() == selected
    assert output.cu_num_generated_tokens == (cu.tolist() if num_drafts else None)
    reference = logits.clone()
    if mode == "processed_logprobs":
        keep = torch.zeros(128, dtype=torch.bool, device=device)
        keep[[7, 9, 11, *custom_ids]] = True
        reference[:first_rows, ~keep] = -torch.inf
    expected = torch.log_softmax(reference, dim=-1).gather(
        1, output.logprob_token_ids.long()
    )
    valid = torch.ones_like(expected, dtype=torch.bool)
    if mixed:
        # Batch-wide top-k can exceed this request's own count (the frontend
        # truncates it); only columns beyond batch-wide top-k are sentinels.
        topk = 3 if explicit_count else 2
        assert (
            output.logprob_token_ids[-1, 1 : topk + 1].tolist()
            == logits[-1].topk(topk).indices.tolist()
        )
        valid[-1, topk + 1 :] = False
        assert torch.isneginf(output.logprobs[-1, topk + 1 :]).all()
    torch.testing.assert_close(
        output.logprobs[valid], expected[valid], atol=2e-5, rtol=1e-5
    )


@pytest.mark.parametrize("max_len", [65536, 131072, 262144])
@pytest.mark.parametrize("spare", [0, 1])
def test_prompt_token_lookup_long_context_graph_boundary(max_len, spare):
    prompt_len = max_len - spare
    storage = torch.full((1, max_len), -91, dtype=torch.int32, device="cuda")
    storage[0, prompt_len - 4 : prompt_len] = torch.tensor([5, 7, 9, 11], device="cuda")
    computed = torch.tensor([prompt_len - 4], dtype=torch.int32, device="cuda")
    prompt_lens = torch.tensor([prompt_len], dtype=torch.int32, device="cuda")
    mapping = torch.zeros(1, dtype=torch.int32, device="cuda")
    query_start = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    get_prompt_logprobs_token_ids(
        4, query_start, mapping, computed, storage, prompt_lens
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids = get_prompt_logprobs_token_ids(
            4, query_start, mapping, computed, storage, prompt_lens
        )
    for start, expected in [
        (prompt_len - 4, [7, 9, 11, 0]),
        (prompt_len - 3, [9, 11, 0, 0]),
    ]:
        computed.fill_(start)
        graph.replay()
        assert ids.tolist() == expected
