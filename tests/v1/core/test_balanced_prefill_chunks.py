# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler


def _prefill_chunks(prompt_tokens: int, budget: int) -> list[int]:
    scheduler = create_scheduler(max_num_batched_tokens=budget, max_model_len=2048)
    (request,) = create_requests(num_requests=1, num_tokens=prompt_tokens)
    scheduler.add_request(request)
    chunks: list[int] = []
    while sum(chunks) < prompt_tokens:
        output = scheduler.schedule()
        chunks.append(output.num_scheduled_tokens[request.request_id])
        done = sum(chunks) >= prompt_tokens
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[0] if done else []],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    return chunks


@pytest.mark.parametrize(
    "prompt_tokens, budget, balanced",
    [
        (1100, 512, [368, 368, 364]),
        (600, 512, [304, 296]),
        (512, 512, [512]),
        (100, 64, [64, 36]),
    ],
)
def test_balanced_prefill_chunks(monkeypatch, prompt_tokens, budget, balanced):
    assert _prefill_chunks(prompt_tokens, budget)[:1] == [min(budget, prompt_tokens)]
    monkeypatch.setenv("VLLM_BALANCED_PREFILL_CHUNKS", "1")
    chunks = _prefill_chunks(prompt_tokens, budget)
    assert sum(chunks) == prompt_tokens
    assert max(chunks) <= budget
    assert len(chunks) == -(-prompt_tokens // budget)
    assert chunks == balanced
