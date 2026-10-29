# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staggered parallel sampling regressions, adapted from vllm-project/vllm#62726.

Source: https://github.com/vllm-project/vllm/pull/62726 (wguo7).
"""

import asyncio
import json
import time
from typing import Any

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

PROMPT = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_output_processor_merge_keeps_prompt_logprobs(flat_logprobs):
    op = OutputProcessor(tokenizer=None, log_stats=False)
    n_plp = len(PROMPT) - 1
    params = SamplingParams(
        n=2,
        logprobs=1,
        prompt_logprobs=1,
        detokenize=False,
        flat_logprobs=flat_logprobs,
        output_kind=RequestOutputKind.DELTA,
    )

    def make_request(request_id, sp):
        return EngineCoreRequest(
            request_id=request_id,
            external_req_id="parent",
            prompt_token_ids=PROMPT,
            mm_features=None,
            sampling_params=sp,
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )

    parent = ParentRequest(make_request("parent", params))
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id="parent")
    child_ids = []
    for idx in range(2):
        cid, cparams = parent.get_child_info(idx)
        child_ids.append(cid)
        op.add_request(make_request(cid, cparams), None, parent, idx, queue)

    def out(child, token, first):
        plp = None
        if first:
            plp = LogprobsTensors(
                torch.tensor([[t, 100] for t in PROMPT[1:]]),
                torch.full((n_plp, 2), -2.0),
                torch.ones(n_plp, dtype=torch.long),
            )
        return EngineCoreOutput(
            request_id=child_ids[child],
            new_token_ids=[token],
            new_logprobs=LogprobsLists(
                np.array([[token, 100]]),
                np.array([[-0.5, -1.5]], dtype=np.float32),
                np.array([1]),
            ),
            new_prompt_logprobs_tensors=plp,
        )

    op.process_outputs([out(0, 10, True)], time.monotonic(), None)
    first = queue.get_nowait()
    assert [o.index for o in first.outputs] == [0]
    assert len(first.prompt_logprobs) == len(PROMPT)

    # child 0's 2nd token and child 1's 1st token in the same engine step
    op.process_outputs([out(0, 11, False), out(1, 20, True)], time.monotonic(), None)
    merged = queue.get_nowait()
    assert [o.index for o in merged.outputs] == [0, 1]
    assert merged.prompt_logprobs is not None, "child 1 prompt logprobs dropped"
    assert len(merged.prompt_logprobs) == len(PROMPT)


def _serving():
    s = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    s.enable_force_include_usage = False
    s.return_tokens_as_token_ids = False
    s.system_fingerprint = None
    s.enable_prompt_tokens_details = False
    return s


def _ro(outputs, prompt_logprobs, finished=False):
    return RequestOutput(
        request_id="parent",
        prompt="Test prompt",
        prompt_token_ids=PROMPT,
        prompt_logprobs=prompt_logprobs,
        outputs=outputs,
        finished=finished,
    )


def _co(index, token, finish_reason=None):
    return CompletionOutput(
        index=index,
        text=f"<{token}>",
        token_ids=[token],
        cumulative_logprob=None,
        logprobs=[{token: Logprob(logprob=-0.5)}],
        finish_reason=finish_reason,
    )


def test_completion_stream_echo_logprobs_n2():
    plp = [None] + [{t: Logprob(logprob=-1.0)} for t in PROMPT[1:]]

    async def results():
        # what the API server reads from the shared RequestOutputCollector
        q = RequestOutputCollector(RequestOutputKind.DELTA, request_id="parent")
        q.put(_ro([_co(0, 100)], plp))
        yield 0, q.get_nowait()
        q.put(_ro([_co(0, 101, "length")], None))  # child 0, 2nd delta
        q.put(_ro([_co(1, 200)], plp))  # child 1, 1st delta, same step
        yield 0, q.get_nowait()
        q.put(_ro([_co(1, 201, "length")], None, finished=True))
        yield 0, q.get_nowait()

    req = CompletionRequest(
        model="m",
        prompt="Test prompt",
        n=2,
        echo=True,
        logprobs=1,
        max_tokens=2,
        stream=True,
        return_tokens_as_token_ids=True,
    )

    async def run():
        gen = _serving().completion_stream_generator(
            req,
            [{"prompt_token_ids": PROMPT}],
            results(),
            "cmpl-test",
            0,
            "m",
            1,
            None,
            RequestResponseMetadata(request_id="cmpl-test"),
        )
        return [c async for c in gen]

    chunks = asyncio.run(run())  # the generator itself never raises
    assert chunks[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(c.removeprefix("data: ")) for c in chunks[:-1]]
    errors = [p for p in payloads if "error" in p]
    assert errors == [], errors
    first: dict[int, Any] = {}
    for p in payloads:
        for ch in p["choices"]:
            first.setdefault(ch["index"], ch)
    assert set(first) == {0, 1}
    for ch in first.values():
        assert ch["text"].startswith("Test prompt")
        assert len(ch["logprobs"]["tokens"]) == len(ch["logprobs"]["token_logprobs"])


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.CUMULATIVE, RequestOutputKind.FINAL_ONLY]
)
def test_nonstream_parallel_prompt_logprobs_unchanged(output_kind):
    plp = [None] + [{t: Logprob(logprob=-1.0)} for t in PROMPT[1:]]
    queue = RequestOutputCollector(output_kind, request_id="parent")
    queue.put(_ro([_co(0, 100, "length")], plp))
    queue.put(_ro([_co(1, 200, "length")], plp, finished=True))
    result = queue.get_nowait()
    assert result.prompt_logprobs is plp
    assert [o.index for o in result.outputs] == [0, 1]
    assert [list(o.token_ids) for o in result.outputs] == [[100], [200]]
    assert [o.logprobs[0][int(o.token_ids[0])].logprob for o in result.outputs] == [
        -0.5,
        -0.5,
    ]
    assert result.finished
