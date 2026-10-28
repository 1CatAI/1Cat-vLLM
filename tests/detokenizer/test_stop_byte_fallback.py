# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (
    FastIncrementalDetokenizer,
    SlowIncrementalDetokenizer,
)


@pytest.mark.parametrize(
    "detokenizer_cls", [FastIncrementalDetokenizer, SlowIncrementalDetokenizer]
)
@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("split", [0, 1, 2, 3])
def test_stop_after_incomplete_utf8_bytes(detokenizer_cls, include, split):
    # A real tokenizer decoder buffers incomplete UTF-8 rather than emitting
    # one character per token. The final token also contains text after stop.
    backend = Tokenizer(
        models.WordLevel(
            {
                "a": 0,
                "<0xE4>": 1,
                "<0xB8>": 2,
                "<0xAD>": 3,
                "!tail": 4,
                "x": 5,
            }
        )
    )
    backend.decoder = decoders.ByteFallback()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    request = EngineCoreRequest(
        request_id="bytes",
        external_req_id="bytes",
        prompt_token_ids=[],
        mm_features=None,
        sampling_params=SamplingParams(
            stop=["中!"],
            include_stop_str_in_output=include,
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    detokenizer = detokenizer_cls(tokenizer, request)
    ids = [0, 1, 2, 3, 4, 5]
    assert detokenizer.update(ids[:split], False) is None
    assert detokenizer.update(ids[split:], False) == "中!"
    assert detokenizer.output_text == ("a中!" if include else "a")
    assert detokenizer.output_token_ids == ids[:5]
