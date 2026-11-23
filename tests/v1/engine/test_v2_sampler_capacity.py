# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2 keeps allowed token ids, logit bias, the stop ids masked by
min_tokens and the tokenized bad words in fixed-width rows per request. A
request over one of those limits must get a validation error from the input
processor: once it reaches the workers, every worker fails and the engine
dies."""

from unittest.mock import MagicMock

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import (
    MAX_BAD_WORDS_TOTAL_TOKENS,
    MAX_NUM_ALLOWED_TOKEN_IDS,
    MAX_NUM_BAD_WORDS,
    MAX_NUM_LOGIT_BIAS_TOKENS,
    MAX_NUM_STOP_TOKEN_IDS,
    SamplingParams,
)
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor

EOS = 0


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


class _Tokenizer:
    """One token per character. A leading space is folded into the first
    token, so, as with a BPE tokenizer, every bad word is stored twice."""

    max_token_id = 1999

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" "):
            return [1000 + ord(text[1]), *map(ord, text[2:])]
        return list(map(ord, text))


def _words(n: int) -> list[str]:
    return [f"w{i:03d}" for i in range(n)]


# (parameter, kwargs at the limit, kwargs one past it)
CASES = [
    (
        "allowed_token_ids",
        dict(allowed_token_ids=list(range(MAX_NUM_ALLOWED_TOKEN_IDS))),
        dict(allowed_token_ids=list(range(MAX_NUM_ALLOWED_TOKEN_IDS + 1))),
    ),
    (
        "logit_bias",
        dict(logit_bias={i: 1.0 for i in range(MAX_NUM_LOGIT_BIAS_TOKENS)}),
        dict(logit_bias={i: 1.0 for i in range(MAX_NUM_LOGIT_BIAS_TOKENS + 1)}),
    ),
    # The EOS id is added to the user's stop ids before the worker counts them.
    (
        "stop_token_ids",
        dict(min_tokens=1, stop_token_ids=list(range(1, MAX_NUM_STOP_TOKEN_IDS))),
        dict(min_tokens=1, stop_token_ids=list(range(1, MAX_NUM_STOP_TOKEN_IDS + 1))),
    ),
    # Each word is stored with and without a leading space.
    (
        "bad_words",
        dict(bad_words=_words(MAX_NUM_BAD_WORDS // 2)),
        dict(bad_words=_words(MAX_NUM_BAD_WORDS // 2 + 1)),
    ),
    (
        "bad_words",
        dict(bad_words=["a" * (MAX_BAD_WORDS_TOTAL_TOKENS // 2)]),
        dict(bad_words=["a" * (MAX_BAD_WORDS_TOTAL_TOKENS // 2 + 1)]),
    ),
]


def _as_sent_to_the_engine(**kwargs) -> SamplingParams:
    params = SamplingParams(**kwargs)
    params.update_from_generation_config({}, EOS)
    params.update_from_tokenizer(_Tokenizer())
    return params


@pytest.mark.parametrize(("parameter", "at_limit", "over_limit"), CASES)
def test_limits_match_the_worker_rows(parameter, at_limit, over_limit):
    _as_sent_to_the_engine(**at_limit).verify_v2_sampler_capacity()

    with pytest.raises(VLLMValidationError) as error:
        _as_sent_to_the_engine(**over_limit).verify_v2_sampler_capacity()
    assert error.value.parameter == parameter


def test_stop_ids_are_only_stored_with_min_tokens():
    many = list(range(1, 4 * MAX_NUM_STOP_TOKEN_IDS))
    _as_sent_to_the_engine(stop_token_ids=many).verify_v2_sampler_capacity()


def test_generation_config_eos_ids_count_too():
    params = SamplingParams(
        min_tokens=1, stop_token_ids=list(range(1, MAX_NUM_STOP_TOKEN_IDS - 1))
    )
    params.update_from_generation_config({}, EOS)
    params.verify_v2_sampler_capacity()  # 126 + EOS

    params.update_from_generation_config({"eos_token_id": [EOS, 5000, 5001]}, EOS)
    with pytest.raises(VLLMValidationError, match="stop token IDs"):
        params.verify_v2_sampler_capacity()  # 126 + 3 EOS ids


def _input_processor(use_v2_model_runner: bool) -> InputProcessor:
    processor = object.__new__(InputProcessor)
    processor.use_v2_model_runner = use_v2_model_runner
    processor._validate_params = MagicMock()
    processor._validate_lora = MagicMock()
    processor._validate_model_inputs = MagicMock()
    processor.vllm_config = MagicMock()
    processor.vllm_config.parallel_config.data_parallel_size = 1
    processor.vllm_config.parallel_config.data_parallel_size_local = 1
    processor.vllm_config.parallel_config.local_engines_only = False
    processor.model_config = MagicMock()
    processor.model_config.max_model_len = 128
    processor.generation_config_fields = {}
    processor.renderer = MagicMock()
    processor.renderer.get_eos_token_id.return_value = EOS
    processor.renderer.tokenizer = _Tokenizer()
    return processor


@pytest.mark.parametrize(("parameter", "at_limit", "over_limit"), CASES)
@pytest.mark.parametrize("use_v2_model_runner", [True, False])
def test_input_processor_rejects_for_model_runner_v2_only(
    parameter, at_limit, over_limit, use_v2_model_runner
):
    processor = _input_processor(use_v2_model_runner)
    prompt = {"type": "token", "prompt_token_ids": [1, 2, 3]}

    request = processor.process_inputs(
        "at_limit", prompt, SamplingParams(**at_limit), ("generate",)
    )
    assert isinstance(request, EngineCoreRequest)

    if not use_v2_model_runner:
        # The V1 sampler has no per-request rows; nothing changes there.
        processor.process_inputs(
            "over_limit", prompt, SamplingParams(**over_limit), ("generate",)
        )
        return
    with pytest.raises(VLLMValidationError) as error:
        processor.process_inputs(
            "over_limit", prompt, SamplingParams(**over_limit), ("generate",)
        )
    assert error.value.parameter == parameter


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA")
@pytest.mark.parametrize(("parameter", "at_limit", "over_limit"), CASES)
def test_worker_rows_take_what_the_frontend_accepts(parameter, at_limit, over_limit):
    import torch

    from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    logit_bias = LogitBiasState(1, device)
    bad_words = BadWordsState(RequestState(1, 32, 8, 0, 128, device))

    params = _as_sent_to_the_engine(**at_limit)
    logit_bias.add_request(0, 3, params)
    bad_words.add_request(0, params)

    params = _as_sent_to_the_engine(**over_limit)
    with pytest.raises(ValueError):
        logit_bias.add_request(0, 3, params)
        bad_words.add_request(0, params)
