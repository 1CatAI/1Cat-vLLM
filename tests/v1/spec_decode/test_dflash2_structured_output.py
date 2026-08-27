# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured-output state safety for DFlash2/speculative draft windows."""

from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

TOKENIZER = "gpt2"
NUM_SPEC_TOKENS = 4


def _make_manager_and_request(prompt_str: str = '{"a": "b"}'):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode(prompt_str)
    config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        speculative_config=SpeculativeConfig(
            model="[ngram]",
            num_speculative_tokens=NUM_SPEC_TOKENS,
        ),
    )
    manager = StructuredOutputManager(config)
    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    assert sampling_params.structured_outputs is not None
    sampling_params.structured_outputs._backend = "xgrammar"
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)
    request = Request(
        "dflash2-structured-output",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    manager.grammar_init(request)
    assert request.structured_output_request is not None
    while not request.structured_output_request._check_grammar_completion():
        pass
    return tokenizer, manager, request, prompt


def _grammar(request: Request):
    assert request.structured_output_request is not None
    grammar = request.structured_output_request.grammar
    assert grammar is not None
    return grammar


def test_dflash2_bonus_row_remains_constrained_after_invalid_draft_padding():
    tokenizer, manager, request, prompt = _make_manager_and_request('{"a"')
    grammar = _grammar(request)
    assert grammar.accept_tokens(request.request_id, prompt)

    drafts = [tokenizer.encode(":")[0], -1, -1, -1]
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == NUM_SPEC_TOKENS + 1
    assert not (bitmask[-1] == -1).all()
    assert not grammar.is_terminated()


def test_dflash2_post_reasoning_invalid_draft_does_not_advance_fsm(caplog):
    tokenizer, manager, request, prompt = _make_manager_and_request("{")
    grammar = _grammar(request)
    assert grammar.accept_tokens(request.request_id, prompt)

    marker = tokenizer.encode("\n")[0]

    class MarkerReasoner:
        def __init__(self, *args, **kwargs):
            pass

        def is_reasoning_end(self, input_ids):
            return marker in list(input_ids)

        def is_reasoning_end_streaming(self, input_ids, delta_ids):
            return marker in list(delta_ids)

    manager.reasoner_cls = MarkerReasoner
    assert request.structured_output_request is not None
    request.structured_output_request.reasoner = MarkerReasoner()
    request.structured_output_request.reasoning_ended = False

    drafts = [tokenizer.encode(" ")[0], marker, tokenizer.encode("z")[0]]
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert (bitmask[0] == -1).all()
    assert (bitmask[1] == -1).all()
    assert not (bitmask[2] == -1).all()
    assert not (bitmask[-1] == -1).all()
    assert not grammar.is_terminated()
    assert "Failed to advance FSM" not in caplog.text


def test_xgrammar_stops_accepting_at_termination(capfd):
    tokenizer, _, request, prompt = _make_manager_and_request()
    grammar = _grammar(request)
    assert grammar.accept_tokens(request.request_id, prompt)

    eos = tokenizer.eos_token_id
    trailing = tokenizer.encode("\n")[0]
    processed_before = grammar.num_processed_tokens

    assert grammar.accept_tokens(request.request_id, [eos, trailing])
    assert grammar.is_terminated()
    assert grammar.num_processed_tokens == processed_before + 1
    assert "trying to accept new token" not in capfd.readouterr().err

    processed_after = grammar.num_processed_tokens
    assert grammar.accept_tokens(request.request_id, [trailing])
    assert grammar.num_processed_tokens == processed_after

    grammar.reset()
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0


def test_xgrammar_validation_rolls_back_at_termination(capfd):
    tokenizer, _, request, prompt = _make_manager_and_request()
    grammar = _grammar(request)
    assert grammar.accept_tokens(request.request_id, prompt)

    eos = tokenizer.eos_token_id
    trailing = tokenizer.encode("\n")[0]
    assert grammar.validate_tokens([eos, trailing]) == [eos]
    assert "trying to accept new token" not in capfd.readouterr().err
    assert not grammar.matcher.is_terminated()

    assert grammar.accept_tokens(request.request_id, [eos])
    assert grammar.is_terminated()
    assert grammar.validate_tokens([trailing]) == []
