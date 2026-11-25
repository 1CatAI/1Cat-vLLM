# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for v1.5.5 evaluation audit fixes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser
from vllm.tool_parsers.qwen_xml_tool_grammar import (
    maybe_inject_qwen_xml_response_format_hint,
)
from vllm.tool_parsers.utils import (
    _params_or_empty_object,
    get_json_schema_from_tools,
    is_closed_empty_object_schema,
)
from vllm.v1.core import kv_cache_utils

ROOT = Path(__file__).resolve().parents[2]


def _weather_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _ping_tool(params: dict | None) -> dict:
    fn: dict = {"name": "ping", "description": "Health check."}
    if params is not None:
        fn["parameters"] = params
    return {"type": "function", "function": fn}


def test_qwen_xml_response_format_hint_injected_only_when_auto_and_json_rf():
    base_messages = [{"role": "user", "content": "What is 2 + 2?"}]
    req_with_rf = ChatCompletionRequest(
        model="qwen3.8-27b",
        messages=base_messages,
        tools=[_weather_tool()],
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    out = maybe_inject_qwen_xml_response_format_hint(
        req_with_rf, base_messages, Qwen3CoderToolParser
    )
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert "raw JSON object" in out[0]["content"]
    # Original list must not be mutated in place.
    assert len(base_messages) == 1

    # Existing leading system prompt is appended cleanly without duplication.
    sys_messages = [{"role": "system", "content": "Be concise."}, *base_messages]
    out_sys = maybe_inject_qwen_xml_response_format_hint(
        req_with_rf, sys_messages, Qwen3XMLToolParser
    )
    assert len(out_sys) == 2
    assert out_sys[0]["content"].startswith("Be concise.\n\n")
    out_idempotent = maybe_inject_qwen_xml_response_format_hint(
        req_with_rf, out_sys, Qwen3XMLToolParser
    )
    assert out_idempotent == out_sys

    # Ordinary tool_choice="auto" WITHOUT response_format must remain 100% untouched.
    req_no_rf = ChatCompletionRequest(
        model="qwen3.8-27b",
        messages=base_messages,
        tools=[_weather_tool()],
        tool_choice="auto",
    )
    assert (
        maybe_inject_qwen_xml_response_format_hint(
            req_no_rf, base_messages, Qwen3CoderToolParser
        )
        is base_messages
    )


def test_empty_parameter_tool_closes_additional_properties_and_whitespace():
    for raw_params in (None, {}, {"type": "object", "properties": {}}):
        schema = _params_or_empty_object(raw_params)
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["additionalProperties"] is False
        assert is_closed_empty_object_schema(schema)

    # Explicit open object schema (additionalProperties=True) must be respected.
    open_schema = _params_or_empty_object(
        {"type": "object", "properties": {}, "additionalProperties": True}
    )
    assert open_schema["additionalProperties"] is True
    assert not is_closed_empty_object_schema(open_schema)

    req = ChatCompletionRequest(
        model="qwen3.8-27b",
        messages=[{"role": "user", "content": "Call ping."}],
        tools=[_ping_tool({"type": "object", "properties": {}})],
        tool_choice={"type": "function", "function": {"name": "ping"}},
    )
    derived = get_json_schema_from_tools(req.tool_choice, req.tools)
    assert is_closed_empty_object_schema(derived)

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {"<tool_call>": 1, "</tool_call>": 2}
    parser = Qwen3CoderToolParser(tokenizer)
    adjusted = parser.adjust_request(req)
    assert adjusted.structured_outputs is not None
    assert adjusted.structured_outputs.disable_any_whitespace is True
    assert adjusted.structured_outputs.json["additionalProperties"] is False


def test_host_kv_history_blocks_auto_caps_on_64gb_host(monkeypatch):
    layout = SimpleNamespace(
        main_kv_names=["layer.0", "layer.1"],
        host_main_kv_names={"layer.0", "layer.1"},
        main_kv_page_sizes=[3 * (1 << 20), 3 * (1 << 20)],  # 6 MiB/rank/block
        compressed_names=[],
        host_compressed_names=set(),
        compressed_page_sizes=[],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            data_parallel_size_local=1,
        )
    )
    # Simulate 64 GiB total RAM with 34 GiB MemAvailable after PLE load:
    # reserve = 16 GiB -> 18 GiB budget / (4 * 6 MiB = 24 MiB/block) = 768 total
    # blocks -> 768 - 176 state_blocks = 592 history blocks (< 768 target).
    monkeypatch.setattr(
        kv_cache_utils,
        "_read_linux_host_memory_bytes",
        lambda: (64 << 30, 34 << 30),
    )
    capped = kv_cache_utils._cap_host_kv_history_blocks_by_ram(
        vllm_config,
        layout,
        state_blocks=176,
        min_history_blocks=129,
        target_history_blocks=768,
    )
    assert capped == 592

    # Simulate 128 GiB host with 80 GiB MemAvailable -> full 768 preserved.
    monkeypatch.setattr(
        kv_cache_utils,
        "_read_linux_host_memory_bytes",
        lambda: (128 << 30, 80 << 30),
    )
    uncapped = kv_cache_utils._cap_host_kv_history_blocks_by_ram(
        vllm_config,
        layout,
        state_blocks=176,
        min_history_blocks=129,
        target_history_blocks=768,
    )
    assert uncapped == 768

    # Never drop below required min_history_blocks.
    monkeypatch.setattr(
        kv_cache_utils,
        "_read_linux_host_memory_bytes",
        lambda: (64 << 30, 17 << 30),
    )
    floored = kv_cache_utils._cap_host_kv_history_blocks_by_ram(
        vllm_config,
        layout,
        state_blocks=176,
        min_history_blocks=129,
        target_history_blocks=768,
    )
    assert floored == 129


def test_release_shell_scripts_syntax_and_27b_int8_parity():
    serve_sm70 = ROOT / "docker" / "sm70_release" / "serve.sh"
    serve_fn = ROOT / "scripts" / "serve_flash_next_nvfp4_v100.sh"
    for script in (serve_sm70, serve_fn):
        subprocess.run(["bash", "-n", str(script)], check=True)
    sm70_text = serve_sm70.read_text()
    assert "--kv-cache-dtype int8_per_token_head" in sm70_text
    assert "--limit-mm-per-prompt '{\"video\": 0}'" in sm70_text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for Triton")
def test_dflash2_selector_walk_applies_penalties_identically_to_target():
    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        _selector_walk_kernel,
        _selector_walk_tail_kernel,
    )

    device = torch.device("cuda:0")
    num_reqs = 1
    num_steps = 7
    walk_steps = 6
    top_k = 16
    block_k = triton.next_power_of_2(top_k)
    vocab_size = 256

    # Candidate 0 (token 10) has slightly higher raw score than candidate 1 (token 20),
    # so without presence_penalty token 10 wins; with presence_penalty=1.5 on token 10
    # (already in output_bin_counts), token 20 wins AND realized_scores reflect -1.5/T.
    scores = torch.full(
        (num_reqs, num_steps, top_k, top_k), -5.0, dtype=torch.float32, device=device
    )
    scores[:, :, :, 0] = 2.0
    scores[:, :, :, 1] = 1.5

    candidates = (
        torch.arange(top_k, dtype=torch.int64, device=device)
        .view(1, 1, top_k)
        .repeat(num_reqs, num_steps, 1)
        .contiguous()
    )
    candidates[:, :, 0] = 10
    candidates[:, :, 1] = 20

    sample_pos = torch.arange(1, num_steps + 1, dtype=torch.int32, device=device).view(
        1, num_steps
    )
    req_state = torch.zeros((num_reqs, num_steps), dtype=torch.int32, device=device)
    temperature = torch.tensor([0.6], dtype=torch.float32, device=device)
    seeds = torch.tensor([42], dtype=torch.int64, device=device)
    tokens_unpen = torch.empty((num_reqs, num_steps), dtype=torch.int32, device=device)
    realized_unpen = torch.empty(
        (num_reqs, num_steps, top_k), dtype=torch.float32, device=device
    )
    path_state = torch.zeros((num_reqs,), dtype=torch.int32, device=device)

    rep_pen = torch.tensor([1.0], dtype=torch.float32, device=device)
    freq_pen = torch.tensor([0.0], dtype=torch.float32, device=device)
    pres_pen = torch.tensor([1.5], dtype=torch.float32, device=device)
    prompt_mask = torch.zeros((1, vocab_size // 32), dtype=torch.int32, device=device)
    out_counts = torch.zeros((1, vocab_size), dtype=torch.int32, device=device)
    out_counts[0, 10] = 2  # token 10 already generated

    _selector_walk_kernel[(num_reqs,)](
        scores,
        candidates,
        sample_pos,
        req_state,
        temperature,
        seeds,
        tokens_unpen,
        realized_unpen,
        path_state,
        num_steps=num_steps,
        walk_steps=walk_steps,
        top_k=top_k,
        BLOCK_K=block_k,
        SAMPLE_PROBABILISTIC=True,
        USE_FP64=False,
        PROPOSAL_TEMPERATURE_SCALE=1.0,
        PROPOSAL_TOP_P=1.0,
        HAS_PENALTIES=False,
        num_warps=1,
    )

    tokens_pen = torch.empty_like(tokens_unpen)
    realized_pen = torch.empty_like(realized_unpen)
    path_state_pen = torch.zeros_like(path_state)
    _selector_walk_kernel[(num_reqs,)](
        scores,
        candidates,
        sample_pos,
        req_state,
        temperature,
        seeds,
        tokens_pen,
        realized_pen,
        path_state_pen,
        num_steps=num_steps,
        walk_steps=walk_steps,
        top_k=top_k,
        BLOCK_K=block_k,
        SAMPLE_PROBABILISTIC=True,
        USE_FP64=False,
        PROPOSAL_TEMPERATURE_SCALE=1.0,
        PROPOSAL_TOP_P=1.0,
        repetition_penalty_ptr=rep_pen,
        frequency_penalty_ptr=freq_pen,
        presence_penalty_ptr=pres_pen,
        prompt_bin_mask_ptr=prompt_mask,
        prompt_bin_mask_stride=prompt_mask.stride(0),
        output_bin_counts_ptr=out_counts,
        output_bin_counts_stride=out_counts.stride(0),
        vocab_size=vocab_size,
        HAS_PENALTIES=True,
        num_warps=1,
    )
    _selector_walk_tail_kernel[(num_reqs,)](
        scores,
        candidates,
        sample_pos,
        req_state,
        temperature,
        seeds,
        tokens_pen,
        realized_pen,
        path_state_pen,
        num_steps=num_steps,
        top_k=top_k,
        BLOCK_K=block_k,
        SAMPLE_PROBABILISTIC=True,
        USE_FP64=False,
        PROPOSAL_TEMPERATURE_SCALE=1.0,
        PROPOSAL_TOP_P=1.0,
        repetition_penalty_ptr=rep_pen,
        frequency_penalty_ptr=freq_pen,
        presence_penalty_ptr=pres_pen,
        prompt_bin_mask_ptr=prompt_mask,
        prompt_bin_mask_stride=prompt_mask.stride(0),
        output_bin_counts_ptr=out_counts,
        output_bin_counts_stride=out_counts.stride(0),
        vocab_size=vocab_size,
        HAS_PENALTIES=True,
        num_warps=1,
    )

    # Step 0: token 10 score drops from 2.0/0.6 to (2.0 - 1.5)/0.6 = 0.5/0.6.
    assert np.isclose(realized_unpen[0, 0, 0].item(), 2.0 / 0.6, atol=1e-4)
    assert np.isclose(realized_pen[0, 0, 0].item(), (2.0 - 1.5) / 0.6, atol=1e-4)
    # Token 20 is unpenalized at step 0, then once chosen at step 0, step 1..6
    # self-penalize token 20 via intra-draft counts as well!
    assert np.isclose(realized_pen[0, 0, 1].item(), 1.5 / 0.6, atol=1e-4)
    if tokens_pen[0, 0].item() == 20:
        assert np.isclose(realized_pen[0, 1, 1].item(), (1.5 - 1.5) / 0.6, atol=1e-4)
    # Tail slot (step 6) also applied presence_penalty to token 10.
    assert np.isclose(realized_pen[0, 6, 0].item(), (2.0 - 1.5) / 0.6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for UVA/Graph")
def test_dflash2_cuda_graph_penalties_stay_fresh_across_uva_rotations():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
    from vllm.v1.worker.gpu.sample.states import SamplingStates
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        DFlash2Speculator,
        _selector_walk_kernel,
    )

    device = torch.device("cuda:0")
    max_num_reqs = 4
    vocab_size = 256
    req_states = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        vocab_size=vocab_size,
        device=device,
        prefill_len=SimpleNamespace(
            np=np.ones(max_num_reqs, dtype=np.int32),
            gpu=torch.ones(max_num_reqs, dtype=torch.int32, device=device),
        ),
        prompt_len=SimpleNamespace(
            gpu=torch.ones(max_num_reqs, dtype=torch.int32, device=device)
        ),
        all_token_ids=SimpleNamespace(
            gpu=torch.ones((max_num_reqs, 8), dtype=torch.int32, device=device)
        ),
    )

    # Every rotating UVA pool slot must be initialized to valid neutral values.
    penalties = PenaltiesState(req_states)
    for buf in penalties.repetition_penalty.pool._uva_bufs:
        assert np.all(buf.np == 1.0)
    sampling_states = SamplingStates(max_num_reqs, vocab_size)
    for buf in sampling_states.top_k.pool._uva_bufs:
        assert np.all(buf.np == vocab_size)
    for buf in sampling_states.top_p.pool._uva_bufs:
        assert np.all(buf.np == 1.0)

    # Wire a lightweight DFlash2Speculator around its graph-stable mirrors.
    spec = object.__new__(DFlash2Speculator)
    spec.device = device
    spec.max_num_reqs = max_num_reqs
    spec.draft_block = 7
    spec.selector_top_k = 16
    spec._use_sm70_tail = False
    spec.use_fp64_gumbel = False
    spec.proposal_temperature_scale = 1.0
    spec.proposal_top_p = 1.0
    spec.draft_logits = torch.empty((1,), device=device)
    spec.sample_pos = torch.arange(1, 8, dtype=torch.int64, device=device)
    spec.sample_idx_mapping = torch.zeros(7, dtype=torch.int32, device=device)
    spec.temperature = torch.full(
        (max_num_reqs,), 1.0, dtype=torch.float32, device=device
    )
    spec.seeds = torch.full((max_num_reqs,), 42, dtype=torch.int64, device=device)
    spec._selector_tokens = torch.zeros((1, 7), dtype=torch.int64, device=device)
    spec._selector_scores = torch.zeros((1, 7, 16), dtype=torch.float32, device=device)
    spec._selector_path_state = torch.zeros((1,), dtype=torch.int32, device=device)
    spec._repetition_penalty = torch.ones(
        max_num_reqs, dtype=torch.float32, device=device
    )
    spec._frequency_penalty = torch.zeros(
        max_num_reqs, dtype=torch.float32, device=device
    )
    spec._presence_penalty = torch.zeros(
        max_num_reqs, dtype=torch.float32, device=device
    )
    spec._penalties_state = None
    spec.set_penalties_state(penalties)

    scores = torch.zeros((1, 7, 16, 16), dtype=torch.float32, device=device)
    scores[:, :, :, 0] = 5.0
    scores[:, :, :, 1] = 4.5
    cands = (
        torch.arange(16, dtype=torch.int64, device=device)
        .view(1, 1, 16)
        .expand(1, 7, 16)
        .contiguous()
    )
    penalties.output_bin_counts[0, 0] = 1

    # Capture _sample_path into a CUDA graph (mirrors DFlashSpeculator.capture()).
    spec._sample_path(cands, scores, 1)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        spec._sample_path(cands, scores, 1)
    torch.accelerator.synchronize()

    # Step 1 (odd UVA rotation): presence_penalty=2.0 must take effect immediately.
    penalties.add_request(0, SamplingParams(presence_penalty=2.0))
    penalties.apply_staged_writes()
    penalties.output_bin_counts[0, 0] = 1
    spec._refresh_penalties()
    graph.replay()
    torch.accelerator.synchronize()
    assert np.isclose(spec._selector_scores[0, 0, 0].item(), 3.0, atol=1e-4)

    # Step 2 (even UVA rotation): presence_penalty=2.0 stays consistent.
    penalties.apply_staged_writes()
    spec._refresh_penalties()
    graph.replay()
    torch.accelerator.synchronize()
    assert np.isclose(spec._selector_scores[0, 0, 0].item(), 3.0, atol=1e-4)

    # Step 3 (switch back to thinking mode: pres=0.0, use_penalty=False so
    # output_bin_counts is NOT cleared). Stale penalty must NOT leak into step 3!
    penalties.add_request(0, SamplingParams(presence_penalty=0.0))
    penalties.apply_staged_writes()
    spec._refresh_penalties()
    graph.replay()
    torch.accelerator.synchronize()
    assert np.isclose(spec._selector_scores[0, 0, 0].item(), 5.0, atol=1e-4)

    # Step 4: repetition_penalty=2.0 divides positive logit 5.0 -> 2.5.
    penalties.add_request(0, SamplingParams(repetition_penalty=2.0))
    penalties.apply_staged_writes()
    penalties.output_bin_counts[0, 0] = 1
    spec._refresh_penalties()
    graph.replay()
    torch.accelerator.synchronize()
    assert np.isclose(spec._selector_scores[0, 0, 0].item(), 2.5, atol=1e-4)

    # Defensive guard: direct kernel call with rep_penalty=0.0 must stay finite.
    zero_rep = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    out_scores = torch.zeros_like(spec._selector_scores)
    _selector_walk_kernel[(1,)](
        scores,
        cands,
        spec.sample_pos,
        spec.sample_idx_mapping,
        spec.temperature,
        spec.seeds,
        spec._selector_tokens,
        out_scores,
        spec._selector_path_state,
        num_steps=7,
        walk_steps=7,
        top_k=16,
        BLOCK_K=16,
        SAMPLE_PROBABILISTIC=True,
        USE_FP64=False,
        PROPOSAL_TEMPERATURE_SCALE=1.0,
        PROPOSAL_TOP_P=0.95,
        repetition_penalty_ptr=zero_rep,
        frequency_penalty_ptr=spec._frequency_penalty,
        presence_penalty_ptr=spec._presence_penalty,
        prompt_bin_mask_ptr=penalties.prompt_bin_mask,
        prompt_bin_mask_stride=penalties.prompt_bin_mask.stride(0),
        output_bin_counts_ptr=penalties.output_bin_counts,
        output_bin_counts_stride=penalties.output_bin_counts.stride(0),
        vocab_size=vocab_size,
        HAS_PENALTIES=True,
        num_warps=1,
    )
    torch.accelerator.synchronize()
    assert torch.isfinite(out_scores[0, 0, 0])
    assert np.isclose(out_scores[0, 0, 0].item(), 5.0, atol=1e-4)


def test_batch_chat_completion_and_anthropic_sampling_defaults():
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
    from vllm.entrypoints.openai.chat_completion.protocol import (
        BatchChatCompletionRequest,
    )
    from vllm.models.sampling_defaults import QWEN38_NON_THINKING_SAMPLING_DEFAULTS

    batch_req = BatchChatCompletionRequest(
        model="qwen3.8-27b",
        messages=[[{"role": "user", "content": "hi"}]],
    )
    single = batch_req.to_chat_completion_request(batch_req.messages[0])
    assert single.temperature is None
    assert single.top_p is None
    assert single.presence_penalty is None
    assert single.frequency_penalty is None
    assert single.repetition_penalty is None
    sp = single.to_sampling_params(128, dict(QWEN38_NON_THINKING_SAMPLING_DEFAULTS))
    assert sp.temperature == 0.7
    assert sp.top_p == 0.80
    assert sp.presence_penalty == 1.5

    anth_default = AnthropicMessagesRequest(
        model="qwen3.8-27b",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
    )
    converted_default = AnthropicServingMessages._build_base_request(
        anth_default, [{"role": "user", "content": "hi"}]
    )
    assert converted_default.presence_penalty is None
    assert converted_default.frequency_penalty is None
    assert converted_default.repetition_penalty is None

    anth_pen = AnthropicMessagesRequest(
        model="qwen3.8-27b",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        presence_penalty=1.2,
        frequency_penalty=0.3,
        repetition_penalty=1.1,
    )
    converted_pen = AnthropicServingMessages._build_base_request(
        anth_pen, [{"role": "user", "content": "hi"}]
    )
    assert converted_pen.presence_penalty == 1.2
    assert converted_pen.frequency_penalty == 0.3
    assert converted_pen.repetition_penalty == 1.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for Triton")
def test_topk_topp_triton_duplicate_logit_tie_break(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_triton

    monkeypatch.setattr(topk_topp_triton, "_NO_FENCE", True)
    device = torch.device("cuda:0")
    vocab_size = 2048
    logits = torch.full((2, vocab_size), -10.0, dtype=torch.float32, device=device)
    # 5 higher unique tokens + 10 tied tokens at 1.0 -> with k=8, must keep 5 + 3 = 8!
    logits[:, :5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.5], device=device)
    logits[:, 5:15] = 1.0
    k = torch.tensor([8, 8], dtype=torch.int32, device=device)
    out = topk_topp_triton.apply_top_k_top_p_triton(logits, k, None)
    torch.accelerator.synchronize()
    for row in range(2):
        kept = int(torch.isfinite(out[row]).sum().item())
        assert kept == 8, (
            f"Row {row}: expected 8 surviving tokens (5 + 3 ties), got {kept}"
        )
