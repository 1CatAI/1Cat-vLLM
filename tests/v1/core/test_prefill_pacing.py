# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler pacing for unfinished prefills sharing steps with decode."""

import json
from pathlib import Path

import pytest

import vllm.envs as envs
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _create_pacing_scheduler(monkeypatch, pace_steps: int, tmp_path: Path):
    monkeypatch.setenv("VLLM_1CAT_PREFILL_PACE_STEPS", str(pace_steps))
    envs.disable_envs_cache()
    model_path = tmp_path / "tiny-opt"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "vocab_size": 1024,
                "hidden_size": 16,
                "word_embed_proj_dim": 16,
                "ffn_dim": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "max_position_embeddings": 256,
                "pad_token_id": 1,
                "bos_token_id": 2,
                "eos_token_id": 2,
                "do_layer_norm_before": True,
                "enable_bias": True,
            }
        ),
        encoding="utf-8",
    )
    return create_scheduler(
        model=str(model_path),
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=256,
        skip_tokenizer_init=True,
    )


def _complete_scheduled_step(scheduler, step, decode_req_id: str | None = None):
    req_ids = list(step.num_scheduled_tokens)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [1000 + scheduler.current_step] if req_id == decode_req_id else []
                for req_id in req_ids
            ],
        ),
    )


def _make_running_decode(scheduler):
    (decode,) = create_requests(
        num_requests=1,
        num_tokens=8,
        max_tokens=100,
        req_ids=["decode"],
    )
    scheduler.add_request(decode)
    initial = scheduler.schedule()
    assert initial.num_scheduled_tokens[decode.request_id] == 8
    _complete_scheduled_step(scheduler, initial, decode.request_id)
    return decode


def test_unfinished_prefill_runs_once_per_four_steps_with_decode(monkeypatch, tmp_path):
    scheduler = _create_pacing_scheduler(monkeypatch, pace_steps=4, tmp_path=tmp_path)
    decode = _make_running_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=128,
        max_tokens=100,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill)

    prefill_steps = []
    for step_number in range(2, 13):
        step = scheduler.schedule()
        assert step.num_scheduled_tokens[decode.request_id] == 1
        if prefill.request_id in step.num_scheduled_tokens:
            prefill_steps.append(step_number)
        _complete_scheduled_step(scheduler, step, decode.request_id)

    # The waiting request gets its first chunk at step 2. Once it is running,
    # each unfinished prefill chunk is separated by three skipped steps.
    assert prefill_steps == [2, 3, 7, 11]
    assert prefill.next_decode_eligible_step == 15


def test_lone_prefill_is_not_paced(monkeypatch, tmp_path):
    scheduler = _create_pacing_scheduler(monkeypatch, pace_steps=4, tmp_path=tmp_path)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=128,
        max_tokens=100,
        req_ids=["solo-prefill"],
    )
    scheduler.add_request(prefill)

    for _ in range(4):
        step = scheduler.schedule()
        assert step.num_scheduled_tokens[prefill.request_id] == 8
        assert prefill.next_decode_eligible_step == 0
        _complete_scheduled_step(scheduler, step)


def test_zero_pacing_keeps_prefill_eligible_each_step_with_decode(
    monkeypatch, tmp_path
):
    scheduler = _create_pacing_scheduler(monkeypatch, pace_steps=0, tmp_path=tmp_path)
    decode = _make_running_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=128,
        max_tokens=100,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill)

    for _ in range(6):
        step = scheduler.schedule()
        assert step.num_scheduled_tokens[decode.request_id] == 1
        assert step.num_scheduled_tokens[prefill.request_id] > 0
        assert prefill.next_decode_eligible_step == 0
        _complete_scheduled_step(scheduler, step, decode.request_id)
