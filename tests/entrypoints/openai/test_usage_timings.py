# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.entrypoints.anthropic.protocol import AnthropicUsage
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.engine.usage import get_request_usage_timings


def _metrics(**overrides):
    values = dict(
        queued_ts=10.0,
        scheduled_ts=12.0,
        first_token_ts=15.0,
        last_token_ts=19.0,
        first_token_latency=6.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_engine_phase_durations_keep_frontend_ttft():
    assert get_request_usage_timings(_metrics()) == {
        "queue_s": 2.0,
        "prefill_s": 3.0,
        "decode_s": 4.0,
        "inference_s": 7.0,
        "ttft_s": 6.0,
    }


def test_zero_duration_is_a_measured_value():
    timings = get_request_usage_timings(
        _metrics(
            scheduled_ts=10.0,
            first_token_ts=10.0,
            last_token_ts=10.0,
            first_token_latency=0.0,
        )
    )
    assert timings is not None
    assert set(timings.values()) == {0.0}


@pytest.mark.parametrize(
    "overrides",
    [
        {"queued_ts": 0.0},
        {"scheduled_ts": 0.0},
        {"first_token_ts": 0.0},
        {"last_token_ts": 0.0},
        # A preemption can move queued_ts past the first scheduled timestamp.
        {"queued_ts": 13.0},
        {"first_token_ts": 11.0},
        {"last_token_ts": 14.0},
        {"first_token_latency": -1.0},
        *[
            {key: float(value)}
            for key in (
                "queued_ts",
                "scheduled_ts",
                "first_token_ts",
                "last_token_ts",
                "first_token_latency",
            )
            for value in ("nan", "inf", "-inf")
        ],
    ],
)
def test_incomplete_or_reordered_measurements_are_omitted(overrides):
    assert get_request_usage_timings(_metrics(**overrides)) is None


def test_stats_disabled_and_multi_choice_do_not_invent_timings():
    assert get_request_usage_timings(None) is None
    assert get_request_usage_timings(_metrics(), num_choices=2) is None


@pytest.mark.parametrize("usage_type", [UsageInfo, AnthropicUsage])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_unavailable_timings_are_absent_from_json(usage_type, explicit_none):
    counts = (
        {"input_tokens": 0, "output_tokens": 0} if usage_type is AnthropicUsage else {}
    )
    usage = usage_type(**counts, **({"timings": None} if explicit_none else {}))
    assert "timings" not in usage.model_dump()
    assert "timings" not in usage.model_dump(exclude_unset=True)
    assert "timings" not in json.loads(usage.model_dump_json())
    # Assignment after construction must preserve the omission contract.
    usage.timings = None
    assert "timings" not in usage.model_dump(exclude_unset=True)


@pytest.mark.parametrize("usage_type", [UsageInfo, AnthropicUsage])
def test_measured_timings_round_trip_including_zero(usage_type):
    timings = get_request_usage_timings(_metrics(first_token_latency=0.0))
    counts = (
        {"input_tokens": 0, "output_tokens": 0} if usage_type is AnthropicUsage else {}
    )
    usage = usage_type(**counts, timings=timings)
    assert usage.model_dump()["timings"] == timings
    assert usage_type.model_validate_json(usage.model_dump_json()).timings == timings
