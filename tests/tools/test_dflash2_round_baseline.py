# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-round statistics must not become target-event or per-token timing."""

import pytest

from benchmarks.benchmark_sm70_dflash2_round_baseline import engine_cost, steady_rounds


def test_warmup_is_excluded_and_emitted_tokens_do_not_divide_round_latency():
    events = [(float(i), [0]) for i in range(20)]
    events += [(20.0, [1]), (20.014, [2, 3]), (20.028, [4, 5, 6, 7])]
    result = steady_rounds(events)
    assert result["steady_rounds"] == 2
    assert result["stream_complete_round_mean_ms"] == pytest.approx(14.0)
    assert result["steady_emitted_tokens_per_round"] == 3
    assert result["steady_emitted_tokens_per_second"] == pytest.approx(6 / 0.028)


def test_metric_round_acceptance_and_emitted_token_cost_have_distinct_units():
    delta = {
        "vllm:spec_decode_num_drafts_total": 100,
        "vllm:spec_decode_num_accepted_tokens_total": 200,
        "vllm:spec_decode_num_draft_tokens_total": 700,
        "vllm:request_decode_time_seconds_sum": 1.4,
    }
    result = engine_cost(delta, 301)
    assert result["engine_complete_round_mean_ms"] == pytest.approx(14)
    assert result["accepted_drafts_per_round"] == 2
    assert result["accepted_over_proposed"] == pytest.approx(2 / 7)
    assert result["engine_decode_ms_per_emitted_token"] == pytest.approx(1400 / 300)


def test_no_speculation_is_not_reported_as_zero_cost_or_zero_acceptance():
    result = engine_cost({"vllm:request_decode_time_seconds_sum": 1.4}, 101)
    assert result["engine_complete_round_mean_ms"] is None
    assert result["accepted_drafts_per_round"] is None
    assert result["engine_decode_ms_per_emitted_token"] == pytest.approx(14)


@pytest.mark.parametrize("events", [[(0.0, [1])], [(1.0, [1]), (1.0, [2])]])
def test_incomplete_or_unordered_round_evidence_is_rejected(events):
    with pytest.raises(ValueError):
        steady_rounds(events, warmup=0)


def test_stale_engine_timing_cannot_be_reported_as_zero_round_latency():
    with pytest.raises(ValueError, match="decode timing"):
        engine_cost({"vllm:spec_decode_num_drafts_total": 10}, 31)
