# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real Scheduler trace integration; no model weights or CUDA work."""

import json

import pytest
from transformers import OPTConfig

from tests.v1.core import utils
from tests.v1.core.test_pp_decode_stagger import (
    FeedbackQueue,
    make_scheduler,
)
from vllm.config import VllmConfig
from vllm.v1.core.sched.pp_causal_trace import PPCausalTrace
from vllm.v1.request import RequestStatus


@pytest.fixture
def local_model(tmp_path):
    model = tmp_path / "metadata-only"
    OPTConfig(
        hidden_size=32,
        num_hidden_layers=14,
        num_attention_heads=4,
        ffn_dim=64,
        max_position_embeddings=2048,
        vocab_size=65536,
        architectures=["OPTForCausalLM"],
    ).save_pretrained(model)
    return str(model)


def scheduler(monkeypatch, local_model, directory, *, stagger, trace):
    def config(**kwargs):
        kwargs["additional_config"] = (
            {
                "pp_causal_trace": {
                    "directory": str(directory),
                    "tags": ["causal-"],
                    "max_events": 5000,
                }
            }
            if trace
            else {}
        )
        return VllmConfig(**kwargs)

    monkeypatch.setattr(utils, "VllmConfig", config)
    return make_scheduler(monkeypatch, local_model, enabled=stagger)


@pytest.mark.parametrize("stagger", [False, True])
def test_actual_trace_records_feedback_dependencies_and_phase(
    monkeypatch, local_model, tmp_path, stagger
):
    directory = tmp_path / "trace"
    engine = scheduler(monkeypatch, local_model, directory, stagger=stagger, trace=True)
    assert engine._pp_causal_trace is not None and not directory.exists()
    requests = utils.create_requests(
        num_requests=2, num_tokens=8, max_tokens=128, req_ids=["causal-a", "causal-b"]
    )
    for request in requests:
        engine.add_request(request)
    queue = FeedbackQueue(engine, accepted=(1, 2, 5))
    queue.step()
    queue.step()
    for _ in range(6):
        queue.step()
    engine.finish_requests(
        [r.request_id for r in requests], RequestStatus.FINISHED_ABORTED
    )
    queue.step()
    queue.step()
    engine._pp_causal_trace.flush()
    files = sorted(directory.glob("*.jsonl"))
    assert files
    events = []
    for path in files:
        data = [json.loads(line) for line in path.read_text().splitlines()]
        assert data[-1]["dropped_events"] == 0 and data[-1]["pending_steps"] == []
        events.extend(data[1:-1])
    begins = {
        row["scheduler_current_step"]: row
        for row in events
        if row["event"] == "schedule_begin"
    }
    model_pending = set()
    peak = 0
    tail_shapes = []
    for row in events:
        if row["event"] == "schedule_end":
            step = row["scheduled_step"]
            for request_id in row["query_tokens"]:
                assert (
                    begins[step]["requests"][request_id]["next_decode_eligible_step"]
                    <= step
                )
            if row["is_model_forward"]:
                model_pending.add(step)
                peak = max(peak, len(model_pending))
            if 3 <= step <= 8:
                tail_shapes.append(
                    (row["total_query_tokens"], len(row["query_tokens"]))
                )
        elif row["event"] == "feedback_end":
            model_pending.discard(row["scheduled_step"])
    assert not model_pending
    assert peak == (2 if stagger else 1)
    assert tail_shapes == ([(5, 1)] * 6 if stagger else [(10, 2), (0, 0)] * 3)


@pytest.mark.parametrize("trace", [False, True])
def test_disabled_or_unmatched_trace_leaves_output_and_files_unchanged(
    monkeypatch, local_model, tmp_path, trace
):
    directory = tmp_path / "trace"
    engine = scheduler(monkeypatch, local_model, directory, stagger=True, trace=trace)
    if not trace:
        assert engine._pp_causal_trace is None

        def forbidden(*args, **kwargs):
            raise AssertionError("Disabled trace callback was invoked")

        for name in (
            "schedule_begin",
            "schedule_end",
            "feedback_begin",
            "feedback_end",
        ):
            monkeypatch.setattr(PPCausalTrace, name, forbidden)
    requests = utils.create_requests(
        num_requests=2,
        num_tokens=8,
        max_tokens=12,
        req_ids=["ordinary-a", "ordinary-b"],
    )
    for request in requests:
        engine.add_request(request)
    queue = FeedbackQueue(engine, accepted=(1, 2, 5))
    for _ in range(100):
        queue.step()
        if engine.get_num_unfinished_requests() == 0:
            break
    else:
        pytest.fail("Finite scheduler run failed to complete")
    assert [list(r.output_token_ids) for r in requests] == [list(range(1000, 1012))] * 2
    if trace:
        assert engine._pp_causal_trace.events == []
        engine._pp_causal_trace.flush()
    assert not directory.exists()
