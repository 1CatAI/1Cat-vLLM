# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore fills worker drafts into the bitmask path when no step of a
structured-output request is in flight (the deferred path never runs)."""

from unittest.mock import MagicMock

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import DraftTokenIds


def _scheduler_output(spec: dict[str, list[int]], structured: bool = True):
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1 + len(toks) for req_id, toks in spec.items()},
        total_num_scheduled_tokens=sum(1 + len(t) for t in spec.values()),
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens=spec,
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        has_structured_output_requests=structured,
    )


def _core(requests: dict, drafts: DraftTokenIds | None):
    core = EngineCore.__new__(EngineCore)
    core.use_spec_decode = True
    core.scheduler = MagicMock()
    core.scheduler.requests = requests
    core.model_executor = MagicMock()
    core.model_executor.take_draft_token_ids.return_value = drafts
    return core


def _request(structured: bool, placeholders: int):
    req = MagicMock()
    req.use_structured_output = structured
    req.num_output_placeholders = placeholders
    return req


def test_fills_drafts_when_only_this_step_is_outstanding():
    drafts = DraftTokenIds(["0"], [[7, 8, 9]], None)
    out = _scheduler_output({"0": [-1, -1, -1]})
    core = _core({"0": _request(True, placeholders=4)}, drafts)
    core._fill_structured_output_drafts(out)
    core.model_executor.take_draft_token_ids.assert_called_once()
    core.scheduler.update_draft_token_ids_in_output.assert_called_once_with(drafts, out)


def test_skips_when_a_previous_step_is_still_in_flight():
    out = _scheduler_output({"0": [-1, -1, -1]})
    core = _core({"0": _request(True, placeholders=8)}, None)
    core._fill_structured_output_drafts(out)
    core.model_executor.take_draft_token_ids.assert_not_called()


def test_skips_without_structured_requests_or_placeholders():
    core = _core({"0": _request(False, placeholders=4)}, None)
    core._fill_structured_output_drafts(_scheduler_output({"0": [-1, -1, -1]}))
    core.model_executor.take_draft_token_ids.assert_not_called()

    core = _core({"0": _request(True, placeholders=4)}, None)
    core._fill_structured_output_drafts(_scheduler_output({"0": [5, 6, 7]}))
    core.model_executor.take_draft_token_ids.assert_not_called()

    core = _core({"0": _request(True, placeholders=4)}, None)
    core._fill_structured_output_drafts(_scheduler_output({"0": [-1, -1]}, structured=False))
    core.model_executor.take_draft_token_ids.assert_not_called()
