# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput


def test_sm70_dflash2_batch_trace_records_completed_decode_iteration() -> None:
    core = EngineCore.__new__(EngineCore)
    core._sm70_dflash2_batch_trace_enabled = True
    core._sm70_dflash2_batch_trace_every = 16
    core.scheduler = SimpleNamespace(get_num_unfinished_requests=lambda: 0)
    core._reset_sm70_dflash2_batch_trace()

    scheduled_spec_tokens = {
        "r0": [10, 11, 12, 13, 14, 15, 16],
        "r1": [20, 21, 22, 23, 24, 25, 26],
    }
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(is_context_phase=lambda _req_id: False),
        num_scheduled_tokens={"r0": 8, "r1": 8},
        scheduled_spec_decode_tokens=scheduled_spec_tokens,
        num_invalid_spec_tokens={"r1": 1},
    )
    model_output = ModelRunnerOutput(
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        sampled_token_ids=[[30, 31, 32, 33], [40, 41]],
    )
    engine_core_outputs = {
        0: EngineCoreOutputs(
            outputs=[
                EngineCoreOutput("r0", [30, 31, 32, 33]),
                EngineCoreOutput("r1", [40, 41]),
            ]
        )
    }

    with patch("vllm.v1.engine.core.logger.info") as log_info:
        core._record_sm70_dflash2_batch_trace(
            scheduler_output,
            model_output,
            engine_core_outputs,
            queue_len=0,
        )

    log_info.assert_called_once()
    log_args = log_info.call_args.args
    message = log_args[0] % log_args[1:]
    assert "steps=1" in message
    assert "phase_hist=decode:1" in message
    assert "gen_req_hist=2:1" in message
    assert "target_m_hist=16:1" in message
    assert "accept_len_hist=2:1,4:1" in message
    assert "queue_len_hist=0:1" in message
    assert "draft_tokens=13" in message
    assert "accepted_draft_tokens=4" in message
    assert "sampled_tokens=6 emitted_tokens=6 finished_reqs=0" in message
