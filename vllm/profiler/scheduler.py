# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-only scheduler metadata for joining pipeline profiler scopes."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


def scheduler_step_annotation(output: "SchedulerOutput") -> str:
    """Identify request positions without recording prompt or generated tokens."""
    offsets = {
        request.req_id: request.num_computed_tokens
        for request in output.scheduled_new_reqs
    }
    cached = output.scheduled_cached_reqs
    offsets.update(zip(cached.req_ids, cached.num_computed_tokens))
    scheduled = output.num_scheduled_tokens
    request_ids = sorted(offsets.keys() | scheduled.keys())
    return json.dumps(
        {
            "kind": "model"
            if output.total_num_scheduled_tokens
            else "zero_token_control",
            "requests": [
                {
                    "id": request_id,
                    "computed": offsets.get(request_id),
                    "scheduled": scheduled.get(request_id, 0),
                }
                for request_id in request_ids
            ],
        },
        separators=(",", ":"),
    )
