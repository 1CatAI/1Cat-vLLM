# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Counter semantics must not depend on GPU synchronization or tensor contents."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.models.pp_decode_transfer import (
    PPTransportMetrics,
    decode_fallback_reason,
)
from vllm.v1.core.sched.output import SchedulerOutput

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("value", [True, -1, 1.5, "100", None])
def test_metrics_interval_validation(value):
    with pytest.raises(ValueError, match="nonnegative int"):
        PPTransportMetrics.create({"pp_transport_metrics_interval": value}, 0)


def test_metrics_disabled_by_default():
    assert PPTransportMetrics.create({}, 0) is None
    assert PPTransportMetrics.create(None, 0) is None


def test_payload_counts_exclude_static_padding_and_keep_direction():
    metrics = PPTransportMetrics.create({"pp_transport_metrics_interval": 100}, 3)
    assert metrics is not None
    tensor = torch.empty(20, 4, 8, dtype=torch.float16, device="meta")
    metrics.record_step(static=True, reason=None)
    metrics.record_payload("send", "decode_static", {"hidden": tensor}, num_tokens=5)
    metrics.record_payload("recv", "dynamic", {"hidden": tensor})
    metrics.record_step(static=False, reason="prefill")
    metrics.record_step(static=False, reason="prefill")
    metrics.record_step(static=False, reason=None)
    metrics.record_send_wait(0.125)
    snapshot = metrics.snapshot()
    assert snapshot["rank"] == 3
    assert snapshot["steps"] == 4
    assert snapshot["static_fraction"] == 0.25
    assert snapshot["fallbacks"] == {"prefill": 2, "disabled": 1}
    assert snapshot["payload_bytes"] == {
        "send_decode_static": 320,
        "recv_dynamic": 1280,
    }
    assert snapshot["tensor_messages"] == {
        "send_decode_static": 1,
        "recv_dynamic": 1,
    }
    assert snapshot["send_wait_calls"] == 1
    assert snapshot["send_wait_host_seconds"] == 0.125
    snapshot["fallbacks"]["prefill"] = 1000
    assert metrics.snapshot()["fallbacks"]["prefill"] == 2
    assert not torch.cuda.is_initialized()


def test_fallback_reason_distinguishes_prefill_from_incomplete_draft():
    output = SchedulerOutput.make_empty()
    assert decode_fallback_reason(output, 5, 40) == "empty"
    output.num_scheduled_tokens = {"a": 5}
    output.total_num_scheduled_tokens = 5
    output.scheduled_cached_reqs.req_ids = ["a"]
    output.scheduled_cached_reqs.num_output_tokens = [0]
    assert decode_fallback_reason(output, 5, 40) == "prefill"
    output.scheduled_cached_reqs.num_output_tokens = [1]
    assert decode_fallback_reason(output, 5, 40) == "incomplete_draft"
    output.scheduled_spec_decode_tokens = {"a": [-1] * 4}
    assert decode_fallback_reason(output, 5, 40) is None


def test_dynamic_send_counts_actual_tp_slice_and_skips_empty_tensors():
    metrics = PPTransportMetrics(0, 100)
    pp = GroupCoordinator.__new__(GroupCoordinator)
    tp = cast(GroupCoordinator, SimpleNamespace(world_size=4))
    full = torch.empty(20, 32, dtype=torch.float16, device="meta")
    tensors = {"hidden": full, "empty": full[:0]}
    metrics.record_dynamic_send(pp, tensors, tp, {})
    metrics.record_payload("recv", "dynamic", {"hidden": full[:5], "empty": full[:0]})
    snapshot = metrics.snapshot()
    assert snapshot["payload_bytes"] == {"send_dynamic": 320, "recv_dynamic": 320}
    assert snapshot["tensor_messages"] == {"send_dynamic": 1, "recv_dynamic": 1}
    metrics.record_dynamic_send(pp, {"hidden": full}, tp, {"hidden": False})
    assert metrics.snapshot()["payload_bytes"]["send_dynamic"] == 1600
    assert not torch.cuda.is_initialized()
