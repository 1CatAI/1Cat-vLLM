# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comparison counters and injected diagnostics retain the original behavior."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import debug_compare, impl
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

pytestmark = pytest.mark.cpu_test


def _instance():
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.compare_bhmd_out_dir = "/unused"
    instance.compare_bhmd_out_max_calls = 2
    instance._compare_bhmd_out_calls = 0
    instance.compare_triton_out_dir = "/unused"
    instance.compare_triton_out_max_calls = 2
    instance._compare_triton_out_calls = 0
    return instance


def test_counter_quota_is_shared_by_layer_executors_but_not_other_layers():
    instance = _instance()
    other = _instance()
    first = instance._new_comparison_executor()
    second = instance._new_comparison_executor()
    assert first._reserve_bhmd_compare_call() == 0
    assert second._reserve_bhmd_compare_call() == 1
    assert instance._reserve_bhmd_compare_call() is None
    assert other._reserve_bhmd_compare_call() == 0
    assert "_compare_bhmd_out_calls" not in vars(instance)
    instance._compare_bhmd_out_calls = 0
    assert first._reserve_bhmd_compare_call() == 0


def test_partially_initialized_enabled_comparison_retains_missing_counter_error():
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.compare_bhmd_out_dir = "/unused"
    instance.compare_bhmd_out_max_calls = 1
    with pytest.raises(AttributeError, match="_compare_bhmd_out_calls"):
        instance._reserve_bhmd_compare_call()


def test_capture_guard_preserves_quota_reservation_without_running_reference(
    monkeypatch,
):
    instance = _instance()
    instance.compare_triton_out_max_calls = 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    query = SimpleNamespace(is_cuda=True)
    owner = instance._new_comparison_executor()
    for _ in range(2):
        owner._maybe_compare_triton_output(
            None, query, None, None, None, None, None, None, None, "decode"
        )
    assert instance._compare_triton_out_calls == 1


def test_reference_failure_propagates_after_reserving_its_call():
    def fail(*args):
        raise ValueError("reference failed")

    instance = _instance()
    owner = debug_compare.ComparisonExecutor(
        instance._policy(),
        1.0,
        "auto",
        debug_compare.ComparisonOps(fail, None),
        instance.comparison_state,
    )
    query = torch.zeros((1, 1, 4))
    with pytest.raises(ValueError, match="reference failed"):
        owner._maybe_compare_triton_output(
            None, query, query, query, query, None, query, None, None, "decode"
        )
    assert instance._compare_triton_out_calls == 1
    assert torch.count_nonzero(query) == 0


def test_bhmd_operator_and_json_report_use_owned_policy_and_counter(tmp_path):
    instance = _instance()
    instance.scale = 0.125
    instance.kv_cache_dtype = "auto"
    instance.compare_bhmd_out_dir = str(tmp_path)
    calls = []

    def native(*args, **kwargs):
        calls.append(kwargs["softmax_scale"])
        kwargs["out"].fill_(1)

    instance.flash_attn_prefill_paged_bhmd = native
    tensor = torch.ones((1, 1, 2, 4), dtype=torch.float16)
    instance._maybe_compare_bhmd_out(
        SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0),
        tensor.permute(0, 2, 1, 3),
        tensor,
        tensor,
        torch.zeros((1, 1), dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        tensor,
    )
    (report,) = tmp_path.glob("bhmd_compare_*.json")
    data = json.loads(report.read_text())
    assert data["equal"] and data["call_idx"] == 0 and data["max_diff"] == 0
    assert calls == [0.125] and instance._compare_bhmd_out_calls == 1


def test_static_helper_override_reaches_real_triton_comparison(monkeypatch):
    instance = _instance()
    instance._layer_debug_info = lambda _: {"custom_layer": "preserved"}
    instance._maybe_write_triton_tensor_dump = lambda *args: {}
    reports = []
    instance._write_triton_compare_report = lambda *args: reports.append(args[-1])

    def reference(self, layer, q, k, v, cache, metadata, output, *args):
        output.fill_(2)
        return output

    monkeypatch.setattr(TritonAttentionImpl, "forward", reference)
    query = torch.zeros((1, 1, 4))
    metadata = SimpleNamespace(num_actual_tokens=1, max_query_len=1, max_seq_len=1)
    instance._maybe_compare_triton_output(
        None, query, query, query, query, metadata, query, None, None, "decode"
    )
    assert reports[0]["custom_layer"] == "preserved"
