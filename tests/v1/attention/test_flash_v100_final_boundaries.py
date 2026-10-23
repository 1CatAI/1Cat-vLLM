# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final diagnostic subscription and backend layering contract."""

from unittest.mock import Mock

import pytest

from tools.pre_commit.check_layering import measure
from vllm.v1.attention.backends.flash_v100 import debug
from vllm.v1.attention.backends.flash_v100.plan import diagnostics

pytestmark = pytest.mark.cpu_test


def test_diagnostic_events_keep_dynamic_limits_and_message_order(monkeypatch):
    sink = Mock()
    monkeypatch.setattr(debug, "logger", sink)
    monkeypatch.setattr(diagnostics, "_draft_graph_debug_counts", {})
    monkeypatch.setenv("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG_LIMIT", "1")
    diagnostics.draft_graph_debug_log("same", "first %s", "A")
    diagnostics.graph_metadata_debug_log("same", "suppressed")
    monkeypatch.setenv("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG_LIMIT", "2")
    diagnostics.graph_metadata_debug_log("same", "second %s", "B")
    assert [call.args for call in sink.info.call_args_list] == [
        ("FLASH_ATTN_V100 draft graph debug[%s#%d]: %s", "same", 0, "first A"),
        ("FLASH_ATTN_V100 graph metadata debug[%s#%d]: %s", "same", 1, "second B"),
    ]


def test_backend_model_check_only_exempts_spec_and_route_declarations():
    common = "vllm/v1/attention/backends/flash_v100/prefill.py"
    assert measure(common, "is_dflash = True")["flash_v100_model"] == 1
    assert "flash_v100_model" not in measure(common, 'ROUTE_SPECS = {"dflash": 1}')
    assert "flash_v100_model" not in measure(
        common.replace("/prefill.py", "/spec/prefill.py"), "is_dflash = True"
    )
