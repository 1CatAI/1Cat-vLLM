# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide keyed logging preserves legacy decode observation controls."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm import logger as logging_api
from vllm.v1.attention.backends import flash_attn_v100 as legacy
from vllm.v1.attention.backends.flash_v100 import decode, impl, state

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("level", ["info", "warning"])
def test_explicit_key_survives_argument_logger_and_lru_changes(monkeypatch, level):
    first = logging_api.init_logger("vllm.test.once.first")
    second = logging_api.init_logger("vllm.test.once.second")
    records = []
    key = object()
    for logger in (first, second):
        monkeypatch.setattr(logger, level, lambda *a, **k: records.append(a))
    try:
        getattr(first, level + "_once")("first %s", 1, scope="process", key=key)
        getattr(second, level + "_once")("changed %s", 2, scope="process", key=key)
        assert records == [("first %s", 1)]
        for index in range(256):
            getattr(first, level + "_once")("unrelated %s", index, scope="process")
        before = len(records)
        getattr(first, level + "_once")("after eviction", scope="process", key=key)
        assert len(records) == before
        logging_api.set_log_once_state(key, False)
        getattr(second, level + "_once")("reset", scope="process", key=key)
        assert records[-1] == ("reset",)
    finally:
        logging_api.set_log_once_state(key, False)


def test_explicit_key_is_not_consumed_by_scope_decline_or_emission_error(monkeypatch):
    logger = logging_api.init_logger("vllm.test.once.failure")
    key = object()
    original = logging_api._should_log_with_scope
    monkeypatch.setattr(logging_api, "_should_log_with_scope", lambda scope: False)
    logger.info_once("skipped", scope="global", key=key)
    assert not logging_api.log_once_seen(key)
    monkeypatch.setattr(logging_api, "_should_log_with_scope", original)
    monkeypatch.setattr(logger, "info", MagicMock(side_effect=RuntimeError("sink")))
    with pytest.raises(RuntimeError, match="sink"):
        logger.info_once("failed", scope="process", key=key)
    assert not logging_api.log_once_seen(key)


def test_unkeyed_once_keeps_original_message_and_argument_granularity(monkeypatch):
    logger = logging_api.init_logger("vllm.test.once.original")
    records = []
    monkeypatch.setattr(logger, "info", lambda *a, **k: records.append(a))
    message = str(object())
    logger.info_once(message, 1, scope="process")
    logger.info_once(message, 1, scope="process")
    logger.info_once(message, 2, scope="process")
    assert records == [(message, 1), (message, 2)]


def test_legacy_flag_and_real_logger_share_one_process_observation(monkeypatch):
    key = state.LOG_KEYS["_logged_decode_dense_reference"]
    records = []
    monkeypatch.setattr(legacy, "_logged_decode_dense_reference", False)
    monkeypatch.setattr(decode.logger, "warning", lambda *a, **k: records.append(a))
    query = torch.empty((0, 6, 256), dtype=torch.float16)
    metadata = SimpleNamespace(num_actual_tokens=0)
    for _ in range(2):
        instance = object.__new__(impl.FlashAttnV100Impl)
        instance.kv_cache_dtype = "auto"
        assert (
            instance._flash_v100_decode_dense_reference(
                None, query, query, metadata, query
            )
            is query
        )
    assert len(records) == 1
    assert logging_api.log_once_seen(key)
    assert legacy._logged_decode_dense_reference
    assert "_logged_decode_dense_reference" not in vars(state)
    legacy._logged_decode_dense_reference = False
    instance._flash_v100_decode_dense_reference(None, query, query, metadata, query)
    assert len(records) == 2


def test_auditor_rejects_virtual_flag_read_only_by_test(tmp_path):
    source = tmp_path / "test_unconsumed.py"
    report = tmp_path / "audit.json"
    source.write_text(
        "from vllm.v1.attention.backends import flash_attn_v100 as legacy\n"
        "def test_unused(monkeypatch):\n"
        "    monkeypatch.setattr(legacy, '_logged_decode_flash', False)\n"
        "    assert legacy._logged_decode_flash is False\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--import-mode=importlib",
            "--confcutdir=" + str(tmp_path),
            "-p",
            "tools.sm70.flash_v100_shim_audit",
            "--shim-audit=" + str(report),
            "--require-shim-use",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Unconsumed shim patches: ['_logged_decode_flash']" in result.stdout
    observation = json.loads(report.read_text())["_logged_decode_flash"]
    assert observation["patched_in"]
    assert observation["read_from"] == observation["called_from"] == []
