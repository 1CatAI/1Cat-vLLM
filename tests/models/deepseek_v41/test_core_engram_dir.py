# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram row directory defaults to the model directory; the knob overrides it; a hub id fails loud."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm.models.deepseek_v41.sm70.model import ENGRAM_DIR_KNOB, engram_row_dir


def test_engram_dir_defaults_to_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENGRAM_DIR_KNOB, raising=False)
    assert engram_row_dir(str(tmp_path)) == str(tmp_path)


def test_engram_dir_knob_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "fast-disk"
    monkeypatch.setenv(ENGRAM_DIR_KNOB, str(other))
    assert engram_row_dir(str(tmp_path)) == str(other)


def test_engram_dir_refuses_hub_id_without_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENGRAM_DIR_KNOB, raising=False)
    with pytest.raises(ValueError, match=ENGRAM_DIR_KNOB):
        engram_row_dir("deepseek-ai/DeepSeek-V4.1-Flash")
