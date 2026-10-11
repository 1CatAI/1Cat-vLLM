# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the actual adapter; these do not qualify GPU kernels."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def adapter():
    path = ROOT / "vllm/model_executor/kernels/flashinfer_sm70.py"
    spec = importlib.util.spec_from_file_location("fi_sm70_adapter_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePlan:
    def __init__(self, rows, requests, columns, device):
        self.max_columns = columns
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "scores", "visible"

    def provenance(self):
        return {"test_only": True}


def provider(reason=None):
    return SimpleNamespace(qsa_score_reason=lambda *args: reason, QsaScorePlan=FakePlan)


def args():
    return [
        SimpleNamespace(device="cuda:0", shape=(20, 4, 128)),
        object(),
        object(),
        object(),
        object(),
        object(),
    ]


def test_candidate_records_rejection_without_allocating(adapter, monkeypatch):
    monkeypatch.setattr(
        adapter, "_load_provider", lambda: provider("requires_FP16_query_and_cache")
    )
    scorer = adapter.FlashInferQsaScorer("candidate")
    assert scorer.score(*args(), 4, 1024) is None
    assert scorer.plan is None
    assert scorer.last_route == "requires_FP16_query_and_cache"


def test_force_cannot_silently_fall_back(adapter, monkeypatch):
    monkeypatch.setattr(
        adapter, "_load_provider", lambda: provider("requires_M1_32_H4_D128")
    )
    scorer = adapter.FlashInferQsaScorer("force")
    with pytest.raises(RuntimeError, match="requires_M1_32_H4_D128"):
        scorer.validate_force(*args())
    with pytest.raises(RuntimeError, match="requires_M1_32_H4_D128"):
        scorer.score(*args(), 4, 1024)
    with pytest.raises(RuntimeError, match="prefill_cublas"):
        scorer.record_legacy_path("prefill_cublas")


def test_workspaces_are_reused_bounded_and_independent(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_load_provider", provider)
    first, second = (
        adapter.FlashInferQsaScorer("candidate"),
        adapter.FlashInferQsaScorer("candidate"),
    )
    original = args()
    assert first.score(*original, 4, 1024) == ("scores", "visible")
    plan = first.plan
    replacement = args()
    first.score(*replacement, 4, 512)
    assert first.plan is plan
    assert plan.calls[-1][0][1] is replacement[1]  # no stale cache pointer
    second.score(*original, 4, 1024)
    assert second.plan is not first.plan
    assert first.score(*original, 4, 2048) is None
    assert first.plan is plan
    assert first.last_route == "prepared_column_capacity_exceeded"


@pytest.mark.parametrize("mode", ["candidate", "force"])
def test_budget_is_checked_before_plan_allocation(adapter, monkeypatch, mode):
    monkeypatch.setattr(adapter, "_load_provider", provider)
    scorer = adapter.FlashInferQsaScorer(mode, workspace_limit_bytes=1)
    if mode == "force":
        with pytest.raises(RuntimeError, match="workspace_budget_exceeded"):
            scorer.score(*args(), 4, 1024)
    else:
        assert scorer.score(*args(), 4, 1024) is None
    assert scorer.plan is None


def test_wrong_provider_version_is_rejected(adapter, monkeypatch):
    package = SimpleNamespace(
        __version__="0.6.11.post2", __file__="official/flashinfer"
    )
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: package)
    with pytest.raises(RuntimeError, match="requires flashinfer-python"):
        adapter._load_provider()


def test_provider_version_matches_library_source(adapter):
    assert (
        ROOT / "flashinfer-sm70/version.txt"
    ).read_text().strip() == adapter._REQUIRED_VERSION


def test_paged_score_hands_off_before_legacy_allocation():
    source = ROOT / "vllm/models/qwen4_exp/nvidia/ops/qsa.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "qsa_mqa_paged"
    )
    namespace: dict[str, Any] = {
        "torch": torch,
        "math": math,
        "_validate_mqa": lambda q: None,
        "HAS_TRITON": True,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    q = SimpleNamespace(shape=(20, 4, 128), is_cuda=True)
    cache = SimpleNamespace(shape=(64, 204, 1, 128), ndim=4)
    table = SimpleNamespace(shape=(4, 16), ndim=2)
    rows = SimpleNamespace(shape=(20,))
    lengths = SimpleNamespace(shape=(4,))
    calls = []

    def score(*args):
        calls.append(args)
        return "native scores", "native visible"

    selected = namespace["qsa_mqa_paged"](
        q,
        cache,
        table,
        rows,
        rows,
        lengths,
        4,
        score_provider=SimpleNamespace(score=score),
    )
    assert selected == ("native scores", "native visible")
    assert calls[0][-2:] == (3264, math.sqrt(128))


def test_forced_selection_cannot_bypass_scorer_for_prefill():
    source = ROOT / "vllm/models/qwen4_exp/nvidia/ops/qsa.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "qsa_select_paged_tokens"
    )
    calls = []

    def unsupported_legacy(*args):
        raise AssertionError("forced scorer must not enter a legacy prefill route")

    class StopAfterScore(Exception):
        pass

    def score(*args, **kwargs):
        calls.append(kwargs["score_provider"])
        raise StopAfterScore

    namespace: dict[str, Any] = {
        "torch": torch,
        "_qsa_select_by_request": unsupported_legacy,
        "_use_sm70_qsa_indexer_cublas": unsupported_legacy,
        "_use_sm70_qsa_lexicographic_topk": lambda k: True,
        "_LOGITS_WORKSPACE_BYTES": 16 * 1024 * 1024,
        "qsa_mqa_paged": score,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    scorer = SimpleNamespace(mode="force", validate_force=lambda *args: None)
    with pytest.raises(StopAfterScore):
        namespace["qsa_select_paged_tokens"](
            torch.empty(20, 4, 128),
            torch.empty(64, 204, 1, 128),
            torch.empty(4, 16, dtype=torch.int32),
            torch.empty(20, dtype=torch.int32),
            torch.empty(20, dtype=torch.int64),
            torch.empty(4, dtype=torch.int32),
            2048,
            4,
            query_start_loc_cpu=torch.arange(0, 21, 5),
            score_provider=scorer,
        )
    assert calls == [scorer]
