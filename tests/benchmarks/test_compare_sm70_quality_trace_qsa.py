# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_tool():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "compare_sm70_quality_trace.py"
    )
    spec = importlib.util.spec_from_file_location("compare_sm70_quality_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qsa_block_overlap_uses_page_ids_and_ignores_padding() -> None:
    tool = _load_tool()
    left = torch.tensor([[0, 1, 4, 5, -1], [8, 9, -1, -1, -1]])
    right = torch.tensor([[2, 3, 8, 9, -1], [8, 11, -1, -1, -1]])

    stats = tool._qsa_block_overlap_stats(left, right, compress_ratio=4)

    assert stats["exact_row_ratio"] == pytest.approx(0.5)
    assert stats["mean_row_jaccard"] == pytest.approx((1 / 3 + 1) / 2)
    assert stats["mean_left_block_recall"] == pytest.approx(0.75)
    assert stats["micro_jaccard"] == pytest.approx(2 / 4)


def test_layer_dump_index_accepts_eager_direct_save_names(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "pid123_layer03_qsa_qsa_selected_indices_000.pt"
    torch.save(
        {
            "label": "qsa_selected_indices",
            "layer_idx": 3,
            "layer_type": "qsa",
            "count": 0,
            "pid": 123,
            "shape": (8, 4),
            "tensor": torch.zeros((8, 4), dtype=torch.int32),
        },
        path,
    )

    indexed = tool._index_layer_dumps(tmp_path)

    assert indexed == {(0, 0, 3, "qsa", "qsa_selected_indices", "8x4"): path}


def test_tensor_stats_reports_scale_aware_error() -> None:
    tool = _load_tool()

    stats = tool._tensor_stats(
        torch.tensor([3.0, 4.0]),
        torch.tensor([0.0, 4.0]),
    )

    assert stats["rmse"] == pytest.approx((9 / 2) ** 0.5)
    assert stats["left_mean_abs"] == pytest.approx(3.5)
    assert stats["relative_l2"] == pytest.approx(3 / 5)
    assert stats["cosine_similarity"] == pytest.approx(0.8)
