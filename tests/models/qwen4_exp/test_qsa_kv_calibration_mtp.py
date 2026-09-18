# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2 CPU tests for the QSA KV calibration tool: 13-layer support
(12 target + MTP layer id 48), 12-layer regression, and the new
`compare-targets` old-vs-new target envelope report (D5)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip_global_cleanup

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools" / "qwen4_exp" / "qsa_kv_calibration.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("qsa_kv_cal_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tool = _load_tool()


def _stat(max_abs, bins=16):
    hist = [0] * bins
    hist[-1] = 100
    return {
        "count": 100,
        "finite_count": 100,
        "nonzero_count": 100,
        "max_abs": float(max_abs),
        "histogram": hist,
    }


def _record(layer_id, max_abs, shard="s0", bins=16):
    return {
        "schema_version": 1,
        "layer_id": layer_id,
        "corpus_shard": shard,
        "histogram": {"bins": bins, "log2_min": -16.0, "log2_max": 16.0},
        "k": _stat(max_abs),
        "v": _stat(max_abs * 0.5),
    }


def _write_traces(d: Path, layer_ids):
    p = d / "qsa-kv-rank0-pid1.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for lid in layer_ids:
            f.write(json.dumps(_record(lid, 1.0 + lid)) + "\n")
    return p


def _summarize(tmp_path: Path, layer_ids, expected):
    src = tmp_path / "traces"
    src.mkdir()
    _write_traces(src, layer_ids)
    out = tmp_path / "report.json"
    tool.summarize(
        argparse.Namespace(
            input_dir=[str(src)],
            exclude_corpus_shard=[],
            output=str(out),
            expected_layers=expected,
        )
    )
    return json.loads(out.read_text(encoding="utf-8"))


def test_summarize_12_layer_regression(tmp_path):
    report = _summarize(tmp_path, list(range(12)), 12)
    assert report["tensor_count"] == 24
    assert report["qsa_layer_ids"] == list(range(12))
    assert "model.layers.0.self_attn.k_scale" in report["tensors"]
    assert "model.layers.48.self_attn.k_scale" not in report["tensors"]


def test_summarize_13_layer_includes_mtp(tmp_path):
    report = _summarize(tmp_path, list(range(12)) + [48], 13)
    assert report["tensor_count"] == 26
    assert 48 in report["qsa_layer_ids"]
    assert "model.layers.48.self_attn.k_scale" in report["tensors"]
    assert "model.layers.48.self_attn.v_scale" in report["tensors"]


def test_summarize_layer_count_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="Expected 13 QSA layers"):
        _summarize(tmp_path, list(range(12)), 13)


def _report(scales: dict[str, float]) -> dict:
    return {
        "schema_version": 1,
        "tensor_count": len(scales),
        "tensors": {
            name: {"scale": s, "max_abs": s * 448.0} for name, s in scales.items()
        },
    }


def test_compare_targets_excludes_mtp_and_reports_ratio(tmp_path):
    names = {f"model.layers.{i}.self_attn.k_scale": 0.1 * (i + 1) for i in range(12)}
    names["model.layers.48.self_attn.k_scale"] = 0.99  # MTP layer, must be excluded
    old = _report(names)
    new_scales = dict(names)
    new_scales["model.layers.0.self_attn.k_scale"] *= 1.05  # 5% target drift
    new_scales["model.layers.48.self_attn.k_scale"] = 0.5  # different MTP, ignored
    new = _report(new_scales)
    old_p, new_p, out_p = (tmp_path / n for n in ("old.json", "new.json", "cmp.json"))
    old_p.write_text(json.dumps(old))
    new_p.write_text(json.dumps(new))
    tool.compare_targets(
        argparse.Namespace(
            old_report=str(old_p),
            new_report=str(new_p),
            output=str(out_p),
            mtp_layer_id=48,
        )
    )
    cmp = json.loads(out_p.read_text(encoding="utf-8"))
    assert cmp["target_tensor_count"] == 12  # layer 48 excluded
    assert all("layers.48." not in row["tensor"] for row in cmp["tensors"])
    assert cmp["max_scale_ratio_deviation"] == pytest.approx(0.05, abs=1e-6)


REAL_MANIFEST = Path(
    "/home/l/models/qwen38-flash-next-e4m3-kvscales/kvscales-manifest.json"
)


@pytest.mark.skipif(
    not REAL_MANIFEST.is_file(), reason="published scale-pack manifest not present"
)
def test_compare_targets_reads_published_manifest(tmp_path):
    # The published manifest lists tensors as {kind, layer, scale, max_abs, ...};
    # the normalizer must map them onto model.layers.<id>.self_attn.<kind>_scale.
    old_t = tool._load_scale_tensors(REAL_MANIFEST)
    assert len(old_t) == 24
    assert all(n.startswith("model.layers.") and ".self_attn." in n for n in old_t)
    assert "model.layers.48.self_attn.k_scale" not in old_t
    drift_name = "model.layers.3.self_attn.k_scale"
    assert drift_name in old_t
    new_tensors = {}
    for name, details in old_t.items():
        factor = 1.02 if name == drift_name else 1.0
        new_tensors[name] = {
            "scale": details["scale"] * factor,
            "max_abs": details["max_abs"],
        }
    new_report = tmp_path / "new.json"
    new_report.write_text(json.dumps({"tensors": new_tensors}))
    out = tmp_path / "cmp.json"
    tool.compare_targets(
        argparse.Namespace(
            old_report=str(REAL_MANIFEST),
            new_report=str(new_report),
            output=str(out),
            mtp_layer_id=48,
        )
    )
    cmp = json.loads(out.read_text(encoding="utf-8"))
    assert cmp["target_tensor_count"] == 24
    assert cmp["max_scale_ratio_deviation"] == pytest.approx(0.02, abs=1e-6)
