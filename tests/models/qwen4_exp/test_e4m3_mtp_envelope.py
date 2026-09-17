# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2b CPU tests for the materializer's --target-scale-source envelope:
raise each published target scale to max(published, W2 report) max_abs / 448 so
W2 activations that exceed 448*published_scale are no longer clipped. Uses a
tiny synthetic pack (manifest list format) + W2 report; no GPU, no real model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

MATERIALIZER = Path("/home/l/work/flash-next/dev-kv/tools/materialize_overlay.py")
pytestmark = pytest.mark.skipif(
    not MATERIALIZER.is_file(), reason="phase-2 materializer not present"
)


def _load_mat():
    spec = importlib.util.spec_from_file_location("mtp_materializer", MATERIALIZER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_scales(path: Path) -> dict[str, float]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        start = 8 + n
        out = {}
        for name, d in header.items():
            if name == "__metadata__":
                continue
            f.seek(start + d["data_offsets"][0])
            out[name] = struct.unpack("<f", f.read(4))[0]
        return out


# (layer, kind): (published_max_abs, w2_max_abs). layer3 k and layer7 v are
# raised by W2; the others keep the published envelope.
CASES = {
    (3, "k"): (9.0, 14.984375),
    (3, "v"): (5.0, 4.0),
    (7, "k"): (8.3359375, 8.3671875),
    (7, "v"): (10.828125, 8.0),
}
E4M3_MAX = 448.0


def _build(tmp_path, mat):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model-bf16-0001.safetensors").write_bytes(b"\x00" * 16)
    index_path = base / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "model.embed_tokens.weight": "model-bf16-0001.safetensors"
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    on_disk = {
        (layer, kind): f"model.language_model.layers.{layer}.self_attn.{kind}_scale"
        for (layer, kind) in CASES
    }
    published_scales = {
        on_disk[key]: mat._ceil_float32(pub / E4M3_MAX)
        for key, (pub, _w2) in CASES.items()
    }
    pack = tmp_path / "pack"
    pack.mkdir()
    scale_file = pack / "model-kvscales.safetensors"
    mat.save_scale_shard(scale_file, published_scales, {"format": "pt"})
    manifest = {
        "schema_version": 1,
        "artifact_id": "test-envelope",
        "scale_contract": "e4m3fn(x/scale), max_abs/448",
        "loader_contract": "merged model.safetensors.index.json",
        "base_checkpoint": {
            "repo_id": "test/base",
            "index_sha256": _sha256(index_path),
        },
        "scale_file": {
            "filename": "model-kvscales.safetensors",
            "sha256": _sha256(scale_file),
            "tensor_names": sorted(on_disk.values()),
            "tensor_count": len(on_disk),
        },
        "tensors": [
            {"layer": layer, "kind": kind, "max_abs": pub, "scale": pub / E4M3_MAX}
            for (layer, kind), (pub, _w2) in CASES.items()
        ],
    }
    (pack / "kvscales-manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "tensors": {
            f"model.layers.{layer}.self_attn.{kind}_scale": {"max_abs": w2}
            for (layer, kind), (_pub, w2) in CASES.items()
        }
    }
    report_path = tmp_path / "w2-report.json"
    report_path.write_text(json.dumps(report))
    return base, pack, report_path, on_disk


def _ns(mat, base, pack, out, **kw):
    defaults = dict(
        base_checkpoint=str(base),
        output_dir=str(out),
        pack_dir=str(pack),
        mtp_report=None,
        mtp_layer_id=48,
        mtp_scale_from_layer=None,
        mtp_unit_scale=True,
        target_scale_source="published",
        target_report=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_envelope_raises_only_exceeding_tensors(tmp_path):
    mat = _load_mat()
    base, pack, report_path, on_disk = _build(tmp_path, mat)
    out = tmp_path / "overlay"
    mat.materialize(
        _ns(
            mat,
            base,
            pack,
            out,
            target_scale_source="envelope",
            target_report=str(report_path),
        )
    )
    scales = _read_scales(out / "model-kvscales.safetensors")
    for (layer, kind), (pub, w2) in CASES.items():
        name = on_disk[(layer, kind)]
        expected = mat._ceil_float32(max(pub, w2) / E4M3_MAX)
        assert scales[name] == pytest.approx(expected), name
    prov = json.loads((out / "kvscales-provenance.json").read_text())
    assert prov["target_scale_source"] == "envelope"
    tp = prov["target_scales"]
    assert tp[on_disk[(3, "k")]]["source"] == "envelope:w2"  # 14.98 > 9.0
    assert tp[on_disk[(7, "k")]]["source"] == "envelope:w2"  # 8.367 > 8.336
    assert tp[on_disk[(3, "v")]]["source"] == "published"  # 4.0 < 5.0
    assert tp[on_disk[(7, "v")]]["source"] == "published"  # 8.0 < 10.83
    # A raised tensor's scale must exceed the published one.
    assert scales[on_disk[(3, "k")]] > mat._ceil_float32(9.0 / E4M3_MAX)


def test_published_source_copies_verbatim(tmp_path):
    mat = _load_mat()
    base, pack, _report, _on_disk = _build(tmp_path, mat)
    out = tmp_path / "overlay_pub"
    mat.materialize(_ns(mat, base, pack, out))  # default: published
    copied = out / "model-kvscales.safetensors"
    assert _sha256(copied) == _sha256(pack / "model-kvscales.safetensors")
    prov = json.loads((out / "kvscales-provenance.json").read_text())
    assert prov["target_scale_source"] == "published"
    assert prov["target_scales"] is None


def test_envelope_requires_report(tmp_path):
    mat = _load_mat()
    base, pack, _report, _on_disk = _build(tmp_path, mat)
    out = tmp_path / "overlay_err"
    with pytest.raises(ValueError, match="target-report is required"):
        mat.materialize(_ns(mat, base, pack, out, target_scale_source="envelope"))
