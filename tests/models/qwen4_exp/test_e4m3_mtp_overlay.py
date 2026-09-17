# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2 CPU tests for the MTP KV-scale overlay (D3 draft visibility + D4
materializer). Builds a tiny synthetic base checkpoint + scale pack and proves:
the drafter-visible shard is selected by the first allow-pattern, MTP scales are
written under mtp.layers.0.self_attn.{k,v}_scale, the target path is unchanged,
provenance labels the source, and an existing output dir is refused."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP, _remap_mtp_weight_name

MATERIALIZER = Path("/home/l/work/flash-next/dev-kv/tools/materialize_overlay.py")
pytestmark = pytest.mark.skipif(
    not MATERIALIZER.is_file(), reason="phase-2 materializer not present"
)


def _load_materializer():
    spec = importlib.util.spec_from_file_location("mtp_materializer", MATERIALIZER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_safetensors_scales(path: Path) -> dict[str, float]:
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


def _build_pack_and_base(tmp_path, mat):
    base = tmp_path / "base"
    base.mkdir()
    # a couple of ordinary base shards + a bf16 shard the drafter would read
    (base / "model-bf16-0001.safetensors").write_bytes(b"\x00" * 16)
    (base / "config.json").write_text("{}")
    weight_map = {"model.embed_tokens.weight": "model-bf16-0001.safetensors"}
    index = {"metadata": {"total_size": 16}, "weight_map": weight_map}
    index_path = base / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    pack = tmp_path / "pack"
    pack.mkdir()
    target_names = [
        f"model.layers.{i}.self_attn.{k}_scale" for i in range(2) for k in ("k", "v")
    ]
    target_scales = {n: 0.1 + 0.01 * i for i, n in enumerate(target_names)}
    target_file = pack / "model-kvscales.safetensors"
    mat.save_scale_shard(target_file, target_scales, {"format": "pt"})
    manifest = {
        "schema_version": 1,
        "artifact_id": "test-e4m3",
        "scale_contract": "e4m3fn(x/scale), max_abs/448",
        "loader_contract": "merged model.safetensors.index.json",
        "base_checkpoint": {
            "repo_id": "test/base",
            "index_sha256": _sha256(index_path),
        },
        "scale_file": {
            "filename": "model-kvscales.safetensors",
            "sha256": _sha256(target_file),
            "tensor_names": target_names,
            "tensor_count": len(target_names),
        },
    }
    (pack / "kvscales-manifest.json").write_text(json.dumps(manifest, indent=2))
    return base, pack, target_names


def _ns(mat, base, pack, out, **kw):
    import argparse

    defaults = dict(
        base_checkpoint=str(base),
        output_dir=str(out),
        pack_dir=str(pack),
        mtp_report=None,
        mtp_layer_id=48,
        mtp_scale_from_layer=None,
        mtp_unit_scale=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_materialize_unit_scale_overlay(tmp_path):
    mat = _load_materializer()
    base, pack, target_names = _build_pack_and_base(tmp_path, mat)
    out = tmp_path / "overlay"
    mat.materialize(_ns(mat, base, pack, out, mtp_unit_scale=True))

    mtp_file = out / "model-bf16-kvscales-mtp.safetensors"
    assert mtp_file.is_file()
    scales = _read_safetensors_scales(mtp_file)
    assert scales == {
        "mtp.layers.0.self_attn.k_scale": 1.0,
        "mtp.layers.0.self_attn.v_scale": 1.0,
    }
    # Index carries BOTH target and MTP scales, each pointing at its shard.
    index = json.loads((out / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    assert wm["mtp.layers.0.self_attn.k_scale"] == "model-bf16-kvscales-mtp.safetensors"
    for name in target_names:
        assert wm[name] == "model-kvscales.safetensors"
    prov = json.loads((out / "kvscales-provenance.json").read_text())
    assert prov["mtp_scale_source"] == "prototype:unit"
    assert prov["target_tensor_count"] == len(target_names)


def test_materialize_from_layer_and_calibrated(tmp_path):
    mat = _load_materializer()
    base, pack, _ = _build_pack_and_base(tmp_path, mat)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "tensors": {
                    "model.layers.47.self_attn.k_scale": {"scale": 0.7},
                    "model.layers.47.self_attn.v_scale": {"scale": 0.8},
                    "model.layers.48.self_attn.k_scale": {"scale": 0.3},
                    "model.layers.48.self_attn.v_scale": {"scale": 0.4},
                }
            }
        )
    )
    out1 = tmp_path / "ov_from47"
    mat.materialize(
        _ns(mat, base, pack, out1, mtp_report=str(report), mtp_scale_from_layer=47)
    )
    s1 = _read_safetensors_scales(out1 / "model-bf16-kvscales-mtp.safetensors")
    assert s1["mtp.layers.0.self_attn.k_scale"] == pytest.approx(0.7)
    assert s1["mtp.layers.0.self_attn.v_scale"] == pytest.approx(0.8)
    prov1 = json.loads((out1 / "kvscales-provenance.json").read_text())
    assert prov1["mtp_scale_source"] == "prototype:from-layer-47"

    out2 = tmp_path / "ov_calibrated"
    mat.materialize(_ns(mat, base, pack, out2, mtp_report=str(report)))
    s2 = _read_safetensors_scales(out2 / "model-bf16-kvscales-mtp.safetensors")
    assert s2["mtp.layers.0.self_attn.k_scale"] == pytest.approx(0.3)  # layer 48
    prov2 = json.loads((out2 / "kvscales-provenance.json").read_text())
    assert prov2["mtp_scale_source"] == "calibrated"


def test_materialize_refuses_existing_dir(tmp_path):
    mat = _load_materializer()
    base, pack, _ = _build_pack_and_base(tmp_path, mat)
    out = tmp_path / "overlay"
    out.mkdir()
    with pytest.raises(FileExistsError):
        mat.materialize(_ns(mat, base, pack, out, mtp_unit_scale=True))


def test_draft_first_allow_pattern_selects_mtp_shard():
    patterns = Qwen4ExpMTP.allow_patterns_overrides
    assert patterns[0] == "model-bf16-*.safetensors"
    # The MTP scale shard is matched by the drafter's FIRST pattern; the target
    # scale shard is NOT (so the drafter never loads target scales).
    assert fnmatch.fnmatch("model-bf16-kvscales-mtp.safetensors", patterns[0])
    assert not fnmatch.fnmatch("model-kvscales.safetensors", patterns[0])


def test_mtp_shard_names_map_onto_draft_modules():
    assert (
        _remap_mtp_weight_name("mtp.layers.0.self_attn.k_scale")
        == "model.layers.0.self_attn.k_scale"
    )
    # Target scale in the other shard is not rerouted by the drafter.
    assert _remap_mtp_weight_name("model.layers.1.self_attn.k_scale") is None
