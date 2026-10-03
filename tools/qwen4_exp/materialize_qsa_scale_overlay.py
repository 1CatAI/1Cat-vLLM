#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a revision-checked QSA E4M3 target and MTP scale overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import regex as re

E4M3_MAX = 448.0
INDEX_FILENAME = "model.safetensors.index.json"
MANIFEST_FILENAME = "kvscales-manifest.json"
MTP_SCALE_FILENAME = "model-bf16-kvscales-mtp.safetensors"
_SCALE_NAME = re.compile(r"layers\.(\d+)\.self_attn\.([kv])_scale$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _ceil_float32(value: float) -> float:
    """Return the smallest positive float32 that is at least ``value``."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Expected a finite positive scale, got {value}")
    packed = struct.pack("<f", value)
    rounded = struct.unpack("<f", packed)[0]
    if rounded >= value:
        return rounded
    bits = struct.unpack("<I", packed)[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0]


def save_scale_shard(
    path: Path, scales: dict[str, float], metadata: dict[str, str]
) -> None:
    """Write FP32 scalar tensors without importing torch or safetensors."""
    if not scales:
        raise ValueError("Refusing to write an empty scale shard")
    header: dict[str, Any] = {
        "__metadata__": {str(key): str(value) for key, value in metadata.items()}
    }
    payload = bytearray()
    for name in sorted(scales):
        value = float(scales[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid calibrated scale for {name}: {value}")
        start = len(payload)
        payload.extend(struct.pack("<f", value))
        header[name] = {
            "dtype": "F32",
            "shape": [],
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _read_scale_shard(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size < 2 or header_size > file_size - 8:
            raise ValueError(f"Invalid safetensors header size: {header_size}")
        header = json.loads(handle.read(header_size))
        if not isinstance(header, dict):
            raise TypeError("Safetensors header is not a JSON object")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict):
            raise TypeError("Invalid safetensors metadata")
        data_start = 8 + header_size
        values: dict[str, float] = {}
        for name, details in header.items():
            if not isinstance(details, dict):
                raise TypeError(f"Invalid tensor header for {name}")
            if details.get("dtype") != "F32" or details.get("shape") not in ([], [1]):
                raise ValueError(f"Expected an FP32 scalar for {name}")
            offsets = details.get("data_offsets")
            if not (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(offset, int) for offset in offsets)
                and offsets[1] - offsets[0] == 4
            ):
                raise ValueError(f"Invalid scalar offsets for {name}")
            handle.seek(data_start + offsets[0])
            raw_value = handle.read(4)
            if len(raw_value) != 4:
                raise ValueError(f"Truncated tensor data for {name}")
            value = struct.unpack("<f", raw_value)[0]
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Invalid calibrated scale for {name}: {value}")
            values[name] = value
    return values, {str(key): str(value) for key, value in metadata.items()}


def _scale_key(name: str) -> tuple[int, str]:
    match = _SCALE_NAME.search(name)
    if match is None:
        raise ValueError(f"Unrecognized QSA scale tensor name: {name}")
    return int(match.group(1)), match.group(2)


def _target_scales(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    published: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any] | None]:
    source = getattr(args, "target_scale_source", "published")
    if source == "published":
        return published, None
    if source != "envelope":
        raise ValueError(f"Unknown target scale source: {source}")
    report_arg = getattr(args, "target_report", None)
    if not report_arg:
        raise ValueError("target-report is required for the envelope source")
    report = _load_json(Path(report_arg))
    report_tensors = report.get("tensors")
    if not isinstance(report_tensors, dict):
        raise TypeError("Target report lacks a tensors object")
    published_max = {
        (int(item["layer"]), str(item["kind"])): float(item["max_abs"])
        for item in manifest.get("tensors", [])
    }
    output: dict[str, float] = {}
    provenance: dict[str, Any] = {}
    for name, published_scale in published.items():
        layer, kind = _scale_key(name)
        old_max = published_max.get((layer, kind), published_scale * E4M3_MAX)
        report_name = f"model.layers.{layer}.self_attn.{kind}_scale"
        details = report_tensors.get(report_name)
        if not isinstance(details, dict) or "max_abs" not in details:
            raise ValueError(f"Target report is missing {report_name}")
        new_max = float(details["max_abs"])
        use_report = new_max > old_max
        output[name] = _ceil_float32(max(old_max, new_max) / E4M3_MAX)
        provenance[name] = {
            "source": "envelope:w2" if use_report else "published",
            "published_max_abs": old_max,
            "report_max_abs": new_max,
        }
    return output, provenance


def _draft_scales(args: argparse.Namespace) -> tuple[dict[str, float], str]:
    names = {
        "k": "mtp.layers.0.self_attn.k_scale",
        "v": "mtp.layers.0.self_attn.v_scale",
    }
    if getattr(args, "mtp_unit_scale", False):
        return {name: 1.0 for name in names.values()}, "prototype:unit"
    report_arg = getattr(args, "mtp_report", None)
    if not report_arg:
        raise ValueError("mtp-report is required unless --mtp-unit-scale is used")
    report = _load_json(Path(report_arg))
    tensors = report.get("tensors")
    if not isinstance(tensors, dict):
        raise TypeError("MTP report lacks a tensors object")
    from_layer = getattr(args, "mtp_scale_from_layer", None)
    layer = int(from_layer if from_layer is not None else args.mtp_layer_id)
    scales = {}
    for kind, output_name in names.items():
        report_name = f"model.layers.{layer}.self_attn.{kind}_scale"
        details = tensors.get(report_name)
        if not isinstance(details, dict) or "scale" not in details:
            raise ValueError(f"MTP report is missing {report_name}")
        scales[output_name] = float(details["scale"])
    source = f"prototype:from-layer-{layer}" if from_layer is not None else "calibrated"
    return scales, source


def materialize(args: argparse.Namespace) -> None:
    pack_dir = Path(args.pack_dir).resolve()
    base = Path(args.base_checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if base == output or base in output.parents:
        raise ValueError("Output must not be inside the base checkpoint directory")

    manifest = _load_json(pack_dir / MANIFEST_FILENAME)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported scale-pack manifest version")
    base_index_path = base / INDEX_FILENAME
    base_index_hash = _sha256(base_index_path)
    expected_base_hash = manifest["base_checkpoint"]["index_sha256"]
    if base_index_hash != expected_base_hash:
        raise ValueError("Base checkpoint revision does not match the scale pack")

    scale_details = manifest["scale_file"]
    published_path = pack_dir / scale_details["filename"]
    if _sha256(published_path) != scale_details["sha256"]:
        raise ValueError("Published scale file SHA-256 does not match its manifest")
    published, metadata = _read_scale_shard(published_path)
    expected_names = set(scale_details["tensor_names"])
    if (
        set(published) != expected_names
        or len(published) != scale_details["tensor_count"]
    ):
        raise ValueError("Published scale tensor coverage does not match its manifest")

    target, target_provenance = _target_scales(args, manifest, published)
    draft, draft_source = _draft_scales(args)
    base_index = _load_json(base_index_path)
    weight_map = base_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("Base checkpoint index lacks a weight_map object")
    collisions = sorted((set(target) | set(draft)) & set(weight_map))
    if collisions:
        raise ValueError(
            f"Base checkpoint already contains scale tensors: {collisions}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    try:
        staging.mkdir()
        excluded = {
            INDEX_FILENAME,
            scale_details["filename"],
            MTP_SCALE_FILENAME,
            "kvscales-provenance.json",
        }
        for source in base.iterdir():
            if source.name in excluded:
                continue
            (staging / source.name).symlink_to(
                os.path.relpath(source, start=staging),
                target_is_directory=source.is_dir(),
            )

        target_path = staging / scale_details["filename"]
        if target_provenance is None:
            shutil.copy2(published_path, target_path)
        else:
            save_scale_shard(target_path, target, metadata)
        draft_path = staging / MTP_SCALE_FILENAME
        save_scale_shard(draft_path, draft, {"source": draft_source})

        merged_weight_map = dict(weight_map)
        merged_weight_map.update({name: target_path.name for name in target})
        merged_weight_map.update({name: draft_path.name for name in draft})
        merged_index = dict(base_index)
        merged_index["weight_map"] = merged_weight_map
        index_metadata = dict(base_index.get("metadata", {}))
        if isinstance(index_metadata.get("total_size"), int):
            index_metadata["total_size"] += target_path.stat().st_size
            index_metadata["total_size"] += draft_path.stat().st_size
        merged_index["metadata"] = index_metadata
        (staging / INDEX_FILENAME).write_text(
            json.dumps(merged_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provenance = {
            "schema_version": 1,
            "artifact_id": manifest["artifact_id"],
            "base_index_sha256": base_index_hash,
            "target_scale_file_sha256": _sha256(target_path),
            "target_scale_source": getattr(args, "target_scale_source", "published"),
            "target_scales": target_provenance,
            "target_tensor_count": len(target),
            "mtp_scale_file_sha256": _sha256(draft_path),
            "mtp_scale_source": draft_source,
            "mtp_tensor_count": len(draft),
        }
        (staging / "kvscales-provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--mtp-report")
    parser.add_argument("--mtp-layer-id", type=int, default=48)
    parser.add_argument("--mtp-scale-from-layer", type=int)
    parser.add_argument("--mtp-unit-scale", action="store_true")
    parser.add_argument(
        "--target-scale-source",
        choices=("published", "envelope"),
        default="published",
    )
    parser.add_argument("--target-report")
    return parser.parse_args()


if __name__ == "__main__":
    materialize(parse_args())
