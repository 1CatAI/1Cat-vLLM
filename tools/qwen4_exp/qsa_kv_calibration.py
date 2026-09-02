#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize QSA KV calibration traces and build a checkpoint overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

E4M3_MAX = 448.0
SCALE_FILENAME = "model-kvscales.safetensors"
INDEX_FILENAME = "model.safetensors.index.json"


def _ceil_float32(value: float) -> float:
    """Return the least finite FP32 value greater than or equal to value."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Cannot round invalid positive FP32 scale: {value}")
    rounded = struct.unpack(">f", struct.pack(">f", value))[0]
    if rounded < value:
        bits = struct.unpack(">I", struct.pack(">f", rounded))[0] + 1
        rounded = struct.unpack(">f", struct.pack(">I", bits))[0]
    if not math.isfinite(rounded):
        raise ValueError(f"Scale overflows FP32: {value}")
    return rounded


def prepare(args: argparse.Namespace) -> None:
    """Freeze source text/messages into exact checkpoint prompt token IDs."""
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    source = Path(args.input_jsonl).resolve()
    output = Path(args.output_jsonl).resolve()
    provenance_path = output.with_name(output.name + ".manifest.json")
    if output.exists():
        raise FileExistsError(f"Prepared calibration input already exists: {output}")
    if provenance_path.exists():
        raise FileExistsError(f"Preparation manifest already exists: {provenance_path}")
    audit_path = (
        Path(args.processor_audit_jsonl).resolve()
        if args.processor_audit_jsonl
        else None
    )
    corpus_manifest_path = (
        Path(args.corpus_manifest).resolve() if args.corpus_manifest else None
    )
    if (audit_path is None) != (corpus_manifest_path is None):
        raise ValueError(
            "--processor-audit-jsonl and --corpus-manifest must be supplied together"
        )

    audits: list[dict[str, Any]] | None = None
    if corpus_manifest_path is not None and audit_path is not None:
        corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        if _sha256(source) != corpus_manifest.get("corpus_sha256"):
            raise ValueError("Calibration corpus SHA-256 does not match its manifest")
        if _sha256(audit_path) != corpus_manifest.get("processor_audit_sha256"):
            raise ValueError("Processor audit SHA-256 does not match corpus manifest")
        audits = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(audits) != corpus_manifest.get("record_count"):
            raise ValueError("Processor audit record count does not match manifest")

    model = str(Path(args.model).resolve())
    processor = None
    if audits is not None:
        processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            local_files_only=True,
        )
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if audits is not None and line_number > len(audits):
                raise ValueError(
                    "Calibration corpus has more rows than processor audit"
                )
            audit = audits[line_number - 1] if audits is not None else None
            add_generation_prompt = bool(
                record.get("messages")
                and record["messages"][-1].get("role") != "assistant"
            )
            template_kwargs = {}
            if isinstance(record.get("tools"), list):
                template_kwargs["tools"] = record["tools"]
            if processor is not None:
                if "messages" not in record:
                    raise ValueError(
                        f"Audited line {line_number} requires messages content"
                    )
                rendered = processor.apply_chat_template(
                    [record["messages"]],
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
                inputs = processor(text=rendered, padding=True, return_tensors="pt")
                input_ids = inputs["input_ids"].detach().cpu().contiguous()
                attention_mask = inputs["attention_mask"].detach().cpu().contiguous()
                token_ids = input_ids[0].tolist()
                assert audit is not None
                actual_audit = {
                    "id": record.get("id", str(line_number)),
                    "category": record.get("category", "unspecified"),
                    "add_generation_prompt": add_generation_prompt,
                    "active_tokens": int(attention_mask.sum().item()),
                    "input_ids_shape": list(input_ids.shape),
                    "input_ids_sha256": hashlib.sha256(
                        input_ids.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "attention_mask_sha256": hashlib.sha256(
                        attention_mask.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
                mismatches = [
                    f"{key}: expected {audit.get(key)!r}, got {value!r}"
                    for key, value in actual_audit.items()
                    if audit.get(key) != value
                ]
                if mismatches:
                    raise ValueError(
                        f"Processor audit mismatch at line {line_number}: "
                        + "; ".join(mismatches)
                    )
            elif "messages" in record:
                assert tokenizer is not None
                token_ids = tokenizer.apply_chat_template(
                    record["messages"],
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
            elif isinstance(record.get("text"), str):
                assert tokenizer is not None
                token_ids = tokenizer.encode(record["text"], add_special_tokens=True)
            else:
                raise ValueError(
                    f"Line {line_number} requires messages or text source content"
                )
            if hasattr(token_ids, "input_ids"):
                token_ids = token_ids.input_ids
            if hasattr(token_ids, "ids"):
                token_ids = token_ids.ids
            token_ids = list(token_ids)
            original_tokens = len(token_ids)
            if args.max_prompt_tokens is not None:
                token_ids = token_ids[: args.max_prompt_tokens]
            if not token_ids:
                raise ValueError(f"Line {line_number} tokenized to an empty prompt")
            rows.append(
                {
                    "prompt_token_ids": token_ids,
                    "calibration_metadata": {
                        "source_line": line_number,
                        "source_id": record.get("id", str(line_number)),
                        "category": record.get("category", "unspecified"),
                        "has_tools": bool(template_kwargs),
                        "original_tokens": original_tokens,
                        "used_tokens": len(token_ids),
                    },
                }
            )
    if not rows:
        raise ValueError("Calibration source input is empty")
    if audits is not None and len(rows) != len(audits):
        raise ValueError("Calibration corpus and processor audit lengths differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    provenance = {
        "schema_version": 1,
        "model": model,
        "source_jsonl": str(source),
        "source_sha256": _sha256(source),
        "prepared_jsonl": str(output),
        "prepared_sha256": _sha256(output),
        "record_count": len(rows),
        "active_tokens": sum(
            int(row["calibration_metadata"]["used_tokens"]) for row in rows
        ),
        "max_prompt_tokens": args.max_prompt_tokens,
        "processor_audit_verified": audits is not None,
        "processor_audit_jsonl": str(audit_path) if audit_path else None,
        "processor_audit_sha256": _sha256(audit_path) if audit_path else None,
        "corpus_manifest": str(corpus_manifest_path) if corpus_manifest_path else None,
        "corpus_manifest_sha256": (
            _sha256(corpus_manifest_path) if corpus_manifest_path else None
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_prepared_prompts(input_path: Path) -> list[dict[str, list[int]]]:
    prompts: list[dict[str, list[int]]] = []
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            token_ids = row if isinstance(row, list) else row.get("prompt_token_ids")
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or not all(isinstance(token, int) and token >= 0 for token in token_ids)
            ):
                raise ValueError(
                    f"Invalid prompt_token_ids at {input_path}:{line_number}"
                )
            prompts.append({"prompt_token_ids": token_ids})
    if not prompts:
        raise ValueError(f"Calibration input is empty: {input_path}")
    return prompts


def collect(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "COLLECTING"
    marker.unlink(missing_ok=True)

    if args.shard_manifest:
        if args.input_jsonl or args.corpus_shard:
            raise ValueError(
                "--shard-manifest cannot be combined with --input-jsonl/--corpus-shard"
            )
        shard_manifest_path = Path(args.shard_manifest).resolve()
        shard_document = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
        shard_rows = (
            shard_document.get("shards")
            if isinstance(shard_document, dict)
            else shard_document
        )
        if not isinstance(shard_rows, list) or not shard_rows:
            raise ValueError("Shard manifest requires a non-empty shards list")
        collection_id = args.collection_id or shard_manifest_path.stem
        shard_specs = [
            {
                "corpus_shard": str(row["corpus_shard"]),
                "input_path": Path(row["input_jsonl"]).resolve(),
            }
            for row in shard_rows
        ]
    else:
        if not args.input_jsonl or not args.corpus_shard:
            raise ValueError(
                "Single-shard collection requires --input-jsonl and --corpus-shard"
            )
        collection_id = args.collection_id or args.corpus_shard
        shard_specs = [
            {
                "corpus_shard": args.corpus_shard,
                "input_path": Path(args.input_jsonl).resolve(),
            }
        ]
        shard_manifest_path = None
    shard_names = [str(spec["corpus_shard"]) for spec in shard_specs]
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("Collection corpus_shard names must be unique")

    manifest_path = output_dir / f"collection-{collection_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Collection manifest already exists: {manifest_path}")
    os.environ["VLLM_QSA_KV_CALIBRATION_DIR"] = str(output_dir)
    os.environ["VLLM_QSA_KV_CALIBRATION_CORPUS_SHARD"] = shard_names[0]
    prepared_shards = [
        {**spec, "prompts": _load_prepared_prompts(spec["input_path"])}
        for spec in shard_specs
    ]

    # Import only after the calibration environment is fixed for worker spawn.
    from vllm import LLM, SamplingParams

    engine_kwargs: dict[str, Any] = {
        "model": str(Path(args.model).resolve()),
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "float16",
        "kv_cache_dtype": "float16",
        "language_model_only": True,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": True,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**engine_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    for shard in prepared_shards:
        marker.write_text(str(shard["corpus_shard"]) + "\n", encoding="utf-8")
        try:
            llm.generate(shard["prompts"], sampling, use_tqdm=True)
        finally:
            marker.unlink(missing_ok=True)

    try:
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unknown"
    manifest = {
        "schema_version": 1,
        "model": str(Path(args.model).resolve()),
        "collection_id": collection_id,
        "shard_manifest": str(shard_manifest_path) if shard_manifest_path else None,
        "shard_manifest_sha256": (
            _sha256(shard_manifest_path) if shard_manifest_path else None
        ),
        "shards": [
            {
                "input_jsonl": str(shard["input_path"]),
                "input_sha256": _sha256(shard["input_path"]),
                "corpus_shard": shard["corpus_shard"],
                "prompt_count": len(shard["prompts"]),
            }
            for shard in prepared_shards
        ],
        "prompt_count": sum(len(shard["prompts"]) for shard in prepared_shards),
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "float16",
        "kv_cache_dtype": "float16",
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "language_model_only": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "source_revision": source_revision,
        "warmup_excluded_by_marker": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_from_log_hist(
    histogram: list[int],
    zeros: int,
    total: int,
    log2_min: float,
    log2_max: float,
    q: float,
) -> float:
    target = math.ceil(q * total)
    if target <= zeros:
        return 0.0
    cumulative = zeros
    width = (log2_max - log2_min) / len(histogram)
    for index, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            # Return the bin upper bound, making the reported percentile
            # conservative at the resolution of the calibration histogram.
            return 2.0 ** (log2_min + (index + 1) * width)
    return 2.0**log2_max


def _empty_accumulator(bins: int) -> dict[str, Any]:
    return {
        "count": 0,
        "finite_count": 0,
        "nonzero_count": 0,
        "max_abs": 0.0,
        "histogram": [0] * bins,
    }


def _merge_stat(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["count"] += int(source["count"])
    target["finite_count"] += int(source["finite_count"])
    target["nonzero_count"] += int(source["nonzero_count"])
    target["max_abs"] = max(float(target["max_abs"]), float(source["max_abs"]))
    source_hist = source["histogram"]
    if len(source_hist) != len(target["histogram"]):
        raise ValueError("Calibration records use inconsistent histogram sizes")
    target["histogram"] = [
        left + int(right) for left, right in zip(target["histogram"], source_hist)
    ]


def summarize(args: argparse.Namespace) -> None:
    input_dirs = (
        args.input_dir if isinstance(args.input_dir, list) else [args.input_dir]
    )
    inputs = sorted(
        path
        for input_dir in input_dirs
        for path in Path(input_dir).glob("qsa-kv-rank*-pid*.jsonl")
    )
    if not inputs:
        raise ValueError(
            "No QSA KV calibration JSONL files found in " + ", ".join(input_dirs)
        )
    excluded_shards = set(getattr(args, "exclude_corpus_shard", []) or [])

    aggregate: dict[tuple[int, str], dict[str, Any]] = {}
    by_shard: dict[tuple[str, int, str], dict[str, Any]] = {}
    histogram_spec: dict[str, Any] | None = None
    records = 0
    for path in inputs:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("schema_version") != 1:
                    raise ValueError(f"Unsupported calibration schema in {path}")
                current_spec = record["histogram"]
                if histogram_spec is None:
                    histogram_spec = current_spec
                elif current_spec != histogram_spec:
                    raise ValueError("Calibration records use inconsistent histograms")
                layer_id = int(record["layer_id"])
                shard = str(record["corpus_shard"])
                if shard in excluded_shards:
                    continue
                bins = int(current_spec["bins"])
                for kind in ("k", "v"):
                    global_key = (layer_id, kind)
                    shard_key = (shard, layer_id, kind)
                    aggregate.setdefault(global_key, _empty_accumulator(bins))
                    by_shard.setdefault(shard_key, _empty_accumulator(bins))
                    _merge_stat(aggregate[global_key], record[kind])
                    _merge_stat(by_shard[shard_key], record[kind])
                records += 1

    assert histogram_spec is not None
    layers = sorted({layer_id for layer_id, _ in aggregate})
    if len(layers) != args.expected_layers:
        raise ValueError(
            f"Expected {args.expected_layers} QSA layers, "
            f"observed {len(layers)}: {layers}"
        )
    expected_pairs = {(layer_id, kind) for layer_id in layers for kind in ("k", "v")}
    if set(aggregate) != expected_pairs:
        raise ValueError(
            "Calibration is missing one or more per-layer K/V distributions"
        )

    log2_min = float(histogram_spec["log2_min"])
    log2_max = float(histogram_spec["log2_max"])
    tensors: dict[str, Any] = {}
    for layer_id in layers:
        for kind in ("k", "v"):
            stat = aggregate[(layer_id, kind)]
            if stat["finite_count"] != stat["count"]:
                raise ValueError(
                    f"Non-finite activation found in layer {layer_id} {kind.upper()}"
                )
            if stat["max_abs"] <= 0.0 or not math.isfinite(stat["max_abs"]):
                raise ValueError(
                    f"Invalid activation range in layer {layer_id} {kind.upper()}"
                )
            # The overlay persists FP32. Round upward in that exact storage
            # type so the runtime converter cannot saturate the observed max
            # merely because max_abs/448 rounded down during serialization.
            scale = _ceil_float32(stat["max_abs"] / E4M3_MAX)
            zeros = stat["finite_count"] - stat["nonzero_count"]
            tensor_name = f"model.layers.{layer_id}.self_attn.{kind}_scale"
            shard_scales = {
                shard: _ceil_float32(shard_stat["max_abs"] / E4M3_MAX)
                for (shard, shard_layer, shard_kind), shard_stat in sorted(
                    by_shard.items()
                )
                if shard_layer == layer_id and shard_kind == kind
            }
            tensors[tensor_name] = {
                "layer_id": layer_id,
                "kind": kind,
                "max_abs": stat["max_abs"],
                "scale": scale,
                "saturation_ratio": 0.0,
                "p99_9_abs_upper_bound": _quantile_from_log_hist(
                    stat["histogram"],
                    zeros,
                    stat["finite_count"],
                    log2_min,
                    log2_max,
                    0.999,
                ),
                "p99_99_abs_upper_bound": _quantile_from_log_hist(
                    stat["histogram"],
                    zeros,
                    stat["finite_count"],
                    log2_min,
                    log2_max,
                    0.9999,
                ),
                "count": stat["count"],
                "shard_scales": shard_scales,
                "shard_scale_min_over_global": min(shard_scales.values()) / scale,
                "shard_scale_max_over_global": max(shard_scales.values()) / scale,
            }

    report = {
        "schema_version": 1,
        "converter_contract": "stored=e4m3fn(x/scale); reconstructed=stored*scale",
        "scale_method": "per-layer per-tensor max_abs/448",
        "scale_storage": "float32 rounded upward",
        "e4m3_max": E4M3_MAX,
        "saturation_definition": "abs(x) > 448*scale",
        "histogram": histogram_spec,
        "input_files": [str(path.resolve()) for path in inputs],
        "excluded_corpus_shards": sorted(excluded_shards),
        "records": records,
        "qsa_layer_ids": layers,
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_scale_envelope(args: argparse.Namespace) -> None:
    """Keep one report's percentiles while enveloping observed maxima."""
    base_path = Path(args.base_report).resolve()
    scale_paths = [Path(path).resolve() for path in args.scale_report]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Scale-envelope report already exists: {output}")

    report = json.loads(base_path.read_text(encoding="utf-8"))
    tensors = report.get("tensors", {})
    if not tensors or report.get("tensor_count") != len(tensors):
        raise ValueError("Base calibration report has an invalid tensor set")
    envelope_inputs = [base_path]
    for scale_path in scale_paths:
        candidate = json.loads(scale_path.read_text(encoding="utf-8"))
        candidate_tensors = candidate.get("tensors", {})
        if set(candidate_tensors) != set(tensors):
            raise ValueError(
                f"Scale report tensor set differs from base report: {scale_path}"
            )
        if candidate.get("e4m3_max") != report.get("e4m3_max"):
            raise ValueError(
                f"Scale report converter range differs from base report: {scale_path}"
            )
        envelope_inputs.append(scale_path)
        for name, candidate_details in candidate_tensors.items():
            details = tensors[name]
            details.setdefault("distribution_max_abs", details["max_abs"])
            details["max_abs"] = max(
                float(details["max_abs"]), float(candidate_details["max_abs"])
            )
            details["scale"] = _ceil_float32(details["max_abs"] / E4M3_MAX)
            details["saturation_ratio"] = 0.0
            shard_scales = details.setdefault("shard_scales", {})
            for shard, scale in candidate_details.get("shard_scales", {}).items():
                existing = shard_scales.get(shard)
                if existing is not None and existing != scale:
                    raise ValueError(
                        f"Conflicting scale for shard {shard!r} in tensor {name}"
                    )
                shard_scales[shard] = scale
            details["shard_scale_min_over_global"] = (
                min(shard_scales.values()) / details["scale"]
            )
            details["shard_scale_max_over_global"] = (
                max(shard_scales.values()) / details["scale"]
            )

    report["scale_method"] = "per-layer per-tensor observed max envelope/448"
    report["distribution_statistics_report"] = {
        "path": str(base_path),
        "sha256": _sha256(base_path),
    }
    report["scale_envelope_reports"] = [
        {"path": str(path), "sha256": _sha256(path)} for path in envelope_inputs
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_overlay(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import save_file

    base = Path(args.base_checkpoint).resolve()
    report_path = Path(args.report).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Overlay output already exists: {output}")
    base_index_path = base / INDEX_FILENAME
    if not base_index_path.is_file():
        raise FileNotFoundError(f"Base checkpoint lacks {INDEX_FILENAME}: {base}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tensors = report.get("tensors", {})
    if (
        report.get("tensor_count") != args.expected_layers * 2
        or len(tensors) != args.expected_layers * 2
    ):
        raise ValueError(
            "Calibration report does not contain the required K/V scale count"
        )

    base_index = json.loads(base_index_path.read_text(encoding="utf-8"))
    weight_map = dict(base_index["weight_map"])
    layer_prefixes: dict[str, set[int]] = {}
    for name in weight_map:
        if ".layers." not in name or not name.endswith(".self_attn.k_proj.weight"):
            continue
        prefix, remainder = name.split(".layers.", 1)
        layer_id_text = remainder.split(".", 1)[0]
        if not layer_id_text.isdigit():
            continue
        full_prefix = prefix + ".layers"
        layer_prefixes.setdefault(full_prefix, set()).add(int(layer_id_text))
    if not layer_prefixes:
        raise ValueError("Cannot find checkpoint self-attention K projections")
    largest_count = max(len(layer_ids) for layer_ids in layer_prefixes.values())
    candidates = sorted(
        prefix
        for prefix, layer_ids in layer_prefixes.items()
        if len(layer_ids) == largest_count
    )
    if len(candidates) != 1:
        raise ValueError(
            "Cannot uniquely infer the main checkpoint layer prefix from "
            f"self-attention K projections: {candidates}"
        )
    checkpoint_layer_prefix = candidates[0]

    scale_values: dict[str, torch.Tensor] = {}
    for name, details in sorted(tensors.items()):
        report_prefix = "model.layers."
        if not name.startswith(report_prefix):
            raise ValueError(f"Unexpected calibration tensor name: {name}")
        checkpoint_name = f"{checkpoint_layer_prefix}." + name.removeprefix(
            report_prefix
        )
        value = 1.0 if args.unit_scale_negative_control else float(details["scale"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid scale for {checkpoint_name}: {value}")
        scale_values[checkpoint_name] = torch.tensor(value, dtype=torch.float32)

    collisions = sorted(set(scale_values) & set(weight_map))
    if collisions:
        raise ValueError(
            f"Base checkpoint already contains QSA scale tensors: {collisions}"
        )

    output.mkdir(parents=True)
    excluded = {INDEX_FILENAME, SCALE_FILENAME, "kvscales-provenance.json"}
    for source in base.iterdir():
        if source.name in excluded:
            continue
        os.symlink(source, output / source.name, target_is_directory=source.is_dir())

    scale_path = output / SCALE_FILENAME
    save_file(
        scale_values,
        str(scale_path),
        metadata={
            "format": "pt",
            "qsa_kv_scale_contract": "e4m3fn(x/scale), max_abs/448",
            "negative_control": str(bool(args.unit_scale_negative_control)).lower(),
        },
    )
    weight_map.update({name: SCALE_FILENAME for name in scale_values})
    metadata = dict(base_index.get("metadata", {}))
    if isinstance(metadata.get("total_size"), int):
        metadata["total_size"] += scale_path.stat().st_size
    merged_index = {"metadata": metadata, "weight_map": weight_map}
    (output / INDEX_FILENAME).write_text(
        json.dumps(merged_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "base_checkpoint": str(base),
        "base_index_sha256": _sha256(base_index_path),
        "calibration_report": str(report_path),
        "calibration_report_sha256": _sha256(report_path),
        "scale_file": SCALE_FILENAME,
        "scale_file_sha256": _sha256(scale_path),
        "tensor_count": len(scale_values),
        "checkpoint_layer_prefix": checkpoint_layer_prefix,
        "unit_scale_negative_control": bool(args.unit_scale_negative_control),
        "loader_contract": f"merged {INDEX_FILENAME}",
    }
    (output / "kvscales-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preparer = subparsers.add_parser("prepare")
    preparer.add_argument("--model", required=True)
    preparer.add_argument("--input-jsonl", required=True)
    preparer.add_argument("--output-jsonl", required=True)
    preparer.add_argument("--max-prompt-tokens", type=int)
    preparer.add_argument("--processor-audit-jsonl")
    preparer.add_argument("--corpus-manifest")
    preparer.set_defaults(func=prepare)

    collector = subparsers.add_parser("collect")
    collector.add_argument("--model", required=True)
    collector.add_argument("--input-jsonl")
    collector.add_argument("--output-dir", required=True)
    collector.add_argument("--corpus-shard")
    collector.add_argument("--shard-manifest")
    collector.add_argument("--collection-id")
    collector.add_argument("--tensor-parallel-size", type=int, default=4)
    collector.add_argument("--max-model-len", type=int)
    collector.add_argument("--max-num-batched-tokens", type=int, default=2048)
    collector.add_argument("--max-num-seqs", type=int, default=32)
    collector.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    collector.add_argument("--max-tokens", type=int, default=1)
    collector.set_defaults(func=collect)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--input-dir", action="append", required=True)
    summary.add_argument("--exclude-corpus-shard", action="append", default=[])
    summary.add_argument("--output", required=True)
    summary.add_argument("--expected-layers", type=int, default=12)
    summary.set_defaults(func=summarize)

    envelope = subparsers.add_parser("envelope")
    envelope.add_argument("--base-report", required=True)
    envelope.add_argument("--scale-report", action="append", required=True)
    envelope.add_argument("--output", required=True)
    envelope.set_defaults(func=build_scale_envelope)

    overlay = subparsers.add_parser("overlay")
    overlay.add_argument("--base-checkpoint", required=True)
    overlay.add_argument("--report", required=True)
    overlay.add_argument("--output-dir", required=True)
    overlay.add_argument("--expected-layers", type=int, default=12)
    overlay.add_argument("--unit-scale-negative-control", action="store_true")
    overlay.set_defaults(func=build_overlay)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
