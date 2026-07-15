# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and evaluate static draft-vocabulary rankings from model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer


@dataclass(frozen=True)
class Sample:
    token_ids: tuple[int, ...]
    source: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--background", type=Path, action="append")
    parser.add_argument("--background-field", action="append")
    parser.add_argument("--background-limit-per-file", type=int, default=20)
    parser.add_argument("--path-regex")
    parser.add_argument("--subset-size", type=int, action="append")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--model-vocab-size", type=int)
    parser.add_argument("--include-quality-fragments", action="store_true")
    parser.add_argument("--include-explicit-quality-failures", action="store_true")
    parser.add_argument("--top-missing", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--ranking-out", type=Path)
    return parser.parse_args()


def _expand_paths(roots: list[Path], pattern: re.Pattern[str] | None) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file():
            paths.add(root.resolve())
            continue
        elif root.is_dir():
            candidates = [
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in {".json", ".jsonl"}
            ]
        else:
            raise FileNotFoundError(root)
        for path in candidates:
            if pattern is None or pattern.search(str(path.relative_to(root))):
                paths.add(path.resolve())
    return sorted(paths)


def _model_vocab_size(model: Path) -> int:
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("Could not find vocab_size in the model config.")
    return int(vocab_size)


def _token_hash(token_ids: list[int]) -> str:
    values = array("I", token_ids)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _record_output(record: dict[str, Any]) -> tuple[list[int] | None, str | None]:
    choice = record.get("choice")
    if isinstance(choice, dict):
        token_ids = choice.get("token_ids")
        if isinstance(token_ids, list) and all(isinstance(x, int) for x in token_ids):
            return token_ids, None
        text = choice.get("text")
        if isinstance(text, str):
            return None, text
    pred = record.get("pred")
    if isinstance(pred, str):
        return None, pred
    return None, None


def _load_file(path: Path) -> Any:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON at {path}:{line_number}"
                        ) from exc
        return rows
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _collect_samples(
    paths: list[Path],
    tokenizer: Any,
    tokenizer_vocab_size: int,
    include_quality_fragments: bool,
    include_explicit_quality_failures: bool,
) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    seen_hashes: set[str] = set()
    stats = Counter()

    def add(
        token_ids: list[int] | None,
        text: str | None,
        source: str,
    ) -> None:
        if token_ids is None:
            if text is None or not text:
                return
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids = [int(token_id) for token_id in token_ids]
        if not token_ids:
            return
        invalid = [
            token_id
            for token_id in token_ids
            if token_id < 0 or token_id >= tokenizer_vocab_size
        ]
        if invalid:
            raise ValueError(f"Out-of-range token IDs in {source}: {invalid[:5]}")
        digest = _token_hash(token_ids)
        if digest in seen_hashes:
            stats["duplicate_samples"] += 1
            return
        seen_hashes.add(digest)
        samples.append(Sample(tuple(token_ids), source))
        stats["samples"] += 1
        stats["tokens"] += len(token_ids)

    for path in paths:
        payload = _load_file(path)
        stats["files"] += 1
        root = payload if isinstance(payload, dict) else None
        quality = root.get("quality") if root is not None else None
        root_failed = isinstance(quality, dict) and quality.get("passed") is False
        if root_failed and not include_explicit_quality_failures:
            stats["explicit_failed_files"] += 1
        else:
            records = root.get("records") if root is not None else payload
            if isinstance(records, list):
                for index, record in enumerate(records):
                    if not isinstance(record, dict):
                        continue
                    token_ids, text = _record_output(record)
                    record_id = record.get("id", index)
                    add(token_ids, text, f"{path}:{record_id}")

        if not include_quality_fragments or not isinstance(quality, dict):
            continue
        quality_records = quality.get("records")
        if not isinstance(quality_records, list):
            continue
        for index, record in enumerate(quality_records):
            if not isinstance(record, dict):
                continue
            metrics = record.get("metrics")
            record_failed = isinstance(metrics, dict) and metrics.get("passed") is False
            if record_failed and not include_explicit_quality_failures:
                stats["explicit_failed_fragments"] += 1
                continue
            record_id = record.get("id", index)
            for field in ("preview", "tail"):
                text = record.get(field)
                if isinstance(text, str):
                    add(None, text, f"{path}:quality:{record_id}:{field}")

    stats["unique_samples"] = len(samples)
    return samples, dict(stats)


def _collect_background_counts(
    paths: list[Path],
    tokenizer: Any,
    tokenizer_vocab_size: int,
    fields: list[str],
    limit_per_file: int,
) -> tuple[Counter[int], dict[str, int]]:
    counts: Counter[int] = Counter()
    stats = Counter()
    seen_text: set[str] = set()
    for path in paths:
        payload = _load_file(path)
        rows = payload if isinstance(payload, list) else [payload]
        stats["files"] += 1
        used_rows = 0
        for row in rows:
            if used_rows >= limit_per_file:
                break
            if not isinstance(row, dict):
                continue
            used_rows += 1
            stats["rows"] += 1
            for field in fields:
                text = row.get(field)
                if not isinstance(text, str) or not text:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen_text:
                    stats["duplicate_texts"] += 1
                    continue
                seen_text.add(digest)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                invalid = [
                    token_id
                    for token_id in token_ids
                    if token_id < 0 or token_id >= tokenizer_vocab_size
                ]
                if invalid:
                    raise ValueError(
                        f"Out-of-range background token IDs in {path}: {invalid[:5]}"
                    )
                counts.update(token_ids)
                stats["texts"] += 1
                stats["tokens"] += len(token_ids)
    stats["unique_tokens"] = len(counts)
    return counts, dict(stats)


def _count_tokens(samples: list[Sample]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        counts.update(sample.token_ids)
    return counts


def _counter_fingerprint(counts: Counter[int]) -> str:
    digest = hashlib.sha256()
    for token_id, count in sorted(counts.items()):
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
        digest.update(int(count).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _ranking_fingerprint(token_ids: list[int]) -> str:
    return hashlib.sha256(array("I", token_ids).tobytes()).hexdigest()


def _rank_ids(
    token_ids: range,
    counts: Counter[int],
    background_counts: Counter[int],
    mandatory: set[int],
) -> list[int]:
    return sorted(
        token_ids,
        key=lambda token_id: (
            token_id not in mandatory,
            -counts[token_id],
            -background_counts[token_id],
            token_id,
        ),
    )


def _coverage(
    samples: list[Sample],
    eval_counts: Counter[int],
    selected: set[int],
) -> dict[str, Any]:
    total = sum(eval_counts.values())
    hits = sum(count for token_id, count in eval_counts.items() if token_id in selected)
    full_samples = sum(
        1
        for sample in samples
        if all(token_id in selected for token_id in sample.token_ids)
    )
    return {
        "token_hits": hits,
        "token_total": total,
        "token_coverage": hits / total if total else 1.0,
        "missing_occurrences": total - hits,
        "missing_unique_tokens": sum(
            1 for token_id in eval_counts if token_id not in selected
        ),
        "fully_covered_samples": full_samples,
        "sample_total": len(samples),
        "full_sample_coverage": full_samples / len(samples) if samples else 1.0,
    }


def _decode_missing(
    tokenizer: Any,
    eval_counts: Counter[int],
    selected: set[int],
    limit: int,
) -> list[dict[str, Any]]:
    missing = [
        (count, token_id)
        for token_id, count in eval_counts.items()
        if token_id not in selected
    ]
    missing.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "token_id": token_id,
            "count": count,
            "text": tokenizer.decode([token_id]),
        }
        for count, token_id in missing[:limit]
    ]


def main() -> int:
    args = _parse_args()
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive.")
    if args.background_limit_per_file <= 0:
        raise ValueError("--background-limit-per-file must be positive.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = args.model_vocab_size or _model_vocab_size(args.model)
    if model_vocab_size < tokenizer_vocab_size:
        raise ValueError("Model vocabulary cannot be smaller than the tokenizer.")

    subset_sizes = args.subset_size or [131072, 98304, 65536, 32768]
    if any(size <= 0 or size > model_vocab_size for size in subset_sizes):
        raise ValueError("Subset sizes must be in [1, model vocab size].")

    path_pattern = (
        re.compile(args.path_regex, re.IGNORECASE) if args.path_regex else None
    )
    train_paths = _expand_paths(args.train, path_pattern)
    eval_paths = _expand_paths(args.eval, path_pattern)
    background_paths = _expand_paths(args.background or [], None)
    if not train_paths or not eval_paths:
        raise ValueError("Training and evaluation file sets must both be non-empty.")

    collect_args = (
        tokenizer,
        tokenizer_vocab_size,
        args.include_quality_fragments,
        args.include_explicit_quality_failures,
    )
    train_samples, train_stats = _collect_samples(train_paths, *collect_args)
    eval_samples, eval_stats = _collect_samples(eval_paths, *collect_args)
    if not train_samples or not eval_samples:
        raise ValueError("No usable output samples were found.")

    train_counts = _count_tokens(train_samples)
    eval_counts = _count_tokens(eval_samples)
    background_fields = args.background_field or ["context", "input"]
    background_counts, background_stats = _collect_background_counts(
        background_paths,
        tokenizer,
        tokenizer_vocab_size,
        background_fields,
        args.background_limit_per_file,
    )
    mandatory = {
        int(token_id)
        for token_id in tokenizer.all_special_ids
        if 0 <= int(token_id) < model_vocab_size
    }
    global_ranking = _rank_ids(
        range(model_vocab_size), train_counts, background_counts, mandatory
    )

    shard_width = math.ceil(model_vocab_size / args.tp_size)
    local_rankings = []
    for rank in range(args.tp_size):
        start = rank * shard_width
        end = min(start + shard_width, model_vocab_size)
        local_rankings.append(
            _rank_ids(range(start, end), train_counts, background_counts, mandatory)
        )

    seen_ids = set(train_counts)
    cases = []
    for subset_size in subset_sizes:
        global_ids = global_ranking[:subset_size]
        global_selected = set(global_ids)
        global_rank_counts = [
            sum(
                rank * shard_width
                <= token_id
                < min((rank + 1) * shard_width, model_vocab_size)
                for token_id in global_ids
            )
            for rank in range(args.tp_size)
        ]

        quotas = [
            subset_size // args.tp_size + (rank < subset_size % args.tp_size)
            for rank in range(args.tp_size)
        ]
        balanced_by_rank = [
            local_rankings[rank][: quotas[rank]] for rank in range(args.tp_size)
        ]
        balanced_selected = {
            token_id for rank_ids in balanced_by_rank for token_id in rank_ids
        }
        cases.append(
            {
                "subset_size": subset_size,
                "global_top": {
                    "rank_counts": global_rank_counts,
                    "max_local_width": max(global_rank_counts),
                    "coverage": _coverage(eval_samples, eval_counts, global_selected),
                    "top_missing": _decode_missing(
                        tokenizer,
                        eval_counts,
                        global_selected,
                        args.top_missing,
                    ),
                },
                "tp_balanced": {
                    "rank_counts": [len(ids) for ids in balanced_by_rank],
                    "max_local_width": max(len(ids) for ids in balanced_by_rank),
                    "coverage": _coverage(eval_samples, eval_counts, balanced_selected),
                    "top_missing": _decode_missing(
                        tokenizer,
                        eval_counts,
                        balanced_selected,
                        args.top_missing,
                    ),
                },
            }
        )

    seen_coverage = _coverage(eval_samples, eval_counts, seen_ids)
    result = {
        "model": str(args.model),
        "model_vocab_size": model_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "tp_size": args.tp_size,
        "shard_width": shard_width,
        "mandatory_token_ids": sorted(mandatory),
        "ranking_policy": (
            "special tokens first, then target-output frequency descending, "
            "then background frequency descending, then token ID ascending"
        ),
        "fingerprints": {
            "train_frequency_sha256": _counter_fingerprint(train_counts),
            "background_frequency_sha256": _counter_fingerprint(background_counts),
            "global_ranking_sha256": _ranking_fingerprint(global_ranking),
        },
        "train": {
            **train_stats,
            "unique_tokens": len(train_counts),
            "paths": [str(path) for path in train_paths],
        },
        "eval": {
            **eval_stats,
            "unique_tokens": len(eval_counts),
            "paths": [str(path) for path in eval_paths],
            "seen_in_train_coverage": seen_coverage,
        },
        "background": {
            **background_stats,
            "fields": background_fields,
            "limit_per_file": args.background_limit_per_file,
            "paths": [str(path) for path in background_paths],
        },
        "cases": cases,
    }
    text = json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.ranking_out is not None:
        args.ranking_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_vocab_size": model_vocab_size,
                "tokenizer_vocab_size": tokenizer_vocab_size,
                "tp_size": args.tp_size,
                "shard_width": shard_width,
                "mandatory_token_ids": torch.tensor(
                    sorted(mandatory), dtype=torch.int32
                ),
                "global_ranking": torch.tensor(global_ranking, dtype=torch.int32),
                "local_rankings": [
                    torch.tensor(ids, dtype=torch.int32) for ids in local_rankings
                ],
                "fingerprints": result["fingerprints"],
            },
            args.ranking_out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
