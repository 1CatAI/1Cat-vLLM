# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure lossless-compression headroom in DeepSeek V4 FP8 dense weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

DEFAULT_WEIGHTS = (
    "layers.2.attn.wq_a.weight",
    "layers.2.attn.wkv.weight",
    "layers.2.attn.wq_b.weight",
    "layers.2.attn.wo_a.weight",
    "layers.2.attn.wo_b.weight",
    "layers.2.ffn.shared_experts.w1.weight",
    "layers.2.ffn.shared_experts.w3.weight",
    "layers.2.ffn.shared_experts.w2.weight",
    "layers.2.attn.indexer.wq_b.weight",
)
BLOCK = 128


def _percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.to(torch.float32), q).item())


def _sample_block_unique_counts(raw: torch.Tensor, max_blocks: int) -> torch.Tensor:
    n, k = raw.shape
    if n % BLOCK != 0 or k % BLOCK != 0:
        return torch.empty(0, dtype=torch.int64)
    blocks = (
        raw.reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        .permute(0, 2, 1, 3)
        .reshape(-1, BLOCK * BLOCK)
    )
    if max_blocks > 0 and blocks.shape[0] > max_blocks:
        indices = torch.linspace(
            0, blocks.shape[0] - 1, steps=max_blocks, dtype=torch.float64
        ).round()
        blocks = blocks.index_select(0, indices.to(torch.int64))

    unique_counts: list[torch.Tensor] = []
    for chunk in blocks.split(64):
        ordered = torch.sort(chunk, dim=1).values
        counts = 1 + (ordered[:, 1:] != ordered[:, :-1]).sum(dim=1)
        unique_counts.append(counts)
    return torch.cat(unique_counts)


def _analyze_weight(weight: torch.Tensor, max_blocks: int) -> dict[str, Any]:
    if weight.dtype != torch.float8_e4m3fn:
        return {
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "skip": "not_float8_e4m3fn",
        }
    if weight.ndim != 2:
        return {
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "skip": "not_2d",
        }

    raw = weight.contiguous().view(torch.uint8)
    histogram = torch.bincount(raw.reshape(-1).to(torch.int64), minlength=256)
    used = histogram > 0
    probabilities = histogram[used].to(torch.float64) / raw.numel()
    entropy_bits = float((-(probabilities * torch.log2(probabilities))).sum().item())
    global_unique = int(used.sum().item())
    global_fixed_bits = max(1, math.ceil(math.log2(global_unique)))
    positive_zero_fraction = float(histogram[0].item() / raw.numel())
    negative_zero_fraction = float(histogram[128].item() / raw.numel())

    unique_counts = _sample_block_unique_counts(raw, max_blocks)
    block_summary: dict[str, Any] = {"sampled_blocks": int(unique_counts.numel())}
    if unique_counts.numel() > 0:
        index_bits = torch.ceil(torch.log2(unique_counts.to(torch.float64))).clamp_min(
            1
        )
        packed_bytes = (BLOCK * BLOCK * index_bits / 8).ceil() + unique_counts
        block_summary.update(
            {
                "unique_min": int(unique_counts.min().item()),
                "unique_p50": _percentile(unique_counts, 0.50),
                "unique_p90": _percentile(unique_counts, 0.90),
                "unique_p99": _percentile(unique_counts, 0.99),
                "unique_max": int(unique_counts.max().item()),
                "blocks_le_16_fraction": float(
                    (unique_counts <= 16).to(torch.float32).mean().item()
                ),
                "blocks_le_32_fraction": float(
                    (unique_counts <= 32).to(torch.float32).mean().item()
                ),
                "blocks_le_64_fraction": float(
                    (unique_counts <= 64).to(torch.float32).mean().item()
                ),
                "blocks_le_128_fraction": float(
                    (unique_counts <= 128).to(torch.float32).mean().item()
                ),
                "sampled_codebook_ratio": float(
                    (packed_bytes / (BLOCK * BLOCK)).mean().item()
                ),
            }
        )

    top_counts, top_codes = torch.topk(histogram, 16)
    return {
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "numel": weight.numel(),
        "global_unique_codes": global_unique,
        "global_fixed_bits": global_fixed_bits,
        "shannon_entropy_bits": entropy_bits,
        "entropy_lower_bound_ratio": entropy_bits / 8,
        "positive_zero_code_fraction": positive_zero_fraction,
        "negative_zero_code_fraction": negative_zero_fraction,
        "zero_value_fraction": positive_zero_fraction + negative_zero_fraction,
        "top_codes": [
            {
                "code": int(code.item()),
                "count": int(count.item()),
                "fraction": float(count.item() / raw.numel()),
            }
            for count, code in zip(top_counts, top_codes, strict=True)
        ],
        "block_128x128": block_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--weight", action="append", dest="weights")
    parser.add_argument("--max-blocks", type=int, default=1024)
    args = parser.parse_args()

    index_path = args.model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    selected = tuple(args.weights or DEFAULT_WEIGHTS)
    missing = [name for name in selected if name not in weight_map]
    if missing:
        raise KeyError(f"Weights missing from checkpoint index: {missing}")

    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        filename = weight_map[name]
        with safe_open(args.model / filename, framework="pt", device="cpu") as handle:
            results[name] = _analyze_weight(
                handle.get_tensor(name), max_blocks=args.max_blocks
            )

    payload = {
        "contract": {
            "model": str(args.model),
            "format": "lossless raw E4M3 code analysis",
            "block": [BLOCK, BLOCK],
            "max_sampled_blocks_per_weight": args.max_blocks,
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
