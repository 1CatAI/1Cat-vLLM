# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in activation statistics for offline QSA KV calibration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

_CALIBRATION_DIR_ENV = "VLLM_QSA_KV_CALIBRATION_DIR"
_CORPUS_SHARD_ENV = "VLLM_QSA_KV_CALIBRATION_CORPUS_SHARD"
_HISTOGRAM_BINS = 4096
_LOG2_MIN = -24.0
_LOG2_MAX = 16.0


def _summarize(tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.detach().float().abs().reshape(-1)
    finite = torch.isfinite(values)
    finite_values = values[finite]
    nonzero = finite_values[finite_values > 0]
    histogram = torch.histc(
        torch.log2(nonzero).clamp(_LOG2_MIN, _LOG2_MAX),
        bins=_HISTOGRAM_BINS,
        min=_LOG2_MIN,
        max=_LOG2_MAX,
    )
    return {
        "count": values.numel(),
        "finite_count": int(finite.sum().item()),
        "nonzero_count": nonzero.numel(),
        "max_abs": float(finite_values.max().item()) if finite_values.numel() else 0.0,
        "histogram": histogram.to(dtype=torch.int64, device="cpu").tolist(),
    }


def observe_qsa_kv(layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
    """Append one normal-forward K/V observation when calibration is enabled."""
    output_dir = os.getenv(_CALIBRATION_DIR_ENV)
    if not output_dir:
        return
    path = Path(output_dir)
    # The worker receives its environment before engine warmup. The collector
    # creates this marker only after initialization so dummy/profile forwards
    # can never enter the calibration distribution.
    marker = path / "COLLECTING"
    if not marker.is_file():
        return
    # Reading the tiny marker at each forward lets one initialized engine
    # collect multiple explicit corpus shards without admitting its startup
    # warmup into any shard. The environment remains a compatibility fallback.
    corpus_shard = marker.read_text(encoding="utf-8").strip() or os.getenv(
        _CORPUS_SHARD_ENV, "unspecified"
    )
    path.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = str(torch.distributed.get_rank())
    else:
        rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    record = {
        "schema_version": 1,
        "layer_id": layer_id,
        "corpus_shard": corpus_shard,
        "rank": rank,
        "pid": os.getpid(),
        "histogram": {
            "bins": _HISTOGRAM_BINS,
            "log2_min": _LOG2_MIN,
            "log2_max": _LOG2_MAX,
        },
        "k": _summarize(key),
        "v": _summarize(value),
    }
    output_file = path / f"qsa-kv-rank{rank}-pid{os.getpid()}.jsonl"
    with output_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


__all__ = ["observe_qsa_kv"]
