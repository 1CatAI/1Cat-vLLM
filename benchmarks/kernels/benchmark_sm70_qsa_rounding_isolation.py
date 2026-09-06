# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical counterfactual on retained real QSA inputs.

All arithmetic is FP64 except the explicitly modeled probability and output
materializations. This isolates their effect without GPU acquisition; it is
not a CUDA-kernel emulator, model score, performance test, or release gate.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def materialize(result, gate):
    result = result.half()
    if gate is not None:
        result = (result.float() * gate.reshape_as(result).float().sigmoid()).half()
    return result


def isolate(capture, rounded):
    q, k, v = (capture[name] for name in ("q", "k", "v"))
    assert k.shape[2] == 1, "This diagnostic models one KV head per rank"
    indices, table, requests = (
        capture[name].long() for name in ("indices", "table", "requests")
    )
    safe = indices.clamp_min(0)
    valid = (
        (indices >= 0)
        & (safe // k.shape[1] < table.shape[1])
        & (requests[:, None] >= 0)
        & (requests[:, None] < table.shape[0])
    )
    pages = table[
        requests.clamp(0, table.shape[0] - 1)[:, None],
        (safe // k.shape[1]).clamp_max(table.shape[1] - 1),
    ]
    valid &= (pages >= 0) & (pages < k.shape[0])
    keys = k[pages.clamp(0, k.shape[0] - 1), safe % k.shape[1], 0].double()
    values = v[pages.clamp(0, v.shape[0] - 1), safe % v.shape[1], 0].double()
    # Invalid values may be NaN; masking their probability alone is not enough.
    keys = torch.where(valid[:, :, None], keys, 0.0)
    values = torch.where(valid[:, :, None], values, 0.0)
    scores = torch.bmm(q.double(), keys.transpose(1, 2)) / math.sqrt(q.shape[2])
    scores.masked_fill_(~valid[:, None], -1e20)
    tiles = (indices.shape[1] + 15) // 16
    target = 64 if len(q) <= 8 else 32
    splits = min(1 << (tiles.bit_length() - 1), target)
    partials, lses = [], []
    for split in range(splits):
        maximum = torch.full(q.shape[:2], -1e20, dtype=torch.float64)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros(q.shape, dtype=torch.float64)
        for tile in range(split * tiles // splits, (split + 1) * tiles // splits):
            start, end = tile * 16, min((tile + 1) * 16, indices.shape[1])
            logits = scores[:, :, start:end]
            next_max = torch.maximum(maximum, logits.max(-1).values)
            alpha = (maximum - next_max).exp()
            p = torch.where(
                valid[:, None, start:end], (logits - next_max[:, :, None]).exp(), 0
            )
            pv = p.half().double() if rounded else p
            numerator = numerator * alpha[:, :, None] + torch.bmm(
                pv, values[:, start:end]
            )
            denominator = denominator * alpha + p.sum(-1)
            maximum = next_max
        partials.append(numerator / denominator.clamp_min(1e-20)[:, :, None])
        lses.append(
            torch.where(denominator > 0, maximum + denominator.log(), -torch.inf)
        )
    lse = torch.stack(lses)
    weights = (lse - lse.max(0).values).exp().nan_to_num(0)
    merged = (torch.stack(partials) * weights[:, :, :, None]).sum(0)
    merged /= weights.sum(0).clamp_min(1e-20)[:, :, None]
    return materialize(merged, capture["gate"])


def error(a, b):
    a, b = a.double(), b.double()
    return {
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float((a - b).abs().max()),
        "changed": int((a != b).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh output path")
    paths = sorted(args.captures.glob("qsa-r*-n*.pt"))
    if not paths:
        parser.error("No retained QSA inputs found")
    torch.set_num_threads(4)
    results = []
    for path in paths:
        capture = torch.load(path, weights_only=True, map_location="cpu")
        rounded, full = isolate(capture, True), isolate(capture, False)
        record = {
            "capture": path.name,
            "rounded_vs_reference": error(rounded, capture["reference"]),
            "full_vs_reference": error(full, capture["reference"]),
            "rounded_vs_native": error(rounded, capture["candidate"]),
            "full_vs_native": error(full, capture["candidate"]),
            "full_vs_oracle": error(full, capture["oracle"]),
        }
        results.append(record)
        print(json.dumps(record), flush=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": "CPU FP64 counterfactual, not CUDA emulation or quality proof",
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
