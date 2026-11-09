# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Project 25/20% raw-slot increments against an observed full 30% run.

Reads checkpoint headers and an existing resource report; allocates no GPU or
model payload. Projections are labeled explicitly and are not runtime results
for the alternative fractions. Raw storage can be qualified separately with
shared_checkpoint_probe --offload.
"""

import argparse
import json
from pathlib import Path

import gguf

from vllm.model_executor.layers.quantization.gguf_shared_layout import (
    SharedExpertBudget,
    SharedExpertLayout,
)


def capacities(root, observation):
    types = {}
    for path in sorted((root / "gguf").glob("*-of-00004.gguf")):
        reader = gguf.GGUFReader(path)
        types.update({t.name: int(t.tensor_type) for t in reader.tensors})
    geometries = []
    for layer in range(40):
        geometries.append(
            tuple(
                SharedExpertLayout.for_type(
                    5120, 2304, 4, 0, role, types[f"blk.{layer}.ffn_{name}_exps.weight"]
                )
                for role, name in (("w1", "gate"), ("w3", "up"), ("w2", "down"))
            )
        )
    sizes = tuple(
        sum(g.source_rows * g.source_row_bytes for g in layer) for layer in geometries
    )
    records = []
    for fraction in (0.30, 0.25, 0.20):
        budget = SharedExpertBudget(384, fraction, sizes)
        raw = 0
        for layer in geometries:
            for g in layer:
                raw = (raw + 255) // 256 * 256
                raw += budget.gpu_slots * g.device_rows * g.device_row_bytes
                raw += g.guard_bytes
        records.append(
            dict(
                offload_fraction=fraction,
                cold_experts=budget.cold_experts,
                gpu_slots=budget.gpu_slots,
                shared_cold_bytes=budget.cold_bytes,
                exchange_bytes=budget.exchange_bytes,
                raw_gpu_bytes_per_rank=raw,
            )
        )
    if observation["offload_fraction"] != 0.30 or not observation["dspark"]:
        raise ValueError("Projection requires a full DSpark 30% observation")
    for record in records:
        delta = record["raw_gpu_bytes_per_rank"] - records[0]["raw_gpu_bytes_per_rank"]
        record["raw_gpu_increment_per_rank"] = delta
        record["projected_driver_peaks"] = [
            dict(
                uuid=gpu["uuid"],
                projected_peak_bytes=gpu["peak_used_bytes"] + delta,
                total_bytes=gpu["total_bytes"],
                projected_deficit_bytes=max(
                    0, gpu["peak_used_bytes"] + delta - gpu["total_bytes"]
                ),
            )
            for gpu in observation["resources"]["gpu"]
        ]
    return dict(
        scope="Alternative-fraction peak projections; no full model execution",
        assumption="Same non-expert live allocations, workspace and driver footprint",
        baseline_status=observation["status"],
        baseline_context_cases=[
            c["name"] for c in observation["cases"] if c["name"].startswith("context-")
        ],
        fractions=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capacities(args.model_root, json.loads(args.observation.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
