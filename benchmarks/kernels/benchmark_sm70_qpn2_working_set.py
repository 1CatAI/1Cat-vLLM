# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Race QPN2 kernels over prepared weights from four consecutive TP4 layers.

Snapshots contain the actual runtime codes/scales and dispatch parameters,
including fused projections and padding. Activations are frozen synthetic
FP16 inputs. This measures a sequential projection working set, not a complete
verification round. The optional candidate library registers _qpn2_candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import regex as re
import torch


def save_model_snapshot(model: torch.nn.Module, directory: Path, rank: int) -> int:
    """Export prepared runtime weights once, before graph capture/profiling."""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
        _SM70_NVFP4_QPN2_CONFIGS,
    )

    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, module in model.named_modules():
        match = re.search(r"\.layers\.(\d+)\.", name)
        if (
            match is None
            or int(match[1]) >= 4
            or not getattr(module, "sm70_nvfp4_qpn2", False)
        ):
            continue
        k = module.input_size_per_partition
        n = module.sm70_nvfp4_qpn2_output_size
        gated = module.sm70_nvfp4_qpn2_gated_silu
        split_k, nacc = _SM70_NVFP4_QPN2_CONFIGS[k, n, gated]
        torch.save(
            {
                "name": name,
                "layer": int(match[1]),
                "rank": rank,
                "k": k,
                "n": n,
                "gated": gated,
                "split_k": split_k,
                "nacc": nacc,
                "global_scale": module.sm70_nvfp4_qpn2_global_scale,
                "codes": module.sm70_nvfp4_qpn2_codes.detach().cpu(),
                "scales": module.sm70_nvfp4_qpn2_scales.detach().cpu(),
            },
            directory / f"{count:02}-{name}.pt",
        )
        count += 1
    if count != 16:
        raise ValueError(f"Expected sixteen runtime QPN2 projections, got {count}")
    return count


def main() -> None:
    from vllm import _sm70_ops as ops

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--candidate-namespace", default="_qpn2_candidate")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.iterations < 1 or args.trials < 2:
        parser.error("positive iterations and at least two alternating trials required")
    if torch.cuda.get_device_capability() != (7, 0):
        raise ValueError("SM70 is required")
    if args.candidate_library:
        torch.ops.load_library(str(args.candidate_library))
    candidate_ops = getattr(torch.ops, args.candidate_namespace)
    torch.manual_seed(20260908)
    items, provenance = [], []
    for path in sorted(args.snapshots.glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        provenance.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **{k: v for k, v in item.items() if not isinstance(v, torch.Tensor)},
            }
        )
        item["codes"] = item["codes"].cuda()
        item["scales"] = item["scales"].cuda()
        item["candidate_scales"] = item["scales"]
        if args.candidate_library and hasattr(candidate_ops, "prepare_scales"):
            item["candidate_scales"] = candidate_ops.prepare_scales(
                item["scales"], item["global_scale"]
            )
        item["input"] = torch.randn(8, item["k"], device="cuda", dtype=torch.float16)
        width = item["n"] // 2 if item["gated"] else item["n"]
        item["outputs"] = [
            torch.empty(8, width, device="cuda", dtype=torch.float16) for _ in range(2)
        ]
        items.append(item)
    shapes = {(x["k"], x["n"]) for x in items}
    if (
        len(items) != 16
        or {x["layer"] for x in items} != {0, 1, 2, 3}
        or len({x["rank"] for x in items}) != 1
        or shapes
        != {(1536, 5120), (5120, 3584), (5120, 4128), (4352, 5120), (5120, 8704)}
    ):
        raise ValueError("Expected all five shapes in sixteen consecutive projections")

    def run(arm: int) -> None:
        for item in items:
            if arm and args.candidate_library:
                op = candidate_ops.gated if item["gated"] else candidate_ops.gemm
            else:
                op = (
                    ops.nvfp4_qpn2_gated_sm70_out
                    if item["gated"]
                    else ops.nvfp4_qpn2_gemm_sm70_out
                )
            op(
                item["outputs"][arm],
                item["input"],
                item["codes"],
                item["candidate_scales"] if arm else item["scales"],
                item["global_scale"],
                item["split_k"],
                item["nacc"],
            )

    graphs = []
    for arm in range(2):
        for _ in range(3):
            run(arm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(arm)
        graphs.append(graph)

    def check_outputs() -> None:
        exact = [
            torch.equal(
                x["outputs"][0].view(torch.uint8), x["outputs"][1].view(torch.uint8)
            )
            for x in items
        ]
        finite = all(
            torch.isfinite(x["outputs"][a]).all() for x in items for a in range(2)
        )
        if not finite or not all(exact):
            args.output.write_text(
                json.dumps({"bitwise_equal": exact, "finite": bool(finite)})
            )
            raise RuntimeError("Copy/load candidate changed numerical output")

    for graph in graphs:
        graph.replay()
    check_outputs()
    for _ in range(20):
        for graph in graphs:
            graph.replay()
    samples = [[], []]
    for trial in range(args.trials):
        for arm in range(2) if trial % 2 == 0 else range(1, -1, -1):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.iterations):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) / args.iterations)
    check_outputs()
    control_library = Path(sys.modules["vllm._C"].__file__).resolve()
    result = {
        "snapshots": provenance,
        "activation_seed": 20260908,
        "activation_kind": "synthetic_fp16_normal",
        "shape": "B1/q8 TP4 four consecutive layers",
        "weight_working_set_bytes": sum(
            x[k].numel() * x[k].element_size()
            for x in items
            for k in ("codes", "scales")
        ),
        "candidate_weight_working_set_bytes": sum(
            x[k].numel() * x[k].element_size()
            for x in items
            for k in ("codes", "candidate_scales")
        ),
        "candidate_scale_dtype": str(items[0]["candidate_scales"].dtype),
        "candidate_namespace": args.candidate_namespace
        if args.candidate_library
        else None,
        "candidate_library_sha256": hashlib.sha256(
            args.candidate_library.read_bytes()
        ).hexdigest()
        if args.candidate_library
        else None,
        "all_outputs_bitwise_equal": True,
        "parity_checked": "after first graph replay and after all timing trials",
        "control_library": str(control_library),
        "control_library_sha256": hashlib.sha256(
            control_library.read_bytes()
        ).hexdigest(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "iterations": args.iterations,
        "trials": args.trials,
        "working_set_samples_ms": samples,
        "working_set_medians_ms": [statistics.median(s) for s in samples],
        "paired_saved_ms": [a - b for a, b in zip(*samples)],
        "complete_round_performance": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "snapshots"}, indent=2))


if __name__ == "__main__":
    main()
