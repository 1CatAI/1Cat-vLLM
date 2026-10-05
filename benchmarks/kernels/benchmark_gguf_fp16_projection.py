# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GGUF floating shards: installed operator math, graph and cold-L2 timing.

Run under the shared GPU reservation and the individual GPU lock. This script
uses installed package sources only, with no extension loading or source overlay.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
from pathlib import Path

import gguf
import torch

from vllm.model_executor.layers.quantization.gguf_turbomind import (
    GGUFPreparedProjection,
)
from vllm.model_executor.model_loader.gguf_adapters import get_gguf_adapter
from vllm.transformers_utils.gguf_config import load_gguf_config


def load_shards(path, layer, rank, tp):
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    adapter = get_gguf_adapter(load_gguf_config(path), tp_size=tp)
    shards = {}
    for role, raw_suffix in (("b", "beta"), ("a", "alpha")):
        raw_name = f"blk.{layer}.ssm_{raw_suffix}.weight"
        tensor = tensors[raw_name]
        name = f"model.layers.{layer}.linear_attn.in_proj_{role}.weight"
        loaded = dict(
            adapter.weights({raw_name: tensor}, {raw_name: name}, torch.float16)
        )
        payload = loaded[name.removesuffix(".weight") + ".qweight"]
        source_type = int(loaded[name.removesuffix(".weight") + ".qweight_type"])
        official = torch.from_numpy(
            gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        )
        official = adapter.restore(name, official)
        # Verify the normal adapter's conversion independently of its payload.
        torch.testing.assert_close(
            payload.view(torch.int16), official.half().view(torch.int16), rtol=0, atol=0
        )
        n = payload.shape[0] // tp
        converted = payload.narrow(0, rank * n, n).contiguous()
        reference = official.narrow(0, rank * n, n).contiguous()
        assert converted.shape == (12, 5120)
        assert source_type == 30
        assert torch.isfinite(converted).all() and torch.isfinite(reference).all()
        shards[role] = {
            "weight": converted,
            "official": reference,
            "source_type": source_type,
            "source_bytes": converted.numel() * converted.element_size(),
            "conversion": {
                "max_source_abs": float(reference.abs().max()),
                "max_half_abs": float(converted.abs().max()),
                "max_conversion_abs": float(
                    (reference - converted.float()).abs().max()
                ),
                "finite": True,
                "half_operand_bit_differences": 0,
            },
        }
    return shards


def error(actual, reference):
    difference = actual.float() - reference.float()
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            difference.norm() / reference.float().norm().clamp_min(1e-30)
        ),
        "half_bit_differences": int(
            (actual.view(torch.int16) != reference.view(torch.int16)).sum()
        ),
        "finite": bool(torch.isfinite(actual).all()),
    }


def clock():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def graph_time(call, flush):
    for _ in range(3):
        call()
    torch.accelerator.synchronize()
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(12)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for start, end in events:
            flush.fill_(0)
            start.record()
            call()
            end.record()
    samples = []
    for _ in range(7):
        graph.replay()
        torch.accelerator.synchronize()
        samples.extend(start.elapsed_time(end) * 1000 for start, end in events)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples": len(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--operator-sha", required=True)
    parser.add_argument("--native-base-sha", required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    shards = load_shards(args.model, args.layer, args.rank, args.tp)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.manual_seed(958)
    projections = {}
    for role, shard in shards.items():
        shard["weight"] = shard["weight"].cuda()
        shard["official"] = shard["official"].cuda()
        projections[role] = GGUFPreparedProjection(
            shard["weight"], 30, torch.float16, True, 8
        )
        assert projections[role].fp16_capabilities[0].reason is None

    from vllm.model_executor.layers.quantization import (
        gguf_fp16_projection as implementation,
    )

    source_path = Path(implementation.__file__).resolve()
    assert "site-packages" in source_path.parts
    assert "worktrees" not in source_path.parts
    record = {
        "operator_sha": args.operator_sha,
        "native_base_sha": args.native_base_sha,
        "native_scope": (
            "Reused declared normal native package; "
            "does not validate new disk-PLE operations"
        ),
        "package_version": importlib.metadata.version("1cat-vllm"),
        "installed_wrapper_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "layer": args.layer,
        "tp": args.tp,
        "rank": args.rank,
        "shape": [12, 5120],
        "source_type": "BF16",
        "activation_dtype": "float16",
        "accumulation_dtype": "float32",
        "conversion": {role: shard["conversion"] for role, shard in shards.items()},
        "admission": {
            role: projection.admission() for role, projection in projections.items()
        },
        "runtime_graphs": [],
        "timing": [],
    }
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(
        projections["b"], backend=backend, dynamic=True, fullgraph=True
    )
    for m in (512, 8, 1, 16, 32, 8):
        x = torch.randn(m, 5120, device="cuda", dtype=torch.float16)
        reference = (x.float() @ shards["b"]["official"].T).half()
        actual = compiled(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(x)
        graph.replay()
        torch.accelerator.synchronize()
        numerical = error(actual, reference)
        assert numerical["finite"] and numerical["relative_l2"] < 0.001
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            graph.replay()
            torch.accelerator.synchronize()
        kernel_names = [
            event.name
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        row_hit = any("_fp16_gemv_silu_ranges_kernel" in name for name in kernel_names)
        assert row_hit == (m == 8), (m, kernel_names)
        row = {
            "m": m,
            "math": numerical,
            "bitwise_graph_replay": True,
            "row_kernel_hit": row_hit,
            "kernels": kernel_names,
        }
        record["runtime_graphs"].append(row)
        print(json.dumps(row), flush=True)
    record["dynamo_graphs"] = len(graphs)
    for graph in graphs:
        nodes = [node for node in graph.graph.nodes if node.op == "call_function"]
        assert len(nodes) == 1 and "prepared_gguf_fp16_projection" in str(
            nodes[0].target
        )

    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for role, shard in shards.items():
        reference = (x.float() @ shard["official"].T).half()
        actual = projections[role](x)
        baseline = torch.mm(x, shard["weight"].T)
        record.setdefault("m8_math", {})[role] = {
            "row_vs_official": error(actual, reference),
            "mm_vs_official": error(baseline, reference),
            "row_vs_mm": error(actual, baseline),
        }
        assert record["m8_math"][role]["row_vs_official"]["relative_l2"] < 0.001
        assert torch.isfinite(actual).all()
        for _ in range(400):
            projections[role](x)
        torch.accelerator.synchronize()
        for label in ("torch.mm", "row_gemv", "row_gemv", "torch.mm"):
            call = (
                (lambda projection=projections[role]: projection(x))
                if label == "row_gemv"
                else (lambda weight=shard["weight"]: torch.mm(x, weight.T))
            )
            before = clock()
            timing = graph_time(call, flush)
            timing.update(
                {
                    "role": role,
                    "path": label,
                    "m": 8,
                    "source_bytes": shard["source_bytes"],
                    "weight_GBps": shard["source_bytes"] / timing["median_us"] / 1000,
                    "clock_before": before,
                    "clock_after": clock(),
                }
            )
            record["timing"].append(timing)
            print(json.dumps(timing), flush=True)
    record["complete"] = True
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
