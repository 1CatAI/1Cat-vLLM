# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare packaged IQ3_S gated projection with its canonical fallback.

Run under the shared GPU lock. Model files and results are external inputs.
No private extension is loaded. Cold L2 eviction is excluded from timing.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import gguf
import torch

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_iq3_gated import (
    apply_iq3_gated_pair,
    prepare_iq3_gated_pair,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    prepare_gguf_projections,
)


def clocks():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def cold_graph(call, flush):
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
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=6)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    assert torch.cuda.get_device_capability() == (7, 0)
    assert hasattr(torch.ops._C, "gguf_iq3_gated_sm70_out")
    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    raw = [tensors[f"blk.{args.layer}.ffn_{role}.weight"] for role in ("gate", "up")]
    assert all(int(tensor.tensor_type) == 21 for tensor in raw)
    payload = [tensor.data[:4352].copy() for tensor in raw]
    sources = [(torch.from_numpy(data).cuda(), 21) for data in payload]
    layer = torch.nn.Module()
    layer.prefix = f"model.layers.{args.layer}.mlp.gate_up_proj"
    projections = prepare_gguf_projections(sources, torch.float16, True, 8)
    layer.gguf_tm_projections = torch.nn.ModuleList(projections)
    admission = prepare_iq3_gated_pair(layer, sources, projections, True)
    assert admission["reason"] is None, admission
    references = [
        torch.from_numpy(
            gguf.quants.dequantize(data, gguf.GGMLQuantizationType.IQ3_S)
        ).cuda()
        for data in payload
    ]
    torch.manual_seed(131)
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)

    def canonical(rows):
        parts = [projection(rows) for projection in projections]
        pair = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        output = torch.empty(rows.shape[0], 4352, dtype=rows.dtype, device=rows.device)
        torch.ops._C.silu_and_mul(output, pair)
        return output

    def official(rows):
        gate, up = [(rows.float() @ weight.T).half() for weight in references]
        return (gate.float() / (1 + torch.exp(-gate.float()))).half() * up

    checks = []
    for m in (8, 1, 16, 32, 512, 8):
        rows = torch.randn(m, 5120, device="cuda", dtype=torch.float16)
        result = apply_iq3_gated_pair(layer, rows)
        reference = official(rows)
        error = result.float() - reference.float()
        relative = float(error.norm() / reference.float().norm())
        assert relative < 0.001 and bool(torch.isfinite(result).all())
        if m != 8:
            torch.testing.assert_close(result, canonical(rows), rtol=0, atol=0)
        checks.append(
            {
                "m": m,
                "relative_l2": relative,
                "max_abs": float(error.abs().max()),
                "route": "native_pair" if m == 8 else "canonical",
            }
        )
    flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    samples = []
    for label, call in (
        ("canonical", lambda: canonical(x)),
        ("native_pair", lambda: apply_iq3_gated_pair(layer, x)),
        ("native_pair", lambda: apply_iq3_gated_pair(layer, x)),
        ("canonical", lambda: canonical(x)),
    ):
        before = clocks()
        us = cold_graph(call, flush)
        samples.append(
            {
                "route": label,
                "median_us": us,
                "source_bytes": sum(data.nbytes for data in payload),
                "source_gbps": sum(data.nbytes for data in payload) / us / 1000,
                "clocks_before": before,
                "clocks_after": clocks(),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "complete": True,
                "packaged_operator": True,
                "model": args.model.name,
                "layer": args.layer,
                "source_sha256": [hashlib.sha256(data).hexdigest() for data in payload],
                "admission": admission,
                "canonical_output_sizes": [p.logical_output_size for p in projections],
                "checks": checks,
                "samples": samples,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "weight_decoder": (
                    "exact source-byte permutation; separate original scales"
                ),
                "accumulation": "fp32",
                "tp": 4,
                "rank": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
