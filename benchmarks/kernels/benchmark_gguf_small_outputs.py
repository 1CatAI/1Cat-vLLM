# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen shared-A output projections, folding GDN head order into A loads."""

import argparse
import json
from pathlib import Path

import gguf
import torch
from benchmark_gguf_iq3_gated import clocks, cold_graph

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.model_executor.layers.quantization.gguf_native_pair import _SOURCE_PACKERS
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    assert 0 <= args.rank < 4
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tensors = gguf.GGUFReader(args.model).tensors
    report = {"model": args.model.name, "rank": args.rank, "cases": []}
    layout = GGUFHeadTilingLayout(3, 128)
    flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    seen = set()
    for tensor in tensors:
        gdn = tensor.name.endswith(".ssm_out.weight")
        attention = tensor.name.endswith(".attn_output.weight")
        kind = int(tensor.tensor_type)
        if not (gdn or attention) or kind not in (16, 17, 18, 21, 22, 23, 29):
            continue
        role = "gdn_out" if gdn else "attention_o"
        if (role, kind) in seen:
            continue
        seen.add((role, kind))
        block, size = quant_size(kind)
        full = torch.from_numpy(tensor.data.copy())
        if gdn:
            raw = (
                layout.shard_weight(
                    full,
                    dim=1,
                    logical_size=6144,
                    block_size=block,
                    tp_rank=args.rank,
                    tp_size=4,
                )
                .numpy()
                .copy()
            )
        else:
            columns = 1536 // block * size
            raw = (
                full[:, args.rank * columns : (args.rank + 1) * columns].numpy().copy()
            )
        source = torch.from_numpy(raw).cuda()
        projection = prepare_gguf_projections(
            [(source, kind)],
            torch.float16,
            True,
            8,
            input_layout=layout if gdn else None,
        )[0]
        reference = torch.from_numpy(gguf.quants.dequantize(raw, tensor.tensor_type))
        if gdn:
            reference = layout.weight_to_vllm(reference, dim=1)
        reference = reference.cuda()
        weight = torch.from_numpy(_SOURCE_PACKERS[kind](raw)).cuda()
        partials = torch.empty(80, 2, 512, dtype=torch.float32, device="cuda")
        counters = torch.zeros(80, dtype=torch.int32, device="cuda")
        output = torch.empty(8, 5120, dtype=torch.float16, device="cuda")

        def candidate(
            x,
            split,
            weight=weight,
            partials=partials,
            counters=counters,
            output=output,
            kind=kind,
            gdn=gdn,
        ):
            torch.ops._C.gguf_small_output_sm70_out(
                output, x, weight, partials, counters, kind, split, gdn
            )
            return output

        def canonical(x, projection=projection, gdn=gdn):
            if gdn and not projection.input_layout_restored:
                x = layout.input_to_gguf(x)
            return apply_prepared_gguf_projections(x, [projection])

        checks = []
        for seed in (131, 132, 133):
            torch.manual_seed(seed)
            x = torch.randn(8, 1536, dtype=torch.float16, device="cuda")
            oracle = (x.float() @ reference.T).half()
            for split in (1, 2):
                actual = candidate(x, split).clone()
                difference = actual.float() - oracle.float()
                relative = float(difference.norm() / oracle.float().norm())
                assert bool(torch.isfinite(actual).all()) and relative < 0.001
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay = candidate(x, split)
                for _ in range(1000):
                    graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(replay, actual, rtol=0, atol=0)
                assert torch.count_nonzero(counters).item() == 0
                checks.append(
                    {
                        "seed": seed,
                        "split": split,
                        "relative_l2": relative,
                        "max_abs": float(difference.abs().max()),
                    }
                )
        timings = []
        for split in (1, 2):
            for label in ("canonical", "candidate", "candidate", "canonical"):
                call = (
                    (lambda x=x: canonical(x))
                    if label == "canonical"
                    else (lambda x=x, split=split: candidate(x, split))
                )
                time = cold_graph(call, flush)
                timings.append(
                    {"route": label, "split": split, "us": time, "clock": clocks()}
                )
        report["cases"].append(
            {
                "tensor": tensor.name,
                "role": role,
                "source_type": kind,
                "m": 8,
                "n": 5120,
                "k": 1536,
                "source_bytes": raw.nbytes,
                "canonical_input_layout_restored": projection.input_layout_restored,
                "checks": checks,
                "timings": timings,
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
