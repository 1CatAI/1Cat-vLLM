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
    parser.add_argument("--wired", action="store_true")
    parser.add_argument("--compiled", action="store_true")
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
        if not (gdn or attention) or kind not in (12, 16, 17, 18, 21, 22, 23, 29):
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
        if kind == 12:
            assert not gdn or projection.input_layout_restored
            weight = projection.codes
            partitions = (2,)
            stream_bytes = sum(
                t.numel() * t.element_size() for t in (weight, projection.stats)
            )
        else:
            weight = torch.from_numpy(_SOURCE_PACKERS[kind](raw)).cuda()
            partitions = (1, 2)
            stream_bytes = weight.numel() * weight.element_size()
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
            projection=projection,
        ):
            if kind == 12:
                torch.ops._C.gguf_canonical_linear_n64_sm70_out(
                    output, x, weight, projection.stats, partials, counters, 4, 32
                )
            else:
                torch.ops._C.gguf_small_output_sm70_out(
                    output, x, weight, partials, counters, kind, split, gdn
                )
            return output

        def canonical(x, projection=projection, gdn=gdn):
            if gdn and not projection.input_layout_restored:
                x = layout.input_to_gguf(x)
            return apply_prepared_gguf_projections(x, [projection])

        admission = None
        runtime_checks = []
        if args.wired:
            from vllm.model_executor.layers.quantization.gguf_small_output import (
                apply_small_output,
                prepare_small_output,
            )

            module = torch.nn.Module()
            layer_id = int(tensor.name.split(".")[1])
            module.prefix = f"model.layers.{layer_id}." + (
                "linear_attn.out_proj" if gdn else "self_attn.o_proj"
            )
            module.gguf_tm_projections = torch.nn.ModuleList([projection])
            admission = prepare_small_output(
                module, [(source, kind)], [projection], True, layout if gdn else None
            )
            if kind == 12:
                assert admission["reason"] == "measured_route_not_faster"
                continue
            assert admission["reason"] is None, admission
            partitions = (1,)

            def wired(x, module=module):
                return apply_small_output(module, x)

            if args.compiled:
                torch._dynamo.reset()
                wired = torch.compile(wired, dynamic=True, fullgraph=True)
                wired(torch.randn(512, 1536, dtype=torch.float16, device="cuda"))

            def candidate(x, split, wired=wired):
                assert split == 1
                return wired(x)

            for m in (512, 8, 1, 5, 16, 20, 32, 8):
                rows = torch.randn(m, 1536, dtype=torch.float16, device="cuda")
                actual = wired(rows)
                if m != 8:
                    torch.testing.assert_close(actual, canonical(rows), rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay = wired(rows)
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(replay, actual, rtol=0, atol=0)
                runtime_checks.append({"m": m, "bitwise_graph": True})

        checks = []
        for seed in (131, 132, 133):
            torch.manual_seed(seed)
            x = torch.randn(8, 1536, dtype=torch.float16, device="cuda")
            oracle = (x.float() @ reference.T).half()
            for split in partitions:
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
        for split in partitions:
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
                "admission": admission,
                "compiled": args.compiled,
                "runtime_checks": runtime_checks,
                "candidate_weight_stream_bytes": stream_bytes,
                "candidate_source": "canonical_u4" if kind == 12 else "original",
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
