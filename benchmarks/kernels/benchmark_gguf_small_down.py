# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real TP4 GGUF expert down projections with cold rotated weight banks."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import vllm._C as core
from benchmark_gguf_turbomind import elapsed, prepare_projection

import vllm
from vllm.model_executor.layers.quantization.gguf_lut_transcode import transcode_lut4
from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def error(actual, expected):
    delta = actual.float() - expected
    return dict(
        max_abs=delta.abs().max().item(),
        relative_l2=(delta.norm() / expected.norm()).item(),
        finite=torch.isfinite(actual).all().item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--weight-banks", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.weight_banks < 1:
        parser.error("--weight-banks must be positive")
    assert "site-packages" in vllm.__file__
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(str(args.gguf))
    name = f"blk.{args.layer}.ffn_down_exps.weight"
    tensor = next(t for t in reader.tensors if t.name == name)
    kind = int(tensor.tensor_type)
    if kind not in (20, 42):
        parser.error("Requires IQ4_NL or Q2_0 down projection")
    prepared = []
    for source in tensor.data:
        projection = (transcode_lut4 if kind == 20 else transcode_affine)(source, kind)
        local = projection.tp_slice(args.rank, 4, axis=1)
        prepared.append(prepare_projection(local))
    n, k = local.codes.shape
    experts = len(prepared)
    metadata = prepared[0][2].tolist()
    weights = torch.stack([p[0] for p in prepared])
    stats = torch.stack([p[1] for p in prepared])
    banks = []
    for i in range(args.weight_banks):
        w, s = (weights, stats) if i == 0 else (weights.clone(), stats.clone())
        wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(w, s, *metadata, experts)
        banks.append((w, s, wp, sp))
    results = []
    for m in (1, 5, 20):
        torch.manual_seed(20261004 + m)
        top_k = 10
        ids = torch.randn(m, experts, device="cuda").topk(top_k, dim=1).indices
        sorted_ids, _ = ids.flatten().sort(stable=True)
        offsets = torch.searchsorted(
            sorted_ids, torch.arange(experts + 1, device="cuda")
        ).int()
        x = torch.randn(m * top_k, k, device="cuda", dtype=torch.float16)
        old_out = torch.empty((m * top_k, n), device="cuda", dtype=x.dtype)
        new_out = torch.empty_like(old_out)

        def old(bank=banks[0], old_out=old_out, x=x, offsets=offsets):
            _, _, wp, sp = bank
            if kind == 20:
                torch.ops._C.gguf_lut4_grouped_gemm_sm70_out(
                    old_out, x, offsets, wp, sp, 0, experts, 32
                )
            else:
                torch.ops._C.gguf_affine_grouped_gemm_sm70_out(
                    old_out, x, offsets, wp, sp, 2, experts, 32
                )

        def new(bank=banks[0], new_out=new_out, x=x, offsets=offsets):
            torch.ops._C.gguf_small_grouped_vec_sm70_out(
                new_out, x, offsets, bank[2], bank[3], kind, experts, 32
            )

        def cold_old():
            for bank in banks:
                old(bank)

        def cold_new():
            for bank in banks:
                new(bank)

        old_us = elapsed(cold_old, 100) / len(banks)
        new_us = elapsed(cold_new, 100) / len(banks)
        old()
        new()
        route_ids = sorted_ids.cpu().tolist()
        active = sorted(set(route_ids))
        decoded = {
            expert: torch.from_numpy(
                dequantize(tensor.data[expert], kind)[
                    :, args.rank * k : (args.rank + 1) * k
                ].copy()
            ).cuda()
            for expert in active
        }
        reference = torch.stack([decoded[e] for e in route_ids])
        expected = torch.einsum("rk,rnk->rn", x.float(), reference)
        row = dict(
            m=m,
            routes=m * top_k,
            active_experts=len(active),
            canonical_us=old_us,
            vector_us=new_us,
            saved_us=old_us - new_us,
            canonical_error=error(old_out, expected),
            vector_error=error(new_out, expected),
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                version=vllm.__version__,
                core_sha256=hashlib.sha256(
                    Path(core.__file__).read_bytes()
                ).hexdigest(),
                tensor=name,
                source_type=kind,
                local_shape=[experts, n, k],
                weight_banks=len(banks),
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
