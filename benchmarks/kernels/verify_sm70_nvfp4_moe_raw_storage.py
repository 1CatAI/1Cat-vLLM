# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real TP4 checkpoint shards through the production MoE loader/apply.

This single-GPU diagnostic uses synthetic activations and all 512 experts in
each selected layer. It is not an end-to-end quality benchmark. Example:

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    benchmarks/kernels/verify_sm70_nvfp4_moe_raw_storage.py \
    --model /path/to/checkpoint --layers 0,23,47 --ranks 0,3 \
    --tokens 1,4,8,16,784 --out raw-storage.json
"""

import argparse
import gc
import json
import os
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from vllm import _sm70_ops as ops
from vllm import envs
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
    MoEActivation,
)


def error(a, b):
    d = a.float() - b.float()
    return dict(
        exact=torch.equal(a, b),
        max_abs=d.abs().max().item(),
        relative_l2=(d.norm() / b.float().norm().clamp_min(1e-12)).item(),
        mismatches=(a != b).sum().item(),
        finite=torch.isfinite(a).all().item(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--ranks", default="0")
    parser.add_argument("--tokens", default="1,3,4,7,8,16,32,128,784")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Audit grouped decode instead of raw-scale storage",
    )
    args = parser.parse_args()
    layers = tuple(map(int, args.layers.split(",")))
    ranks = tuple(map(int, args.ranks.split(",")))
    token_counts = tuple(map(int, args.tokens.split(",")))
    if any(rank not in range(4) for rank in ranks):
        parser.error("--ranks must select TP4 shards 0..3")
    if any(layer < 0 for layer in layers) or any(m <= 0 for m in token_counts):
        parser.error("layer indices must be nonnegative; token counts positive")
    os.environ["VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE"] = str(int(args.dynamic))
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This kernel audit requires SM70")
    if args.grouped and not ops.has_nvfp4_grouped_decode_dispatch():
        raise RuntimeError("Build the grouped-decode native operators first")
    if not args.grouped and not ops.has_nvfp4_qpn_raw_scale_dispatch():
        raise RuntimeError("Build the raw-scale native operators first")
    model = args.model
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    rows = []
    cfg = SimpleNamespace(
        num_experts=512,
        experts_per_token=10,
        hidden_dim=2560,
        intermediate_size_per_partition=160,
        tp_size=4,
        has_bias=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    method = object.__new__(ModelOptNvFp4SM70MoEMethod)
    method.moe = cfg
    for layer_no in layers:
        for rank in ranks:
            prefix = f"model.language_model.layers.{layer_no}.mlp.experts"
            with ExitStack() as stack:
                handles = {}

                def load(
                    expert, suffix, *, prefix=prefix, handles=handles, stack=stack
                ):
                    key = f"{prefix}.{expert}.{suffix}"
                    shard = index[key]
                    if shard not in handles:
                        handles[shard] = stack.enter_context(
                            safe_open(model / shard, framework="pt", device="cpu")
                        )
                    return handles[shard].get_tensor(key)

                tensors = {
                    k: []
                    for k in (
                        "w13_weight",
                        "w13_weight_scale",
                        "w13_weight_scale_2",
                        "w2_weight",
                        "w2_weight_scale",
                        "w2_weight_scale_2",
                    )
                }
                for e in range(512):
                    for suffix, dst in (
                        ("weight", "w13_weight"),
                        ("weight_scale", "w13_weight_scale"),
                    ):
                        tensors[dst].append(
                            torch.cat(
                                [
                                    load(e, f"{p}.{suffix}")[
                                        rank * 160 : (rank + 1) * 160
                                    ]
                                    for p in ("gate_proj", "up_proj")
                                ]
                            )
                        )
                    tensors["w13_weight_scale_2"].append(
                        torch.stack(
                            [
                                load(e, f"{p}.weight_scale_2").reshape(())
                                for p in ("gate_proj", "up_proj")
                            ]
                        )
                    )
                    tensors["w2_weight"].append(
                        load(e, "down_proj.weight")[:, rank * 80 : (rank + 1) * 80]
                    )
                    tensors["w2_weight_scale"].append(
                        load(e, "down_proj.weight_scale")[
                            :, rank * 10 : (rank + 1) * 10
                        ]
                    )
                    tensors["w2_weight_scale_2"].append(
                        load(e, "down_proj.weight_scale_2").reshape(())
                    )
                tensors = {
                    k: torch.stack(v).cuda().contiguous() for k, v in tensors.items()
                }

            def make(raw, tensors=tensors):
                os.environ["VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE"] = str(
                    int(raw and not args.grouped)
                )
                os.environ["VLLM_SM70_NVFP4_MOE_GROUPED_DECODE"] = str(
                    int(raw and args.grouped)
                )
                envs.disable_envs_cache()
                layer = SimpleNamespace(
                    moe_config=cfg,
                    local_num_experts=512,
                    global_num_experts=512,
                    activation=MoEActivation.SILU,
                    apply_router_weight_on_input=False,
                    expert_map=None,
                    swiglu_limit=None,
                    w13_input_scale=None,
                    w2_input_scale=None,
                    **tensors,
                )
                method.process_weights_after_loading(layer)
                return layer

            prepared, raw = make(False), make(True)
            del make, tensors
            scales = {}
            interleaved_w13 = raw.sm70_nvfp4_qwen38_fused_swiglu_prefill
            for stage, interleaved in (
                () if args.grouped else (("w13", interleaved_w13), ("w2", False))
            ):
                expected = getattr(prepared, f"{stage}_tm_scales")
                dest = torch.empty_like(expected)
                codes = getattr(raw, f"{stage}_raw_scale_codes")
                glob = getattr(raw, f"{stage}_raw_global_scales")
                for fast in (False, True):
                    ops.nvfp4_expand_raw_scales_sm70_out(
                        dest, codes, glob, interleaved, fast
                    )
                    scales[f"{stage}_{fast}"] = error(
                        dest, expected * (16384.0 if fast else 1.0)
                    )
            print(
                json.dumps(dict(layer=layer_no, rank=rank, scales=scales)), flush=True
            )
            for tokens in token_counts:
                torch.manual_seed(20260905 + tokens)
                x = torch.randn(tokens, 2560, device="cuda", dtype=torch.float16) * 0.1
                scores = torch.randn(tokens, 512, device="cuda")
                vals, ids = torch.topk(scores, 10, dim=-1)
                ids = ids.int().contiguous()
                weights = torch.softmax(vals, dim=-1)

                def apply(layer, tokens=tokens, x=x, weights=weights, ids=ids):
                    context = ForwardContext(
                        no_compile_layers={},
                        slot_mapping={},
                        attn_metadata={
                            "attn": SimpleNamespace(
                                max_query_len=1 if tokens <= 16 else tokens
                            )
                        },
                    )
                    with override_forward_context(context):
                        return method.apply(layer, x, weights, ids, None, None)

                # Run same allocation/shape twice to distinguish nondeterminism.
                a = apply(prepared).clone()
                aa = apply(prepared).clone()
                b = apply(raw).clone()
                bb = apply(raw).clone()
                if args.grouped:
                    torch.testing.assert_close(b, a, rtol=1e-3, atol=1e-5)
                row = dict(
                    layer=layer_no,
                    rank=rank,
                    tokens=tokens,
                    dynamic_decode=args.dynamic,
                    grouped_decode=args.grouped,
                    scales=scales,
                    **{
                        (
                            "grouped_vs_prepared" if args.grouped else "raw_vs_prepared"
                        ): error(b, a)
                    },
                    prepared_repeat=error(aa, a),
                    raw_repeat=error(bb, b),
                )
                if tokens <= 16:
                    graph = torch.cuda.CUDAGraph()
                    torch.accelerator.synchronize()
                    with torch.cuda.graph(graph):
                        bgraph = apply(raw)
                    for _ in range(3):
                        graph.replay()
                    row["graph_vs_eager"] = error(bgraph, b)
                    del graph
                rows.append(row)
                print(json.dumps(row), flush=True)
            del prepared, raw
            gc.collect()
            torch.accelerator.empty_cache()
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        checks = [row[k] for k in ("prepared_repeat", "raw_repeat")]
        if not args.grouped:
            checks.append(row["raw_vs_prepared"])
        elif row["tokens"] != 16:
            checks.append(row["grouped_vs_prepared"])
        checks.extend(row["scales"].values())
        if "graph_vs_eager" in row:
            checks.append(row["graph_vs_eager"])
        if any(not check["exact"] or not check["finite"] for check in checks):
            raise SystemExit(f"Numerical mismatch; inspect {args.out}")


if __name__ == "__main__":
    main()
