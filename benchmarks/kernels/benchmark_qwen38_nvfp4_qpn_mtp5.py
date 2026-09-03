#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Screen native-NVFP4 direct 50-route MoE kernels for Qwen3.8 MTP4.

This benchmark does not quantize activations or alter checkpoint weights. It
compares the existing TurboMind W4A16_NVFP4 compact grouped path against a
direct route kernel reading the same prepared NVFP4 weights with FP16 inputs
and FP32 accumulation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    _prepare_compact_slot_groups,
)
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    unpack_mxfp4_weight,
)
from vllm.triton_utils import tl, triton

MODEL = Path("/data/models/RadixArk/Qwen3.8-Flash-Next-NVFP4")
PREFIX = "model.language_model.layers.0.mlp.experts"
HIDDEN = 2560
INTERMEDIATE = 160
TOKENS = 5
TOP_K = 10
ROUTES = TOKENS * TOP_K
EXPERTS = 512
SOURCE_EXPERT_IDS = (0, 7, 55, 99, 120, 211, 256, 310, 401, 470)


@triton.jit
def _mtp_weighted_reduce_kernel(
    expert_output_ptr,
    topk_weights_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(0, TOP_K):
        route = token * TOP_K + slot
        values = tl.load(
            expert_output_ptr + route * HIDDEN + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(topk_weights_ptr + route)
        acc += values.to(tl.float32) * weight
    tl.store(output_ptr + token * HIDDEN + offsets, acc, mask=mask)


def mtp_weighted_reduce(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    tokens, top_k = topk_weights.shape
    hidden = expert_output.shape[1]
    block = 256
    _mtp_weighted_reduce_kernel[(tokens, triton.cdiv(hidden, block))](
        expert_output,
        topk_weights,
        output,
        HIDDEN=hidden,
        TOP_K=top_k,
        BLOCK=block,
        num_warps=4,
    )


def graph_us(fn: Callable[[], None], *, unroll: int = 32, replays: int = 100) -> float:
    for _ in range(8):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            fn()
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (unroll * replays))
    return statistics.median(samples)


def cold_us(
    fn: Callable[[], None], l2_flush: torch.Tensor, *, trials: int = 101
) -> float:
    for _ in range(8):
        l2_flush.zero_()
        fn()
    torch.accelerator.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(trials):
        l2_flush.zero_()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = actual.float() - expected.float()
    return {
        "bitwise_equal": torch.equal(actual, expected),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.library is not None:
        torch.ops.load_library(str(args.library))
    if not sm70_ops.has_nvfp4_qpn_mtp5_dispatch():
        raise RuntimeError(
            "benchmark requires nvfp4_moe_qpn_mtp5_sm70_out in the "
            "production _C or _C_qwen38 namespace"
        )
    op = sm70_ops.nvfp4_moe_qpn_mtp5_sm70_out
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("benchmark requires exact SM70")

    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def load(expert: int, suffix: str) -> torch.Tensor:
        key = f"{PREFIX}.{expert}.{suffix}"
        with safe_open(
            args.model / weight_map[key], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(key).cuda().contiguous()

    def prepare_expert(expert: int):
        gate = load(expert, "gate_proj.weight")[:INTERMEDIATE]
        up = load(expert, "up_proj.weight")[:INTERMEDIATE]
        gate_scale = (
            load(expert, "gate_proj.weight_scale")[:INTERMEDIATE].float()
            * load(expert, "gate_proj.weight_scale_2").float()
        )
        up_scale = (
            load(expert, "up_proj.weight_scale")[:INTERMEDIATE].float()
            * load(expert, "up_proj.weight_scale_2").float()
        )
        w13 = sm70_ops.nvfp4_sm70_prepare(
            unpack_mxfp4_weight(torch.cat((gate, up)).contiguous()),
            torch.cat((gate_scale, up_scale)).half().t().contiguous(),
            16,
        )
        down = load(expert, "down_proj.weight")[:, : INTERMEDIATE // 2]
        down_scale = (
            load(expert, "down_proj.weight_scale")[:, : INTERMEDIATE // 16].float()
            * load(expert, "down_proj.weight_scale_2").float()
        )
        w2 = sm70_ops.nvfp4_sm70_prepare(
            unpack_mxfp4_weight(down),
            down_scale.half().t().contiguous(),
            16,
        )
        return w13, w2

    def stack_prepared(items, copies: int):
        weights = torch.stack([item[0] for item in items]).repeat(copies, 1, 1)
        scales = torch.stack([item[1] for item in items]).repeat(copies, 1, 1)
        meta = items[0][2]
        ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            weights,
            scales,
            int(meta[0].item()),
            int(meta[1].item()),
            weights.shape[0],
        )
        return weights, scales, ptrs

    def pad_experts(tensor: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            (EXPERTS, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        )
        indices = torch.arange(ROUTES, dtype=torch.int64, device=tensor.device)
        out.index_copy_(0, indices, tensor)
        return out

    torch.manual_seed(args.seed)
    prepared = [prepare_expert(expert) for expert in SOURCE_EXPERT_IDS]
    w13, s13, p13 = stack_prepared([item[0] for item in prepared], TOKENS)
    w2, s2, p2 = stack_prepared([item[1] for item in prepared], TOKENS)
    qpn_w13, qpn_s13 = pad_experts(w13), pad_experts(s13)
    qpn_w2, qpn_s2 = pad_experts(w2), pad_experts(s2)

    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.float16).mul_(0.1)
    expanded = x.repeat_interleave(TOP_K, dim=0)
    offsets = torch.arange(ROUTES + 1, device="cuda", dtype=torch.int32)
    l2_flush = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    result: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(),
        "seed": args.seed,
        "precision_contract": {
            "checkpoint_weights": "native W4A16_NVFP4 (unchanged)",
            "activations": "FP16 (no activation quantization)",
            "accumulation": "FP32 HMMA",
        },
        "shapes": {
            "tokens": TOKENS,
            "top_k": TOP_K,
            "routes": ROUTES,
            "w13": [ROUTES, HIDDEN, 2 * INTERMEDIATE],
            "w2": [ROUTES, INTERMEDIATE, HIDDEN],
        },
        "patterns": {},
    }

    patterns = {
        "overlap_10_experts": torch.arange(TOP_K, device="cuda").repeat(TOKENS),
        "distinct_50_experts": torch.arange(ROUTES, device="cuda"),
    }
    for pattern_name, token_order_ids_i64 in patterns.items():
        token_order_ids = token_order_ids_i64.to(torch.int32).contiguous()
        sort_index = torch.argsort(token_order_ids_i64, stable=True)
        inverse_index = torch.empty_like(sort_index)
        inverse_index[sort_index] = torch.arange(ROUTES, device="cuda")
        sorted_input = expanded.index_select(0, sort_index).contiguous()
        sorted_ids = token_order_ids.index_select(0, sort_index).contiguous()

        baseline_w13 = torch.empty(
            ROUTES, 2 * INTERMEDIATE, device="cuda", dtype=torch.float16
        )
        candidate_w13 = torch.empty_like(baseline_w13)
        baseline_intermediate = torch.empty(
            ROUTES, INTERMEDIATE, device="cuda", dtype=torch.float16
        )
        candidate_intermediate = torch.empty_like(baseline_intermediate)
        baseline_w2 = torch.empty(ROUTES, HIDDEN, device="cuda", dtype=torch.float16)
        candidate_w2 = torch.empty_like(baseline_w2)

        def baseline_stage13() -> None:
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                baseline_w13,
                sorted_input,
                offsets,
                sorted_ids,
                p13[0],
                p13[1],
                ROUTES,
                HIDDEN,
                2 * INTERMEDIATE,
                16,
            )

        baseline_stage13()
        torch.ops._C.silu_and_mul(baseline_intermediate, baseline_w13)

        def baseline_stage2() -> None:
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                baseline_w2,
                baseline_intermediate,
                offsets,
                sorted_ids,
                p2[0],
                p2[1],
                ROUTES,
                INTERMEDIATE,
                HIDDEN,
                16,
            )

        baseline_stage2()
        torch.accelerator.synchronize()
        pattern: dict[str, object] = {
            "unique_experts": int(token_order_ids.unique().numel()),
            "baseline_us": {
                "w13_warm": graph_us(baseline_stage13),
                "w13_cold": cold_us(baseline_stage13, l2_flush),
                "w2_warm": graph_us(baseline_stage2),
                "w2_cold": cold_us(baseline_stage2, l2_flush),
            },
            "candidate_w13": [],
            "candidate_w2": [],
        }

        for split_k in (4, 5, 8, 10, 16, 20, 32):

            def candidate_stage13(split_k: int = split_k) -> None:
                op(
                    candidate_w13,
                    x,
                    qpn_w13,
                    qpn_s13,
                    token_order_ids,
                    True,
                    split_k,
                )

            candidate_stage13()
            torch.accelerator.synchronize()
            pattern["candidate_w13"].append(
                {
                    "split_k": split_k,
                    "warm_us": graph_us(candidate_stage13),
                    "cold_us": cold_us(candidate_stage13, l2_flush),
                    **error(
                        candidate_w13,
                        baseline_w13.index_select(0, inverse_index),
                    ),
                }
            )

        expanded_candidate_w13 = torch.empty_like(candidate_w13)
        op(
            candidate_w13,
            x,
            qpn_w13,
            qpn_s13,
            token_order_ids,
            True,
            4,
        )
        op(
            expanded_candidate_w13,
            expanded,
            qpn_w13,
            qpn_s13,
            token_order_ids,
            False,
            4,
        )
        torch.accelerator.synchronize()
        pattern["broadcast_vs_expanded_w13_split4"] = error(
            candidate_w13, expanded_candidate_w13
        )

        for split_k in (1, 2, 5, 10):

            def candidate_stage2(split_k: int = split_k) -> None:
                op(
                    candidate_w2,
                    candidate_intermediate,
                    qpn_w2,
                    qpn_s2,
                    token_order_ids,
                    False,
                    split_k,
                )

            candidate_intermediate.copy_(
                baseline_intermediate.index_select(0, inverse_index)
            )
            candidate_stage2()
            torch.accelerator.synchronize()
            pattern["candidate_w2"].append(
                {
                    "split_k": split_k,
                    "warm_us": graph_us(candidate_stage2),
                    "cold_us": cold_us(candidate_stage2, l2_flush),
                    **error(
                        candidate_w2,
                        baseline_w2.index_select(0, inverse_index),
                    ),
                }
            )

        topk_ids = token_order_ids.view(TOKENS, TOP_K)
        topk_weights = torch.softmax(
            torch.randn(TOKENS, TOP_K, device="cuda", dtype=torch.float32),
            dim=-1,
        )
        topk_ids_buffer = torch.empty_like(topk_ids)
        token_expert_indices = torch.arange(
            ROUTES, device="cuda", dtype=torch.int32
        ).view(TOKENS, TOP_K)
        permuted_input = torch.empty_like(expanded)
        full_offsets64 = torch.empty(EXPERTS + 1, device="cuda", dtype=torch.int64)
        full_offsets = torch.empty(EXPERTS + 1, device="cuda", dtype=torch.int32)
        inv_permuted_idx = torch.empty_like(topk_ids)
        permuted_idx = torch.empty(ROUTES, device="cuda", dtype=torch.int32)
        permuted_expert_ids = torch.empty_like(permuted_idx)
        sorted_row_idx = torch.empty_like(permuted_idx)
        topk_ids_for_sort = torch.empty_like(permuted_idx)
        sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            ROUTES, EXPERTS
        )
        sort_workspace = torch.empty(
            sort_workspace_size, device="cuda", dtype=torch.int8
        )
        compact_offsets = torch.empty(ROUTES + 1, device="cuda", dtype=torch.int32)
        active_expert_ids = torch.empty_like(permuted_idx)
        baseline_output = torch.empty(
            TOKENS, HIDDEN, device="cuda", dtype=torch.float16
        )
        candidate_output = torch.empty_like(baseline_output)

        def baseline_full_path() -> None:
            baseline_output.zero_()
            topk_ids_buffer.copy_(topk_ids)
            permuted_idx.fill_(ROUTES)
            torch.ops._moe_C.moe_permute_with_scratch(
                x,
                topk_ids_buffer,
                token_expert_indices,
                None,
                EXPERTS,
                EXPERTS,
                TOP_K,
                permuted_input,
                full_offsets64,
                inv_permuted_idx,
                permuted_idx,
                sort_workspace,
                permuted_expert_ids,
                sorted_row_idx,
                topk_ids_for_sort,
            )
            full_offsets.copy_(full_offsets64)
            _prepare_compact_slot_groups(
                permuted_expert_ids,
                compact_offsets,
                active_expert_ids,
            )
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                baseline_w13,
                permuted_input,
                compact_offsets,
                active_expert_ids,
                p13[0],
                p13[1],
                ROUTES,
                HIDDEN,
                2 * INTERMEDIATE,
                16,
            )
            torch.ops._C.silu_and_mul(baseline_intermediate, baseline_w13)
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                baseline_w2,
                baseline_intermediate,
                compact_offsets,
                active_expert_ids,
                p2[0],
                p2[1],
                ROUTES,
                INTERMEDIATE,
                HIDDEN,
                16,
            )
            torch.ops._moe_C.moe_unpermute(
                baseline_w2,
                topk_weights,
                inv_permuted_idx,
                full_offsets64,
                TOP_K,
                baseline_output,
            )

        def candidate_full_path() -> None:
            op(
                candidate_w13,
                x,
                qpn_w13,
                qpn_s13,
                token_order_ids,
                True,
                4,
            )
            torch.ops._C.silu_and_mul(candidate_intermediate, candidate_w13)
            op(
                candidate_w2,
                candidate_intermediate,
                qpn_w2,
                qpn_s2,
                token_order_ids,
                False,
                1,
            )
            mtp_weighted_reduce(candidate_w2, topk_weights, candidate_output)

        baseline_full_path()
        candidate_full_path()
        torch.accelerator.synchronize()
        full_error = error(candidate_output, baseline_output)
        baseline_warm = graph_us(baseline_full_path)
        candidate_warm = graph_us(candidate_full_path)
        baseline_cold = cold_us(baseline_full_path, l2_flush)
        candidate_cold = cold_us(candidate_full_path, l2_flush)
        pattern["full_path"] = {
            "baseline_warm_us": baseline_warm,
            "candidate_warm_us": candidate_warm,
            "baseline_cold_us": baseline_cold,
            "candidate_cold_us": candidate_cold,
            "warm_saving_us_per_layer": baseline_warm - candidate_warm,
            "cold_saving_us_per_layer": baseline_cold - candidate_cold,
            "projected_warm_saving_ms_per_48_layers": (baseline_warm - candidate_warm)
            * 48
            / 1000,
            "projected_cold_saving_ms_per_48_layers": (baseline_cold - candidate_cold)
            * 48
            / 1000,
            **full_error,
        }

        result["patterns"][pattern_name] = pattern

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
