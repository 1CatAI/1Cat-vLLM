# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Compare the production TP4 and TP-local EP4 FlashNext NVFP4 MoE chains.

The four TP ranks receive the same token rows. The control keeps all 512
experts on every rank with a 160-wide intermediate shard. EP4 keeps 128 full
640-wide experts on each rank, maps global routes to their owning rank, and
uses the same final four-rank all-reduce. Timings include routing, W13,
SwiGLU, W2, weighted unpermutation, and NCCL all-reduce.

This is a one-layer screening benchmark. It does not replace end-to-end model
or quality validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    unpack_mxfp4_weight,
)

HIDDEN = 2560
INTERMEDIATE = 640
GLOBAL_EXPERTS = 512
LOCAL_EXPERTS = 128
TOP_K = 10
GROUP_SIZE = 16


@dataclass
class PreparedWeights:
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    p13w: torch.Tensor
    p13s: torch.Tensor
    p2w: torch.Tensor
    p2s: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_contract() -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an SM70 GPU.")
    for namespace, name in (
        (torch.ops._C, "nvfp4_sm70_prepare"),
        (torch.ops._C, "nvfp4_moe_dense_stage_sm70_out"),
        (torch.ops._C, "awq_moe_build_strided_ptrs"),
        (torch.ops._C, "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out"),
        (torch.ops._C, "nvfp4_moe_qpn_w2_reduce_sm70_out"),
        (torch.ops._C, "nvfp4_moe_fused_swiglu_stage_sm70_out"),
        (torch.ops._C, "nvfp4_qwen38_ep4_permute_sm70_out"),
        (torch.ops._C, "nvfp4_qwen38_ep4_combine_sm70_out"),
        (torch.ops._C, "nvfp4_grouped_w13_sm70_out"),
        (torch.ops._C, "nvfp4_grouped_w2_sm70_out"),
        (torch.ops._C, "nvfp4_grouped_ep4_w13_sm70_out"),
        (torch.ops._C, "nvfp4_grouped_ep4_w2_sm70_out"),
        (torch.ops._moe_C, "moe_permute_with_scratch"),
        (torch.ops._moe_C, "moe_unpermute"),
    ):
        if not hasattr(namespace, name):
            raise RuntimeError(f"Required operator is missing: {namespace}.{name}")


def _prepare(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        sm70_ops.nvfp4_sm70_prepare(
            unpack_mxfp4_weight(packed.cuda()),
            scales.half().t().contiguous().cuda(),
            GROUP_SIZE,
            interleave_gated_silu=interleaved,
        )
    )


def _load_weights(
    model: Path, layer: int, rank: int
) -> tuple[PreparedWeights, PreparedWeights]:
    index_path = model / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    tp: list[list[torch.Tensor]] = [[], [], [], []]
    ep: list[list[torch.Tensor]] = [[], [], [], []]
    tp_meta: tuple[torch.Tensor, torch.Tensor] | None = None
    ep_meta: tuple[torch.Tensor, torch.Tensor] | None = None
    ep_start = rank * LOCAL_EXPERTS
    ep_stop = ep_start + LOCAL_EXPERTS

    with ExitStack() as stack:
        handles: dict[str, object] = {}

        def get(expert: int, projection: str, suffix: str) -> torch.Tensor:
            key = (
                f"model.language_model.layers.{layer}.mlp.experts."
                f"{expert}.{projection}.{suffix}"
            )
            shard = weight_map[key]
            if shard not in handles:
                handles[shard] = stack.enter_context(
                    safe_open(model / shard, framework="pt", device="cpu")
                )
            return handles[shard].get_tensor(key)  # type: ignore[union-attr]

        for expert in range(GLOBAL_EXPERTS):
            gate_w = get(expert, "gate_proj", "weight")
            up_w = get(expert, "up_proj", "weight")
            gate_scale = get(expert, "gate_proj", "weight_scale").float()
            up_scale = get(expert, "up_proj", "weight_scale").float()
            gate_global = get(expert, "gate_proj", "weight_scale_2").float()
            up_global = get(expert, "up_proj", "weight_scale_2").float()
            down_w = get(expert, "down_proj", "weight")
            down_scale = get(expert, "down_proj", "weight_scale").float()
            down_global = get(expert, "down_proj", "weight_scale_2").float()

            lo = rank * (INTERMEDIATE // 4)
            hi = lo + INTERMEDIATE // 4
            tp_w13, tp_s13, meta13 = _prepare(
                torch.cat((gate_w[lo:hi], up_w[lo:hi])),
                torch.cat(
                    (gate_scale[lo:hi] * gate_global, up_scale[lo:hi] * up_global)
                ),
                interleaved=True,
            )
            tp_w2, tp_s2, meta2 = _prepare(
                down_w[:, lo // 2 : hi // 2].contiguous(),
                down_scale[:, lo // GROUP_SIZE : hi // GROUP_SIZE] * down_global,
            )
            for destination, value in zip(tp, (tp_w13, tp_s13, tp_w2, tp_s2)):
                destination.append(value)
            if tp_meta is None:
                tp_meta = (meta13, meta2)

            if ep_start <= expert < ep_stop:
                ep_w13, ep_s13, ep_meta13 = _prepare(
                    torch.cat((gate_w, up_w)),
                    torch.cat((gate_scale * gate_global, up_scale * up_global)),
                    interleaved=True,
                )
                ep_w2, ep_s2, ep_meta2 = _prepare(
                    down_w,
                    down_scale * down_global,
                )
                for destination, value in zip(ep, (ep_w13, ep_s13, ep_w2, ep_s2)):
                    destination.append(value)
                if ep_meta is None:
                    ep_meta = (ep_meta13, ep_meta2)

    if tp_meta is None or ep_meta is None:
        raise RuntimeError("Weight preparation produced no metadata.")

    def finish(
        values: list[list[torch.Tensor]],
        metadata: tuple[torch.Tensor, torch.Tensor],
        experts: int,
    ) -> PreparedWeights:
        w13, s13, w2, s2 = (torch.stack(items) for items in values)
        meta13, meta2 = metadata
        p13w, p13s = sm70_ops.awq_moe_build_strided_ptrs(
            w13,
            s13,
            int(meta13[0].item()),
            int(meta13[1].item()),
            experts,
        )
        p2w, p2s = sm70_ops.awq_moe_build_strided_ptrs(
            w2,
            s2,
            int(meta2[0].item()),
            int(meta2[1].item()),
            experts,
        )
        return PreparedWeights(w13, s13, w2, s2, p13w, p13s, p2w, p2s)

    return (
        finish(tp, tp_meta, GLOBAL_EXPERTS),
        finish(ep, ep_meta, LOCAL_EXPERTS),
    )


def _time_distributed(
    fn: Callable[[], None],
    *,
    rank: int,
    repeats: int,
    samples: int,
    warmup: int = 5,
) -> tuple[float, list[float], list[list[float]]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    max_samples: list[float] = []
    rank_samples: list[list[float]] = []
    for _ in range(samples):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        local_us = start.elapsed_time(end) * 1000.0 / repeats
        gathered: list[float | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_us)
        per_rank = [float(value) for value in gathered if value is not None]
        rank_samples.append(per_rank)
        max_samples.append(max(per_rank))
    median = statistics.median(max_samples)
    if rank == 0:
        print(
            json.dumps({"median_us": median, "samples_us": max_samples}),
            flush=True,
        )
    return median, max_samples, rank_samples


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = actual.float() - expected.float()
    expected_f = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1), expected_f.reshape(1, -1)
    )
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float(
            (delta.norm() / expected_f.norm().clamp_min(1e-12)).item()
        ),
        "cosine": float(cosine.item()),
    }


def _load_routes(path: Path, tokens: int) -> torch.Tensor:
    value = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(value, dict):
        value = value["tensor"]
    if value.ndim != 2 or value.shape[1] != TOP_K or value.shape[0] < tokens:
        raise ValueError(
            f"Expected at least [{tokens},{TOP_K}] routes, got {value.shape}"
        )
    return value[:tokens].to(torch.int32).contiguous().cuda()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", default="4,8,16")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--ep4-blocks", default="80,160,240,320")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"This benchmark requires four ranks, got {world_size}.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    _require_contract()
    torch.manual_seed(20260907)

    if rank == 0:
        print("Preparing TP4 and EP4 checkpoint weights...", flush=True)
    tp, ep = _load_weights(args.model, args.layer, rank)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("Checkpoint weights prepared.", flush=True)

    cases: list[dict[str, object]] = []
    for tokens in map(int, args.tokens.split(",")):
        if tokens not in (4, 8, 16):
            parser.error("Production QPN control supports token counts 4, 8, and 16.")
        slots = tokens * TOP_K
        x = torch.randn(tokens, HIDDEN, dtype=torch.float16, device="cuda") * 0.02
        topk_ids = _load_routes(args.routes, tokens)
        topk_weights = torch.softmax(
            torch.randn(tokens, TOP_K, dtype=torch.float32, device="cuda"), dim=-1
        )
        for tensor in (x, topk_ids, topk_weights):
            dist.broadcast(tensor, src=0)

        tp_intermediate = torch.empty(
            slots, INTERMEDIATE // 4, dtype=torch.float16, device="cuda"
        )
        tp_routed = torch.empty(slots, HIDDEN, dtype=torch.float16, device="cuda")
        tp_output = torch.empty(tokens, HIDDEN, dtype=torch.float16, device="cuda")
        tp_grouped_rows = torch.empty(160, 8, dtype=torch.int32, device="cuda")
        tp_grouped_experts = torch.empty(160, dtype=torch.int32, device="cuda")
        tp_grouped_sizes = torch.empty(160, dtype=torch.int32, device="cuda")
        tp_grouped_total = torch.empty(1, dtype=torch.int32, device="cuda")

        expert_map = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.int32, device="cuda")
        ep_start = rank * LOCAL_EXPERTS
        expert_map[ep_start : ep_start + LOCAL_EXPERTS] = torch.arange(
            LOCAL_EXPERTS, dtype=torch.int32, device="cuda"
        )
        topk_ids_buffer = torch.empty_like(topk_ids)
        token_expert_indices = torch.arange(
            slots, dtype=torch.int32, device="cuda"
        ).view(tokens, TOP_K)
        permuted_input = torch.empty(slots, HIDDEN, dtype=torch.float16, device="cuda")
        expert_offsets64 = torch.empty(
            LOCAL_EXPERTS + 1, dtype=torch.int64, device="cuda"
        )
        expert_offsets = torch.empty(
            LOCAL_EXPERTS + 1, dtype=torch.int32, device="cuda"
        )
        inv_permuted_idx = torch.empty_like(topk_ids)
        permuted_idx = torch.empty(slots, dtype=torch.int32, device="cuda")
        permuted_expert_ids = torch.empty(slots, dtype=torch.int32, device="cuda")
        sorted_row_idx = torch.empty(slots, dtype=torch.int32, device="cuda")
        topk_ids_for_sort = torch.empty(slots, dtype=torch.int32, device="cuda")
        workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            slots, GLOBAL_EXPERTS
        )
        sort_workspace = torch.empty(workspace_size, dtype=torch.int8, device="cuda")
        dense_expert_ids = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device="cuda")
        ep_gate_up = torch.empty(
            slots, 2 * INTERMEDIATE, dtype=torch.float16, device="cuda"
        )
        ep_intermediate = torch.empty(
            slots, INTERMEDIATE, dtype=torch.float16, device="cuda"
        )
        ep_sorted_output = torch.empty(
            slots, HIDDEN, dtype=torch.float16, device="cuda"
        )
        ep_output = torch.empty_like(tp_output)
        ep_grouped_intermediate = torch.empty_like(ep_intermediate)
        ep_grouped_routed = torch.empty_like(ep_sorted_output)
        ep_grouped_output = torch.empty_like(tp_output)
        ep_grouped_rows = torch.empty(160, 8, dtype=torch.int32, device="cuda")
        ep_grouped_experts = torch.empty(160, dtype=torch.int32, device="cuda")
        ep_grouped_sizes = torch.empty(160, dtype=torch.int32, device="cuda")
        ep_grouped_total = torch.empty(1, dtype=torch.int32, device="cuda")

        def tp_qpn_chain() -> None:
            sm70_ops.nvfp4_moe_qpn_w13_swiglu_batch_sm70_out(
                tp_intermediate,
                x,
                tp.w13,
                tp.s13,
                topk_ids.view(-1),
                True,
            )
            sm70_ops.nvfp4_moe_qpn_w2_reduce_sm70_out(
                tp_output,
                tp_intermediate,
                tp.w2,
                tp.s2,
                topk_ids.view(-1),
                topk_weights,
            )
            dist.all_reduce(tp_output)

        def tp_grouped_chain() -> None:
            sm70_ops.nvfp4_grouped_w13_sm70_out(
                tp_intermediate,
                x,
                tp.w13,
                tp.s13,
                topk_ids.view(-1),
                tp_grouped_rows,
                tp_grouped_experts,
                tp_grouped_sizes,
                tp_grouped_total,
                4 if tokens == 8 else 8,
                True,
            )
            sm70_ops.nvfp4_grouped_w2_sm70_out(
                tp_output,
                tp_routed,
                tp_intermediate,
                tp.w2,
                tp.s2,
                topk_weights,
                tp_grouped_rows,
                tp_grouped_experts,
                tp_grouped_sizes,
                tp_grouped_total,
            )
            dist.all_reduce(tp_output)

        tp_chain = tp_grouped_chain if tokens in (8, 16) else tp_qpn_chain

        def ep_legacy_permute() -> None:
            ep_output.zero_()
            topk_ids_buffer.copy_(topk_ids)
            permuted_idx.fill_(slots)
            torch.ops._moe_C.moe_permute_with_scratch(
                x,
                topk_ids_buffer,
                token_expert_indices,
                expert_map,
                GLOBAL_EXPERTS,
                LOCAL_EXPERTS,
                TOP_K,
                permuted_input,
                expert_offsets64,
                inv_permuted_idx,
                permuted_idx,
                sort_workspace,
                permuted_expert_ids,
                sorted_row_idx,
                topk_ids_for_sort,
            )
            expert_offsets.copy_(expert_offsets64)

        def ep_legacy_w13() -> None:
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                ep_gate_up,
                permuted_input,
                expert_offsets,
                dense_expert_ids,
                ep.p13w,
                ep.p13s,
                LOCAL_EXPERTS,
                HIDDEN,
                2 * INTERMEDIATE,
                GROUP_SIZE,
            )

        def ep_activation() -> None:
            torch.ops._C.silu_and_mul_interleaved(ep_intermediate, ep_gate_up)

        def ep_fast_permute() -> None:
            sm70_ops.nvfp4_qwen38_ep4_permute_sm70_out(
                permuted_input,
                expert_offsets,
                inv_permuted_idx,
                x,
                topk_ids,
                ep_start,
            )

        def ep_fast_w13() -> None:
            sm70_ops.nvfp4_moe_fused_swiglu_stage_sm70_out(
                ep_intermediate,
                permuted_input,
                expert_offsets,
                dense_expert_ids,
                ep.p13w,
                ep.p13s,
                LOCAL_EXPERTS,
                HIDDEN,
                2 * INTERMEDIATE,
                GROUP_SIZE,
            )

        def ep_w2() -> None:
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                ep_sorted_output,
                ep_intermediate,
                expert_offsets,
                dense_expert_ids,
                ep.p2w,
                ep.p2s,
                LOCAL_EXPERTS,
                INTERMEDIATE,
                HIDDEN,
                GROUP_SIZE,
            )

        def ep_legacy_unpermute() -> None:
            ep_output.zero_()
            torch.ops._moe_C.moe_unpermute(
                ep_sorted_output,
                topk_weights,
                inv_permuted_idx,
                expert_offsets64,
                TOP_K,
                ep_output,
            )

        def ep_fast_combine() -> None:
            sm70_ops.nvfp4_qwen38_ep4_combine_sm70_out(
                ep_output,
                ep_sorted_output,
                topk_weights,
                inv_permuted_idx,
            )

        def ep_legacy_chain() -> None:
            ep_legacy_permute()
            ep_legacy_w13()
            ep_activation()
            ep_w2()
            ep_legacy_unpermute()
            dist.all_reduce(ep_output)

        def ep_chain() -> None:
            ep_fast_permute()
            ep_fast_w13()
            ep_w2()
            ep_fast_combine()
            dist.all_reduce(ep_output)

        def ep_grouped_chain(blocks: int) -> None:
            sm70_ops.nvfp4_grouped_ep4_w13_sm70_out(
                ep_grouped_intermediate,
                x,
                ep.w13,
                ep.s13,
                topk_ids.view(-1),
                ep_grouped_rows,
                ep_grouped_experts,
                ep_grouped_sizes,
                ep_grouped_total,
                ep_start,
                4 if tokens == 8 else 8,
                True,
                blocks,
            )
            sm70_ops.nvfp4_grouped_ep4_w2_sm70_out(
                ep_grouped_output,
                ep_grouped_routed,
                ep_grouped_intermediate,
                ep.w2,
                ep.s2,
                topk_weights,
                topk_ids.view(-1),
                ep_grouped_rows,
                ep_grouped_experts,
                ep_grouped_sizes,
                ep_grouped_total,
                ep_start,
                blocks,
            )
            dist.all_reduce(ep_grouped_output)

        tp_chain()
        tp_reference = tp_output.clone()
        ep_legacy_chain()
        ep_legacy_reference = ep_output.clone()
        ep_chain()
        torch.cuda.synchronize()
        correctness = _error(ep_output, tp_reference)
        fastpath_correctness = _error(ep_output, ep_legacy_reference)
        if not correctness["finite"]:
            raise RuntimeError(f"EP4 produced non-finite output: {correctness}")

        if rank == 0:
            print(f"Timing M={tokens} TP4 control...", flush=True)
        tp_us, tp_samples, tp_rank_samples = _time_distributed(
            tp_chain,
            rank=rank,
            repeats=args.repeats,
            samples=args.samples,
        )
        if rank == 0:
            print(f"Timing M={tokens} EP4 legacy candidate...", flush=True)
        ep_legacy_us, _, _ = _time_distributed(
            ep_legacy_chain,
            rank=rank,
            repeats=args.repeats,
            samples=args.samples,
        )
        if rank == 0:
            print(f"Timing M={tokens} EP4 fast candidate...", flush=True)
        ep_us, ep_samples, ep_rank_samples = _time_distributed(
            ep_chain,
            rank=rank,
            repeats=args.repeats,
            samples=args.samples,
        )

        ep_grouped_results: list[dict[str, object]] = []
        for blocks in map(int, args.ep4_blocks.split(",")):
            ep_grouped_chain(blocks)
            torch.cuda.synchronize()
            grouped_vs_tp = _error(ep_grouped_output, tp_reference)
            grouped_vs_ep = _error(ep_grouped_output, ep_legacy_reference)
            if not grouped_vs_ep["finite"]:
                raise RuntimeError(
                    f"Grouped EP4 produced non-finite output: {grouped_vs_ep}"
                )
            if rank == 0:
                print(
                    f"Timing M={tokens} grouped EP4 candidate blocks={blocks}...",
                    flush=True,
                )
            grouped_us, grouped_samples, grouped_rank_samples = _time_distributed(
                lambda blocks=blocks: ep_grouped_chain(blocks),
                rank=rank,
                repeats=args.repeats,
                samples=args.samples,
            )
            ep_grouped_results.append(
                {
                    "blocks": blocks,
                    "full_chain_us": grouped_us,
                    "saving_vs_tp_us": tp_us - grouped_us,
                    "speedup_vs_tp": tp_us / grouped_us,
                    "correctness_vs_tp4": grouped_vs_tp,
                    "correctness_vs_legacy_ep4": grouped_vs_ep,
                    "max_rank_samples_us": grouped_samples,
                    "per_rank_samples_us": grouped_rank_samples,
                }
            )

        ep_fast_permute()
        ep_fast_w13()
        ep_w2()
        ep_fast_combine()
        stage_times: dict[str, float] = {}
        for name, fn in (
            ("permute", ep_fast_permute),
            ("w13_fused_swiglu", ep_fast_w13),
            ("w2", ep_w2),
            ("combine", ep_fast_combine),
        ):
            median, _, _ = _time_distributed(
                fn,
                rank=rank,
                repeats=max(args.repeats, 40),
                samples=max(3, args.samples // 2),
                warmup=3,
            )
            stage_times[name] = median
        comm_tensor = torch.zeros_like(ep_output)
        allreduce_us, _, _ = _time_distributed(
            lambda: dist.all_reduce(comm_tensor),
            rank=rank,
            repeats=max(args.repeats, 40),
            samples=max(3, args.samples // 2),
            warmup=3,
        )
        stage_times["allreduce"] = allreduce_us

        local_routes = int(
            ((topk_ids >= ep_start) & (topk_ids < ep_start + LOCAL_EXPERTS))
            .sum()
            .item()
        )
        route_counts: list[int | None] = [None] * world_size
        dist.all_gather_object(route_counts, local_routes)
        case = {
            "tokens": tokens,
            "route_counts_per_rank": [int(v) for v in route_counts if v is not None],
            "tp4_full_chain_us": tp_us,
            "ep4_legacy_full_chain_us": ep_legacy_us,
            "ep4_full_chain_us": ep_us,
            "saving_us": tp_us - ep_us,
            "speedup": tp_us / ep_us,
            "correctness_vs_tp4": correctness,
            "fastpath_correctness_vs_legacy_ep4": fastpath_correctness,
            "tp4_max_rank_samples_us": tp_samples,
            "ep4_max_rank_samples_us": ep_samples,
            "tp4_per_rank_samples_us": tp_rank_samples,
            "ep4_per_rank_samples_us": ep_rank_samples,
            "ep4_stage_diagnostics_us": stage_times,
            "ep4_grouped_candidates": ep_grouped_results,
        }
        cases.append(case)
        if rank == 0:
            print(json.dumps(case), flush=True)

    if rank == 0:
        report = {
            "model": str(args.model),
            "layer": args.layer,
            "routes": str(args.routes),
            "route_sha256": _sha256(args.routes),
            "world_size": world_size,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "mode": "eager full chain with NCCL all-reduce",
            "cases": cases,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.out}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
