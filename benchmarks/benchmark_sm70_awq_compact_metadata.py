# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate compact SM70 AWQ metadata against the regular representation.

The synthetic shapes match one Qwen3.8 TP4 expert shard.  The check is strict:
the reconstructed metadata and every GEMM output must be bitwise identical.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from vllm import _sm70_ops as sm70_ops

GROUP_SIZE = 32
QWEN38_NUM_EXPERTS = 512
QWEN38_NUM_MOE_LAYERS = 48
SHAPES = {
    "w13": (2560, 320),
    "w2": (160, 2560),
}


def _require_sm70() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")
    if not hasattr(torch.ops._C, "awq_sm70_prepare_compact"):
        raise RuntimeError("The build does not contain awq_sm70_prepare_compact.")
    if not hasattr(torch.ops._C, "uint4_sm70_prepare"):
        raise RuntimeError("The build does not contain uint4_sm70_prepare.")


def _make_awq_inputs(
    k: int, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    qweight = torch.randint(
        -(1 << 31),
        1 << 31,
        (k, n // 8),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    qzeros = torch.randint(
        -(1 << 31),
        1 << 31,
        (k // GROUP_SIZE, n // 8),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    scales = (
        torch.rand(
            (k // GROUP_SIZE, n),
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        * 0.02
        + 0.0001
    )
    return qweight, scales, qzeros


def _bitwise_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return (
        lhs.shape == rhs.shape
        and lhs.dtype == rhs.dtype
        and torch.equal(
            lhs.contiguous().view(torch.uint8), rhs.contiguous().view(torch.uint8)
        )
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _prepare_pair(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    interleave_gated_silu: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    regular = sm70_ops.awq_sm70_prepare(
        qweight, scales, qzeros, GROUP_SIZE, interleave_gated_silu
    )
    compact = sm70_ops.awq_sm70_prepare_compact(
        qweight, scales, qzeros, GROUP_SIZE, interleave_gated_silu
    )
    torch.cuda.synchronize()
    return regular, compact


def _rebuild_regular_stats(compact: torch.Tensor) -> torch.Tensor:
    scale = compact[..., :2].contiguous().view(torch.float16).squeeze(-1)
    zero = compact[..., 2].to(torch.float16)
    bias = (-zero * scale).to(torch.float16)
    rebuilt = torch.empty(
        (*compact.shape[:-1], 4), dtype=torch.uint8, device=compact.device
    )
    rebuilt[..., :2] = compact[..., :2]
    rebuilt[..., 2:] = (
        bias.contiguous().view(torch.uint8).reshape(*compact.shape[:-1], 2)
    )
    return rebuilt.contiguous().view(torch.int32).squeeze(-1)


def _gemm_out(
    out: torch.Tensor,
    x: torch.Tensor,
    prepared: list[torch.Tensor],
) -> None:
    weight, stats, meta = prepared
    sm70_ops.awq_gemm_sm70_out(
        out,
        x,
        weight,
        stats,
        GROUP_SIZE,
        int(meta[0].item()),
        int(meta[1].item()),
        False,
    )


def _time_ms(fn: Any, warmup: int, iters: int, rounds: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def _check_dense_case(
    name: str,
    k: int,
    n: int,
    rows: list[int],
    warmup: int,
    iters: int,
    rounds: int,
) -> dict[str, Any]:
    qweight, scales, qzeros = _make_awq_inputs(k, n, seed=100)
    regular, compact = _prepare_pair(
        qweight, scales, qzeros, interleave_gated_silu=name == "w13"
    )
    regular_weight, regular_stats, regular_meta = regular
    compact_weight, compact_stats, compact_meta = compact

    rebuilt = _rebuild_regular_stats(compact_stats)
    checks = {
        "weight_equal": torch.equal(regular_weight, compact_weight),
        "stats_rebuild_equal": torch.equal(regular_stats, rebuilt),
        "k_ld_equal": int(regular_meta[0]) == int(compact_meta[0]),
        "q_ld_sign_contract": int(regular_meta[1]) == -int(compact_meta[1]),
    }
    if not all(checks.values()):
        raise AssertionError(f"{name} prepare mismatch: {checks}")

    timings = []
    for m in rows:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(1000 + m)
        x = torch.randn((m, k), dtype=torch.float16, device="cuda", generator=generator)
        regular_out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        compact_out = torch.empty_like(regular_out)
        _gemm_out(regular_out, x, regular)
        _gemm_out(compact_out, x, compact)
        torch.cuda.synchronize()
        if not _bitwise_equal(regular_out, compact_out):
            max_diff = float((regular_out - compact_out).abs().max().item())
            raise AssertionError(f"{name} m={m} output mismatch: {max_diff=}")

        regular_ms = _time_ms(
            lambda out=regular_out, input_x=x: _gemm_out(out, input_x, regular),
            warmup,
            iters,
            rounds,
        )
        compact_ms = _time_ms(
            lambda out=compact_out, input_x=x: _gemm_out(out, input_x, compact),
            warmup,
            iters,
            rounds,
        )
        timings.append(
            {
                "m": m,
                "regular_ms": regular_ms,
                "compact_ms": compact_ms,
                "compact_overhead_pct": (compact_ms / regular_ms - 1.0) * 100.0,
                "output_equal": True,
            }
        )

    graph_x = torch.randn((1, k), dtype=torch.float16, device="cuda")
    graph_out = torch.empty((1, n), dtype=torch.float16, device="cuda")
    _gemm_out(graph_out, graph_x, compact)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _gemm_out(graph_out, graph_x, compact)
    graph_x.copy_(torch.randn_like(graph_x))
    graph.replay()
    regular_graph_out = torch.empty_like(graph_out)
    _gemm_out(regular_graph_out, graph_x, regular)
    torch.cuda.synchronize()
    if not _bitwise_equal(graph_out, regular_graph_out):
        raise AssertionError(f"{name} CUDA Graph replay mismatch")

    return {
        "name": name,
        "k": k,
        "n": n,
        "checks": checks,
        "regular_bytes": regular_stats.nbytes,
        "compact_bytes": compact_stats.nbytes,
        "saving_bytes": regular_stats.nbytes - compact_stats.nbytes,
        "saving_pct": (regular_stats.nbytes - compact_stats.nbytes)
        / regular_stats.nbytes
        * 100.0,
        "cuda_graph_equal": True,
        "timings": timings,
    }


def _check_dense_uint4_case() -> dict[str, Any]:
    """Exercise the untagged uint32 statistics iterator used by dense uint4."""
    k, n, m = 2560, 320, 64
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260903)
    qweight = torch.randint(
        0,
        16,
        (k, n),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    scales = (
        torch.rand(
            (k // GROUP_SIZE, n),
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        * 0.02
        + 0.0001
    )
    zeros = torch.randint(
        0,
        16,
        (k // GROUP_SIZE, n),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    ).to(torch.float16)
    prepared = sm70_ops.uint4_sm70_prepare(qweight, scales, zeros, GROUP_SIZE, False)
    if int(prepared[2][1].item()) <= 0:
        raise AssertionError("dense uint4 must retain an untagged positive q_ld")

    x = torch.randn((m, k), dtype=torch.float16, device="cuda", generator=generator)
    first = torch.empty((m, n), dtype=torch.float16, device="cuda")
    second = torch.empty_like(first)
    _gemm_out(first, x, prepared)
    _gemm_out(second, x, prepared)
    torch.cuda.synchronize()
    if not _bitwise_equal(first, second):
        raise AssertionError("dense uint4 repeat output mismatch")

    return {
        "m": m,
        "k": k,
        "n": n,
        "group_size": GROUP_SIZE,
        "q_ld": int(prepared[2][1].item()),
        "repeat_bitwise_equal": True,
        "output_sha256": _tensor_sha256(first),
    }


def _prepare_moe_stack(
    name: str, experts: int
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    k, n = SHAPES[name]
    regular_weights = []
    regular_stats = []
    compact_weights = []
    compact_stats = []
    regular_meta = compact_meta = None
    for expert_id in range(experts):
        inputs = _make_awq_inputs(k, n, seed=200 + expert_id + 1000 * n)
        regular, compact = _prepare_pair(*inputs, interleave_gated_silu=name == "w13")
        regular_weights.append(regular[0])
        regular_stats.append(regular[1])
        compact_weights.append(compact[0])
        compact_stats.append(compact[1])
        regular_meta, compact_meta = regular[2], compact[2]

    assert regular_meta is not None and compact_meta is not None
    regular_weights_t = torch.stack(regular_weights)
    regular_stats_t = torch.stack(regular_stats)
    compact_weights_t = torch.stack(compact_weights)
    compact_stats_t = torch.stack(compact_stats)
    if not torch.equal(regular_weights_t, compact_weights_t):
        raise AssertionError(f"{name} stacked weights differ")
    if not torch.equal(regular_stats_t, _rebuild_regular_stats(compact_stats_t)):
        raise AssertionError(f"{name} stacked metadata reconstruction differs")
    regular_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
        regular_weights_t,
        regular_stats_t,
        int(regular_meta[0]),
        int(regular_meta[1]),
        experts,
    )
    compact_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
        compact_weights_t,
        compact_stats_t,
        int(compact_meta[0]),
        int(compact_meta[1]),
        experts,
    )
    return (
        (regular_weights_t, regular_stats_t),
        (compact_weights_t, compact_stats_t),
        regular_ptrs,
        compact_ptrs,
    )


def _check_moe_pointer_path(experts: int, rows_per_expert: int) -> dict[str, Any]:
    if experts != QWEN38_NUM_EXPERTS:
        raise ValueError(f"--experts must be {QWEN38_NUM_EXPERTS}.")
    regular_w13, _compact_w13, regular_ptrs, compact_ptrs = _prepare_moe_stack(
        "w13", experts
    )
    regular_w2, _compact_w2, regular_w2_ptrs, compact_w2_ptrs = _prepare_moe_stack(
        "w2", experts
    )
    # StridedPtr stores raw, non-owning device addresses. Keep both stacked
    # weight/stat tensors alive until every pointer-path launch has completed.

    k, n = SHAPES["w13"]
    total_rows = experts * rows_per_expert
    x = torch.randn((total_rows, k), dtype=torch.float16, device="cuda")
    offsets = torch.arange(
        0,
        total_rows + 1,
        rows_per_expert,
        dtype=torch.int32,
        device="cuda",
    )
    regular_out = torch.empty((total_rows, n), dtype=torch.float16, device="cuda")
    compact_out = torch.empty_like(regular_out)
    sm70_ops.awq_moe_gemm_sm70_per_expert_dispatch_out(
        regular_out,
        x,
        offsets,
        *regular_ptrs,
        experts,
        k,
        n,
        GROUP_SIZE,
        False,
    )
    torch.cuda.synchronize()
    regular_snapshot = regular_out.clone()
    sm70_ops.awq_moe_gemm_sm70_per_expert_dispatch_out(
        compact_out,
        x,
        offsets,
        *compact_ptrs,
        experts,
        k,
        n,
        GROUP_SIZE,
        False,
    )
    torch.cuda.synchronize()
    if not _bitwise_equal(regular_snapshot, compact_out):
        finite = torch.isfinite(regular_snapshot) & torch.isfinite(compact_out)
        max_diff = (
            float((regular_snapshot[finite] - compact_out[finite]).abs().max().item())
            if finite.any()
            else None
        )
        details = {
            "max_finite_diff": max_diff,
            "different_bytes": int(
                (
                    regular_snapshot.contiguous().view(torch.uint8)
                    != compact_out.contiguous().view(torch.uint8)
                )
                .sum()
                .item()
            ),
            "regular_nan": int(torch.isnan(regular_snapshot).sum().item()),
            "compact_nan": int(torch.isnan(compact_out).sum().item()),
            "regular_inf": int(torch.isinf(regular_snapshot).sum().item()),
            "compact_inf": int(torch.isinf(compact_out).sum().item()),
            "regular_corrupted_after_compact": not _bitwise_equal(
                regular_snapshot, regular_out
            ),
        }
        raise AssertionError(f"MoE pointer path mismatch: {details}")

    top_k = min(experts, 10)
    topk_ids = torch.arange(top_k - 1, -1, -1, dtype=torch.int32, device="cuda").view(
        1, top_k
    )
    topk_weights = torch.arange(1, top_k + 1, dtype=torch.float32, device="cuda").view(
        1, top_k
    )
    topk_weights /= topk_weights.sum()
    single_input = torch.randn((1, k), dtype=torch.float16, device="cuda")

    def _legacy_single_token(
        w13_ptrs: list[torch.Tensor], w2_ptrs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_bytes = int(w13_ptrs[0].numel() // experts)
        final_output = torch.zeros((1, k), dtype=torch.float16, device="cuda")
        intermediate = torch.empty(
            (top_k, SHAPES["w2"][0]), dtype=torch.float16, device="cuda"
        )
        sorted_output = torch.empty(
            (top_k, SHAPES["w2"][1]), dtype=torch.float16, device="cuda"
        )
        sm70_ops.awq_moe_single_token_sm70_out(
            final_output,
            single_input,
            topk_weights,
            topk_ids,
            w13_ptrs[0].view(experts, row_bytes),
            w13_ptrs[1].view(experts, row_bytes),
            w2_ptrs[0].view(experts, row_bytes),
            w2_ptrs[1].view(experts, row_bytes),
            torch.empty((top_k, k), dtype=torch.float16, device="cuda"),
            intermediate,
            sorted_output,
            torch.empty(top_k, dtype=torch.float32, device="cuda"),
            torch.empty((top_k, row_bytes), dtype=torch.uint8, device="cuda"),
            torch.empty((top_k, row_bytes), dtype=torch.uint8, device="cuda"),
            torch.empty((top_k, row_bytes), dtype=torch.uint8, device="cuda"),
            torch.empty((top_k, row_bytes), dtype=torch.uint8, device="cuda"),
            torch.empty(top_k + 1, dtype=torch.int32, device="cuda"),
            torch.empty(top_k, dtype=torch.int32, device="cuda"),
            int(regular_w13[0].shape[1]),
            SHAPES["w13"][1],
            int(regular_w2[0].shape[1]),
            SHAPES["w2"][1],
            GROUP_SIZE,
            k,
        )
        return final_output, intermediate, sorted_output

    regular_single = _legacy_single_token(regular_ptrs, regular_w2_ptrs)
    compact_single = _legacy_single_token(compact_ptrs, compact_w2_ptrs)
    torch.cuda.synchronize()
    single_token_equal = all(
        _bitwise_equal(regular_tensor, compact_tensor)
        for regular_tensor, compact_tensor in zip(regular_single, compact_single)
    )
    if not single_token_equal:
        raise AssertionError("Legacy single-token MoE path mismatch")
    return {
        "experts": experts,
        "rows_per_expert": rows_per_expert,
        "output_equal": True,
        "legacy_single_token_equal": single_token_equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,64,512")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--experts", type=int, default=QWEN38_NUM_EXPERTS)
    parser.add_argument("--rows-per-expert", type=int, default=4)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    _require_sm70()
    rows = [int(value) for value in args.rows.split(",")]
    cases = [
        _check_dense_case(
            name,
            *shape,
            rows,
            args.warmup,
            args.iters,
            args.rounds,
        )
        for name, shape in SHAPES.items()
    ]
    dense_uint4 = _check_dense_uint4_case()
    moe_pointer = _check_moe_pointer_path(args.experts, args.rows_per_expert)
    per_card_saving = sum(case["saving_bytes"] for case in cases)
    projected_saving = per_card_saving * QWEN38_NUM_EXPERTS * QWEN38_NUM_MOE_LAYERS
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "strict_bitwise": True,
        "cases": cases,
        "dense_uint4": dense_uint4,
        "moe_pointer_path": moe_pointer,
        "qwen38_tp4_projected_saving_bytes_per_card": projected_saving,
        "qwen38_tp4_projected_saving_gib_per_card": projected_saving / (1 << 30),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
