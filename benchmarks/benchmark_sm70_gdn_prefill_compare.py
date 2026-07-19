# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare SM70 GDN prefill recurrent backends on the vLLM tensor contract."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from flash_qla.ops.gated_delta_rule.chunk.sm70 import (
    chunk_gated_delta_rule_fwd_sm70_vlk_varlen,
    resolve_column_groups_per_block_sm70,
)
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE


def _bench_ms(fn: Callable[[], Any], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    delta = (a.float() - b.float()).abs()
    return {
        "max": float(delta.max().item()),
        "mean": float(delta.mean().item()),
        "num_different": int((delta != 0).sum().item()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--v-heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dim != 128:
        raise ValueError("SM70 FlashQLA path currently requires dim=128")

    device = torch.device(args.device)
    torch.accelerator.set_device_index(device)
    dtype = torch.float16
    torch.manual_seed(1234)
    scale = args.dim**-0.5

    q = torch.randn(1, args.tokens, args.q_heads, args.dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(dtype).contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(dtype).contiguous()
    v = torch.randn(
        1, args.tokens, args.v_heads, args.dim, device=device, dtype=dtype
    ).contiguous()
    g = (
        torch.randn(
            1,
            args.tokens,
            args.v_heads,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
        - 0.04
    ).contiguous()
    g_exp = torch.exp(g).contiguous()
    beta = torch.rand(
        1,
        args.tokens,
        args.v_heads,
        device=device,
        dtype=torch.float32,
    ).contiguous()
    initial_state = (
        torch.randn(
            1,
            args.v_heads,
            args.dim,
            args.dim,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    ).contiguous()
    cu_seqlens = torch.tensor([0, args.tokens], device=device, dtype=torch.int32)
    cu_seqlens_cpu = torch.tensor([0, args.tokens], dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens_cpu, FLA_CHUNK_SIZE).to(
        device=device, non_blocking=True
    )
    chunk_offsets = prepare_chunk_offsets(cu_seqlens_cpu, FLA_CHUNK_SIZE).to(
        device=device, non_blocking=True
    )
    fla_core_out = torch.empty_like(v)
    qla_direct_out = torch.empty_like(v)
    qla_direct_exp_out = torch.empty_like(v)
    qla_copy_out = torch.empty_like(v)

    def fla_call() -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=False,
        )

    def fla_core_out_call() -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=False,
            core_attn_out=fla_core_out,
        )

    def qla_call() -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            validate_cu_seqlens=False,
        )

    def qla_direct_out_call() -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            validate_cu_seqlens=False,
            output=qla_direct_out,
        )

    def qla_exp_gate_call() -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q=q,
            k=k,
            v=v,
            g=g_exp,
            beta=beta,
            cu_seqlens=cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            validate_cu_seqlens=False,
            gate_is_exp=True,
        )

    def qla_exp_gate_direct_out_call() -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q=q,
            k=k,
            v=v,
            g=g_exp,
            beta=beta,
            cu_seqlens=cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            validate_cu_seqlens=False,
            gate_is_exp=True,
            output=qla_direct_exp_out,
        )

    def qla_exp_gate_copy_to_core_call() -> tuple[torch.Tensor, torch.Tensor | None]:
        out, state = qla_exp_gate_call()
        qla_copy_out.copy_(out)
        return qla_copy_out, state

    fla_out, fla_state = fla_call()
    qla_out, qla_state = qla_call()
    qla_direct_out_result, qla_direct_state = qla_direct_out_call()
    qla_exp_out, qla_exp_state = qla_exp_gate_call()
    qla_exp_direct_out_result, qla_exp_direct_state = qla_exp_gate_direct_out_call()
    qla_exp_copy_out, qla_exp_copy_state = qla_exp_gate_copy_to_core_call()
    fla_core_out_result, fla_core_state = fla_core_out_call()
    torch.accelerator.synchronize()

    result = {
        "shape": {
            "tokens": args.tokens,
            "q_heads": args.q_heads,
            "v_heads": args.v_heads,
            "dim": args.dim,
            "dtype": args.dtype,
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK": os.environ.get(
                "FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK"
            ),
            "VLLM_SM70_GDN_CHUNK_O_SCHEDULE": os.environ.get(
                "VLLM_SM70_GDN_CHUNK_O_SCHEDULE"
            ),
            "VLLM_SM70_GDN_KKT_SCHEDULE": os.environ.get("VLLM_SM70_GDN_KKT_SCHEDULE"),
            "VLLM_SM70_GDN_DELTA_H_SCHEDULE": os.environ.get(
                "VLLM_SM70_GDN_DELTA_H_SCHEDULE"
            ),
        },
        "resolved_column_groups_per_block": resolve_column_groups_per_block_sm70(
            args.tokens, args.q_heads, args.v_heads
        ),
        "chunk": {
            "size": FLA_CHUNK_SIZE,
            "num_chunks": int(chunk_indices.shape[0]),
        },
        "timing_ms": {
            "fla": _bench_ms(fla_call, args.warmup, args.repeat),
            "fla_core_out": _bench_ms(fla_core_out_call, args.warmup, args.repeat),
            "flashqla_vlk_varlen": _bench_ms(qla_call, args.warmup, args.repeat),
            "flashqla_vlk_varlen_direct_out": _bench_ms(
                qla_direct_out_call,
                args.warmup,
                args.repeat,
            ),
            "flashqla_vlk_varlen_exp_gate": _bench_ms(
                qla_exp_gate_call,
                args.warmup,
                args.repeat,
            ),
            "flashqla_vlk_varlen_exp_gate_direct_out": _bench_ms(
                qla_exp_gate_direct_out_call,
                args.warmup,
                args.repeat,
            ),
            "flashqla_vlk_varlen_exp_gate_copy_to_core": _bench_ms(
                qla_exp_gate_copy_to_core_call,
                args.warmup,
                args.repeat,
            ),
        },
        "diff_vs_fla": {
            "flashqla_out": _diff(fla_out, qla_out),
            "flashqla_state": _diff(fla_state, qla_state),
            "flashqla_direct_out": _diff(fla_out, qla_direct_out_result),
            "flashqla_direct_state": _diff(fla_state, qla_direct_state),
            "flashqla_exp_gate_out": _diff(fla_out, qla_exp_out),
            "flashqla_exp_gate_state": _diff(fla_state, qla_exp_state),
            "flashqla_exp_gate_direct_out": _diff(fla_out, qla_exp_direct_out_result),
            "flashqla_exp_gate_direct_state": _diff(fla_state, qla_exp_direct_state),
            "flashqla_exp_gate_copy_out": _diff(fla_out, qla_exp_copy_out),
            "flashqla_exp_gate_copy_state": _diff(fla_state, qla_exp_copy_state),
            "fla_core_out": _diff(fla_out, fla_core_out_result),
            "fla_core_state": _diff(fla_state, fla_core_state),
        },
        "diff_exp_gate_vs_log_gate": {
            "out": _diff(qla_out, qla_exp_out),
            "state": _diff(qla_state, qla_exp_state),
        },
        "direct_output_reused": {
            "flashqla_vlk_varlen_direct_out": (
                qla_direct_out_result.data_ptr() == qla_direct_out.data_ptr()
            ),
            "flashqla_vlk_varlen_exp_gate_direct_out": (
                qla_exp_direct_out_result.data_ptr() == qla_direct_exp_out.data_ptr()
            ),
        },
    }
    timings = result["timing_ms"]
    result["speedup_vs_fla"] = {
        "flashqla_vlk_varlen": timings["fla"] / timings["flashqla_vlk_varlen"],
        "flashqla_vlk_varlen_direct_out": (
            timings["fla"] / timings["flashqla_vlk_varlen_direct_out"]
        ),
        "flashqla_vlk_varlen_exp_gate": (
            timings["fla"] / timings["flashqla_vlk_varlen_exp_gate"]
        ),
        "flashqla_vlk_varlen_exp_gate_direct_out": (
            timings["fla"] / timings["flashqla_vlk_varlen_exp_gate_direct_out"]
        ),
        "flashqla_vlk_varlen_exp_gate_copy_to_core": (
            timings["fla"] / timings["flashqla_vlk_varlen_exp_gate_copy_to_core"]
        ),
        "fla_core_out": timings["fla"] / timings["fla_core_out"],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
