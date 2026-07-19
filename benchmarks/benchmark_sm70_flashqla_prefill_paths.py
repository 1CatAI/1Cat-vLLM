# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare SM70 FlashQLA public, vLLM-layout, and wrapper-like prefill paths."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

import torch

from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.sm70 import (
    chunk_gated_delta_rule_fwd_sm70_vlk_varlen,
    resolve_column_groups_per_block_sm70,
)
from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)


def _bench_ms(fn: Callable[[], object], warmup: int, repeat: int) -> float:
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


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    delta = (a.float() - b.float()).abs()
    return {
        "max": float(delta.max().item()),
        "mean": float(delta.mean().item()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--v-heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--state-slots", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dim != 128:
        raise ValueError("FlashQLA SM70 path currently requires dim=128")

    device = torch.device(args.device)
    torch.accelerator.set_device_index(device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    torch.manual_seed(1234)
    scale = args.dim**-0.5

    q = (
        torch.randn(
            1,
            args.tokens,
            args.q_heads,
            args.dim,
            device=device,
            dtype=dtype,
        )
        * 0.05
    ).contiguous()
    k = (torch.randn_like(q) * 0.05).contiguous()
    v = (
        torch.randn(
            1,
            args.tokens,
            args.v_heads,
            args.dim,
            device=device,
            dtype=dtype,
        )
        * 0.1
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
    beta = torch.rand(
        1,
        args.tokens,
        args.v_heads,
        device=device,
        dtype=torch.float32,
    ).contiguous()

    public_state = (
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
    # vLLM stores state as [slots, Hv, V, K]. K==V for Qwen, so use a
    # transpose to preserve values while exercising the real layout contract.
    vlk_state = public_state.transpose(-1, -2).contiguous()
    cu_seqlens = torch.tensor([0, args.tokens], device=device, dtype=torch.int32)
    state_cache = torch.empty(
        args.state_slots,
        args.v_heads,
        args.dim,
        args.dim,
        device=device,
        dtype=torch.float32,
    )
    state_cache[0].copy_(vlk_state[0])
    state_indices = torch.zeros(1, device=device, dtype=torch.long)
    has_initial_state = torch.ones(1, device=device, dtype=torch.bool)
    core_out = torch.empty_like(v)
    conv_output = torch.cat(
        (
            q.reshape(args.tokens, -1),
            k.reshape(args.tokens, -1),
            v.reshape(args.tokens, -1),
        ),
        dim=-1,
    ).contiguous()
    a = torch.randn(
        args.tokens,
        args.v_heads,
        device=device,
        dtype=dtype,
    ).contiguous()
    b = torch.randn_like(a).contiguous()
    A_log = torch.randn(args.v_heads, device=device, dtype=torch.float32).contiguous()
    dt_bias = torch.randn(args.v_heads, device=device, dtype=torch.float32).contiguous()

    def public_call():
        return chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=public_state,
            output_final_state=True,
        )

    def vlk_call():
        return chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=cu_seqlens,
            scale=scale,
            initial_state=vlk_state,
            output_final_state=True,
        )

    def wrapper_like_call():
        initial_state = state_cache[state_indices].contiguous()
        initial_state[~has_initial_state] = 0
        out, final_state = chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
        )
        core_out.copy_(out)
        state_cache[state_indices] = final_state.to(state_cache.dtype)
        return core_out, final_state

    def post_conv_prep_call():
        return fused_post_conv_prep(
            conv_output,
            a,
            b,
            A_log,
            dt_bias,
            args.q_heads,
            args.dim,
            args.dim,
            apply_l2norm=True,
            output_g_exp=False,
        )

    def post_conv_plus_vlk_call():
        q2, k2, v2, g2, beta2 = post_conv_prep_call()
        return chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q2.unsqueeze(0),
            k2.unsqueeze(0),
            v2.unsqueeze(0),
            g2.unsqueeze(0),
            beta2.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            scale=scale,
            initial_state=vlk_state,
            output_final_state=True,
        )

    public_out, public_final = public_call()
    vlk_out, vlk_final = vlk_call()
    wrapper_out, wrapper_final = wrapper_like_call()
    prep_q, prep_k, prep_v, prep_g, prep_beta = post_conv_prep_call()
    post_conv_out, post_conv_final = post_conv_plus_vlk_call()
    torch.accelerator.synchronize()

    public_ms = _bench_ms(public_call, args.warmup, args.repeat)
    vlk_ms = _bench_ms(vlk_call, args.warmup, args.repeat)
    wrapper_ms = _bench_ms(wrapper_like_call, args.warmup, args.repeat)
    post_conv_ms = _bench_ms(post_conv_prep_call, args.warmup, args.repeat)
    post_conv_plus_vlk_ms = _bench_ms(
        post_conv_plus_vlk_call,
        args.warmup,
        args.repeat,
    )

    state_cache[0].copy_(vlk_state[0])
    wrapper_out, wrapper_final = wrapper_like_call()
    torch.accelerator.synchronize()

    result = {
        "shape": {
            "tokens": args.tokens,
            "q_heads": args.q_heads,
            "v_heads": args.v_heads,
            "dim": args.dim,
            "state_slots": args.state_slots,
            "dtype": args.dtype,
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK": os.environ.get(
                "FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK"
            ),
        },
        "resolved_column_groups_per_block": resolve_column_groups_per_block_sm70(
            args.tokens,
            args.q_heads,
            args.v_heads,
        ),
        "timing_ms": {
            "public": public_ms,
            "vlk_varlen": vlk_ms,
            "wrapper_like": wrapper_ms,
            "wrapper_over_vlk": wrapper_ms - vlk_ms,
            "post_conv_prep": post_conv_ms,
            "post_conv_plus_vlk": post_conv_plus_vlk_ms,
            "post_conv_plus_vlk_over_vlk": post_conv_plus_vlk_ms - vlk_ms,
        },
        "diff_vs_public": {
            "vlk_out": _diff(public_out, vlk_out),
            "vlk_state_transposed": _diff(public_final.transpose(-1, -2), vlk_final),
            "wrapper_out": _diff(public_out, wrapper_out),
            "wrapper_state_transposed": _diff(
                public_final.transpose(-1, -2),
                wrapper_final,
            ),
        },
        "contiguous": {
            "q": q.is_contiguous(),
            "k": k.is_contiguous(),
            "v": v.is_contiguous(),
            "g": g.is_contiguous(),
            "beta": beta.is_contiguous(),
            "vlk_state": vlk_state.is_contiguous(),
            "core_out": core_out.is_contiguous(),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
