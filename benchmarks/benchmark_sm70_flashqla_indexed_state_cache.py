# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark SM70 original FlashQLA indexed state-cache mode.

This isolates the vLLM integration question: whether passing the full state
cache plus state_indices is faster than gathering one sequence state tensor
and writing final_state back outside the kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flash_qla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd_sm70_tilelang,
)


def _bench_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--v-heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--state-slots", type=int, default=1024)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dim != 128:
        raise ValueError("SM70 FlashQLA original path requires dim=128")
    if args.state_index < 0 or args.state_index >= args.state_slots:
        raise ValueError("--state-index must be inside --state-slots")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    dtype = torch.float16
    scale = args.dim**-0.5

    q = (torch.randn(1, args.tokens, args.q_heads, args.dim,
                     device=device, dtype=dtype) * 0.05).contiguous()
    k = (torch.randn_like(q) * 0.05).contiguous()
    v = (torch.randn(1, args.tokens, args.v_heads, args.dim,
                     device=device, dtype=dtype) * 0.1).contiguous()
    g = (torch.randn(1, args.tokens, args.v_heads,
                     device=device, dtype=torch.float32) * 0.02 - 0.04).contiguous()
    beta = torch.rand(1, args.tokens, args.v_heads,
                      device=device, dtype=torch.float32).contiguous()
    cu_seqlens = torch.tensor([0, args.tokens], device=device, dtype=torch.int32)
    state_indices = torch.tensor([args.state_index], device=device, dtype=torch.long)
    has_initial_state = torch.ones(1, device=device, dtype=torch.bool)

    base_state_cache = (torch.randn(
        args.state_slots,
        args.v_heads,
        args.dim,
        args.dim,
        device=device,
        dtype=torch.float32,
    ) * 0.01).contiguous()
    gather_cache = base_state_cache.clone()
    indexed_cache = base_state_cache.clone()
    gather_out = torch.empty_like(v)
    indexed_out = torch.empty_like(v)

    def gather_writeback_call():
        initial_state = gather_cache[state_indices].contiguous()
        initial_state[~has_initial_state] = 0
        _, _, out, _, final_state = chunk_gated_delta_rule_fwd_sm70_tilelang(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            scale=scale,
            output_final_state=True,
            output_h=False,
            auto_cp=False,
            state_layout_vlk=True,
            output=gather_out,
            inplace_final_state=False,
        )
        gather_cache[state_indices] = final_state
        return out

    def indexed_call():
        _, _, out, _, final_state = chunk_gated_delta_rule_fwd_sm70_tilelang(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            initial_state=indexed_cache,
            scale=scale,
            output_final_state=False,
            output_h=False,
            auto_cp=False,
            state_layout_vlk=True,
            output=indexed_out,
            inplace_final_state=True,
        )
        if final_state is not None:
            raise AssertionError("indexed inplace mode should not return final_state")
        return out

    gather_cache.copy_(base_state_cache)
    indexed_cache.copy_(base_state_cache)
    gather_result = gather_writeback_call()
    indexed_result = indexed_call()
    torch.cuda.synchronize()

    correctness = {
        "out": _diff(gather_result, indexed_result),
        "state_slot": _diff(gather_cache[args.state_index],
                            indexed_cache[args.state_index]),
    }
    if args.state_slots > 1:
        untouched_index = 0 if args.state_index != 0 else args.state_slots - 1
        correctness["untouched_slot"] = _diff(
            gather_cache[untouched_index],
            indexed_cache[untouched_index],
        )

    gather_cache.copy_(base_state_cache)
    indexed_cache.copy_(base_state_cache)
    gather_ms = _bench_ms(gather_writeback_call, args.warmup, args.repeat)
    indexed_ms = _bench_ms(indexed_call, args.warmup, args.repeat)
    torch.cuda.synchronize()

    result = {
        "shape": {
            "tokens": args.tokens,
            "q_heads": args.q_heads,
            "v_heads": args.v_heads,
            "dim": args.dim,
            "state_slots": args.state_slots,
            "state_index": args.state_index,
        },
        "timing_ms": {
            "gather_writeback": gather_ms,
            "indexed_inplace": indexed_ms,
            "delta_ms": gather_ms - indexed_ms,
            "speedup": gather_ms / indexed_ms,
        },
        "correctness": correctness,
        "ptr_match": {
            "gather_out": gather_result.data_ptr() == gather_out.data_ptr(),
            "indexed_out": indexed_result.data_ptr() == indexed_out.data_ptr(),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
