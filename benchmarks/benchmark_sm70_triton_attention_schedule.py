# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict schedule sweep for latest Triton attention on SM70.

This benchmark exercises the current vLLM ``unified_attention`` implementation
directly. It compares SM70-specific launch meta-parameter overrides against the
default schedule with strict ``torch.equal`` and ``max_diff == 0`` semantics.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode


@dataclass
class ScheduleConfig:
    name: str
    prefill_tile: int = 0
    decode_tile: int = 0
    num_warps: int = 0
    prefill_num_warps: int = 0
    decode_num_warps: int = 0


@dataclass
class CaseResult:
    case: str
    schedule: str
    equal: bool
    max_diff: float
    mean_diff: float
    num_different: int
    median_ms: float
    reference_median_ms: float
    speedup_vs_default: float
    shape: list[int]
    seq_len: int
    query_len: int
    q_heads: int
    kv_heads: int
    head_dim: int
    block_size: int
    prefill_tile: int
    decode_tile: int
    num_warps: int
    prefill_num_warps: int
    decode_num_warps: int


@contextmanager
def _schedule_env(config: ScheduleConfig):
    keys = {
        "VLLM_SM70_TRITON_ATTN_PREFILL_TILE_SIZE": config.prefill_tile,
        "VLLM_SM70_TRITON_ATTN_DECODE_TILE_SIZE": config.decode_tile,
        "VLLM_SM70_TRITON_ATTN_NUM_WARPS": config.num_warps,
        "VLLM_SM70_TRITON_ATTN_PREFILL_NUM_WARPS": config.prefill_num_warps,
        "VLLM_SM70_TRITON_ATTN_DECODE_NUM_WARPS": config.decode_num_warps,
    }
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in keys.items():
            if value:
                os.environ[key] = str(value)
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _make_inputs(
    *,
    seq_len: int,
    query_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (query_len, q_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    k_tokens = torch.randn(
        (seq_len, kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    v_tokens = torch.randn(
        (seq_len, kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )

    num_blocks = (seq_len + block_size - 1) // block_size
    k_cache = torch.zeros(
        (num_blocks, block_size, kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    v_cache = torch.zeros_like(k_cache)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        if end > start:
            k_cache[block_idx, : end - start].copy_(k_tokens[start:end])
            v_cache[block_idx, : end - start].copy_(v_tokens[start:end])

    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, block_table, seq_lens


def _run_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    q_heads: int,
    head_dim: int,
) -> torch.Tensor:
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=query_len,
        seqused_k=seq_lens,
        max_seqlen_k=seq_len,
        softmax_scale=head_dim**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.NONE,
    )
    return out


def _time_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    q_heads: int,
    head_dim: int,
    warmup: int,
    repeats: int,
) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        out = _run_attention(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            head_dim=head_dim,
        )
    torch.cuda.synchronize()

    times = []
    out = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = _run_attention(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            head_dim=head_dim,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    assert out is not None
    times.sort()
    return out, times[len(times) // 2]


def _diff_result(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[bool, float, float, int]:
    diff = (actual.float() - expected.float()).abs()
    return (
        bool(torch.equal(actual, expected)),
        float(diff.max().item()) if diff.numel() else 0.0,
        float(diff.mean().item()) if diff.numel() else 0.0,
        int((actual != expected).sum().item()),
    )


def _parse_schedule(value: str) -> ScheduleConfig:
    parts = value.split(",")
    name = parts[0]
    kwargs = {"name": name}
    for part in parts[1:]:
        key, raw = part.split("=", 1)
        if key == "prefill":
            kwargs["prefill_tile"] = int(raw)
        elif key == "decode":
            kwargs["decode_tile"] = int(raw)
        elif key == "warps":
            kwargs["num_warps"] = int(raw)
        elif key in ("prefill_warps", "pwarps"):
            kwargs["prefill_num_warps"] = int(raw)
        elif key in ("decode_warps", "dwarps"):
            kwargs["decode_num_warps"] = int(raw)
        else:
            raise ValueError(f"Unknown schedule key {key!r} in {value!r}")
    return ScheduleConfig(**kwargs)


def _default_schedules() -> list[ScheduleConfig]:
    return [
        ScheduleConfig("warps1", num_warps=1),
        ScheduleConfig("warps2", num_warps=2),
        ScheduleConfig("warps4", num_warps=4),
        ScheduleConfig("warps8", num_warps=8),
        ScheduleConfig("prefill4_decode8", prefill_num_warps=4, decode_num_warps=8),
        ScheduleConfig("tile16", prefill_tile=16, decode_tile=16),
        ScheduleConfig("tile64", prefill_tile=64, decode_tile=64),
    ]


def _run_case(args: argparse.Namespace, case: str, query_len: int) -> list[CaseResult]:
    q, k_cache, v_cache, block_table, seq_lens = _make_inputs(
        seq_len=args.seq_len,
        query_len=query_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        seed=args.seed + query_len,
    )

    default_config = ScheduleConfig("default")
    with _schedule_env(default_config):
        reference, reference_ms = _time_attention(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            seq_len=args.seq_len,
            query_len=query_len,
            q_heads=args.q_heads,
            head_dim=args.head_dim,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    results = []
    for config in [default_config, *args.schedule]:
        with _schedule_env(config):
            actual, median_ms = _time_attention(
                q,
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                seq_len=args.seq_len,
                query_len=query_len,
                q_heads=args.q_heads,
                head_dim=args.head_dim,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        equal, max_diff, mean_diff, num_different = _diff_result(actual, reference)
        results.append(
            CaseResult(
                case=case,
                schedule=config.name,
                equal=equal,
                max_diff=max_diff,
                mean_diff=mean_diff,
                num_different=num_different,
                median_ms=median_ms,
                reference_median_ms=reference_ms,
                speedup_vs_default=reference_ms / median_ms if median_ms > 0 else 0.0,
                shape=list(actual.shape),
                seq_len=args.seq_len,
                query_len=query_len,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                prefill_tile=config.prefill_tile,
                decode_tile=config.decode_tile,
                num_warps=config.num_warps,
                prefill_num_warps=config.prefill_num_warps,
                decode_num_warps=config.decode_num_warps,
            )
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--prefill-query-len", type=int, default=128)
    parser.add_argument("--decode-query-len", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=528)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--schedule",
        type=_parse_schedule,
        action="append",
        default=None,
        help=(
            "Schedule spec NAME[,prefill=N][,decode=N][,warps=N]. "
            "Use pwarps=N and dwarps=N for split prefill/decode warps. "
            "May be repeated."
        ),
    )
    args = parser.parse_args()
    if args.schedule is None:
        args.schedule = _default_schedules()
    return args


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    results = []
    results.extend(_run_case(args, "prefill", args.prefill_query_len))
    results.extend(_run_case(args, "decode", args.decode_query_len))
    payload = {
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "results": [asdict(result) for result in results],
    }
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
