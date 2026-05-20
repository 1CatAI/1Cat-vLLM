# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V100 fp16 AllReduce overflow regression test.

Exercises ``vllm.distributed.tensor_model_parallel_all_reduce`` against the
patterns from humanjesse/vllm-v100#11 (bayley) and asserts NCCL primitive
behavior on this build matches expectations. Catches the class of breakage
where a NCCL/torch upgrade silently changes reduction precision in a way that
would invalidate the per-model fp32-promote pattern we use in models like
``minimax_m2.py`` (see ``vllm/model_executor/models/minimax_m2.py`` and
``VLLM_ALLREDUCE_OVERFLOW_STRATEGY``).

Hard assertions (must all hold on Volta/V100 + NCCL with fp16-internal AR):
  * Scenario 4: all+10000 (true sum 80000) overflows fp16, result is Inf.
  * Scenario 5: asymmetric +60000x7,-64000 (true sum 356000) overflows, Inf.
  * Scenario fp32: same patterns in fp32 stay finite and match the sum.

Observational (logged, not asserted -- platform/algo dependent):
  * Scenario 1: bayley's symmetric ±20000 may or may not overflow depending
    on NCCL's ring/tree partial-sum ordering. On our 8x V100 SXM2 NCCL falls
    through to the PyNccl path (no CustomAllReduce -- Volta no symm-mem), and
    today the symmetric pattern cancels cleanly to 0. If this flips to Inf
    after a NCCL upgrade, models patched with the fp32-AR pattern are still
    safe; models without it may regress.

Launch (NOT pytest -- requires torchrun for distributed init):
    cd /path/to/vllm-v100
    torchrun --nproc-per-node=8 --master-port=29500 \\
        tests/distributed/test_fp16_allreduce_overflow.py

Exit code 0 = all hard assertions passed. Non-zero = a primitive behavior
changed; investigate whether VLLM_ALLREDUCE_OVERFLOW_STRATEGY default in
fp16-MoE model classes still does what it claims.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.distributed as dist

from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import get_tp_group


def _fill(rank: int, dtype: torch.dtype, n_elems: int, value_fn) -> torch.Tensor:
    return torch.full(
        (n_elems,), value_fn(rank), dtype=dtype, device=torch.device(f"cuda:{rank}")
    )


def _print_rank0(rank: int, *args, **kwargs) -> None:
    if rank == 0:
        print(*args, **kwargs, flush=True)


# Per-rank record of failed assertions; rank 0 prints summary at end.
_failures: list[str] = []


def _expect(rank: int, cond: bool, label: str) -> None:
    """Assert ``cond`` on rank 0 only -- the other ranks have already
    collectively executed the AR primitive so the rank-0 observation is
    sufficient. Failures are accumulated; we report all of them before
    exiting non-zero so a single run surfaces every regression."""
    if rank != 0:
        return
    tag = "OK" if cond else "FAIL"
    _print_rank0(rank, f"  [{tag}] {label}")
    if not cond:
        _failures.append(label)


def run_all(rank: int, tp_size: int) -> None:
    torch.cuda.set_device(torch.device(f"cuda:{rank}"))
    if rank == 0:
        g = get_tp_group()
        print(
            "\n=== TP group config ===\n"
            f"  world_size={tp_size}\n"
            f"  CustomAllReduce={getattr(g, 'ca_comm', None) is not None}\n"
            f"  PyNcclCommunicator={getattr(g, 'pynccl_comm', None) is not None}\n",
            flush=True,
        )

    # Scenario 1 -- bayley symmetric pattern (observational).
    _print_rank0(rank, "=== Scenario 1: ±20000 symmetric (observational) ===")
    fn = lambda r: 20000.0 if r < tp_size // 2 else -20000.0
    y = tensor_model_parallel_all_reduce(_fill(rank, torch.float16, 1, fn).clone())
    if rank == 0:
        first = y.flatten()[0].item()
        print(
            f"  fp16: observed={first:+.4f}  finite={math.isfinite(first)}",
            flush=True,
        )
    dist.barrier()

    # Scenario 4 -- guaranteed overflow, must produce Inf in fp16 if the AR
    # primitive is doing fp16-internal accumulation. If a future NCCL or
    # build flips to fp32 internal, this will FAIL with a finite value; that
    # is significant (it means our per-model fp32-promote work is redundant
    # and the helper-vs-per-model debate should be revisited).
    _print_rank0(rank, "\n=== Scenario 4: all +10000, true sum 80000 (must overflow fp16) ===")
    y = tensor_model_parallel_all_reduce(
        _fill(rank, torch.float16, 1, lambda r: 10000.0).clone()
    )
    if rank == 0:
        first = y.flatten()[0].item()
        _expect(rank, math.isinf(first), f"fp16 result is Inf (got {first})")

    y32 = tensor_model_parallel_all_reduce(
        _fill(rank, torch.float32, 1, lambda r: 10000.0).clone()
    )
    if rank == 0:
        first = y32.flatten()[0].item()
        expected = 10000.0 * tp_size
        _expect(
            rank,
            math.isfinite(first) and abs(first - expected) < max(1.0, expected * 0.01),
            f"fp32 result ~= {expected} (got {first})",
        )
    dist.barrier()

    # Scenario 5 -- asymmetric overflow stress.
    _print_rank0(rank, f"\n=== Scenario 5: +60000x{tp_size - 1},-64000 (must overflow fp16) ===")
    fn = lambda r: -64000.0 if r == tp_size - 1 else 60000.0
    y = tensor_model_parallel_all_reduce(_fill(rank, torch.float16, 1, fn).clone())
    if rank == 0:
        first = y.flatten()[0].item()
        _expect(rank, math.isinf(first), f"fp16 result is Inf (got {first})")

    y32 = tensor_model_parallel_all_reduce(_fill(rank, torch.float32, 1, fn).clone())
    if rank == 0:
        first = y32.flatten()[0].item()
        expected = 60000.0 * (tp_size - 1) + (-64000.0)
        _expect(
            rank,
            math.isfinite(first) and abs(first - expected) < max(1.0, abs(expected) * 0.01),
            f"fp32 result ~= {expected} (got {first})",
        )
    dist.barrier()

    # Scenario small-magnitude control -- must stay finite and ~= sum.
    _print_rank0(rank, "\n=== Control: ±5000 small magnitudes (must stay finite) ===")
    fn = lambda r: 5000.0 if r < tp_size // 2 else -5000.0
    y = tensor_model_parallel_all_reduce(_fill(rank, torch.float16, 1, fn).clone())
    if rank == 0:
        first = y.flatten()[0].item()
        _expect(
            rank,
            math.isfinite(first) and abs(first) < 1.0,
            f"fp16 cancels symmetric small magnitudes to ~0 (got {first})",
        )
    dist.barrier()


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed_environment(
        world_size=world_size, rank=rank,
        distributed_init_method="env://",
        local_rank=int(os.environ["LOCAL_RANK"]), backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        run_all(rank, world_size)
    finally:
        dist.barrier()
    if rank == 0:
        if _failures:
            print(
                f"\n=== {len(_failures)} FAILURE(S) ===", flush=True,
            )
            for label in _failures:
                print(f"  - {label}", flush=True)
            return 1
        print("\n=== all hard assertions passed ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
