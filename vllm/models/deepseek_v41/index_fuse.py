# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_INDEX_FUSE: fused SM70 prefill index scores.

The prefill indexer (``attention_ops.select_packed_positions`` ->
``select_compressed_positions``) scores each all-reduce tile of R query rows
as ``torch.mm`` -> ``relu_`` -> ``mul_`` -> ``sum(1)``: a [R, 8, N] fp16 tile
written once and read three times per rank (~96 bytes per query-key pair),
then the per-tile TP all-reduce and one 32-CTA top-k launch per tile at long
context. With the knob on, ``select`` replaces that loop:

1. ``torch.ops.v41pidx.index_head_sum`` (benchmarks/deepseek_v41/v41pidx)
   computes the per-rank head sums of a block of whole tiles in one kernel
   (mma.m8n8k4 scores in registers; 2 bytes per pair written);
2. every tile of the block goes through the same ``all_reduce`` call as
   today: the same rows, the same shape, the same contents, so the same
   collective and the same bits;
3. one ``index_top_k`` launch over the block's rows (top-k is per row; the
   same kernel and inputs per row).

Exactness: the selected index sets equal today's if and only if step 1 gives
today's per-rank partial sums bit for bit. Two things decide that, and both
are GPU facts: whether cuBLAS feeds the K quads of each dot product in order
(``index_head_scores`` vs ``torch.mm``) and which FP32 order PyTorch's
``sum(1)`` uses for 8 heads. The knob value names the order (seq, ilp2,
ilp4, tree); the GPU check (benchmarks/deepseek_v41/v41pidx/check.py)
finds the one that matches, and the knob stays 0 if none does. Columns at or
past the live key count are +0 here and uninitialized-key scores today; top-k
never reads them.

Knobs (registered in vllm/envs.py):
VLLM_V41_INDEX_FUSE 0 (default: off, nothing loads) | seq | ilp2 | ilp4 | tree;
VLLM_V41_INDEX_FUSE_LIB path of v41pidx.so (required when on);
VLLM_V41_INDEX_FUSE_MB scored block budget in MiB (default 64).
Activation: one "V4.1 indexer: VLLM_V41_INDEX_FUSE=<order> ..." line per rank
at import and one "V4.1 indexer: v41pidx fused prefill select first call"
line per rank at the first long prefill.

With VLLM_V41_INDEX_OVERLAP=1 too, a block's reductions and top-k run on the
L4 side stream while the next block is scored (same kernels, same order of
collectives on every rank).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm import envs
from vllm.logger import init_logger

KNOB = "VLLM_V41_INDEX_FUSE"
LIB_KNOB = "VLLM_V41_INDEX_FUSE_LIB"
MB_KNOB = "VLLM_V41_INDEX_FUSE_MB"
# by2 (block.y 2, vt0 4) adds in the same order as ilp2, so it is not listed.
ORDERS = {"seq": 0, "ilp2": 1, "ilp4": 2, "tree": 3}
HEADS = 8
DIM = 128

logger = init_logger(__name__)
_ANNOUNCED = [False]


@dataclass(frozen=True)
class Fuse:
    """The knob turned on: a loaded library, a checked order and budget."""

    name: str
    order: int
    budget: int
    scorer: Callable


def fuse_order(value: str | None) -> int | None:
    """None = off ("0" or unset); else the head-sum order code. Anything else
    refuses, so a typo never runs an unchecked order."""
    text = str(value if value is not None else "0").strip() or "0"
    if text == "0":
        return None
    if text not in ORDERS:
        raise ValueError(f"{KNOB}={text!r}: use 0 (off) or one of {sorted(ORDERS)}")
    return ORDERS[text]


def fuse_budget(value: str | None) -> int:
    """VLLM_V41_INDEX_FUSE_MB in bytes (>= 1 MiB)."""
    text = str(value if value is not None else "64").strip()
    try:
        mib = int(text)
    except ValueError as err:
        raise ValueError(
            f"VLLM_V41_INDEX_FUSE_MB must be an integer, got {text!r}"
        ) from err
    if mib < 1:
        raise ValueError(f"VLLM_V41_INDEX_FUSE_MB must be >= 1, got {mib}")
    return mib << 20


def today_tile(tile_bytes: int, heads: int, keys: int) -> int:
    """select_compressed_positions' all-reduce tile (rows), unchanged."""
    budget = tile_bytes // max(1, heads * keys * 2)
    return max(32, min(1024, budget // 32 * 32))


def block_rows(tile: int, keys: int, budget: int) -> int:
    """Rows of one fused block: a whole number (>= 1) of all-reduce tiles."""
    return tile * max(1, budget // max(1, tile * keys * 2))


def load_library(path: str | None) -> None:
    """Load v41pidx.so from VLLM_V41_INDEX_FUSE_LIB; fail loudly otherwise."""
    if not path:
        raise RuntimeError(f"{KNOB} is on but {LIB_KNOB} is not set")
    if not os.path.isfile(path):
        raise RuntimeError(f"{KNOB}: {LIB_KNOB}={path!r} does not exist")
    torch.ops.load_library(path)
    if not hasattr(torch.ops.v41pidx, "index_head_sum"):
        raise RuntimeError(f"{KNOB}: {path!r} has no v41pidx.index_head_sum")


def cuda_scorer(order: int) -> Callable:
    def score(out, q, w, keys, live):
        torch.ops.v41pidx.index_head_sum(out, q, w, keys, order, live)

    return score


def from_env(getenv: Callable[[str], str | None] | None = None) -> Fuse | None:
    """None when VLLM_V41_INDEX_FUSE is 0/unset (nothing loads); else the
    loaded Fuse. Any bad value or a missing library raises. getenv defaults
    to the registered vllm.envs values."""
    if getenv is None:
        getenv = {
            KNOB: envs.VLLM_V41_INDEX_FUSE,
            LIB_KNOB: envs.VLLM_V41_INDEX_FUSE_LIB,
            MB_KNOB: envs.VLLM_V41_INDEX_FUSE_MB,
        }.get
    order = fuse_order(getenv(KNOB))
    if order is None:
        return None
    name = next(k for k, v in ORDERS.items() if v == order)
    budget = fuse_budget(getenv(MB_KNOB))
    lib = getenv(LIB_KNOB)
    load_library(lib)
    logger.info(
        "V4.1 indexer: %s=%s v41pidx fused prefill scorer (%s, block %d MiB)",
        KNOB,
        name,
        lib,
        budget >> 20,
    )
    return Fuse(name, order, budget, cuda_scorer(order))


def select(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    live: int,
    positions: torch.Tensor,
    valid_tokens: torch.Tensor,
    ratio: int,
    all_reduce: Callable,
    *,
    top_k: int,
    tile: int,
    budget: int,
    scorer: Callable,
    index_topk: Callable,
    side: Callable | None = None,
    join: Callable | None = None,
) -> torch.Tensor:
    """Top-k index rows of select_compressed_positions (fused-top-k branch).

    keys: fp16 [K, 128] with K >= top_k (rows >= live unset); tile: today's
    all-reduce tile; scorer(out, q, w, keys, live) writes the per-rank head
    sums; side(tensor) -> context manager and join() for the L4 overlap.
    """
    tokens, heads = q.shape[0], q.shape[1]
    width = keys.shape[0]
    if (
        q.dtype != torch.float16
        or q.shape[2] != DIM
        or heads != HEADS
        or weights.shape != (tokens, heads)
        or weights.dtype != torch.float16
        or keys.dtype != torch.float16
        or keys.shape[1] != DIM
        or width < top_k
        or not 0 <= live <= width
    ):
        raise ValueError(
            f"{KNOB}: needs fp16 q [T, {HEADS}, {DIM}], weights [T, {HEADS}], "
            f"fp16 keys [K >= top_k, {DIM}] and 0 <= live <= K"
        )
    if (side is None) != (join is None):
        raise ValueError(f"{KNOB}: side and join go together")
    q = q.contiguous()
    weights = weights.contiguous()
    result = torch.empty(tokens, top_k, dtype=torch.int32, device=q.device)
    rows = block_rows(tile, width, budget)
    if not _ANNOUNCED[0]:
        _ANNOUNCED[0] = True
        logger.info(
            "V4.1 indexer: v41pidx fused prefill select first call "
            "(%d rows, %d keys, %d live, tile %d, block %d)",
            tokens,
            width,
            live,
            tile,
            rows,
        )
    for first in range(0, tokens, rows):
        last = min(first + rows, tokens)
        partial = torch.empty(last - first, width, dtype=torch.float16, device=q.device)
        scorer(partial, q[first:last], weights[first:last], keys, live)
        with side(partial) if side is not None else contextlib.nullcontext():
            reduced = torch.empty_like(partial)
            for a in range(0, last - first, tile):
                b = min(a + tile, last - first)
                reduced[a:b].copy_(all_reduce(partial[a:b]))
            index_topk(
                result[first:last],
                reduced,
                positions[first:last],
                valid_tokens[first:last],
                ratio,
                top_k,
            )
    if join is not None:
        join()
    return result
