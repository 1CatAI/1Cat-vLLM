# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit SM70 block-sparse attention for pre-tiled DiT operands."""

import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path


@lru_cache(maxsize=1)
def sparse_extension():
    try:
        return import_module("vllm._sm70_sparse_attention_C")
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    cutlass_path = os.environ.get("VLLM_CUTLASS_SRC_DIR")
    if not cutlass_path:
        raise RuntimeError("SM70 sparse attention requires CUTLASS v4.4.2 sources")
    cutlass = Path(cutlass_path)
    root = Path(__file__).resolve().parents[3]
    return load(
        name="onecat_sm70_sparse_attention",
        sources=[str(root / "flash-attention-v100/kernel/h3/forward_sparse.cu")],
        extra_include_paths=[
            str(cutlass / "include"),
            str(cutlass / "examples/41_fused_multi_head_attention"),
        ],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=False,
    )


def block_sparse_attention(q, k, v, block_map, block_sizes, *, scale):
    """Execute only selected 64-token blocks, excluding every padded edge.

    Operands use FP16 [B,64*N,H,128]; block_map is bool [B,H,N,N] and
    block_sizes is int32 [N]. Each selected block is visited in ascending
    logical order. Unused output rows remain zero. No dense fallback exists.
    """
    return sparse_extension().forward(q, k, v, block_map, block_sizes, scale)


def _h3_block_sparse_attention(q, k, v, block_map, block_sizes, *, scale):
    """Private route for H3-owned sizes and a nonempty mask built by H3.

    Arbitrary external maps must use block_sparse_attention and its value
    checks. Older wheels retain that checked entrypoint until rebuilt.
    """
    ops = sparse_extension()
    forward = getattr(ops, "_forward_prevalidated", ops.forward)
    return forward(q, k, v, block_map, block_sizes, scale)
