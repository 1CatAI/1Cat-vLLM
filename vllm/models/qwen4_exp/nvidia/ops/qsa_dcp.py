# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP localization for 1Cat QSA's token-ID / -1 selection ABI.

Adapted from vllm-project/vllm#57431. Unlike that patch's packed selection,
every column here is a token ID; there is no trailing count column. The
localized buffer must be separate because subsequent MTP steps reuse the
original global selection.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _localize_indices(
    src,
    dst,
    src_stride,
    dst_stride,
    WIDTH: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token = tl.load(src + row * src_stride + col, col < WIDTH, other=-1)
    owned = (col < WIDTH) & (token >= 0) & ((token // INTERLEAVE) % WORLD == RANK)
    local = token // (WORLD * INTERLEAVE) * INTERLEAVE + token % INTERLEAVE
    position = tl.cumsum(owned.to(tl.int32), 0) - 1
    count = tl.sum(owned.to(tl.int32), 0)
    # Disjoint writes avoid races between warp-local compaction and padding.
    tl.store(dst + row * dst_stride + col, -1, (col < WIDTH) & (col >= count))
    tl.store(dst + row * dst_stride + position, local, owned)


def qsa_localize_dcp_indices(
    indices: torch.Tensor,
    out: torch.Tensor,
    *,
    dcp_world_size: int,
    dcp_rank: int,
    interleave_size: int,
    local_block_size: int,
) -> torch.Tensor:
    """Compact this rank's selected IDs into an independent persistent buffer.

    The output has the same shape; unused entries are -1. Its IDs address the
    rank's compact local token space and can be looked up with its block table.
    Requiring interleave to divide a local page makes flat localization agree
    with block-table slot mapping at every global page boundary.
    """
    if indices.ndim != 2 or out.shape != indices.shape or indices.shape[1] == 0:
        raise ValueError("QSA DCP needs matching [rows, selection_width] buffers")
    if indices.dtype != torch.int32 or out.dtype != torch.int32:
        raise ValueError("QSA DCP indices must be int32")
    if indices.device != out.device:
        raise ValueError("QSA DCP buffers must be on the same device")
    if indices.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("QSA DCP selection columns must be contiguous")
    if indices.stride(0) < indices.shape[1] or out.stride(0) < out.shape[1]:
        raise ValueError("QSA DCP rows must not overlap")
    if dcp_world_size < 1 or not 0 <= dcp_rank < dcp_world_size:
        raise ValueError("QSA DCP rank must belong to a positive world size")
    if (
        interleave_size < 1
        or local_block_size < 1
        or local_block_size % interleave_size
    ):
        raise ValueError("QSA DCP interleave must divide the positive local block size")
    if indices.numel() == 0:
        return out
    if indices.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
        raise ValueError("QSA DCP output must not alias the MTP selection buffer")
    if dcp_world_size == 1:
        out.copy_(indices)
        return out
    if not indices.is_cuda:
        raise ValueError("QSA DCP localization requires CUDA buffers")
    _localize_indices[(indices.shape[0],)](
        indices,
        out,
        indices.stride(0),
        out.stride(0),
        WIDTH=indices.shape[1],
        WORLD=dcp_world_size,
        RANK=dcp_rank,
        INTERLEAVE=interleave_size,
        BLOCK=triton.next_power_of_2(indices.shape[1]),
    )
    return out
