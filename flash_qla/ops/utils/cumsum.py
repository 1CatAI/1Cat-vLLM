# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_indices


@tilelang.jit(
    # out_idx=[-1],
)
def tilelang_chunk_local_cumsum(
    H,
    chunk_size,
    accum_dtype,
    g_dtype,
    seqlen_dtype,
    is_varlen,
    reverse,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    g_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(
        bb,
        bc,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        g_raw,
        g_cumsum,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        g_fragment = T.alloc_fragment((H, block_S), dtype=accum_dtype)
        gT_fragment = T.alloc_fragment((block_S, H), dtype=g_dtype)
        gT_shared = T.alloc_shared((block_S, H + 1), dtype=g_dtype)

        if right <= seq_end_idx:
            T.copy(g_raw[bb, left:right, 0:H], gT_fragment)
        else:
            for j, i in T.Parallel(block_S, H):
                if left + j < seq_end_idx:
                    gT_fragment[j, i] = g_raw[bb, left + j, i]
                else:
                    gT_fragment[j, i] = 0
        T.copy(gT_fragment, gT_shared[:, :H])

        for i, j in T.Parallel(H, block_S):
            g_fragment[i, j] = gT_shared[j, i]

        T.cumsum(g_fragment, dim=1, reverse=reverse)

        for i, j in T.Parallel(H, block_S):
            gT_shared[j, i] = g_fragment[i, j]

        T.copy(gT_shared[:, :H], gT_fragment)
        if right <= seq_end_idx:
            T.copy(gT_fragment, g_cumsum[bb, left:right, 0:H])
        else:
            for j, i in T.Parallel(block_S, H):
                if left + j < seq_end_idx:
                    g_cumsum[bb, left + j, i] = gT_fragment[j, i]

    if is_varlen:

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        ):
            with T.Kernel(num_chunks, threads=128) as (bc,):
                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    bc,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

    else:

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks, threads=128) as (bc,):
                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

    return tilelang_chunk_local_cumsum_kernel


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    reverse: bool = False,
):
    batch_size, num_tokens, H = g.shape
    assert g.stride(-1) == 1

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        else:
            if chunk_indices.ndim != 2 or chunk_indices.shape[-1] != 2:
                raise ValueError("chunk_indices must have shape [num_chunks, 2]")
            if chunk_indices.device != cu_seqlens.device:
                raise ValueError(
                    "chunk_indices must be on the same device as cu_seqlens"
                )
            chunk_indices = chunk_indices.to(cu_seqlens.dtype).contiguous()
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    g_cumsum = torch.empty_like(g)

    tilelang_chunk_local_cumsum_kernel = tilelang_chunk_local_cumsum(
        H,
        chunk_size,
        g_dtype=g.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
        reverse=reverse,
    )
    if is_varlen:
        tilelang_chunk_local_cumsum_kernel(g, cu_seqlens, chunk_indices, g_cumsum)
    else:
        tilelang_chunk_local_cumsum_kernel(g, g_cumsum, num_chunks)

    return g_cumsum
