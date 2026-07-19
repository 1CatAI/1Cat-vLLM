# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from typing import Optional

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_indices


@tilelang.jit(
    # out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        # tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: True,
    },
)
def tilelang_kkt_solve(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(
        bb,
        bc,
        bh,
        bhg,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        k,
        b,
        a,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
        a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

        a16i_row = T.alloc_fragment((4, 16), dtype=accum_dtype)
        a16i_sum = T.alloc_fragment((4, 16), dtype=accum_dtype)

        a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
        a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)

        a32i_fragment = T.alloc_fragment((2, 32, 32), dtype=accum_dtype)
        a32i0_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

        a64_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )

        k_is_ready = T.alloc_barrier(arrive_count=32)
        a_is_ready = T.alloc_barrier(arrive_count=128)

        tx = T.get_thread_binding()

        PRODUCER_NREG = 24
        CONSUMER_NREG = 64

        if tx < 128:
            T.set_max_nreg(CONSUMER_NREG, 1)

            # Load b
            if right <= seq_end_idx:
                for j_s in T.Parallel(block_S):
                    b_shared[j_s] = b[bb, left + j_s, bh]
            else:
                for j_s in T.Parallel(block_S):
                    if left + j_s < seq_end_idx:
                        b_shared[j_s] = b[bb, left + j_s, bh]
                    else:
                        b_shared[j_s] = 0

            T.barrier_wait(k_is_ready, 0)

            # A = K @ K^T
            T.gemm_v1(
                k_shared, k_shared, a64_fragment, transpose_B=True, clear_accum=True
            )

            # A = b * A
            for j_s, j_t in T.Parallel(block_S, block_S):
                a64_fragment[j_s, j_t] *= b_shared[j_s]

            # A = I + StrictLower(A)
            for j_s, j_t in T.Parallel(block_S, block_S):
                if j_s < j_t:
                    a64_fragment[j_s, j_t] = 0
                elif j_s == j_t:
                    a64_fragment[j_s, j_t] = 1

            # Prepare inversion input
            for j_s, j_t in T.Parallel(block_S, block_S):
                if j_s >= 32 and j_t < 32:
                    a32o_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t]
                elif (j_s // 16) == (j_t // 16) + 1:
                    a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a64_fragment[j_s, j_t]
                elif (j_s // 16) == (j_t // 16):
                    a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a64_fragment[j_s, j_t]

            # Diagonal 4x16x16
            T.clear(a16i_row)
            for k_s in T.unroll(1, 16):
                for j_s, k_t in T.Parallel(4, 16):
                    if k_t < k_s:
                        a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
                T.clear(a16i_sum)
                for k_r in T.unroll(k_s):
                    for j_s, k_t in T.Parallel(4, 16):
                        a16i_sum[j_s, k_t] -= (
                            a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                        )
                for j_s, k_t in T.Parallel(4, 16):
                    if k_t < k_s:
                        a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

            # First level 2x16x16
            T.clear(a16o_fragment)
            for k_r in T.unroll(16):
                for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                    a16o_fragment[j_s, k_s, k_t] += (
                        a16i_shared[j_s * 2 + 1, k_s, k_r] * a16o_shared[j_s, k_r, k_t]
                    )
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
            T.clear(a16o_fragment)
            for k_r in T.unroll(16):
                for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                    a16o_fragment[j_s, k_s, k_t] += (
                        a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
                    )
            T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

            # Second level 1x32x32
            for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                if k_s < 16 and k_t >= 16:
                    a32i_fragment[j_s, k_s, k_t] = 0
            for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                if k_s >= 16 and k_t < 16:
                    a32i_fragment[j_s, k_s, k_t] = a16o_shared[j_s, k_s - 16, k_t]
            for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                if k_s // 16 == k_t // 16:
                    a32i_fragment[j_s, k_s, k_t] = a16i_shared[
                        j_s * 2 + k_s // 16, k_s % 16, k_t % 16
                    ]
            for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                if j_s == 0:
                    a32i0_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
                else:
                    a32i1_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
            T.gemm_v1(a32i1_shared, a32o_shared, a32o_fragment, clear_accum=True)
            T.copy(a32o_fragment, a32o_shared)
            T.gemm_v1(a32o_shared, a32i0_shared, a32o_fragment, clear_accum=True)

            # Combine inversion output
            for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                a64_shared[j_s * 32 + k_s, j_s * 32 + k_t] = a32i_fragment[
                    j_s, k_s, k_t
                ]
            for k_s, k_t in T.Parallel(32, 32):
                a64_shared[32 + k_s, k_t] = a32o_fragment[k_s, k_t]
            for k_s, k_t in T.Parallel(32, 32):
                a64_shared[k_s, 32 + k_t] = 0

            T.barrier_arrive(a_is_ready)

        else:
            T.set_max_nreg(PRODUCER_NREG, 0)

            if tx < 128 + 32:
                # Load K
                T.copy(k[bb, left:right, bhg, 0:DK], k_shared)

                T.barrier_arrive(k_is_ready)

            elif tx < 128 + 64:
                T.barrier_wait(a_is_ready, 0)

                # Save A (unmasked)
                if right <= seq_end_idx:
                    T.copy(a64_shared, a[bb, left:right, bh, 0:block_S])

            else:
                T.barrier_wait(a_is_ready, 0)

                # Save A (masked)
                if right > seq_end_idx:
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if left + j_s < seq_end_idx:
                            a[bb, left + j_s, bh, j_t] = a64_shared[j_s, j_t]

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=256) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    else:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=256) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    return tilelang_kkt_solve_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_kkt_solve_sm70(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(
        bb,
        bc,
        bh,
        bhg,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        k,
        b,
        a,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
        a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

        a16i_row = T.alloc_fragment((4, 16), dtype=accum_dtype)
        a16i_sum = T.alloc_fragment((4, 16), dtype=accum_dtype)

        a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
        a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)

        a32i_fragment = T.alloc_fragment((2, 32, 32), dtype=accum_dtype)
        a32i0_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

        a64_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )

        if right <= seq_end_idx:
            T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = b[bb, left + j_s, bh]
        else:
            for j_s, j_d in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    k_shared[j_s, j_d] = k[bb, left + j_s, bhg, j_d]
                else:
                    k_shared[j_s, j_d] = 0
            for j_s in T.Parallel(block_S):
                if left + j_s < seq_end_idx:
                    b_shared[j_s] = b[bb, left + j_s, bh]
                else:
                    b_shared[j_s] = 0
        T.sync_threads()

        T.gemm_v1(k_shared, k_shared, a64_fragment, transpose_B=True, clear_accum=True)

        for j_s, j_t in T.Parallel(block_S, block_S):
            a64_fragment[j_s, j_t] *= b_shared[j_s]

        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a64_fragment[j_s, j_t] = 0
            elif j_s == j_t:
                a64_fragment[j_s, j_t] = 1

        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s >= 32 and j_t < 32:
                a32o_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t]
            elif (j_s // 16) == (j_t // 16) + 1:
                a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a64_fragment[j_s, j_t]
            elif (j_s // 16) == (j_t // 16):
                a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a64_fragment[j_s, j_t]

        T.clear(a16i_row)
        for k_s in T.unroll(1, 16):
            for j_s, k_t in T.Parallel(4, 16):
                if k_t < k_s:
                    a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
            T.clear(a16i_sum)
            for k_r in T.unroll(k_s):
                for j_s, k_t in T.Parallel(4, 16):
                    a16i_sum[j_s, k_t] -= (
                        a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                    )
            for j_s, k_t in T.Parallel(4, 16):
                if k_t < k_s:
                    a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += (
                    a16i_shared[j_s * 2 + 1, k_s, k_r] * a16o_shared[j_s, k_r, k_t]
                )
        for j_s, k_s, k_t in T.Parallel(2, 16, 16):
            a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += (
                    a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
                )
        T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s < 16 and k_t >= 16:
                a32i_fragment[j_s, k_s, k_t] = 0
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s >= 16 and k_t < 16:
                a32i_fragment[j_s, k_s, k_t] = a16o_shared[j_s, k_s - 16, k_t]
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s // 16 == k_t // 16:
                a32i_fragment[j_s, k_s, k_t] = a16i_shared[
                    j_s * 2 + k_s // 16, k_s % 16, k_t % 16
                ]
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if j_s == 0:
                a32i0_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
            else:
                a32i1_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]

        T.clear(a32o_fragment)
        for k_r in T.serial(32):
            for k_s, k_t in T.Parallel(32, 32):
                a32o_fragment[k_s, k_t] += (
                    a32i1_shared[k_s, k_r] * a32o_shared[k_r, k_t]
                )
        T.copy(a32o_fragment, a32o_shared)

        T.clear(a32o_fragment)
        for k_r in T.serial(32):
            for k_s, k_t in T.Parallel(32, 32):
                a32o_fragment[k_s, k_t] += (
                    a32o_shared[k_s, k_r] * a32i0_shared[k_r, k_t]
                )

        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            a64_shared[j_s * 32 + k_s, j_s * 32 + k_t] = a32i_fragment[j_s, k_s, k_t]
        for k_s, k_t in T.Parallel(32, 32):
            a64_shared[32 + k_s, k_t] = a32o_fragment[k_s, k_t]
        for k_s, k_t in T.Parallel(32, 32):
            a64_shared[k_s, 32 + k_t] = 0
        T.sync_threads()

        if right <= seq_end_idx:
            T.copy(a64_shared, a[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a[bb, left + j_s, bh, j_t] = a64_shared[j_s, j_t]

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    else:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    return tilelang_kkt_solve_kernel


def kkt_solve(
    k: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H = b.shape
    assert K == 128
    assert chunk_size == 64

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

    a = torch.empty(
        (batch_size, num_tokens, H, chunk_size), dtype=k.dtype, device=k.device
    )

    target_compute_version = tilelang.contrib.nvcc.get_target_compute_version()
    kkt_solve_jit = (
        tilelang_kkt_solve_sm70
        if target_compute_version in ("7.0", "7.5")
        else tilelang_kkt_solve
    )

    tilelang_kkt_solve_kernel = kkt_solve_jit(
        H,
        Hg,
        K,
        chunk_size,
        qkva_dtype=k.dtype,
        b_dtype=b.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
    )
    if is_varlen:
        tilelang_kkt_solve_kernel(k, b, cu_seqlens, chunk_indices, a)
    else:
        tilelang_kkt_solve_kernel(k, b, a, num_chunks)

    return a
