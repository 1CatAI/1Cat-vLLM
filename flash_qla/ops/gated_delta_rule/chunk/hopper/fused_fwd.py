# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_offsets


MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count
TARGET_NUM_CTAS = int(MULTI_PROCESSOR_COUNT * 0.7)


@tilelang.jit(
    # out_idx=[-3, -2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def tilelang_fused_chunk_gdr_fwd(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    h_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    store_h,
    store_o,
    is_varlen,
    is_cp,
    state_layout_vlk=False,
    block_DV=128,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    raw_batch_size = T.dynamic("raw_batch_size")
    block_S = chunk_size

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        o_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        h_shape = (1, num_chunks, H, DK, DV)
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        o_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        h_shape = (batch_size, num_chunks, H, DK, DV)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (raw_batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        cp_seq_map: T.Tensor([batch_size], dtype=seqlen_dtype),
        raw_cu_seqlens: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=512) as (bbhv,):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            seq_split_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")
            chunk_split_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            raw_batch_idx = T.alloc_var("int32")
            raw_seq_end_idx = T.alloc_var("int32")
            need_store_final_state = T.alloc_var("bool")
            raw_batch_idx = cp_seq_map[bb] if is_cp else bb
            raw_seq_end_idx = (
                raw_cu_seqlens[raw_batch_idx + 1] if is_cp else seq_end_idx
            )
            need_store_final_state = store_final_state & (
                raw_seq_end_idx == seq_end_idx
            )

            num_iters = T.alloc_var("int32")
            num_unmasked_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            num_unmasked_iters = (seq_end_idx - seq_start_idx) // block_S

            q_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((2, block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((2, block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")

            o_shared = T.alloc_shared((block_S, block_DV), dtype=o_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            data_is_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_is_free = T.alloc_barrier(arrive_count=[384] * 2)

            bar_o = T.alloc_barrier(arrive_count=128)
            bar_0 = T.alloc_barrier(arrive_count=416)
            bar_1 = T.alloc_barrier(arrive_count=256)
            _bar_2 = T.alloc_barrier(arrive_count=128)
            bar_3 = T.alloc_barrier(arrive_count=128)
            bar_4 = T.alloc_barrier(arrive_count=128)
            bar_5 = T.alloc_barrier(arrive_count=416)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 32
            CONSUMER_V_NREG = 128
            CONSUMER_S_NREG = 160
            CONSUMER_O_NREG = 128

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                # Initialize S
                if use_initial_state:
                    T.copy(
                        h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                        h_fragment,
                    )
                else:
                    T.clear(h_fragment)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE 0]
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 0] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # S4[S] S
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    # [STAGE 0] 2, 3, 4
                    T.barrier_wait(bar_1, i_s % 2)
                    # S = g_last * S
                    g_last_local[0] = g_exp_shared[block_S - 1]
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] *= g_last_local[0]
                    T.barrier_arrive(bar_5)

                    # [STAGE 0] 5
                    T.barrier_wait(bar_5, i_s % 2)
                    # S += K^T @ V'
                    T.gemm_v1(
                        k_shared[i_s % 2, :, :],
                        vn_shared,
                        h_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )

                    T.barrier_arrive(data_is_free[i_s % 2])

                # Store final S
                if need_store_final_state:
                    T.copy(
                        h_fragment,
                        ht[
                            raw_batch_idx, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV
                        ],
                    )

            elif tx < 256:
                T.set_max_nreg(CONSUMER_V_NREG, 1)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE 0]
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 0] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # Precompute g, g_last/g
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[i_s % 2, j_s] * 1.442695)
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.if_then_else(
                            seq_start_idx + i_s * block_S + j_s < seq_end_idx,
                            T.exp2(
                                (
                                    g_shared[i_s % 2, block_S - 1]
                                    - g_shared[i_s % 2, j_s]
                                )
                                * 1.442695
                            ),
                            0.0,
                        )
                    T.barrier_arrive(bar_1)

                    # [STAGE 0] 1
                    T.barrier_wait(bar_1, i_s % 2)
                    # U = K @ S
                    T.gemm_v1(
                        k_shared[i_s % 2, :, :], h_shared, u_fragment, clear_accum=True
                    )

                    # [STAGE 0] 2
                    # W = V - g * U
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        u_fragment[j_s, j_v] += v_shared[i_s % 2, j_s, j_v]
                    # S2[V] W
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_shared[i_s % 2, j_s, j_v] = u_fragment[j_s, j_v]

                    # [STAGE 0] 3
                    T.barrier_wait(bar_3, i_s % 2)
                    # Vd = Ag @ W
                    T.gemm_v1(
                        a_shared[i_s % 2, :, :],
                        v_shared[i_s % 2, :, :],
                        v_fragment,
                        clear_accum=True,
                    )
                    # S2[2] Vd
                    T.copy(v_fragment, vd_shared)
                    T.barrier_arrive(bar_4)

                    # [STAGE 0] 4
                    # V' = g_last/g Vd
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    # S2[1] V'
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_5)

                    T.barrier_wait(bar_5, i_s % 2)

                    T.barrier_arrive(data_is_free[i_s % 2])

            elif tx < 384:
                T.set_max_nreg(CONSUMER_O_NREG, 1)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE 0]
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 0] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # P = Q K^T
                    T.gemm_v1(
                        q_shared[i_s % 2, :, :],
                        k_shared[i_s % 2, :, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )

                    # [STAGE 0] 1
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = (
                            g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            g_fragment[j_s, j_t] = T.exp2(
                                g_fragment[j_s, j_t] * 1.442695
                            )
                        else:
                            g_fragment[j_s, j_t] = 0
                    # Ag = G * Ar * b
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = a_shared[i_s % 2, j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[i_s % 2, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_shared[i_s % 2, j_s, j_t] = a_fragment[j_s, j_t]

                    # [STAGE 0] 2
                    T.barrier_wait(bar_1, i_s % 2)
                    # O = Q @ S
                    T.gemm_v1(
                        q_shared[i_s % 2, :, :], h_shared, o_fragment, clear_accum=True
                    )

                    # [STAGE 0] 3
                    # Pg = s * G * P
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    # S1[1] Pg
                    T.copy(p_fragment, p_shared)
                    T.barrier_arrive(bar_3)
                    # O = s * g * O
                    for j_s, j_k in T.Parallel(block_S, DK):
                        o_fragment[j_s, j_k] *= scale * g_exp_shared[j_s]

                    # [STAGE 0] 4
                    T.barrier_wait(bar_4, i_s % 2)
                    # O += Pg @ Vd
                    T.gemm_v1(p_shared, vd_shared, o_fragment, clear_accum=False)
                    T.barrier_arrive(bar_5)

                    # [STAGE 0] 5
                    T.barrier_wait(bar_5, i_s % 2)
                    # S2[S] O
                    T.copy(o_fragment, o_shared)

                    T.barrier_arrive(data_is_free[i_s % 2])

                T.barrier_arrive(bar_o)

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load Q
                        T.copy(
                            q[batch_idx, left:right, bhg, 0:DK], q_shared[i_s % 2, :, :]
                        )
                        # Load K
                        T.copy(
                            k[batch_idx, left:right, bhg, 0:DK], k_shared[i_s % 2, :, :]
                        )

                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load V
                        T.copy(
                            v[
                                batch_idx,
                                left:right,
                                bh,
                                bv * block_DV : (bv + 1) * block_DV,
                            ],
                            v_shared[i_s % 2, :, :],
                        )
                        # Load beta
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % 2, j_s] = b[batch_idx, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    b_shared[i_s % 2, j_s] = b[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    b_shared[i_s % 2, j_s] = 0

                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load A
                        T.copy(
                            a[batch_idx, left:right, bh, 0:block_S],
                            a_shared[i_s % 2, :, :],
                        )
                        # Load gamma
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % 2, j_s] = g[batch_idx, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    g_shared[i_s % 2, j_s] = g[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    g_shared[i_s % 2, j_s] = g[
                                        batch_idx, seq_end_idx - 1, bh
                                    ]

                        T.barrier_arrive(data_is_ready[i_s % 2])

                else:
                    for i_s in T.serial(num_unmasked_iters):
                        right = seq_start_idx + i_s * block_S
                        left = right - block_S

                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, i_s % 2)
                        # Store O
                        if i_s > 0 and store_o:
                            T.copy(
                                o_shared,
                                o[
                                    batch_idx,
                                    left:right,
                                    bh,
                                    bv * block_DV : (bv + 1) * block_DV,
                                ],
                            )
                        T.barrier_arrive(bar_5)

                        T.barrier_wait(bar_1, i_s % 2)
                        # Store S
                        if store_h:
                            T.copy(
                                h_shared,
                                h[
                                    batch_idx,
                                    chunk_start_idx + i_s,
                                    bh,
                                    0:DK,
                                    bv * block_DV : (bv + 1) * block_DV,
                                ],
                            )

                    if num_unmasked_iters < num_iters:
                        seq_split_idx = seq_start_idx + num_unmasked_iters * block_S
                        chunk_split_idx = chunk_start_idx + num_unmasked_iters

                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, num_unmasked_iters % 2)
                        # Store O
                        if num_unmasked_iters > 0 and store_o:
                            T.copy(
                                o_shared,
                                o[
                                    batch_idx,
                                    seq_split_idx - block_S : seq_split_idx,
                                    bh,
                                    bv * block_DV : (bv + 1) * block_DV,
                                ],
                            )
                        T.barrier_arrive(bar_5)

                        T.barrier_wait(bar_1, num_unmasked_iters % 2)
                        # Store S
                        if store_h:
                            T.copy(
                                h_shared,
                                h[
                                    batch_idx,
                                    chunk_split_idx,
                                    bh,
                                    0:DK,
                                    bv * block_DV : (bv + 1) * block_DV,
                                ],
                            )

                    seq_split_idx = seq_start_idx + (num_iters - 1) * block_S

                    # Store O
                    T.barrier_wait(bar_o, 0)
                    if store_o:
                        for j_s, j_v in T.Parallel(block_S, block_DV):
                            with T.If(seq_split_idx + j_s < seq_end_idx):
                                with T.Then():
                                    o[
                                        batch_idx,
                                        seq_split_idx + j_s,
                                        bh,
                                        bv * block_DV + j_v,
                                    ] = o_shared[j_s, j_v]

    return tilelang_fused_chunk_gdr_fwd_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_gdr_fwd_sm70(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    h_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    store_h,
    store_o,
    is_varlen,
    is_cp,
    state_layout_vlk,
    indexed_state=False,
    block_DV=128,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    raw_batch_size = T.dynamic("raw_batch_size")
    state_cache_size = T.dynamic("state_cache_size")
    block_S = chunk_size

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        o_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        h_shape = (1, num_chunks, H, DK, DV)
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        o_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        h_shape = (batch_size, num_chunks, H, DK, DV)
    h0_batch = state_cache_size if indexed_state else batch_size
    h0_shape = (h0_batch, H, DK, DV)
    ht_shape = (raw_batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        cp_seq_map: T.Tensor([batch_size], dtype=seqlen_dtype),
        raw_cu_seqlens: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
        state_indices: T.Tensor([batch_size], dtype="int32"),
        has_initial_state: T.Tensor([batch_size], dtype="bool"),
        o: T.Tensor(o_shape, dtype=o_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=128) as (bbhv,):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            raw_batch_idx = T.alloc_var("int32")
            raw_seq_end_idx = T.alloc_var("int32")
            need_store_final_state = T.alloc_var("bool")
            raw_batch_idx = cp_seq_map[bb] if is_cp else bb
            raw_seq_end_idx = (
                raw_cu_seqlens[raw_batch_idx + 1] if is_cp else seq_end_idx
            )
            need_store_final_state = store_final_state & (
                raw_seq_end_idx == seq_end_idx
            )

            state_slot = T.alloc_var("int32")
            state_slot = state_indices[bb] if indexed_state else bb

            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            kt_shared = T.alloc_shared((DK, block_S), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            if use_initial_state:
                if indexed_state and state_layout_vlk:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] = T.if_then_else(
                            has_initial_state[bb],
                            h0[
                                state_slot,
                                bh,
                                bv * block_DV + j_v,
                                j_k,
                            ],
                            0.0,
                        )
                elif indexed_state:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] = T.if_then_else(
                            has_initial_state[bb],
                            h0[
                                state_slot,
                                bh,
                                j_k,
                                bv * block_DV + j_v,
                            ],
                            0.0,
                        )
                elif state_layout_vlk:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] = h0[
                            bb,
                            bh,
                            bv * block_DV + j_v,
                            j_k,
                        ]
                else:
                    T.copy(
                        h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                        h_fragment,
                    )
            else:
                T.clear(h_fragment)

            for i_s in T.serial(num_iters):
                left = seq_start_idx + i_s * block_S
                right = left + block_S

                if right <= seq_end_idx:
                    T.copy(q[batch_idx, left:right, bhg, 0:DK], q_shared)
                    T.copy(k[batch_idx, left:right, bhg, 0:DK], k_shared)
                    T.copy(
                        v[
                            batch_idx,
                            left:right,
                            bh,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                        v_shared,
                    )
                    T.copy(a[batch_idx, left:right, bh, 0:block_S], a_shared)
                    for j_s in T.Parallel(block_S):
                        g_shared[j_s] = g[batch_idx, left + j_s, bh]
                    for j_s in T.Parallel(block_S):
                        b_shared[j_s] = b[batch_idx, left + j_s, bh]
                else:
                    for j_s, j_k in T.Parallel(block_S, DK):
                        if left + j_s < seq_end_idx:
                            q_shared[j_s, j_k] = q[batch_idx, left + j_s, bhg, j_k]
                            k_shared[j_s, j_k] = k[batch_idx, left + j_s, bhg, j_k]
                        else:
                            q_shared[j_s, j_k] = 0
                            k_shared[j_s, j_k] = 0
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        if left + j_s < seq_end_idx:
                            v_shared[j_s, j_v] = v[
                                batch_idx,
                                left + j_s,
                                bh,
                                bv * block_DV + j_v,
                            ]
                        else:
                            v_shared[j_s, j_v] = 0
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if left + j_s < seq_end_idx:
                            a_shared[j_s, j_t] = a[batch_idx, left + j_s, bh, j_t]
                        else:
                            a_shared[j_s, j_t] = 0
                    for j_s in T.Parallel(block_S):
                        if left + j_s < seq_end_idx:
                            g_shared[j_s] = g[batch_idx, left + j_s, bh]
                            b_shared[j_s] = b[batch_idx, left + j_s, bh]
                        else:
                            g_shared[j_s] = g[batch_idx, seq_end_idx - 1, bh]
                            b_shared[j_s] = 0
                T.sync_threads()
                for j_s, j_k in T.Parallel(block_S, DK):
                    kt_shared[j_k, j_s] = k_shared[j_s, j_k]
                T.sync_threads()

                T.copy(h_fragment, h_shared)
                if store_h:
                    T.copy(
                        h_shared,
                        h[
                            batch_idx,
                            chunk_start_idx + i_s,
                            bh,
                            0:DK,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                    )
                T.sync_threads()

                for j_s in T.Parallel(block_S):
                    g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                for j_s in T.Parallel(block_S):
                    g_rev_exp_shared[j_s] = T.if_then_else(
                        left + j_s < seq_end_idx,
                        T.exp2((g_shared[block_S - 1] - g_shared[j_s]) * 1.442695),
                        0.0,
                    )
                T.sync_threads()

                T.gemm_v1(
                    q_shared, k_shared, p_fragment, transpose_B=True, clear_accum=True
                )

                for j_s, j_t in T.Parallel(block_S, block_S):
                    g_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s >= j_t:
                        g_fragment[j_s, j_t] = T.exp2(g_fragment[j_s, j_t] * 1.442695)
                    else:
                        g_fragment[j_s, j_t] = 0
                for j_s, j_t in T.Parallel(block_S, block_S):
                    a_fragment[j_s, j_t] = a_shared[j_s, j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    a_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    a_fragment[j_s, j_t] *= b_shared[j_t]
                T.copy(a_fragment, a_shared)
                T.sync_threads()

                T.gemm_v1(k_shared, h_shared, u_fragment, clear_accum=True)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                T.copy(u_fragment, v_shared)
                T.sync_threads()

                T.gemm_v1(a_shared, v_shared, v_fragment, clear_accum=True)
                T.copy(v_fragment, vd_shared)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                T.copy(v_fragment, vn_shared)
                T.sync_threads()

                T.gemm_v1(q_shared, h_shared, o_fragment, clear_accum=True)
                for j_s, j_t in T.Parallel(block_S, block_S):
                    p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                T.copy(p_fragment, p_shared)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.sync_threads()

                T.gemm_v1(p_shared, vd_shared, o_fragment, clear_accum=False)

                if store_o:
                    if right <= seq_end_idx:
                        T.copy(
                            o_fragment,
                            o[
                                batch_idx,
                                left:right,
                                bh,
                                bv * block_DV : (bv + 1) * block_DV,
                            ],
                        )
                    else:
                        for j_s, j_v in T.Parallel(block_S, block_DV):
                            if left + j_s < seq_end_idx:
                                o[
                                    batch_idx,
                                    left + j_s,
                                    bh,
                                    bv * block_DV + j_v,
                                ] = o_fragment[j_s, j_v]

                g_last_local[0] = g_exp_shared[block_S - 1]
                for j_k, j_v in T.Parallel(DK, block_DV):
                    h_fragment[j_k, j_v] *= g_last_local[0]
                T.gemm_v1(kt_shared, vn_shared, h_fragment, clear_accum=False)

            if need_store_final_state:
                if indexed_state and state_layout_vlk:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h0[state_slot, bh, bv * block_DV + j_v, j_k] = h_fragment[
                            j_k, j_v
                        ]
                elif indexed_state:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h0[state_slot, bh, j_k, bv * block_DV + j_v] = h_fragment[
                            j_k, j_v
                        ]
                elif state_layout_vlk:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        ht[raw_batch_idx, bh, bv * block_DV + j_v, j_k] = h_fragment[
                            j_k, j_v
                        ]
                else:
                    T.copy(
                        h_fragment,
                        ht[
                            raw_batch_idx,
                            bh,
                            0:DK,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                    )

    return tilelang_fused_chunk_gdr_fwd_kernel


def fused_gdr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    output_o: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cp_seq_map: torch.LongTensor | None = None,
    raw_cu_seqlens: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    chunk_size: int = 64,
    state_layout_vlk: bool = False,
    sm70_original: bool = False,
    output: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = False,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 64

    if cu_seqlens is None:
        real_batch_size = batch_size
        num_chunks = tilelang.cdiv(num_tokens, chunk_size) if output_h else 0
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        chunk_offsets = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        seqlen_dtype = torch.int32
        is_varlen = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        if chunk_offsets is None:
            chunk_offsets, prepared_num_chunks = prepare_chunk_offsets(
                cu_seqlens,
                chunk_size,
            )
        else:
            if chunk_offsets.shape != cu_seqlens.shape:
                raise ValueError("chunk_offsets must have the same shape as cu_seqlens")
            if chunk_offsets.device != cu_seqlens.device:
                raise ValueError(
                    "chunk_offsets must be on the same device as cu_seqlens"
                )
            chunk_offsets = chunk_offsets.contiguous()
            prepared_num_chunks = int(chunk_offsets[-1].item()) if output_h else 0
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        num_chunks = prepared_num_chunks if output_h else 0
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    if cp_seq_map is None:
        cp_seq_map = torch.empty(
            (real_batch_size,), dtype=seqlen_dtype, device=k.device
        )
        is_cp = False
    else:
        is_cp = True

    indexed_state = state_indices is not None
    if indexed_state:
        if not sm70_original:
            raise ValueError(
                "indexed state-cache mode is only supported by SM70 original TileLang."
            )
        if not state_layout_vlk:
            raise ValueError("indexed state-cache mode requires vLLM VLK state layout.")
        if cu_seqlens is None:
            raise ValueError("indexed state-cache mode requires cu_seqlens.")
        if initial_state is None:
            raise ValueError("indexed state-cache mode requires a state cache tensor.")
        if not inplace_final_state:
            raise ValueError(
                "indexed state-cache mode requires inplace_final_state=True."
            )
        if state_indices.ndim != 1 or state_indices.shape[0] != real_batch_size:
            raise ValueError("state_indices must have shape [num_sequences].")
        if state_indices.device != k.device:
            raise ValueError("state_indices must be on the same device as q/k/v.")
        state_indices_arg = state_indices.to(dtype=torch.int32).contiguous()
        if has_initial_state is None:
            has_initial_state_arg = torch.ones(
                (real_batch_size,), dtype=torch.bool, device=k.device
            )
        else:
            if has_initial_state.shape != (real_batch_size,):
                raise ValueError("has_initial_state must have shape [num_sequences].")
            if has_initial_state.device != k.device:
                raise ValueError(
                    "has_initial_state must be on the same device as q/k/v."
                )
            if has_initial_state.dtype != torch.bool:
                raise ValueError("has_initial_state must be bool.")
            has_initial_state_arg = has_initial_state.contiguous()
    else:
        state_indices_arg = torch.empty(
            (real_batch_size,), dtype=torch.int32, device=k.device
        )
        has_initial_state_arg = torch.empty(
            (real_batch_size,), dtype=torch.bool, device=k.device
        )

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )
    h = torch.empty((batch_size, num_chunks, H, K, V), dtype=k.dtype, device=k.device)
    if raw_cu_seqlens is None:
        raw_cu_seqlens = torch.empty(
            (real_batch_size + 1,), dtype=seqlen_dtype, device=k.device
        )
        final_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )
    else:
        final_state = torch.empty(
            (raw_cu_seqlens.shape[0] - 1, H, K, V), dtype=torch.float32, device=k.device
        )
    if output is None:
        o = torch.empty_like(v)
    else:
        if output.shape != v.shape:
            raise ValueError(
                f"output must have shape {tuple(v.shape)}, got {tuple(output.shape)}"
            )
        if output.dtype != v.dtype:
            raise ValueError(f"output must have dtype {v.dtype}, got {output.dtype}")
        if output.device != v.device:
            raise ValueError(
                f"output must be on device {v.device}, got {output.device}"
            )
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")
        o = output

    target_compute_version = tilelang.contrib.nvcc.get_target_compute_version()
    enable_sm70_original = os.getenv("FLASH_QLA_SM70_ORIGINAL_FUSED_FWD", "0") == "1"
    use_sm70_original_fused = target_compute_version in ("7.0", "7.5") and (
        enable_sm70_original or sm70_original
    )

    grid_size = real_batch_size * H
    if use_sm70_original_fused:
        block_DV = 32
    elif grid_size >= TARGET_NUM_CTAS:
        block_DV = 128
    elif grid_size * 2 >= TARGET_NUM_CTAS:
        block_DV = 64
    else:
        block_DV = 32

    fused_gdr_fwd_jit = (
        tilelang_fused_chunk_gdr_fwd_sm70
        if use_sm70_original_fused
        else tilelang_fused_chunk_gdr_fwd
    )

    tilelang_fused_chunk_gdr_fwd_kernel = fused_gdr_fwd_jit(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        h_dtype=h.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state or inplace_final_state,
        store_h=output_h,
        store_o=output_o,
        is_varlen=is_varlen,
        is_cp=is_cp,
        state_layout_vlk=state_layout_vlk,
        **({"indexed_state": indexed_state} if use_sm70_original_fused else {}),
        block_DV=block_DV,
    )
    if use_sm70_original_fused:
        tilelang_fused_chunk_gdr_fwd_kernel(
            q,
            k,
            v,
            a,
            g,
            b,
            initial_state,
            cu_seqlens,
            chunk_offsets,
            cp_seq_map,
            raw_cu_seqlens,
            state_indices_arg,
            has_initial_state_arg,
            o,
            h,
            final_state,
        )
    else:
        tilelang_fused_chunk_gdr_fwd_kernel(
            q,
            k,
            v,
            a,
            g,
            b,
            initial_state,
            cu_seqlens,
            chunk_offsets,
            cp_seq_map,
            raw_cu_seqlens,
            o,
            h,
            final_state,
        )

    if inplace_final_state:
        final_state = None
    elif not output_final_state:
        final_state = None
    if not output_h:
        h = None
    if not output_o:
        o = None

    return o, h, final_state
