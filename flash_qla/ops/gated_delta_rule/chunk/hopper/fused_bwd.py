# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_offsets


@tilelang.jit(
    # out_idx=[-5, -4, -3, -2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_fused_chunk_gdr_bwd(
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
    h_dtype,
    o_dtype,
    seqlen_dtype,
    is_varlen,
    use_dht,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
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
    ht_shape = (batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_bwd_kernel(
        do: T.Tensor(o_shape, dtype=o_dtype),
        dht: T.Tensor(ht_shape, dtype=accum_dtype),
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        dq: T.Tensor(v_shape, dtype=qkva_dtype),
        dk: T.Tensor(v_shape, dtype=qkva_dtype),
        dv: T.Tensor(v_shape, dtype=qkva_dtype),
        dg: T.Tensor(g_shape, dtype=g_dtype),
        db: T.Tensor(b_shape, dtype=b_dtype),
        dh0: T.Tensor(h0_shape, dtype=accum_dtype),
    ):
        with T.Kernel(batch_size * H, threads=512) as (bbh,):
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

            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)

            # 2+2+2+2 + 1 + 4 = 13 units
            do_shared = T.alloc_shared((block_S, DV), dtype=o_dtype)
            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, DV), dtype=h_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            # 2 units
            dqkv_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            dg_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            db_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            # 1+1 + 2+2+2 + 4 = 12 units
            tmp_shared_1_1 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_1_2 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_1_3 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_2_1 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_2_2 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_2_3 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_4_1 = T.alloc_shared((DK, DV), dtype=qkva_dtype)

            # CONSUMER_K
            dk_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            odot_fragment_1 = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_fragment_1 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_last_local_1 = T.alloc_fragment((1), dtype=accum_dtype)

            # CONSUMER_A
            mask_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dp_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            da_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dq_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            db_fragment = T.alloc_fragment((block_S), dtype=accum_dtype)
            odot_fragment_2 = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_fragment_2 = T.alloc_fragment((block_S), dtype=accum_dtype)

            # CONSUMER_S
            dh_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            _odot_fragment_3 = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            reduce_fragment = T.alloc_fragment((128, 2), dtype=accum_dtype)
            dg_last_local_3 = T.alloc_fragment((1), dtype=accum_dtype)
            g_last_local_3 = T.alloc_local((1), dtype=accum_dtype)

            # 16 stages
            bar_00 = T.alloc_barrier(arrive_count=448)
            bar_01 = T.alloc_barrier(arrive_count=384)
            bar_02 = T.alloc_barrier(arrive_count=288)
            bar_03 = T.alloc_barrier(arrive_count=256)
            bar_04 = T.alloc_barrier(arrive_count=416)
            bar_05 = T.alloc_barrier(arrive_count=288)
            bar_06 = T.alloc_barrier(arrive_count=256)
            bar_07 = T.alloc_barrier(arrive_count=256)
            bar_08 = T.alloc_barrier(arrive_count=384)
            bar_09 = T.alloc_barrier(arrive_count=256)
            bar_10 = T.alloc_barrier(arrive_count=288)
            bar_11 = T.alloc_barrier(arrive_count=256)
            bar_12 = T.alloc_barrier(arrive_count=128)
            bar_13 = T.alloc_barrier(arrive_count=256)
            bar_14 = T.alloc_barrier(arrive_count=256)
            bar_15 = T.alloc_barrier(arrive_count=256)

            T.annotate_layout(
                {
                    do_shared: tilelang.layout.make_swizzled_layout(do_shared),
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    h_shared: tilelang.layout.make_swizzled_layout(h_shared),
                    dqkv_shared: tilelang.layout.make_swizzled_layout(dqkv_shared),
                    tmp_shared_1_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_1
                    ),
                    tmp_shared_1_2: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_2
                    ),
                    tmp_shared_1_3: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_3
                    ),
                    tmp_shared_2_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_1
                    ),
                    tmp_shared_2_2: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_2
                    ),
                    tmp_shared_2_3: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_3
                    ),
                    tmp_shared_4_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_4_1
                    ),
                }
            )

            # T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_K_NREG = 144
            CONSUMER_A_NREG = 176
            CONSUMER_S_NREG = 160

            # Prefetch the last chunk of data
            T.copy(
                h[batch_idx, chunk_start_idx + num_iters - 1, bh, 0:DK, 0:DV], h_shared
            )
            for j_s, j_k in T.Parallel(block_S, DK):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    q_shared[j_s, j_k] = q[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bhg,
                        j_k,
                    ]
                else:
                    q_shared[j_s, j_k] = 0
            for j_s, j_k in T.Parallel(block_S, DK):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    k_shared[j_s, j_k] = k[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bhg,
                        j_k,
                    ]
                else:
                    k_shared[j_s, j_k] = 0
            for j_s, j_v in T.Parallel(block_S, DV):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    v_shared[j_s, j_v] = v[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bh,
                        j_v,
                    ]
                else:
                    v_shared[j_s, j_v] = 0
            for j_s, j_t in T.Parallel(block_S, block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    a_shared[j_s, j_t] = a[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bh,
                        j_t,
                    ]
                else:
                    a_shared[j_s, j_t] = 0
            for j_s, j_v in T.Parallel(block_S, DV):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    do_shared[j_s, j_v] = do[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bh,
                        j_v,
                    ]
                else:
                    do_shared[j_s, j_v] = 0
            for j_s in T.Parallel(block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    g_shared[j_s] = g[
                        batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh
                    ]
                else:
                    g_shared[j_s] = g[batch_idx, seq_end_idx - 1, bh]
            for j_s in T.Parallel(block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    b_shared[j_s] = b[
                        batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh
                    ]
                else:
                    b_shared[j_s] = 0

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                if use_dht:
                    T.copy(dht[bb, bh, 0:DK, 0:DV], dh_fragment)
                else:
                    T.clear(dh_fragment)
                T.copy(dh_fragment, tmp_shared_4_1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                        )
                    T.barrier_arrive(bar_01)

                    # 01, 02, 03
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    g_last_local_3[0] = g_exp_shared[block_S - 1]
                    # dS0 = g_last * dSt
                    for j_k, j_v in T.Parallel(DK, DV):
                        dh_fragment[j_k, j_v] *= g_last_local_3[0]
                    T.barrier_arrive(bar_04)

                    # 04, 05, 06, 07
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # dg_last += sum(dS0 * S0)
                    T.clear(reduce_fragment)
                    for j_k, j_v in T.Parallel(DK, DV):
                        reduce_fragment[
                            j_k % 64 // 16 * 32 + j_k % 8 * 4 + j_v % 8 // 2, j_v % 2
                        ] += dh_fragment[j_k, j_v] * h_shared[j_k, j_v]
                    T.barrier_arrive(bar_08)
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    T.barrier_wait(bar_09, (i_s + 0) % 2)

                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    T.reduce_sum(
                        T.reshape(reduce_fragment, (128 * 2,)),
                        dg_last_local_3,
                        dim=0,
                        clear=True,
                    )
                    dg_shared[block_S - 1] += dg_last_local_3[0]
                    T.barrier_arrive(bar_11)

                    # 11
                    T.barrier_wait(bar_11, (i_s + 0) % 2)
                    # dS0 += K^T @ dVg
                    T.gemm_v1(
                        tmp_shared_2_2,
                        tmp_shared_2_3,
                        dh_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_12)
                    T.barrier_wait(bar_12, (i_s + 0) % 2)

                    # 13
                    T.barrier_wait(bar_13, (i_s + 0) % 2)
                    # dOg = s * g * dO
                    for j_s, j_v in T.Parallel(block_S, DV):
                        tmp_shared_2_3[j_s, j_v] = (
                            scale * do_shared[j_s, j_v] * g_exp_shared[j_s]
                        )
                    T.barrier_arrive(bar_14)

                    # 14
                    T.barrier_wait(bar_14, (i_s + 0) % 2)
                    # dS0 += Q^T @ dOg
                    T.gemm_v1(
                        tmp_shared_2_1,
                        tmp_shared_2_3,
                        dh_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_15)

                    # 15
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    # S4[1] = dS0
                    T.copy(dh_fragment, tmp_shared_4_1)

                if use_dht:
                    T.copy(dh_fragment, dh0[bb, bh, 0:DK, 0:DV])

            elif tx < 256:
                T.set_max_nreg(CONSUMER_K_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 16 == 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    # S2[S] dK
                    if i_s > 0:
                        T.copy(dk_fragment, dqkv_shared)
                    T.barrier_arrive(bar_01)

                    # 01
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    # dV' = K @ dSt
                    T.gemm_v1(k_shared, tmp_shared_4_1, dv_fragment, clear_accum=True)
                    # dV' = g_last/g * dV'
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    T.barrier_arrive(bar_02)

                    # 02
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    # dV' += Pg^T @ dO
                    T.gemm_v1(
                        tmp_shared_1_1,
                        do_shared,
                        dv_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_03)

                    # 03
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # S2[1] dV'
                    T.copy(dv_fragment, tmp_shared_2_1)
                    T.barrier_arrive(bar_04)

                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # dV = Ag^T @ dV'
                    T.gemm_v1(
                        tmp_shared_1_2,
                        tmp_shared_2_1,
                        dv_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    # S2[S] dV
                    T.copy(dv_fragment, dqkv_shared)
                    T.barrier_arrive(bar_05)

                    # 05
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    # dVg = -g * dV
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] = (
                            -dv_fragment[j_s, j_v] * g_exp_shared[j_s]
                        )
                    # dg += sum(dVg * U)
                    T.copy(tmp_shared_2_3, odot_fragment_1)
                    for j_s, j_v in T.Parallel(block_S, DV):
                        odot_fragment_1[j_s, j_v] *= dv_fragment[j_s, j_v]
                    T.reduce_sum(odot_fragment_1, dg_fragment_1, dim=1, clear=True)
                    T.copy(dg_fragment_1, dg_shared)
                    # S2[3] dVg
                    T.copy(dv_fragment, tmp_shared_2_3)
                    T.barrier_arrive(bar_06)

                    # 06
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    # S2[2] K
                    T.copy(k_shared, odot_fragment_1)
                    T.copy(odot_fragment_1, tmp_shared_2_2)
                    T.barrier_arrive(bar_07)

                    # 07
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    # dK = V' @ dSt^T
                    T.gemm_v1(
                        tmp_shared_2_1,
                        tmp_shared_4_1,
                        dk_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_08)

                    # 08
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    # dK = g_last/g * dK
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dk_fragment[j_s, j_k] *= g_rev_exp_shared[j_s]
                    # dg -= sum(K * dK)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        odot_fragment_1[j_s, j_k] *= -dk_fragment[j_s, j_k]
                    T.reduce_sum(odot_fragment_1, dg_fragment_1, dim=1, clear=True)
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_1[j_s]
                    # dg_last += sum(K * dK)
                    T.reduce_sum(dg_fragment_1, dg_last_local_1, dim=0, clear=True)
                    # Sg[S] dg
                    dg_shared[block_S - 1] -= dg_last_local_1[0]
                    T.barrier_arrive(bar_09)

                    # 09
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # dK += dVg @ S0^T
                    T.gemm_v1(
                        tmp_shared_2_3,
                        h_shared,
                        dk_fragment,
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_10)
                    T.barrier_wait(bar_10, (i_s + 0) % 2)

                    # 12
                    T.barrier_wait(bar_12, (i_s + 0) % 2)
                    # dK += dP^T @ Q
                    T.gemm_v1(
                        tmp_shared_1_1,
                        tmp_shared_2_1,
                        dk_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_13)
                    T.barrier_wait(bar_13, (i_s + 0) % 2)

                    # 15
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    # dK += dAs @ K
                    T.gemm_v1(
                        tmp_shared_1_2, tmp_shared_2_2, dk_fragment, clear_accum=False
                    )

                for j_s, j_k in T.Parallel(block_S, DK):
                    if seq_start_idx + j_s < seq_end_idx:
                        dk[batch_idx, seq_start_idx + j_s, bh, j_k] = dk_fragment[
                            j_s, j_k
                        ]

            elif tx < 384:
                T.set_max_nreg(CONSUMER_A_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    # P = Q @ K^T
                    T.gemm_v1(
                        q_shared,
                        k_shared,
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_01)

                    # 01
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        mask_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            mask_fragment[j_s, j_t] = T.exp2(
                                mask_fragment[j_s, j_t] * 1.442695
                            )
                        else:
                            mask_fragment[j_s, j_t] = 0
                    # Pg = s * P * G
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale
                    # S1[1] Pg
                    T.copy(p_fragment, tmp_shared_1_1)
                    T.barrier_arrive(bar_02)

                    # 02
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    # Ab = Ar * b
                    T.copy(a_shared, a_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[j_t]
                    # Ag = G * Ab
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    # S1[2] Ag
                    T.copy(a_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_03)

                    # 03
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # U = K @ S0
                    T.gemm_v1(k_shared, h_shared, u_fragment, clear_accum=True)
                    T.barrier_arrive(bar_04)

                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # S2[3] U
                    T.copy(u_fragment, tmp_shared_2_3)
                    # W = V - g * U
                    for j_s, j_v in T.Parallel(block_S, DV):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, DV):
                        u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                    # S2[2] W
                    T.copy(u_fragment, tmp_shared_2_2)
                    T.barrier_arrive(bar_05)

                    # 05
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    # dAg = dV' @ W^T
                    T.gemm_v1(
                        tmp_shared_2_1,
                        tmp_shared_2_2,
                        da_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    # V' = Ag @ W
                    T.gemm_v1(
                        tmp_shared_1_2, tmp_shared_2_2, u_fragment, clear_accum=True
                    )
                    # S2[1] V'
                    T.copy(u_fragment, tmp_shared_2_1)
                    T.barrier_arrive(bar_06)

                    # 06
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    # dPg = dO @ V'^T
                    T.gemm_v1(
                        do_shared,
                        tmp_shared_2_1,
                        dp_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_07)

                    # 07
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    # dAb = G * dAg
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    # dg += sum((dPg * P) - (dPg * P)^T)
                    T.copy(tmp_shared_1_1, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= dp_fragment[j_s, j_t]
                    T.copy(p_fragment, tmp_shared_1_1)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] -= tmp_shared_1_1[j_t, j_s]
                    T.reduce_sum(p_fragment, dg_fragment_2, dim=1, clear=True)
                    # dP = s * G * dPg
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= scale
                    # S1[1] dP
                    T.copy(dp_fragment, tmp_shared_1_1)
                    T.barrier_arrive(bar_08)

                    # 08
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    # dQ = dO @ S0^T
                    T.gemm_v1(
                        do_shared,
                        h_shared,
                        dq_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_09)

                    # 09
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # dQ = s * g * dQ
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dq_fragment[j_s, j_k] *= g_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dq_fragment[j_s, j_k] *= scale
                    # S2[1] Q
                    T.copy(q_shared, odot_fragment_2)
                    # dg += sum(Q * dQ)
                    T.copy(odot_fragment_2, tmp_shared_2_1)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        odot_fragment_2[j_s, j_k] *= dq_fragment[j_s, j_k]
                    T.reduce_sum(odot_fragment_2, dg_fragment_2, dim=1, clear=False)
                    T.barrier_arrive(bar_10)

                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    # dQ += dP @ K
                    T.gemm_v1(
                        tmp_shared_1_1, tmp_shared_2_2, dq_fragment, clear_accum=False
                    )
                    # S2[S] dQ
                    T.copy(dq_fragment, dqkv_shared)
                    T.barrier_arrive(bar_11)

                    # 11, 12
                    T.barrier_wait(bar_11, (i_s + 0) % 2)
                    # dAb * Ar
                    T.copy(a_shared, a_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= da_fragment[j_s, j_t]
                    T.copy(a_fragment, tmp_shared_1_3)
                    # dAb * Ab [ = G * dAg * Ab ]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[j_t]
                    # dg += sum((dAb * Ab) - (dAb * Ab)^T)
                    T.copy(a_fragment, tmp_shared_1_2)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] -= tmp_shared_1_2[j_t, j_s]
                    T.reduce_sum(a_fragment, dg_fragment_2, dim=1, clear=False)
                    # Sg[S] dg
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_2[j_s]
                    # db = sum((dAb * Ar)^T)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = tmp_shared_1_3[j_t, j_s]
                    T.reduce_sum(a_fragment, db_fragment, dim=1, clear=True)
                    # dAr = dAb * b
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= b_shared[j_t]
                    # S1[2] dAr
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_13)

                    # 13
                    T.barrier_wait(bar_13, (i_s + 0) % 2)
                    # dA = -Ar^T @ dAr @ Ar^T
                    T.gemm_v1(
                        a_shared,
                        tmp_shared_1_2,
                        da_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.gemm_v1(
                        tmp_shared_1_2,
                        a_shared,
                        da_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    # At = K @ K^T
                    T.gemm_v1(
                        tmp_shared_2_2,
                        tmp_shared_2_2,
                        a_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_14)

                    # 14
                    T.barrier_wait(bar_14, (i_s + 0) % 2)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s <= j_t:
                            da_fragment[j_s, j_t] = 0
                        else:
                            da_fragment[j_s, j_t] = -da_fragment[j_s, j_t]
                    # db += sum(dA * At)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= da_fragment[j_s, j_t]
                    T.reduce_sum(a_fragment, db_fragment, dim=1, clear=False)
                    T.copy(db_fragment, db_shared)
                    # dAt = b * dA
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= b_shared[j_s]
                    # dAs = dAt + dAt^T
                    T.copy(da_fragment, tmp_shared_1_2)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] += tmp_shared_1_2[j_t, j_s]
                    # S1[1] dAs
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_15)
                    T.barrier_wait(bar_15, (i_s + 0) % 2)

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    for i_s in T.serial(num_iters - 1):
                        chunk_idx = num_iters - i_s - 2
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, (i_s + 0) % 2)

                        T.barrier_wait(bar_03, (i_s + 0) % 2)
                        for j_s in T.Parallel(block_S):
                            g_shared[j_s] = g[batch_idx, left + j_s, bh]

                        T.barrier_wait(bar_05, (i_s + 0) % 2)
                        T.copy(v[batch_idx, left:right, bh, 0:DV], v_shared)

                        T.barrier_wait(bar_07, (i_s + 0) % 2)
                        T.copy(k[batch_idx, left:right, bhg, 0:DK], k_shared)

                        T.barrier_wait(bar_10, (i_s + 0) % 2)
                        T.copy(q[batch_idx, left:right, bhg, 0:DK], q_shared)

                    if num_iters > 0:
                        T.barrier_arrive(bar_00)

                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        left = seq_start_idx + (num_iters - i_s - 1) * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, (i_s + 0) % 2)

                        T.barrier_wait(bar_01, (i_s + 0) % 2)
                        if i_s == 1:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + block_S + j_s < seq_end_idx:
                                    dk[batch_idx, left + block_S + j_s, bh, j_k] = (
                                        dqkv_shared[j_s, j_k]
                                    )
                        elif i_s > 1:
                            T.copy(
                                dqkv_shared,
                                dk[
                                    batch_idx,
                                    left + block_S : right + block_S,
                                    bh,
                                    0:DK,
                                ],
                            )
                        T.barrier_arrive(bar_04)
                        T.barrier_wait(bar_04, (i_s + 0) % 2)

                        T.barrier_wait(bar_05, (i_s + 0) % 2)
                        if i_s == 0:
                            for j_s, j_v in T.Parallel(block_S, DV):
                                if left + j_s < seq_end_idx:
                                    dv[batch_idx, left + j_s, bh, j_v] = dqkv_shared[
                                        j_s, j_v
                                    ]
                        else:
                            T.copy(dqkv_shared, dv[batch_idx, left:right, bh, 0:DV])
                        T.barrier_arrive(bar_10)
                        T.barrier_wait(bar_10, (i_s + 0) % 2)

                        T.barrier_wait(bar_11, (i_s + 0) % 2)
                        if i_s == 0:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    dq[batch_idx, left + j_s, bh, j_k] = dqkv_shared[
                                        j_s, j_k
                                    ]
                        else:
                            T.copy(dqkv_shared, dq[batch_idx, left:right, bh, 0:DK])

                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters - 1):
                        chunk_idx = num_iters - i_s - 2
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_02)
                        T.barrier_wait(bar_02, (i_s + 0) % 2)

                        T.barrier_wait(bar_10, (i_s + 0) % 2)
                        T.copy(
                            h[batch_idx, chunk_start_idx + chunk_idx, bh, 0:DK, 0:DV],
                            h_shared,
                        )

                        T.barrier_wait(bar_14, (i_s + 0) % 2)
                        T.copy(a[batch_idx, left:right, bh, 0:block_S], a_shared)

                        T.copy(do[batch_idx, left:right, bh, 0:DV], do_shared)

                        T.barrier_wait(bar_15, (i_s + 0) % 2)
                        for j_s in T.Parallel(block_S):
                            b_shared[j_s] = b[batch_idx, left + j_s, bh]

                    if num_iters > 0:
                        T.barrier_wait(bar_00, (num_iters - 1) % 2)
                        T.barrier_arrive(bar_02)

                else:
                    for i_s in T.serial(num_iters):
                        left = seq_start_idx + (num_iters - i_s - 1) * block_S

                        T.barrier_arrive(bar_05)
                        T.barrier_wait(bar_05, (i_s + 0) % 2)

                        T.barrier_wait(bar_15, (i_s + 0) % 2)

                        if i_s == 0:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    dg[batch_idx, left + j_s, bh] = dg_shared[j_s]
                            if (seq_end_idx - seq_start_idx) % block_S > 0:
                                dg[batch_idx, seq_end_idx - 1, bh] += dg_shared[
                                    block_S - 1
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                dg[batch_idx, left + j_s, bh] = dg_shared[j_s]

                        if i_s == 0:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    db[batch_idx, left + j_s, bh] = db_shared[j_s]
                        else:
                            for j_s in T.Parallel(block_S):
                                db[batch_idx, left + j_s, bh] = db_shared[j_s]

    return tilelang_fused_chunk_gdr_bwd_kernel


def fused_gdr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    h: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 64

    if cu_seqlens is None:
        real_batch_size = batch_size
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        chunk_offsets = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        is_varlen = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        is_varlen = True

    use_dht = dht is not None
    if dht is None:
        dht = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )
    dq = torch.empty_like(v)
    dk = torch.empty_like(v)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    db = torch.empty_like(b)
    dh0 = torch.empty_like(dht)

    tilelang_fused_chunk_gdr_bwd_kernel = tilelang_fused_chunk_gdr_bwd(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h_dtype=h.dtype,
        o_dtype=do.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
        use_dht=use_dht,
    )
    tilelang_fused_chunk_gdr_bwd_kernel(
        do,
        dht,
        q,
        k,
        v,
        a,
        g,
        b,
        h,
        cu_seqlens,
        chunk_offsets,
        dq,
        dk,
        dv,
        dg,
        db,
        dh0,
    )

    if not use_dht:
        dh0 = None

    return dq, dk, dv, dg, db, dh0
