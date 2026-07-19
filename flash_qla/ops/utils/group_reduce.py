# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    # out_idx=[-1],
)
def tilelang_group_reduce_vector(
    H,
    Hg,
    DK,
    accum_dtype,
    qkva_dtype,
    block_size: int = 16,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")

    group_size = H // Hg

    buffer_shape = (batch_size, num_tokens, H, DK)
    dqk_shape = (batch_size, num_tokens, Hg, DK)

    @T.prim_func
    def tilelang_group_reduce_vector_kernel(
        buffer: T.Tensor(buffer_shape, dtype=qkva_dtype),
        result: T.Tensor(dqk_shape, dtype=qkva_dtype),
    ):
        with T.Kernel(
            tilelang.cdiv(num_tokens, block_size), Hg, batch_size, threads=128
        ) as (bt, bhg, bb):
            buffer_fragment = T.alloc_fragment((block_size, DK), dtype=accum_dtype)
            result_fragment = T.alloc_fragment((block_size, DK), dtype=accum_dtype)

            T.clear(result_fragment)
            for i in T.serial(group_size):
                T.copy(
                    buffer[
                        bb,
                        bt * block_size : (bt + 1) * block_size,
                        bhg * group_size + i,
                        0:DK,
                    ],
                    buffer_fragment,
                )
                for j, k in T.Parallel(block_size, DK):
                    result_fragment[j, k] += buffer_fragment[j, k]
            T.copy(
                result_fragment,
                result[bb, bt * block_size : (bt + 1) * block_size, bhg, 0:DK],
            )

    return tilelang_group_reduce_vector_kernel


def group_reduce_vector(
    buffer: torch.Tensor,
    Hg: int,
):
    batch_size, num_tokens, H, K = buffer.shape

    result = torch.empty(
        (batch_size, num_tokens, Hg, K), dtype=buffer.dtype, device=buffer.device
    )

    tilelang_group_reduce_vector_kernel = tilelang_group_reduce_vector(
        H,
        Hg,
        K,
        qkva_dtype=buffer.dtype,
        accum_dtype="float32",
    )
    tilelang_group_reduce_vector_kernel(buffer, result)

    return result
