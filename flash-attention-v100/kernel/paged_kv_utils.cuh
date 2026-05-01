#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__
size_t resolve_paged_kv_offset(
    const int token_idx,
    const int page_block_size,
    const int* __restrict__ block_table,
    const int kv_head_offset,
    const int num_kv_heads,
    const int head_dim
) {

    const int virtual_page_idx = token_idx / page_block_size;
    const int page_offset = token_idx % page_block_size;

    const int physical_block_idx = block_table[virtual_page_idx];

    const size_t physical_offset =
        (size_t)physical_block_idx * page_block_size * num_kv_heads * head_dim
        + (size_t)page_offset * num_kv_heads * head_dim
        + (size_t)kv_head_offset;

    return physical_offset;
}

template<int D>
__device__ __forceinline__
void load_kv_tile_paged(
    const __half* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int start_token_idx,
    const int num_tokens,
    const int kv_head_idx,
    const int num_kv_heads,
    const int head_dim,
    const int page_block_size,
    __half* smem_dst,
    const int smem_stride,
    const int tid,
    const int num_threads
) {
    constexpr int PER_UINT4 = 8;
    const int d_stride_uint4 = (D + PER_UINT4 - 1) / PER_UINT4;
    const int smem_stride_uint4 = (smem_stride + PER_UINT4 - 1) / PER_UINT4;

    const int kv_head_offset = kv_head_idx * head_dim;

    uint4* smem_vec = reinterpret_cast<uint4*>(smem_dst);

    #pragma unroll 2
    for (int idx = tid; idx < (num_tokens * d_stride_uint4); idx += num_threads) {
        const int token_offset = idx / d_stride_uint4;
        const int vec_col = idx % d_stride_uint4;

        uint4 kv_val = make_uint4(0, 0, 0, 0);

        if (token_offset < num_tokens && vec_col < d_stride_uint4) {
            const int global_token_idx = start_token_idx + token_offset;

            const size_t base_offset = resolve_paged_kv_offset(
                global_token_idx,
                page_block_size,
                block_table,
                kv_head_offset,
                num_kv_heads,
                head_dim
            );

            const uint4* kv_vec = reinterpret_cast<const uint4*>(kv_cache + base_offset);
            kv_val = __ldg(&kv_vec[vec_col]);
        }

        smem_vec[token_offset * smem_stride_uint4 + vec_col] = kv_val;
    }
}

#endif
