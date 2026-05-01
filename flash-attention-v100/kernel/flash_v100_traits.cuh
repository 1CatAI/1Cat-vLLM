#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

template<int D>
struct FlashV100Traits {

    static constexpr int BLOCK_M_16  = 16;
    static constexpr int BLOCK_N_16  = 512;
    static constexpr int BLOCK_M_32  = 32;
    static constexpr int BLOCK_N_32  = 256;
    static constexpr int BLOCK_M_64  = 64;
    static constexpr int BLOCK_N_64  = 128;
    static constexpr int BLOCK_M_128 = 32;
    static constexpr int BLOCK_N_128 = 176;
    static constexpr int BLOCK_M_256 = 32;
    static constexpr int BLOCK_N_256 = 64;
    static constexpr int WARPS_PER_BLOCK = 16;
    static constexpr int THREADS_PER_WARP = 32;

    static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 :
                                   (D == 32) ? BLOCK_M_32 :
                                   (D == 64) ? BLOCK_M_64 :
                                   (D == 128) ? BLOCK_M_128 : BLOCK_M_256;

    static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 :
                                   (D == 32) ? BLOCK_N_32 :
                                   (D == 64) ? BLOCK_N_64 :
                                   (D == 128) ? BLOCK_N_128 : BLOCK_N_256;

    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * THREADS_PER_WARP;
    static constexpr int THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M;

    static constexpr int kBlockM = BLOCK_M;
    static constexpr int kBlockN = BLOCK_N;
    static constexpr int kHeadDim = D;

    static constexpr int kGmemThreadsPerRow = THREADS_PER_ROW;
    static constexpr int kGmemRowsPerThread = 1;
    static constexpr int kGmemElemsPerLoad = 8;

    static constexpr int PER_UINT4 = 8;
    static constexpr int d_stride_uint4 = (D + PER_UINT4 - 1) / PER_UINT4;
};

template<typename Traits>
__device__ __forceinline__
int64_t resolve_thread_kv_page_slice_offset(
    const int tidx,
    const int n_block,
    const int page_block_size,
    const int* __restrict__ block_table,
    const int page_stride,
    const int row_stride,
    const int partial_block_size = -1
) {
    constexpr int kGmemThreadsPerRow = Traits::kGmemThreadsPerRow;
    constexpr int kGmemRowsPerThread = Traits::kGmemRowsPerThread;
    constexpr int kGmemElemsPerLoad = Traits::kGmemElemsPerLoad;
    constexpr int kBlockN = Traits::kBlockN;

    const int64_t col_offset = (tidx % kGmemThreadsPerRow) * kGmemElemsPerLoad;
    int64_t block_row_offset = (tidx / kGmemThreadsPerRow) * kGmemRowsPerThread;

    if (partial_block_size > 0) {
        const int final_row_offset = max(partial_block_size - 1, 0);
        const int final_thread_row_offset =
            ((final_row_offset + kGmemRowsPerThread - 1) / kGmemRowsPerThread) * kGmemRowsPerThread;
        block_row_offset = min(block_row_offset, (int64_t)final_thread_row_offset);
    }

    const int64_t global_row_offset = block_row_offset + n_block * kBlockN;

    const int64_t page_offset = global_row_offset % page_block_size;
    const int64_t virtual_page_idx = global_row_offset / page_block_size;

    const int64_t physical_offset =
        ((int64_t)block_table[virtual_page_idx]) * ((int64_t)page_stride)
        + page_offset * ((int64_t)row_stride)
        + col_offset;

    return physical_offset;
}

template<typename Traits>
__device__ __forceinline__
void load_kv_tile_paged(
    const __half* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int start_token_idx,
    const int num_tokens,
    const int n_block,
    const int page_block_size,
    const int page_stride,
    const int row_stride,
    __half* smem_dst,
    const int smem_stride,
    const int tid,
    const int num_threads,
    const int partial_block_size = -1
) {
    constexpr int PER_UINT4 = Traits::PER_UINT4;
    constexpr int d_stride_uint4 = Traits::d_stride_uint4;
    constexpr int kBlockN = Traits::kBlockN;

    const uint4* kv_vec = reinterpret_cast<const uint4*>(kv_cache);
    uint4* smem_vec = reinterpret_cast<uint4*>(smem_dst);
    const int smem_stride_uint4 = (smem_stride + PER_UINT4 - 1) / PER_UINT4;

    #pragma unroll 2
    for (int idx = tid; idx < (num_tokens * d_stride_uint4); idx += num_threads) {
        const int token_offset = idx / d_stride_uint4;
        const int vec_col = idx % d_stride_uint4;

        uint4 kv_val = make_uint4(0, 0, 0, 0);

        if (token_offset < num_tokens && vec_col < d_stride_uint4) {

            const int global_token_idx = start_token_idx + token_offset;

            const int64_t physical_offset_halfs = resolve_thread_kv_page_slice_offset<Traits>(
                tid,
                n_block,
                page_block_size,
                block_table,
                page_stride,
                row_stride,
                partial_block_size
            );

            const int64_t physical_offset_uint4 = physical_offset_halfs / PER_UINT4;

            kv_val = __ldg(&kv_vec[physical_offset_uint4 + vec_col]);
        }

        smem_vec[token_offset * smem_stride_uint4 + vec_col] = kv_val;
    }
}
