#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <mma.h>
using namespace nvcuda::wmma;

#include "flash_v100_traits.cuh"
#include "fp8_kv_utils.cuh"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define MAX_THREADS_PER_WARP    32
#define MAX_THREADS_PER_SM      2048
#define MAX_THREAD_BLOCK_SIZE   1024
#define MAX_THREAD_BLOCK_PER_SM 32
#define MAX_WARPS_PER_SM        64
#define MAX_SM_PER_GPU          80
#define MAX_SMEM_PER_SM         98304

#define WARP_ALLOC_GROUP        4

namespace {

int kv_cache_dtype_code_from_string(const std::string& kv_cache_dtype) {
    if (kv_cache_dtype == "auto" || kv_cache_dtype == "bfloat16") {
        return flash_v100::KV_CACHE_DTYPE_FP16;
    }
    if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
        return flash_v100::KV_CACHE_DTYPE_FP8_E4M3;
    }
    if (kv_cache_dtype == "fp8_e5m2") {
        return flash_v100::KV_CACHE_DTYPE_FP8_E5M2;
    }
    return -1;
}

}  // namespace

#define BLOCK_M_16  16
#define BLOCK_N_16  512
#define WARPS_16    16

#define BLOCK_M_32  32
#define BLOCK_N_32  256
#define WARPS_32    16

#define BLOCK_M_64  64
#define BLOCK_N_64  128
#define WARPS_64    16

#define BLOCK_M_128 32
#define BLOCK_N_128 176
#define WARPS_128   16

#define BLOCK_M_256 32
#define BLOCK_N_256 64
#define WARPS_256   16

#define BLOCK_M_256_LOW_SMEM 16
#define BLOCK_N_256_LOW_SMEM 128
#define KV_STAGE_N_256_LOW_SMEM 1
#define WARPS_256_LOW_SMEM   16
#define BLOCK_N_256_LOW_SMEM_SCALAR_QK 32
#define KV_STAGE_N_256_LOW_SMEM_SCALAR_QK BLOCK_N_256_LOW_SMEM_SCALAR_QK
#define LOW_SMEM_PAGE_SIZE   16

template<int D, bool LOW_SMEM = false, bool LOW_SMEM_SCALAR_QK = false,
         bool LOW_SMEM_BM32 = false,
         bool D256_OUTPUT_STRIDE_268 = false,
         bool D256_SW_PIPELINE_QK = false,
         bool D256_SW_PIPELINE_PV = false>
struct KernelConfig {
    static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 : (D == 32) ? BLOCK_M_32 : (D == 64) ? BLOCK_M_64 : (D == 128) ? BLOCK_M_128 : (LOW_SMEM ? (LOW_SMEM_BM32 ? BLOCK_M_256 : BLOCK_M_256_LOW_SMEM) : BLOCK_M_256);
    static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 : (D == 32) ? BLOCK_N_32 : (D == 64) ? BLOCK_N_64 : (D == 128) ? BLOCK_N_128 : (LOW_SMEM ? (LOW_SMEM_SCALAR_QK ? BLOCK_N_256_LOW_SMEM_SCALAR_QK : BLOCK_N_256_LOW_SMEM) : BLOCK_N_256);
    static constexpr int WARPS_PER_BLOCK = (D == 16) ? WARPS_16 : (D == 32) ? WARPS_32 : (D == 64) ? WARPS_64 : (D == 128) ? WARPS_128 : (LOW_SMEM ? WARPS_256_LOW_SMEM : WARPS_256);

    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
    static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int PAD               = (8 - (D % 32) + 32) % 32;
    static constexpr int Q_STRIDE          = D + PAD;
    static constexpr int KV_STRIDE         = D + PAD;
    static constexpr int KV_STAGE_N        = (D == 256 && LOW_SMEM) ? (LOW_SMEM_SCALAR_QK ? KV_STAGE_N_256_LOW_SMEM_SCALAR_QK : KV_STAGE_N_256_LOW_SMEM) : BLOCK_N;
    static constexpr int S_STRIDE          = BLOCK_N + PAD;
    static constexpr int P_SUB_TILE        = 32;
    static constexpr int P_STRIDE          = P_SUB_TILE + PAD;
    static constexpr int P_STRICT_ELEMENTS = (D == 256) ? BLOCK_M * P_STRIDE : 1;
    static constexpr int PIPELINE_S_ELEMENTS =
        (D == 256 && LOW_SMEM && D256_SW_PIPELINE_QK
         && D256_SW_PIPELINE_PV)
            ? BLOCK_M * S_STRIDE
            : 1;
    static constexpr int PIPELINE_PAGE_ELEMENTS =
        (D == 256 && LOW_SMEM && D256_SW_PIPELINE_QK
         && D256_SW_PIPELINE_PV)
            ? BLOCK_N / LOW_SMEM_PAGE_SIZE
            : 1;
    static constexpr bool PIPELINE_SWIZZLED_Q =
        D == 256 && LOW_SMEM && D256_SW_PIPELINE_QK
        && D256_SW_PIPELINE_PV;
    static constexpr int Q_ELEMENTS =
        PIPELINE_SWIZZLED_Q ? BLOCK_M * D : BLOCK_M * Q_STRIDE;
    // The D256 low-SMEM candidate changes only the output accumulator pitch
    // from 264 to 268 floats. This preserves all WMMA/softmax arithmetic and
    // targets the largest 8-way shared-memory load/store groups.
    static constexpr int OUTPUT_EXTRA_PAD =
        (D == 256 && LOW_SMEM && D256_OUTPUT_STRIDE_268) ? 4 : 0;
    static constexpr int O_STRIDE          = D + PAD + OUTPUT_EXTRA_PAD;
    static constexpr int PER_UINT4         = 8;
    static constexpr int LOW_SMEM_PAGE_COUNT =
        (D == 256 && LOW_SMEM) ? (BLOCK_N / LOW_SMEM_PAGE_SIZE) : 1;
    struct alignas(128) SmemLayout {
        alignas(16) __half q      [Q_ELEMENTS];
    union {
        alignas(16) __half k      [KV_STAGE_N * KV_STRIDE];
        alignas(16) __half v      [KV_STAGE_N * KV_STRIDE];
    } reuse_kv;
    union {
        alignas(16) float  s      [BLOCK_M * S_STRIDE];
        alignas(16) __half p      [BLOCK_M * S_STRIDE];
    } reuse_sp;
        alignas(16) __half p_strict[P_STRICT_ELEMENTS];
        alignas(16) float  pipeline_s[PIPELINE_S_ELEMENTS];
        alignas(16) int    pipeline_page_idx[PIPELINE_PAGE_ELEMENTS];
        alignas(16) int    pipeline_page_offset[PIPELINE_PAGE_ELEMENTS];
        alignas(16) float  o      [BLOCK_M * O_STRIDE];
        alignas(16) float  row_max[BLOCK_M];
        alignas(16) float  row_sum[BLOCK_M];
        alignas(16) int    page_idx[LOW_SMEM_PAGE_COUNT];
        alignas(16) int    page_offset[LOW_SMEM_PAGE_COUNT];
    };

    static constexpr size_t TOTAL_SMEM = ((sizeof(SmemLayout) + 127) & ~size_t(127));
};

template<typename Config>
__device__ __forceinline__ void init_smem(char* smem_raw) {
    constexpr int N_U4 = Config::TOTAL_SMEM / 16;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    const int tid = threadIdx.x;

    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    #pragma unroll 1
    for (int i = tid; i < N_U4; i += THREADS_PER_BLOCK) {
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};"
                     :: "r"(addr + (i << 4)), "r"(0) : "memory");
    }
    __syncthreads();
}

__device__ __forceinline__ int pipeline_swizzled_q_row_slot(int row) {
    return (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
}

__device__ __forceinline__ int pipeline_swizzled_matrix_a_offset(
    int row,
    int col) {
    const int tile = col / WMMA_K;
    const int within_tile = col - tile * WMMA_K;
    const int plane = within_tile >> 3;
    const int inner = within_tile & 7;
    return tile * WMMA_M * WMMA_K + plane * WMMA_M * 8
           + pipeline_swizzled_q_row_slot(row) * 8 + inner;
}

__device__ __forceinline__ void load_pipeline_swizzled_matrix_a_fragment(
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major>& frag,
    const __half* __restrict__ matrix,
    int k_offset) {
    const int lane = threadIdx.x & 31;
    const int row =
        (lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8;
    const int slot = pipeline_swizzled_q_row_slot(row);
    const int tile_offset = (k_offset / WMMA_K) * WMMA_M * WMMA_K;
    uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(matrix + tile_offset + slot * 8));
    uint32_t* words = reinterpret_cast<uint32_t*>(&frag);
    asm volatile(
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3])
        : "r"(address)
        : "memory");
    address += WMMA_M * 8 * sizeof(__half);
    asm volatile(
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(words[4]), "=r"(words[5]), "=r"(words[6]), "=r"(words[7])
        : "r"(address)
        : "memory");
}

template<int Q_STRIDE, int S_STRIDE, int K_TILES, bool IS_CAUSAL,
         bool SWIZZLED_Q = false>
__device__ __forceinline__ void pipeline_qk_slice(
    const __half* __restrict__ sQ,
    float* __restrict__ score,
    const __half* __restrict__ k_cache,
    const int* __restrict__ page_idx,
    const int* __restrict__ page_offset,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int kv_head_id,
    int start_col,
    int valid_k_rows,
    int start_row,
    int valid_q_rows,
    int causal_q_offset,
    int tile_n,
    int k_begin,
    bool load_partial,
    bool finalize,
    float softmax_scale,
    int window_size_left,
    int window_size_right,
    float neg_inf) {
    if (tile_n >= valid_k_rows) {
        return;
    }

    const int page_slot = tile_n >> 4;
    const int block_offset = page_offset[page_slot];
    const int physical_block_idx = page_idx[page_slot];
    const __half* k_tile_base =
        k_cache + (int64_t)physical_block_idx * k_block_stride
        + (int64_t)block_offset * k_token_stride
        + (int64_t)kv_head_id * k_head_stride;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    if (load_partial) {
        load_matrix_sync(
            acc_frag, score + tile_n, S_STRIDE, mem_row_major);
    } else {
        fill_fragment(acc_frag, 0.0f);
    }

    #pragma unroll
    for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
        const int k_offset = k_begin + k_tile * WMMA_K;
        if constexpr (SWIZZLED_Q) {
            load_pipeline_swizzled_matrix_a_fragment(
                a_frag, sQ, k_offset);
        } else {
            load_matrix_sync(a_frag, sQ + k_offset, Q_STRIDE);
        }
        load_matrix_sync(
            b_frag, k_tile_base + k_offset, k_token_stride);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (finalize) {
        const int lane_id = threadIdx.x & 31;
        const unsigned row_causal =
            (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8
            + ((lane_id >> 4) & 0b1) * 4;
        const unsigned col_causal =
            ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            const unsigned col =
                col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
            const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;
            const int global_m = start_row + row;
            const int global_n = start_col + tile_n + col;
            const int global_q_pos = global_m + causal_q_offset;

            const bool is_valid =
                global_m < start_row + valid_q_rows
                && global_n < start_col + valid_k_rows;
            bool is_causal_valid = true;
            if constexpr (IS_CAUSAL) {
                is_causal_valid = global_n <= global_q_pos;
            }
            bool is_window_valid = true;
            if (window_size_left >= 0) {
                is_window_valid =
                    global_n >= global_q_pos - window_size_left;
            }
            if (window_size_right >= 0) {
                is_window_valid =
                    is_window_valid
                    && global_n <= global_q_pos + window_size_right;
            }
            acc_frag.x[i] =
                (is_valid && is_causal_valid && is_window_valid)
                    ? acc_frag.x[i] * softmax_scale
                    : neg_inf;
        }
    }

    store_matrix_sync(score + tile_n, acc_frag, S_STRIDE, mem_row_major);
}

template<int P_STRIDE, int O_STRIDE, bool SWIZZLED_P = false>
__device__ __forceinline__ void pipeline_pv_tile(
    const __half* __restrict__ sP,
    float* __restrict__ sO,
    const __half* __restrict__ v_cache,
    const int* __restrict__ page_idx,
    const int* __restrict__ page_offset,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    int kv_head_id,
    int sub_start,
    int sub_valid_k_rows,
    int tile_d) {
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    load_matrix_sync(
        acc_frag, sO + tile_d, O_STRIDE, mem_row_major);

    #pragma unroll
    for (int tile_k = 0; tile_k < 2; ++tile_k) {
        const int k_offset = tile_k * WMMA_K;
        if (k_offset >= sub_valid_k_rows) {
            break;
        }
        const int token_offset = sub_start + k_offset;
        const int page_slot = token_offset >> 4;
        const int block_offset =
            page_offset[page_slot] + (token_offset & 15);
        const int physical_block_idx = page_idx[page_slot];
        const __half* v_tile_ptr =
            v_cache + (int64_t)physical_block_idx * v_block_stride
            + (int64_t)block_offset * v_token_stride
            + (int64_t)kv_head_id * v_head_stride + tile_d;
        if constexpr (SWIZZLED_P) {
            load_pipeline_swizzled_matrix_a_fragment(
                a_frag, sP, k_offset);
        } else {
            load_matrix_sync(a_frag, sP + k_offset, P_STRIDE);
        }
        load_matrix_sync(b_frag, v_tile_ptr, v_token_stride);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    store_matrix_sync(sO + tile_d, acc_frag, O_STRIDE, mem_row_major);
}

template<int D, bool LOW_SMEM, bool LOW_SMEM_CONTIG_FAST,
         bool LOW_SMEM_SCALAR_QK, bool LOW_SMEM_BM32, bool SPLIT_KV,
         bool IS_CAUSAL, int KV_DTYPE, bool D256_OUTPUT_STRIDE_268 = false,
         bool D256_SW_PIPELINE_QK = false,
         bool D256_SW_PIPELINE_PV = false>
__global__ void __launch_bounds__(
    KernelConfig<D, LOW_SMEM, LOW_SMEM_SCALAR_QK,
                 LOW_SMEM_BM32,
                 D256_OUTPUT_STRIDE_268,
                 D256_SW_PIPELINE_QK,
                 D256_SW_PIPELINE_PV>::THREADS_PER_BLOCK, 2)
flash_attention_forward_kernel_paged(
    const __half* __restrict__ Q,
    const void* __restrict__ K_cache,
    const void* __restrict__ V_cache,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
    const int* __restrict__ block_table,
    const int* __restrict__ seqused_k,
    const int B,
    const int H,
    const int M,
    const int N,
    const int* __restrict__ bfla_block_mask,
    const int bfla_mask_block_n,
    const int64_t bfla_mask_stride_b,
    const int64_t bfla_mask_stride_h,
    const int64_t bfla_mask_stride_q,
    const int64_t bfla_mask_stride_k,
    const int page_block_size,
    const int num_kv_heads,
    const int64_t k_block_stride,
    const int64_t k_token_stride,
    const int64_t k_head_stride,
    const int64_t v_block_stride,
    const int64_t v_token_stride,
    const int64_t v_head_stride,
    const float softmax_scale,
    const float k_scale,
    const float v_scale,
    const int window_size_left,
    const int window_size_right,
          float* __restrict__ split_tmp_out,
          float* __restrict__ split_tmp_row_max,
          float* __restrict__ split_tmp_row_sum,
    const int split_kv_tiles
) {
    using Config = KernelConfig<D, LOW_SMEM, LOW_SMEM_SCALAR_QK,
                                LOW_SMEM_BM32, D256_OUTPUT_STRIDE_268,
                                D256_SW_PIPELINE_QK,
                                D256_SW_PIPELINE_PV>;
    using Traits = FlashV100Traits<D>;

    constexpr int BLOCK_M           = Config::BLOCK_M;
    constexpr int BLOCK_N           = Config::BLOCK_N;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW   = Config::THREADS_PER_ROW;
    constexpr int WARPS_PER_BLOCK   = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE          = Config::Q_STRIDE;
    constexpr int KV_STRIDE         = Config::KV_STRIDE;
    constexpr int S_STRIDE          = Config::S_STRIDE;
    constexpr int P_SUB_TILE        = Config::P_SUB_TILE;
    constexpr int P_STRIDE          = Config::P_STRIDE;
    constexpr int O_STRIDE          = Config::O_STRIDE;
    constexpr int PER_UINT4         = Config::PER_UINT4;
    constexpr bool USE_SW_PIPELINE =
        D256_SW_PIPELINE_QK || D256_SW_PIPELINE_PV;
    if constexpr (USE_SW_PIPELINE) {
        static_assert(D == 256 && LOW_SMEM && !LOW_SMEM_CONTIG_FAST
                          && !LOW_SMEM_SCALAR_QK && !LOW_SMEM_BM32
                          && !SPLIT_KV && BLOCK_M == 16 && BLOCK_N == 128
                          && WARPS_PER_BLOCK == 16,
                      "software pipeline is specialized for D256 BM16 BN128");
    }
    const float NEG_INF = -1e30f;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int batch_id = batch_head_id / H;
    const int q_head_id = batch_head_id % H;
    const int kv_group_size = H / num_kv_heads;
    const int kv_head_id = q_head_id / kv_group_size;

    const int block_m = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;

    const int actual_N = seqused_k[batch_id];
    int num_n_tiles = (actual_N + BLOCK_N - 1) / BLOCK_N;
    const int valid_q_rows = min(BLOCK_M, M - start_row);
    const int causal_q_offset = max(actual_N - M, 0);

    int max_key_pos = actual_N - 1;
    if constexpr (IS_CAUSAL) {
        max_key_pos = min(max_key_pos,
                          start_row + valid_q_rows - 1 + causal_q_offset);
    }
    if (window_size_right >= 0) {
        max_key_pos = min(max_key_pos,
                          start_row + valid_q_rows - 1 + causal_q_offset
                              + window_size_right);
    }
    if (max_key_pos < 0) {
        num_n_tiles = 0;
    } else {
        num_n_tiles = min(num_n_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
    }
    const int min_key_pos = window_size_left >= 0
                                ? max(0, start_row + causal_q_offset
                                             - window_size_left)
                                : 0;

    const int tid = threadIdx.x;
    const int warp_id = tid / MAX_THREADS_PER_WARP;
    const int lane_id = tid % MAX_THREADS_PER_WARP;

    const size_t q_head_linear = (size_t)batch_id * H + q_head_id;
    const __half* q_ptr = Q + q_head_linear * M * D + start_row * D;
    __half* out_ptr = Out + q_head_linear * M * D + start_row * D;
    float* softmax_lse_ptr = softmax_lse + q_head_linear * M + start_row;

    const int max_num_blocks_per_seq = (N + page_block_size - 1) / page_block_size;
    const int* block_table_seq = block_table + batch_id * max_num_blocks_per_seq;

    const int64_t k_row_stride = k_token_stride;
    const int64_t v_row_stride = v_token_stride;

    extern __shared__ char smem_raw[];
    init_smem<Config>(smem_raw);
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* sQ      = smem.q;
    __half* sK      = smem.reuse_kv.k;
    __half* sV      = smem.reuse_kv.v;
    float*  sS      = smem.reuse_sp.s;
    __half* sP      = (D == 256) ? smem.p_strict : smem.reuse_sp.p;
    const int p_stride = (D == 256) ? P_STRIDE : S_STRIDE;
    const int p_tile_capacity = (D == 256) ? P_SUB_TILE : BLOCK_N;
    float*  sO      = smem.o;
    float*  sRowMax = smem.row_max;
    float*  sRowSum = smem.row_sum;
    int*    sPageIdx = smem.page_idx;
    int*    sPageOffset = smem.page_offset;

    const int  d_stride_uint4 = (D + PER_UINT4 - 1) / PER_UINT4;
    const int  q_stride_uint4 = (Q_STRIDE  + PER_UINT4 - 1) / PER_UINT4;
    const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

    if (tid < BLOCK_M) {
        sRowMax[tid] = NEG_INF;
        sRowSum[tid] = 0.0f;
    }
    constexpr int O_UINT4_PER_ROW = D / 4;
    constexpr int O_UINT4_STRIDE = O_STRIDE / 4;
    for (int idx = tid; idx < BLOCK_M * O_UINT4_PER_ROW;
         idx += THREADS_PER_BLOCK) {
        const int row = idx / O_UINT4_PER_ROW;
        const int col = idx % O_UINT4_PER_ROW;
        reinterpret_cast<uint4*>(sO)[row * O_UINT4_STRIDE + col] =
            make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    const uint4*      q_vec = reinterpret_cast<const uint4*>(q_ptr);
    uint4*           sQ_vec = reinterpret_cast<uint4*>(sQ);

    #pragma unroll 4
    for (int idx = tid; idx < (valid_q_rows * d_stride_uint4); idx += THREADS_PER_BLOCK) {
        const int row = idx / d_stride_uint4;
        const int vec_col = idx % d_stride_uint4;
        uint4 q_val = make_uint4(0, 0, 0, 0);
        if (row < valid_q_rows && vec_col < d_stride_uint4) {
            q_val = __ldg(&q_vec[row * d_stride_uint4 + vec_col]);
        }
        if constexpr (Config::PIPELINE_SWIZZLED_Q) {
            const int k_tile = vec_col >> 1;
            const int plane = vec_col & 1;
            const int slot = pipeline_swizzled_q_row_slot(row);
            sQ_vec[k_tile * (2 * BLOCK_M) + plane * BLOCK_M + slot] =
                q_val;
        } else {
            sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
        }
    }
    __syncthreads();

    int cross_block_first_n_tile = 0;
    if constexpr (D256_SW_PIPELINE_QK && D256_SW_PIPELINE_PV) {
        const bool use_cross_block_pipeline =
            valid_q_rows == BLOCK_M && actual_N > 0
            && page_block_size == 784 && bfla_block_mask == nullptr
            && window_size_left < 0 && window_size_right < 0;
        if (use_cross_block_pipeline) {
            const __half* k_cache_h =
                reinterpret_cast<const __half*>(K_cache);
            const __half* v_cache_h =
                reinterpret_cast<const __half*>(V_cache);

            if (tid < Config::LOW_SMEM_PAGE_COUNT) {
                const int global_token_idx = tid * LOW_SMEM_PAGE_SIZE;
                const int virtual_block_idx =
                    global_token_idx / page_block_size;
                sPageIdx[tid] =
                    __ldg(&block_table_seq[virtual_block_idx]);
                sPageOffset[tid] =
                    global_token_idx
                    - virtual_block_idx * page_block_size;
            }
            __syncthreads();

            if (warp_id < BLOCK_N / WMMA_N) {
                const int valid_k_rows = min(BLOCK_N, actual_N);
                pipeline_qk_slice<Q_STRIDE, S_STRIDE, D / WMMA_K,
                                  IS_CAUSAL, true>(
                    sQ, sS, k_cache_h, sPageIdx, sPageOffset,
                    k_block_stride, k_token_stride,
                    k_head_stride, kv_head_id, 0, valid_k_rows, start_row,
                    valid_q_rows, causal_q_offset, warp_id * WMMA_N, 0, false,
                    true, softmax_scale, window_size_left, window_size_right,
                    NEG_INF);
            }
            __syncthreads();

            // Keep the hot loop on a compile-time steady-state path. The
            // existing exact path below drains the final block, avoiding a
            // loop-carried has_next predicate in every softmax/PV panel.
            const int steady_n_tiles = num_n_tiles - 1;
            for (int block_n = 0; block_n < steady_n_tiles; ++block_n) {
                const int start_col = block_n * BLOCK_N;
                const int valid_k_rows = BLOCK_N;
                const int next_start_col = start_col + BLOCK_N;
                const int next_valid_k_rows =
                    min(BLOCK_N, actual_N - next_start_col);
                const bool odd_block = (block_n & 1) != 0;
                float* const score_current =
                    odd_block ? smem.pipeline_s : sS;
                float* const score_next =
                    odd_block ? sS : smem.pipeline_s;
                int* const page_idx_current =
                    odd_block ? smem.pipeline_page_idx : sPageIdx;
                int* const page_offset_current =
                    odd_block ? smem.pipeline_page_offset : sPageOffset;
                int* const page_idx_next =
                    odd_block ? sPageIdx : smem.pipeline_page_idx;
                int* const page_offset_next =
                    odd_block ? sPageOffset : smem.pipeline_page_offset;

                if (tid < Config::LOW_SMEM_PAGE_COUNT) {
                    const int page_token_offset =
                        tid * LOW_SMEM_PAGE_SIZE;
                    const int global_token_idx =
                        next_start_col + page_token_offset;
                    const int virtual_block_idx =
                        global_token_idx / page_block_size;
                    page_idx_next[tid] =
                        __ldg(&block_table_seq[virtual_block_idx]);
                    page_offset_next[tid] =
                        global_token_idx
                        - virtual_block_idx * page_block_size;
                }
                __syncthreads();

                for (int sub_start = 0; sub_start < valid_k_rows;
                     sub_start += P_SUB_TILE) {
                    const int sub_valid_k_rows =
                        min(P_SUB_TILE, valid_k_rows - sub_start);
                    const int row = warp_id;
                    const int thread_in_row = lane_id;
                    const unsigned mask = 0xFFFFFFFFU;
                    float* sS_row_f =
                        score_current + row * S_STRIDE + sub_start;

                    const int vec_cols = sub_valid_k_rows >> 2;
                    const int vecs_per_thread =
                        (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
                    const int tail_start = vec_cols << 2;
                    float thread_max = NEG_INF;
                    float4* sS_vec4 = reinterpret_cast<float4*>(sS_row_f);

                    #pragma unroll 4
                    for (int j = 0; j < vecs_per_thread; ++j) {
                        const int vc =
                            thread_in_row + j * THREADS_PER_ROW;
                        if (vc < vec_cols) {
                            const float4 v4 = sS_vec4[vc];
                            thread_max = fmaxf(
                                thread_max,
                                fmaxf(
                                    fmaxf(v4.x, v4.y),
                                    fmaxf(v4.z, v4.w)));
                        }
                    }
                    #pragma unroll
                    for (int c = tail_start + thread_in_row;
                         c < sub_valid_k_rows; c += THREADS_PER_ROW) {
                        thread_max = fmaxf(thread_max, sS_row_f[c]);
                    }
                    #pragma unroll
                    for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
                        thread_max = fmaxf(
                            thread_max,
                            __shfl_down_sync(mask, thread_max, o));
                    }

                    const float row_max =
                        __shfl_sync(mask, thread_max, 0);
                    const float old_max = sRowMax[row];
                    const float new_max = fmaxf(old_max, row_max);
                    const float exp_diff = __expf(old_max - new_max);
                    float thread_sum = 0.0f;
                    int vc_base = thread_in_row;

                    #pragma unroll 4
                    for (int j = 0; j < vecs_per_thread;
                         ++j, vc_base += THREADS_PER_ROW) {
                        if (vc_base < vec_cols) {
                            const float4 v4 = sS_vec4[vc_base];
                            const float e0 =
                                __expf(fmaxf(v4.x - new_max, -80.0f));
                            const float e1 =
                                __expf(fmaxf(v4.y - new_max, -80.0f));
                            const float e2 =
                                __expf(fmaxf(v4.z - new_max, -80.0f));
                            const float e3 =
                                __expf(fmaxf(v4.w - new_max, -80.0f));
                            thread_sum += (e0 + e1) + (e2 + e3);
                            const int p_col = vc_base * 4;
                            __half2* p_half2 = reinterpret_cast<__half2*>(
                                sP + pipeline_swizzled_matrix_a_offset(
                                         row, p_col));
                            p_half2[0] =
                                __float22half2_rn(make_float2(e0, e1));
                            p_half2[1] =
                                __float22half2_rn(make_float2(e2, e3));
                        }
                    }
                    #pragma unroll 4
                    for (int c = tail_start + thread_in_row;
                         c < sub_valid_k_rows; c += THREADS_PER_ROW) {
                        const float e = __expf(
                            fmaxf(sS_row_f[c] - new_max, -80.0f));
                        thread_sum += e;
                        sP[pipeline_swizzled_matrix_a_offset(row, c)] =
                            __float2half_rn(e);
                    }
                    #pragma unroll
                    for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
                        thread_sum +=
                            __shfl_down_sync(mask, thread_sum, o);
                    }
                    const float row_sum =
                        __shfl_sync(mask, thread_sum, 0);
                    if (thread_in_row == 0) {
                        sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
                        sRowMax[row] = new_max;
                    }

                    #pragma unroll 4
                    for (int c = tail_start + thread_in_row;
                        c < P_SUB_TILE; c += THREADS_PER_ROW) {
                        if (c >= sub_valid_k_rows) {
                            sP[pipeline_swizzled_matrix_a_offset(row, c)] =
                                __float2half(0.0f);
                        }
                    }

                    if (block_n > 0 || sub_start > 0) {
                        float4* sO_vec = reinterpret_cast<float4*>(
                            sO + row * O_STRIDE);
                        #pragma unroll 4
                        for (int ov = thread_in_row; ov < D / 4;
                             ov += THREADS_PER_ROW) {
                            float4 v = sO_vec[ov];
                            v.x *= exp_diff;
                            v.y *= exp_diff;
                            v.z *= exp_diff;
                            v.w *= exp_diff;
                            sO_vec[ov] = v;
                        }
                    }
                    __syncthreads();

                    if (sub_start == 0) {
                        if (warp_id < BLOCK_N / WMMA_N) {
                            pipeline_qk_slice<
                                Q_STRIDE, S_STRIDE, D / WMMA_K, IS_CAUSAL,
                                true>(
                                sQ, score_next, k_cache_h, page_idx_next,
                                page_offset_next, k_block_stride,
                                k_token_stride, k_head_stride, kv_head_id,
                                next_start_col, next_valid_k_rows, start_row,
                                valid_q_rows, causal_q_offset,
                                warp_id * WMMA_N, 0, false, true,
                                softmax_scale, window_size_left,
                                window_size_right, NEG_INF);
                        } else {
                            const int pv_warp =
                                warp_id - BLOCK_N / WMMA_N;
                            pipeline_pv_tile<P_STRIDE, O_STRIDE, true>(
                                sP, sO, v_cache_h, page_idx_current,
                                page_offset_current, v_block_stride,
                                v_token_stride, v_head_stride, kv_head_id,
                                sub_start, sub_valid_k_rows,
                                pv_warp * WMMA_N);
                            pipeline_pv_tile<P_STRIDE, O_STRIDE, true>(
                                sP, sO, v_cache_h, page_idx_current,
                                page_offset_current, v_block_stride,
                                v_token_stride, v_head_stride, kv_head_id,
                                sub_start, sub_valid_k_rows,
                                (pv_warp + BLOCK_N / WMMA_N) * WMMA_N);
                        }
                    } else {
                        pipeline_pv_tile<P_STRIDE, O_STRIDE, true>(
                            sP, sO, v_cache_h, page_idx_current,
                            page_offset_current, v_block_stride,
                            v_token_stride, v_head_stride, kv_head_id,
                            sub_start, sub_valid_k_rows, warp_id * WMMA_N);
                    }
                    __syncthreads();
                }

            }
            cross_block_first_n_tile = steady_n_tiles;
        }
    }

    int first_n_tile = cross_block_first_n_tile;
    int last_n_tile = num_n_tiles;
    if constexpr (SPLIT_KV) {
        const int partition_id = blockIdx.y;
        const int tiles_per_partition =
            split_kv_tiles > 1 ? split_kv_tiles : 1;
        first_n_tile = partition_id * tiles_per_partition;
        last_n_tile = min(num_n_tiles, first_n_tile + tiles_per_partition);
    }

    for (int block_n = first_n_tile; block_n < last_n_tile; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= actual_N) break;
        const int valid_k_rows = min(BLOCK_N, actual_N - start_col);

        if (start_col + valid_k_rows <= min_key_pos) {
            continue;
        }

        if (bfla_block_mask != nullptr && bfla_mask_block_n > 0) {
            const int mask_q_idx = start_row / bfla_mask_block_n;
            const int mask_k_idx = start_col / bfla_mask_block_n;
            const int keep_tile = __ldg(
                bfla_block_mask
                + (int64_t)batch_id * bfla_mask_stride_b
                + (int64_t)kv_head_id * bfla_mask_stride_h
                + (int64_t)mask_q_idx * bfla_mask_stride_q
                + (int64_t)mask_k_idx * bfla_mask_stride_k);
            if (keep_tile == 0) {
                continue;
            }
        }

        const int partial_block_size = (block_n == num_n_tiles - 1 && actual_N % BLOCK_N != 0)
                                       ? (actual_N % BLOCK_N) : -1;

        uint4* sK_vec = reinterpret_cast<uint4*>(sK);
        const int64_t row_stride_uint4 = k_row_stride / PER_UINT4;
        const int start_page = start_col / page_block_size;
        const int page_offset = start_col % page_block_size;
        const bool single_page_tile =
            (page_offset + valid_k_rows) <= page_block_size;
        const bool two_page_tile =
            !single_page_tile &&
            (page_offset + valid_k_rows) <= (page_block_size * 2);
        const int first_page_rows =
            single_page_tile ? valid_k_rows
                             : min(valid_k_rows, page_block_size - page_offset);
        const int second_page_rows = valid_k_rows - first_page_rows;
        const int physical_block_idx0 = __ldg(&block_table_seq[start_page]);
        const int physical_block_idx1 =
            second_page_rows > 0 ? __ldg(&block_table_seq[start_page + 1]) : -1;
        const bool four_page_aligned_tile =
            D == 256 && page_block_size == 16 && page_offset == 0
            && BLOCK_N == page_block_size * 4
            && valid_k_rows == BLOCK_N
            && k_block_stride == (int64_t)page_block_size * k_token_stride
            && v_block_stride == (int64_t)page_block_size * v_token_stride;
        const int physical_block_idx2 =
            four_page_aligned_tile ? __ldg(&block_table_seq[start_page + 2]) : -1;
        const int physical_block_idx3 =
            four_page_aligned_tile ? __ldg(&block_table_seq[start_page + 3]) : -1;
        const bool four_page_contiguous_tile =
            four_page_aligned_tile
            && physical_block_idx1 == physical_block_idx0 + 1
            && physical_block_idx2 == physical_block_idx0 + 2
            && physical_block_idx3 == physical_block_idx0 + 3;
        bool low_smem_contiguous_page_tile = false;
        if constexpr (LOW_SMEM) {
            static_assert(D == 256, "low-smem paged prefill is D=256 only");
            static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16,
                          "low-smem paged prefill is fp16-KV only");
            static_assert(BLOCK_N % LOW_SMEM_PAGE_SIZE == 0,
                          "low-smem BLOCK_N must be page aligned");

            constexpr int LOW_SMEM_PAGE_COUNT =
                Config::LOW_SMEM_PAGE_COUNT;
            if (tid < LOW_SMEM_PAGE_COUNT) {
                const int page_token_offset = tid * LOW_SMEM_PAGE_SIZE;
                if (page_token_offset < valid_k_rows) {
                    const int global_token_idx =
                        start_col + page_token_offset;
                    const int virtual_block_idx =
                        global_token_idx / page_block_size;
                    sPageIdx[tid] = __ldg(
                        &block_table_seq[virtual_block_idx]);
                    sPageOffset[tid] =
                        global_token_idx
                        - virtual_block_idx * page_block_size;
                } else {
                    sPageIdx[tid] = -1;
                    sPageOffset[tid] = 0;
                }
            }
            __syncthreads();

            if constexpr (LOW_SMEM_CONTIG_FAST) {
                low_smem_contiguous_page_tile =
                    valid_k_rows == BLOCK_N
                    && k_block_stride
                        == (int64_t)page_block_size * k_token_stride
                    && v_block_stride
                        == (int64_t)page_block_size * v_token_stride
                    && sPageIdx[0] >= 0;
                #pragma unroll
                for (int i = 1; i < LOW_SMEM_PAGE_COUNT; ++i) {
                    const int linear_offset =
                        sPageOffset[0] + i * LOW_SMEM_PAGE_SIZE;
                    const int expected_page_delta =
                        linear_offset / page_block_size;
                    const int expected_page_offset =
                        linear_offset
                        - expected_page_delta * page_block_size;
                    low_smem_contiguous_page_tile =
                        low_smem_contiguous_page_tile
                        && sPageIdx[i] == sPageIdx[0] + expected_page_delta
                        && sPageOffset[i] == expected_page_offset;
                }
            } else {
                low_smem_contiguous_page_tile = false;
            }

            const __half* K_cache_h = reinterpret_cast<const __half*>(K_cache);

            if constexpr (LOW_SMEM_SCALAR_QK) {
                uint4* sK_vec = reinterpret_cast<uint4*>(sK);
                for (int idx = tid; idx < valid_k_rows * d_stride_uint4;
                     idx += THREADS_PER_BLOCK) {
                    const int row = idx / d_stride_uint4;
                    const int vec_col = idx % d_stride_uint4;
                    uint4 k_val = make_uint4(0, 0, 0, 0);
                    if (row < valid_k_rows && vec_col < d_stride_uint4) {
                        const int page_slot = row >> 4;
                        const int block_offset =
                            sPageOffset[page_slot]
                            + (row - page_slot * LOW_SMEM_PAGE_SIZE);
                        const int physical_block_idx_direct =
                            sPageIdx[page_slot];
                        const __half* k_row_ptr;
                        if constexpr (LOW_SMEM_CONTIG_FAST) {
                            k_row_ptr =
                                low_smem_contiguous_page_tile
                                    ? K_cache_h
                                          + (int64_t)sPageIdx[0] * k_block_stride
                                          + (int64_t)(sPageOffset[0] + row)
                                                * k_token_stride
                                          + (int64_t)kv_head_id * k_head_stride
                                    : K_cache_h
                                          + (int64_t)physical_block_idx_direct
                                                * k_block_stride
                                          + (int64_t)block_offset
                                                * k_token_stride
                                          + (int64_t)kv_head_id * k_head_stride;
                        } else {
                            k_row_ptr =
                                K_cache_h
                                + (int64_t)physical_block_idx_direct
                                      * k_block_stride
                                + (int64_t)block_offset * k_token_stride
                                + (int64_t)kv_head_id * k_head_stride;
                        }
                        const uint4* k_row_vec =
                            reinterpret_cast<const uint4*>(k_row_ptr);
                        k_val = __ldg(&k_row_vec[vec_col]);
                    }
                    sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
                }
                __syncthreads();

                const int total_scores = valid_q_rows * valid_k_rows;
                for (int idx = tid; idx < total_scores;
                     idx += THREADS_PER_BLOCK) {
                    const int row = idx / valid_k_rows;
                    const int col = idx - row * valid_k_rows;
                    const int global_m = start_row + row;
                    const int global_n = start_col + col;
                    const int global_q_pos = global_m + causal_q_offset;

                    bool is_valid = true;
                    if constexpr (IS_CAUSAL) {
                        is_valid = global_n <= global_q_pos;
                    }
                    if (window_size_left >= 0) {
                        is_valid =
                            is_valid
                            && global_n >= global_q_pos - window_size_left;
                    }
                    if (window_size_right >= 0) {
                        is_valid =
                            is_valid
                            && global_n <= global_q_pos + window_size_right;
                    }

                    float acc = 0.0f;
                    if (is_valid) {
                        #pragma unroll 8
                        for (int d = 0; d < D; d += 2) {
                            const __half2 q_h2 =
                                *reinterpret_cast<const __half2*>(
                                    sQ + row * Q_STRIDE + d);
                            const __half2 k_h2 =
                                *reinterpret_cast<const __half2*>(
                                    sK + col * KV_STRIDE + d);
                            const float2 q_f2 = __half22float2(q_h2);
                            const float2 k_f2 = __half22float2(k_h2);
                            acc = fmaf(q_f2.x, k_f2.x, acc);
                            acc = fmaf(q_f2.y, k_f2.y, acc);
                        }
                    }
                    sS[row * S_STRIDE + col] =
                        is_valid ? acc * softmax_scale : NEG_INF;
                }
            } else if constexpr (D256_SW_PIPELINE_QK) {
                constexpr int QK_TILES = BLOCK_N / WMMA_N;
                if (warp_id < QK_TILES) {
                    pipeline_qk_slice<Q_STRIDE, S_STRIDE, D / WMMA_K,
                                      IS_CAUSAL, D256_SW_PIPELINE_PV>(
                        sQ, sS, K_cache_h, sPageIdx, sPageOffset,
                        k_block_stride, k_token_stride, k_head_stride,
                        kv_head_id, start_col, valid_k_rows, start_row,
                        valid_q_rows, causal_q_offset, warp_id * WMMA_N, 0,
                        false, true, softmax_scale, window_size_left,
                        window_size_right, NEG_INF);
                }
            } else {
                const int num_tiles_m_qk = (BLOCK_M + WMMA_M - 1) / WMMA_M;
                const int num_tiles_n_qk = (BLOCK_N + WMMA_N - 1) / WMMA_N;
                const int num_tiles_k_qk = (D + WMMA_K - 1) / WMMA_K;
                const int total_tiles_qk = num_tiles_m_qk * num_tiles_n_qk;
                const int tiles_per_warp_qk =
                    (total_tiles_qk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                const unsigned row_causal =
                    (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8
                    + ((lane_id >> 4) & 0b1) * 4;
                const unsigned col_causal =
                    ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

                for (int tile_idx = 0; tile_idx < tiles_per_warp_qk;
                     ++tile_idx) {
                    const int global_tile_idx =
                        warp_id * tiles_per_warp_qk + tile_idx;
                    if (global_tile_idx >= total_tiles_qk) break;

                    const int tile_m_idx = global_tile_idx / num_tiles_n_qk;
                    const int tile_n_idx = global_tile_idx % num_tiles_n_qk;

                    const int tile_m = tile_m_idx * WMMA_M;
                    const int tile_n = tile_n_idx * WMMA_N;

                    if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) {
                        continue;
                    }

                    const int page_slot = tile_n >> 4;
                    const int block_offset = sPageOffset[page_slot];
                    const int physical_block_idx_direct = sPageIdx[page_slot];

                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major>
                        a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major>
                        b_frag;
                    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>
                        acc_frag;
                    fill_fragment(acc_frag, 0.0f);

                    #pragma unroll
                    for (int k_tile = 0; k_tile < num_tiles_k_qk; ++k_tile) {
                        const int k_offset = k_tile * WMMA_K;
                        if (k_offset >= D) break;

                        const __half* k_tile_ptr;
                        if constexpr (LOW_SMEM_CONTIG_FAST) {
                            k_tile_ptr =
                                low_smem_contiguous_page_tile
                                    ? K_cache_h
                                          + (int64_t)sPageIdx[0] * k_block_stride
                                          + (int64_t)(sPageOffset[0] + tile_n)
                                                * k_token_stride
                                          + (int64_t)kv_head_id * k_head_stride
                                          + k_offset
                                    : K_cache_h
                                          + (int64_t)physical_block_idx_direct
                                                * k_block_stride
                                          + (int64_t)block_offset
                                                * k_token_stride
                                          + (int64_t)kv_head_id * k_head_stride
                                          + k_offset;
                        } else {
                            k_tile_ptr =
                                K_cache_h
                                + (int64_t)physical_block_idx_direct
                                      * k_block_stride
                                + (int64_t)block_offset * k_token_stride
                                + (int64_t)kv_head_id * k_head_stride
                                + k_offset;
                        }

                        load_matrix_sync(
                            a_frag,
                            sQ + tile_m * Q_STRIDE + k_offset,
                            Q_STRIDE);
                        load_matrix_sync(b_frag, k_tile_ptr, k_token_stride);
                        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                    }

                    #pragma unroll
                    for (int i = 0; i < acc_frag.num_elements; ++i) {
                        const unsigned col =
                            col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                        const unsigned row =
                            row_causal + ((i >> 1) & 0b1) * 2;

                        const int global_m = start_row + tile_m + row;
                        const int global_n = start_col + tile_n + col;
                        const int global_q_pos = global_m + causal_q_offset;

                        const bool is_valid =
                            (global_m < start_row + valid_q_rows)
                            && (global_n < start_col + valid_k_rows);
                        bool is_causal_valid = true;
                        if constexpr (IS_CAUSAL) {
                            is_causal_valid = global_n <= global_q_pos;
                        }
                        bool is_window_valid = true;
                        if (window_size_left >= 0) {
                            is_window_valid =
                                is_window_valid
                                && global_n
                                       >= global_q_pos - window_size_left;
                        }
                        if (window_size_right >= 0) {
                            is_window_valid =
                                is_window_valid
                                && global_n
                                       <= global_q_pos + window_size_right;
                        }

                        acc_frag.x[i] =
                            (is_valid && is_causal_valid && is_window_valid)
                                ? acc_frag.x[i] * softmax_scale
                                : NEG_INF;
                    }

                    store_matrix_sync(
                        sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE,
                        mem_row_major);
                }
            }
            __syncthreads();
        } else {
        if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
            const __half* K_cache_h = reinterpret_cast<const __half*>(K_cache);
            const uint4* k_page0_vec = reinterpret_cast<const uint4*>(
                K_cache_h + (int64_t)physical_block_idx0 * k_block_stride
                + (int64_t)page_offset * k_token_stride
                + (int64_t)kv_head_id * k_head_stride);
            const uint4* k_page1_vec =
                two_page_tile && second_page_rows > 0
                    ? reinterpret_cast<const uint4*>(
                          K_cache_h + (int64_t)physical_block_idx1 * k_block_stride
                          + (int64_t)kv_head_id * k_head_stride)
                    : nullptr;

            #pragma unroll 2
            for (int idx = tid; idx < (valid_k_rows * d_stride_uint4);
                 idx += THREADS_PER_BLOCK) {
                const int row = idx / d_stride_uint4;
                const int vec_col = idx % d_stride_uint4;

                uint4 k_val = make_uint4(0, 0, 0, 0);
                if (row < valid_k_rows && vec_col < d_stride_uint4) {
                    if (four_page_contiguous_tile) {
                        k_val = __ldg(
                            &k_page0_vec[row * row_stride_uint4 + vec_col]);
                    } else if (single_page_tile) {
                        k_val = __ldg(&k_page0_vec[row * row_stride_uint4 + vec_col]);
                    } else if (two_page_tile) {
                        if (row < first_page_rows) {
                            k_val = __ldg(
                                &k_page0_vec[row * row_stride_uint4 + vec_col]);
                        } else {
                            const int row_page1 = row - first_page_rows;
                            k_val = __ldg(
                                &k_page1_vec[row_page1 * row_stride_uint4 + vec_col]);
                        }
                    } else {
                        const uint4* k_vec = reinterpret_cast<const uint4*>(K_cache_h);
                        const int global_token_idx = start_col + row;
                        const int virtual_block_idx =
                            global_token_idx / page_block_size;
                        const int block_offset = global_token_idx % page_block_size;
                        const int physical_block_idx_slow =
                            __ldg(&block_table_seq[virtual_block_idx]);
                        const int64_t physical_offset_halfs =
                            (int64_t)physical_block_idx_slow * k_block_stride
                            + (int64_t)block_offset * k_token_stride
                            + (int64_t)kv_head_id * k_head_stride;
                        const int64_t physical_offset_uint4 =
                            (physical_offset_halfs / PER_UINT4) + vec_col;
                        k_val = __ldg(&k_vec[physical_offset_uint4]);
                    }
                }
                sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
            }
        } else {
            for (int idx = tid; idx < valid_k_rows * D; idx += THREADS_PER_BLOCK) {
                const int row = idx / D;
                const int col = idx % D;
                const int global_token_idx = start_col + row;
                const int virtual_block_idx = global_token_idx / page_block_size;
                const int block_offset = global_token_idx % page_block_size;
                const int physical_block_idx_slow =
                    __ldg(&block_table_seq[virtual_block_idx]);
                const int64_t physical_offset =
                    (int64_t)physical_block_idx_slow * k_block_stride
                    + (int64_t)block_offset * k_token_stride
                    + (int64_t)kv_head_id * k_head_stride + col;
                sK[row * KV_STRIDE + col] =
                    flash_v100::load_kv_cache_half<KV_DTYPE>(
                        K_cache, physical_offset, k_scale);
            }
        }
        __syncthreads();

        const int num_tiles_m_qk    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_qk    = (BLOCK_N + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_qk    = (D + WMMA_K - 1) / WMMA_K;
        const int total_tiles_qk    = num_tiles_m_qk * num_tiles_n_qk;
        const int tiles_per_warp_qk = (total_tiles_qk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned row_causal   = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 4;
        const unsigned col_causal   = ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

        for (int tile_idx = 0; tile_idx < tiles_per_warp_qk; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp_qk + tile_idx;
            if (global_tile_idx >= total_tiles_qk) break;

            const int tile_m_idx = global_tile_idx / num_tiles_n_qk;
            const int tile_n_idx = global_tile_idx % num_tiles_n_qk;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k_qk; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= D) break;

                load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

                const int global_m = start_row + tile_m + row;
                const int global_n = start_col + tile_n + col;
                const int global_q_pos = global_m + causal_q_offset;

                const bool is_valid = (global_m < start_row + valid_q_rows) &&
                                      (global_n < start_col + valid_k_rows);
                bool is_causal_valid = true;
                if constexpr (IS_CAUSAL) {
                    is_causal_valid = global_n <= global_q_pos;
                }
                bool is_window_valid = true;
                if (window_size_left >= 0) {
                    is_window_valid = is_window_valid &&
                                      global_n >= global_q_pos - window_size_left;
                }
                if (window_size_right >= 0) {
                    is_window_valid = is_window_valid &&
                                      global_n <= global_q_pos + window_size_right;
                }

                acc_frag.x[i] = (is_valid && is_causal_valid && is_window_valid)
                    ? acc_frag.x[i] * softmax_scale
                    : NEG_INF;
            }

            store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();
        }

        uint4* sV_vec = reinterpret_cast<uint4*>(sV);
        const int64_t v_row_stride_uint4 = v_row_stride / PER_UINT4;
        if constexpr (!LOW_SMEM) {
        if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
            const __half* V_cache_h = reinterpret_cast<const __half*>(V_cache);
            const uint4* v_page0_vec = reinterpret_cast<const uint4*>(
                V_cache_h + (int64_t)physical_block_idx0 * v_block_stride
                + (int64_t)page_offset * v_token_stride
                + (int64_t)kv_head_id * v_head_stride);
            const uint4* v_page1_vec =
                two_page_tile && second_page_rows > 0
                    ? reinterpret_cast<const uint4*>(
                          V_cache_h + (int64_t)physical_block_idx1 * v_block_stride
                          + (int64_t)kv_head_id * v_head_stride)
                    : nullptr;

            #pragma unroll 2
            for (int idx = tid; idx < (valid_k_rows * d_stride_uint4);
                 idx += THREADS_PER_BLOCK) {
                const int row = idx / d_stride_uint4;
                const int vec_col = idx % d_stride_uint4;

                uint4 v_val = make_uint4(0, 0, 0, 0);
                if (row < valid_k_rows && vec_col < d_stride_uint4) {
                    if (four_page_contiguous_tile) {
                        v_val = __ldg(
                            &v_page0_vec[row * v_row_stride_uint4 + vec_col]);
                    } else if (single_page_tile) {
                        v_val = __ldg(&v_page0_vec[row * v_row_stride_uint4 + vec_col]);
                    } else if (two_page_tile) {
                        if (row < first_page_rows) {
                            v_val = __ldg(
                                &v_page0_vec[row * v_row_stride_uint4 + vec_col]);
                        } else {
                            const int row_page1 = row - first_page_rows;
                            v_val = __ldg(
                                &v_page1_vec[row_page1 * v_row_stride_uint4 + vec_col]);
                        }
                    } else {
                        const uint4* v_vec = reinterpret_cast<const uint4*>(V_cache_h);
                        const int global_token_idx = start_col + row;
                        const int virtual_block_idx =
                            global_token_idx / page_block_size;
                        const int block_offset = global_token_idx % page_block_size;
                        const int physical_block_idx_slow =
                            __ldg(&block_table_seq[virtual_block_idx]);
                        const int64_t physical_offset_halfs =
                            (int64_t)physical_block_idx_slow * v_block_stride
                            + (int64_t)block_offset * v_token_stride
                            + (int64_t)kv_head_id * v_head_stride;
                        const int64_t physical_offset_uint4 =
                            (physical_offset_halfs / PER_UINT4) + vec_col;
                        v_val = __ldg(&v_vec[physical_offset_uint4]);
                    }
                }
                sV_vec[row * kv_stride_uint4 + vec_col] = v_val;
            }
        } else {
            for (int idx = tid; idx < valid_k_rows * D; idx += THREADS_PER_BLOCK) {
                const int row = idx / D;
                const int col = idx % D;
                const int global_token_idx = start_col + row;
                const int virtual_block_idx = global_token_idx / page_block_size;
                const int block_offset = global_token_idx % page_block_size;
                const int physical_block_idx_slow =
                    __ldg(&block_table_seq[virtual_block_idx]);
                const int64_t physical_offset =
                    (int64_t)physical_block_idx_slow * v_block_stride
                    + (int64_t)block_offset * v_token_stride
                    + (int64_t)kv_head_id * v_head_stride + col;
                sV[row * KV_STRIDE + col] =
                    flash_v100::load_kv_cache_half<KV_DTYPE>(
                        V_cache, physical_offset, v_scale);
            }
        }
        __syncthreads();
        }

        const int num_tiles_m_pv = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_pv = (D + WMMA_N - 1) / WMMA_N;
        const int total_tiles_pv = num_tiles_m_pv * num_tiles_n_pv;
        const int tiles_per_warp_pv =
            (total_tiles_pv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const int softmax_sub_tile = (D == 256) ? P_SUB_TILE : p_tile_capacity;

        for (int sub_start = 0; sub_start < valid_k_rows;
             sub_start += softmax_sub_tile) {
            const int sub_valid_k_rows = min(softmax_sub_tile,
                                             valid_k_rows - sub_start);

            if (tid < valid_q_rows * THREADS_PER_ROW) {
                const int row = tid / THREADS_PER_ROW;
                const int thread_in_row = tid % THREADS_PER_ROW;
                const unsigned mask = (valid_q_rows == BLOCK_M)
                                          ? 0xFFFFFFFFU
                                          : __activemask();
                const int row_leader = __ffs(mask) - 1;

                float*  sS_row_f = sS + row * S_STRIDE + sub_start;
                __half* sP_row_h = sP + row * p_stride;

                const int vec_cols = sub_valid_k_rows >> 2;
                const int vecs_per_thread =
                    (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
                const int tail_start = vec_cols << 2;

                float thread_max = NEG_INF;
                float4* sS_vec4 = reinterpret_cast<float4*>(sS_row_f);

                #pragma unroll 4
                for (int j = 0; j < vecs_per_thread; ++j) {
                    int vc = thread_in_row + j * THREADS_PER_ROW;
                    if (vc < vec_cols) {
                        float4 v4 = sS_vec4[vc];
                        thread_max = fmaxf(
                            thread_max,
                            fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
                    }
                }

                #pragma unroll
                for (int c = tail_start + thread_in_row; c < sub_valid_k_rows;
                     c += THREADS_PER_ROW) {
                    thread_max = fmaxf(thread_max, sS_row_f[c]);
                }

                #pragma unroll
                for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
                    thread_max = fmaxf(
                        thread_max,
                        __shfl_down_sync(mask, thread_max, o, THREADS_PER_ROW));
                }

                const float row_max =
                    __shfl_sync(mask, thread_max, row_leader, THREADS_PER_ROW);
                const float old_max = sRowMax[row];
                const float new_max = fmaxf(old_max, row_max);
                const float exp_diff = __expf(old_max - new_max);

                float thread_sum = 0.0f;
                __half2 half_buffer[20];
                int vc_base = thread_in_row;
                int h2_idx = 0;
                int tail_col = -1;
                __half tail_value = __float2half(0.f);
                __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);

                #pragma unroll 4
                for (int j = 0; j < vecs_per_thread; ++j,
                         vc_base += THREADS_PER_ROW) {
                    if (vc_base < vec_cols) {
                        float4 v4 = sS_vec4[vc_base];

                        float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
                        float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
                        float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
                        float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));

                        thread_sum += (e0 + e1) + (e2 + e3);

                        const __half2 p01 =
                            __float22half2_rn(make_float2(e0, e1));
                        const __half2 p23 =
                            __float22half2_rn(make_float2(e2, e3));
                        if constexpr (D256_SW_PIPELINE_PV) {
                            // D256 owns a separate probability buffer, so the
                            // values can leave registers before row reduction.
                            const int base_offset = vc_base * 2;
                            sP_half2[base_offset] = p01;
                            sP_half2[base_offset + 1] = p23;
                        } else {
                            half_buffer[h2_idx++] = p01;
                            half_buffer[h2_idx++] = p23;
                        }
                    }
                }

                #pragma unroll 4
                for (int c = tail_start + thread_in_row; c < sub_valid_k_rows;
                     c += THREADS_PER_ROW) {
                    float v = sS_row_f[c];
                    float e = __expf(fmaxf(v - new_max, -80.0f));
                    thread_sum += e;
                    if constexpr (D256_SW_PIPELINE_PV) {
                        sP_row_h[c] = __float2half_rn(e);
                    } else {
                        tail_col = c;
                        tail_value = __float2half_rn(e);
                    }
                }

                #pragma unroll
                for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
                    thread_sum +=
                        __shfl_down_sync(mask, thread_sum, o, THREADS_PER_ROW);
                }

                float row_sum =
                    __shfl_sync(mask, thread_sum, row_leader, THREADS_PER_ROW);

                if (thread_in_row == 0) {
                    sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
                    sRowMax[row] = new_max;
                }

                if constexpr (!D256_SW_PIPELINE_PV) {
                    h2_idx = 0;
                    vc_base = thread_in_row;
                    #pragma unroll 4
                    for (int j = 0; j < vecs_per_thread; ++j,
                             vc_base += THREADS_PER_ROW) {
                        if (vc_base < vec_cols) {
                            int base_offset = vc_base * 2;
                            sP_half2[base_offset] = half_buffer[h2_idx++];
                            sP_half2[base_offset + 1] = half_buffer[h2_idx++];
                        }
                    }

                    if (tail_col >= 0) {
                        sP_row_h[tail_col] = tail_value;
                    }
                }

                #pragma unroll 4
                for (int c = tail_start + thread_in_row; c < p_tile_capacity;
                     c += THREADS_PER_ROW) {
                    if (c >= sub_valid_k_rows) {
                        sP_row_h[c] = __float2half(0.f);
                    }
                }

                if (block_n > 0 || sub_start > 0) {
                    float*  sO_row = sO + row * O_STRIDE;
                    float4* sO_vec = reinterpret_cast<float4*>(sO_row);
                    const int o_vec_count = D / 4;
                    float scale = exp_diff;

                    #pragma unroll 4
                    for (int ov = thread_in_row; ov < o_vec_count;
                         ov += THREADS_PER_ROW) {
                        float4 v = sO_vec[ov];
                        v.x *= scale;
                        v.y *= scale;
                        v.z *= scale;
                        v.w *= scale;
                        sO_vec[ov] = v;
                    }
                }
            }
            __syncthreads();

            const int num_tiles_k_pv =
                (p_tile_capacity + WMMA_K - 1) / WMMA_K;

            for (int tile_idx = 0; tile_idx < tiles_per_warp_pv; ++tile_idx) {
                const int global_tile_idx = warp_id * tiles_per_warp_pv + tile_idx;
                if (global_tile_idx >= total_tiles_pv) break;

                const int tile_m_idx = global_tile_idx / num_tiles_n_pv;
                const int tile_d_idx = global_tile_idx % num_tiles_n_pv;

                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_d = tile_d_idx * WMMA_N;

                if (tile_m >= valid_q_rows) continue;

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

                load_matrix_sync(acc_frag, sO + tile_m * O_STRIDE + tile_d,
                                 O_STRIDE, mem_row_major);

                #pragma unroll
                for (int tile_k = 0; tile_k < num_tiles_k_pv; ++tile_k) {
                    const int k_offset = tile_k * WMMA_K;
                    if (k_offset >= sub_valid_k_rows) break;

                    load_matrix_sync(a_frag,
                                     sP + tile_m * p_stride + k_offset,
                                     p_stride);
                    if constexpr (LOW_SMEM) {
                        const __half* V_cache_h =
                            reinterpret_cast<const __half*>(V_cache);
                        const int token_offset = sub_start + k_offset;
                        const int page_slot = token_offset >> 4;
                        const int block_offset = sPageOffset[page_slot];
                        const int physical_block_idx_direct =
                            sPageIdx[page_slot];
                        const __half* v_tile_ptr;
                        if constexpr (LOW_SMEM_CONTIG_FAST) {
                            v_tile_ptr =
                                low_smem_contiguous_page_tile
                                    ? V_cache_h
                                          + (int64_t)sPageIdx[0] * v_block_stride
                                          + (int64_t)(sPageOffset[0] + token_offset)
                                                * v_token_stride
                                          + (int64_t)kv_head_id * v_head_stride
                                          + tile_d
                                    : V_cache_h
                                          + (int64_t)physical_block_idx_direct
                                                * v_block_stride
                                          + (int64_t)block_offset * v_token_stride
                                          + (int64_t)kv_head_id * v_head_stride
                                          + tile_d;
                        } else {
                            v_tile_ptr =
                                V_cache_h
                                + (int64_t)physical_block_idx_direct
                                      * v_block_stride
                                + (int64_t)block_offset * v_token_stride
                                + (int64_t)kv_head_id * v_head_stride + tile_d;
                        }
                        load_matrix_sync(b_frag, v_tile_ptr, v_token_stride);
                    } else {
                        load_matrix_sync(
                            b_frag,
                            sV + (sub_start + k_offset) * KV_STRIDE + tile_d,
                            KV_STRIDE);
                    }
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                store_matrix_sync(sO + tile_m * O_STRIDE + tile_d, acc_frag,
                                  O_STRIDE, mem_row_major);
            }
            __syncthreads();
        }

    }

    if constexpr (SPLIT_KV) {
        const int partition_id = blockIdx.y;
        const int num_partitions = gridDim.y;
        const int64_t row_base =
            ((int64_t)batch_head_id * num_partitions + partition_id) * M
            + start_row;

        for (int i = tid; i < valid_q_rows * D; i += THREADS_PER_BLOCK) {
            const int row = i / D;
            const int col = i - row * D;
            split_tmp_out[(row_base + row) * D + col] =
                sO[row * O_STRIDE + col];
        }

        if (tid < valid_q_rows) {
            split_tmp_row_max[row_base + tid] = sRowMax[tid];
            split_tmp_row_sum[row_base + tid] = sRowSum[tid];
        }
        return;
    }

    const int total_fp16_x4 = (valid_q_rows * D) / 4;

    for (int i = tid; i < total_fp16_x4; i += THREADS_PER_BLOCK) {
        const int row = i / (D / 4);
        const int col = (i % (D / 4)) * 4;

        const float sum_clamped = fmaxf(sRowSum[row], 1e-24f);
        const float inv_sum = 1.0f / sum_clamped;
        const float* sO_row = sO + row * O_STRIDE;

        const __half h0 = __float2half_rn(sO_row[col + 0] * inv_sum);
        const __half h1 = __float2half_rn(sO_row[col + 1] * inv_sum);
        const __half h2 = __float2half_rn(sO_row[col + 2] * inv_sum);
        const __half h3 = __float2half_rn(sO_row[col + 3] * inv_sum);

        asm volatile(
            "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
            :
            : "l"(out_ptr + row * D + col),
              "h"(__half_as_ushort(h0)),
              "h"(__half_as_ushort(h1)),
              "h"(__half_as_ushort(h2)),
              "h"(__half_as_ushort(h3))
            : "memory"
        );
    }

    if (tid < valid_q_rows) {
        const float sum = fmaxf(sRowSum[tid], 1e-24f);
        softmax_lse_ptr[tid] = sRowMax[tid] + logf(sum);
    }
}

inline bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    return raw != nullptr && std::strcmp(raw, "0") != 0;
}

inline bool env_flag_default_enabled(const char* name) {
    const char* raw = std::getenv(name);
    return raw == nullptr || std::strcmp(raw, "0") != 0;
}

template<int D, int KV_DTYPE, bool LOW_SMEM, bool LOW_SMEM_CONTIG_FAST,
         bool LOW_SMEM_SCALAR_QK, bool LOW_SMEM_BM32,
         bool D256_OUTPUT_STRIDE_268 = false,
         bool D256_SW_PIPELINE_QK = false,
         bool D256_SW_PIPELINE_PV = false>
void launcher_flash_attention_forward_paged_impl(
    const torch::Tensor& Q,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    const int* bfla_mask_ptr,
    int bfla_mask_block_n,
    int64_t bfla_mask_stride_b,
    int64_t bfla_mask_stride_h,
    int64_t bfla_mask_stride_q,
    int64_t bfla_mask_stride_k,
    float softmax_scale,
    bool is_causal,
    float k_scale,
    float v_scale,
    int window_size_left,
    int window_size_right,
    cudaStream_t stream
) {
    using Config = KernelConfig<D, LOW_SMEM, LOW_SMEM_SCALAR_QK,
                                LOW_SMEM_BM32, D256_OUTPUT_STRIDE_268,
                                D256_SW_PIPELINE_QK,
                                D256_SW_PIPELINE_PV>;

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int page_block_size = K_cache.size(1);
    const int num_kv_heads = K_cache.size(2);
    const int max_num_blocks = block_table.size(1);
    const int N = max_num_blocks * page_block_size;
    const int64_t k_block_stride = K_cache.stride(0);
    const int64_t k_token_stride = K_cache.stride(1);
    const int64_t k_head_stride = K_cache.stride(2);
    const int64_t v_block_stride = V_cache.stride(0);
    const int64_t v_token_stride = V_cache.stride(1);
    const int64_t v_head_stride = V_cache.stride(2);

    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB: ", smem,
                " bytes");

    auto kernel = is_causal
                      ? (void*)flash_attention_forward_kernel_paged<
                            D, LOW_SMEM, LOW_SMEM_CONTIG_FAST,
                            LOW_SMEM_SCALAR_QK, LOW_SMEM_BM32, false, true,
                            KV_DTYPE, D256_OUTPUT_STRIDE_268,
                            D256_SW_PIPELINE_QK, D256_SW_PIPELINE_PV>
                      : (void*)flash_attention_forward_kernel_paged<
                            D, LOW_SMEM, LOW_SMEM_CONTIG_FAST,
                            LOW_SMEM_SCALAR_QK, LOW_SMEM_BM32, false, false,
                            KV_DTYPE, D256_OUTPUT_STRIDE_268,
                            D256_SW_PIPELINE_QK, D256_SW_PIPELINE_PV>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    if (is_causal) {
        flash_attention_forward_kernel_paged<
            D, LOW_SMEM, LOW_SMEM_CONTIG_FAST, LOW_SMEM_SCALAR_QK,
            LOW_SMEM_BM32, false, true, KV_DTYPE, D256_OUTPUT_STRIDE_268,
            D256_SW_PIPELINE_QK, D256_SW_PIPELINE_PV>
            <<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            K_cache.data_ptr(),
            V_cache.data_ptr(),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            block_table.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            B,
            H,
            M,
            N,
            bfla_mask_ptr,
            bfla_mask_block_n,
            bfla_mask_stride_b,
            bfla_mask_stride_h,
            bfla_mask_stride_q,
            bfla_mask_stride_k,
            page_block_size,
            num_kv_heads,
            k_block_stride,
            k_token_stride,
            k_head_stride,
            v_block_stride,
            v_token_stride,
            v_head_stride,
            softmax_scale,
            k_scale,
            v_scale,
            window_size_left,
            window_size_right,
            nullptr,
            nullptr,
            nullptr,
            0
        );
    } else {
        flash_attention_forward_kernel_paged<
            D, LOW_SMEM, LOW_SMEM_CONTIG_FAST, LOW_SMEM_SCALAR_QK,
            LOW_SMEM_BM32, false, false, KV_DTYPE, D256_OUTPUT_STRIDE_268,
            D256_SW_PIPELINE_QK, D256_SW_PIPELINE_PV>
            <<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            K_cache.data_ptr(),
            V_cache.data_ptr(),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            block_table.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            B,
            H,
            M,
            N,
            bfla_mask_ptr,
            bfla_mask_block_n,
            bfla_mask_stride_b,
            bfla_mask_stride_h,
            bfla_mask_stride_q,
            bfla_mask_stride_k,
            page_block_size,
            num_kv_heads,
            k_block_stride,
            k_token_stride,
            k_head_stride,
            v_block_stride,
            v_token_stride,
            v_head_stride,
            softmax_scale,
            k_scale,
            v_scale,
            window_size_left,
            window_size_right,
            nullptr,
            nullptr,
            nullptr,
            0
        );
    }
}

template<int D>
__global__ void flash_attention_forward_paged_splitkv_merge_kernel(
    const float* __restrict__ split_tmp_out,
    const float* __restrict__ split_tmp_row_max,
    const float* __restrict__ split_tmp_row_sum,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
    const int B,
    const int H,
    const int M,
    const int num_partitions
) {
    static_assert(D == 256, "split-KV paged prefill merge is D=256 only");
    constexpr int BLOCK_M = BLOCK_M_256_LOW_SMEM;
    constexpr int THREADS = 512;
    const float NEG_INF = -1e30f;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_m = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;
    const int valid_q_rows = min(BLOCK_M, M - start_row);
    const int tid = threadIdx.x;

    __shared__ float sFinalMax[BLOCK_M];
    __shared__ float sFinalSum[BLOCK_M];

    if (tid < valid_q_rows) {
        const int row = start_row + tid;
        float final_max = NEG_INF;
        #pragma unroll 1
        for (int p = 0; p < num_partitions; ++p) {
            const int64_t state_idx =
                ((int64_t)batch_head_id * num_partitions + p) * M + row;
            final_max = fmaxf(final_max, split_tmp_row_max[state_idx]);
        }

        float final_sum = 0.0f;
        #pragma unroll 1
        for (int p = 0; p < num_partitions; ++p) {
            const int64_t state_idx =
                ((int64_t)batch_head_id * num_partitions + p) * M + row;
            const float part_sum = split_tmp_row_sum[state_idx];
            if (part_sum > 0.0f) {
                final_sum +=
                    __expf(fmaxf(split_tmp_row_max[state_idx] - final_max,
                                  -80.0f)) * part_sum;
            }
        }
        sFinalMax[tid] = final_max;
        sFinalSum[tid] = final_sum;
        softmax_lse[(int64_t)batch_head_id * M + row] =
            final_max + logf(fmaxf(final_sum, 1e-24f));
    }
    __syncthreads();

    for (int idx = tid; idx < valid_q_rows * D; idx += THREADS) {
        const int row_local = idx / D;
        const int col = idx - row_local * D;
        const int row = start_row + row_local;
        const float final_max = sFinalMax[row_local];
        const float inv_sum = 1.0f / fmaxf(sFinalSum[row_local], 1e-24f);

        float acc = 0.0f;
        #pragma unroll 1
        for (int p = 0; p < num_partitions; ++p) {
            const int64_t state_idx =
                ((int64_t)batch_head_id * num_partitions + p) * M + row;
            const float part_sum = split_tmp_row_sum[state_idx];
            if (part_sum > 0.0f) {
                const float scale =
                    __expf(fmaxf(split_tmp_row_max[state_idx] - final_max,
                                  -80.0f));
                const int64_t out_idx = state_idx * D + col;
                acc = fmaf(scale, split_tmp_out[out_idx], acc);
            }
        }
        Out[((int64_t)batch_head_id * M + row) * D + col] =
            __float2half_rn(acc * inv_sum);
    }
}

template<int D, bool LOW_SMEM_CONTIG_FAST, bool LOW_SMEM_SCALAR_QK>
void launcher_flash_attention_forward_paged_splitkv_impl(
    const torch::Tensor& Q,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    torch::Tensor& split_tmp_out,
    torch::Tensor& split_tmp_row_max,
    torch::Tensor& split_tmp_row_sum,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    float softmax_scale,
    bool is_causal,
    float k_scale,
    float v_scale,
    int window_size_left,
    int window_size_right,
    int split_kv_tokens,
    int max_seq_len_hint,
    cudaStream_t stream
) {
    static_assert(D == 256, "split-KV paged prefill is D=256 only");
    using Config = KernelConfig<D, true, LOW_SMEM_SCALAR_QK>;
    constexpr int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16;

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int page_block_size = K_cache.size(1);
    const int num_kv_heads = K_cache.size(2);
    const int max_num_blocks = block_table.size(1);
    const int N = max_num_blocks * page_block_size;
    const int64_t k_block_stride = K_cache.stride(0);
    const int64_t k_token_stride = K_cache.stride(1);
    const int64_t k_head_stride = K_cache.stride(2);
    const int64_t v_block_stride = V_cache.stride(0);
    const int64_t v_token_stride = V_cache.stride(1);
    const int64_t v_head_stride = V_cache.stride(2);

    split_kv_tokens = std::max(split_kv_tokens, Config::BLOCK_N);
    max_seq_len_hint = std::max(max_seq_len_hint, 1);
    const int split_kv_tiles =
        std::max(1, (split_kv_tokens + Config::BLOCK_N - 1) / Config::BLOCK_N);
    const int max_kv_tiles =
        std::max(1, (max_seq_len_hint + Config::BLOCK_N - 1) / Config::BLOCK_N);
    const int num_partitions =
        std::max(1, (max_kv_tiles + split_kv_tiles - 1) / split_kv_tiles);

    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, num_partitions, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB: ", smem,
                " bytes");
    TORCH_CHECK(split_tmp_out.size(2) == num_partitions,
                "split_tmp_out partition mismatch");
    TORCH_CHECK(split_tmp_row_max.size(2) == num_partitions,
                "split_tmp_row_max partition mismatch");
    TORCH_CHECK(split_tmp_row_sum.size(2) == num_partitions,
                "split_tmp_row_sum partition mismatch");

    auto kernel = is_causal
                      ? (void*)flash_attention_forward_kernel_paged<
                            D, true, LOW_SMEM_CONTIG_FAST,
                            LOW_SMEM_SCALAR_QK, false, true, true, KV_DTYPE>
                      : (void*)flash_attention_forward_kernel_paged<
                            D, true, LOW_SMEM_CONTIG_FAST,
                            LOW_SMEM_SCALAR_QK, false, true, false, KV_DTYPE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    if (is_causal) {
        flash_attention_forward_kernel_paged<
            D, true, LOW_SMEM_CONTIG_FAST, LOW_SMEM_SCALAR_QK, false, true, true,
            KV_DTYPE>
            <<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            K_cache.data_ptr(),
            V_cache.data_ptr(),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            block_table.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            B,
            H,
            M,
            N,
            nullptr,
            0,
            0,
            0,
            0,
            0,
            page_block_size,
            num_kv_heads,
            k_block_stride,
            k_token_stride,
            k_head_stride,
            v_block_stride,
            v_token_stride,
            v_head_stride,
            softmax_scale,
            k_scale,
            v_scale,
            window_size_left,
            window_size_right,
            split_tmp_out.data_ptr<float>(),
            split_tmp_row_max.data_ptr<float>(),
            split_tmp_row_sum.data_ptr<float>(),
            split_kv_tiles
        );
    } else {
        flash_attention_forward_kernel_paged<
            D, true, LOW_SMEM_CONTIG_FAST, LOW_SMEM_SCALAR_QK, false, true, false,
            KV_DTYPE>
            <<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            K_cache.data_ptr(),
            V_cache.data_ptr(),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            block_table.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            B,
            H,
            M,
            N,
            nullptr,
            0,
            0,
            0,
            0,
            0,
            page_block_size,
            num_kv_heads,
            k_block_stride,
            k_token_stride,
            k_head_stride,
            v_block_stride,
            v_token_stride,
            v_head_stride,
            softmax_scale,
            k_scale,
            v_scale,
            window_size_left,
            window_size_right,
            split_tmp_out.data_ptr<float>(),
            split_tmp_row_max.data_ptr<float>(),
            split_tmp_row_sum.data_ptr<float>(),
            split_kv_tiles
        );
    }

    const dim3 merge_grid(grid_x, 1, B * H);
    const dim3 merge_block(512);
    flash_attention_forward_paged_splitkv_merge_kernel<D>
        <<<merge_grid, merge_block, 0, stream>>>(
            split_tmp_out.data_ptr<float>(),
            split_tmp_row_max.data_ptr<float>(),
            split_tmp_row_sum.data_ptr<float>(),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            B,
            H,
            M,
            num_partitions);
}

constexpr int D256_BM32_PHASE_BLOCK_M = 32;
constexpr int D256_BM32_PHASE_BLOCK_N = 128;
constexpr int D256_BM32_PHASE_PANEL_M = 16;
constexpr int D256_BM32_PHASE_SOFTMAX_N = 32;
constexpr int D256_BM32_PHASE_D = 256;
constexpr int D256_BM32_PHASE_THREADS = 512;
constexpr int D256_BM32_PHASE_PAGE_SIZE = 16;
constexpr int D256_BM32_PHASE_PAGE_BLOCK_SIZE = 784;
constexpr int D256_BM32_PHASE_SPLIT_PARTS = 3;
constexpr int D256_BM32_PHASE_PAGE_SLOTS =
    D256_BM32_PHASE_BLOCK_N / D256_BM32_PHASE_PAGE_SIZE;
constexpr int D256_BM32_PHASE_PANELS =
    D256_BM32_PHASE_BLOCK_N / D256_BM32_PHASE_SOFTMAX_N;
constexpr int D256_BM32_PHASE_PROBABILITY_ELEMENTS =
    D256_BM32_PHASE_PANEL_M * D256_BM32_PHASE_SOFTMAX_N;

template<int PROBABILITY_PANELS>
struct alignas(16) D256BM32PhaseSharedStorage {
    __half query[D256_BM32_PHASE_BLOCK_M * D256_BM32_PHASE_D];
    float score[D256_BM32_PHASE_BLOCK_M * D256_BM32_PHASE_BLOCK_N];
    __half probability_top[PROBABILITY_PANELS
                           * D256_BM32_PHASE_PROBABILITY_ELEMENTS];
    __half probability_bottom[PROBABILITY_PANELS
                              * D256_BM32_PHASE_PROBABILITY_ELEMENTS];
    float row_max[D256_BM32_PHASE_BLOCK_M];
    float row_sum[D256_BM32_PHASE_BLOCK_M];
    float row_exp_diff[PROBABILITY_PANELS * D256_BM32_PHASE_BLOCK_M];
    int page_idx[D256_BM32_PHASE_PAGE_SLOTS];
    int page_offset[D256_BM32_PHASE_PAGE_SLOTS];
    uint64_t k_tile_ptr[D256_BM32_PHASE_PAGE_SLOTS];
    uint64_t v_tile_ptr[D256_BM32_PHASE_PAGE_SLOTS];
    int block_index;
    int batch_id;
    int kv_head_id;
    int actual_n;
};

template<int PROBABILITY_PANELS>
struct alignas(16) D256BM32PhaseSplitSharedStorage
    : D256BM32PhaseSharedStorage<PROBABILITY_PANELS> {
    int split_end_block;
};

template<bool SPLIT_KV, int PROBABILITY_PANELS>
struct D256BM32PhaseSharedStorageSelector {
    using Type = D256BM32PhaseSharedStorage<PROBABILITY_PANELS>;
};

template<int PROBABILITY_PANELS>
struct D256BM32PhaseSharedStorageSelector<true, PROBABILITY_PANELS> {
    using Type = D256BM32PhaseSplitSharedStorage<PROBABILITY_PANELS>;
};

using D256BM32PhaseSharedStorageSingle = D256BM32PhaseSharedStorage<1>;
using D256BM32PhaseSharedStorageAllP =
    D256BM32PhaseSharedStorage<D256_BM32_PHASE_PANELS>;

static_assert(sizeof(D256BM32PhaseSharedStorageSingle) == 35408,
              "D256 BM32 phase shared layout changed unexpectedly");
static_assert(sizeof(D256BM32PhaseSharedStorageAllP) == 41936,
              "D256 BM32 all-P shared layout changed unexpectedly");
static_assert(sizeof(D256BM32PhaseSharedStorageAllP) <= 48 * 1024,
              "D256 BM32 phase shared storage exceeds 48 KiB");
static_assert(sizeof(D256BM32PhaseSplitSharedStorage<
                      D256_BM32_PHASE_PANELS>) == 41952,
              "D256 BM32 split shared layout changed unexpectedly");

using D256BM32PhaseMatrixAFragment =
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major>;
using D256BM32PhaseQKMatrixBFragment =
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major>;
using D256BM32PhasePVMatrixBFragment =
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major>;
using D256BM32PhaseAccumulatorFragment =
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

__device__ __forceinline__ int d256_bm32_phase_swizzled_row_slot(int row) {
    return (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
}

__device__ __forceinline__ int d256_bm32_phase_matrix_a_offset(
    int row,
    int column) {
    const int tile = column / WMMA_K;
    const int within_tile = column - tile * WMMA_K;
    const int plane = within_tile >> 3;
    const int inner = within_tile & 7;
    return tile * WMMA_M * WMMA_K + plane * WMMA_M * 8
           + d256_bm32_phase_swizzled_row_slot(row) * 8 + inner;
}

__device__ __forceinline__ void d256_bm32_phase_stage_query(
    const __half* __restrict__ source,
    __half* __restrict__ destination) {
    constexpr int HALF_PER_UINT4 = sizeof(uint4) / sizeof(__half);
    constexpr int VECTORS_PER_ROW = D256_BM32_PHASE_D / HALF_PER_UINT4;
    constexpr int VECTORS_PER_PANEL =
        D256_BM32_PHASE_PANEL_M * VECTORS_PER_ROW;
    constexpr int QUERY_VECTORS =
        D256_BM32_PHASE_BLOCK_M * VECTORS_PER_ROW;

    const uint4* source_vectors = reinterpret_cast<const uint4*>(source);
    uint4* destination_vectors = reinterpret_cast<uint4*>(destination);

    #pragma unroll
    for (int index = threadIdx.x; index < QUERY_VECTORS;
         index += D256_BM32_PHASE_THREADS) {
        const int row = index / VECTORS_PER_ROW;
        const int vector_column = index % VECTORS_PER_ROW;
        const int panel = row / D256_BM32_PHASE_PANEL_M;
        const int panel_row = row - panel * D256_BM32_PHASE_PANEL_M;
        const int k_tile = vector_column >> 1;
        const int plane = vector_column & 1;
        const int slot = d256_bm32_phase_swizzled_row_slot(panel_row);
        destination_vectors[panel * VECTORS_PER_PANEL
                            + k_tile * (2 * D256_BM32_PHASE_PANEL_M)
                            + plane * D256_BM32_PHASE_PANEL_M + slot] =
            __ldg(source_vectors + index);
    }
}

__device__ __forceinline__ void d256_bm32_phase_stage_query_partial(
    const __half* __restrict__ source,
    __half* __restrict__ destination,
    int valid_rows) {
    constexpr int HALF_PER_UINT4 = sizeof(uint4) / sizeof(__half);
    constexpr int VECTORS_PER_ROW = D256_BM32_PHASE_D / HALF_PER_UINT4;
    constexpr int VECTORS_PER_PANEL =
        D256_BM32_PHASE_PANEL_M * VECTORS_PER_ROW;
    constexpr int QUERY_VECTORS =
        D256_BM32_PHASE_BLOCK_M * VECTORS_PER_ROW;

    const uint4* source_vectors = reinterpret_cast<const uint4*>(source);
    uint4* destination_vectors = reinterpret_cast<uint4*>(destination);

    #pragma unroll
    for (int index = threadIdx.x; index < QUERY_VECTORS;
         index += D256_BM32_PHASE_THREADS) {
        const int row = index / VECTORS_PER_ROW;
        const int vector_column = index % VECTORS_PER_ROW;
        const int panel = row / D256_BM32_PHASE_PANEL_M;
        const int panel_row = row - panel * D256_BM32_PHASE_PANEL_M;
        const int k_tile = vector_column >> 1;
        const int plane = vector_column & 1;
        const int slot = d256_bm32_phase_swizzled_row_slot(panel_row);
        uint4 value = {0, 0, 0, 0};
        if (row < valid_rows) {
            value = __ldg(source_vectors + index);
        }
        destination_vectors[panel * VECTORS_PER_PANEL
                            + k_tile * (2 * D256_BM32_PHASE_PANEL_M)
                            + plane * D256_BM32_PHASE_PANEL_M + slot] = value;
    }
}

__device__ __forceinline__ void d256_bm32_phase_load_matrix_a(
    D256BM32PhaseMatrixAFragment& fragment,
    const __half* __restrict__ matrix,
    int k_offset) {
    const int lane = threadIdx.x & 31;
    const int row =
        (lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8;
    const int slot = d256_bm32_phase_swizzled_row_slot(row);
    const int tile_offset = (k_offset / WMMA_K) * WMMA_M * WMMA_K;
    uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(matrix + tile_offset + slot * 8));
    uint32_t* words = reinterpret_cast<uint32_t*>(&fragment);
    asm volatile(
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3])
        : "r"(address)
        : "memory");
    address += WMMA_M * 8 * sizeof(__half);
    asm volatile(
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(words[4]), "=r"(words[5]), "=r"(words[6]), "=r"(words[7])
        : "r"(address)
        : "memory");
}

__device__ __forceinline__ int d256_bm32_phase_accumulator_row(
    int lane,
    int element) {
    const int row_base =
        (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
    return row_base + ((element >> 1) & 1) * 2;
}

__device__ __forceinline__ int d256_bm32_phase_accumulator_column(
    int lane,
    int element) {
    const int column_base = ((lane >> 1) & 1) * 2 + ((lane >> 3) & 1) * 8;
    return column_base + (element & 1) + ((element >> 2) & 1) * 4;
}

__device__ __forceinline__ void d256_bm32_phase_spill_pair_scratch(
    float* __restrict__ score,
    D256BM32PhaseAccumulatorFragment& accumulator_top,
    D256BM32PhaseAccumulatorFragment& accumulator_bottom) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_pair = warp >> 1;
    const int warp_in_pair = warp & 1;
    const int pair_column = warp_pair * 32 + (lane & 15) * 2;
    const int lane_row = lane >> 4;

    #pragma unroll
    for (int element_pair = 0; element_pair < 4; ++element_pair) {
        const int element = element_pair * 2;
        const int offset =
            (warp_in_pair * 16 + element_pair * 2 + lane_row)
                * D256_BM32_PHASE_BLOCK_N
            + pair_column;
        const uint32_t address = static_cast<uint32_t>(
            __cvta_generic_to_shared(score + offset));
        asm volatile(
            "st.shared.v2.u32 [%0], {%1, %2};"
            :
            : "r"(address),
              "r"(__float_as_uint(accumulator_top.x[element])),
              "r"(__float_as_uint(accumulator_top.x[element + 1]))
            : "memory");
        asm volatile(
            "st.shared.v2.u32 [%0+4096], {%1, %2};"
            :
            : "r"(address),
              "r"(__float_as_uint(accumulator_bottom.x[element])),
              "r"(__float_as_uint(accumulator_bottom.x[element + 1]))
            : "memory");
    }
}

__device__ __forceinline__ void d256_bm32_phase_reload_pair_scratch(
    const float* __restrict__ score,
    D256BM32PhaseAccumulatorFragment& accumulator_top,
    D256BM32PhaseAccumulatorFragment& accumulator_bottom) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_pair = warp >> 1;
    const int warp_in_pair = warp & 1;
    const int pair_column = warp_pair * 32 + (lane & 15) * 2;
    const int lane_row = lane >> 4;

    #pragma unroll
    for (int element_pair = 0; element_pair < 4; ++element_pair) {
        const int element = element_pair * 2;
        const int offset =
            (warp_in_pair * 16 + element_pair * 2 + lane_row)
                * D256_BM32_PHASE_BLOCK_N
            + pair_column;
        const uint32_t address = static_cast<uint32_t>(
            __cvta_generic_to_shared(score + offset));
        uint32_t first_word;
        uint32_t second_word;
        asm volatile(
            "ld.shared.v2.u32 {%0, %1}, [%2];"
            : "=r"(first_word), "=r"(second_word)
            : "r"(address)
            : "memory");
        accumulator_top.x[element] = __uint_as_float(first_word);
        accumulator_top.x[element + 1] = __uint_as_float(second_word);
        asm volatile(
            "ld.shared.v2.u32 {%0, %1}, [%2+4096];"
            : "=r"(first_word), "=r"(second_word)
            : "r"(address)
            : "memory");
        accumulator_bottom.x[element] = __uint_as_float(first_word);
        accumulator_bottom.x[element + 1] = __uint_as_float(second_word);
    }
}

__device__ __forceinline__ void d256_bm32_phase_sync_warp_pair(int warp_pair) {
    const int barrier_id = warp_pair + 1;
    asm volatile("bar.sync %0, 64;" : : "r"(barrier_id) : "memory");
}

__device__ __forceinline__ float d256_bm32_phase_make_probability_row(
    const float* __restrict__ score_row,
    __half* __restrict__ probability,
    int probability_row,
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    int state_row,
    int panel,
    float neg_inf) {
    const int lane = threadIdx.x & 31;
    const float* panel_score =
        score_row + panel * D256_BM32_PHASE_SOFTMAX_N;
    float thread_max = neg_inf;
    if (lane < D256_BM32_PHASE_SOFTMAX_N / 4) {
        const float4 values =
            reinterpret_cast<const float4*>(panel_score)[lane];
        thread_max = fmaxf(fmaxf(values.x, values.y),
                           fmaxf(values.z, values.w));
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(
            thread_max, __shfl_down_sync(0xffffffffU, thread_max, offset));
    }
    const float panel_max = __shfl_sync(0xffffffffU, thread_max, 0);
    const float old_max = row_max[state_row];
    const float new_max = fmaxf(old_max, panel_max);
    const float exp_diff = __expf(old_max - new_max);

    float thread_sum = 0.0f;
    if (lane < D256_BM32_PHASE_SOFTMAX_N / 4) {
        float4 values = reinterpret_cast<const float4*>(panel_score)[lane];
        values.x = __expf(fmaxf(values.x - new_max, -80.0f));
        values.y = __expf(fmaxf(values.y - new_max, -80.0f));
        values.z = __expf(fmaxf(values.z - new_max, -80.0f));
        values.w = __expf(fmaxf(values.w - new_max, -80.0f));
        thread_sum = (values.x + values.y) + (values.z + values.w);
        const int column = lane * 4;
        __half2* first_pair = reinterpret_cast<__half2*>(
            probability + d256_bm32_phase_matrix_a_offset(
                              probability_row, column));
        __half2* second_pair = reinterpret_cast<__half2*>(
            probability + d256_bm32_phase_matrix_a_offset(
                              probability_row, column + 2));
        *first_pair = __float22half2_rn(make_float2(values.x, values.y));
        *second_pair = __float22half2_rn(make_float2(values.z, values.w));
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum +=
            __shfl_down_sync(0xffffffffU, thread_sum, offset);
    }
    const float panel_sum = __shfl_sync(0xffffffffU, thread_sum, 0);
    if (lane == 0) {
        row_sum[state_row] = exp_diff * row_sum[state_row] + panel_sum;
        row_max[state_row] = new_max;
    }
    return exp_diff;
}

__device__ __forceinline__ float
d256_bm32_phase_make_split_probability_row(
    const float* __restrict__ score_row,
    __half* __restrict__ probability,
    int probability_row,
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    int state_row,
    int panel,
    float neg_inf,
    bool has_visible_value) {
    if (!has_visible_value) {
        const int lane = threadIdx.x & 31;
        if (lane < D256_BM32_PHASE_SOFTMAX_N / 4) {
            const int column = lane * 4;
            __half2* first_pair = reinterpret_cast<__half2*>(
                probability + d256_bm32_phase_matrix_a_offset(
                                  probability_row, column));
            __half2* second_pair = reinterpret_cast<__half2*>(
                probability + d256_bm32_phase_matrix_a_offset(
                                  probability_row, column + 2));
            const __half2 zero = __float2half2_rn(0.0f);
            *first_pair = zero;
            *second_pair = zero;
        }
        return 1.0f;
    }
    return d256_bm32_phase_make_probability_row(
        score_row, probability, probability_row, row_max, row_sum,
        state_row, panel, neg_inf);
}

__device__ __forceinline__ void d256_bm32_phase_scale_accumulator_rows(
    D256BM32PhaseAccumulatorFragment& accumulator,
    float first_row_scale,
    float second_row_scale) {
    accumulator.x[0] *= first_row_scale;
    accumulator.x[1] *= first_row_scale;
    accumulator.x[2] *= second_row_scale;
    accumulator.x[3] *= second_row_scale;
    accumulator.x[4] *= first_row_scale;
    accumulator.x[5] *= first_row_scale;
    accumulator.x[6] *= second_row_scale;
    accumulator.x[7] *= second_row_scale;
}

__device__ __forceinline__ void d256_bm32_phase_scale_accumulators(
    D256BM32PhaseAccumulatorFragment& accumulator_top,
    D256BM32PhaseAccumulatorFragment& accumulator_bottom,
    const float* __restrict__ row_exp_diff) {
    const int row = d256_bm32_phase_accumulator_row(threadIdx.x & 31, 0);
    d256_bm32_phase_scale_accumulator_rows(
        accumulator_top, row_exp_diff[row], row_exp_diff[row + 2]);
    d256_bm32_phase_scale_accumulator_rows(
        accumulator_bottom,
        row_exp_diff[D256_BM32_PHASE_PANEL_M + row],
        row_exp_diff[D256_BM32_PHASE_PANEL_M + row + 2]);
}

__device__ __forceinline__ void d256_bm32_phase_update_pv_panel(
    const __half* __restrict__ probability_top,
    const __half* __restrict__ probability_bottom,
    const __half* __restrict__ value_k0,
    const __half* __restrict__ value_k16,
    int64_t value_token_stride,
    int d_offset,
    int valid_columns,
    D256BM32PhaseAccumulatorFragment& accumulator_top,
    D256BM32PhaseAccumulatorFragment& accumulator_bottom) {
    D256BM32PhaseMatrixAFragment a_fragment;
    D256BM32PhasePVMatrixBFragment b_fragment;

    load_matrix_sync(b_fragment, value_k0 + d_offset, value_token_stride);
    d256_bm32_phase_load_matrix_a(a_fragment, probability_top, 0);
    mma_sync(accumulator_top, a_fragment, b_fragment, accumulator_top);
    d256_bm32_phase_load_matrix_a(a_fragment, probability_bottom, 0);
    mma_sync(accumulator_bottom, a_fragment, b_fragment, accumulator_bottom);
    if (valid_columns > WMMA_K) {
        load_matrix_sync(
            b_fragment, value_k16 + d_offset, value_token_stride);
        d256_bm32_phase_load_matrix_a(a_fragment, probability_top, WMMA_K);
        mma_sync(accumulator_top, a_fragment, b_fragment, accumulator_top);
        d256_bm32_phase_load_matrix_a(a_fragment, probability_bottom, WMMA_K);
        mma_sync(accumulator_bottom, a_fragment, b_fragment,
                 accumulator_bottom);
    }
}

__device__ __forceinline__ void d256_bm32_phase_store_output(
    const D256BM32PhaseAccumulatorFragment& accumulator,
    __half* __restrict__ output,
    const float* __restrict__ row_sum,
    int row_offset,
    int d_offset) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int element = 0; element < accumulator.num_elements; ++element) {
        const int row = row_offset
                        + d256_bm32_phase_accumulator_row(lane, element);
        const int column = d_offset
                           + d256_bm32_phase_accumulator_column(lane, element);
        const float inverse_sum = 1.0f / fmaxf(row_sum[row], 1e-24f);
        output[row * D256_BM32_PHASE_D + column] =
            __float2half_rn(accumulator.x[element] * inverse_sum);
    }
}

__device__ __forceinline__ void d256_bm32_phase_store_output_partial(
    const D256BM32PhaseAccumulatorFragment& accumulator,
    __half* __restrict__ output,
    const float* __restrict__ row_sum,
    int row_offset,
    int d_offset,
    int valid_rows) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int element = 0; element < accumulator.num_elements; ++element) {
        const int row = row_offset
                        + d256_bm32_phase_accumulator_row(lane, element);
        if (row < valid_rows) {
            const int column =
                d_offset
                + d256_bm32_phase_accumulator_column(lane, element);
            const float inverse_sum = 1.0f / fmaxf(row_sum[row], 1e-24f);
            output[row * D256_BM32_PHASE_D + column] =
                __float2half_rn(accumulator.x[element] * inverse_sum);
        }
    }
}

__device__ __forceinline__ void d256_bm32_phase_store_unnormalized_partial(
    const D256BM32PhaseAccumulatorFragment& accumulator,
    float* __restrict__ output,
    int row_offset,
    int d_offset,
    int valid_rows) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int element = 0; element < accumulator.num_elements; ++element) {
        const int row = row_offset
                        + d256_bm32_phase_accumulator_row(lane, element);
        if (row < valid_rows) {
            const int column =
                d_offset
                + d256_bm32_phase_accumulator_column(lane, element);
            output[row * D256_BM32_PHASE_D + column] = accumulator.x[element];
        }
    }
}

template<bool IS_CAUSAL, bool ALL_P = false, bool PAIR_SCRATCH = false,
         bool SPLIT_KV = false, bool CHECK_SPLIT_EMPTY = true>
__device__ __forceinline__
void flash_attention_forward_paged_d256_bm32_phase_body(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
    const int* __restrict__ block_table,
    const int* __restrict__ seqused_k,
    int H,
    int M,
    int max_num_blocks_per_seq,
    int num_kv_heads,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    float softmax_scale,
    float* __restrict__ split_tmp_out,
    float* __restrict__ split_tmp_row_max,
    float* __restrict__ split_tmp_row_sum,
    int split_actual_n) {
    constexpr float NEG_INF = -1e30f;
    static_assert(!SPLIT_KV || (ALL_P && PAIR_SCRATCH),
                  "BM32 split-KV requires the accepted all-P pair-scratch body");
    __shared__ __align__(16)
        typename D256BM32PhaseSharedStorageSelector<
            SPLIT_KV,
            ALL_P ? D256_BM32_PHASE_PANELS : 1>::Type shared;

    const int start_row = blockIdx.x * D256_BM32_PHASE_BLOCK_M;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int valid_q_rows =
        min(D256_BM32_PHASE_BLOCK_M, M - start_row);

    {
        const __half* q_ptr =
            Q + static_cast<size_t>(blockIdx.z) * M * D256_BM32_PHASE_D
            + start_row * D256_BM32_PHASE_D;
        if (valid_q_rows == D256_BM32_PHASE_BLOCK_M) {
            d256_bm32_phase_stage_query(q_ptr, shared.query);
        } else {
            d256_bm32_phase_stage_query_partial(
                q_ptr, shared.query, valid_q_rows);
        }
    }
    if (tid < D256_BM32_PHASE_BLOCK_M) {
        shared.row_max[tid] = NEG_INF;
        shared.row_sum[tid] = 0.0f;
    }
    if (tid == 0) {
        shared.block_index = 0;
        shared.batch_id = blockIdx.z / H;
        shared.kv_head_id =
            (blockIdx.z % H) / (H / num_kv_heads);
        if constexpr (SPLIT_KV) {
            shared.actual_n = split_actual_n;
        } else {
            shared.actual_n = seqused_k[shared.batch_id];
        }
    }
    __syncthreads();

    const int total_n_blocks =
        (shared.actual_n + D256_BM32_PHASE_BLOCK_N - 1)
        / D256_BM32_PHASE_BLOCK_N;
    if constexpr (SPLIT_KV) {
        if (tid == 0) {
            shared.block_index =
                static_cast<int>(blockIdx.y) * total_n_blocks
                / D256_BM32_PHASE_SPLIT_PARTS;
            shared.split_end_block =
                (static_cast<int>(blockIdx.y) + 1) * total_n_blocks
                / D256_BM32_PHASE_SPLIT_PARTS;
        }
        __syncthreads();
    }

    D256BM32PhaseAccumulatorFragment accumulator_top;
    D256BM32PhaseAccumulatorFragment accumulator_bottom;
    fill_fragment(accumulator_top, 0.0f);
    fill_fragment(accumulator_bottom, 0.0f);

    for (;;) {
        if constexpr (SPLIT_KV) {
            if (shared.block_index >= shared.split_end_block) {
                break;
            }
        } else {
            if (shared.block_index >= total_n_blocks) {
                break;
            }
        }
        const int start_col =
            shared.block_index * D256_BM32_PHASE_BLOCK_N;
        const int valid_k_rows = min(D256_BM32_PHASE_BLOCK_N,
                                     shared.actual_n - start_col);

        if (tid < D256_BM32_PHASE_PAGE_SLOTS) {
            const int token_offset = tid * D256_BM32_PHASE_PAGE_SIZE;
            if (token_offset < valid_k_rows) {
                const int global_token_idx = start_col + token_offset;
                const int virtual_block_idx =
                    global_token_idx / D256_BM32_PHASE_PAGE_BLOCK_SIZE;
                shared.page_idx[tid] =
                    __ldg(&block_table[shared.batch_id
                                       * max_num_blocks_per_seq
                                       + virtual_block_idx]);
                shared.page_offset[tid] =
                    global_token_idx
                    - virtual_block_idx * D256_BM32_PHASE_PAGE_BLOCK_SIZE;
                shared.k_tile_ptr[tid] = reinterpret_cast<uint64_t>(
                    K_cache
                    + (int64_t)shared.page_idx[tid] * k_block_stride
                    + (int64_t)shared.page_offset[tid] * k_token_stride
                    + (int64_t)shared.kv_head_id * k_head_stride);
                shared.v_tile_ptr[tid] = reinterpret_cast<uint64_t>(
                    V_cache
                    + (int64_t)shared.page_idx[tid] * v_block_stride
                    + (int64_t)shared.page_offset[tid] * v_token_stride
                    + (int64_t)shared.kv_head_id * v_head_stride);
            } else {
                shared.page_idx[tid] = -1;
                shared.page_offset[tid] = 0;
                shared.k_tile_ptr[tid] = 0;
                shared.v_tile_ptr[tid] = 0;
            }
        }
        __syncthreads();

        if (warp_id < D256_BM32_PHASE_BLOCK_N / WMMA_N
            && warp_id * WMMA_N < valid_k_rows) {
            const int tile_n = warp_id * WMMA_N;
            const __half* k_tile =
                reinterpret_cast<const __half*>(shared.k_tile_ptr[warp_id]);
            if constexpr (PAIR_SCRATCH) {
                d256_bm32_phase_spill_pair_scratch(
                    shared.score, accumulator_top, accumulator_bottom);
            } else {
                store_matrix_sync(shared.score + tile_n, accumulator_top,
                                  D256_BM32_PHASE_BLOCK_N, mem_row_major);
                store_matrix_sync(
                    shared.score + D256_BM32_PHASE_PANEL_M
                                       * D256_BM32_PHASE_BLOCK_N + tile_n,
                    accumulator_bottom, D256_BM32_PHASE_BLOCK_N,
                    mem_row_major);
            }
            asm volatile("" ::: "memory");

            D256BM32PhaseAccumulatorFragment qk_top;
            D256BM32PhaseAccumulatorFragment qk_bottom;
            {
                D256BM32PhaseMatrixAFragment a_fragment;
                D256BM32PhaseQKMatrixBFragment b_fragment;
                fill_fragment(qk_top, 0.0f);
                fill_fragment(qk_bottom, 0.0f);

                #pragma unroll
                for (int k_offset = 0; k_offset < D256_BM32_PHASE_D;
                     k_offset += WMMA_K) {
                    load_matrix_sync(b_fragment, k_tile + k_offset,
                                     k_token_stride);
                    d256_bm32_phase_load_matrix_a(
                        a_fragment, shared.query, k_offset);
                    mma_sync(qk_top, a_fragment, b_fragment, qk_top);
                    d256_bm32_phase_load_matrix_a(
                        a_fragment,
                        shared.query
                            + D256_BM32_PHASE_PANEL_M
                                  * D256_BM32_PHASE_D,
                        k_offset);
                    mma_sync(qk_bottom, a_fragment, b_fragment, qk_bottom);
                }
            }
            asm volatile("" ::: "memory");

            if constexpr (PAIR_SCRATCH) {
                d256_bm32_phase_reload_pair_scratch(
                    shared.score, accumulator_top, accumulator_bottom);
                const int partner_warp = warp_id ^ 1;
                if (partner_warp * WMMA_N < valid_k_rows) {
                    d256_bm32_phase_sync_warp_pair(warp_id >> 1);
                }
            } else {
                load_matrix_sync(accumulator_top, shared.score + tile_n,
                                 D256_BM32_PHASE_BLOCK_N, mem_row_major);
                load_matrix_sync(
                    accumulator_bottom,
                    shared.score + D256_BM32_PHASE_PANEL_M
                                       * D256_BM32_PHASE_BLOCK_N + tile_n,
                    D256_BM32_PHASE_BLOCK_N, mem_row_major);
            }

            store_matrix_sync(shared.score + tile_n, qk_top,
                              D256_BM32_PHASE_BLOCK_N, mem_row_major);
            store_matrix_sync(
                shared.score + D256_BM32_PHASE_PANEL_M
                                   * D256_BM32_PHASE_BLOCK_N + tile_n,
                qk_bottom, D256_BM32_PHASE_BLOCK_N, mem_row_major);
        }
        __syncthreads();

        const int causal_q_offset = max(shared.actual_n - M, 0);
        #pragma unroll 1
        for (int index = tid;
             index < D256_BM32_PHASE_BLOCK_M * D256_BM32_PHASE_BLOCK_N;
             index += D256_BM32_PHASE_THREADS) {
            const int row = index / D256_BM32_PHASE_BLOCK_N;
            const int column = index - row * D256_BM32_PHASE_BLOCK_N;
            const int global_m = start_row + row;
            const int global_n = start_col + column;
            const bool is_valid = global_m < M
                                  && global_n < start_col + valid_k_rows;
            if constexpr (IS_CAUSAL) {
                if (is_valid && global_n <= global_m + causal_q_offset) {
                    shared.score[index] *= softmax_scale;
                } else {
                    shared.score[index] = NEG_INF;
                }
            } else {
                if (is_valid) {
                    shared.score[index] *= softmax_scale;
                } else {
                    shared.score[index] = NEG_INF;
                }
            }
        }
        __syncthreads();

        if constexpr (ALL_P) {
            #pragma unroll
            for (int panel = 0; panel < D256_BM32_PHASE_PANELS;
                 ++panel) {
                const int panel_start =
                    panel * D256_BM32_PHASE_SOFTMAX_N;
                const int valid_panel_columns =
                    min(D256_BM32_PHASE_SOFTMAX_N,
                        valid_k_rows - panel_start);
                if (valid_panel_columns > 0) {
                    __half* probability_top =
                        shared.probability_top
                        + panel * D256_BM32_PHASE_PROBABILITY_ELEMENTS;
                    __half* probability_bottom =
                        shared.probability_bottom
                        + panel * D256_BM32_PHASE_PROBABILITY_ELEMENTS;
                    float* row_exp_diff =
                        shared.row_exp_diff
                        + panel * D256_BM32_PHASE_BLOCK_M;
                    float top_exp_diff;
                    float bottom_exp_diff;
                    if constexpr (SPLIT_KV && CHECK_SPLIT_EMPTY) {
                        const int first_global_n = start_col + panel_start;
                        const bool top_has_visible_value =
                            warp_id < valid_q_rows
                            && (!IS_CAUSAL
                                || first_global_n
                                       <= start_row + warp_id
                                              + causal_q_offset);
                        const bool bottom_has_visible_value =
                            D256_BM32_PHASE_PANEL_M + warp_id < valid_q_rows
                            && (!IS_CAUSAL
                                || first_global_n
                                       <= start_row
                                              + D256_BM32_PHASE_PANEL_M
                                              + warp_id + causal_q_offset);
                        top_exp_diff =
                            d256_bm32_phase_make_split_probability_row(
                                shared.score
                                    + warp_id * D256_BM32_PHASE_BLOCK_N,
                                probability_top, warp_id, shared.row_max,
                                shared.row_sum, warp_id, panel, NEG_INF,
                                top_has_visible_value);
                        bottom_exp_diff =
                            d256_bm32_phase_make_split_probability_row(
                                shared.score
                                    + (D256_BM32_PHASE_PANEL_M + warp_id)
                                          * D256_BM32_PHASE_BLOCK_N,
                                probability_bottom, warp_id, shared.row_max,
                                shared.row_sum,
                                D256_BM32_PHASE_PANEL_M + warp_id, panel,
                                NEG_INF, bottom_has_visible_value);
                    } else {
                        top_exp_diff = d256_bm32_phase_make_probability_row(
                            shared.score
                                + warp_id * D256_BM32_PHASE_BLOCK_N,
                            probability_top, warp_id, shared.row_max,
                            shared.row_sum, warp_id, panel, NEG_INF);
                        bottom_exp_diff =
                            d256_bm32_phase_make_probability_row(
                                shared.score
                                    + (D256_BM32_PHASE_PANEL_M + warp_id)
                                          * D256_BM32_PHASE_BLOCK_N,
                                probability_bottom, warp_id, shared.row_max,
                                shared.row_sum,
                                D256_BM32_PHASE_PANEL_M + warp_id, panel,
                                NEG_INF);
                    }
                    if (lane_id == 0) {
                        row_exp_diff[warp_id] = top_exp_diff;
                        row_exp_diff[D256_BM32_PHASE_PANEL_M + warp_id] =
                            bottom_exp_diff;
                    }
                }
                // Publish lane 0's online-softmax state to this warp before
                // it advances to the next panel.
                __syncwarp();
            }
            __syncthreads();

            #pragma unroll
            for (int panel = 0; panel < D256_BM32_PHASE_PANELS;
                 ++panel) {
                const int panel_start =
                    panel * D256_BM32_PHASE_SOFTMAX_N;
                const int valid_panel_columns =
                    min(D256_BM32_PHASE_SOFTMAX_N,
                        valid_k_rows - panel_start);
                if (valid_panel_columns > 0) {
                    const __half* probability_top =
                        shared.probability_top
                        + panel * D256_BM32_PHASE_PROBABILITY_ELEMENTS;
                    const __half* probability_bottom =
                        shared.probability_bottom
                        + panel * D256_BM32_PHASE_PROBABILITY_ELEMENTS;
                    const float* row_exp_diff =
                        shared.row_exp_diff
                        + panel * D256_BM32_PHASE_BLOCK_M;
                    if constexpr (SPLIT_KV) {
                        d256_bm32_phase_scale_accumulators(
                            accumulator_top, accumulator_bottom,
                            row_exp_diff);
                    } else if (shared.block_index != 0 || panel != 0) {
                        d256_bm32_phase_scale_accumulators(
                            accumulator_top, accumulator_bottom,
                            row_exp_diff);
                    }
                    const int page_slot_k0 = panel_start >> 4;
                    const __half* value_k0 =
                        reinterpret_cast<const __half*>(
                            shared.v_tile_ptr[page_slot_k0]);
                    const __half* value_k16 = nullptr;
                    if (valid_panel_columns > WMMA_K) {
                        const int page_slot_k16 =
                            (panel_start + WMMA_K) >> 4;
                        value_k16 =
                            reinterpret_cast<const __half*>(
                                shared.v_tile_ptr[page_slot_k16]);
                    }
                    d256_bm32_phase_update_pv_panel(
                        probability_top, probability_bottom,
                        value_k0, value_k16, v_token_stride,
                        warp_id * WMMA_N,
                        valid_panel_columns,
                        accumulator_top, accumulator_bottom);
                }
            }
            // Protect the current V pointers and block index from the next
            // iteration while slower warps finish the last PV panel.
            __syncthreads();
        } else {
            #pragma unroll
            for (int panel = 0; panel < D256_BM32_PHASE_PANELS;
                 ++panel) {
                const int panel_start =
                    panel * D256_BM32_PHASE_SOFTMAX_N;
                const int valid_panel_columns =
                    min(D256_BM32_PHASE_SOFTMAX_N,
                        valid_k_rows - panel_start);
                if (valid_panel_columns > 0) {
                    const float top_exp_diff =
                        d256_bm32_phase_make_probability_row(
                            shared.score
                                + warp_id * D256_BM32_PHASE_BLOCK_N,
                            shared.probability_top, warp_id, shared.row_max,
                            shared.row_sum, warp_id, panel, NEG_INF);
                    const float bottom_exp_diff =
                        d256_bm32_phase_make_probability_row(
                            shared.score
                                + (D256_BM32_PHASE_PANEL_M + warp_id)
                                      * D256_BM32_PHASE_BLOCK_N,
                            shared.probability_bottom, warp_id,
                            shared.row_max, shared.row_sum,
                            D256_BM32_PHASE_PANEL_M + warp_id, panel,
                            NEG_INF);
                    if (lane_id == 0) {
                        shared.row_exp_diff[warp_id] = top_exp_diff;
                        shared.row_exp_diff[
                            D256_BM32_PHASE_PANEL_M + warp_id] =
                            bottom_exp_diff;
                    }
                }
                __syncthreads();

                if (valid_panel_columns > 0) {
                    if constexpr (SPLIT_KV) {
                        d256_bm32_phase_scale_accumulators(
                            accumulator_top, accumulator_bottom,
                            shared.row_exp_diff);
                    } else if (shared.block_index != 0 || panel != 0) {
                        d256_bm32_phase_scale_accumulators(
                            accumulator_top, accumulator_bottom,
                            shared.row_exp_diff);
                    }
                    const int page_slot_k0 = panel_start >> 4;
                    const __half* value_k0 =
                        reinterpret_cast<const __half*>(
                            shared.v_tile_ptr[page_slot_k0]);
                    const __half* value_k16 = nullptr;
                    if (valid_panel_columns > WMMA_K) {
                        const int page_slot_k16 =
                            (panel_start + WMMA_K) >> 4;
                        value_k16 =
                            reinterpret_cast<const __half*>(
                                shared.v_tile_ptr[page_slot_k16]);
                    }
                    d256_bm32_phase_update_pv_panel(
                        shared.probability_top,
                        shared.probability_bottom,
                        value_k0, value_k16, v_token_stride,
                        warp_id * WMMA_N,
                        valid_panel_columns,
                        accumulator_top, accumulator_bottom);
                }
                __syncthreads();
            }
        }

        if (tid == 0) {
            ++shared.block_index;
        }
        __syncthreads();
    }

    {
        const size_t q_head_linear = static_cast<size_t>(blockIdx.z);
        if constexpr (SPLIT_KV) {
            const size_t partial_row_base =
                (q_head_linear * D256_BM32_PHASE_SPLIT_PARTS
                 + static_cast<int>(blockIdx.y))
                    * M
                + start_row;
            float* partial_out =
                split_tmp_out
                + partial_row_base * D256_BM32_PHASE_D;
            d256_bm32_phase_store_unnormalized_partial(
                accumulator_top, partial_out, 0,
                warp_id * WMMA_N, valid_q_rows);
            d256_bm32_phase_store_unnormalized_partial(
                accumulator_bottom, partial_out,
                D256_BM32_PHASE_PANEL_M, warp_id * WMMA_N, valid_q_rows);
            if (tid < valid_q_rows) {
                split_tmp_row_max[partial_row_base + tid] =
                    shared.row_max[tid];
                split_tmp_row_sum[partial_row_base + tid] =
                    shared.row_sum[tid];
            }
        } else {
            __half* out_ptr =
                Out + q_head_linear * M * D256_BM32_PHASE_D
                + start_row * D256_BM32_PHASE_D;
            float* softmax_lse_ptr =
                softmax_lse + q_head_linear * M + start_row;
            if (valid_q_rows == D256_BM32_PHASE_BLOCK_M) {
                d256_bm32_phase_store_output(
                    accumulator_top, out_ptr, shared.row_sum, 0,
                    warp_id * WMMA_N);
                d256_bm32_phase_store_output(
                    accumulator_bottom, out_ptr, shared.row_sum,
                    D256_BM32_PHASE_PANEL_M, warp_id * WMMA_N);
            } else {
                d256_bm32_phase_store_output_partial(
                    accumulator_top, out_ptr, shared.row_sum, 0,
                    warp_id * WMMA_N, valid_q_rows);
                d256_bm32_phase_store_output_partial(
                    accumulator_bottom, out_ptr, shared.row_sum,
                    D256_BM32_PHASE_PANEL_M, warp_id * WMMA_N, valid_q_rows);
            }

            if (tid < valid_q_rows) {
                const float sum = fmaxf(shared.row_sum[tid], 1e-24f);
                softmax_lse_ptr[tid] = shared.row_max[tid] + logf(sum);
            }
        }
    }
}

template<bool IS_CAUSAL, bool ALL_P = false, bool PAIR_SCRATCH = false>
__global__ __launch_bounds__(D256_BM32_PHASE_THREADS, 2)
void flash_attention_forward_paged_d256_bm32_phase_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
    const int* __restrict__ block_table,
    const int* __restrict__ seqused_k,
    int H,
    int M,
    int max_num_blocks_per_seq,
    int num_kv_heads,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    float softmax_scale) {
    flash_attention_forward_paged_d256_bm32_phase_body<
        IS_CAUSAL, ALL_P, PAIR_SCRATCH, false>(
            Q, K_cache, V_cache, Out, softmax_lse, block_table, seqused_k,
            H, M, max_num_blocks_per_seq, num_kv_heads, k_block_stride,
            k_token_stride, k_head_stride, v_block_stride, v_token_stride,
            v_head_stride, softmax_scale, nullptr, nullptr, nullptr, 0);
}

template<bool CHECK_SPLIT_EMPTY>
__global__ __launch_bounds__(D256_BM32_PHASE_THREADS, 2)
void flash_attention_forward_paged_d256_bm32_splitkv3_partial_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache,
    const int* __restrict__ block_table,
    int H,
    int M,
    int actual_n,
    int max_num_blocks_per_seq,
    int num_kv_heads,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    float softmax_scale,
    float* __restrict__ split_tmp_out,
    float* __restrict__ split_tmp_row_max,
    float* __restrict__ split_tmp_row_sum) {
    flash_attention_forward_paged_d256_bm32_phase_body<
        true, true, true, true, CHECK_SPLIT_EMPTY>(
            Q, K_cache, V_cache, nullptr, nullptr, block_table, nullptr,
            H, M, max_num_blocks_per_seq, num_kv_heads, k_block_stride,
            k_token_stride, k_head_stride, v_block_stride, v_token_stride,
            v_head_stride, softmax_scale, split_tmp_out,
            split_tmp_row_max, split_tmp_row_sum, actual_n);
}

template<bool IS_CAUSAL, bool ALL_P, bool PAIR_SCRATCH>
void launch_flash_attention_forward_paged_d256_bm32_phase_kernel(
    const torch::Tensor& Q,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    float softmax_scale,
    cudaStream_t stream) {
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int num_kv_heads = K_cache.size(2);
    const int max_num_blocks_per_seq = block_table.size(1);
    const int64_t k_block_stride = K_cache.stride(0);
    const int64_t k_token_stride = K_cache.stride(1);
    const int64_t k_head_stride = K_cache.stride(2);
    const int64_t v_block_stride = V_cache.stride(0);
    const int64_t v_token_stride = V_cache.stride(1);
    const int64_t v_head_stride = V_cache.stride(2);
    const dim3 grid((M + D256_BM32_PHASE_BLOCK_M - 1)
                        / D256_BM32_PHASE_BLOCK_M,
                    1, Q.size(0) * H);
    const dim3 block(D256_BM32_PHASE_THREADS);
    flash_attention_forward_paged_d256_bm32_phase_kernel<
        IS_CAUSAL, ALL_P, PAIR_SCRATCH><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K_cache.data_ptr()),
            reinterpret_cast<const __half*>(V_cache.data_ptr()),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(), block_table.data_ptr<int>(),
            seq_lens.data_ptr<int>(), H, M, max_num_blocks_per_seq,
            num_kv_heads, k_block_stride, k_token_stride, k_head_stride,
            v_block_stride, v_token_stride, v_head_stride, softmax_scale);
}

__global__ __launch_bounds__(256)
void flash_attention_forward_paged_d256_bm32_splitkv3_merge_kernel(
    const float* __restrict__ split_tmp_out,
    const float* __restrict__ split_tmp_row_max,
    const float* __restrict__ split_tmp_row_sum,
    __half* __restrict__ Out,
    float* __restrict__ softmax_lse,
    int M) {
    const int start_row = blockIdx.x * D256_BM32_PHASE_BLOCK_M;
    const int row_local = threadIdx.x >> 3;
    const int lane_in_row = threadIdx.x & 7;
    const int row = start_row + row_local;
    if (row >= M) {
        return;
    }

    const size_t q_head_linear = static_cast<size_t>(blockIdx.z);
    const size_t first_state =
        (q_head_linear * D256_BM32_PHASE_SPLIT_PARTS) * M + row;
    const float max_0 = split_tmp_row_max[first_state];
    const float max_1 = split_tmp_row_max[first_state + M];
    const float max_2 = split_tmp_row_max[first_state + 2 * M];
    const float sum_0 = split_tmp_row_sum[first_state];
    const float sum_1 = split_tmp_row_sum[first_state + M];
    const float sum_2 = split_tmp_row_sum[first_state + 2 * M];
    const float merged_max = fmaxf(max_0, fmaxf(max_1, max_2));
    const float weight_0 =
        sum_0 > 0.0f ? __expf(fmaxf(max_0 - merged_max, -80.0f)) : 0.0f;
    const float weight_1 =
        sum_1 > 0.0f ? __expf(fmaxf(max_1 - merged_max, -80.0f)) : 0.0f;
    const float weight_2 =
        sum_2 > 0.0f ? __expf(fmaxf(max_2 - merged_max, -80.0f)) : 0.0f;
    const float denominator =
        weight_0 * sum_0 + weight_1 * sum_1 + weight_2 * sum_2;
    const float inverse_denominator =
        1.0f / fmaxf(denominator, 1e-24f);

    if (lane_in_row == 0) {
        softmax_lse[q_head_linear * M + row] =
            merged_max + logf(fmaxf(denominator, 1e-24f));
    }

    const size_t output_base =
        (q_head_linear * M + row) * D256_BM32_PHASE_D;
    const size_t partial_base_0 = first_state * D256_BM32_PHASE_D;
    const size_t partial_base_1 =
        (first_state + M) * D256_BM32_PHASE_D;
    const size_t partial_base_2 =
        (first_state + 2 * M) * D256_BM32_PHASE_D;
    #pragma unroll
    for (int column = lane_in_row; column < D256_BM32_PHASE_D; column += 8) {
        const float numerator =
            weight_0 * split_tmp_out[partial_base_0 + column]
            + weight_1 * split_tmp_out[partial_base_1 + column]
            + weight_2 * split_tmp_out[partial_base_2 + column];
        Out[output_base + column] =
            __float2half_rn(numerator * inverse_denominator);
    }
}

void launch_flash_attention_forward_paged_d256_bm32_splitkv3_kernel(
    const torch::Tensor& Q,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    torch::Tensor& split_tmp_out,
    torch::Tensor& split_tmp_row_max,
    torch::Tensor& split_tmp_row_sum,
    const torch::Tensor& block_table,
    int actual_n,
    float softmax_scale,
    cudaStream_t stream) {
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int num_kv_heads = K_cache.size(2);
    const int max_num_blocks_per_seq = block_table.size(1);
    const int64_t k_block_stride = K_cache.stride(0);
    const int64_t k_token_stride = K_cache.stride(1);
    const int64_t k_head_stride = K_cache.stride(2);
    const int64_t v_block_stride = V_cache.stride(0);
    const int64_t v_token_stride = V_cache.stride(1);
    const int64_t v_head_stride = V_cache.stride(2);
    const dim3 partial_grid(
        (M + D256_BM32_PHASE_BLOCK_M - 1) / D256_BM32_PHASE_BLOCK_M,
        D256_BM32_PHASE_SPLIT_PARTS,
        Q.size(0) * H);
    const dim3 partial_block(D256_BM32_PHASE_THREADS);
    const int64_t total_n_blocks =
        (static_cast<int64_t>(actual_n) + D256_BM32_PHASE_BLOCK_N - 1)
        / D256_BM32_PHASE_BLOCK_N;
    const int64_t last_split_begin =
        (D256_BM32_PHASE_SPLIT_PARTS - 1) * total_n_blocks
        / D256_BM32_PHASE_SPLIT_PARTS * D256_BM32_PHASE_BLOCK_N;
    const bool check_split_empty =
        total_n_blocks < D256_BM32_PHASE_SPLIT_PARTS
        || last_split_begin > static_cast<int64_t>(actual_n) - M;
    if (check_split_empty) {
        flash_attention_forward_paged_d256_bm32_splitkv3_partial_kernel<true>
            <<<partial_grid, partial_block, 0, stream>>>(
                reinterpret_cast<const __half*>(Q.data_ptr()),
                reinterpret_cast<const __half*>(K_cache.data_ptr()),
                reinterpret_cast<const __half*>(V_cache.data_ptr()),
                block_table.data_ptr<int>(), H, M, actual_n,
                max_num_blocks_per_seq, num_kv_heads, k_block_stride,
                k_token_stride, k_head_stride, v_block_stride,
                v_token_stride, v_head_stride, softmax_scale,
                split_tmp_out.data_ptr<float>(),
                split_tmp_row_max.data_ptr<float>(),
                split_tmp_row_sum.data_ptr<float>());
    } else {
        flash_attention_forward_paged_d256_bm32_splitkv3_partial_kernel<false>
            <<<partial_grid, partial_block, 0, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K_cache.data_ptr()),
            reinterpret_cast<const __half*>(V_cache.data_ptr()),
                block_table.data_ptr<int>(), H, M, actual_n,
                max_num_blocks_per_seq, num_kv_heads, k_block_stride,
                k_token_stride, k_head_stride, v_block_stride,
                v_token_stride, v_head_stride, softmax_scale,
                split_tmp_out.data_ptr<float>(),
                split_tmp_row_max.data_ptr<float>(),
                split_tmp_row_sum.data_ptr<float>());
    }

    const dim3 merge_grid(
        (M + D256_BM32_PHASE_BLOCK_M - 1) / D256_BM32_PHASE_BLOCK_M,
        1,
        Q.size(0) * H);
    flash_attention_forward_paged_d256_bm32_splitkv3_merge_kernel
        <<<merge_grid, 256, 0, stream>>>(
            split_tmp_out.data_ptr<float>(),
            split_tmp_row_max.data_ptr<float>(),
            split_tmp_row_sum.data_ptr<float>(),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(), M);
}

void launcher_flash_attention_forward_paged_d256_bm32_phase(
    const torch::Tensor& Q,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream) {
    const bool use_all_p =
        env_flag_default_enabled("VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P");
    const bool use_pair_scratch =
        use_all_p
        && env_flag_default_enabled(
            "VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH");

    if (is_causal) {
        if (use_all_p) {
            if (use_pair_scratch) {
                launch_flash_attention_forward_paged_d256_bm32_phase_kernel<
                    true, true, true>(
                    Q, K_cache, V_cache, Out, softmax_lse, block_table,
                    seq_lens, softmax_scale, stream);
            } else {
                launch_flash_attention_forward_paged_d256_bm32_phase_kernel<
                    true, true, false>(
                    Q, K_cache, V_cache, Out, softmax_lse, block_table,
                    seq_lens, softmax_scale, stream);
            }
        } else {
            launch_flash_attention_forward_paged_d256_bm32_phase_kernel<
                true, false, false>(
                Q, K_cache, V_cache, Out, softmax_lse, block_table, seq_lens,
                softmax_scale, stream);
        }
        return;
    }

    if (use_all_p) {
        if (use_pair_scratch) {
            launch_flash_attention_forward_paged_d256_bm32_phase_kernel<
                false, true, true>(
                Q, K_cache, V_cache, Out, softmax_lse, block_table, seq_lens,
                softmax_scale, stream);
        } else {
            launch_flash_attention_forward_paged_d256_bm32_phase_kernel<
                false, true, false>(
                Q, K_cache, V_cache, Out, softmax_lse, block_table, seq_lens,
                softmax_scale, stream);
        }
    } else {
        launch_flash_attention_forward_paged_d256_bm32_phase_kernel<
            false, false, false>(
            Q, K_cache, V_cache, Out, softmax_lse, block_table, seq_lens,
            softmax_scale, stream);
    }
}

template<int D, int KV_DTYPE>
void launcher_flash_attention_forward_paged(
    const torch::Tensor& Q,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    float softmax_scale,
    bool is_causal,
    float k_scale,
    float v_scale,
    int window_size_left,
    int window_size_right,
    cudaStream_t stream,
    const int* bfla_mask_ptr = nullptr,
    int bfla_mask_block_n = 0,
    int64_t bfla_mask_stride_b = 0,
    int64_t bfla_mask_stride_h = 0,
    int64_t bfla_mask_stride_q = 0,
    int64_t bfla_mask_stride_k = 0
) {
    if constexpr (D == 256 && KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
        const int M = Q.size(2);
        const int page_block_size = K_cache.size(1);
        const bool use_low_smem =
            M > 1 && page_block_size >= 16 && (page_block_size % 16) == 0
            && env_flag_default_enabled(
                "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM");
        if (use_low_smem) {
            const bool use_d256_bm32_phase =
                env_flag_default_enabled(
                    "VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE")
                && page_block_size == D256_BM32_PHASE_PAGE_BLOCK_SIZE
                && M >= D256_BM32_PHASE_BLOCK_M
                && bfla_mask_ptr == nullptr && window_size_left < 0
                && window_size_right < 0;
            if (use_d256_bm32_phase) {
                launcher_flash_attention_forward_paged_d256_bm32_phase(
                    Q, K_cache, V_cache, Out, softmax_lse, block_table,
                    seq_lens, softmax_scale, is_causal, stream);
                return;
            }
            const bool use_low_smem_contig_fast =
                page_block_size == 16 ||
                env_flag_enabled("VLLM_FLASH_V100_PREFILL_CONTIG_FAST");
            const bool use_low_smem_scalar_qk =
                env_flag_enabled("VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK");
            const bool use_low_smem_bm32 =
                env_flag_enabled("VLLM_FLASH_V100_PREFILL_D256_BM32");
            // Default only for the measured hybrid-cache page size. Other
            // page sizes retain the prior layout unless explicitly enabled.
            const bool use_low_smem_output_stride_268 =
                page_block_size == 784
                    ? env_flag_default_enabled(
                          "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268")
                    : env_flag_enabled(
                          "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268");
            const bool use_sw_pipeline =
                page_block_size == 784
                && env_flag_enabled(
                    "VLLM_FLASH_V100_PREFILL_D256_SOFTWARE_PIPELINE");
            const bool use_sw_pipeline_qk =
                page_block_size == 784
                && (use_sw_pipeline
                    || env_flag_default_enabled(
                        "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK"));
            const bool use_sw_pipeline_pv =
                page_block_size == 784
                && (use_sw_pipeline
                    || env_flag_default_enabled(
                        "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV"));
            if (use_low_smem_contig_fast) {
                if (use_low_smem_scalar_qk) {
                    if (use_low_smem_bm32) {
                        launcher_flash_attention_forward_paged_impl<
                            D, KV_DTYPE, true, true, true, true>(
                            Q, K_cache, V_cache, Out, softmax_lse, block_table,
                            seq_lens, bfla_mask_ptr, bfla_mask_block_n,
                            bfla_mask_stride_b, bfla_mask_stride_h,
                            bfla_mask_stride_q, bfla_mask_stride_k,
                            softmax_scale, is_causal, k_scale, v_scale,
                            window_size_left, window_size_right, stream);
                    } else {
                        launcher_flash_attention_forward_paged_impl<
                            D, KV_DTYPE, true, true, true, false>(
                            Q, K_cache, V_cache, Out, softmax_lse, block_table,
                            seq_lens, bfla_mask_ptr, bfla_mask_block_n,
                            bfla_mask_stride_b, bfla_mask_stride_h,
                            bfla_mask_stride_q, bfla_mask_stride_k,
                            softmax_scale, is_causal, k_scale, v_scale,
                            window_size_left, window_size_right, stream);
                    }
                } else {
                    if (use_low_smem_bm32) {
                        launcher_flash_attention_forward_paged_impl<
                            D, KV_DTYPE, true, true, false, true>(
                            Q, K_cache, V_cache, Out, softmax_lse, block_table,
                            seq_lens, bfla_mask_ptr, bfla_mask_block_n,
                            bfla_mask_stride_b, bfla_mask_stride_h,
                            bfla_mask_stride_q, bfla_mask_stride_k,
                            softmax_scale, is_causal, k_scale, v_scale,
                            window_size_left, window_size_right, stream);
                    } else {
                        launcher_flash_attention_forward_paged_impl<
                            D, KV_DTYPE, true, true, false, false>(
                            Q, K_cache, V_cache, Out, softmax_lse, block_table,
                            seq_lens, bfla_mask_ptr, bfla_mask_block_n,
                            bfla_mask_stride_b, bfla_mask_stride_h,
                            bfla_mask_stride_q, bfla_mask_stride_k,
                            softmax_scale, is_causal, k_scale, v_scale,
                            window_size_left, window_size_right, stream);
                    }
                }
            } else {
                if (use_low_smem_scalar_qk) {
                    if (use_low_smem_bm32) {
                        launcher_flash_attention_forward_paged_impl<
                            D, KV_DTYPE, true, false, true, true>(
                            Q, K_cache, V_cache, Out, softmax_lse, block_table,
                            seq_lens, bfla_mask_ptr, bfla_mask_block_n,
                            bfla_mask_stride_b, bfla_mask_stride_h,
                            bfla_mask_stride_q, bfla_mask_stride_k,
                            softmax_scale, is_causal, k_scale, v_scale,
                            window_size_left, window_size_right, stream);
                    } else {
                        launcher_flash_attention_forward_paged_impl<
                            D, KV_DTYPE, true, false, true, false>(
                            Q, K_cache, V_cache, Out, softmax_lse, block_table,
                            seq_lens, bfla_mask_ptr, bfla_mask_block_n,
                            bfla_mask_stride_b, bfla_mask_stride_h,
                            bfla_mask_stride_q, bfla_mask_stride_k,
                            softmax_scale, is_causal, k_scale, v_scale,
                            window_size_left, window_size_right, stream);
                    }
                } else {
                    if (use_low_smem_bm32) {
                        launcher_flash_attention_forward_paged_impl<
                            D, KV_DTYPE, true, false, false, true>(
                            Q, K_cache, V_cache, Out, softmax_lse, block_table,
                            seq_lens, bfla_mask_ptr, bfla_mask_block_n,
                            bfla_mask_stride_b, bfla_mask_stride_h,
                            bfla_mask_stride_q, bfla_mask_stride_k,
                            softmax_scale, is_causal, k_scale, v_scale,
                            window_size_left, window_size_right, stream);
                    } else {
                        if (use_low_smem_output_stride_268) {
                            if (use_sw_pipeline_qk) {
                                if (use_sw_pipeline_pv) {
                                    launcher_flash_attention_forward_paged_impl<
                                        D, KV_DTYPE, true, false, false,
                                        false, true, true, true>(
                                        Q, K_cache, V_cache, Out,
                                        softmax_lse, block_table, seq_lens,
                                        bfla_mask_ptr, bfla_mask_block_n,
                                        bfla_mask_stride_b,
                                        bfla_mask_stride_h,
                                        bfla_mask_stride_q,
                                        bfla_mask_stride_k, softmax_scale,
                                        is_causal, k_scale, v_scale,
                                        window_size_left,
                                        window_size_right, stream);
                                } else {
                                    launcher_flash_attention_forward_paged_impl<
                                        D, KV_DTYPE, true, false, false, false,
                                        true, true, false>(
                                        Q, K_cache, V_cache, Out, softmax_lse,
                                        block_table, seq_lens, bfla_mask_ptr,
                                        bfla_mask_block_n, bfla_mask_stride_b,
                                        bfla_mask_stride_h,
                                        bfla_mask_stride_q,
                                        bfla_mask_stride_k, softmax_scale,
                                        is_causal, k_scale, v_scale,
                                        window_size_left, window_size_right,
                                        stream);
                                }
                            } else if (use_sw_pipeline_pv) {
                                launcher_flash_attention_forward_paged_impl<
                                    D, KV_DTYPE, true, false, false, false,
                                    true, false, true>(
                                    Q, K_cache, V_cache, Out, softmax_lse,
                                    block_table, seq_lens, bfla_mask_ptr,
                                    bfla_mask_block_n, bfla_mask_stride_b,
                                    bfla_mask_stride_h, bfla_mask_stride_q,
                                    bfla_mask_stride_k, softmax_scale,
                                    is_causal, k_scale, v_scale,
                                    window_size_left, window_size_right,
                                    stream);
                            } else {
                                launcher_flash_attention_forward_paged_impl<
                                    D, KV_DTYPE, true, false, false, false,
                                    true>(
                                    Q, K_cache, V_cache, Out, softmax_lse,
                                    block_table, seq_lens, bfla_mask_ptr,
                                    bfla_mask_block_n, bfla_mask_stride_b,
                                    bfla_mask_stride_h, bfla_mask_stride_q,
                                    bfla_mask_stride_k, softmax_scale,
                                    is_causal, k_scale, v_scale,
                                    window_size_left, window_size_right,
                                    stream);
                            }
                        } else {
                            launcher_flash_attention_forward_paged_impl<
                                D, KV_DTYPE, true, false, false, false>(
                                Q, K_cache, V_cache, Out, softmax_lse,
                                block_table, seq_lens, bfla_mask_ptr,
                                bfla_mask_block_n, bfla_mask_stride_b,
                                bfla_mask_stride_h, bfla_mask_stride_q,
                                bfla_mask_stride_k, softmax_scale, is_causal,
                                k_scale, v_scale, window_size_left,
                                window_size_right, stream);
                        }
                    }
                }
            }
        } else {
            launcher_flash_attention_forward_paged_impl<
                D, KV_DTYPE, false, false, false, false>(
                Q, K_cache, V_cache, Out, softmax_lse, block_table, seq_lens,
                bfla_mask_ptr, bfla_mask_block_n, bfla_mask_stride_b,
                bfla_mask_stride_h, bfla_mask_stride_q, bfla_mask_stride_k,
                softmax_scale, is_causal, k_scale, v_scale, window_size_left,
                window_size_right, stream);
        }
    } else {
        launcher_flash_attention_forward_paged_impl<
            D, KV_DTYPE, false, false, false, false>(
            Q, K_cache, V_cache, Out, softmax_lse, block_table, seq_lens,
            bfla_mask_ptr, bfla_mask_block_n, bfla_mask_stride_b,
            bfla_mask_stride_h, bfla_mask_stride_q, bfla_mask_stride_k,
            softmax_scale, is_causal, k_scale, v_scale, window_size_left,
            window_size_right, stream);
    }
}

at::Tensor flash_attention_prefill_paged_splitkv(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const float softmax_scale,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale,
    const bool is_causal,
    const int window_size_left,
    const int window_size_right,
    const int split_kv_tokens,
    const int max_seq_len_hint
) {
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
    TORCH_CHECK(kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16,
                "split-KV paged prefill supports fp16 KV cache only");
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
    TORCH_CHECK(v_cache.dtype() == torch::kFloat16, "v_cache must be fp16");
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "Tensors must be on CUDA");
    TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
                "block_table and seq_lens must be CUDA tensors");
    TORCH_CHECK(q.stride(-1) == 1 && k_cache.stride(-1) == 1 &&
                    v_cache.stride(-1) == 1,
                "Last dim must be contiguous");
    TORCH_CHECK(k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0 &&
                    v_cache.stride(1) % 8 == 0 && v_cache.stride(2) % 8 == 0,
                "Paged KV strides must be divisible by 8 half elements");

    const int B = q.size(0);
    const int H = q.size(1);
    const int M = q.size(2);
    const int D = q.size(3);
    const int num_kv_heads = k_cache.size(2);
    const int page_block_size = k_cache.size(1);

    TORCH_CHECK(D == 256, "split-KV paged prefill supports D=256 only");
    TORCH_CHECK(H % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(M > 1, "split-KV paged prefill requires prefill M > 1");
    TORCH_CHECK(page_block_size >= 16 && (page_block_size % 16) == 0,
                "split-KV paged prefill requires page block size multiple of 16");
    TORCH_CHECK(window_size_left >= -1 && window_size_right >= -1,
                "window sizes must be >= -1");

    at::Tensor out_fp16 = out_.has_value() ? out_.value() : torch::zeros_like(q);
    auto softmax_lse = torch::zeros({B, H, M},
                                    torch::dtype(torch::kFloat32).device(q.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props = at::cuda::getCurrentDeviceProperties();
    bool sm70 = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    const bool use_low_smem_contig_fast =
        page_block_size == 16 ||
        env_flag_enabled("VLLM_FLASH_V100_PREFILL_CONTIG_FAST");
    const bool use_low_smem_scalar_qk =
        env_flag_enabled("VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK");
    const int block_n =
        use_low_smem_scalar_qk
            ? BLOCK_N_256_LOW_SMEM_SCALAR_QK
            : BLOCK_N_256_LOW_SMEM;
    const int split_tokens_rounded = std::max(split_kv_tokens, block_n);
    const int split_kv_tiles =
        std::max(1, (split_tokens_rounded + block_n - 1) / block_n);
    const int max_hint = std::max(max_seq_len_hint, 1);
    const int max_kv_tiles = std::max(1, (max_hint + block_n - 1) / block_n);
    const int num_partitions =
        std::max(1, (max_kv_tiles + split_kv_tiles - 1) / split_kv_tiles);

    if (num_partitions <= 1) {
        launcher_flash_attention_forward_paged<256, flash_v100::KV_CACHE_DTYPE_FP16>(
            q, k_cache, v_cache, out_fp16, softmax_lse, block_table, seq_lens,
            softmax_scale, is_causal, k_scale, v_scale, window_size_left,
            window_size_right, stream);
        return out_fp16;
    }

    auto tmp_options = torch::dtype(torch::kFloat32).device(q.device());
    auto split_tmp_out =
        torch::empty({B, H, num_partitions, M, D}, tmp_options);
    auto split_tmp_row_max =
        torch::empty({B, H, num_partitions, M}, tmp_options);
    auto split_tmp_row_sum =
        torch::empty({B, H, num_partitions, M}, tmp_options);

    if (use_low_smem_contig_fast) {
        if (use_low_smem_scalar_qk) {
            launcher_flash_attention_forward_paged_splitkv_impl<
                256, true, true>(
                q, k_cache, v_cache, out_fp16, softmax_lse, split_tmp_out,
                split_tmp_row_max, split_tmp_row_sum, block_table, seq_lens,
                softmax_scale, is_causal, k_scale, v_scale, window_size_left,
                window_size_right, split_kv_tokens, max_seq_len_hint, stream);
        } else {
            launcher_flash_attention_forward_paged_splitkv_impl<
                256, true, false>(
                q, k_cache, v_cache, out_fp16, softmax_lse, split_tmp_out,
                split_tmp_row_max, split_tmp_row_sum, block_table, seq_lens,
                softmax_scale, is_causal, k_scale, v_scale, window_size_left,
                window_size_right, split_kv_tokens, max_seq_len_hint, stream);
        }
    } else {
        if (use_low_smem_scalar_qk) {
            launcher_flash_attention_forward_paged_splitkv_impl<
                256, false, true>(
                q, k_cache, v_cache, out_fp16, softmax_lse, split_tmp_out,
                split_tmp_row_max, split_tmp_row_sum, block_table, seq_lens,
                softmax_scale, is_causal, k_scale, v_scale, window_size_left,
                window_size_right, split_kv_tokens, max_seq_len_hint, stream);
        } else {
            launcher_flash_attention_forward_paged_splitkv_impl<
                256, false, false>(
                q, k_cache, v_cache, out_fp16, softmax_lse, split_tmp_out,
                split_tmp_row_max, split_tmp_row_sum, block_table, seq_lens,
                softmax_scale, is_causal, k_scale, v_scale, window_size_left,
                window_size_right, split_kv_tokens, max_seq_len_hint, stream);
        }
    }

    return out_fp16;
}

at::Tensor flash_attention_prefill_paged(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const float softmax_scale,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale,
    const bool is_causal,
    const int window_size_left,
    const int window_size_right
) {
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
    TORCH_CHECK(kv_dtype_code >= 0, "Unsupported kv_cache_dtype: ", kv_cache_dtype);
    if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
        TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
        TORCH_CHECK(v_cache.dtype() == torch::kFloat16, "v_cache must be fp16");
    } else {
        TORCH_CHECK(k_cache.dtype() == torch::kUInt8,
                    "fp8 k_cache must be stored as uint8");
        TORCH_CHECK(v_cache.dtype() == torch::kUInt8,
                    "fp8 v_cache must be stored as uint8");
        TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
                    "fp8 k/v scales must be positive");
    }
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "Tensors must be on CUDA");
    TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
                "block_table and seq_lens must be CUDA tensors");
    TORCH_CHECK(q.stride(-1) == 1 && k_cache.stride(-1) == 1 &&
                    v_cache.stride(-1) == 1,
                "Last dim must be contiguous");
    TORCH_CHECK(k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0 &&
                    v_cache.stride(1) % 8 == 0 && v_cache.stride(2) % 8 == 0,
                "Paged KV strides must be divisible by 8 half elements");

    const int B = q.size(0);
    const int H = q.size(1);
    const int M = q.size(2);
    const int D = q.size(3);
    const int num_kv_heads = k_cache.size(2);

    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0,
                "D must be even, <=256, multiple of 8");
    TORCH_CHECK(H % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(window_size_left >= -1 && window_size_right >= -1,
                "window sizes must be >= -1");

    at::Tensor out_fp16 = out_.has_value() ? out_.value() : torch::zeros_like(q);
    auto softmax_lse = torch::zeros({B, H, M},
                                    torch::dtype(torch::kFloat32).device(q.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props = at::cuda::getCurrentDeviceProperties();
    bool sm70 = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    #define LAUNCH_PAGED_TYPED(HDIM, KV_DTYPE_CODE)                             \
        launcher_flash_attention_forward_paged<HDIM, KV_DTYPE_CODE>(            \
            q, k_cache, v_cache, out_fp16, softmax_lse, block_table, seq_lens,  \
            softmax_scale, is_causal, k_scale, v_scale, window_size_left,       \
            window_size_right, stream)

    #define LAUNCH_PAGED_BY_KV(HDIM)                                            \
        do {                                                                    \
            switch (kv_dtype_code) {                                            \
                case flash_v100::KV_CACHE_DTYPE_FP16:                           \
                    LAUNCH_PAGED_TYPED(HDIM, flash_v100::KV_CACHE_DTYPE_FP16);  \
                    break;                                                      \
                case flash_v100::KV_CACHE_DTYPE_FP8_E4M3:                       \
                    LAUNCH_PAGED_TYPED(HDIM, flash_v100::KV_CACHE_DTYPE_FP8_E4M3); \
                    break;                                                      \
                case flash_v100::KV_CACHE_DTYPE_FP8_E5M2:                       \
                    LAUNCH_PAGED_TYPED(HDIM, flash_v100::KV_CACHE_DTYPE_FP8_E5M2); \
                    break;                                                      \
                default:                                                        \
                    TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype); \
            }                                                                   \
        } while (0)

    switch (D) {
        case 16:
            LAUNCH_PAGED_BY_KV(16);
            break;
        case 32:
            LAUNCH_PAGED_BY_KV(32);
            break;
        case 64:
            LAUNCH_PAGED_BY_KV(64);
            break;
        case 128:
            LAUNCH_PAGED_BY_KV(128);
            break;
        case 256:
            LAUNCH_PAGED_BY_KV(256);
            break;
        default:
            TORCH_CHECK(false, "Unsupported D: ", D);
    }

    #undef LAUNCH_PAGED_BY_KV
    #undef LAUNCH_PAGED_TYPED

    return out_fp16;
}

std::vector<at::Tensor>
flash_attention_prefill_paged_d256_bm32_allp_pair_scratch(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    std::optional<at::Tensor>& softmax_lse_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const float softmax_scale
) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, H, M, D]");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
                "k_cache and v_cache must have shape [blocks, page, Hkv, D]");
    TORCH_CHECK(block_table.dim() == 2,
                "block_table must have shape [B, max_num_blocks]");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16,
                "k_cache must be fp16");
    TORCH_CHECK(v_cache.dtype() == torch::kFloat16,
                "v_cache must be fp16");
    TORCH_CHECK(block_table.dtype() == torch::kInt32,
                "block_table must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "q, k_cache, and v_cache must be CUDA tensors");
    TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
                "block_table and seq_lens must be CUDA tensors");
    TORCH_CHECK(
        q.device() == k_cache.device() && q.device() == v_cache.device()
            && q.device() == block_table.device() && q.device() == seq_lens.device(),
        "all input tensors must be on the q device");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous [B, H, M, D]");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
                "k_cache and v_cache must have the same shape");
    TORCH_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
                "K/V head dimensions must be contiguous");
    TORCH_CHECK(k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0
                    && v_cache.stride(1) % 8 == 0
                    && v_cache.stride(2) % 8 == 0,
                "K/V strides must be divisible by 8 half elements");

    const int B = q.size(0);
    const int H = q.size(1);
    const int M = q.size(2);
    const int D = q.size(3);
    const int num_kv_heads = k_cache.size(2);

    TORCH_CHECK(B > 0 && H > 0, "q batch and head dimensions must be positive");
    TORCH_CHECK(M >= D256_BM32_PHASE_BLOCK_M,
                "fixed BM32 prefill requires M >= ",
                D256_BM32_PHASE_BLOCK_M);
    TORCH_CHECK(D == D256_BM32_PHASE_D,
                "fixed BM32 prefill requires D=", D256_BM32_PHASE_D);
    TORCH_CHECK(k_cache.size(0) > 0 && num_kv_heads > 0,
                "K/V cache block and head dimensions must be positive");
    TORCH_CHECK(k_cache.size(1) == D256_BM32_PHASE_PAGE_BLOCK_SIZE,
                "fixed BM32 prefill requires page size ",
                D256_BM32_PHASE_PAGE_BLOCK_SIZE);
    TORCH_CHECK(k_cache.size(3) == D256_BM32_PHASE_D,
                "K/V cache head dimension must be D=",
                D256_BM32_PHASE_D);
    TORCH_CHECK(H % num_kv_heads == 0,
                "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(block_table.size(0) == B && seq_lens.size(0) == B,
                "block_table and seq_lens batch dimensions must equal q batch");
    TORCH_CHECK(block_table.size(1) > 0,
                "block_table must provide at least one page index");

    if (out_.has_value()) {
        const at::Tensor& out = out_.value();
        TORCH_CHECK(out.is_cuda() && out.device() == q.device(),
                    "out must be a CUDA tensor on the q device");
        TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
        TORCH_CHECK(out.sizes() == q.sizes(), "out must have shape [B, H, M, D]");
        TORCH_CHECK(out.is_contiguous(), "out must be contiguous [B, H, M, D]");
    }
    if (softmax_lse_.has_value()) {
        const at::Tensor& softmax_lse = softmax_lse_.value();
        TORCH_CHECK(softmax_lse.is_cuda() && softmax_lse.device() == q.device(),
                    "softmax_lse must be a CUDA tensor on the q device");
        TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32,
                    "softmax_lse must be fp32");
        TORCH_CHECK(softmax_lse.dim() == 3 && softmax_lse.size(0) == B
                        && softmax_lse.size(1) == H && softmax_lse.size(2) == M,
                    "softmax_lse must have shape [B, H, M]");
        TORCH_CHECK(softmax_lse.is_contiguous(),
                    "softmax_lse must be contiguous [B, H, M]");
    }

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t capture_error = cudaStreamIsCapturing(stream, &capture_status);
    TORCH_CHECK(capture_error == cudaSuccess, "cudaStreamIsCapturing failed: ",
                cudaGetErrorString(capture_error));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "fixed BM32 prefill CUDA graph capture is not validated");

    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0,
                "fixed BM32 prefill supports only SM70 GPUs");

    at::Tensor out_fp16 = out_.has_value() ? out_.value() : torch::empty_like(q);
    at::Tensor softmax_lse =
        softmax_lse_.has_value()
            ? softmax_lse_.value()
            : torch::empty({B, H, M},
                           torch::dtype(torch::kFloat32).device(q.device()));

    launch_flash_attention_forward_paged_d256_bm32_phase_kernel<true, true, true>(
        q, k_cache, v_cache, out_fp16, softmax_lse, block_table, seq_lens,
        softmax_scale, stream);

    return {out_fp16, softmax_lse};
}

std::vector<at::Tensor>
flash_attention_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    std::optional<at::Tensor>& softmax_lse_,
    at::Tensor& split_tmp_out,
    at::Tensor& split_tmp_row_max,
    at::Tensor& split_tmp_row_sum,
    const at::Tensor& block_table,
    const int64_t actual_n,
    const float softmax_scale
) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, H, M, D]");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
                "k_cache and v_cache must have shape [blocks, page, Hkv, D]");
    TORCH_CHECK(block_table.dim() == 2,
                "block_table must have shape [B, max_num_blocks]");
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16,
                "k_cache must be fp16");
    TORCH_CHECK(v_cache.dtype() == torch::kFloat16,
                "v_cache must be fp16");
    TORCH_CHECK(block_table.dtype() == torch::kInt32,
                "block_table must be int32");
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "q, k_cache, and v_cache must be CUDA tensors");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
    TORCH_CHECK(
        q.device() == k_cache.device() && q.device() == v_cache.device()
            && q.device() == block_table.device(),
        "all input tensors must be on the q device");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous [B, H, M, D]");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
                "k_cache and v_cache must have the same shape");
    TORCH_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
                "K/V head dimensions must be contiguous");
    TORCH_CHECK(k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0
                    && v_cache.stride(1) % 8 == 0
                    && v_cache.stride(2) % 8 == 0,
                "K/V strides must be divisible by 8 half elements");

    const int B = q.size(0);
    const int H = q.size(1);
    const int M = q.size(2);
    const int D = q.size(3);
    const int num_kv_heads = k_cache.size(2);

    TORCH_CHECK(B == 1,
                "fixed BM32 split-KV host-length path requires B=1");
    TORCH_CHECK(H > 0, "q head dimension must be positive");
    TORCH_CHECK(M >= D256_BM32_PHASE_BLOCK_M,
                "fixed BM32 split-KV prefill requires M >= ",
                D256_BM32_PHASE_BLOCK_M);
    TORCH_CHECK(D == D256_BM32_PHASE_D,
                "fixed BM32 split-KV prefill requires D=",
                D256_BM32_PHASE_D);
    TORCH_CHECK(k_cache.size(0) > 0 && num_kv_heads > 0,
                "K/V cache block and head dimensions must be positive");
    TORCH_CHECK(k_cache.size(1) == D256_BM32_PHASE_PAGE_BLOCK_SIZE,
                "fixed BM32 split-KV prefill requires page size ",
                D256_BM32_PHASE_PAGE_BLOCK_SIZE);
    TORCH_CHECK(k_cache.size(3) == D256_BM32_PHASE_D,
                "K/V cache head dimension must be D=",
                D256_BM32_PHASE_D);
    TORCH_CHECK(H % num_kv_heads == 0,
                "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(block_table.size(0) == B,
                "block_table batch dimension must equal q batch");
    TORCH_CHECK(actual_n >= M && actual_n <= 2147483647LL,
                "actual_n must satisfy M <= actual_n <= INT_MAX");
    const int64_t required_pages =
        (actual_n + D256_BM32_PHASE_PAGE_BLOCK_SIZE - 1)
        / D256_BM32_PHASE_PAGE_BLOCK_SIZE;
    TORCH_CHECK(block_table.size(1) >= required_pages,
                "block_table does not cover actual_n host length");

    if (out_.has_value()) {
        const at::Tensor& out = out_.value();
        TORCH_CHECK(out.is_cuda() && out.device() == q.device(),
                    "out must be a CUDA tensor on the q device");
        TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
        TORCH_CHECK(out.sizes() == q.sizes(), "out must have shape [B, H, M, D]");
        TORCH_CHECK(out.is_contiguous(), "out must be contiguous [B, H, M, D]");
    }
    if (softmax_lse_.has_value()) {
        const at::Tensor& softmax_lse = softmax_lse_.value();
        TORCH_CHECK(softmax_lse.is_cuda() && softmax_lse.device() == q.device(),
                    "softmax_lse must be a CUDA tensor on the q device");
        TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32,
                    "softmax_lse must be fp32");
        TORCH_CHECK(softmax_lse.dim() == 3 && softmax_lse.size(0) == B
                        && softmax_lse.size(1) == H && softmax_lse.size(2) == M,
                    "softmax_lse must have shape [B, H, M]");
        TORCH_CHECK(softmax_lse.is_contiguous(),
                    "softmax_lse must be contiguous [B, H, M]");
    }

    const std::vector<int64_t> expected_partial_out = {
        B, H, D256_BM32_PHASE_SPLIT_PARTS, M, D};
    const std::vector<int64_t> expected_partial_rows = {
        B, H, D256_BM32_PHASE_SPLIT_PARTS, M};
    for (const at::Tensor* workspace : {
             &split_tmp_out, &split_tmp_row_max, &split_tmp_row_sum}) {
        TORCH_CHECK(workspace->is_cuda() && workspace->device() == q.device(),
                    "split-KV workspaces must be CUDA tensors on the q device");
        TORCH_CHECK(workspace->dtype() == torch::kFloat32,
                    "split-KV workspaces must be fp32");
        TORCH_CHECK(workspace->is_contiguous(),
                    "split-KV workspaces must be contiguous");
    }
    TORCH_CHECK(split_tmp_out.sizes() == expected_partial_out,
                "split_tmp_out must have shape [B, H, 3, M, D]");
    TORCH_CHECK(split_tmp_row_max.sizes() == expected_partial_rows,
                "split_tmp_row_max must have shape [B, H, 3, M]");
    TORCH_CHECK(split_tmp_row_sum.sizes() == expected_partial_rows,
                "split_tmp_row_sum must have shape [B, H, 3, M]");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t capture_error = cudaStreamIsCapturing(stream, &capture_status);
    TORCH_CHECK(capture_error == cudaSuccess, "cudaStreamIsCapturing failed: ",
                cudaGetErrorString(capture_error));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "fixed BM32 split-KV CUDA graph capture is not validated");

    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0,
                "fixed BM32 split-KV prefill supports only SM70 GPUs");

    at::Tensor out_fp16 = out_.has_value() ? out_.value() : torch::empty_like(q);
    at::Tensor softmax_lse =
        softmax_lse_.has_value()
            ? softmax_lse_.value()
            : torch::empty({B, H, M},
                           torch::dtype(torch::kFloat32).device(q.device()));
    launch_flash_attention_forward_paged_d256_bm32_splitkv3_kernel(
        q, k_cache, v_cache, out_fp16, softmax_lse,
        split_tmp_out, split_tmp_row_max, split_tmp_row_sum,
        block_table, static_cast<int>(actual_n),
        softmax_scale, stream);

    return {out_fp16, softmax_lse};
}

at::Tensor flash_attention_prefill_paged_bfla(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& bfla_block_mask,
    const int bfla_mask_block_n,
    const float softmax_scale,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale,
    const bool is_causal,
    const int window_size_left,
    const int window_size_right
) {
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
    TORCH_CHECK(kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16,
                "BFLA paged prefill supports fp16 KV cache only");
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
    TORCH_CHECK(v_cache.dtype() == torch::kFloat16, "v_cache must be fp16");
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "Tensors must be on CUDA");
    TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
                "block_table and seq_lens must be CUDA tensors");
    TORCH_CHECK(bfla_block_mask.is_cuda(),
                "bfla_block_mask must be a CUDA tensor");
    TORCH_CHECK(bfla_block_mask.dtype() == torch::kInt32,
                "bfla_block_mask must be int32");
    TORCH_CHECK(bfla_block_mask.dim() == 4,
                "bfla_block_mask must have shape [B, Hkv, q_tiles, kv_tiles]");
    TORCH_CHECK(q.stride(-1) == 1 && k_cache.stride(-1) == 1 &&
                    v_cache.stride(-1) == 1,
                "Last dim must be contiguous");
    TORCH_CHECK(k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0 &&
                    v_cache.stride(1) % 8 == 0 && v_cache.stride(2) % 8 == 0,
                "Paged KV strides must be divisible by 8 half elements");

    const int B = q.size(0);
    const int H = q.size(1);
    const int M = q.size(2);
    const int D = q.size(3);
    const int num_kv_heads = k_cache.size(2);
    const int page_block_size = k_cache.size(1);

    TORCH_CHECK(D == 256, "BFLA paged prefill supports D=256 only");
    TORCH_CHECK(H % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(M > 1, "BFLA paged prefill requires prefill M > 1");
    TORCH_CHECK(page_block_size >= 16 && (page_block_size % 16) == 0,
                "BFLA paged prefill requires page block size multiple of 16");
    TORCH_CHECK(is_causal, "BFLA paged prefill supports causal attention only");
    TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                "BFLA paged prefill does not support sliding window");
    TORCH_CHECK(bfla_mask_block_n > 0,
                "bfla_mask_block_n must be positive");
    TORCH_CHECK(B <= bfla_block_mask.size(0),
                "bfla_block_mask batch dimension must cover q");
    TORCH_CHECK(num_kv_heads <= bfla_block_mask.size(1),
                "bfla_block_mask head dimension must cover KV heads");

    at::Tensor out_fp16 = out_.has_value() ? out_.value() : torch::zeros_like(q);
    auto softmax_lse = torch::zeros({B, H, M},
                                    torch::dtype(torch::kFloat32).device(q.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props = at::cuda::getCurrentDeviceProperties();
    bool sm70 = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    launcher_flash_attention_forward_paged<256, flash_v100::KV_CACHE_DTYPE_FP16>(
        q, k_cache, v_cache, out_fp16, softmax_lse, block_table, seq_lens,
        softmax_scale, is_causal, k_scale, v_scale, window_size_left,
        window_size_right, stream, bfla_block_mask.data_ptr<int>(),
        bfla_mask_block_n, bfla_block_mask.stride(0),
        bfla_block_mask.stride(1), bfla_block_mask.stride(2),
        bfla_block_mask.stride(3));

    return out_fp16;
}

at::Tensor flash_attention_decode_paged_wmma(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const float softmax_scale,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale
) {
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
    TORCH_CHECK(kv_dtype_code >= 0, "Unsupported kv_cache_dtype: ", kv_cache_dtype);
    if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
        TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
        TORCH_CHECK(v_cache.dtype() == torch::kFloat16, "v_cache must be fp16");
    } else {
        TORCH_CHECK(k_cache.dtype() == torch::kUInt8,
                    "fp8 k_cache must be stored as uint8");
        TORCH_CHECK(v_cache.dtype() == torch::kUInt8,
                    "fp8 v_cache must be stored as uint8");
        TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
                    "fp8 k/v scales must be positive");
    }
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "Tensors must be on CUDA");
    TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
                "block_table and seq_lens must be CUDA tensors");
    TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
    TORCH_CHECK(block_table.dim() == 2,
                "block_table must have shape [B, max_num_blocks]");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
    TORCH_CHECK(q.stride(-1) == 1 && k_cache.stride(-1) == 1 &&
                    v_cache.stride(-1) == 1,
                "Last dim must be contiguous");
    TORCH_CHECK(k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0 &&
                    v_cache.stride(1) % 8 == 0 && v_cache.stride(2) % 8 == 0,
                "Paged KV strides must be divisible by 8 half elements");

    const int B = q.size(0);
    const int H = q.size(1);
    const int D = q.size(2);
    const int num_kv_heads = k_cache.size(2);

    TORCH_CHECK(B <= block_table.size(0), "block_table batch size must cover q");
    TORCH_CHECK(B <= seq_lens.size(0), "seq_lens batch size must cover q");
    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0,
                "D must be even, <=256, multiple of 8");
    TORCH_CHECK(H % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads");

    at::Tensor out_fp16 = out_.has_value() ? out_.value() : torch::zeros_like(q);
    TORCH_CHECK(out_fp16.is_cuda(), "out must be on CUDA");
    TORCH_CHECK(out_fp16.dtype() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(out_fp16.sizes() == q.sizes(), "out must have same shape as q");
    TORCH_CHECK(out_fp16.stride(-1) == 1, "out last dim must be contiguous");

    at::Tensor q_m1 = q.unsqueeze(2);
    at::Tensor out_m1 = out_fp16.unsqueeze(2);
    auto softmax_lse = torch::zeros({B, H, 1},
                                    torch::dtype(torch::kFloat32).device(q.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props = at::cuda::getCurrentDeviceProperties();
    bool sm70 = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    #define LAUNCH_DECODE_WMMA_TYPED(HDIM, KV_DTYPE_CODE)                       \
        launcher_flash_attention_forward_paged<HDIM, KV_DTYPE_CODE>(            \
            q_m1, k_cache, v_cache, out_m1, softmax_lse, block_table, seq_lens, \
            softmax_scale, true, k_scale, v_scale, -1, -1, stream)

    #define LAUNCH_DECODE_WMMA_BY_KV(HDIM)                                      \
        do {                                                                    \
            switch (kv_dtype_code) {                                            \
                case flash_v100::KV_CACHE_DTYPE_FP16:                           \
                    LAUNCH_DECODE_WMMA_TYPED(HDIM, flash_v100::KV_CACHE_DTYPE_FP16); \
                    break;                                                      \
                case flash_v100::KV_CACHE_DTYPE_FP8_E4M3:                       \
                    LAUNCH_DECODE_WMMA_TYPED(HDIM, flash_v100::KV_CACHE_DTYPE_FP8_E4M3); \
                    break;                                                      \
                case flash_v100::KV_CACHE_DTYPE_FP8_E5M2:                       \
                    LAUNCH_DECODE_WMMA_TYPED(HDIM, flash_v100::KV_CACHE_DTYPE_FP8_E5M2); \
                    break;                                                      \
                default:                                                        \
                    TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype); \
            }                                                                   \
        } while (0)

    switch (D) {
        case 16:
            LAUNCH_DECODE_WMMA_BY_KV(16);
            break;
        case 32:
            LAUNCH_DECODE_WMMA_BY_KV(32);
            break;
        case 64:
            LAUNCH_DECODE_WMMA_BY_KV(64);
            break;
        case 128:
            LAUNCH_DECODE_WMMA_BY_KV(128);
            break;
        case 256:
            LAUNCH_DECODE_WMMA_BY_KV(256);
            break;
        default:
            TORCH_CHECK(false, "Unsupported D: ", D);
    }

    #undef LAUNCH_DECODE_WMMA_BY_KV
    #undef LAUNCH_DECODE_WMMA_TYPED

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out_fp16;
}
