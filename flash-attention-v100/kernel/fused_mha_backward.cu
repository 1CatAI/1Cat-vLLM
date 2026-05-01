#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <mma.h>
using namespace nvcuda::wmma;

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

#define MAX_REG_PER_UNIT        256
#define MAX_REG_PER_THREAD      255
#define MAX_REG_PER_BLOCK       65536
#define MAX_REG_BUFFER          65536

#define BLOCK_M_16  16
#define BLOCK_N_16  256
#define WARPS_16    16

#define BLOCK_M_32  32
#define BLOCK_N_32  128
#define WARPS_32    16

#define BLOCK_M_64  64
#define BLOCK_N_64  80
#define WARPS_64    16

#define BLOCK_M_128 32
#define BLOCK_N_128 112
#define WARPS_128   16

#define BLOCK_M_256 32
#define BLOCK_N_256 32
#define WARPS_256   16

#define BLOCK_KV_16  32
#define BLOCK_Q_16   224
#define WARPS_DKV_16 14

#define BLOCK_KV_32  32
#define BLOCK_Q_32   192
#define WARPS_DKV_32 12

#define BLOCK_KV_64  32
#define BLOCK_Q_64   128
#define WARPS_DKV_64  8

#define BLOCK_KV_128  16
#define BLOCK_Q_128   144
#define WARPS_DKV_128 12

#define BLOCK_KV_256  16
#define BLOCK_Q_256   64
#define WARPS_DKV_256 16

template<int D>
struct dQKernelConfig {
    static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 : (D == 32) ? BLOCK_M_32 : (D == 64)  ? BLOCK_M_64 : (D == 128) ? BLOCK_M_128 : BLOCK_M_256;
    static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 : (D == 32) ? BLOCK_N_32 : (D == 64)  ? BLOCK_N_64 : (D == 128) ? BLOCK_N_128 : BLOCK_N_256;
    static constexpr int WARPS_PER_BLOCK = (D == 16) ? WARPS_16 : (D == 32) ? WARPS_32 : (D == 64) ? WARPS_64 : (D == 128) ? WARPS_128 : WARPS_256;

    static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
    static constexpr int THREADS_PER_ROW    = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int PAD                = (8 - (D % 32) + 32) % 32;
    static constexpr int Q_STRIDE           = D + PAD;
    static constexpr int KV_STRIDE          = D + PAD;
    static constexpr int S_STRIDE           = BLOCK_N + PAD;
    static constexpr int PER_UINT4          = 8;
    static constexpr int NUM_UINT4_Q_BLOCK  = BLOCK_M * ((D + PER_UINT4 - 1) / PER_UINT4);
    static constexpr int NUM_UINT4_KV_BLOCK = BLOCK_N * ((D + PER_UINT4 - 1) / PER_UINT4);

    struct alignas(128) SmemLayout {
        union {
            alignas(16) __half k     [BLOCK_N * KV_STRIDE];
            alignas(16) __half v     [BLOCK_N * KV_STRIDE];
        } reuse_kv;
            alignas(16) __half dO    [BLOCK_M * Q_STRIDE];
            alignas(16) __half q     [BLOCK_M * Q_STRIDE];
            alignas(16) float  s     [BLOCK_M * S_STRIDE];
        union {
            alignas(16) float  dOV   [BLOCK_M * S_STRIDE];
            alignas(16) __half dS    [BLOCK_M * S_STRIDE];
        } reuse_sdOVS;
            alignas(16) float row_dot[BLOCK_M];
            alignas(16) float lse    [BLOCK_M];
            alignas(16) float dQ     [BLOCK_M * Q_STRIDE];
    };

    static constexpr size_t TOTAL_SMEM = ((sizeof(SmemLayout) + 127) & ~size_t(127));
};

template<typename Config>
__device__ __forceinline__ void init_smem(char* smem_raw) {
    constexpr int N_U4 = Config::TOTAL_SMEM / 16;
    const int lane_id = threadIdx.x & 31;

    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    #pragma unroll 1
    for (int i = lane_id; i < N_U4; i += 32) {
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};"
                     :: "r"(addr + (i << 4)), "r"(0) : "memory");
    }
    __syncwarp();
}

template<int D>
struct dKVKernelConfig {
    static constexpr int BLOCK_M = (D == 16) ? BLOCK_KV_16 : (D == 32) ? BLOCK_KV_32 : (D == 64) ? BLOCK_KV_64 : (D == 128) ? BLOCK_KV_128 : BLOCK_KV_256;
    static constexpr int BLOCK_N = (D == 16) ? BLOCK_Q_16 : (D == 32) ? BLOCK_Q_32 : (D == 64) ? BLOCK_Q_64 : (D == 128) ? BLOCK_Q_128 : BLOCK_Q_256;
    static constexpr int WARPS_PER_BLOCK = (D == 16) ? WARPS_DKV_16 : (D == 32) ? WARPS_DKV_32 : (D == 64) ? WARPS_DKV_64 : (D == 128) ? WARPS_DKV_128 : WARPS_DKV_256;

    static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
    static constexpr int THREADS_PER_ROW    = THREADS_PER_BLOCK / BLOCK_N;
    static constexpr int PAD                = 8;
    static constexpr int Q_STRIDE           = D + PAD;
    static constexpr int KV_STRIDE          = D + PAD;
    static constexpr int S_STRIDE           = BLOCK_M + PAD;
    static constexpr int PER_UINT4          = 8;
    static constexpr int NUM_UINT4_Q_BLOCK  = BLOCK_N * ((D + PER_UINT4 - 1) / PER_UINT4);
    static constexpr int NUM_UINT4_KV_BLOCK = BLOCK_M * ((D + PER_UINT4 - 1) / PER_UINT4);

    struct alignas(128) SmemLayout {
            alignas(16) __half k     [BLOCK_M * KV_STRIDE];
            alignas(16) __half v     [BLOCK_M * KV_STRIDE];
        union {
            alignas(16) __half dO    [BLOCK_N * Q_STRIDE];
            alignas(16) __half q     [BLOCK_N * Q_STRIDE];
        } reuse_qdO;
        union {
            alignas(16) float  s     [BLOCK_N * S_STRIDE];
            alignas(16) __half p     [BLOCK_N * BLOCK_M];
        } reuse_sp;
        union {
            alignas(16) float  dOV   [BLOCK_N * S_STRIDE];
            alignas(16) __half dS    [BLOCK_N * BLOCK_M];
        } reuse_dOVS;
            alignas(16) float row_dot[BLOCK_N];
            alignas(16) float lse    [BLOCK_N];
            alignas(16) float dK     [BLOCK_M * KV_STRIDE];
            alignas(16) float dV     [BLOCK_M * KV_STRIDE];
    };

    static constexpr size_t TOTAL_SMEM = ((sizeof(SmemLayout) + 127) & ~size_t(127));
};

template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(dQKernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_backward_dq_kernel(
    const __half*  __restrict__ Q,
    const __half*  __restrict__ K,
    const __half*  __restrict__ V,
    const __half*  __restrict__ O,
    const __half*  __restrict__ dO,
    const  float*  __restrict__ softmax_lse,
          __half*  __restrict__ dQ,
    const int B,
    const int H,
    const int M,
    const int N,
    const float softmax_scale
) {
    using Config = dQKernelConfig<D>;

    constexpr int BLOCK_M           = Config::BLOCK_M;
    constexpr int BLOCK_N           = Config::BLOCK_N;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW   = Config::THREADS_PER_ROW;
    constexpr int WARPS_PER_BLOCK   = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE          = Config::Q_STRIDE;
    constexpr int KV_STRIDE         = Config::KV_STRIDE;
    constexpr int S_STRIDE          = Config::S_STRIDE;
    constexpr int PER_UINT4         = Config::PER_UINT4;
    constexpr int NUM_UINT4_Q_BLOCK  = Config::NUM_UINT4_Q_BLOCK;
    constexpr int NUM_UINT4_KV_BLOCK = Config::NUM_UINT4_KV_BLOCK;

    const float NEG_INF = -1e30f;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_m   = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;

    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int valid_q_rows = min(BLOCK_M, M - start_row);

    if constexpr (IS_CAUSAL) {
        const int max_key_pos = start_row + valid_q_rows - 1;
        if (max_key_pos < 0) {
            num_n_tiles = 0;
        } else {
            num_n_tiles = min(num_n_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
        }
    }

    const int tid          = threadIdx.x;
    const int warp_id      = tid / MAX_THREADS_PER_WARP;
    const int lane_id      = tid % MAX_THREADS_PER_WARP;

    const __half* q_ptr   = Q           + (size_t)batch_head_id * M * D + start_row * D;
    const __half* k_ptr   = K           + (size_t)batch_head_id * N * D;
    const __half* v_ptr   = V           + (size_t)batch_head_id * N * D;
    const __half* o_ptr   = O           + (size_t)batch_head_id * M * D + start_row * D;
    const __half* dO_ptr  = dO          + (size_t)batch_head_id * M * D + start_row * D;
          __half* dQ_ptr  = dQ          + (size_t)batch_head_id * M * D + start_row * D;
    const float*  lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_row;

    extern __shared__ char smem_raw[];
    init_smem<Config>(smem_raw);
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* sK      = smem.reuse_kv.k;
    __half* sV      = smem.reuse_kv.v;
    __half* sdO     = smem.dO;
    __half* sQ      = smem.q;
     float* sS      = smem.s;
     float* sdOV    = smem.reuse_sdOVS.dOV;
    __half* sdS     = smem.reuse_sdOVS.dS;
     float* sRowDot = smem.row_dot;
     float* sLse    = smem.lse;
     float* sdQ     = smem.dQ;

    const int  d_stride_uint4 = (D + PER_UINT4 - 1) / PER_UINT4;
    const int  q_stride_uint4 = (Q_STRIDE  + PER_UINT4 - 1) / PER_UINT4;
    const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

    const uint4* q_vec  = reinterpret_cast<const uint4*>(q_ptr);
    const uint4* do_vec = reinterpret_cast<const uint4*>(dO_ptr);
    uint4* sQ_vec  = reinterpret_cast<uint4*>(sQ);
    uint4* sdO_vec = reinterpret_cast<uint4*>(sdO);

    #pragma unroll 2
    for (int idx = tid; idx < NUM_UINT4_Q_BLOCK; idx += THREADS_PER_BLOCK) {
        const int row = idx / d_stride_uint4;
        const int vec_col = idx % d_stride_uint4;

        uint4 q_val  = make_uint4(0, 0, 0, 0);
        uint4 do_val = make_uint4(0, 0, 0, 0);

        if (row < valid_q_rows && vec_col < d_stride_uint4) {
            q_val =  __ldg(&q_vec[row * d_stride_uint4 + vec_col]);
            do_val = __ldg(&do_vec[row * d_stride_uint4 + vec_col]);
        }
         sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
        sdO_vec[row * q_stride_uint4 + vec_col] = do_val;
    }
    __syncthreads();

    if (tid < valid_q_rows * THREADS_PER_ROW) {
        const int row = tid / THREADS_PER_ROW;
        const int thread_in_row = tid % THREADS_PER_ROW;
        const int fp16_x4_per_row = D / 4;
        const int work_per_thread = (fp16_x4_per_row + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
        const unsigned mask = (valid_q_rows == BLOCK_M) ? 0xFFFFFFFFU : __activemask();

        float thread_dot = 0.0f;

        #pragma unroll
        for (int j = 0; j < work_per_thread; ++j) {
            const int chunk_idx = thread_in_row + j * THREADS_PER_ROW;
            if (chunk_idx >= fp16_x4_per_row) break;
            const int col = chunk_idx * 4;

            const __half* o_addr = o_ptr + row * D + col;
            ushort o_h0, o_h1, o_h2, o_h3;
            asm volatile(
                "ld.global.v4.u16 {%0, %1, %2, %3}, [%4];"
                : "=h"(o_h0), "=h"(o_h1), "=h"(o_h2), "=h"(o_h3)
                : "l"(o_addr)
                : "memory"
            );

            const __half* dO_addr = sdO + row * Q_STRIDE + col;
            ushort d_h0, d_h1, d_h2, d_h3;
            const uint32_t ptr_dO = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(dO_addr)));
            asm volatile(
                "ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
                : "=h"(d_h0), "=h"(d_h1), "=h"(d_h2), "=h"(d_h3)
                : "r"(ptr_dO)
                : "memory"
            );

            const float fo_0 = __half2float(__ushort_as_half(o_h0));
            const float fo_1 = __half2float(__ushort_as_half(o_h1));
            const float fo_2 = __half2float(__ushort_as_half(o_h2));
            const float fo_3 = __half2float(__ushort_as_half(o_h3));

            const float fd_0 = __half2float(__ushort_as_half(d_h0));
            const float fd_1 = __half2float(__ushort_as_half(d_h1));
            const float fd_2 = __half2float(__ushort_as_half(d_h2));
            const float fd_3 = __half2float(__ushort_as_half(d_h3));

            thread_dot = __fmaf_rn(fo_0, fd_0, thread_dot);
            thread_dot = __fmaf_rn(fo_1, fd_1, thread_dot);
            thread_dot = __fmaf_rn(fo_2, fd_2, thread_dot);
            thread_dot = __fmaf_rn(fo_3, fd_3, thread_dot);
        }

        #pragma unroll
        for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
            thread_dot += __shfl_down_sync(mask, thread_dot, o, THREADS_PER_ROW);

        if (thread_in_row == 0) {
            sRowDot[row] = thread_dot;
        }
    }

    if (tid < valid_q_rows) { sLse[tid] = lse_ptr[tid]; }
    __syncthreads();

    for (int block_n = 0; block_n < num_n_tiles; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= N) break;
        const int valid_k_rows = min(BLOCK_N, N - start_col);

        if constexpr (IS_CAUSAL) {
            if (start_col >= start_row + valid_q_rows) { continue; }
        }

        const uint4* v_vec        = reinterpret_cast<const uint4*>(v_ptr + start_col * D);
        uint4*       sV_vec       = reinterpret_cast<uint4*>(sV);

        #pragma unroll 2
        for (int idx = tid; idx < NUM_UINT4_KV_BLOCK; idx += THREADS_PER_BLOCK) {
            const int row = idx / d_stride_uint4;
            const int vec_col = idx % d_stride_uint4;
            uint4 v_val = make_uint4(0, 0, 0, 0);
            if (row < valid_k_rows && vec_col < d_stride_uint4) {
                v_val = __ldg(&v_vec[row * d_stride_uint4 + vec_col]);
            }
            sV_vec[row * kv_stride_uint4 + vec_col] = v_val;
        }
        __syncthreads();

        const int num_tiles_m_dov    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_dov    = (BLOCK_N + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_dov    = (D + WMMA_K - 1) / WMMA_K;
        const int total_tiles_dov    = num_tiles_m_dov * num_tiles_n_dov;
        const int tiles_per_warp_dov = (total_tiles_dov + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        for (int tile_idx = 0; tile_idx < tiles_per_warp_dov; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp_dov + tile_idx;

            if (global_tile_idx >= total_tiles_dov) break;

            const int tile_m_idx = global_tile_idx / num_tiles_n_dov;
            const int tile_n_idx = global_tile_idx % num_tiles_n_dov;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k_dov; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= D) break;
                load_matrix_sync(a_frag, sdO + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sV + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            store_matrix_sync(sdOV + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        const uint4* k_vec     = reinterpret_cast<const uint4*>(k_ptr + start_col * D);
        uint4*       sK_vec    = reinterpret_cast<uint4*>(sK);

        #pragma unroll 2
        for (int idx = tid; idx < NUM_UINT4_KV_BLOCK; idx += THREADS_PER_BLOCK) {
            const int row = idx / d_stride_uint4;
            const int vec_col = idx % d_stride_uint4;
            uint4 k_val = make_uint4(0, 0, 0, 0);
            if (row < valid_k_rows && vec_col < d_stride_uint4) {
                k_val = __ldg(&k_vec[row * d_stride_uint4 + vec_col]);
            }
            sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
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

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
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

            if constexpr (IS_CAUSAL) {
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                    const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

                    const int global_m = start_row + tile_m + row;
                    const int global_n = start_col + tile_n + col;

                    const bool is_valid = (global_m < start_row + valid_q_rows) &&
                                          (global_n < start_col + valid_k_rows);

                    acc_frag.x[i] = is_valid
                        ? ((global_n > global_m) ? NEG_INF : acc_frag.x[i] * softmax_scale)
                        : NEG_INF;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    acc_frag.x[i] *= softmax_scale;
                }
            }
            store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;

            float*  sS_row   = sS   + row * S_STRIDE;
            float*  sdOV_row = sdOV + row * S_STRIDE;
            __half* sdS_row  = sdS  + row * S_STRIDE;

            const float lse_val     = sLse[row];
            const float row_dot_val = sRowDot[row];

            const int vec8_cols = valid_k_rows / 8;
            const int vec8_per_thread = (vec8_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
            const int tail_start = vec8_cols * 8;

            float4* sS_vec4    = reinterpret_cast<float4*>(sS_row);
            float4* sdOV_vec4  = reinterpret_cast<float4*>(sdOV_row);
            uint4*  sdS_vec_u4 = reinterpret_cast<uint4*>(sdS_row);

            uint4 buf[8]; int cnt = 0;

            #pragma unroll
            for (int j = 0; j < vec8_per_thread; ++j) {
                const int v8 = thread_in_row + j * THREADS_PER_ROW;
                if (v8 >= vec8_cols) break;

                float4 s0 = sS_vec4[v8 * 2],   s1 = sS_vec4[v8 * 2 + 1];
                float4 d0 = sdOV_vec4[v8 * 2], d1 = sdOV_vec4[v8 * 2 + 1];

                #define COMP(i, sf, df) \
                    float sh##i = (sf) - lse_val; \
                    float p##i = (sh##i < -80.0f) ? 0.0f : __expf(sh##i); \
                    float ds##i = p##i * softmax_scale * ((df) - row_dot_val);

                COMP(0, s0.x, d0.x) COMP(1, s0.y, d0.y) COMP(2, s0.z, d0.z) COMP(3, s0.w, d0.w)
                COMP(4, s1.x, d1.x) COMP(5, s1.y, d1.y) COMP(6, s1.z, d1.z) COMP(7, s1.w, d1.w)
                #undef COMP

                uint4 res;
                asm volatile(
                    "{ mov.b32 %0, {%4,%5}; mov.b32 %1, {%6,%7}; mov.b32 %2, {%8,%9}; mov.b32 %3, {%10,%11}; }\n"
                    : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
                    : "h"(__half_as_ushort(__float2half_rn(ds0))),
                      "h"(__half_as_ushort(__float2half_rn(ds1))),
                      "h"(__half_as_ushort(__float2half_rn(ds2))),
                      "h"(__half_as_ushort(__float2half_rn(ds3))),
                      "h"(__half_as_ushort(__float2half_rn(ds4))),
                      "h"(__half_as_ushort(__float2half_rn(ds5))),
                      "h"(__half_as_ushort(__float2half_rn(ds6))),
                      "h"(__half_as_ushort(__float2half_rn(ds7)))
                );
                buf[cnt++] = res;
            }

            #pragma unroll
            for (int c = tail_start + thread_in_row; c < BLOCK_N; c += THREADS_PER_ROW) {
                float s = (c < valid_k_rows) ? sS_row[c] : NEG_INF;
                float dov = (c < valid_k_rows) ? sdOV_row[c] : 0.0f;
                float p = (s - lse_val < -80.0f) ? 0.0f : __expf(s - lse_val);
                float ds = p * softmax_scale * ((c < valid_k_rows) ? (dov - row_dot_val) : 0.0f);
                sdS_row[c] = __float2half_rn(ds);
            }

            #pragma unroll
            for (int i = 0; i < cnt; ++i) {
                    const int v8 = thread_in_row + i * THREADS_PER_ROW;
                    if (v8 < vec8_cols) {
                        sdS_vec_u4[v8] = buf[i];
                    }
            }
        }
        __syncthreads();

        const int num_tiles_m_dq    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_dq    = (D + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_dq    = (BLOCK_N + WMMA_K - 1) / WMMA_K;
        const int total_tiles_dq    = num_tiles_m_dq * num_tiles_n_dq;
        const int tiles_per_warp_dq = (total_tiles_dq + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        for (int tile_local = 0; tile_local < tiles_per_warp_dq; ++tile_local) {
            const int global_tile_idx = warp_id * tiles_per_warp_dq + tile_local;
            if (global_tile_idx >= total_tiles_dq) break;

            const int tile_m_idx = global_tile_idx / num_tiles_n_dq;
            const int tile_n_idx = global_tile_idx % num_tiles_n_dq;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_q_rows || tile_n >= D) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k_dq; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= valid_k_rows) break;

                load_matrix_sync(a_frag, sdS + tile_m * S_STRIDE + k_offset, S_STRIDE);
                load_matrix_sync(b_frag, sK + k_offset * KV_STRIDE + tile_n, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> curr_frag;
            load_matrix_sync(curr_frag, sdQ + tile_m * Q_STRIDE + tile_n, Q_STRIDE, mem_row_major);

            #pragma unroll
            for (int i = 0; i < curr_frag.num_elements; ++i) {
                curr_frag.x[i] += acc_frag.x[i];
            }
            store_matrix_sync(sdQ + tile_m * Q_STRIDE + tile_n, curr_frag, Q_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    const int total_fp16_x4 = (valid_q_rows * D) / 4;
    for (int i = tid; i < total_fp16_x4; i += THREADS_PER_BLOCK) {
        const int row = i / (D / 4);
        const int col = (i % (D / 4)) * 4;

        const float* s_dQ_row = sdQ + row * Q_STRIDE;

        const __half h0 = __float2half_rn(s_dQ_row[col + 0]);
        const __half h1 = __float2half_rn(s_dQ_row[col + 1]);
        const __half h2 = __float2half_rn(s_dQ_row[col + 2]);
        const __half h3 = __float2half_rn(s_dQ_row[col + 3]);

        asm volatile(
            "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dQ_ptr + row * D + col),
              "h"(__half_as_ushort(h0)),
              "h"(__half_as_ushort(h1)),
              "h"(__half_as_ushort(h2)),
              "h"(__half_as_ushort(h3))
            : "memory"
        );
    }
}

template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(dKVKernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_backward_dkv_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    const  float* __restrict__ softmax_lse,
          __half* __restrict__ dK,
          __half* __restrict__ dV,
    const int B,
    const int H,
    const int M,
    const int N,
    const float softmax_scale
) {
    using Config = dKVKernelConfig<D>;
    constexpr int BLOCK_M            = Config::BLOCK_M;
    constexpr int BLOCK_N            = Config::BLOCK_N;
    constexpr int THREADS_PER_BLOCK  = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW    = Config::THREADS_PER_ROW;
    constexpr int WARPS_PER_BLOCK    = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE           = Config::Q_STRIDE;
    constexpr int KV_STRIDE          = Config::KV_STRIDE;
    constexpr int S_STRIDE           = Config::S_STRIDE;
    constexpr int PER_UINT4          = Config::PER_UINT4;
    constexpr int NUM_UINT4_Q_BLOCK  = Config::NUM_UINT4_Q_BLOCK;
    constexpr int NUM_UINT4_KV_BLOCK = Config::NUM_UINT4_KV_BLOCK;

    const float NEG_INF = -1e30f;
    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_kv = blockIdx.x;
    const int start_kv = block_kv * BLOCK_M;
    if (start_kv >= N) return;

    int num_q_tiles = (M + BLOCK_N - 1) / BLOCK_N;
    const int valid_kv_rows = min(BLOCK_M, N - start_kv);

    const int tid     = threadIdx.x;
    const int warp_id = tid / MAX_THREADS_PER_WARP;
    const int lane_id = tid % MAX_THREADS_PER_WARP;

    const __half*   q_ptr = Q           + (size_t)batch_head_id * M * D;
    const __half*   k_ptr = K           + (size_t)batch_head_id * N * D + start_kv * D;
    const __half*   v_ptr = V           + (size_t)batch_head_id * N * D + start_kv * D;
    const __half*   o_ptr = O           + (size_t)batch_head_id * M * D;
    const __half*  dO_ptr = dO          + (size_t)batch_head_id * M * D;
    const  float* lse_ptr = softmax_lse + (size_t)batch_head_id * M;
          __half*  dK_ptr = dK          + (size_t)batch_head_id * N * D + start_kv * D;
          __half*  dV_ptr = dV          + (size_t)batch_head_id * N * D + start_kv * D;

    extern __shared__ char smem_raw[];
    init_smem<Config>(smem_raw);
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* sK       = smem.k;
    __half* sV       = smem.v;
    __half* sdO      = smem.reuse_qdO.dO;
    __half* sQ       = smem.reuse_qdO.q;
     float* sS       = smem.reuse_sp.s;
    __half* sP       = smem.reuse_sp.p;
     float* sdOV     = smem.reuse_dOVS.dOV;
    __half* sdS      = smem.reuse_dOVS.dS;
     float* sRowDot  = smem.row_dot;
     float* sLse     = smem.lse;
     float* sdK      = smem.dK;
     float* sdV      = smem.dV;

    const int  d_stride_uint4 = (D + PER_UINT4 - 1) / PER_UINT4;
    const int  q_stride_uint4 = (Q_STRIDE  + PER_UINT4 - 1) / PER_UINT4;
    const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

    const uint4* k_vec = reinterpret_cast<const uint4*>(k_ptr);
    const uint4* v_vec = reinterpret_cast<const uint4*>(v_ptr);
    uint4* sK_vec = reinterpret_cast<uint4*>(sK);
    uint4* sV_vec = reinterpret_cast<uint4*>(sV);

    #pragma unroll 2
    for (int idx = tid; idx < NUM_UINT4_KV_BLOCK; idx += THREADS_PER_BLOCK) {
        const int row = idx / d_stride_uint4;
        const int vec_col = idx % d_stride_uint4;

        uint4 k_val = make_uint4(0, 0, 0, 0);
        uint4 v_val = make_uint4(0, 0, 0, 0);

        if (row < valid_kv_rows && vec_col < d_stride_uint4) {
            k_val = __ldg(&k_vec[row * d_stride_uint4 + vec_col]);
            v_val = __ldg(&v_vec[row * d_stride_uint4 + vec_col]);
        }
        sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
        sV_vec[row * kv_stride_uint4 + vec_col] = v_val;
    }
    __syncthreads();

    for (int block_n = 0; block_n < num_q_tiles; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= M) break;
        const int valid_q_rows = min(BLOCK_N, M - start_col);

        if constexpr (IS_CAUSAL) {
            if (start_kv >= start_col + valid_q_rows) { continue; }
        }

        const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr + start_col * D);
        uint4*      sQ_vec = reinterpret_cast<uint4*>(sdO);

        #pragma unroll 2
        for (int idx = tid; idx < NUM_UINT4_Q_BLOCK; idx += THREADS_PER_BLOCK) {
            const int row = idx / d_stride_uint4;
            const int vec_col = idx % d_stride_uint4;
            uint4 q_val = make_uint4(0, 0, 0, 0);
            if (row < valid_q_rows && vec_col < d_stride_uint4) {
                q_val = __ldg(&q_vec[row * d_stride_uint4 + vec_col]);
            }
            sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
        }
        __syncthreads();

        const int num_tiles_m_qk    = (BLOCK_N + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_qk    = (BLOCK_M + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_qk    = (D + WMMA_K - 1) / WMMA_K;
        const int total_tiles_qk    = num_tiles_m_qk * num_tiles_n_qk;
        const int tiles_per_warp_qk = (total_tiles_qk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned row_causal   = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 4;
        const unsigned col_causal   = ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

        for (int tile_local = 0; tile_local < tiles_per_warp_qk; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_qk + tile_local;
            if (tile_idx >= total_tiles_qk) break;

            const int tile_m_idx = tile_idx / num_tiles_n_qk;
            const int tile_n_idx = tile_idx % num_tiles_n_qk;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_q_rows || tile_n >= valid_kv_rows) continue;

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

            if constexpr (IS_CAUSAL) {
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                    const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

                    const int global_m = start_col + tile_m + row;
                    const int global_n = start_kv  + tile_n + col;

                    const bool is_valid = (global_m < start_col + valid_q_rows) &&
                                          (global_n < start_kv  + valid_kv_rows);

                    acc_frag.x[i] = is_valid
                        ? ((global_n > global_m) ? NEG_INF : acc_frag.x[i] * softmax_scale)
                        : NEG_INF;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    acc_frag.x[i] *= softmax_scale;
                }
            }
            store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        const uint4* do_vec = reinterpret_cast<const uint4*>(dO_ptr + start_col * D);
        uint4*      sdO_vec = reinterpret_cast<uint4*>(sdO);

        #pragma unroll 2
        for (int idx = tid; idx < NUM_UINT4_Q_BLOCK; idx += THREADS_PER_BLOCK) {
            const int row = idx / d_stride_uint4;
            const int vec_col = idx % d_stride_uint4;
            uint4 do_val = make_uint4(0, 0, 0, 0);
            if (row < valid_q_rows && vec_col < d_stride_uint4) {
                do_val = __ldg(&do_vec[row * d_stride_uint4 + vec_col]);
            }
            sdO_vec[row * q_stride_uint4 + vec_col] = do_val;
        }
        __syncthreads();

        const __half* current_o_ptr = o_ptr + start_col * D;

        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;
            const int fp16_x4_per_row = D / 4;
            const int work_per_thread = (fp16_x4_per_row + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
            const unsigned mask = (valid_q_rows == BLOCK_N) ? 0xFFFFFFFFU : __activemask();

            float thread_dot = 0.0f;

            #pragma unroll
            for (int j = 0; j < work_per_thread; ++j) {
                const int chunk_idx = thread_in_row + j * THREADS_PER_ROW;
                if (chunk_idx >= fp16_x4_per_row) break;
                const int col = chunk_idx * 4;

                const half* o_addr = current_o_ptr + row * D + col;
                ushort o_h0, o_h1, o_h2, o_h3;
                asm volatile(
                    "ld.global.v4.u16 {%0, %1, %2, %3}, [%4];"
                    : "=h"(o_h0), "=h"(o_h1), "=h"(o_h2), "=h"(o_h3)
                    : "l"(o_addr)
                    : "memory"
                );

                const half* dO_addr = sdO + row * Q_STRIDE + col;
                ushort d_h0, d_h1, d_h2, d_h3;
                const uint32_t ptr_dO = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(dO_addr)));
                asm volatile(
                    "ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
                    : "=h"(d_h0), "=h"(d_h1), "=h"(d_h2), "=h"(d_h3)
                    : "r"(ptr_dO)
                    : "memory"
                );

                const float fo_0 = __half2float(__ushort_as_half(o_h0));
                const float fo_1 = __half2float(__ushort_as_half(o_h1));
                const float fo_2 = __half2float(__ushort_as_half(o_h2));
                const float fo_3 = __half2float(__ushort_as_half(o_h3));

                const float fd_0 = __half2float(__ushort_as_half(d_h0));
                const float fd_1 = __half2float(__ushort_as_half(d_h1));
                const float fd_2 = __half2float(__ushort_as_half(d_h2));
                const float fd_3 = __half2float(__ushort_as_half(d_h3));

                thread_dot = __fmaf_rn(fo_0, fd_0, thread_dot);
                thread_dot = __fmaf_rn(fo_1, fd_1, thread_dot);
                thread_dot = __fmaf_rn(fo_2, fd_2, thread_dot);
                thread_dot = __fmaf_rn(fo_3, fd_3, thread_dot);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_dot += __shfl_down_sync(mask, thread_dot, o, THREADS_PER_ROW);

            if (thread_in_row == 0) { sRowDot[row] = thread_dot; }
        }

        if (tid < valid_q_rows) { sLse[tid] = lse_ptr[start_col + tid]; }
        __syncthreads();

        const int num_tiles_m_dov    = (BLOCK_N + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_dov    = (BLOCK_M + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_dov    = (D + WMMA_K - 1) / WMMA_K;
        const int total_tiles_dov    = num_tiles_m_dov * num_tiles_n_dov;
        const int tiles_per_warp_dov = (total_tiles_dov + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        for (int tile_local = 0; tile_local < tiles_per_warp_dov; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_dov + tile_local;
            if (tile_idx >= total_tiles_dov) break;

            const int tile_m_idx = tile_idx / num_tiles_n_dov;
            const int tile_n_idx = tile_idx % num_tiles_n_dov;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_q_rows || tile_n >= valid_kv_rows) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k_dov; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= D) break;
                load_matrix_sync(a_frag, sdO + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sV + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            store_matrix_sync(sdOV + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        const int total_elements = BLOCK_N * BLOCK_M;
        const int total_pairs = (total_elements + 1) / 2;

        #pragma unroll 2
        for (int i = tid; i < total_pairs; i += THREADS_PER_BLOCK) {
            const int linear_idx0 = i * 2;
            const int linear_idx1 = linear_idx0 + 1;

            const int row0 = linear_idx0 / BLOCK_M;
            const int col0 = linear_idx0 % BLOCK_M;
            const bool has_pair = (linear_idx1 < total_elements);
            const int row1 = has_pair ? linear_idx1 / BLOCK_M : row0;
            const int col1 = has_pair ? linear_idx1 % BLOCK_M : col0 + 1;

            float s0 = 0.0f, s1 = 0.0f;
            float dov0 = 0.0f, dov1 = 0.0f;

            const bool in_bounds0 = (row0 < valid_q_rows) && (col0 < valid_kv_rows);
            const bool causal_ok0 = !IS_CAUSAL || ((start_kv + col0) <= (start_col + row0));
            const bool valid0 = in_bounds0 && causal_ok0;

            bool valid1 = false;
            if (has_pair) {
                const bool in_bounds1 = (row1 < valid_q_rows) && (col1 < valid_kv_rows);
                const bool causal_ok1 = !IS_CAUSAL || ((start_kv + col1) <= (start_col + row1));
                valid1 = in_bounds1 && causal_ok1;
            }

            if (valid0) { s0 = sS[row0 * S_STRIDE + col0]; dov0 = sdOV[row0 * S_STRIDE + col0]; } else { s0 = NEG_INF; }
            if (valid1) { s1 = sS[row1 * S_STRIDE + col1]; dov1 = sdOV[row1 * S_STRIDE + col1]; } else { s1 = NEG_INF; }

            float lse0 = sLse[row0];
            float lse1 = (valid1 && row1 != row0) ? sLse[row1] : lse0;
            float row_dot0 = sRowDot[row0];
            float row_dot1 = (valid1 && row1 != row0) ? sRowDot[row1] : row_dot0;

            float shifted0 = s0 - lse0;
            float shifted1 = s1 - lse1;

            float p0 = (shifted0 < -80.0f) ? 0.0f : __expf(shifted0);
            float p1 = (shifted1 < -80.0f) ? 0.0f : __expf(shifted1);

            float diff0 = dov0 - row_dot0;
            float diff1 = dov1 - row_dot1;
            float ds0 = valid0 ? fmaf(p0, softmax_scale * diff0, 0.0f) : 0.0f;
            float ds1 = valid1 ? fmaf(p1, softmax_scale * diff1, 0.0f) : 0.0f;

            __half2 p_h2 = __float22half2_rn(make_float2(p0, p1));
            __half2 ds_h2 = __float22half2_rn(make_float2(ds0, ds1));

            if (col1 < BLOCK_M && row1 == row0) {
                __half* p_dst  = sP  + row0 * BLOCK_M + col0;
                __half* ds_dst = sdS + row0 * BLOCK_M + col0;
                p_dst[0]  = p_h2.x;  p_dst[1]  = p_h2.y;
                ds_dst[0] = ds_h2.x; ds_dst[1] = ds_h2.y;
            } else {
                if (valid0) { sP[row0 * BLOCK_M + col0] = p_h2.x; sdS[row0 * BLOCK_M + col0] = ds_h2.x; }
                if (valid1) { sP[row1 * BLOCK_M + col1] = p_h2.y; sdS[row1 * BLOCK_M + col1] = ds_h2.y; }
            }
        }
        __syncthreads();

        const int num_tiles_m_dv    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_dv    = (D + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_dv    = (BLOCK_N + WMMA_K - 1) / WMMA_K;
        const int total_tiles_dv    = num_tiles_m_dv * num_tiles_n_dv;
        const int tiles_per_warp_dv = (total_tiles_dv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        for (int tile_local = 0; tile_local < tiles_per_warp_dv; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_dv + tile_local;
            if (tile_idx >= total_tiles_dv) break;

            const int tile_m_idx = tile_idx / num_tiles_n_dv;
            const int tile_n_idx = tile_idx % num_tiles_n_dv;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_kv_rows || tile_n >= D) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            load_matrix_sync(acc_frag, sdV + tile_m * KV_STRIDE + tile_n, KV_STRIDE, mem_row_major);

            #pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k_dv; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= valid_q_rows) break;

                load_matrix_sync(a_frag, sP + k_offset * BLOCK_M + tile_m, BLOCK_M);
                load_matrix_sync(b_frag, sdO + k_offset * Q_STRIDE + tile_n, Q_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            store_matrix_sync(sdV + tile_m * KV_STRIDE + tile_n, acc_frag, KV_STRIDE, mem_row_major);
        }
        __syncthreads();

        __half* sQ = sdO;
        #pragma unroll 2
        for (int idx = tid; idx < NUM_UINT4_Q_BLOCK; idx += THREADS_PER_BLOCK) {
            const int row = idx / d_stride_uint4;
            const int vec_col = idx % d_stride_uint4;
            uint4 q_val = make_uint4(0, 0, 0, 0);
            if (row < valid_q_rows && vec_col < d_stride_uint4) {
                q_val = __ldg(&q_vec[row * d_stride_uint4 + vec_col]);
            }
            sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
        }
        __syncthreads();

        const int num_tiles_m_dk    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_dk    = (D + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_dk    = (BLOCK_N + WMMA_K - 1) / WMMA_K;
        const int total_tiles_dk    = num_tiles_m_dk * num_tiles_n_dk;
        const int tiles_per_warp_dk = (total_tiles_dk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        for (int tile_local = 0; tile_local < tiles_per_warp_dk; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_dk + tile_local;
            if (tile_idx >= total_tiles_dk) break;

            const int tile_m_idx = tile_idx / num_tiles_n_dk;
            const int tile_n_idx = tile_idx % num_tiles_n_dk;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_kv_rows || tile_n >= D) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            load_matrix_sync(acc_frag, sdK + tile_m * KV_STRIDE + tile_n, KV_STRIDE, mem_row_major);

            #pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k_dk; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= valid_q_rows) break;

                load_matrix_sync(a_frag, sdS + k_offset * BLOCK_M + tile_m, BLOCK_M);
                load_matrix_sync(b_frag, sQ + k_offset * Q_STRIDE + tile_n, Q_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            store_matrix_sync(sdK + tile_m * KV_STRIDE + tile_n, acc_frag, KV_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    const int total_fp16_x4 = (valid_kv_rows * D) / 4;
    for (int i = tid; i < total_fp16_x4; i += THREADS_PER_BLOCK) {
        const int row = i / (D / 4);
        const int col = (i % (D / 4)) * 4;

        const float* s_dK_row = sdK + row * KV_STRIDE;
        const float* s_dV_row = sdV + row * KV_STRIDE;

        const __half hk0 = __float2half_rn(s_dK_row[col + 0]);
        const __half hk1 = __float2half_rn(s_dK_row[col + 1]);
        const __half hk2 = __float2half_rn(s_dK_row[col + 2]);
        const __half hk3 = __float2half_rn(s_dK_row[col + 3]);

        const __half hv0 = __float2half_rn(s_dV_row[col + 0]);
        const __half hv1 = __float2half_rn(s_dV_row[col + 1]);
        const __half hv2 = __float2half_rn(s_dV_row[col + 2]);
        const __half hv3 = __float2half_rn(s_dV_row[col + 3]);

        asm volatile(
            "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dK_ptr + row * D + col),
              "h"(__half_as_ushort(hk0)),
              "h"(__half_as_ushort(hk1)),
              "h"(__half_as_ushort(hk2)),
              "h"(__half_as_ushort(hk3))
            : "memory"
        );

        asm volatile(
            "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dV_ptr + row * D + col),
              "h"(__half_as_ushort(hv0)),
              "h"(__half_as_ushort(hv1)),
              "h"(__half_as_ushort(hv2)),
              "h"(__half_as_ushort(hv3))
            : "memory"
        );
    }
}

template<int D>
void launcher_flash_attention_backward_dq(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    torch::Tensor& dO,
    const torch::Tensor& softmax_lse,
    torch::Tensor& dQ,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    using Config = dQKernelConfig<D>;

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int N = K.size(2);

    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for dQ: ", smem, " bytes");

    auto kernel = is_causal ?
        (void*)flash_attention_backward_dq_kernel<D, true> :
        (void*)flash_attention_backward_dq_kernel<D, false>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    if (is_causal) {
        flash_attention_backward_dq_kernel<D, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<__half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dQ.data_ptr()),
            B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_backward_dq_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<__half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dQ.data_ptr()),
            B, H, M, N, softmax_scale
        );
    }
}

template<int D>
void launcher_flash_attention_backward_dkv(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    torch::Tensor& dO,
    const torch::Tensor& softmax_lse,
    torch::Tensor& dK,
    torch::Tensor& dV,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    using Config = dKVKernelConfig<D>;

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int N = K.size(2);

    const int grid_x = (N + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for unified dKV: ", smem, " bytes (", smem / 1024, " KB)");

    auto kernel = is_causal ?
        (void*)flash_attention_backward_dkv_kernel<D, true> :
        (void*)flash_attention_backward_dkv_kernel<D, false>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    if (is_causal) {
        flash_attention_backward_dkv_kernel<D, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<__half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dK.data_ptr()),
            reinterpret_cast<__half*>(dV.data_ptr()),
            B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_backward_dkv_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<__half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dK.data_ptr()),
            reinterpret_cast<__half*>(dV.data_ptr()),
            B, H, M, N, softmax_scale
        );
    }
}

std::vector<at::Tensor> flash_attention_backward(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor>& dq_,
    std::optional<at::Tensor>& dk_,
    std::optional<at::Tensor>& dv_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor>& rng_state
) {

    TORCH_CHECK(!alibi_slopes_.has_value(), "alibi_slopes not supported");
    TORCH_CHECK(p_dropout == 0.f, "dropout not supported");
    TORCH_CHECK(window_size_left == -1, "window_size_left not supported");
    TORCH_CHECK(window_size_right == -1 || (is_causal && window_size_right == 0), "window not supported");
    TORCH_CHECK(softcap == 0.f, "softcap not supported");
    TORCH_CHECK(!deterministic, "deterministic mode not supported");
    TORCH_CHECK(!gen_.has_value(), "Generator not supported");
    TORCH_CHECK(!rng_state.has_value() || rng_state->numel() == 0, "rng_state not supported");

    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "v must be fp16");
    TORCH_CHECK(dout.dtype() == torch::kFloat16, "dout must be fp16");
    TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    const auto sizes = q.sizes();
    const int B = sizes[0], H = sizes[1], M = sizes[2], D = sizes[3];
    const int N = k.size(2);
    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0, "D must be even, <=256, multiple of 8");

    at::Tensor dq_fp16 = dq_.has_value() ? dq_.value() : torch::zeros_like(q);
    at::Tensor dk_fp16 = dk_.has_value() ? dk_.value() : torch::zeros_like(k);
    at::Tensor dv_fp16 = dv_.has_value() ? dv_.value() : torch::zeros_like(v);

    TORCH_CHECK(dq_fp16.dtype() == torch::kFloat16, "dq must be fp16");
    TORCH_CHECK(dk_fp16.dtype() == torch::kFloat16, "dk must be fp16");
    TORCH_CHECK(dv_fp16.dtype() == torch::kFloat16, "dv must be fp16");

    auto dsoftmax_sum = torch::zeros({B, H, M}, torch::dtype(torch::kFloat32).device(q.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props  = at::cuda::getCurrentDeviceProperties();
    bool sm70   = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    switch (D) {
        case 16:
            launcher_flash_attention_backward_dq<16>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<16>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 32:
            launcher_flash_attention_backward_dq<32>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<32>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 64:
            launcher_flash_attention_backward_dq<64>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<64>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 128:
            launcher_flash_attention_backward_dq<128>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<128>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 256:
            launcher_flash_attention_backward_dq<256>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<256>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    return {dq_fp16, dk_fp16, dv_fp16, dsoftmax_sum};
}
