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

template<int D>
struct KernelConfig {
    static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 : (D == 32) ? BLOCK_M_32 : (D == 64) ? BLOCK_M_64 : (D == 128) ? BLOCK_M_128 : BLOCK_M_256;
    static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 : (D == 32) ? BLOCK_N_32 : (D == 64) ? BLOCK_N_64 : (D == 128) ? BLOCK_N_128 : BLOCK_N_256;
    static constexpr int WARPS_PER_BLOCK = (D == 16) ? WARPS_16 : (D == 32) ? WARPS_32 : (D == 64) ? WARPS_64 : (D == 128) ? WARPS_128 : WARPS_256;

    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
    static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int PAD               = (8 - (D % 32) + 32) % 32;
    static constexpr int Q_STRIDE          = D + PAD;
    static constexpr int KV_STRIDE         = D + PAD;
    static constexpr int S_STRIDE          = BLOCK_N + PAD;
    static constexpr int O_STRIDE          = D + PAD;
    static constexpr int PER_UINT4         = 8;

    struct alignas(128) SmemLayout {
        alignas(16) __half q      [BLOCK_M * Q_STRIDE];
    union {
        alignas(16) __half k      [BLOCK_N * KV_STRIDE];
        alignas(16) __half v      [BLOCK_N * KV_STRIDE];
    } reuse_kv;
    union {
        alignas(16) float  s      [BLOCK_M * S_STRIDE];
        alignas(16) __half p      [BLOCK_M * S_STRIDE];
    } reuse_sp;
        alignas(16) float  o      [BLOCK_M * O_STRIDE];
        alignas(16) float  row_max[BLOCK_M];
        alignas(16) float  row_sum[BLOCK_M];
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

template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_forward_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
    const int B,
    const int H,
    const int H_KV,
    const int M,
    const int N,
    const float softmax_scale
) {
    using Config = KernelConfig<D>;
    constexpr int BLOCK_M           = Config::BLOCK_M;
    constexpr int BLOCK_N           = Config::BLOCK_N;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW   = Config::THREADS_PER_ROW;
    constexpr int WARPS_PER_BLOCK   = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE          = Config::Q_STRIDE;
    constexpr int KV_STRIDE         = Config::KV_STRIDE;
    constexpr int S_STRIDE          = Config::S_STRIDE;
    constexpr int O_STRIDE          = Config::O_STRIDE;
    constexpr int PER_UINT4         = Config::PER_UINT4;
    const float NEG_INF = -1e30f;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;
    const int batch_id = batch_head_id / H;
    const int q_head_id = batch_head_id % H;
    const int kv_group_size = H / H_KV;
    const int kv_head_id = q_head_id / kv_group_size;

    const int block_m = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;

    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int valid_q_rows = min(BLOCK_M, M - start_row);
    const int causal_q_offset = max(N - M, 0);

    if constexpr (IS_CAUSAL) {

        const int max_key_pos = causal_q_offset + start_row + valid_q_rows - 1;
        if (max_key_pos < 0) {
            num_n_tiles = 0;
        } else {
            num_n_tiles = min(num_n_tiles, (max_key_pos + BLOCK_N + 0) / BLOCK_N);
        }
    }

    const int tid = threadIdx.x;
    const int warp_id = tid / MAX_THREADS_PER_WARP;
    const int lane_id = tid % MAX_THREADS_PER_WARP;

    const size_t q_head_linear = (size_t)batch_id * H + q_head_id;
    const size_t kv_head_linear = (size_t)batch_id * H_KV + kv_head_id;
    const __half* q_ptr = Q + q_head_linear * M * D + start_row * D;
    const __half* k_ptr = K + kv_head_linear * N * D;
    const __half* v_ptr = V + kv_head_linear * N * D;
    __half* out_ptr = Out + q_head_linear * M * D + start_row * D;
    float* softmax_lse_ptr = softmax_lse + q_head_linear * M + start_row;

    extern __shared__ char smem_raw[];
    init_smem<Config>(smem_raw);
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* sQ      = smem.q;
    __half* sK      = smem.reuse_kv.k;
    __half* sV      = smem.reuse_kv.v;
    float*  sS      = smem.reuse_sp.s;
    __half* sP      = smem.reuse_sp.p;
    float*  sO      = smem.o;
    float*  sRowMax = smem.row_max;
    float*  sRowSum = smem.row_sum;

    const int  d_stride_uint4 = (D + PER_UINT4 - 1) / PER_UINT4;
    const int  q_stride_uint4 = (Q_STRIDE  + PER_UINT4 - 1) / PER_UINT4;
    const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

    if (tid < BLOCK_M) {
        sRowMax[tid] = NEG_INF;
    }

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
        sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
    }
    __syncthreads();

    for (int block_n = 0; block_n < num_n_tiles; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= N) break;
        const int valid_k_rows = min(BLOCK_N, N - start_col);

        if constexpr (IS_CAUSAL) {
            if (start_col >= causal_q_offset + start_row + valid_q_rows) { continue; }
        }

        const uint4* k_vec     = reinterpret_cast<const uint4*>(k_ptr + start_col * D);
        uint4*       sK_vec    = reinterpret_cast<uint4*>(sK);

        #pragma unroll 2
        for (int idx = tid; idx < (valid_k_rows * d_stride_uint4); idx += THREADS_PER_BLOCK) {
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

                    const int global_m = start_row + tile_m + row;
                    const int global_n = start_col + tile_n + col;
                    const int global_q_pos = global_m + causal_q_offset;

                    const bool is_valid = (global_m < start_row + valid_q_rows) &&
                                          (global_n < start_col + valid_k_rows);

                    acc_frag.x[i] = is_valid
                        ? ((global_n > global_q_pos) ? NEG_INF : acc_frag.x[i] * softmax_scale)
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
            const unsigned mask = (valid_q_rows == BLOCK_M) ? 0xFFFFFFFFU : __activemask();
            const int row_leader = __ffs(mask) - 1;

            float*  sS_row_f = sS + row * S_STRIDE;
            __half* sP_row_h = sP + row * S_STRIDE;

            const int vec_cols = valid_k_rows >> 2;
            const int vecs_per_thread = (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
            const int tail_start = vec_cols << 2;

            float thread_max = NEG_INF;
            float4* sS_vec4 = reinterpret_cast<float4*>(sS_row_f);

            #pragma unroll 4
            for (int j = 0; j < vecs_per_thread; ++j) {
                int vc = thread_in_row + j * THREADS_PER_ROW;
                if (vc < vec_cols) {
                    float4 v4 = sS_vec4[vc];
                    thread_max = fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
                }
            }

            #pragma unroll 4
            for (int c = tail_start + thread_in_row; c < valid_k_rows; c += THREADS_PER_ROW) {
                thread_max = fmaxf(thread_max, sS_row_f[c]);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, THREADS_PER_ROW));

            const float row_max  = __shfl_sync(mask, thread_max, row_leader, THREADS_PER_ROW);
            const float old_max  = sRowMax[row];
            const float new_max  = fmaxf(old_max, row_max);
            const float exp_diff = __expf(old_max - new_max);

            float thread_sum = 0.0f;
            __half2 half_buffer[20];
            int vc_base = thread_in_row;
            int h2_idx = 0;
            int tail_col = -1;
            __half tail_value = __float2half(0.f);

            #pragma unroll 4
            for (int j = 0; j < vecs_per_thread; ++j, vc_base += THREADS_PER_ROW) {
                if (vc_base < vec_cols) {
                    float4 v4 = sS_vec4[vc_base];

                    float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
                    float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
                    float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
                    float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));

                    thread_sum += (e0 + e1) + (e2 + e3);

                    half_buffer[h2_idx++] = __float22half2_rn(make_float2(e0, e1));
                    half_buffer[h2_idx++] = __float22half2_rn(make_float2(e2, e3));
                }
            }

            #pragma unroll 4
            for (int c = tail_start + thread_in_row; c < valid_k_rows; c += THREADS_PER_ROW) {
                float v = sS_row_f[c];
                float e = __expf(fmaxf(v - new_max, -80.0f));
                thread_sum += e;
                tail_col = c;
                tail_value = __float2half_rn(e);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_sum += __shfl_down_sync(mask, thread_sum, o, THREADS_PER_ROW);

            float row_sum = __shfl_sync(mask, thread_sum, row_leader, THREADS_PER_ROW);

            if (thread_in_row == 0) {
                sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
                sRowMax[row] = new_max;
            }

            h2_idx = 0;
            vc_base = thread_in_row;
            __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);

            #pragma unroll 4
            for (int j = 0; j < vecs_per_thread; ++j, vc_base += THREADS_PER_ROW) {
                if (vc_base < vec_cols) {
                    int base_offset = vc_base * 2;

                    sP_half2[base_offset]     = half_buffer[h2_idx++];
                    sP_half2[base_offset + 1] = half_buffer[h2_idx++];
                }
            }

            if (tail_col >= 0) {
                sP_row_h[tail_col] = tail_value;
            }

            #pragma unroll 4
            for (int c = tail_start + thread_in_row; c < BLOCK_N; c += THREADS_PER_ROW) {
                if (c >= valid_k_rows) {
                    sP_row_h[c] = __float2half(0.f);
                }
            }

            if (block_n > 0) {
                float*  sO_row = sO + row * O_STRIDE;
                float4* sO_vec = reinterpret_cast<float4*>(sO_row);
                const int o_vec_count = (O_STRIDE + 3) >> 2;
                float scale = exp_diff;

                #pragma unroll 4
                for (int ov = thread_in_row; ov < o_vec_count; ov += THREADS_PER_ROW) {
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

        const uint4* v_vec     = reinterpret_cast<const uint4*>(v_ptr + start_col * D);
        uint4*       sV_vec    = reinterpret_cast<uint4*>(sV);

        #pragma unroll 2
        for (int idx = tid; idx < (valid_k_rows * d_stride_uint4); idx += THREADS_PER_BLOCK) {
            const int row = idx / d_stride_uint4;
            const int vec_col = idx % d_stride_uint4;

            uint4 v_val = make_uint4(0, 0, 0, 0);
            if (row < valid_k_rows && vec_col < d_stride_uint4) {
                v_val = __ldg(&v_vec[row * d_stride_uint4 + vec_col]);
            }
            sV_vec[row * kv_stride_uint4 + vec_col] = v_val;
        }
        __syncthreads();

        const int num_tiles_m_pv    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_pv    = (D + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_pv    = (BLOCK_N + WMMA_K - 1) / WMMA_K;
        const int total_tiles_pv    = num_tiles_m_pv * num_tiles_n_pv;
        const int tiles_per_warp_pv = (total_tiles_pv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

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

            load_matrix_sync(acc_frag, sO + tile_m * O_STRIDE + tile_d, O_STRIDE, mem_row_major);

            #pragma unroll
            for (int tile_k = 0; tile_k < num_tiles_k_pv; ++tile_k) {
                const int k_offset = tile_k * WMMA_K;
                if (k_offset >= valid_k_rows) break;

                load_matrix_sync(a_frag, sP + tile_m * S_STRIDE + k_offset, S_STRIDE);
                load_matrix_sync(b_frag, sV + k_offset * KV_STRIDE + tile_d, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            store_matrix_sync(sO + tile_m * O_STRIDE + tile_d, acc_frag, O_STRIDE, mem_row_major);
        }
        __syncthreads();
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

template<int D>
void launcher_flash_attention_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int H_KV = K.size(1);
    const int M = Q.size(2);
    const int N = K.size(2);

    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB: ", smem, " bytes");

    auto kernel = is_causal ?
        (void*)flash_attention_forward_kernel<D, true> :
        (void*)flash_attention_forward_kernel<D, false>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    if (is_causal) {
        flash_attention_forward_kernel<D, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            B, H, H_KV, M, N, softmax_scale
        );
    } else {
        flash_attention_forward_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            B, H, H_KV, M, N, softmax_scale
        );
    }
}

std::vector<at::Tensor> flash_attention_forward(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_
) {

    TORCH_CHECK(!alibi_slopes_.has_value(), "alibi_slopes not supported");
    TORCH_CHECK(p_dropout == 0.f, "dropout not supported");
    TORCH_CHECK(window_size_left == -1, "window_size_left not supported");
    TORCH_CHECK(window_size_right == -1 || (is_causal && window_size_right == 0), "window not supported");
    TORCH_CHECK(softcap == 0.f, "softcap not supported");
    TORCH_CHECK(!return_softmax, "return_softmax not supported");
    TORCH_CHECK(!gen_.has_value(), "Generator not supported");

    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "v must be fp16");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Last dim must be contiguous");

    const auto sizes = q.sizes();
    const int B = sizes[0], H = sizes[1], M = sizes[2], D = sizes[3];
    const int H_KV = k.size(1);
    const int N = k.size(2);
    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0, "D must be even, <=256, multiple of 8");
    TORCH_CHECK(H_KV > 0, "num_kv_heads must be positive");
    TORCH_CHECK(H % H_KV == 0, "num_attention_heads must be divisible by num_kv_heads");
    TORCH_CHECK(k.size(0) == B && v.size(0) == B, "K/V batch size must match Q");
    TORCH_CHECK(v.size(1) == H_KV, "K/V num_heads must match");
    TORCH_CHECK(v.size(2) == N, "K/V sequence length must match");
    TORCH_CHECK(k.size(3) == D && v.size(3) == D, "K/V head_dim must match Q");

    at::Tensor out_fp16 = out_.has_value() ? out_.value() : torch::zeros_like(q);
    TORCH_CHECK(out_fp16.dtype() == torch::kFloat16, "out must be fp16");
    auto softmax_lse = torch::zeros({B, H, M}, torch::dtype(torch::kFloat32).device(q.device()));
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props  = at::cuda::getCurrentDeviceProperties();
    bool sm70   = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    switch (D) {
        case 16:  launcher_flash_attention_forward<16>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        case 32:  launcher_flash_attention_forward<32>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        case 64:  launcher_flash_attention_forward<64>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        case 128: launcher_flash_attention_forward<128>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        case 256: launcher_flash_attention_forward<256>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }

    auto p = torch::zeros({0}, q.options());
    auto rng_state = torch::zeros({2}, torch::dtype(torch::kInt64).device(q.device()));
    return {out_fp16, softmax_lse, p, rng_state};
}
