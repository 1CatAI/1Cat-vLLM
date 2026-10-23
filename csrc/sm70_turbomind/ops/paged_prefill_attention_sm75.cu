// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Causal prefill attention over a paged fp16 KV cache for Turing (SM75 and
// later), head dim 256. FlashAttention-2's Turing build runs this shape at
// about a quarter of the m16n8k8 rate: its paged (split-KV) kernel keeps Q in
// shared memory next to K and V, so a 64-row CTA fills the SM's 64 KB, and
// Turing, without cp.async, waits for every tile's loads.
//
// Long prefills stream the whole KV cache once per CTA, which makes the
// kernel DRAM-bound unless many query rows share each loaded tile. Here a CTA
// holds 128 packed query rows in registers (eight warps of 16), so each key is
// read once per 128 rows, and double-buffers K/V tiles of 32 keys in shared
// memory: while a tile computes, each thread fetches its 4 K and 4 V chunks
// of the next one in two halves, each issued one compute phase before it is
// stored (all four at once cost register spills and measured slower).
//
// The query heads of one KV head are packed into the rows of a CTA (row r is
// token r / group, head kv_head * group + r % group), so every tile serves
// all of them. Online softmax as in FlashAttention-2: fp32 scores and
// accumulators, probabilities rounded to fp16 for the P.V product.
//
// One CTA fills an SM (64 KB of shared memory), so a grid of a few waves
// loses up to a whole wave to its last one: 150 CTAs on 72 SMs take three.
// Splitting each CTA's keys into ranges (split-KV) evens the waves out; the
// splits write unnormalized fp32 partial outputs with their row maxima and
// sums, and a second kernel merges them by log-sum-exp.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cmath>

#include "mma_fragments_sm70.cuh"

namespace {

using vllm::sm70::ldmatrix_x4;
using vllm::sm70::ldmatrix_x4_trans;
using vllm::sm70::mma_m16n8k8;
using vllm::sm70::shared_address;

constexpr int kHeadDim = 256;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kBlockM = kWarps * 16;   // packed query rows per CTA
constexpr int kBlockN = 32;            // keys per tile
constexpr int kChunks = kHeadDim / 8;  // 16-byte chunks per row
constexpr int kTileChunks = kBlockN * kChunks;
constexpr int kStageChunks = 2 * kTileChunks;      // K tile then V tile
constexpr int kSmemBytes = 2 * kStageChunks * 16;  // two stages; Q at start
static_assert(kBlockM * kChunks * 16 <= kSmemBytes, "Q stage fits the tiles");
// Each thread moves kSlices K chunks and as many V chunks per tile.
constexpr int kSlices = kTileChunks / kThreads;
static_assert(kSlices == 4, "the compute phases below interleave four slices");

// 512-byte rows: XOR the chunk with the row's low three bits so the eight rows
// an ldmatrix phase reads land in distinct bank groups.
__device__ __forceinline__ int swizzled(int row, int chunk) {
  return row * kChunks + (chunk ^ (row & 7));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  const half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// Offsets are 32-bit (the launcher checks the cache size): the 64-bit
// address arithmetic of the fetch otherwise costs the registers the MMA loop
// needs.
// The page size is a template constant: a runtime divisor makes every slice
// load pay an integer division.
template <int kPageSize>
struct TileFetch {
  const half* k_cache;
  const half* v_cache;
  const int* pages;
  int block_stride, slot_stride, head_offset;
  int seqlen_k;
  int row, chunk;  // this thread's first tile row and its chunk

  // Slice s of the tile at key j0: one K and one V chunk of row row + 8 s.
  __device__ __forceinline__ void load(int j0, int s, uint4& k,
                                       uint4& v) const {
    const int key = j0 + row + 8 * s;
    k = make_uint4(0, 0, 0, 0);
    v = make_uint4(0, 0, 0, 0);
    if (key < seqlen_k) {
      const int offset = pages[key / kPageSize] * block_stride +
                         (key % kPageSize) * slot_stride + head_offset +
                         chunk * 8;
      k = *reinterpret_cast<const uint4*>(k_cache + offset);
      v = *reinterpret_cast<const uint4*>(v_cache + offset);
    }
  }

  __device__ __forceinline__ void store(uint4* stage, int s, const uint4& k,
                                        const uint4& v) const {
    const int r = row + 8 * s;
    stage[swizzled(r, chunk)] = k;
    stage[kTileChunks + swizzled(r, chunk)] = v;
  }
};

// Split-KV partials: unnormalized outputs [split][token][head][256] and the
// row maximum (log2 units) and sum [split][token][head][2].
struct Partials {
  float* o;
  float2* ml;
  int heads;
  int64_t split_stride;  // tokens * heads
};

template <int kPageSize>
__global__ void __launch_bounds__(kThreads, 1) paged_prefill_attention_kernel(
    const half* __restrict__ q, const half* __restrict__ k_cache,
    const half* __restrict__ v_cache, half* __restrict__ out,
    const int* __restrict__ cu_seqlens_q, const int* __restrict__ seqused_k,
    const int* __restrict__ block_table, int block_table_stride,
    int64_t q_token_stride, int64_t q_head_stride, int64_t out_token_stride,
    int64_t out_head_stride, int cache_block_stride, int cache_slot_stride,
    int cache_head_stride, int group, float scale_log2, int num_splits,
    Partials partials) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* stages = reinterpret_cast<uint4*>(smem);

  const int batch = blockIdx.z / num_splits, split = blockIdx.z % num_splits;
  const int kv_head = blockIdx.y;
  const int q_begin = cu_seqlens_q[batch];
  const int q_len = cu_seqlens_q[batch + 1] - q_begin;
  const int rows = q_len * group;
  const int row0 = blockIdx.x * kBlockM;
  if (row0 >= rows) return;  // block-uniform: shorter sequence in the batch
  const int seqlen_k = seqused_k[batch];
  // Causal alignment to the bottom right: query token i sits at key position
  // seqlen_k - q_len + i.
  const int context = seqlen_k - q_len;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int quad = lane & 3;
  const TileFetch<kPageSize> fetch{
      k_cache,
      v_cache,
      block_table + (int64_t)batch * block_table_stride,
      cache_block_stride,
      cache_slot_stride,
      kv_head * cache_head_stride,
      seqlen_k,
      tid / kChunks,
      tid % kChunks};

  // Q tile -> shared memory -> A fragments (rows 16 * warp .. + 15).
  for (int i = tid; i < kBlockM * kChunks; i += kThreads) {
    const int r = i / kChunks, c = i % kChunks;
    const int row = row0 + r;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < rows) {
      const int token = row / group, head = kv_head * group + row % group;
      v = *reinterpret_cast<const uint4*>(
          q + (int64_t)(q_begin + token) * q_token_stride +
          (int64_t)head * q_head_stride + c * 8);
    }
    stages[swizzled(r, c)] = v;
  }
  __syncthreads();
  // ldmatrix x4: matrices (rows 0-7, chunk c), (rows 8-15, chunk c),
  // (rows 0-7, chunk c + 1), (rows 8-15, chunk c + 1) = a0/a1 of two k-steps.
  uint32_t qa[kChunks][2];
  {
    const int r = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int dc = lane >> 4;
#pragma unroll
    for (int ks = 0; ks < kChunks; ks += 2) {
      uint32_t f[4];
      ldmatrix_x4(f, shared_address(&stages[swizzled(r, ks + dc)]));
      qa[ks][0] = f[0];
      qa[ks][1] = f[1];
      qa[ks + 1][0] = f[2];
      qa[ks + 1][1] = f[3];
    }
  }

  // This lane's two rows: lane / 4 and lane / 4 + 8 of its warp's 16.
  int qpos[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = row0 + warp * 16 + (lane >> 2) + 8 * h;
    qpos[h] = row < rows ? context + row / group : -1;  // -1: padding row
  }
  const int last_row = min(row0 + kBlockM, rows) - 1;
  const int kv_end = min(seqlen_k, context + last_row / group + 1);
  // Keys up to the block's first query position need no causal mask.
  const int unmasked_end = context + row0 / group + 1;
  // This split's whole key tiles of the CTA's range [0, kv_end).
  const int tiles = (kv_end + kBlockN - 1) / kBlockN;
  const int split_tiles = (tiles + num_splits - 1) / num_splits;
  const int k_begin = split * split_tiles * kBlockN;
  const int k_end = min(kv_end, k_begin + split_tiles * kBlockN);

  __syncthreads();  // Q fragments read before tile 0 overwrites the stage
#pragma unroll
  for (int s = 0; s < kSlices; s++) {
    uint4 k, v;
    fetch.load(k_begin, s, k, v);
    fetch.store(stages, s, k, v);
  }
  __syncthreads();

  float o[kChunks][4];
#pragma unroll
  for (int dn = 0; dn < kChunks; dn++)
#pragma unroll
    for (int i = 0; i < 4; i++) o[dn][i] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

  // ldmatrix lane addressing shared by all tiles: S takes two key tiles x two
  // k-steps per x4, P.V one key tile x four d tiles.
  const int s_row = ((lane >> 4) * 8) + (lane & 7), s_chunk = (lane >> 3) & 1;
  const int v_row = lane & 7, v_chunk = lane >> 3;

  for (int j0 = k_begin, stage = 0; j0 < k_end; j0 += kBlockN, stage ^= 1) {
    const uint4* k_tile = stages + stage * kStageChunks;
    const uint4* v_tile = k_tile + kTileChunks;
    uint4* next = stages + (stage ^ 1) * kStageChunks;
    const bool prefetch = j0 + kBlockN < k_end;
    uint4 pk[2], pv[2];
    if (prefetch) {
#pragma unroll
      for (int i = 0; i < 2; i++) fetch.load(j0 + kBlockN, i, pk[i], pv[i]);
    }

    // S = Q K^T: 16 rows x 32 keys per warp, the four key tiles interleaved
    // so four accumulator chains are in flight. Matrix 2 i + kk of the x4 for
    // key-tile pair p is key tile 2 p + i at k-step ks + kk.
    float s[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int i = 0; i < 4; i++) s[nt][i] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kChunks; ks += 2) {
      uint32_t f[2][4];
#pragma unroll
      for (int p = 0; p < 2; p++)
        ldmatrix_x4(f[p], shared_address(
                              &k_tile[swizzled(p * 16 + s_row, ks + s_chunk)]));
#pragma unroll
      for (int kk = 0; kk < 2; kk++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
          mma_m16n8k8(s[nt], qa[ks + kk][0], qa[ks + kk][1],
                      f[nt >> 1][(nt & 1) * 2 + kk]);
    }

    if (prefetch) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        fetch.store(next, i, pk[i], pv[i]);
        fetch.load(j0 + kBlockN, 2 + i, pk[i], pv[i]);
      }
    }

    // Online softmax over the tile. s[nt][i]: row (i < 2 ? lo : hi), key
    // j0 + 8 * nt + 2 * quad + (i & 1).
    const bool masked = j0 + kBlockN > unmasked_end || j0 + kBlockN > seqlen_k;
    float tile_max[2] = {m_run[0], m_run[1]};
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float v = s[nt][i] * scale_log2;
        if (masked) {
          const int key = j0 + nt * 8 + 2 * quad + (i & 1);
          if (key > qpos[i >> 1] || key >= seqlen_k) v = -INFINITY;
        }
        s[nt][i] = v;
        tile_max[i >> 1] = fmaxf(tile_max[i >> 1], v);
      }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      tile_max[h] =
          fmaxf(tile_max[h], __shfl_xor_sync(0xffffffff, tile_max[h], 1));
      tile_max[h] =
          fmaxf(tile_max[h], __shfl_xor_sync(0xffffffff, tile_max[h], 2));
    }
    float base[2], alpha[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      // A row without any visible key yet (padding rows) keeps m = -inf;
      // subtracting 0 then turns its scores into exp2(-inf) = 0.
      base[h] = tile_max[h] == -INFINITY ? 0.f : tile_max[h];
      alpha[h] = exp2f(m_run[h] - base[h]);
      m_run[h] = tile_max[h];
      l_run[h] *= alpha[h];
    }
    // Over a long context the row maxima settle after a few tiles; skipping
    // the rescale while every factor of the warp is exactly 1 changes no bit.
    if (__any_sync(0xffffffff, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
      for (int dn = 0; dn < kChunks; dn++) {
        o[dn][0] *= alpha[0];
        o[dn][1] *= alpha[0];
        o[dn][2] *= alpha[1];
        o[dn][3] *= alpha[1];
      }
    }
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        s[nt][i] = exp2f(s[nt][i] - base[i >> 1]);
        l_run[i >> 1] += s[nt][i];
      }

    // O += P V. The m16n8 accumulator of key tile st is the A fragment of
    // k-step st; matrix m of a transposing ldmatrix x4 is keys 8 * st .. + 7
    // at chunk dn + m, the B fragment of n-tile dn + m.
#pragma unroll
    for (int st = 0; st < 4; st++) {
      const uint32_t a0 = pack_half2(s[st][0], s[st][1]);
      const uint32_t a1 = pack_half2(s[st][2], s[st][3]);
#pragma unroll
      for (int dn = 0; dn < kChunks; dn += 4) {
        uint32_t f[4];
        ldmatrix_x4_trans(
            f, shared_address(&v_tile[swizzled(st * 8 + v_row, dn + v_chunk)]));
#pragma unroll
        for (int m = 0; m < 4; m++) mma_m16n8k8(o[dn + m], a0, a1, f[m]);
      }
    }
    if (prefetch) {
#pragma unroll
      for (int i = 0; i < 2; i++) fetch.store(next, 2 + i, pk[i], pv[i]);
    }
    __syncthreads();  // next stage complete, this one consumed
  }

#pragma unroll
  for (int h = 0; h < 2; h++) {
    l_run[h] += __shfl_xor_sync(0xffffffff, l_run[h], 1);
    l_run[h] += __shfl_xor_sync(0xffffffff, l_run[h], 2);
    if (qpos[h] >= 0) {
      const int row = row0 + warp * 16 + (lane >> 2) + 8 * h;
      const int token = row / group, head = kv_head * group + row % group;
      if (num_splits == 1) {
        half* dst = out + (int64_t)(q_begin + token) * out_token_stride +
                    (int64_t)head * out_head_stride + 2 * quad;
        const float inv = 1.f / l_run[h];
#pragma unroll
        for (int dn = 0; dn < kChunks; dn++)
          *reinterpret_cast<half2*>(dst + 8 * dn) =
              __floats2half2_rn(o[dn][2 * h] * inv, o[dn][2 * h + 1] * inv);
      } else {
        // A split without visible keys leaves m = -inf, l = 0 and o = 0.
        const int64_t index = split * partials.split_stride +
                              (int64_t)(q_begin + token) * partials.heads +
                              head;
        float* dst = partials.o + index * kHeadDim + 2 * quad;
#pragma unroll
        for (int dn = 0; dn < kChunks; dn++)
          *reinterpret_cast<float2*>(dst + 8 * dn) =
              make_float2(o[dn][2 * h], o[dn][2 * h + 1]);
        if (quad == 0) partials.ml[index] = make_float2(m_run[h], l_run[h]);
      }
    }
  }
}

// out[token][head] = sum_s 2^(m_s - M) o_s / sum_s 2^(m_s - M) l_s, M the
// largest m_s; split 0 always holds key 0, so M is finite.
__global__ void __launch_bounds__(64)
    merge_splits_kernel(const float4* __restrict__ o,
                        const float2* __restrict__ ml, half* __restrict__ out,
                        int heads, int64_t split_stride, int num_splits,
                        int64_t out_token_stride, int64_t out_head_stride) {
  const int64_t index = blockIdx.x;  // token * heads + head
  const int token = static_cast<int>(index / heads);
  const int head = static_cast<int>(index % heads);
  float m_max = -INFINITY;
  for (int s = 0; s < num_splits; s++)
    m_max = fmaxf(m_max, ml[s * split_stride + index].x);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float l_sum = 0.f;
  for (int s = 0; s < num_splits; s++) {
    const float2 m_l = ml[s * split_stride + index];
    const float w = exp2f(m_l.x - m_max);  // exp2(-inf) = 0 for empty splits
    l_sum += w * m_l.y;
    const float4 v =
        o[(s * split_stride + index) * (kHeadDim / 4) + threadIdx.x];
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  const float inv = 1.f / l_sum;
  half2* dst = reinterpret_cast<half2*>(out + token * out_token_stride +
                                        head * out_head_stride) +
               2 * threadIdx.x;
  dst[0] = __floats2half2_rn(acc.x * inv, acc.y * inv);
  dst[1] = __floats2half2_rn(acc.z * inv, acc.w * inv);
}

constexpr int kMaxSplits = 4;

// The split count that finishes the CTAs in the fewest full-CTA times: with
// s splits a CTA takes 1/s of the time, and ceil(ctas * s / slots) waves run.
int splits_by_waves(int64_t ctas, int64_t slots) {
  int best = 1;
  double best_time = static_cast<double>((ctas + slots - 1) / slots);
  for (int s = 2; s <= kMaxSplits; s++) {
    const double time = static_cast<double>((ctas * s + slots - 1) / slots) / s;
    if (time < best_time) {
      best = s;
      best_time = time;
    }
  }
  return best;
}

}  // namespace

void paged_prefill_attention_sm75(torch::Tensor out, torch::Tensor q,
                                  torch::Tensor k_cache, torch::Tensor v_cache,
                                  torch::Tensor cu_seqlens_q,
                                  torch::Tensor seqused_k,
                                  torch::Tensor block_table,
                                  int64_t max_seqlen_q, double softmax_scale,
                                  int64_t num_kv_splits) {
  const c10::cuda::CUDAGuard device_guard(q.device());
  const cudaDeviceProp* props = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(props->major * 10 + props->minor >= 75,
              "paged prefill attention needs SM75 or later");
  for (const auto& t : {out, q, k_cache, v_cache})
    TORCH_CHECK(
        t.is_cuda() && t.dtype() == torch::kHalf && t.device() == q.device(),
        "fp16 CUDA tensors on one device expected");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kHeadDim && q.stride(2) == 1,
              "q must be [tokens, heads, 256] with contiguous rows");
  TORCH_CHECK(out.sizes() == q.sizes() && out.stride(2) == 1,
              "out must match q");
  TORCH_CHECK(
      k_cache.dim() == 4 && k_cache.size(3) == kHeadDim &&
          k_cache.stride(3) == 1 && v_cache.sizes() == k_cache.sizes() &&
          v_cache.strides() == k_cache.strides(),
      "caches must be [blocks, page, kv_heads, 256] with equal strides");
  const int64_t heads = q.size(1), kv_heads = k_cache.size(2);
  TORCH_CHECK(heads % kv_heads == 0, "query heads must group over KV heads");
  for (const auto& t : {cu_seqlens_q, seqused_k, block_table})
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kInt && t.is_contiguous(),
                "int32 metadata expected");
  TORCH_CHECK(block_table.dim() == 2, "block table must be [batch, pages]");
  const int64_t batch = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(seqused_k.size(0) == batch && block_table.size(0) == batch);
  // The K/V views may interleave (vLLM's [blocks, 2, page, heads, dim]
  // cache), so bound the largest offset the kernel forms, not numel().
  int64_t max_offset = k_cache.size(3);
  for (int d = 0; d < 3; d++)
    max_offset += (k_cache.size(d) - 1) * k_cache.stride(d);
  TORCH_CHECK(
      max_offset <= INT32_MAX,
      "32-bit cache offsets: the cache view spans 2^31 elements or more");
  TORCH_CHECK(num_kv_splits >= 0 && num_kv_splits <= kMaxSplits,
              "num_kv_splits must be 0 (by waves) to ", kMaxSplits);
  const int64_t group = heads / kv_heads;
  const int64_t ctas =
      (max_seqlen_q * group + kBlockM - 1) / kBlockM * kv_heads * batch;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int num_splits = 1;
  Partials partials{nullptr, nullptr, (int)heads, q.size(0) * heads};
  torch::Tensor partial_o, partial_ml;
  // Both stages take Turing's whole 64 KB of shared memory.
  auto launch = [&](auto kernel) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    int occupancy = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, kThreads, kSmemBytes));
    num_splits = num_kv_splits > 0
                     ? static_cast<int>(num_kv_splits)
                     : splits_by_waves(ctas, props->multiProcessorCount *
                                                 (int64_t)occupancy);
    if (num_splits > 1) {
      partial_o = torch::empty({num_splits, q.size(0), heads, kHeadDim},
                               q.options().dtype(torch::kFloat));
      partial_ml = torch::empty({num_splits, q.size(0), heads, 2},
                                q.options().dtype(torch::kFloat));
      partials.o = partial_o.data_ptr<float>();
      partials.ml = reinterpret_cast<float2*>(partial_ml.data_ptr<float>());
    }
    const dim3 grid((unsigned)((max_seqlen_q * group + kBlockM - 1) / kBlockM),
                    (unsigned)kv_heads, (unsigned)(batch * num_splits));
    kernel<<<grid, kThreads, kSmemBytes, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        cu_seqlens_q.data_ptr<int>(), seqused_k.data_ptr<int>(),
        block_table.data_ptr<int>(), (int)block_table.stride(0), q.stride(0),
        q.stride(1), out.stride(0), out.stride(1), (int)k_cache.stride(0),
        (int)k_cache.stride(1), (int)k_cache.stride(2), (int)group,
        (float)(softmax_scale * M_LOG2E), num_splits, partials);
  };
  switch (k_cache.size(1)) {
    case 16:
      launch(paged_prefill_attention_kernel<16>);
      break;
    case 32:
      launch(paged_prefill_attention_kernel<32>);
      break;
    case 64:
      launch(paged_prefill_attention_kernel<64>);
      break;
    default:
      TORCH_CHECK(false,
                  "paged prefill attention supports pages of 16, 32 "
                  "or 64 tokens, got ",
                  k_cache.size(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (num_splits > 1 && q.size(0) > 0) {
    merge_splits_kernel<<<(unsigned)(q.size(0) * heads), kHeadDim / 4, 0,
                          stream>>>(
        reinterpret_cast<const float4*>(partials.o), partials.ml,
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), (int)heads,
        partials.split_stride, num_splits, out.stride(0), out.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
