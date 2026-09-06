// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// This adapter instantiates the actual upstream CUDA decode and cascade
// kernels. Upstream:
// flashinfer-ai/flashinfer@6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f. No
// Triton/Flash-V100 fallback is hidden behind this interface.
#include <flashinfer/attention/decode.cuh>

namespace flashinfer::attention::sm70 {

// QSA's ordered selection is a virtual page-size-one cache. Keeping raw
// selection order and duplicates is essential: physical-page sorting changed
// numerical behavior in the previous grouped-page4 implementation.
struct UnitPage {
  __host__ __device__ operator unsigned int() const { return 1; }
  __device__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n;
    r = 0;
  }
};

struct SafeCachePointer {
  const half* data;
  const half* zero;
  __device__ const half* operator+(size_t offset) const {
    // Invalid selections must not read arbitrary cache values: 0 * NaN in
    // the PV loop is still NaN. Use a persistent zero page, including padding.
    return (offset >> 63) ? zero + (offset & 255) : data + offset;
  }
};

struct SparsePagedKV {
  UnitPage page_size;
  uint32_t batch_size, num_heads, width;
  int64_t head_stride;
  SafeCachePointer k_data, v_data;
  const int32_t* indptr;
  const int32_t* rope_pos_offset = nullptr;
  const int64_t* offsets;

  __device__ uint32_t get_length(uint32_t) const { return width; }
  __device__ size_t protective_get_kv_offset(uint32_t index, uint32_t head,
                                             uint32_t, uint32_t d,
                                             int32_t end) const {
    const int64_t offset = index < end ? offsets[index] : -1;
    return offset < 0 ? (size_t{1} << 63)
                      : size_t(offset + head * head_stride + d);
  }
};

struct QSAParams {
  using DTypeQ = half;
  using DTypeKV = half;
  using DTypeO = float;
  using IdType = int32_t;
  const half* q;
  float* o;
  float* lse;
  SparsePagedKV paged_kv;
  const bool* block_valid_mask = nullptr;
  uint32_t padded_batch_size, num_qo_heads;
  bool partition_kv = true;
  const int32_t* request_indices;
  const int32_t* kv_tile_indices;
  const int32_t* kv_chunk_size_ptr;
  uint32_t q_stride_n, q_stride_h;
};

struct QSAVariant {
  static constexpr bool use_softmax = true;
  float sm_scale_log2 = math::log2e / 16.f;
  __device__ QSAVariant(const QSAParams&, uint32_t, uint8_t*) {}
  __device__ float LogitsTransform(const QSAParams&, float value, uint32_t,
                                   uint32_t, uint32_t, uint32_t,
                                   uint32_t) const {
    return value;
  }
  __device__ bool LogitsMask(const QSAParams& p, uint32_t row, uint32_t,
                             uint32_t index, uint32_t, uint32_t) const {
    return index < p.paged_kv.width &&
           p.paged_kv.offsets[size_t(row) * p.paged_kv.width + index] >= 0;
  }
  __device__ float OutputTransform(const QSAParams&, float value, uint32_t,
                                   uint32_t, uint32_t, float, float denominator,
                                   float) const {
    return denominator > 0.f ? value / denominator : 0.f;
  }
};

// Same visible-index contract as qsa_sparse_paged_attention. Causality is
// applied by the QSA selector/expander before this operator, not reinterpreted
// as dense causal attention. Invalid rows/pages/indices produce empty states.
__global__ void PrepareQSA(const int32_t* indices, const int32_t* table,
                           const int32_t* requests, int64_t* offsets,
                           int32_t* metadata, int rows, int selected, int width,
                           int splits, int page, int table_width, int nrequests,
                           int nblocks, int64_t index_stride,
                           int64_t table_stride, int64_t block_stride,
                           int64_t token_stride) {
  const int row = blockIdx.x;
  const int request = requests[row];
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    int64_t offset = -1;
    const int logical = col < selected ? indices[row * index_stride + col] : -1;
    if (request >= 0 && request < nrequests && logical >= 0 &&
        logical / page < table_width) {
      const int physical = table[request * table_stride + logical / page];
      if (physical >= 0 && physical < nblocks)
        offset = physical * block_stride + (logical % page) * token_stride;
    }
    offsets[size_t(row) * width + col] = offset;
  }
  if (threadIdx.x == 0) {
    metadata[row] = row * width;
    if (row == 0) {
      metadata[rows] = rows * width;
      metadata[rows + 1 + 2 * rows * splits] = width / splits;
    }
  }
  for (int split = threadIdx.x; split < splits; split += blockDim.x) {
    metadata[rows + 1 + row * splits + split] = row;
    metadata[rows + 1 + rows * splits + row * splits + split] = split;
  }
}

template <int Group>
void LaunchQSADecode(QSAParams p, half* output, int rows, int splits,
                     cudaStream_t stream) {
  // First port: upstream SIMT/GQA kernel, no tensor-core emulation. On SM70
  // upstream cp_async.cuh uses synchronous vector loads + block barriers.
  constexpr int stages = 2, tile = 1, vec = 8, bdx = 32, bdz = 1;
  constexpr int smem = 2 * stages * tile * Group * 256 * sizeof(half) +
                       tile * Group * bdx * sizeof(size_t);
  BatchDecodeWithPagedKVCacheKernel<PosEncodingMode::kNone, stages, tile, vec,
                                    bdx, Group, bdz, QSAVariant>
      <<<dim3(rows * splits, p.paged_kv.num_heads), dim3(bdx, Group, bdz), smem,
         stream>>>(p);
  // Keep FP32 partials until the final output cast. This is upstream's actual
  // cascade implementation, not the previous Triton merge under a new name.
  MergeStatesKernel<8, float, half>
      <<<rows, dim3(32, p.num_qo_heads), 0, stream>>>(
          p.o, p.lse, output, nullptr, splits, p.num_qo_heads, 256);
}

}  // namespace flashinfer::attention::sm70
