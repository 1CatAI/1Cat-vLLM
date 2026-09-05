// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_
#define VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace vllm::qsa {

constexpr int kLexicographicTopKThreads = 1024;
constexpr int kLexicographicTopKBins = 256;
constexpr int kLexicographicTopKDecodeCandidateCapacity = 2304;

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  // IEEE -0.0 and +0.0 compare equal, so keep them in the same score bucket
  // and let the block index provide the deterministic tie break.
  if (value == 0.0f) value = 0.0f;
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int TopK>
struct LexicographicTopKShared {
  using BlockScan = cub::BlockScan<uint64_t, kLexicographicTopKThreads>;

  uint32_t histogram[kLexicographicTopKBins];
  typename BlockScan::TempStorage scan;
  uint32_t prefix;
  uint32_t pivot;
  uint32_t remaining;
  uint32_t greater_seen;
  uint32_t equal_seen;
  uint32_t chunk_greater_base;
  uint32_t chunk_equal_base;
};

template <int TopK>
struct LexicographicDecodeTopKShared {
  using BlockScan = cub::BlockScan<uint64_t, kLexicographicTopKThreads>;

  // The decode fast path first selects one coarse radix bucket, then scans
  // only that bucket for the remaining bytes. Keep two buffers so compaction
  // never overwrites input indices that another warp has not consumed yet.
  uint32_t histogram[2][kLexicographicTopKBins + 128];
  int32_t candidates[2][kLexicographicTopKDecodeCandidateCapacity];
  typename BlockScan::TempStorage scan;
  uint32_t prefix;
  uint32_t pivot;
  uint32_t remaining;
  uint32_t remaining_ties;
  uint32_t candidate_count[2];
  uint32_t threshold_bin;
  uint32_t greater_seen;
  uint32_t equal_seen;
  uint32_t chunk_greater_base;
  uint32_t chunk_equal_base;
};

template <int TopK>
__device__ __forceinline__ void decode_suffix_scan_histogram(
    LexicographicDecodeTopKShared<TopK>& shared) {
#pragma unroll
  for (int pass = 0; pass < 8; ++pass) {
    const int distance = 1 << pass;
    const int source = pass & 1;
    if (threadIdx.x < kLexicographicTopKBins) {
      uint32_t value = shared.histogram[source][threadIdx.x];
      if (threadIdx.x + distance < kLexicographicTopKBins) {
        value += shared.histogram[source][threadIdx.x + distance];
      }
      shared.histogram[source ^ 1][threadIdx.x] = value;
    }
    __syncthreads();
  }
}

template <int TopK>
__device__ __forceinline__ void decode_choose_threshold(
    LexicographicDecodeTopKShared<TopK>& shared, int shift) {
  if (threadIdx.x < kLexicographicTopKBins &&
      shared.histogram[0][threadIdx.x] > shared.remaining &&
      shared.histogram[0][threadIdx.x + 1] <= shared.remaining) {
    shared.threshold_bin = threadIdx.x;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t bin = shared.threshold_bin;
    const uint32_t greater = shared.histogram[0][bin + 1];
    shared.remaining -= greater;
    shared.prefix |= bin << shift;
    if (shared.remaining == 0) {
      const uint32_t low_mask = shift == 0 ? 0u : ((uint32_t{1} << shift) - 1);
      shared.pivot = shared.prefix | low_mask;
      shared.remaining_ties = 0;
    } else if (shift == 0) {
      shared.pivot = shared.prefix;
      shared.remaining_ties = shared.remaining;
    }
  }
  __syncthreads();
}

template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_topk_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ output, uint32_t num_rows, uint32_t columns,
    uint32_t stride) {
  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t raw_length = lengths[row];
  const uint32_t length =
      raw_length > 0 ? min(static_cast<uint32_t>(raw_length), columns) : 0;
  const float* row_logits = logits + static_cast<uint64_t>(row) * stride;
  int32_t* row_output = output + static_cast<uint64_t>(row) * TopK;

  if (length <= TopK) {
    for (uint32_t index = tx; index < TopK;
         index += kLexicographicTopKThreads) {
      row_output[index] = index < length ? static_cast<int32_t>(index) : -1;
    }
    return;
  }

  __shared__ LexicographicTopKShared<TopK> shared;
  if (tx == 0) {
    shared.prefix = 0;
    shared.remaining = TopK;
  }
  __syncthreads();

  // Select the exact k-th score with four byte-wide radix passes. The prior
  // implementation also radix-selected the 32-bit index tie-break, requiring
  // eight full scans. We only need the score pivot: the final increasing-index
  // compaction can admit exactly `remaining` values from the pivot bucket.
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    for (uint32_t bin = tx; bin < kLexicographicTopKBins;
         bin += kLexicographicTopKThreads) {
      shared.histogram[bin] = 0;
    }
    __syncthreads();

    const int shift = 24 - pass * 8;
    const uint32_t prefix = shared.prefix;
    const uint32_t prefix_mask = pass == 0 ? 0 : (~uint32_t{0} << (shift + 8));
    for (uint32_t index = tx; index < length;
         index += kLexicographicTopKThreads) {
      const uint32_t key = ordered_float_bits(row_logits[index]);
      if ((key & prefix_mask) == prefix) {
        atomicAdd(&shared.histogram[(key >> shift) & 0xffu], 1u);
      }
    }
    __syncthreads();

    if (tx == 0) {
      uint32_t remaining = shared.remaining;
      for (int bin = kLexicographicTopKBins - 1; bin >= 0; --bin) {
        const uint32_t count = shared.histogram[bin];
        if (remaining > count) {
          remaining -= count;
        } else {
          shared.prefix |= static_cast<uint32_t>(bin) << shift;
          shared.remaining = remaining;
          break;
        }
      }
    }
    __syncthreads();
  }

  if (tx == 0) {
    shared.pivot = shared.prefix;
    shared.greater_seen = 0;
    shared.equal_seen = 0;
  }
  __syncthreads();

  // Compact in increasing index order. A packed 64-bit scan tracks counts of
  // greater and pivot-equal scores at once. Every greater score is selected;
  // only the first `remaining` pivot ties are admitted, preserving the exact
  // lower-index tie break and QSA's canonical accumulation order.
  using BlockScan = typename LexicographicTopKShared<TopK>::BlockScan;
  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads) {
    const uint32_t index = base + tx;
    const uint32_t key =
        index < length ? ordered_float_bits(row_logits[index]) : 0;
    const uint32_t greater = index < length && key > shared.pivot ? 1u : 0u;
    const uint32_t equal = index < length && key == shared.pivot ? 1u : 0u;
    const uint64_t counts = (static_cast<uint64_t>(greater) << 32) | equal;
    uint64_t prefix_counts = 0;
    uint64_t aggregate_counts = 0;
    BlockScan(shared.scan)
        .ExclusiveSum(counts, prefix_counts, aggregate_counts);
    __syncthreads();
    if (tx == 0) {
      shared.chunk_greater_base = shared.greater_seen;
      shared.chunk_equal_base = shared.equal_seen;
      shared.greater_seen += static_cast<uint32_t>(aggregate_counts >> 32);
      shared.equal_seen += static_cast<uint32_t>(aggregate_counts);
    }
    __syncthreads();
    const uint32_t greater_before =
        shared.chunk_greater_base + static_cast<uint32_t>(prefix_counts >> 32);
    const uint32_t equal_before =
        shared.chunk_equal_base + static_cast<uint32_t>(prefix_counts);
    const bool selected = greater || (equal && equal_before < shared.remaining);
    if (selected) {
      const uint32_t offset =
          greater_before + min(equal_before, shared.remaining);
      row_output[offset] = static_cast<int32_t>(index);
    }
    __syncthreads();
  }
}

// Single-token QSA decode has only about two thousand live block scores at the
// common 8K context length. After the first radix byte, scanning all scores for
// the other three bytes wastes most of the work. Compact the selected coarse
// bucket into shared memory and refine that much smaller set instead. Integer
// counters and the final increasing-index pass retain exact tie-breaking.
template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_decode_topk_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ output, uint32_t columns) {
  const uint32_t tx = threadIdx.x;
  const int32_t raw_length = lengths[0];
  const uint32_t length =
      raw_length > 0 ? min(static_cast<uint32_t>(raw_length), columns) : 0;

  if (length <= TopK) {
    for (uint32_t index = tx; index < TopK;
         index += kLexicographicTopKThreads) {
      output[index] = index < length ? static_cast<int32_t>(index) : -1;
    }
    return;
  }

  __shared__ LexicographicDecodeTopKShared<TopK> shared;
  if (tx == 0) {
    shared.prefix = 0;
    shared.remaining = TopK;
    shared.remaining_ties = 0;
    shared.candidate_count[0] = 0;
  }
  __syncthreads();

  if (length > kLexicographicTopKDecodeCandidateCapacity) {
    // Preserve the original exact four-pass algorithm for long contexts,
    // without a host synchronization or a second kernel launch.
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      for (uint32_t bin = tx; bin < kLexicographicTopKBins;
           bin += kLexicographicTopKThreads) {
        shared.histogram[0][bin] = 0;
      }
      __syncthreads();

      const int shift = 24 - pass * 8;
      const uint32_t prefix = shared.prefix;
      const uint32_t prefix_mask =
          pass == 0 ? 0 : (~uint32_t{0} << (shift + 8));
      for (uint32_t index = tx; index < length;
           index += kLexicographicTopKThreads) {
        const uint32_t key = ordered_float_bits(logits[index]);
        if ((key & prefix_mask) == prefix) {
          atomicAdd(&shared.histogram[0][(key >> shift) & 0xffu], 1u);
        }
      }
      __syncthreads();

      if (tx == 0) {
        uint32_t remaining = shared.remaining;
        for (int bin = kLexicographicTopKBins - 1; bin >= 0; --bin) {
          const uint32_t count = shared.histogram[0][bin];
          if (remaining > count) {
            remaining -= count;
          } else {
            shared.prefix |= static_cast<uint32_t>(bin) << shift;
            shared.remaining = remaining;
            break;
          }
        }
      }
      __syncthreads();
    }
    if (tx == 0) {
      shared.pivot = shared.prefix;
      shared.remaining_ties = shared.remaining;
    }
    __syncthreads();
  } else {
    // Coarse pass over the complete score row.
    if (tx < kLexicographicTopKBins + 1) shared.histogram[0][tx] = 0;
    __syncthreads();
    for (uint32_t index = tx; index < length;
         index += kLexicographicTopKThreads) {
      const uint32_t key = ordered_float_bits(logits[index]);
      atomicAdd(&shared.histogram[0][key >> 24], 1u);
    }
    __syncthreads();
    decode_suffix_scan_histogram(shared);
    decode_choose_threshold(shared, 24);

    if (shared.remaining != 0) {
      if (tx < kLexicographicTopKBins + 1) shared.histogram[0][tx] = 0;
      __syncthreads();
      for (uint32_t index = tx; index < length;
           index += kLexicographicTopKThreads) {
        const uint32_t key = ordered_float_bits(logits[index]);
        if ((key & 0xff000000u) == shared.prefix) {
          const uint32_t position = atomicAdd(&shared.candidate_count[0], 1u);
          shared.candidates[0][position] = static_cast<int32_t>(index);
          atomicAdd(&shared.histogram[0][(key >> 16) & 0xffu], 1u);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int radix_pass = 0; radix_pass < 3; ++radix_pass) {
      if (shared.remaining == 0) break;
      const int shift = 16 - radix_pass * 8;
      decode_suffix_scan_histogram(shared);
      decode_choose_threshold(shared, shift);
      if (shared.remaining == 0 || shift == 0) break;

      const int source = radix_pass & 1;
      const int target = source ^ 1;
      if (tx == 0) shared.candidate_count[target] = 0;
      if (tx < kLexicographicTopKBins + 1) shared.histogram[0][tx] = 0;
      __syncthreads();
      const uint32_t count = shared.candidate_count[source];
      const uint32_t prefix_mask = ~uint32_t{0} << shift;
      const int next_shift = shift - 8;
      for (uint32_t item = tx; item < count;
           item += kLexicographicTopKThreads) {
        const int32_t index = shared.candidates[source][item];
        const uint32_t key = ordered_float_bits(logits[index]);
        if ((key & prefix_mask) == shared.prefix) {
          const uint32_t position =
              atomicAdd(&shared.candidate_count[target], 1u);
          shared.candidates[target][position] = index;
          atomicAdd(&shared.histogram[0][(key >> next_shift) & 0xffu], 1u);
        }
      }
      __syncthreads();
    }
  }

  if (tx == 0) {
    shared.greater_seen = 0;
    shared.equal_seen = 0;
  }
  __syncthreads();

  // Emit in original index order, matching QSA's canonical accumulation order.
  using BlockScan = typename LexicographicDecodeTopKShared<TopK>::BlockScan;
  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads) {
    const uint32_t index = base + tx;
    const uint32_t key = index < length ? ordered_float_bits(logits[index]) : 0;
    const uint32_t greater = index < length && key > shared.pivot ? 1u : 0u;
    const uint32_t equal = index < length && key == shared.pivot ? 1u : 0u;
    const uint64_t counts = (static_cast<uint64_t>(greater) << 32) | equal;
    uint64_t prefix_counts = 0;
    uint64_t aggregate_counts = 0;
    BlockScan(shared.scan)
        .ExclusiveSum(counts, prefix_counts, aggregate_counts);
    __syncthreads();
    if (tx == 0) {
      shared.chunk_greater_base = shared.greater_seen;
      shared.chunk_equal_base = shared.equal_seen;
      shared.greater_seen += static_cast<uint32_t>(aggregate_counts >> 32);
      shared.equal_seen += static_cast<uint32_t>(aggregate_counts);
    }
    __syncthreads();
    const uint32_t greater_before =
        shared.chunk_greater_base + static_cast<uint32_t>(prefix_counts >> 32);
    const uint32_t equal_before =
        shared.chunk_equal_base + static_cast<uint32_t>(prefix_counts);
    const bool selected =
        greater || (equal && equal_before < shared.remaining_ties);
    if (selected) {
      const uint32_t offset =
          greater_before + min(equal_before, shared.remaining_ties);
      output[offset] = static_cast<int32_t>(index);
    }
    __syncthreads();
  }
}

template <int TopK>
void launch_qsa_lexicographic_topk(const float* logits, const int32_t* lengths,
                                   int32_t* output, uint32_t num_rows,
                                   uint32_t columns, uint32_t stride,
                                   cudaStream_t stream) {
  if (num_rows == 1) {
    qsa_lexicographic_decode_topk_kernel<TopK>
        <<<1, kLexicographicTopKThreads, 0, stream>>>(logits, lengths, output,
                                                      columns);
  } else {
    qsa_lexicographic_topk_kernel<TopK>
        <<<num_rows, kLexicographicTopKThreads, 0, stream>>>(
            logits, lengths, output, num_rows, columns, stride);
  }
}

}  // namespace vllm::qsa

#endif  // VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_
