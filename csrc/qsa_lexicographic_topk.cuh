// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_
#define VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace vllm::qsa {

constexpr int kLexicographicTopKThreads = 1024;
constexpr int kLexicographicTopKBins = 256;
constexpr int kLexicographicTopKDecodeCandidateCapacity = 2304;
constexpr int kLexicographicTopKCompactMinLength = 32768;

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
__device__ __forceinline__ void qsa_lexicographic_topk_body(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ output, uint32_t num_rows, uint32_t columns,
    uint32_t stride, LexicographicTopKShared<TopK>& shared) {
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

// Each QSA decode row has only about two thousand live block scores at the
// common 8K context length. After the first radix byte, scanning all scores for
// the other three bytes wastes most of the work. Compact the selected coarse
// bucket into shared memory and refine that much smaller set instead. Integer
// counters and the final increasing-index pass retain exact tie-breaking.
template <int TopK>
__device__ __forceinline__ void qsa_lexicographic_decode_topk_body(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ output, uint32_t columns,
    LexicographicDecodeTopKShared<TopK>& shared) {
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
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_topk_kernel(
    const float* logits, const int32_t* lengths, int32_t* output, uint32_t rows,
    uint32_t columns, uint32_t stride) {
  __shared__ LexicographicTopKShared<TopK> shared;
  qsa_lexicographic_topk_body<TopK>(logits, lengths, output, rows, columns,
                                    stride, shared);
}

// A monotone coarse key narrows the candidate set without rounding the scores
// used for the final selection. Signed zero retains the existing tie contract.
__device__ __forceinline__ uint32_t compact_coarse_key(float value) {
  if (value == 0.0f) value = 0.0f;
  const uint16_t bits = __half_as_ushort(__float2half_rn(value));
  return static_cast<uint16_t>((bits & 0x8000u) ? ~bits : (bits | 0x8000u)) >>
         8;
}

struct CompactTopKShared {
  uint32_t histogram[kLexicographicTopKBins];
  int32_t candidates[2][kLexicographicTopKDecodeCandidateCapacity];
  uint32_t remaining;
  uint32_t selected_bin;
  uint32_t count;
  uint32_t next_count;
  uint32_t prefix;
  uint32_t invalid;
};

__device__ __forceinline__ void compact_choose_bin(CompactTopKShared& shared) {
  if (threadIdx.x == 0) {
    for (int bin = kLexicographicTopKBins - 1; bin >= 0; --bin) {
      const uint32_t count = shared.histogram[bin];
      if (shared.remaining > count) {
        shared.remaining -= count;
      } else {
        shared.selected_bin = bin;
        shared.next_count = count;
        break;
      }
    }
  }
  __syncthreads();
}

template <int TopK>
__device__ bool compact_score_pivot(const float* logits, uint32_t length,
                                    CompactTopKShared& shared) {
  const uint32_t tx = threadIdx.x;
  // Probe 32 contiguous scores at 32 positions before scanning the full row.
  // A concentrated tile only declines the optimization: selection still
  // considers every live score, even if the probe is unrepresentative.
  const uint32_t lane = tx & 31;
  const float probe = logits[(tx >> 5) * (length >> 5) + lane];
  const uint32_t probe_key = compact_coarse_key(probe);
  const bool concentrated = __all_sync(
      0xffffffffu, probe_key == __shfl_sync(0xffffffffu, probe_key, 0));
  if (__syncthreads_or(isnan(probe) || concentrated)) return false;
  if (tx < kLexicographicTopKBins) shared.histogram[tx] = 0;
  if (tx == 0) {
    shared.remaining = TopK;
    shared.prefix = 0;
    shared.invalid = 0;
  }
  __syncthreads();
  for (uint32_t index = tx; index < length; index += blockDim.x) {
    const float score = logits[index];
    if (isnan(score)) atomicExch(&shared.invalid, 1u);
    atomicAdd(&shared.histogram[compact_coarse_key(score)], 1u);
  }
  __syncthreads();
  if (shared.invalid) return false;
  compact_choose_bin(shared);
  // Never truncate a bucket: clustered scores use the original exact selector.
  if (shared.next_count > kLexicographicTopKDecodeCandidateCapacity)
    return false;
  const uint32_t coarse_bin = shared.selected_bin;
  if (tx == 0) shared.count = 0;
  __syncthreads();
  for (uint32_t index = tx; index < length; index += blockDim.x) {
    if (compact_coarse_key(logits[index]) == coarse_bin) {
      const uint32_t slot = atomicAdd(&shared.count, 1u);
      shared.candidates[0][slot] = index;
    }
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    const int source = pass & 1;
    const int shift = 24 - 8 * pass;
    if (tx < kLexicographicTopKBins) shared.histogram[tx] = 0;
    __syncthreads();
    for (uint32_t item = tx; item < shared.count; item += blockDim.x) {
      const auto key =
          ordered_float_bits(logits[shared.candidates[source][item]]);
      atomicAdd(&shared.histogram[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    compact_choose_bin(shared);
    if (tx == 0) shared.prefix |= shared.selected_bin << shift;
    __syncthreads();
    if (shift == 0) {
      // An ambiguous cutoff needs the original lower-index tie resolution.
      return shared.next_count == shared.remaining;
    }
    const uint32_t count = shared.count;
    const uint32_t bin = shared.selected_bin;
    if (tx == 0) shared.next_count = 0;
    __syncthreads();
    for (uint32_t item = tx; item < count; item += blockDim.x) {
      const int32_t index = shared.candidates[source][item];
      if (((ordered_float_bits(logits[index]) >> shift) & 255u) == bin) {
        const uint32_t slot = atomicAdd(&shared.next_count, 1u);
        shared.candidates[source ^ 1][slot] = index;
      }
    }
    __syncthreads();
    if (tx == 0) shared.count = shared.next_count;
    __syncthreads();
  }
  return false;
}

template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_compact_topk_kernel(
    const float* logits, const int32_t* lengths, int32_t* output, uint32_t rows,
    uint32_t columns, uint32_t stride, bool decode_batch) {
  using Sort = cub::BlockRadixSort<uint32_t, kLexicographicTopKThreads, 1>;
  __shared__ union {
    CompactTopKShared compact;
    LexicographicTopKShared<TopK> exact;
    LexicographicDecodeTopKShared<TopK> decode;
    typename Sort::TempStorage sort;
  } shared;
  const uint32_t row = blockIdx.x, tx = threadIdx.x;
  const uint32_t length =
      lengths[row] > 0 ? min(uint32_t(lengths[row]), columns) : 0;
  const float* scores = logits + static_cast<uint64_t>(row) * stride;
  int32_t* indices = output + static_cast<uint64_t>(row) * TopK;
  if (length <= TopK) {
    if (tx < TopK) indices[tx] = tx < length ? int32_t(tx) : -1;
    return;
  }
  // Capacity can be large while a replay uses short live sequences. Make the
  // length decision on device and preserve the original per-row short path.
  const bool compact =
      length >= kLexicographicTopKCompactMinLength &&
      compact_score_pivot<TopK>(scores, length, shared.compact);
  if (!compact) {
    __syncthreads();
    if (rows == 1 || (decode_batch && rows <= 16 && rows != 5 && rows != 10)) {
      qsa_lexicographic_decode_topk_body<TopK>(scores, lengths + row, indices,
                                               columns, shared.decode);
    } else {
      qsa_lexicographic_topk_body<TopK>(logits, lengths, output, rows, columns,
                                        stride, shared.exact);
    }
    return;
  }
  const uint32_t pivot = shared.compact.prefix;
  if (tx == 0) shared.compact.count = 0;
  __syncthreads();
  for (uint32_t index = tx; index < length; index += blockDim.x) {
    if (ordered_float_bits(scores[index]) >= pivot) {
      const uint32_t slot = atomicAdd(&shared.compact.count, 1u);
      if (slot < TopK) indices[slot] = index;
    }
  }
  __syncthreads();
  if (shared.compact.count != TopK) {
    __syncthreads();
    qsa_lexicographic_topk_body<TopK>(logits, lengths, output, rows, columns,
                                      stride, shared.exact);
    return;
  }
  uint32_t keys[1] = {tx < TopK ? uint32_t(indices[tx]) : UINT32_MAX};
  __syncthreads();
  Sort(shared.sort).Sort(keys);
  if (tx < TopK) indices[tx] = keys[0];
}

template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_decode_topk_kernel(
    const float* logits, const int32_t* lengths, int32_t* output,
    uint32_t columns, uint32_t stride) {
  __shared__ LexicographicDecodeTopKShared<TopK> shared;
  const uint32_t row = blockIdx.x;
  qsa_lexicographic_decode_topk_body<TopK>(
      logits + static_cast<uint64_t>(row) * stride, lengths + row,
      output + static_cast<uint64_t>(row) * TopK, columns, shared);
}

template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_mtp_topk_kernel(
    const float* logits, const int32_t* lengths, int32_t* output, uint32_t rows,
    uint32_t columns, uint32_t stride) {
  // Only one path runs in a CTA. Sharing its storage avoids reserving both
  // workspaces and unnecessarily reducing the available L1 cache capacity.
  __shared__ union {
    LexicographicTopKShared<TopK> normal;
    LexicographicDecodeTopKShared<TopK> compact;
  } shared;
  const uint32_t row = blockIdx.x;
  if (lengths[row] <= kLexicographicTopKDecodeCandidateCapacity) {
    qsa_lexicographic_decode_topk_body<TopK>(
        logits + static_cast<uint64_t>(row) * stride, lengths + row,
        output + static_cast<uint64_t>(row) * TopK, columns, shared.compact);
  } else {
    qsa_lexicographic_topk_body<TopK>(logits, lengths, output, rows, columns,
                                      stride, shared.normal);
  }
}

template <int TopK>
void launch_qsa_lexicographic_topk(const float* logits, const int32_t* lengths,
                                   int32_t* output, uint32_t num_rows,
                                   uint32_t columns, uint32_t stride,
                                   cudaStream_t stream,
                                   bool decode_batch = false) {
  if (num_rows <= 32 && columns >= kLexicographicTopKCompactMinLength) {
    qsa_lexicographic_compact_topk_kernel<TopK>
        <<<num_rows, kLexicographicTopKThreads, 0, stream>>>(
            logits, lengths, output, num_rows, columns, stride, decode_batch);
  } else if (num_rows == 1) {
    qsa_lexicographic_decode_topk_kernel<TopK>
        <<<1, kLexicographicTopKThreads, 0, stream>>>(logits, lengths, output,
                                                      columns, stride);
  } else if (decode_batch && (num_rows == 5 || num_rows == 10) &&
             columns <= 9216) {
    // Qualified through 32K context, allowing physical page padding. Longer
    // score buffers retain the original kernel and its resource allocation.
    qsa_lexicographic_mtp_topk_kernel<TopK>
        <<<num_rows, kLexicographicTopKThreads, 0, stream>>>(
            logits, lengths, output, num_rows, columns, stride);
  } else if (decode_batch && num_rows <= 16 && num_rows != 5 &&
             num_rows != 10) {
    qsa_lexicographic_decode_topk_kernel<TopK>
        <<<num_rows, kLexicographicTopKThreads, 0, stream>>>(
            logits, lengths, output, columns, stride);
  } else {
    qsa_lexicographic_topk_kernel<TopK>
        <<<num_rows, kLexicographicTopKThreads, 0, stream>>>(
            logits, lengths, output, num_rows, columns, stride);
  }
}

}  // namespace vllm::qsa

#endif  // VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_
