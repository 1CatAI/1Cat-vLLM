
#include "moe_permute_unpermute_kernel.h"
#include <cstdlib>

// moe_permute kernels require at least CUDA 12.0
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)

// CubKeyValueSorter definition begin
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0), num_bits_(sizeof(int) * 8) {}

int CubKeyValueSorter::expertsToBits(int num_experts) {
  // Max value we represent is V = num_experts + (num_experts - 1) = 2 *
  // num_experts - 1 The maximum number of bits is therefore floor(log2(V)) + 1
  return static_cast<int>(log2(2 * num_experts - 1)) + 1;
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
    : num_experts_(num_experts), num_bits_(expertsToBits(num_experts)) {}

void CubKeyValueSorter::updateNumExperts(int const num_experts) {
  num_experts_ = num_experts;
  num_bits_ = expertsToBits(num_experts);
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs,
                                           int const num_experts) {
  int num_bits = expertsToBits(num_experts);
  size_t required_storage = 0;
  int* null_int = nullptr;
  cub::DeviceRadixSort::SortPairs(nullptr, required_storage, null_int, null_int,
                                  null_int, null_int, num_key_value_pairs, 0,
                                  num_bits);

  //   when num_key_value_pairs, num_experts, num_bits, required_storage = 64,
  //   4, 3, 0 The required_storage seems to vary between 0 and 1 for the same
  //   inputs
  if (required_storage == 0) {
    required_storage = 1;
  }
  return required_storage;
}

void CubKeyValueSorter::run(void* workspace, size_t const workspace_size,
                            int const* keys_in, int* keys_out,
                            int const* values_in, int* values_out,
                            size_t const num_key_value_pairs,
                            cudaStream_t stream) {
  size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
  size_t actual_ws_size = workspace_size;

  TORCH_CHECK(expected_ws_size <= workspace_size,
              "[CubKeyValueSorter::run] The allocated workspace is too small "
              "to run this problem.");
  cub::DeviceRadixSort::SortPairs(workspace, actual_ws_size, keys_in, keys_out,
                                  values_in, values_out, num_key_value_pairs, 0,
                                  num_bits_, stream);
}
// CubKeyValueSorter definition end

static inline size_t pad_to_multiple_of_16(size_t const& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

namespace {

constexpr int kSingleTokenFastPathMaxTopK = 32;

bool envFlagEnabled(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && std::atoi(raw) != 0;
}

bool singleTokenPermuteFastPathEnabled() {
  static const bool enabled = []() {
    if (envFlagEnabled("VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH")) {
      return true;
    }
    const char* raw = std::getenv("VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH");
    return raw != nullptr && std::atoi(raw) != 0;
  }();
  return enabled;
}

bool singleTokenUnpermuteFastPathEnabled() {
  static const bool enabled = []() {
    if (envFlagEnabled("VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH")) {
      return true;
    }
    const char* raw = std::getenv("VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH");
    return raw != nullptr && std::atoi(raw) != 0;
  }();
  return enabled;
}

template <typename T>
__global__ void singleTokenMoePermuteKernel(
    T const* input, int const* topk_ids, T* permuted_output,
    int64_t* expert_first_token_offset, int* inv_permuted_idx,
    int* permuted_idx, int* m_indices, int num_experts, int topk,
    int64_t cols) {
  __shared__ int sorted_ids[kSingleTokenFastPathMaxTopK];
  __shared__ int sorted_src[kSingleTokenFastPathMaxTopK];

  if (threadIdx.x < topk) {
    sorted_ids[threadIdx.x] = __ldg(topk_ids + threadIdx.x);
    sorted_src[threadIdx.x] = threadIdx.x;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 1; i < topk; ++i) {
      int expert_id = sorted_ids[i];
      int src_idx = sorted_src[i];
      int j = i - 1;
      while (j >= 0 && sorted_ids[j] > expert_id) {
        sorted_ids[j + 1] = sorted_ids[j];
        sorted_src[j + 1] = sorted_src[j];
        --j;
      }
      sorted_ids[j + 1] = expert_id;
      sorted_src[j + 1] = src_idx;
    }

    int cursor = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      expert_first_token_offset[expert] = cursor;
      while (cursor < topk && sorted_ids[cursor] == expert) {
        ++cursor;
      }
    }
    expert_first_token_offset[num_experts] = cursor;
  }
  __syncthreads();

  if (threadIdx.x < topk) {
    int const sorted_pos = threadIdx.x;
    int const src_idx = sorted_src[sorted_pos];
    inv_permuted_idx[src_idx] = sorted_pos;
    permuted_idx[sorted_pos] = src_idx;
    m_indices[sorted_pos] = sorted_ids[sorted_pos];
  }

  constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
  using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

  auto const* source_row_ptr = reinterpret_cast<DataElem const*>(input);
  auto* dest_row_ptr = reinterpret_cast<DataElem*>(permuted_output);
  int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;
  int64_t const total_elems = num_elems_in_col * topk;

  for (int64_t elem_index = threadIdx.x; elem_index < total_elems;
       elem_index += blockDim.x) {
    int64_t const row = elem_index / num_elems_in_col;
    int64_t const col = elem_index - row * num_elems_in_col;
    dest_row_ptr[row * num_elems_in_col + col] = source_row_ptr[col];
  }
}

template <typename T>
__global__ void singleTokenMoeUnpermuteKernel(
    T const* expanded_permuted_rows, T* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t cols, int64_t topk) {
  __shared__ int inv_perm[kSingleTokenFastPathMaxTopK];
  __shared__ float route_scale[kSingleTokenFastPathMaxTopK];

  if (threadIdx.x < topk) {
    inv_perm[threadIdx.x] =
        __ldg(expanded_source_row_to_expanded_dest_row + threadIdx.x);
    route_scale[threadIdx.x] = __ldg(scales + threadIdx.x);
  }
  __syncthreads();

  constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
  using InputElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

  auto const* expanded_rows_v =
      reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_unpermuted_output);
  int64_t const num_elems_in_col = cols / FINALIZE_ELEM_PER_THREAD;

  for (int64_t elem_index = threadIdx.x; elem_index < num_elems_in_col;
       elem_index += blockDim.x) {
    ComputeElem thread_output;
    thread_output.fill(0);

#pragma unroll
    for (int k_idx = 0; k_idx < kSingleTokenFastPathMaxTopK; ++k_idx) {
      if (k_idx >= topk) {
        break;
      }
      int const expanded_row = inv_perm[k_idx];
      auto const* expanded_row_ptr =
          expanded_rows_v + expanded_row * num_elems_in_col;
      ComputeElem expert_result =
          arrayConvert<InputElem, ComputeElem>(expanded_row_ptr[elem_index]);
      thread_output = thread_output + route_scale[k_idx] * expert_result;
    }

    reduced_row_ptr_v[elem_index] = arrayConvert<ComputeElem, OutputElem>(
        thread_output);
  }
}

}  // namespace

template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices,
                                                      int64_t const arr_length,
                                                      T const target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] >= target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

// Calculates the start offset of the tokens for a given expert. The last
// element is the total number of valid tokens
__global__ void computeExpertFirstTokenOffsetKernel(
    int const* sorted_experts, int64_t const sorted_experts_len,
    int const num_experts, int64_t* expert_first_token_offset) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;

  // Note that expert goes [0, num_experts] (inclusive) because we want a count
  // for the total number of active tokens at the end of the scan.
  if (expert >= num_experts + 1) {
    return;
  }
  expert_first_token_offset[expert] =
      findTotalEltsLessThanTarget(sorted_experts, sorted_experts_len, expert);
}

void computeExpertFirstTokenOffset(int const* sorted_indices,
                                   int const total_indices,
                                   int const num_experts,
                                   int64_t* expert_first_token_offset,
                                   cudaStream_t stream) {
  int const num_entries = num_experts + 1;
  int const threads = std::min(1024, num_entries);
  int const blocks = (num_entries + threads - 1) / threads;

  computeExpertFirstTokenOffsetKernel<<<blocks, threads, 0, stream>>>(
      sorted_indices, total_indices, num_experts, expert_first_token_offset);
}

void sortAndScanExpert(const int* expert_for_source_row, const int* source_rows,
                       int* permuted_experts, int* permuted_rows,
                       int64_t* expert_first_token_offset, int num_rows,
                       int num_experts, int num_experts_per_node, int k,
                       CubKeyValueSorter& sorter, void* sorter_ws,
                       cudaStream_t stream) {
  int64_t const expanded_num_rows = static_cast<int64_t>(k) * num_rows;
  // We need to use the full num_experts because that is the sentinel value used
  // by topk for disabled experts
  sorter.updateNumExperts(num_experts);
  size_t const sorter_ws_size_bytes = pad_to_multiple_of_16(
      sorter.getWorkspaceSize(expanded_num_rows, num_experts));
  sorter.run((void*)sorter_ws, sorter_ws_size_bytes, expert_for_source_row,
             permuted_experts, source_rows, permuted_rows, expanded_num_rows,
             stream);
  computeExpertFirstTokenOffset(permuted_experts, expanded_num_rows,
                                num_experts_per_node, expert_first_token_offset,
                                stream);
}

__global__ void preprocessTopkIdKernel(int* topk_id_ptr, int size,
                                       const int* expert_map_ptr,
                                       int num_experts) {
  auto tidx = threadIdx.x;
  auto bidx = blockIdx.x;
  auto offset = bidx * blockDim.x;
  auto bound = min(offset + blockDim.x, size);
  extern __shared__ int smem_expert_map[];
  // store expert_map in smem
  for (int i = tidx; i < num_experts; i += blockDim.x) {
    smem_expert_map[i] = expert_map_ptr[i];
  }
  __syncthreads();

  // query global expert id in expert map.
  // if global expert id = -1 in exert map, plus n_expert
  // else set global expert id = exert map[global expert id]
  if (offset + tidx < bound) {
    auto topk_id = topk_id_ptr[offset + tidx];
    auto local_expert_idx = smem_expert_map[topk_id];
    if (local_expert_idx == -1) {
      topk_id += num_experts;
    } else {
      topk_id = local_expert_idx;
    }
    __syncwarp();
    topk_id_ptr[offset + tidx] = topk_id;
  }
}
void preprocessTopkIdLauncher(int* topk_id_ptr, int size,
                              const int* expert_map_ptr, int num_experts,
                              cudaStream_t stream) {
  int block = std::min(size, 1024);
  int grid = (size + block - 1) / block;
  int smem_size = (num_experts) * sizeof(int);
  preprocessTopkIdKernel<<<grid, block, smem_size, stream>>>(
      topk_id_ptr, size, expert_map_ptr, num_experts);
}

template <bool ALIGN_BLOCK_SIZE>
__global__ void getMIndicesKernel(int64_t* expert_first_token_offset,
                                  int64_t* align_expert_first_token_offset,
                                  int* m_indices, const int num_local_expert,
                                  const int align_block_size) {
  int eidx = blockIdx.x;
  int tidx = threadIdx.x;
  extern __shared__ int64_t smem_expert_first_token_offset[];
  for (int i = tidx; i <= num_local_expert; i += blockDim.x) {
    smem_expert_first_token_offset[i] = __ldg(expert_first_token_offset + i);
  }
  __syncthreads();
  auto last_token_offset = smem_expert_first_token_offset[eidx + 1];
  auto first_token_offset = smem_expert_first_token_offset[eidx];
  int n_token_in_expert = last_token_offset - first_token_offset;

  if constexpr (ALIGN_BLOCK_SIZE) {
    n_token_in_expert = (n_token_in_expert + align_block_size - 1) /
                        align_block_size * align_block_size;
    // round up to ALIGN_BLOCK_SIZE
    int64_t accumulate_align_offset = 0;
    for (int i = 1; i <= eidx + 1; i++) {
      int n_token = smem_expert_first_token_offset[i] -
                    smem_expert_first_token_offset[i - 1];
      accumulate_align_offset =
          accumulate_align_offset + (n_token + align_block_size - 1) /
                                        align_block_size * align_block_size;
      if (i == eidx) {
        first_token_offset = accumulate_align_offset;
      }
      // last block store align_expert_first_token_offset
      if (eidx == num_local_expert - 1 && threadIdx.x == 0) {
        align_expert_first_token_offset[i] = accumulate_align_offset;
      }
    }
  }
  for (int idx = tidx; idx < n_token_in_expert; idx += blockDim.x) {
    // update m_indice with expert id
    m_indices[first_token_offset + idx] = eidx;
  }
}

void getMIndices(int64_t* expert_first_token_offset,
                 int64_t* align_expert_first_token_offset, int* m_indices,
                 int num_local_expert, const int align_block_size,
                 cudaStream_t stream) {
  int block = 256;
  int grid = num_local_expert;
  int smem_size = sizeof(int64_t) * (num_local_expert + 1);
  if (align_block_size == -1) {
    getMIndicesKernel<false><<<grid, block, smem_size, stream>>>(
        expert_first_token_offset, align_expert_first_token_offset, m_indices,
        num_local_expert, align_block_size);
  } else {
    getMIndicesKernel<true><<<grid, block, smem_size, stream>>>(
        expert_first_token_offset, align_expert_first_token_offset, m_indices,
        num_local_expert, align_block_size);
  }
}

bool canUseSingleTokenMoePermuteFastPath(
    int64_t n_token, int64_t topk, bool has_expert_map,
    const std::optional<int64_t>& align_block_size, int64_t n_expert,
    int64_t n_local_expert) {
  if (!singleTokenPermuteFastPathEnabled()) {
    return false;
  }
  if (n_token != 1 || topk <= 0 || topk > kSingleTokenFastPathMaxTopK) {
    return false;
  }
  if (has_expert_map || align_block_size.has_value()) {
    return false;
  }
  return n_expert > 0 && n_expert == n_local_expert;
}

template <typename T>
void singleTokenMoePermuteLauncher(
    T const* input, int const* topk_ids, T* permuted_output,
    int64_t* expert_first_token_offset, int* inv_permuted_idx,
    int* permuted_idx, int* m_indices, int num_experts, int topk,
    int64_t cols, cudaStream_t stream) {
  constexpr int threads = 256;
  singleTokenMoePermuteKernel<T><<<1, threads, 0, stream>>>(
      input, topk_ids, permuted_output, expert_first_token_offset,
      inv_permuted_idx, permuted_idx, m_indices, num_experts, topk, cols);
}

bool canUseSingleTokenMoeUnpermuteFastPath(int64_t n_token, int64_t topk) {
  return singleTokenUnpermuteFastPathEnabled() && n_token == 1 && topk > 0 &&
         topk <= kSingleTokenFastPathMaxTopK;
}

template <typename T>
void singleTokenMoeUnpermuteLauncher(
    T const* expanded_permuted_rows, T* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t cols, int64_t topk, cudaStream_t stream) {
  constexpr int threads = 256;
  singleTokenMoeUnpermuteKernel<T><<<1, threads, 0, stream>>>(
      expanded_permuted_rows, reduced_unpermuted_output, scales,
      expanded_source_row_to_expanded_dest_row, cols, topk);
}

template void singleTokenMoePermuteLauncher<half>(
    half const* input, int const* topk_ids, half* permuted_output,
    int64_t* expert_first_token_offset, int* inv_permuted_idx,
    int* permuted_idx, int* m_indices, int num_experts, int topk,
    int64_t cols, cudaStream_t stream);

template void singleTokenMoePermuteLauncher<float>(
    float const* input, int const* topk_ids, float* permuted_output,
    int64_t* expert_first_token_offset, int* inv_permuted_idx,
    int* permuted_idx, int* m_indices, int num_experts, int topk,
    int64_t cols, cudaStream_t stream);

template void singleTokenMoePermuteLauncher<__nv_bfloat16>(
    __nv_bfloat16 const* input, int const* topk_ids,
    __nv_bfloat16* permuted_output, int64_t* expert_first_token_offset,
    int* inv_permuted_idx, int* permuted_idx, int* m_indices, int num_experts,
    int topk, int64_t cols, cudaStream_t stream);

template void singleTokenMoeUnpermuteLauncher<half>(
    half const* expanded_permuted_rows, half* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t cols, int64_t topk, cudaStream_t stream);

template void singleTokenMoeUnpermuteLauncher<float>(
    float const* expanded_permuted_rows, float* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t cols, int64_t topk, cudaStream_t stream);

template void singleTokenMoeUnpermuteLauncher<__nv_bfloat16>(
    __nv_bfloat16 const* expanded_permuted_rows,
    __nv_bfloat16* reduced_unpermuted_output, float const* scales,
    int const* expanded_source_row_to_expanded_dest_row, int64_t cols,
    int64_t topk, cudaStream_t stream);

#endif
