// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kM = 32;
constexpr int kPanelM = 16;
constexpr int kN = 128;
constexpr int kK = 256;
constexpr int kNativeN = 16;
constexpr int kNativeK = 16;
constexpr int kBaselineThreads = 512;
constexpr int kCandidateThreads = 256;
constexpr int kBaselineQKWarps = kN / kNativeN;
constexpr int kCandidateQKWarps = kN / kNativeN;
constexpr int kQPanelElements = kPanelM * kK;
constexpr int kQElements = kM * kK;
constexpr int kKeyElementsPerGroup = kN * kK;
constexpr int kOutputElementsPerGroup = kM * kN;

using AFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kPanelM,
                                         kNativeN, kNativeK, __half,
                                         nvcuda::wmma::row_major>;
using BFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kPanelM,
                                         kNativeN, kNativeK, __half,
                                         nvcuda::wmma::col_major>;
using CFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kPanelM,
                                         kNativeN, kNativeK, float>;

struct Args {
  int device = 0;
  int groups = 1024;
  int warmup = 20;
  int rounds = 100;
  int launches_per_sample = 8;
  bool profile_only = false;
  std::string profile_kernel = "both";
};

struct TimingSummary {
  double median_us = 0.0;
  double p90_us = 0.0;
  double mean_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
};

struct PairSummary {
  int count = 0;
  int candidate_faster = 0;
  int baseline_faster = 0;
  int ties = 0;
  double candidate_minus_baseline_median_us = 0.0;
  double candidate_minus_baseline_mean_us = 0.0;
};

struct KernelResources {
  int registers_per_thread = 0;
  size_t static_shared_bytes = 0;
  size_t local_bytes_per_thread = 0;
  int active_ctas_per_sm = 0;
  int resident_total_warps = 0;
  int resident_qk_warps = 0;
  int threads_per_cta = 0;
  int qk_warps_per_cta = 0;
};

struct Exactness {
  bool bitwise_equal = false;
  int mismatch_words = 0;
  uint32_t xor_reduction = 0;
  uint32_t max_word_xor = 0;
  float max_abs_error = 0.0f;
};

void check_cuda(cudaError_t status, const char* expression, const char* file,
                int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << "CUDA failure at " << file << ':' << line << " for "
            << expression << ": " << cudaGetErrorString(status) << '\n';
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ int swizzled_q_row_slot(int row) {
  return (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
}

__device__ __forceinline__ void stage_swizzled_q_panel(
    const __half* __restrict__ query_panel,
    __half* __restrict__ shared_query_panel, int thread, int thread_count) {
  constexpr int kHalfPerUint4 = sizeof(uint4) / sizeof(__half);
  constexpr int kQueryVectors = kQPanelElements / kHalfPerUint4;
  constexpr int kVectorsPerRow = kK / kHalfPerUint4;
  const uint4* query_vectors = reinterpret_cast<const uint4*>(query_panel);
  uint4* shared_vectors = reinterpret_cast<uint4*>(shared_query_panel);

  for (int index = thread; index < kQueryVectors; index += thread_count) {
    const int row = index / kVectorsPerRow;
    const int vector_column = index % kVectorsPerRow;
    const int k_tile = vector_column >> 1;
    const int plane = vector_column & 1;
    const int slot = swizzled_q_row_slot(row);
    shared_vectors[k_tile * (2 * kPanelM) + plane * kPanelM + slot] =
        __ldg(query_vectors + index);
  }
}

__device__ __forceinline__ void load_swizzled_matrix_a_fragment(
    AFragment& fragment, const __half* __restrict__ shared_query_panel,
    int k_offset) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 3) + ((lane >> 4) & 1) * 4 +
                  ((lane >> 2) & 1) * 8;
  const int slot = swizzled_q_row_slot(row);
  const int tile_offset = (k_offset / kNativeK) * kPanelM * kNativeK;
  uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
      shared_query_panel + tile_offset + slot * 8));
  uint32_t* words = reinterpret_cast<uint32_t*>(&fragment);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]),
                 "=r"(words[3])
               : "r"(address)
               : "memory");
  address += kPanelM * 8 * sizeof(__half);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[4]), "=r"(words[5]), "=r"(words[6]),
                 "=r"(words[7])
               : "r"(address)
               : "memory");
}

extern "C" __global__ __launch_bounds__(kBaselineThreads, 2)
void native_bm32_baseline_kernel(const __half* __restrict__ query,
                                 const __half* __restrict__ key,
                                 float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_query[kQPanelElements];

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int m_offset = (block & 1) * kPanelM;
  const int thread = threadIdx.x;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements;
  const __half* key_group = key + static_cast<int64_t>(group) *
                                       kKeyElementsPerGroup;
  stage_swizzled_q_panel(query_group + m_offset * kK, shared_query, thread,
                          kBaselineThreads);
  __syncthreads();

  const int warp = thread >> 5;
  if (warp >= kBaselineQKWarps) {
    return;
  }
  const int n_offset = warp * kNativeN;
  AFragment a_fragment;
  BFragment b_fragment;
  CFragment accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kK; k_offset += kNativeK) {
    load_swizzled_matrix_a_fragment(a_fragment, shared_query, k_offset);
    nvcuda::wmma::load_matrix_sync(
        b_fragment, key_group + n_offset * kK + k_offset, kK);
    nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  }

  float* output_tile = output + static_cast<int64_t>(group) *
                                     kOutputElementsPerGroup +
                       m_offset * kN + n_offset;
  nvcuda::wmma::store_matrix_sync(output_tile, accumulator, kN,
                                  nvcuda::wmma::mem_row_major);
}

extern "C" __global__ __launch_bounds__(kCandidateThreads, 4)
void native_bm32_candidate_kernel(const __half* __restrict__ query,
                                  const __half* __restrict__ key,
                                  float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_query[2 * kQPanelElements];

  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int n_offset = warp * kNativeN;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements;
  const __half* key_group = key + static_cast<int64_t>(group) *
                                       kKeyElementsPerGroup;
  stage_swizzled_q_panel(query_group, shared_query, thread, kCandidateThreads);
  stage_swizzled_q_panel(query_group + kQPanelElements,
                          shared_query + kQPanelElements, thread,
                          kCandidateThreads);
  __syncthreads();

  AFragment top_a_fragment;
  AFragment bottom_a_fragment;
  BFragment b_fragment;
  CFragment top_accumulator;
  CFragment bottom_accumulator;
  nvcuda::wmma::fill_fragment(top_accumulator, 0.0f);
  nvcuda::wmma::fill_fragment(bottom_accumulator, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kK; k_offset += kNativeK) {
    load_swizzled_matrix_a_fragment(top_a_fragment, shared_query, k_offset);
    load_swizzled_matrix_a_fragment(bottom_a_fragment,
                                    shared_query + kQPanelElements, k_offset);
    nvcuda::wmma::load_matrix_sync(
        b_fragment, key_group + n_offset * kK + k_offset, kK);
    nvcuda::wmma::mma_sync(top_accumulator, top_a_fragment, b_fragment,
                           top_accumulator);
    nvcuda::wmma::mma_sync(bottom_accumulator, bottom_a_fragment, b_fragment,
                           bottom_accumulator);
  }

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
  nvcuda::wmma::store_matrix_sync(output_group + n_offset, top_accumulator,
                                  kN, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync(output_group + kPanelM * kN + n_offset,
                                  bottom_accumulator, kN,
                                  nvcuda::wmma::mem_row_major);
}

uint32_t next_random(uint32_t* state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

float random_half_value(uint32_t* state) {
  const int value = static_cast<int>(next_random(state) & 0x7ffu) - 1024;
  return static_cast<float>(value) / 512.0f;
}

uint32_t float_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

TimingSummary summarize(const std::vector<double>& samples) {
  std::vector<double> ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  const size_t middle = ordered.size() / 2;
  const double median = ordered.size() % 2 == 0
                            ? (ordered[middle - 1] + ordered[middle]) / 2.0
                            : ordered[middle];
  return {median,
          ordered[static_cast<size_t>(0.9 * (ordered.size() - 1))],
          sum / static_cast<double>(samples.size()), ordered.front(),
          ordered.back()};
}

PairSummary summarize_pairs(const std::vector<double>& baseline,
                            const std::vector<double>& candidate) {
  std::vector<double> deltas;
  deltas.reserve(baseline.size());
  PairSummary result;
  result.count = static_cast<int>(baseline.size());
  for (size_t index = 0; index < baseline.size(); ++index) {
    const double delta = candidate[index] - baseline[index];
    deltas.push_back(delta);
    if (candidate[index] < baseline[index]) {
      ++result.candidate_faster;
    } else if (baseline[index] < candidate[index]) {
      ++result.baseline_faster;
    } else {
      ++result.ties;
    }
  }
  const TimingSummary summary = summarize(deltas);
  result.candidate_minus_baseline_median_us = summary.median_us;
  result.candidate_minus_baseline_mean_us = summary.mean_us;
  return result;
}

template <typename Kernel>
KernelResources query_resources(Kernel kernel, int threads_per_cta,
                                int qk_warps_per_cta, int device) {
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int active_ctas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_ctas, kernel, threads_per_cta, 0));
  KernelResources result;
  result.registers_per_thread = attributes.numRegs;
  result.static_shared_bytes = attributes.sharedSizeBytes;
  result.local_bytes_per_thread = attributes.localSizeBytes;
  result.active_ctas_per_sm = active_ctas;
  result.threads_per_cta = threads_per_cta;
  result.qk_warps_per_cta = qk_warps_per_cta;
  result.resident_total_warps = active_ctas * (threads_per_cta / 32);
  result.resident_qk_warps = active_ctas * qk_warps_per_cta;
  return result;
}

void print_json_string(const std::string& value) {
  std::cout << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '\\':
        std::cout << "\\\\";
        break;
      case '"':
        std::cout << "\\\"";
        break;
      case '\n':
        std::cout << "\\n";
        break;
      case '\r':
        std::cout << "\\r";
        break;
      case '\t':
        std::cout << "\\t";
        break;
      default:
        if (character < 0x20) {
          std::cout << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(character) << std::dec << std::setfill(' ');
        } else {
          std::cout << character;
        }
    }
  }
  std::cout << '"';
}

void print_timing(const TimingSummary& timing) {
  std::cout << "{\"median_us\": " << std::setprecision(9)
            << timing.median_us << ", \"p90_us\": " << timing.p90_us
            << ", \"mean_us\": " << timing.mean_us
            << ", \"min_us\": " << timing.min_us
            << ", \"max_us\": " << timing.max_us << '}';
}

void print_resources(const KernelResources& resources) {
  std::cout << "{\"registers_per_thread\": "
            << resources.registers_per_thread
            << ", \"static_shared_bytes\": "
            << resources.static_shared_bytes
            << ", \"local_bytes_per_thread\": "
            << resources.local_bytes_per_thread
            << ", \"active_ctas_per_sm\": "
            << resources.active_ctas_per_sm
            << ", \"resident_total_warps\": "
            << resources.resident_total_warps
            << ", \"resident_qk_warps\": "
            << resources.resident_qk_warps
            << ", \"threads_per_cta\": " << resources.threads_per_cta
            << ", \"qk_warps_per_cta\": " << resources.qk_warps_per_cta
            << '}';
}

void print_json(const Args& args, const cudaDeviceProp& properties,
                int runtime_version, int sm_count, const Exactness& exactness,
                const TimingSummary& baseline_timing,
                const TimingSummary& candidate_timing,
                const PairSummary& pairs,
                const KernelResources& baseline_resources,
                const KernelResources& candidate_resources) {
  const double candidate_speedup_pct =
      100.0 * (baseline_timing.median_us - candidate_timing.median_us) /
      baseline_timing.median_us;
  std::cout << "{\n";
  std::cout << "  \"device\": {\n";
  std::cout << "    \"logical_index\": " << args.device << ",\n";
  std::cout << "    \"name\": ";
  print_json_string(properties.name);
  std::cout << ",\n";
  std::cout << "    \"capability\": [" << properties.major << ", "
            << properties.minor << "],\n";
  std::cout << "    \"cuda_runtime\": " << runtime_version << ",\n";
  std::cout << "    \"sm_count\": " << sm_count << "\n";
  std::cout << "  },\n";
  std::cout << "  \"target\": \"sm_70\",\n";
  std::cout << "  \"shape\": {\n";
  std::cout << "    \"groups\": " << args.groups << ",\n";
  std::cout << "    \"output\": \"[groups, M32, N128]\",\n";
  std::cout << "    \"K\": 256,\n";
  std::cout << "    \"query_layout\": \"[group, M32, K256] token-major\",\n";
  std::cout << "    \"key_layout\": \"[group, N128, K256] token-major stride 256\"\n";
  std::cout << "  },\n";
  std::cout << "  \"paths\": {\n";
  std::cout << "    \"baseline\": \"2 CTA/group, BM16xBN128, 512 threads; first 8 warps native WMMA\",\n";
  std::cout << "    \"candidate\": \"1 CTA/group, BM32xBN128, 256 threads; 8 warps native WMMA B reuse\",\n";
  std::cout << "    \"baseline_ctas_per_group\": 2,\n";
  std::cout << "    \"candidate_ctas_per_group\": 1,\n";
  std::cout << "    \"candidate_threads\": 256,\n";
  std::cout << "    \"candidate_qk_warps\": 8,\n";
  std::cout << "    \"candidate_b_fragment_loads_per_k16\": 1,\n";
  std::cout << "    \"candidate_mma_order\": \"top then bottom\",\n";
  std::cout << "    \"shared_q\": \"accepted-path swizzled uint4 staging\"\n";
  std::cout << "  },\n";
  std::cout << "  \"exactness\": {\n";
  std::cout << "    \"word_dtype\": \"uint32\",\n";
  std::cout << "    \"word_count\": "
            << static_cast<int64_t>(args.groups) * kOutputElementsPerGroup
            << ",\n";
  std::cout << "    \"full_32x128\": true,\n";
  std::cout << "    \"bitwise_equal\": "
            << (exactness.bitwise_equal ? "true" : "false") << ",\n";
  std::cout << "    \"mismatch_words\": " << exactness.mismatch_words
            << ",\n";
  std::cout << "    \"xor\": {\"max_word\": "
            << exactness.max_word_xor << ", \"reduction\": "
            << exactness.xor_reduction << "},\n";
  std::cout << "    \"max_abs_error\": " << exactness.max_abs_error << "\n";
  std::cout << "  },\n";
  std::cout << "  \"timing\": {\n";
  std::cout << "    \"unit\": \"us per grid launch\",\n";
  std::cout << "    \"baseline\": ";
  print_timing(baseline_timing);
  std::cout << ",\n    \"candidate\": ";
  print_timing(candidate_timing);
  std::cout << ",\n    \"candidate_speedup_vs_baseline_pct\": "
            << candidate_speedup_pct << "\n";
  std::cout << "  },\n";
  std::cout << "  \"pairs\": {\n";
  std::cout << "    \"count\": " << pairs.count << ",\n";
  std::cout << "    \"candidate_faster\": " << pairs.candidate_faster << ",\n";
  std::cout << "    \"baseline_faster\": " << pairs.baseline_faster
            << ",\n";
  std::cout << "    \"ties\": " << pairs.ties << ",\n";
  std::cout << "    \"candidate_minus_baseline_median_us\": "
            << pairs.candidate_minus_baseline_median_us << ",\n";
  std::cout << "    \"candidate_minus_baseline_mean_us\": "
            << pairs.candidate_minus_baseline_mean_us << "\n";
  std::cout << "  },\n";
  std::cout << "  \"measurement\": {\n";
  std::cout << "    \"warmup_pairs\": " << args.warmup << ",\n";
  std::cout << "    \"rounds\": " << args.rounds << ",\n";
  std::cout << "    \"launches_per_sample\": " << args.launches_per_sample
            << ",\n";
  std::cout << "    \"interleaving\": \"baseline/candidate order alternates every round\"\n";
  std::cout << "  },\n";
  std::cout << "  \"resources\": {\n";
  std::cout << "    \"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ",\n    \"candidate\": ";
  print_resources(candidate_resources);
  std::cout << "\n  }\n";
  std::cout << "}\n";
}

bool profile_kernel_selected(const Args& args, const char* kernel) {
  return args.profile_kernel == "both" || args.profile_kernel == kernel;
}

int run(const Args& args) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (args.device < 0 || args.device >= device_count) {
    std::cerr << "Requested logical CUDA device " << args.device
              << " is unavailable; visible device count is " << device_count << '\n';
    return EXIT_FAILURE;
  }
  if (args.groups < 1 || args.warmup < 0 || args.rounds < 1 ||
      args.launches_per_sample < 1) {
    std::cerr << "groups, rounds, and launches-per-sample must be positive; "
                 "warmup cannot be negative\n";
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "This probe requires SM70, got " << properties.major << '.'
              << properties.minor << '\n';
    return EXIT_FAILURE;
  }

  int runtime_version = 0;
  int sm_count = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    args.device));

  const size_t query_elements = static_cast<size_t>(args.groups) * kQElements;
  const size_t key_elements =
      static_cast<size_t>(args.groups) * kKeyElementsPerGroup;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kOutputElementsPerGroup;
  std::vector<__half> host_query(query_elements);
  std::vector<__half> host_key(key_elements);
  uint32_t random_state = 0x6d2b79f5u;
  for (__half& value : host_query) {
    value = __float2half_rn(random_half_value(&random_state));
  }
  for (__half& value : host_key) {
    value = __float2half_rn(random_half_value(&random_state));
  }

  __half* device_query = nullptr;
  __half* device_key = nullptr;
  float* device_baseline = nullptr;
  float* device_candidate = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_query),
                        query_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_key),
                        key_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_baseline),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_candidate),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_query, host_query.data(),
                        query_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_key, host_key.data(),
                        key_elements * sizeof(__half), cudaMemcpyHostToDevice));

  const dim3 baseline_grid(args.groups * 2);
  const dim3 candidate_grid(args.groups);
  auto launch_baseline = [&] {
    native_bm32_baseline_kernel<<<baseline_grid, kBaselineThreads>>>(
        device_query, device_key, device_baseline, args.groups);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_candidate = [&] {
    native_bm32_candidate_kernel<<<candidate_grid, kCandidateThreads>>>(
        device_query, device_key, device_candidate, args.groups);
    CUDA_CHECK(cudaGetLastError());
  };

  if (args.profile_only) {
    if (profile_kernel_selected(args, "baseline")) {
      launch_baseline();
    }
    if (profile_kernel_selected(args, "candidate")) {
      launch_candidate();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(device_candidate));
    CUDA_CHECK(cudaFree(device_baseline));
    CUDA_CHECK(cudaFree(device_key));
    CUDA_CHECK(cudaFree(device_query));
    return EXIT_SUCCESS;
  }

  for (int warmup = 0; warmup < args.warmup; ++warmup) {
    launch_baseline();
    launch_candidate();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  launch_baseline();
  launch_candidate();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> host_baseline(output_elements);
  std::vector<float> host_candidate(output_elements);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), device_baseline,
                        output_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_candidate.data(), device_candidate,
                        output_elements * sizeof(float), cudaMemcpyDeviceToHost));

  Exactness exactness;
  exactness.bitwise_equal = true;
  for (size_t index = 0; index < output_elements; ++index) {
    const uint32_t word_xor =
        float_bits(host_baseline[index]) ^ float_bits(host_candidate[index]);
    exactness.xor_reduction ^= word_xor;
    exactness.max_word_xor = std::max(exactness.max_word_xor, word_xor);
    exactness.bitwise_equal &= word_xor == 0;
    exactness.mismatch_words += word_xor != 0;
    const float abs_error =
        std::fabs(host_baseline[index] - host_candidate[index]);
    exactness.max_abs_error = std::max(exactness.max_abs_error, abs_error);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto time_launches = [&](bool candidate) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch = 0; launch < args.launches_per_sample; ++launch) {
      if (candidate) {
        launch_candidate();
      } else {
        launch_baseline();
      }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms) * 1000.0 /
           args.launches_per_sample;
  };

  std::vector<double> baseline_samples;
  std::vector<double> candidate_samples;
  baseline_samples.reserve(args.rounds);
  candidate_samples.reserve(args.rounds);
  for (int round = 0; round < args.rounds; ++round) {
    double baseline_us = 0.0;
    double candidate_us = 0.0;
    if ((round & 1) == 0) {
      baseline_us = time_launches(false);
      candidate_us = time_launches(true);
    } else {
      candidate_us = time_launches(true);
      baseline_us = time_launches(false);
    }
    baseline_samples.push_back(baseline_us);
    candidate_samples.push_back(candidate_us);
  }
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  const KernelResources baseline_resources = query_resources(
      native_bm32_baseline_kernel, kBaselineThreads, kBaselineQKWarps,
      args.device);
  const KernelResources candidate_resources = query_resources(
      native_bm32_candidate_kernel, kCandidateThreads, kCandidateQKWarps,
      args.device);
  const TimingSummary baseline_timing = summarize(baseline_samples);
  const TimingSummary candidate_timing = summarize(candidate_samples);
  const PairSummary pairs = summarize_pairs(baseline_samples, candidate_samples);

  CUDA_CHECK(cudaFree(device_candidate));
  CUDA_CHECK(cudaFree(device_baseline));
  CUDA_CHECK(cudaFree(device_key));
  CUDA_CHECK(cudaFree(device_query));
  print_json(args, properties, runtime_version, sm_count, exactness,
             baseline_timing, candidate_timing, pairs, baseline_resources,
             candidate_resources);
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    auto parse_int = [&](int* destination) {
      if (index + 1 >= argc) {
        std::cerr << "Missing value for " << argument << '\n';
        std::exit(EXIT_FAILURE);
      }
      *destination = std::stoi(argv[++index]);
    };
    if (argument == "--device") {
      parse_int(&args.device);
    } else if (argument == "--groups") {
      parse_int(&args.groups);
    } else if (argument == "--warmup") {
      parse_int(&args.warmup);
    } else if (argument == "--rounds") {
      parse_int(&args.rounds);
    } else if (argument == "--launches-per-sample") {
      parse_int(&args.launches_per_sample);
    } else if (argument == "--profile-only") {
      args.profile_only = true;
    } else if (argument == "--profile-kernel" && index + 1 < argc) {
      args.profile_kernel = argv[++index];
      if (args.profile_kernel != "baseline" &&
          args.profile_kernel != "candidate" &&
          args.profile_kernel != "both") {
        std::cerr << "--profile-kernel must be baseline, candidate, or both\n";
        return EXIT_FAILURE;
      }
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--device N] [--groups N] [--warmup N] [--rounds N]"
                   " [--launches-per-sample N] [--profile-only]"
                   " [--profile-kernel baseline|candidate|both]\n";
      return EXIT_FAILURE;
    }
  }
  return run(args);
}
