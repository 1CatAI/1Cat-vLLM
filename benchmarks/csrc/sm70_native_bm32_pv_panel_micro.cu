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
constexpr int kD = 256;
constexpr int kK = 32;
constexpr int kTileM = 16;
constexpr int kTileD = 16;
constexpr int kTileK = 16;
constexpr int kBaselineThreads = 512;
constexpr int kCandidateThreads = 256;
constexpr int kPElementsPerGroup = kM * kK;
constexpr int kBaselinePElements = kTileM * kK;
constexpr int kVElementsPerGroup = kK * kD;
constexpr int kOutputElementsPerGroup = kM * kD;

static_assert(kD / kTileD == 16);
static_assert(kK / kTileK == 2);

using AFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kTileM, kTileD, kTileK,
                          __half, nvcuda::wmma::row_major>;
using BFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTileM, kTileD, kTileK,
                          __half, nvcuda::wmma::row_major>;
using CFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kTileM,
                                          kTileD, kTileK, float>;

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
  int resident_active_warps = 0;
  int threads_per_cta = 0;
  int warps_per_cta = 0;
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

#define CUDA_CHECK(expression) \
  check_cuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ void stage_p_to_shared(
    const __half* __restrict__ source, __half* __restrict__ destination,
    int element_count, int thread_count) {
  constexpr int kHalfPerUint4 = sizeof(uint4) / sizeof(__half);
  const int vector_count = element_count / kHalfPerUint4;
  const uint4* source_vectors = reinterpret_cast<const uint4*>(source);
  uint4* destination_vectors = reinterpret_cast<uint4*>(destination);
  for (int index = threadIdx.x; index < vector_count; index += thread_count) {
    destination_vectors[index] = __ldg(source_vectors + index);
  }
}

extern "C" __global__ __launch_bounds__(kBaselineThreads, 2)
void native_bm32_pv_baseline_kernel(const __half* __restrict__ p,
                                    const __half* __restrict__ v,
                                    float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_p[kBaselinePElements];

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int m_offset = (block & 1) * kTileM;
  const __half* p_tile = p + static_cast<int64_t>(group) *
                                 kPElementsPerGroup +
                         m_offset * kK;
  stage_p_to_shared(p_tile, shared_p, kBaselinePElements,
                    kBaselineThreads);
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int d_offset = warp * kTileD;
  const __half* v_group =
      v + static_cast<int64_t>(group) * kVElementsPerGroup;
  AFragment a_fragment;
  BFragment b_fragment;
  CFragment accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p, kK);
  nvcuda::wmma::load_matrix_sync(b_fragment, v_group + d_offset, kD);
  nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p + kTileK, kK);
  nvcuda::wmma::load_matrix_sync(
      b_fragment, v_group + kTileK * kD + d_offset, kD);
  nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);

  float* output_tile = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup +
                       m_offset * kD + d_offset;
  nvcuda::wmma::store_matrix_sync(output_tile, accumulator, kD,
                                  nvcuda::wmma::mem_row_major);
}

extern "C" __global__ __launch_bounds__(kCandidateThreads, 4)
void native_bm32_pv_candidate_kernel(const __half* __restrict__ p,
                                     const __half* __restrict__ v,
                                     unsigned int groups) {
  __shared__ __align__(16) __half shared_p[kPElementsPerGroup];

  const unsigned int group = blockIdx.x;
  const __half* p_group =
      p + static_cast<uint64_t>(group) * kPElementsPerGroup;
  stage_p_to_shared(p_group, shared_p, kPElementsPerGroup,
                    kCandidateThreads);
  __syncthreads();

  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int d0 = warp * (2 * kTileD);
  const __half* v_group =
      v + static_cast<uint64_t>(group) * kVElementsPerGroup;
  AFragment a_fragment;
  BFragment b_fragment;
  CFragment accumulator_d0_top;
  CFragment accumulator_d0_bottom;
  CFragment accumulator_d1_top;
  CFragment accumulator_d1_bottom;
  nvcuda::wmma::fill_fragment(accumulator_d0_top, 0.0f);
  nvcuda::wmma::fill_fragment(accumulator_d0_bottom, 0.0f);
  nvcuda::wmma::fill_fragment(accumulator_d1_top, 0.0f);
  nvcuda::wmma::fill_fragment(accumulator_d1_bottom, 0.0f);

  const unsigned int d1_early = d0 + kTileD;
  nvcuda::wmma::load_matrix_sync(b_fragment, v_group + d0, kD);
  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p, kK);
  nvcuda::wmma::mma_sync(accumulator_d0_top, a_fragment, b_fragment,
                         accumulator_d0_top);
  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p + kTileM * kK, kK);
  nvcuda::wmma::mma_sync(accumulator_d0_bottom, a_fragment, b_fragment,
                         accumulator_d0_bottom);
  nvcuda::wmma::load_matrix_sync(b_fragment, v_group + d1_early, kD);
  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p, kK);
  nvcuda::wmma::mma_sync(accumulator_d1_top, a_fragment, b_fragment,
                         accumulator_d1_top);
  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p + kTileM * kK, kK);
  nvcuda::wmma::mma_sync(accumulator_d1_bottom, a_fragment, b_fragment,
                         accumulator_d1_bottom);

  nvcuda::wmma::load_matrix_sync(b_fragment,
                                 v_group + kTileK * kD + d0, kD);
  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p + kTileK, kK);
  nvcuda::wmma::mma_sync(accumulator_d0_top, a_fragment, b_fragment,
                         accumulator_d0_top);
  nvcuda::wmma::load_matrix_sync(
      a_fragment, shared_p + kTileM * kK + kTileK, kK);
  nvcuda::wmma::mma_sync(accumulator_d0_bottom, a_fragment, b_fragment,
                         accumulator_d0_bottom);

  float* output = reinterpret_cast<float*>(
      const_cast<__half*>(v) + static_cast<uint64_t>(groups) *
                                   kVElementsPerGroup);
  float* output_group = output + static_cast<uint64_t>(group) *
                                     kOutputElementsPerGroup;
  nvcuda::wmma::store_matrix_sync(output_group + d0, accumulator_d0_top, kD,
                                  nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync(output_group + kTileM * kD + d0,
                                  accumulator_d0_bottom, kD,
                                  nvcuda::wmma::mem_row_major);

  unsigned int late_tid;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(late_tid));
  const unsigned int warp_base = __shfl_sync(0xffffffffU, late_tid, 0);
  const unsigned int d1_late = warp_base + kTileD;
  nvcuda::wmma::load_matrix_sync(
      b_fragment, v_group + kTileK * kD + d1_late, kD);
  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p + kTileK, kK);
  nvcuda::wmma::mma_sync(accumulator_d1_top, a_fragment, b_fragment,
                         accumulator_d1_top);
  nvcuda::wmma::load_matrix_sync(
      a_fragment, shared_p + kTileM * kK + kTileK, kK);
  nvcuda::wmma::mma_sync(accumulator_d1_bottom, a_fragment, b_fragment,
                         accumulator_d1_bottom);
  nvcuda::wmma::store_matrix_sync(output_group + d1_late,
                                  accumulator_d1_top, kD,
                                  nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync(output_group + kTileM * kD + d1_late,
                                  accumulator_d1_bottom, kD,
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

Exactness compare_outputs(const std::vector<float>& baseline,
                          const std::vector<float>& candidate) {
  Exactness result;
  result.bitwise_equal = true;
  for (size_t index = 0; index < baseline.size(); ++index) {
    const uint32_t word_xor =
        float_bits(baseline[index]) ^ float_bits(candidate[index]);
    result.xor_reduction ^= word_xor;
    result.max_word_xor = std::max(result.max_word_xor, word_xor);
    result.bitwise_equal &= word_xor == 0;
    result.mismatch_words += word_xor != 0;
    result.max_abs_error =
        std::max(result.max_abs_error,
                 std::fabs(baseline[index] - candidate[index]));
  }
  return result;
}

template <typename Kernel>
KernelResources query_resources(Kernel kernel, int threads_per_cta) {
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
  result.warps_per_cta = threads_per_cta / 32;
  result.resident_active_warps = active_ctas * result.warps_per_cta;
  return result;
}

void print_json_string(const std::string& value) {
  std::cout << '"';
  for (const unsigned char character : value) {
    if (character == '\\') {
      std::cout << "\\\\";
    } else if (character == '"') {
      std::cout << "\\\"";
    } else if (character == '\n') {
      std::cout << "\\n";
    } else if (character < 0x20) {
      std::cout << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(character) << std::dec << std::setfill(' ');
    } else {
      std::cout << character;
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

void print_pairs(const PairSummary& pairs) {
  std::cout << "{\"count\": " << pairs.count
            << ", \"candidate_faster\": " << pairs.candidate_faster
            << ", \"baseline_faster\": " << pairs.baseline_faster
            << ", \"ties\": " << pairs.ties
            << ", \"candidate_minus_baseline_median_us\": "
            << pairs.candidate_minus_baseline_median_us
            << ", \"candidate_minus_baseline_mean_us\": "
            << pairs.candidate_minus_baseline_mean_us << '}';
}

void print_exactness(const Exactness& exactness) {
  std::cout << "{\"bitwise_equal\": "
            << (exactness.bitwise_equal ? "true" : "false")
            << ", \"mismatch_words\": " << exactness.mismatch_words
            << ", \"xor\": {\"max_word\": " << exactness.max_word_xor
            << ", \"reduction\": " << exactness.xor_reduction << '}'
            << ", \"max_abs_error\": " << exactness.max_abs_error << '}';
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
            << ", \"resident_active_warps\": "
            << resources.resident_active_warps
            << ", \"threads_per_cta\": " << resources.threads_per_cta
            << ", \"warps_per_cta\": " << resources.warps_per_cta << '}';
}

void print_json(const Args& args, const cudaDeviceProp& properties,
                int runtime_version, int sm_count, int max_threads_per_sm,
                const Exactness& exactness,
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
  std::cout << "    \"sm_count\": " << sm_count << ",\n";
  std::cout << "    \"max_threads_per_sm\": " << max_threads_per_sm << "\n";
  std::cout << "  },\n";
  std::cout << "  \"target\": \"sm_70\",\n";
  std::cout << "  \"shape\": {\"groups\": " << args.groups
            << ", \"output\": \"[groups, M32, D256]\", \"K\": 32},\n";
  std::cout << "  \"layouts\": {\"p\": \"[group, M32, K32] row-major\", "
               "\"v\": \"[group, K32, D256] row-major stride 256\"},\n";
  std::cout << "  \"paths\": {\n";
  std::cout << "    \"baseline\": \"2 CTA/group, BM16xD256, 512 threads, 16 warps x D16; two K16 native row.row\",\n";
  std::cout << "    \"candidate\": \"1 CTA/group, BM32xD256, 256 threads, 8 warps x two D16; four accumulators and one B load per D tile/K16 reused top then bottom\"\n";
  std::cout << "  },\n";
  std::cout << "  \"launch_topology\": {\n";
  std::cout << "    \"baseline\": {\"ctas_per_group\": 2, \"threads_per_cta\": 512, \"warps_per_cta\": 16},\n";
  std::cout << "    \"candidate\": {\"ctas_per_group\": 1, \"threads_per_cta\": 256, \"warps_per_cta\": 8}\n";
  std::cout << "  },\n";
  std::cout << "  \"exactness\": {\"word_dtype\": \"uint32\", "
               "\"word_count\": "
            << static_cast<int64_t>(args.groups) * kOutputElementsPerGroup
            << ", \"baseline_vs_candidate\": ";
  print_exactness(exactness);
  std::cout << "},\n";
  std::cout << "  \"timing\": {\"unit\": \"us per grid launch\", "
               "\"baseline\": ";
  print_timing(baseline_timing);
  std::cout << ", \"candidate\": ";
  print_timing(candidate_timing);
  std::cout << ", \"candidate_speedup_vs_baseline_pct\": "
            << candidate_speedup_pct << "},\n";
  std::cout << "  \"pairs\": ";
  print_pairs(pairs);
  std::cout << ",\n";
  std::cout << "  \"measurement\": {\"warmup_pairs\": " << args.warmup
            << ", \"rounds\": " << args.rounds
            << ", \"launches_per_sample\": " << args.launches_per_sample
            << ", \"interleaving\": \"baseline/candidate order alternates each round\"},\n";
  std::cout << "  \"resources\": {\"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ", \"candidate\": ";
  print_resources(candidate_resources);
  std::cout << "}\n";
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
              << " is unavailable; visible device count is " << device_count
              << '\n';
    return EXIT_FAILURE;
  }
  if (args.groups < 1 || args.warmup < 0 || args.rounds < 1 ||
      args.launches_per_sample < 1) {
    std::cerr << "Invalid non-positive benchmark argument\n";
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "This microbenchmark requires SM70, got " << properties.major
              << '.' << properties.minor << '\n';
    return EXIT_FAILURE;
  }

  int runtime_version = 0;
  int sm_count = 0;
  int max_threads_per_sm = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    args.device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_sm,
                                    cudaDevAttrMaxThreadsPerMultiProcessor,
                                    args.device));

  const size_t p_elements =
      static_cast<size_t>(args.groups) * kPElementsPerGroup;
  const size_t v_elements =
      static_cast<size_t>(args.groups) * kVElementsPerGroup;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kOutputElementsPerGroup;
  std::vector<__half> host_p(p_elements);
  std::vector<__half> host_v(v_elements);
  uint32_t random_state = 0x6d2b79f5u;
  for (__half& value : host_p) {
    value = __float2half_rn(random_half_value(&random_state));
  }
  for (__half& value : host_v) {
    value = __float2half_rn(random_half_value(&random_state));
  }

  __half* device_p = nullptr;
  __half* device_v = nullptr;
  float* device_baseline = nullptr;
  float* device_candidate = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_p),
                        p_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_v),
                        v_elements * sizeof(__half) +
                            output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_baseline),
                        output_elements * sizeof(float)));
  device_candidate = reinterpret_cast<float*>(device_v + v_elements);
  CUDA_CHECK(cudaMemcpy(device_p, host_p.data(), p_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_v, host_v.data(), v_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));

  const dim3 baseline_grid(args.groups * 2);
  const dim3 candidate_grid(args.groups);
  auto launch_baseline = [&] {
    native_bm32_pv_baseline_kernel<<<baseline_grid, kBaselineThreads>>>(
        device_p, device_v, device_baseline, args.groups);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_candidate = [&] {
    native_bm32_pv_candidate_kernel<<<candidate_grid, kCandidateThreads>>>(
        device_p, device_v, args.groups);
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
    CUDA_CHECK(cudaFree(device_baseline));
    CUDA_CHECK(cudaFree(device_v));
    CUDA_CHECK(cudaFree(device_p));
    return EXIT_SUCCESS;
  }

  for (int warmup = 0; warmup < args.warmup; ++warmup) {
    if ((warmup & 1) == 0) {
      launch_baseline();
      launch_candidate();
    } else {
      launch_candidate();
      launch_baseline();
    }
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
  const Exactness exactness =
      compare_outputs(host_baseline, host_candidate);

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
    double baseline_us;
    double candidate_us;
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

  const KernelResources baseline_resources =
      query_resources(native_bm32_pv_baseline_kernel, kBaselineThreads);
  const KernelResources candidate_resources =
      query_resources(native_bm32_pv_candidate_kernel, kCandidateThreads);
  const TimingSummary baseline_timing = summarize(baseline_samples);
  const TimingSummary candidate_timing = summarize(candidate_samples);
  const PairSummary pairs =
      summarize_pairs(baseline_samples, candidate_samples);

  CUDA_CHECK(cudaFree(device_baseline));
  CUDA_CHECK(cudaFree(device_v));
  CUDA_CHECK(cudaFree(device_p));
  print_json(args, properties, runtime_version, sm_count, max_threads_per_sm,
             exactness, baseline_timing, candidate_timing, pairs,
             baseline_resources, candidate_resources);
  return EXIT_SUCCESS;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--device" && index + 1 < argc) {
      args.device = std::stoi(argv[++index]);
    } else if (argument == "--groups" && index + 1 < argc) {
      args.groups = std::stoi(argv[++index]);
    } else if (argument == "--warmup" && index + 1 < argc) {
      args.warmup = std::stoi(argv[++index]);
    } else if (argument == "--rounds" && index + 1 < argc) {
      args.rounds = std::stoi(argv[++index]);
    } else if (argument == "--launches-per-sample" && index + 1 < argc) {
      args.launches_per_sample = std::stoi(argv[++index]);
    } else if (argument == "--profile-only") {
      args.profile_only = true;
    } else if (argument == "--profile-kernel" && index + 1 < argc) {
      args.profile_kernel = argv[++index];
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--device N] [--groups N] [--warmup N] [--rounds N]"
                << " [--launches-per-sample N] [--profile-only]"
                << " [--profile-kernel baseline|candidate|both]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  return run(parse_args(argc, argv));
}
