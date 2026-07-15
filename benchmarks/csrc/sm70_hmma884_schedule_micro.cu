// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Isolated SM70 WMMA scheduling target.  The Python harness builds this file
// both as an executable and as a standalone cubin.  The executable launches
// the native kernel as the baseline and a patched copy of the cubin through
// the CUDA driver API as the candidate.

#include <cuda.h>
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
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 16;
constexpr int kTileElements = kTile * kTile;
constexpr int kKTileCount = 2;
constexpr int kWarpsPerBlock = 4;
constexpr int kWarpThreads = 32;
constexpr int kThreads = kWarpsPerBlock * kWarpThreads;

using AFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kTile, kTile,
                                         kTile, __half,
                                         nvcuda::wmma::row_major>;
using BFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTile, kTile,
                                         kTile, __half,
                                         nvcuda::wmma::row_major>;
using CFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kTile,
                                         kTile, kTile, float>;

struct Args {
  int device = 0;
  int groups = 16384;
  int warmup = 24;
  int rounds = 81;
  int launches_per_sample = 8;
  std::string candidate_cubin;
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

struct Exactness {
  bool bitwise_equal = false;
  int mismatch_words = 0;
  uint32_t xor_reduction = 0;
  uint32_t max_word_xor = 0;
};

struct KernelResources {
  int registers_per_thread = 0;
  int static_shared_bytes = 0;
  int local_bytes_per_thread = 0;
  int max_threads_per_block = 0;
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

void check_driver(CUresult status, const char* expression, const char* file,
                  int line) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* detail = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &detail);
  std::cerr << "CUDA driver failure at " << file << ':' << line << " for "
            << expression << ": " << (name == nullptr ? "unknown" : name)
            << " (" << (detail == nullptr ? "unknown" : detail) << ")\n";
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
  check_cuda((expression), #expression, __FILE__, __LINE__)
#define DRIVER_CHECK(expression) \
  check_driver((expression), #expression, __FILE__, __LINE__)

// The single BFragment is deliberately reused.  In native SASS, the second
// load overwrites the B0 register bundle only after the first 16 HMMA.884
// instructions.  A patch may move an individual B1 LDG only after the target
// B0 sub-register's final HMMA use.
extern "C" __global__ __launch_bounds__(kThreads)
void hmma884_patch_target_kernel(const __half* __restrict__ a,
                                 const __half* __restrict__ b,
                                 float* __restrict__ output, int groups) {
  const int group = static_cast<int>(blockIdx.x) * kWarpsPerBlock +
                    static_cast<int>(threadIdx.x) / kWarpThreads;
  if (group >= groups) {
    return;
  }

  const __half* a_group = a + static_cast<int64_t>(group) *
                                   kKTileCount * kTileElements;
  const __half* b_group = b + static_cast<int64_t>(group) *
                                   kKTileCount * kTileElements;
  float* output_group = output + static_cast<int64_t>(group) * kTileElements;

  AFragment a_fragment;
  BFragment b_fragment;
  CFragment accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

  nvcuda::wmma::load_matrix_sync(a_fragment, a_group, kTile);
  nvcuda::wmma::load_matrix_sync(b_fragment, b_group, kTile);
  nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);

  // Keep this B1 load source-ordered after the first opaque WMMA.  The cubin
  // candidate reorders selected existing LDG instructions, never arithmetic.
  nvcuda::wmma::load_matrix_sync(b_fragment, b_group + kTileElements, kTile);
  nvcuda::wmma::load_matrix_sync(a_fragment, a_group + kTileElements, kTile);
  nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  nvcuda::wmma::store_matrix_sync(output_group, accumulator, kTile,
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

float next_half_value(uint32_t* state) {
  const int value = static_cast<int>(next_random(state) & 0x7ffu) - 1024;
  return static_cast<float>(value) / 257.0f;
}

uint32_t float_bits(float value) {
  uint32_t result = 0;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

TimingSummary summarize(std::vector<double> values) {
  TimingSummary result;
  if (values.empty()) {
    return result;
  }
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  result.median_us = values.size() % 2 == 0
                         ? 0.5 * (values[middle - 1] + values[middle])
                         : values[middle];
  const size_t p90 = static_cast<size_t>(std::ceil(0.9 * values.size())) - 1;
  result.p90_us = values[std::min(p90, values.size() - 1)];
  result.mean_us = 0.0;
  for (double value : values) {
    result.mean_us += value;
  }
  result.mean_us /= values.size();
  result.min_us = values.front();
  result.max_us = values.back();
  return result;
}

PairSummary summarize_pairs(const std::vector<double>& baseline,
                            const std::vector<double>& candidate) {
  PairSummary result;
  std::vector<double> deltas;
  deltas.reserve(baseline.size());
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
    result.bitwise_equal &= word_xor == 0;
    result.mismatch_words += word_xor != 0;
    result.xor_reduction ^= word_xor;
    result.max_word_xor = std::max(result.max_word_xor, word_xor);
  }
  return result;
}

KernelResources native_resources() {
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, hmma884_patch_target_kernel));
  KernelResources result;
  result.registers_per_thread = attributes.numRegs;
  result.static_shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
  result.local_bytes_per_thread = static_cast<int>(attributes.localSizeBytes);
  result.max_threads_per_block = attributes.maxThreadsPerBlock;
  return result;
}

KernelResources driver_resources(CUfunction function) {
  KernelResources result;
  DRIVER_CHECK(cuFuncGetAttribute(&result.registers_per_thread,
                                  CU_FUNC_ATTRIBUTE_NUM_REGS, function));
  DRIVER_CHECK(cuFuncGetAttribute(&result.static_shared_bytes,
                                  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                  function));
  DRIVER_CHECK(cuFuncGetAttribute(&result.local_bytes_per_thread,
                                  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                  function));
  DRIVER_CHECK(cuFuncGetAttribute(&result.max_threads_per_block,
                                  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                  function));
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
  std::cout << "{\"median_us\":" << std::setprecision(9)
            << timing.median_us << ",\"p90_us\":" << timing.p90_us
            << ",\"mean_us\":" << timing.mean_us << ",\"min_us\":"
            << timing.min_us << ",\"max_us\":" << timing.max_us << '}';
}

void print_resources(const KernelResources& resources) {
  std::cout << "{\"registers_per_thread\":"
            << resources.registers_per_thread << ",\"static_shared_bytes\":"
            << resources.static_shared_bytes << ",\"local_bytes_per_thread\":"
            << resources.local_bytes_per_thread
            << ",\"max_threads_per_block\":"
            << resources.max_threads_per_block << '}';
}

void print_result(const Args& args, const cudaDeviceProp& properties,
                  const Exactness& exactness,
                  const TimingSummary& baseline_timing,
                  const TimingSummary& candidate_timing,
                  const PairSummary& pairs,
                  const KernelResources& baseline_resources,
                  const KernelResources& candidate_resources) {
  const double speedup =
      100.0 * (baseline_timing.median_us - candidate_timing.median_us) /
      baseline_timing.median_us;
  std::cout << "{\n";
  std::cout << "\"target\":\"sm_70\",\n";
  std::cout << "\"device\":{\"logical_index\":" << args.device
            << ",\"name\":";
  print_json_string(properties.name);
  std::cout << ",\"capability\":[" << properties.major << ','
            << properties.minor << "]},\n";
  std::cout << "\"shape\":{\"groups\":" << args.groups
            << ",\"a\":[\"groups\",2,16,16],\"b\":[\"groups\",2,16,16],"
               "\"output\":[\"groups\",16,16]},\n";
  std::cout << "\"work\":{\"wmma_m16n16k16_calls_per_warp\":2,"
               "\"hmma884_per_wmma\":16,\"b_fragment_instances\":1,"
               "\"arithmetic\":\"A0*B0 + A1*B1, FP16 inputs/FP32 accumulate\"},\n";
  std::cout << "\"exactness\":{\"word_dtype\":\"uint32(fp32)\","
               "\"word_count\":"
            << static_cast<int64_t>(args.groups) * kTileElements
            << ",\"bitwise_equal\":"
            << (exactness.bitwise_equal ? "true" : "false")
            << ",\"mismatch_words\":" << exactness.mismatch_words
            << ",\"xor_reduction\":" << exactness.xor_reduction
            << ",\"max_word_xor\":" << exactness.max_word_xor << "},\n";
  std::cout << "\"timing\":{\"unit\":\"us per grid launch\","
               "\"baseline\":";
  print_timing(baseline_timing);
  std::cout << ",\"candidate\":";
  print_timing(candidate_timing);
  std::cout << ",\"candidate_speedup_vs_baseline_pct\":" << speedup << "},\n";
  std::cout << "\"pairs\":{\"count\":" << pairs.count
            << ",\"candidate_faster\":" << pairs.candidate_faster
            << ",\"baseline_faster\":" << pairs.baseline_faster
            << ",\"ties\":" << pairs.ties
            << ",\"candidate_minus_baseline_median_us\":"
            << pairs.candidate_minus_baseline_median_us
            << ",\"candidate_minus_baseline_mean_us\":"
            << pairs.candidate_minus_baseline_mean_us << "},\n";
  std::cout << "\"measurement\":{\"warmup_pairs\":" << args.warmup
            << ",\"rounds\":" << args.rounds
            << ",\"launches_per_sample\":" << args.launches_per_sample
            << ",\"interleaving\":"
               "\"baseline/candidate order alternates each round\"},\n";
  std::cout << "\"resources\":{\"baseline\":";
  print_resources(baseline_resources);
  std::cout << ",\"candidate\":";
  print_resources(candidate_resources);
  std::cout << "}\n}" << std::endl;
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
    } else if (argument == "--candidate-cubin" && index + 1 < argc) {
      args.candidate_cubin = argv[++index];
    } else {
      std::cerr << "Usage: " << argv[0]
                << " --candidate-cubin PATH [--device N] [--groups N]"
                   " [--warmup N] [--rounds N] [--launches-per-sample N]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if (args.candidate_cubin.empty() || args.groups < 1 || args.warmup < 0 ||
      args.rounds < 1 || args.launches_per_sample < 1) {
    std::cerr << "A candidate cubin and positive benchmark dimensions are required\n";
    std::exit(EXIT_FAILURE);
  }
  return args;
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
  CUDA_CHECK(cudaSetDevice(args.device));
  CUDA_CHECK(cudaFree(nullptr));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "This microbenchmark requires SM70, got " << properties.major
              << '.' << properties.minor << '\n';
    return EXIT_FAILURE;
  }

  DRIVER_CHECK(cuInit(0));
  CUcontext context = nullptr;
  DRIVER_CHECK(cuCtxGetCurrent(&context));
  if (context == nullptr) {
    std::cerr << "CUDA runtime did not establish a current primary context\n";
    return EXIT_FAILURE;
  }
  CUmodule candidate_module = nullptr;
  CUfunction candidate_kernel = nullptr;
  DRIVER_CHECK(cuModuleLoad(&candidate_module, args.candidate_cubin.c_str()));
  DRIVER_CHECK(cuModuleGetFunction(&candidate_kernel, candidate_module,
                                   "hmma884_patch_target_kernel"));

  const size_t input_elements = static_cast<size_t>(args.groups) *
                                kKTileCount * kTileElements;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kTileElements;
  std::vector<__half> host_a(input_elements);
  std::vector<__half> host_b(input_elements);
  uint32_t random_state = 0x6d2b79f5u;
  for (__half& value : host_a) {
    value = __float2half_rn(next_half_value(&random_state));
  }
  for (__half& value : host_b) {
    value = __float2half_rn(next_half_value(&random_state));
  }

  __half* device_a = nullptr;
  __half* device_b = nullptr;
  float* baseline_output = nullptr;
  float* candidate_output = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a),
                        input_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b),
                        input_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&baseline_output),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&candidate_output),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), input_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), input_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));

  const int blocks = (args.groups + kWarpsPerBlock - 1) / kWarpsPerBlock;
  auto launch_baseline = [&] {
    hmma884_patch_target_kernel<<<blocks, kThreads>>>(
        device_a, device_b, baseline_output, args.groups);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_candidate = [&] {
    CUdeviceptr a_argument = reinterpret_cast<CUdeviceptr>(device_a);
    CUdeviceptr b_argument = reinterpret_cast<CUdeviceptr>(device_b);
    CUdeviceptr output_argument =
        reinterpret_cast<CUdeviceptr>(candidate_output);
    int groups_argument = args.groups;
    void* parameters[] = {&a_argument, &b_argument, &output_argument,
                          &groups_argument};
    DRIVER_CHECK(cuLaunchKernel(candidate_kernel, blocks, 1, 1, kThreads, 1, 1,
                                0, nullptr, parameters, nullptr));
  };

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
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), baseline_output,
                        output_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_candidate.data(), candidate_output,
                        output_elements * sizeof(float), cudaMemcpyDeviceToHost));
  const Exactness exactness = compare_outputs(host_baseline, host_candidate);

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

  const KernelResources baseline_resources = native_resources();
  const KernelResources candidate_resources = driver_resources(candidate_kernel);
  const TimingSummary baseline_timing = summarize(baseline_samples);
  const TimingSummary candidate_timing = summarize(candidate_samples);
  const PairSummary pairs = summarize_pairs(baseline_samples, candidate_samples);

  CUDA_CHECK(cudaFree(candidate_output));
  CUDA_CHECK(cudaFree(baseline_output));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_a));
  DRIVER_CHECK(cuModuleUnload(candidate_module));
  print_result(args, properties, exactness, baseline_timing, candidate_timing,
               pairs, baseline_resources, candidate_resources);
  return exactness.bitwise_equal ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
  return run(parse_args(argc, argv));
}
