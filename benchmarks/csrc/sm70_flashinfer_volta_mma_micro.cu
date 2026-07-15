// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <flashinfer/attention/sm70/volta_mma.cuh>

namespace {

constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kMmaK = 16;
constexpr int kReductionK = 256;
constexpr int kKTiles = kReductionK / kMmaK;
constexpr int kAElementsPerGroup = kM * kReductionK;
constexpr int kBElementsPerGroup = kReductionK * kN;
constexpr int kOutputElementsPerGroup = kM * kN;
constexpr int kWarpThreads = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpThreads * kWarpsPerBlock;

enum class Operation : int {
  kQKRowCol = 0,
  kPVRowRow = 1,
};

enum class InputPattern : int {
  kRandom = 0,
  kAlternating = 1,
};

struct Args {
  int device = 0;
  int groups = 4096;
  int warmup = 20;
  int rounds = 100;
  int launches_per_sample = 8;
  bool smoke = false;
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
  int compatibility_first_count = 0;
  int compatibility_faster = 0;
  int reference_faster = 0;
  int ties = 0;
  double compatibility_minus_reference_median_us = 0.0;
  double compatibility_minus_reference_mean_us = 0.0;
};

struct Exactness {
  bool bitwise_equal = false;
  int64_t word_count = 0;
  int64_t mismatch_words = 0;
  uint32_t xor_reduction = 0;
  uint32_t max_word_xor = 0;
  int64_t first_mismatch_index = -1;
  uint32_t first_reference_bits = 0;
  uint32_t first_compatibility_bits = 0;
};

struct CaseResult {
  InputPattern pattern;
  Operation operation;
  Exactness exactness;
  TimingSummary reference_timing;
  TimingSummary compatibility_timing;
  PairSummary pairs;
};

struct HostInputs {
  std::vector<__half> query;
  std::vector<__half> key;
  std::vector<__half> probability;
  std::vector<__half> value;
  std::vector<float> initial_c;
};

struct DeviceBuffers {
  __half* query = nullptr;
  __half* key = nullptr;
  __half* probability = nullptr;
  __half* value = nullptr;
  float* initial_c = nullptr;
  float* reference_output = nullptr;
  float* compatibility_output = nullptr;
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

const char* operation_name(Operation operation) {
  return operation == Operation::kQKRowCol ? "qk_row_col" : "pv_row_row";
}

const char* pattern_name(InputPattern pattern) {
  return pattern == InputPattern::kRandom ? "random" : "alternating";
}

const char* compatibility_name(Operation operation) {
  return operation == Operation::kQKRowCol
             ? "flashinfer::attention::sm70::"
               "mma_sync_m16n16k16_row_col_f16f16f32"
             : "flashinfer::attention::sm70::"
               "mma_sync_m16n16k16_row_row_f16f16f32";
}

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <Operation operation>
__device__ __forceinline__ void direct_wmma_reference(
    float* output, const __half* a, const __half* b, const float* initial_c) {
  namespace wmma = nvcuda::wmma;
  using AFragment =
      wmma::fragment<wmma::matrix_a, kM, kN, kMmaK, __half,
                     wmma::row_major>;
  using CFragment =
      wmma::fragment<wmma::accumulator, kM, kN, kMmaK, float>;

  CFragment accumulator;
  wmma::load_matrix_sync(accumulator, initial_c, kN, wmma::mem_row_major);
  if constexpr (operation == Operation::kQKRowCol) {
    using BFragment = wmma::fragment<wmma::matrix_b, kM, kN, kMmaK, __half,
                                     wmma::col_major>;
    AFragment a_fragment;
    BFragment b_fragment;
#pragma unroll
    for (int k_offset = 0; k_offset < kReductionK; k_offset += kMmaK) {
      wmma::load_matrix_sync(a_fragment, a + k_offset, kReductionK);
      wmma::load_matrix_sync(b_fragment, b + k_offset, kReductionK);
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
  } else {
    using BFragment = wmma::fragment<wmma::matrix_b, kM, kN, kMmaK, __half,
                                     wmma::row_major>;
    AFragment a_fragment;
    BFragment b_fragment;
#pragma unroll
    for (int k_offset = 0; k_offset < kReductionK; k_offset += kMmaK) {
      wmma::load_matrix_sync(a_fragment, a + k_offset, kReductionK);
      wmma::load_matrix_sync(b_fragment, b + k_offset * kN, kN);
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
  }
  wmma::store_matrix_sync(output, accumulator, kN, wmma::mem_row_major);
}

template <Operation operation>
__device__ __forceinline__ void compatibility_mma(float* output,
                                                  const __half* a,
                                                  const __half* b,
                                                  const float* initial_c) {
  namespace sm70 = flashinfer::attention::sm70;
  sm70::AccumulatorFragment accumulator;
  sm70::load_accumulator_fragment(accumulator, initial_c, kN);
  if constexpr (operation == Operation::kQKRowCol) {
    sm70::AFragment a_fragment;
    sm70::QKBFragment b_fragment;
#pragma unroll
    for (int k_offset = 0; k_offset < kReductionK; k_offset += kMmaK) {
      sm70::load_a_fragment(a_fragment, a + k_offset, kReductionK);
      sm70::load_qk_b_fragment(b_fragment, b + k_offset, kReductionK);
      sm70::mma_sync_m16n16k16_row_col_f16f16f32(accumulator, a_fragment,
                                                  b_fragment);
    }
  } else {
    sm70::AFragment a_fragment;
    sm70::PVBFragment b_fragment;
#pragma unroll
    for (int k_offset = 0; k_offset < kReductionK; k_offset += kMmaK) {
      sm70::load_a_fragment(a_fragment, a + k_offset, kReductionK);
      sm70::load_pv_b_fragment(b_fragment, b + k_offset * kN, kN);
      sm70::mma_sync_m16n16k16_row_row_f16f16f32(accumulator, a_fragment,
                                                  b_fragment);
    }
  }
  sm70::store_accumulator_fragment(output, accumulator, kN);
}

template <Operation operation, bool kCompatibility>
__device__ __forceinline__ void run_warp(const __half* a, const __half* b,
                                         const float* initial_c, float* output,
                                         int groups) {
  const int warp = threadIdx.x / kWarpThreads;
  const int group = blockIdx.x * kWarpsPerBlock + warp;
  if (group >= groups) {
    return;
  }

  const int64_t a_offset =
      static_cast<int64_t>(group) * kAElementsPerGroup;
  const int64_t b_offset =
      static_cast<int64_t>(group) * kBElementsPerGroup;
  const int64_t output_offset =
      static_cast<int64_t>(group) * kOutputElementsPerGroup;
  float* output_tile = output + output_offset;
  const float* initial_c_tile = initial_c + output_offset;

  if constexpr (kCompatibility) {
    compatibility_mma<operation>(output_tile, a + a_offset, b + b_offset,
                                 initial_c_tile);
  } else {
    direct_wmma_reference<operation>(output_tile, a + a_offset, b + b_offset,
                                     initial_c_tile);
  }
}

extern "C" __global__ __launch_bounds__(kThreads)
void sm70_flashinfer_volta_mma_qk_reference_kernel(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const float* __restrict__ initial_c, float* __restrict__ output,
    int groups) {
  run_warp<Operation::kQKRowCol, false>(query, key, initial_c, output,
                                         groups);
}

extern "C" __global__ __launch_bounds__(kThreads)
void sm70_flashinfer_volta_mma_qk_compat_kernel(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const float* __restrict__ initial_c, float* __restrict__ output,
    int groups) {
  run_warp<Operation::kQKRowCol, true>(query, key, initial_c, output, groups);
}

extern "C" __global__ __launch_bounds__(kThreads)
void sm70_flashinfer_volta_mma_pv_reference_kernel(
    const __half* __restrict__ probability, const __half* __restrict__ value,
    const float* __restrict__ initial_c, float* __restrict__ output,
    int groups) {
  run_warp<Operation::kPVRowRow, false>(probability, value, initial_c, output,
                                         groups);
}

extern "C" __global__ __launch_bounds__(kThreads)
void sm70_flashinfer_volta_mma_pv_compat_kernel(
    const __half* __restrict__ probability, const __half* __restrict__ value,
    const float* __restrict__ initial_c, float* __restrict__ output,
    int groups) {
  run_warp<Operation::kPVRowRow, true>(probability, value, initial_c, output,
                                        groups);
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
  const int value = static_cast<int>(next_random(state) & 0x7ffU) - 1024;
  return static_cast<float>(value) / 256.0f;
}

float random_float_value(uint32_t* state) {
  const int value = static_cast<int>(next_random(state) & 0xfffU) - 2048;
  return static_cast<float>(value) / 512.0f;
}

float alternating_value(int group, int row, int column, int salt) {
  const int parity = (group * 13 + row * 7 + column * 5 + salt) & 1;
  const int magnitude = 1 + ((group * 3 + row * 11 + column * 17 + salt) % 15);
  const float sign = parity == 0 ? 1.0f : -1.0f;
  return sign * static_cast<float>(magnitude) / 8.0f;
}

float alternating_c_value(int group, int row, int column) {
  return alternating_value(group, row, column, 41) / 2.0f;
}

void fill_half_input(std::vector<__half>* output, int groups, int rows,
                     int columns, InputPattern pattern, uint32_t seed,
                     int salt) {
  uint32_t state = seed;
  const int elements_per_group = rows * columns;
  for (int group = 0; group < groups; ++group) {
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        const float value =
            pattern == InputPattern::kRandom
                ? random_half_value(&state)
                : alternating_value(group, row, column, salt);
        (*output)[static_cast<size_t>(group) * elements_per_group +
                  row * columns + column] = __float2half_rn(value);
      }
    }
  }
}

void fill_initial_c(std::vector<float>* output, InputPattern pattern,
                    uint32_t seed) {
  uint32_t state = seed;
  const int groups =
      static_cast<int>(output->size() / kOutputElementsPerGroup);
  for (int group = 0; group < groups; ++group) {
    for (int row = 0; row < kM; ++row) {
      for (int column = 0; column < kN; ++column) {
        const float value = pattern == InputPattern::kRandom
                                ? random_float_value(&state)
                                : alternating_c_value(group, row, column);
        (*output)[static_cast<size_t>(group) * kOutputElementsPerGroup +
                  row * kN + column] = value;
      }
    }
  }
}

HostInputs make_inputs(int groups, InputPattern pattern) {
  const size_t a_element_count =
      static_cast<size_t>(groups) * kAElementsPerGroup;
  const size_t b_element_count =
      static_cast<size_t>(groups) * kBElementsPerGroup;
  const size_t output_element_count =
      static_cast<size_t>(groups) * kOutputElementsPerGroup;
  HostInputs inputs{
      std::vector<__half>(a_element_count),
      std::vector<__half>(b_element_count),
      std::vector<__half>(a_element_count),
      std::vector<__half>(b_element_count),
      std::vector<float>(output_element_count)};
  fill_half_input(&inputs.query, groups, kM, kReductionK, pattern,
                  0x6d2b79f5U, 3);
  fill_half_input(&inputs.key, groups, kN, kReductionK, pattern,
                  0x9e3779b9U, 7);
  fill_half_input(&inputs.probability, groups, kM, kReductionK, pattern,
                  0x243f6a88U, 11);
  fill_half_input(&inputs.value, groups, kReductionK, kN, pattern,
                  0xb7e15162U, 19);
  fill_initial_c(&inputs.initial_c, pattern, 0x85ebca6bU);
  return inputs;
}

void allocate_buffers(DeviceBuffers* buffers, int groups) {
  const size_t a_bytes = static_cast<size_t>(groups) * kAElementsPerGroup *
                         sizeof(__half);
  const size_t b_bytes = static_cast<size_t>(groups) * kBElementsPerGroup *
                         sizeof(__half);
  const size_t output_bytes =
      static_cast<size_t>(groups) * kOutputElementsPerGroup * sizeof(float);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->query), a_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->key), b_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->probability),
                        a_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->value), b_bytes));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&buffers->initial_c), output_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->reference_output),
                        output_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->compatibility_output),
                        output_bytes));
}

void release_buffers(DeviceBuffers* buffers) {
  CUDA_CHECK(cudaFree(buffers->compatibility_output));
  CUDA_CHECK(cudaFree(buffers->reference_output));
  CUDA_CHECK(cudaFree(buffers->initial_c));
  CUDA_CHECK(cudaFree(buffers->value));
  CUDA_CHECK(cudaFree(buffers->probability));
  CUDA_CHECK(cudaFree(buffers->key));
  CUDA_CHECK(cudaFree(buffers->query));
  *buffers = {};
}

void upload_inputs(const HostInputs& inputs, DeviceBuffers buffers) {
  const size_t query_bytes = inputs.query.size() * sizeof(__half);
  const size_t key_bytes = inputs.key.size() * sizeof(__half);
  const size_t probability_bytes = inputs.probability.size() * sizeof(__half);
  const size_t value_bytes = inputs.value.size() * sizeof(__half);
  const size_t float_bytes = inputs.initial_c.size() * sizeof(float);
  CUDA_CHECK(cudaMemcpy(buffers.query, inputs.query.data(), query_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.key, inputs.key.data(), key_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.probability, inputs.probability.data(),
                        probability_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.value, inputs.value.data(), value_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.initial_c, inputs.initial_c.data(), float_bytes,
                        cudaMemcpyHostToDevice));
}

void launch_kernel(Operation operation, bool compatibility,
                   const DeviceBuffers& buffers, int groups) {
  const dim3 block(kThreads);
  const dim3 grid((groups + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (operation == Operation::kQKRowCol) {
    if (compatibility) {
      sm70_flashinfer_volta_mma_qk_compat_kernel<<<grid, block>>>(
          buffers.query, buffers.key, buffers.initial_c,
          buffers.compatibility_output, groups);
    } else {
      sm70_flashinfer_volta_mma_qk_reference_kernel<<<grid, block>>>(
          buffers.query, buffers.key, buffers.initial_c, buffers.reference_output,
          groups);
    }
  } else if (compatibility) {
    sm70_flashinfer_volta_mma_pv_compat_kernel<<<grid, block>>>(
        buffers.probability, buffers.value, buffers.initial_c,
        buffers.compatibility_output, groups);
  } else {
    sm70_flashinfer_volta_mma_pv_reference_kernel<<<grid, block>>>(
        buffers.probability, buffers.value, buffers.initial_c,
        buffers.reference_output, groups);
  }
  CUDA_CHECK(cudaGetLastError());
}

TimingSummary summarize(std::vector<double> values) {
  TimingSummary result;
  if (values.empty()) {
    return result;
  }
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  result.median_us = values.size() % 2 == 0
                         ? (values[middle - 1] + values[middle]) / 2.0
                         : values[middle];
  const size_t p90 = static_cast<size_t>(std::ceil(0.9 * values.size())) - 1;
  result.p90_us = values[std::min(p90, values.size() - 1)];
  for (double value : values) {
    result.mean_us += value;
  }
  result.mean_us /= static_cast<double>(values.size());
  result.min_us = values.front();
  result.max_us = values.back();
  return result;
}

PairSummary summarize_pairs(const std::vector<double>& reference,
                            const std::vector<double>& compatibility) {
  PairSummary result;
  std::vector<double> deltas;
  deltas.reserve(reference.size());
  result.count = static_cast<int>(reference.size());
  for (size_t index = 0; index < reference.size(); ++index) {
    const double delta = compatibility[index] - reference[index];
    deltas.push_back(delta);
    if (compatibility[index] < reference[index]) {
      ++result.compatibility_faster;
    } else if (reference[index] < compatibility[index]) {
      ++result.reference_faster;
    } else {
      ++result.ties;
    }
  }
  const TimingSummary delta_summary = summarize(std::move(deltas));
  result.compatibility_minus_reference_median_us = delta_summary.median_us;
  result.compatibility_minus_reference_mean_us = delta_summary.mean_us;
  return result;
}

Exactness compare_outputs(const std::vector<float>& reference,
                          const std::vector<float>& compatibility) {
  Exactness result;
  result.bitwise_equal = true;
  result.word_count = static_cast<int64_t>(reference.size());
  for (size_t index = 0; index < reference.size(); ++index) {
    const uint32_t reference_bits = float_bits(reference[index]);
    const uint32_t compatibility_bits = float_bits(compatibility[index]);
    const uint32_t word_xor = reference_bits ^ compatibility_bits;
    result.bitwise_equal &= word_xor == 0;
    result.mismatch_words += word_xor != 0;
    result.xor_reduction ^= word_xor;
    result.max_word_xor = std::max(result.max_word_xor, word_xor);
    if (word_xor != 0 && result.first_mismatch_index < 0) {
      result.first_mismatch_index = static_cast<int64_t>(index);
      result.first_reference_bits = reference_bits;
      result.first_compatibility_bits = compatibility_bits;
    }
  }
  return result;
}

double time_kernel(cudaEvent_t start, cudaEvent_t stop, Operation operation,
                   bool compatibility, const DeviceBuffers& buffers, int groups,
                   int launches_per_sample) {
  CUDA_CHECK(cudaEventRecord(start));
  for (int launch = 0; launch < launches_per_sample; ++launch) {
    launch_kernel(operation, compatibility, buffers, groups);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  return 1000.0 * static_cast<double>(elapsed_ms) / launches_per_sample;
}

CaseResult run_case(InputPattern pattern, Operation operation, const Args& args,
                    const DeviceBuffers& buffers) {
  const size_t element_count =
      static_cast<size_t>(args.groups) * kOutputElementsPerGroup;
  std::vector<float> reference(element_count);
  std::vector<float> compatibility(element_count);

  launch_kernel(operation, false, buffers, args.groups);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(reference.data(), buffers.reference_output,
                        reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
  launch_kernel(operation, true, buffers, args.groups);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(compatibility.data(), buffers.compatibility_output,
                        compatibility.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int warmup = 0; warmup < args.warmup; ++warmup) {
    launch_kernel(operation, false, buffers, args.groups);
    launch_kernel(operation, true, buffers, args.groups);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> reference_samples;
  std::vector<double> compatibility_samples;
  reference_samples.reserve(args.rounds);
  compatibility_samples.reserve(args.rounds);
  int compatibility_first_count = 0;
  for (int round = 0; round < args.rounds; ++round) {
    const bool compatibility_first = (round & 1) != 0;
    compatibility_first_count += compatibility_first;
    if (compatibility_first) {
      compatibility_samples.push_back(
          time_kernel(start, stop, operation, true, buffers, args.groups,
                      args.launches_per_sample));
      reference_samples.push_back(
          time_kernel(start, stop, operation, false, buffers, args.groups,
                      args.launches_per_sample));
    } else {
      reference_samples.push_back(
          time_kernel(start, stop, operation, false, buffers, args.groups,
                      args.launches_per_sample));
      compatibility_samples.push_back(
          time_kernel(start, stop, operation, true, buffers, args.groups,
                      args.launches_per_sample));
    }
  }
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  CaseResult result;
  result.pattern = pattern;
  result.operation = operation;
  result.exactness = compare_outputs(reference, compatibility);
  result.pairs = summarize_pairs(reference_samples, compatibility_samples);
  result.pairs.compatibility_first_count = compatibility_first_count;
  result.reference_timing = summarize(std::move(reference_samples));
  result.compatibility_timing = summarize(std::move(compatibility_samples));
  return result;
}

void print_json_string(const char* value) {
  std::cout << '"';
  for (const unsigned char character : std::string(value)) {
    if (character == '\\') {
      std::cout << "\\\\";
    } else if (character == '"') {
      std::cout << "\\\"";
    } else if (character == '\n') {
      std::cout << "\\n";
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

void print_case(const CaseResult& result) {
  const double relative_delta_pct =
      100.0 * (result.compatibility_timing.median_us -
               result.reference_timing.median_us) /
      result.reference_timing.median_us;
  std::cout << "{\"input_pattern\":";
  print_json_string(pattern_name(result.pattern));
  std::cout << ",\"operation\":";
  print_json_string(operation_name(result.operation));
  std::cout << ",\"reference_implementation\":"
               "\"independent direct nvcuda::wmma K=256 loop\""
            << ",\"compatibility_implementation\":";
  print_json_string(compatibility_name(result.operation));
  std::cout << ",\"accumulator_lifetime\":"
               "\"one load, 16 register updates, one store\"";
  std::cout << ",\"exactness\":{\"word_dtype\":\"uint32(fp32)\","
               "\"word_count\":"
            << result.exactness.word_count << ",\"full_output\":true,"
            << "\"bitwise_equal\":"
            << (result.exactness.bitwise_equal ? "true" : "false")
            << ",\"mismatch_words\":" << result.exactness.mismatch_words
            << ",\"xor_reduction\":" << result.exactness.xor_reduction
            << ",\"max_word_xor\":" << result.exactness.max_word_xor
            << ",\"first_mismatch_index\":"
            << result.exactness.first_mismatch_index
            << ",\"first_reference_bits\":"
            << result.exactness.first_reference_bits
            << ",\"first_compatibility_bits\":"
            << result.exactness.first_compatibility_bits << '}'
            << ",\"timing\":{\"unit\":\"us per grid launch\","
               "\"reference\":";
  print_timing(result.reference_timing);
  std::cout << ",\"compatibility\":";
  print_timing(result.compatibility_timing);
  std::cout << ",\"compatibility_minus_reference_median_us\":"
            << result.pairs.compatibility_minus_reference_median_us
            << ",\"compatibility_relative_delta_pct\":"
            << relative_delta_pct << '}'
            << ",\"pairs\":{\"count\":" << result.pairs.count
            << ",\"compatibility_first_count\":"
            << result.pairs.compatibility_first_count
            << ",\"compatibility_faster\":"
            << result.pairs.compatibility_faster
            << ",\"reference_faster\":" << result.pairs.reference_faster
            << ",\"ties\":" << result.pairs.ties
            << ",\"compatibility_minus_reference_median_us\":"
            << result.pairs.compatibility_minus_reference_median_us
            << ",\"compatibility_minus_reference_mean_us\":"
            << result.pairs.compatibility_minus_reference_mean_us << "}}";
}

void print_results(const Args& args, const cudaDeviceProp& properties,
                   const std::array<CaseResult, 4>& results) {
  bool all_bitwise_equal = true;
  for (const CaseResult& result : results) {
    all_bitwise_equal &= result.exactness.bitwise_equal;
  }
  std::cout << "{\n\"target\":\"sm_70\",\n"
               "\"scope\":\"primitive compatibility only; no attention "
               "speed claim\",\n"
               "\"device\":{\"logical_index\":"
            << args.device << ",\"name\":";
  print_json_string(properties.name);
  std::cout << ",\"capability\":[" << properties.major << ','
            << properties.minor << "]},\n"
               "\"shape\":{\"groups\":"
            << args.groups
            << ",\"a\":[\"groups\",16,256],"
               "\"qk_b_physical\":[\"groups\",16,256],"
               "\"pv_b\":[\"groups\",256,16],"
               "\"c_and_output\":[\"groups\",16,16]},\n"
               "\"work\":{\"k_tiles\":"
            << kKTiles
            << ",\"wmma_m16n16k16_updates_per_warp\":16,"
               "\"expected_hmma884_per_kernel\":256,"
               "\"compatibility_pointer_control_used\":false},\n"
               "\"kernels\":{\"qk_row_col\":{\"reference\":"
               "\"sm70_flashinfer_volta_mma_qk_reference_kernel\","
               "\"compatibility\":"
               "\"sm70_flashinfer_volta_mma_qk_compat_kernel\"},"
               "\"pv_row_row\":{\"reference\":"
               "\"sm70_flashinfer_volta_mma_pv_reference_kernel\","
               "\"compatibility\":"
               "\"sm70_flashinfer_volta_mma_pv_compat_kernel\"}},\n"
               "\"measurement\":{\"warmup_pairs\":"
            << args.warmup << ",\"rounds\":" << args.rounds
            << ",\"launches_per_sample\":" << args.launches_per_sample
            << ",\"interleaving\":"
               "\"reference/compatibility order alternates each round\"},\n"
               "\"results\":[";
  for (size_t index = 0; index < results.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    print_case(results[index]);
  }
  std::cout << "],\n\"all_bitwise_equal\":"
            << (all_bitwise_equal ? "true" : "false") << "\n}" << std::endl;
}

int parse_int(const char* option, const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    std::cerr << "Invalid value for " << option << ": " << value << '\n';
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(parsed);
}

void print_usage(const char* program) {
  std::cout << "Usage: " << program
            << " [--device N] [--groups N] [--warmup N] [--rounds N]"
               " [--launches-per-sample N] [--smoke]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option(argv[index]);
    if (option == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (option == "--smoke") {
      args.smoke = true;
      continue;
    }
    if (index + 1 == argc) {
      std::cerr << "Missing value for " << option << '\n';
      std::exit(EXIT_FAILURE);
    }
    const int value = parse_int(option.c_str(), argv[++index]);
    if (option == "--device") {
      args.device = value;
    } else if (option == "--groups") {
      args.groups = value;
    } else if (option == "--warmup") {
      args.warmup = value;
    } else if (option == "--rounds") {
      args.rounds = value;
    } else if (option == "--launches-per-sample") {
      args.launches_per_sample = value;
    } else {
      std::cerr << "Unknown option: " << option << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  if (args.smoke) {
    args.groups = 1;
    args.warmup = 0;
    args.rounds = 1;
    args.launches_per_sample = 1;
  }
  if (args.device < 0 || args.groups < 1 || args.warmup < 0 || args.rounds < 1 ||
      args.launches_per_sample < 1) {
    std::cerr << "device must be non-negative; groups, rounds, and launches "
                 "must be positive; warmup must be non-negative\n";
    std::exit(EXIT_FAILURE);
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "This primitive probe requires an SM70 device; selected "
              << properties.name << " reports " << properties.major << '.'
              << properties.minor << '\n';
    return EXIT_FAILURE;
  }

  DeviceBuffers buffers;
  allocate_buffers(&buffers, args.groups);
  std::array<CaseResult, 4> results{};
  size_t result_index = 0;
  for (InputPattern pattern : {InputPattern::kRandom, InputPattern::kAlternating}) {
    const HostInputs inputs = make_inputs(args.groups, pattern);
    upload_inputs(inputs, buffers);
    results[result_index++] = run_case(pattern, Operation::kQKRowCol, args, buffers);
    results[result_index++] = run_case(pattern, Operation::kPVRowRow, args, buffers);
  }
  release_buffers(&buffers);
  print_results(args, properties, results);

  for (const CaseResult& result : results) {
    if (!result.exactness.bitwise_equal) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
