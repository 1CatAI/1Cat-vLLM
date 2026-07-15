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
#include <vector>

namespace {

constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kRawN = 32;
constexpr int kK = 16;
constexpr int kKStep = 4;
constexpr int kRawM = 8;
constexpr int kRawRegisters = 8;
constexpr int kAElements = kM * kK;
constexpr int kNativeTileElements = kM * kN;
constexpr int kOutputElements = kM * kRawN;
constexpr int kRawBElements = kK * kRawN;
constexpr int kThreads = 32;

enum class Layout : int {
  kRowCol = 0,
  kRowRow = 1,
};

enum class InputKind : int {
  kRandom = 0,
  kAlternatingSign = 1,
  kExponentSpan = 2,
};

const char* layout_name(Layout layout) {
  return layout == Layout::kRowCol ? "qk_row_col" : "pv_row_row";
}

const char* input_name(InputKind input) {
  switch (input) {
    case InputKind::kRandom:
      return "random";
    case InputKind::kAlternatingSign:
      return "alternating_sign";
    case InputKind::kExponentSpan:
      return "exponent_span";
  }
  return "unknown";
}

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

// These formulas are derived from SM70_MMA_884::thread_offset_C(),
// static_offset_C(), and ReshapeC() in mma_sm70.h. They agree with the
// memory-observable fragment strategy in sm70_wmma_fragment_probe.cu: WMMA is
// never inspected through its unspecified C++ fragment layout.
__host__ __device__ constexpr int raw_a_row(int lane) {
  return (lane & 3) + ((lane >> 4) * 4);
}

__host__ __device__ constexpr int raw_n_tile(int lane) {
  return (lane & 12) * 2;
}

__host__ __device__ constexpr int raw_c_row(int lane, int reg) {
  return (lane & 1) + (reg & 2) + ((lane >> 4) * 4);
}

__host__ __device__ constexpr int raw_c_col(int lane, int reg) {
  return raw_n_tile(lane) + (lane & 2) + (reg & 4) + (reg & 1);
}

// SmemCopy_MMA_884_B::get_offset() selects this column for row.col. For
// row.row, the PTX m8n8k4 row-major B map transposes the register traversal:
// a lane owns one K row and its four consecutive N values.
__host__ __device__ constexpr int raw_b_col_n(int lane) {
  return raw_n_tile(lane) + (lane & 3) + ((lane >> 4) * 4);
}

__host__ __device__ constexpr int raw_b_row_k(int lane) {
  return lane & 3;
}

__host__ __device__ constexpr int raw_b_row_n(int lane, int element) {
  return raw_n_tile(lane) + ((lane >> 4) * 4) + element;
}

__device__ __forceinline__ uint32_t pack_half2(__half low, __half high) {
  return static_cast<uint32_t>(__half_as_ushort(low)) |
         (static_cast<uint32_t>(__half_as_ushort(high)) << 16);
}

__device__ __forceinline__ void mma_m8n8k4_row_col(
    float (&d)[kRawRegisters], uint32_t a0, uint32_t a1, uint32_t b0,
    uint32_t b1, const float (&c)[kRawRegisters]) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
        "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]),
        "f"(c[7]));
}

__device__ __forceinline__ void mma_m8n8k4_row_row(
    float (&d)[kRawRegisters], uint32_t a0, uint32_t a1, uint32_t b0,
    uint32_t b1, const float (&c)[kRawRegisters]) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
        "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]),
        "f"(c[7]));
}

template <bool kRowCol>
__device__ __forceinline__ void native_wmma(
    const __half* shared_a, const __half* shared_b,
    const float* shared_c_tile, float* native_output_tile) {
  namespace wmma = nvcuda::wmma;
  wmma::fragment<wmma::matrix_a, kM, kN, kK, __half, wmma::row_major> a;
  wmma::fragment<wmma::accumulator, kM, kN, kK, float> c;
  wmma::fragment<wmma::accumulator, kM, kN, kK, float> d;
  wmma::load_matrix_sync(a, shared_a, kK);
  wmma::load_matrix_sync(c, shared_c_tile, kRawN, wmma::mem_row_major);
  if constexpr (kRowCol) {
    wmma::fragment<wmma::matrix_b, kM, kN, kK, __half, wmma::col_major> b;
    wmma::load_matrix_sync(b, shared_b, kK);
    wmma::mma_sync(d, a, b, c);
  } else {
    wmma::fragment<wmma::matrix_b, kM, kN, kK, __half, wmma::row_major> b;
    wmma::load_matrix_sync(b, shared_b, kN);
    wmma::mma_sync(d, a, b, c);
  }
  wmma::store_matrix_sync(native_output_tile, d, kRawN,
                          wmma::mem_row_major);
}

template <bool kRowCol>
__global__ __launch_bounds__(kThreads) void raw_hmma_probe_kernel(
    const __half* __restrict__ a, const __half* __restrict__ b,
    const float* __restrict__ c, const int* __restrict__ order,
    float* __restrict__ native_output, float* __restrict__ raw_output) {
  __shared__ __align__(32) __half shared_a[kAElements];
  __shared__ __align__(32) __half shared_b[kNativeTileElements];
  __shared__ __align__(32) float shared_c[kOutputElements];

  const int lane = threadIdx.x;
  for (int index = lane; index < kAElements; index += blockDim.x) {
    shared_a[index] = a[index];
  }
  for (int index = lane; index < kOutputElements; index += blockDim.x) {
    shared_c[index] = c[index];
  }
  __syncthreads();

  // Reference the complete 16x32 result with two native M16N16K16 tiles.
  for (int native_n = 0; native_n < kRawN; native_n += kN) {
    for (int index = lane; index < kNativeTileElements; index += blockDim.x) {
      const int row = index / kN;
      const int col = index % kN;
      if constexpr (kRowCol) {
        shared_b[col * kK + row] = b[row * kRawN + native_n + col];
      } else {
        shared_b[index] = b[row * kRawN + native_n + col];
      }
    }
    __syncthreads();
    native_wmma<kRowCol>(shared_a, shared_b, shared_c + native_n,
                          native_output + native_n);
    __syncthreads();
  }

  float accum_low[kRawRegisters];
  float accum_high[kRawRegisters];
#pragma unroll
  for (int reg = 0; reg < kRawRegisters; ++reg) {
    const int row = raw_c_row(lane, reg);
    const int col = raw_c_col(lane, reg);
    accum_low[reg] = c[row * kRawN + col];
    accum_high[reg] = c[(row + kRawM) * kRawN + col];
  }

  const int raw_row = raw_a_row(lane);
  int order_local[4] = {order[0], order[1], order[2], order[3]};
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    const int k_base = order_local[step] * kKStep;
    __half a_low[kKStep];
    __half a_high[kKStep];
    __half b_values[kKStep];
#pragma unroll
    for (int element = 0; element < kKStep; ++element) {
      a_low[element] = a[raw_row * kK + k_base + element];
      a_high[element] = a[(raw_row + kRawM) * kK + k_base + element];
      if constexpr (kRowCol) {
        b_values[element] = b[(k_base + element) * kRawN + raw_b_col_n(lane)];
      } else {
        b_values[element] =
            b[(k_base + raw_b_row_k(lane)) * kRawN +
              raw_b_row_n(lane, element)];
      }
    }

    float next_low[kRawRegisters];
    float next_high[kRawRegisters];
    const uint32_t a_low_0 = pack_half2(a_low[0], a_low[1]);
    const uint32_t a_low_1 = pack_half2(a_low[2], a_low[3]);
    const uint32_t a_high_0 = pack_half2(a_high[0], a_high[1]);
    const uint32_t a_high_1 = pack_half2(a_high[2], a_high[3]);
    const uint32_t b_0 = pack_half2(b_values[0], b_values[1]);
    const uint32_t b_1 = pack_half2(b_values[2], b_values[3]);
    if constexpr (kRowCol) {
      mma_m8n8k4_row_col(next_low, a_low_0, a_low_1, b_0, b_1, accum_low);
      mma_m8n8k4_row_col(next_high, a_high_0, a_high_1, b_0, b_1,
                           accum_high);
    } else {
      mma_m8n8k4_row_row(next_low, a_low_0, a_low_1, b_0, b_1, accum_low);
      mma_m8n8k4_row_row(next_high, a_high_0, a_high_1, b_0, b_1,
                           accum_high);
    }
#pragma unroll
    for (int reg = 0; reg < kRawRegisters; ++reg) {
      accum_low[reg] = next_low[reg];
      accum_high[reg] = next_high[reg];
    }
  }

#pragma unroll
  for (int reg = 0; reg < kRawRegisters; ++reg) {
    const int row = raw_c_row(lane, reg);
    const int col = raw_c_col(lane, reg);
    raw_output[row * kRawN + col] = accum_low[reg];
    raw_output[(row + kRawM) * kRawN + col] = accum_high[reg];
  }
}

uint32_t next_random(uint32_t* state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

float random_value(uint32_t* state) {
  const int value = static_cast<int>(next_random(state) & 0x7ffu) - 1024;
  return static_cast<float>(value) / 256.0f;
}

float random_nonzero_value(uint32_t* state) {
  int value = static_cast<int>(next_random(state) & 0x3ffu) - 512;
  if (value == 0) {
    value = 1;
  }
  return static_cast<float>(value) / 257.0f;
}

float alternating_value(int row, int col, int scale) {
  const int parity = ((row * 7 + col * 11 + scale) & 1) == 0 ? 1 : -1;
  const int magnitude = 1 + ((row * 5 + col * 3 + scale) % 13);
  return parity * static_cast<float>(magnitude) / 8.0f;
}

float exponent_value(int row, int col, int salt) {
  constexpr int kExponents[] = {-12, -9, -6, -3, 0, 3, 6, 9};
  const int index = (row * 5 + col * 3 + salt) & 7;
  const float mantissa = 1.0f + 0.25f * ((row + col + salt) & 3);
  const float sign = ((row + col + salt) & 1) == 0 ? 1.0f : -1.0f;
  return sign * std::ldexp(mantissa, kExponents[index]);
}

float c_value(InputKind input, int row, int col, uint32_t* state) {
  if (input == InputKind::kRandom) {
    return random_nonzero_value(state);
  }
  if (input == InputKind::kAlternatingSign) {
    return alternating_value(row, col, 19);
  }
  return exponent_value(row, col, 29);
}

void fill_input(InputKind input, std::vector<__half>* a, std::vector<__half>* b,
                std::vector<float>* c) {
  uint32_t state = 0x6d2b79f5u + static_cast<uint32_t>(input) * 0x9e3779b9u;
  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kK; ++col) {
      float value;
      if (input == InputKind::kRandom) {
        value = random_value(&state);
      } else if (input == InputKind::kAlternatingSign) {
        value = alternating_value(row, col, 3);
      } else {
        value = exponent_value(row, col, 7);
      }
      (*a)[row * kK + col] = __float2half_rn(value);
    }
  }
  for (int row = 0; row < kK; ++row) {
    for (int col = 0; col < kRawN; ++col) {
      float value;
      if (input == InputKind::kRandom) {
        value = random_value(&state);
      } else if (input == InputKind::kAlternatingSign) {
        value = alternating_value(row, col, 11);
      } else {
        value = exponent_value(row, col, 17);
      }
      (*b)[row * kRawN + col] = __float2half_rn(value);
    }
  }
  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kRawN; ++col) {
      (*c)[row * kRawN + col] = c_value(input, row, col, &state);
    }
  }
}

struct MappingValidation {
  bool c_exact_cover = false;
  bool a_fourfold_cover = false;
  bool b_row_col_exact_cover = false;
  bool b_row_row_exact_cover = false;
};

MappingValidation validate_mapping() {
  int c_counts[kRawM][kRawN] = {};
  int a_counts[kRawM][kKStep] = {};
  int b_col_counts[kKStep][kRawN] = {};
  int b_row_counts[kKStep][kRawN] = {};
  for (int lane = 0; lane < kThreads; ++lane) {
    for (int reg = 0; reg < kRawRegisters; ++reg) {
      ++c_counts[raw_c_row(lane, reg)][raw_c_col(lane, reg)];
    }
    for (int element = 0; element < kKStep; ++element) {
      ++a_counts[raw_a_row(lane)][element];
      ++b_col_counts[element][raw_b_col_n(lane)];
      ++b_row_counts[raw_b_row_k(lane)][raw_b_row_n(lane, element)];
    }
  }

  MappingValidation result;
  result.c_exact_cover = true;
  result.a_fourfold_cover = true;
  result.b_row_col_exact_cover = true;
  result.b_row_row_exact_cover = true;
  for (int row = 0; row < kRawM; ++row) {
    for (int col = 0; col < kRawN; ++col) {
      result.c_exact_cover &= c_counts[row][col] == 1;
    }
    for (int col = 0; col < kKStep; ++col) {
      result.a_fourfold_cover &= a_counts[row][col] == 4;
    }
  }
  for (int row = 0; row < kKStep; ++row) {
    for (int col = 0; col < kRawN; ++col) {
      result.b_row_col_exact_cover &= b_col_counts[row][col] == 1;
      result.b_row_row_exact_cover &= b_row_counts[row][col] == 1;
    }
  }
  return result;
}

uint32_t float_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct Comparison {
  uint32_t xor_reduction = 0;
  uint32_t max_word_xor = 0;
  int differing_words = 0;
  float max_abs_error = 0.0f;
  bool bitwise_equal = true;
};

Comparison compare_outputs(
    const std::array<float, kOutputElements>& native_output,
    const std::array<float, kOutputElements>& raw_output) {
  Comparison result;
  for (int index = 0; index < kOutputElements; ++index) {
    const uint32_t word_xor =
        float_bits(native_output[index]) ^ float_bits(raw_output[index]);
    result.xor_reduction ^= word_xor;
    result.max_word_xor = std::max(result.max_word_xor, word_xor);
    result.bitwise_equal &= word_xor == 0;
    result.differing_words += word_xor != 0;
    const float abs_error = std::fabs(native_output[index] - raw_output[index]);
    if (std::isnan(abs_error)) {
      result.max_abs_error = std::numeric_limits<float>::infinity();
    } else {
      result.max_abs_error = std::max(result.max_abs_error, abs_error);
    }
  }
  return result;
}

struct Result {
  Layout layout;
  InputKind input;
  int order_index;
  Comparison comparison;
};

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

void print_order(const std::array<int, 4>& order) {
  std::cout << '[' << order[0] << ", " << order[1] << ", " << order[2]
            << ", " << order[3] << ']';
}

void print_float_or_null(float value) {
  if (std::isfinite(value)) {
    std::cout << std::setprecision(9) << value;
  } else {
    std::cout << "null";
  }
}

bool result_matches_all_inputs(const std::vector<Result>& results,
                               Layout layout, int order_index) {
  for (int input = 0; input < 3; ++input) {
    const auto match = std::find_if(
        results.begin(), results.end(), [&](const Result& result) {
          return result.layout == layout && result.input == static_cast<InputKind>(input) &&
                 result.order_index == order_index;
        });
    if (match == results.end() || !match->comparison.bitwise_equal) {
      return false;
    }
  }
  return true;
}

void print_json(const cudaDeviceProp& properties, int device,
                int runtime_version, const MappingValidation& mapping,
                const std::vector<std::array<int, 4>>& orders,
                const std::vector<Result>& results) {
  std::vector<int> qk_matching_orders;
  std::vector<int> pv_matching_orders;
  for (int order_index = 0; order_index < static_cast<int>(orders.size());
       ++order_index) {
    if (result_matches_all_inputs(results, Layout::kRowCol, order_index)) {
      qk_matching_orders.push_back(order_index);
    }
    if (result_matches_all_inputs(results, Layout::kRowRow, order_index)) {
      pv_matching_orders.push_back(order_index);
    }
  }
  const bool mapping_valid = mapping.c_exact_cover && mapping.a_fourfold_cover &&
                             mapping.b_row_col_exact_cover &&
                             mapping.b_row_row_exact_cover;
  const bool gate = mapping_valid && !qk_matching_orders.empty() &&
                    !pv_matching_orders.empty();

  std::cout << "{\n";
  std::cout << "  \"device\": {\n";
  std::cout << "    \"logical_index\": " << device << ",\n";
  std::cout << "    \"name\": ";
  print_json_string(properties.name);
  std::cout << ",\n";
  std::cout << "    \"capability\": [" << properties.major << ", "
            << properties.minor << "],\n";
  std::cout << "    \"cuda_runtime\": " << runtime_version << ",\n";
  std::cout << "    \"cuda_visible_devices\": ";
  print_json_string(std::getenv("CUDA_VISIBLE_DEVICES") == nullptr
                        ? ""
                        : std::getenv("CUDA_VISIBLE_DEVICES"));
  std::cout << "\n  },\n";
  std::cout << "  \"target\": \"sm_70\",\n";
  std::cout << "  \"native\": \"two wmma.m16n16k16.f32 accumulator tiles at N=0,16\",\n";
  std::cout << "  \"raw\": {\n";
  std::cout << "    \"qk\": \"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32\",\n";
  std::cout << "    \"pv\": \"mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32\",\n";
  std::cout << "    \"shape\": \"two M=8 raw fragments form one complete M=16, N=32 result\"\n";
  std::cout << "  },\n";
  std::cout << "  \"mapping_validation\": {\n";
  std::cout << "    \"c_exact_cover_8x32\": "
            << (mapping.c_exact_cover ? "true" : "false") << ",\n";
  std::cout << "    \"a_fourfold_cover_8x4\": "
            << (mapping.a_fourfold_cover ? "true" : "false") << ",\n";
  std::cout << "    \"b_row_col_exact_cover_4x32\": "
            << (mapping.b_row_col_exact_cover ? "true" : "false") << ",\n";
  std::cout << "    \"b_row_row_exact_cover_4x32\": "
            << (mapping.b_row_row_exact_cover ? "true" : "false") << "\n";
  std::cout << "  },\n";
  std::cout << "  \"c_initialization\": \"nonzero_fp32_for_every_case\",\n";
  std::cout << "  \"comparison\": \"512 per-element uint32_t IEEE-754 words over full 16x32 output\",\n";
  std::cout << "  \"results\": [\n";
  for (size_t index = 0; index < results.size(); ++index) {
    const Result& result = results[index];
    const Comparison& comparison = result.comparison;
    std::cout << "    {\"layout\": ";
    print_json_string(layout_name(result.layout));
    std::cout << ", \"input\": ";
    print_json_string(input_name(result.input));
    std::cout << ", \"order\": ";
    print_order(orders[result.order_index]);
    std::cout << ", \"xor\": {\"max_word\": "
              << comparison.max_word_xor
              << ", \"reduction\": " << comparison.xor_reduction << '}';
    std::cout << ", \"differing_words\": " << comparison.differing_words;
    std::cout << ", \"max_abs_error\": ";
    print_float_or_null(comparison.max_abs_error);
    std::cout << ", \"bitwise_equal\": "
              << (comparison.bitwise_equal ? "true" : "false") << '}';
    std::cout << (index + 1 == results.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n";
  std::cout << "  \"gate\": {\n";
  std::cout << "    \"compared_output_words\": " << kOutputElements << ",\n";
  std::cout << "    \"full_16x32\": true,\n";
  std::cout << "    \"criterion\": ";
  print_json_string(
      "each layout needs one fixed k4 order bitwise-equal for every input case");
  std::cout << ",\n    \"qk_row_col_matching_orders\": [";
  for (size_t index = 0; index < qk_matching_orders.size(); ++index) {
    print_order(orders[qk_matching_orders[index]]);
    std::cout << (index + 1 == qk_matching_orders.size() ? "" : ", ");
  }
  std::cout << "],\n    \"pv_row_row_matching_orders\": [";
  for (size_t index = 0; index < pv_matching_orders.size(); ++index) {
    print_order(orders[pv_matching_orders[index]]);
    std::cout << (index + 1 == pv_matching_orders.size() ? "" : ", ");
  }
  std::cout << "],\n    \"enter_single_panel_microkernel\": "
            << (gate ? "true" : "false") << "\n";
  std::cout << "  }\n";
  std::cout << "}\n";
}

int run(int device) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device < 0 || device >= device_count) {
    std::cerr << "Requested logical CUDA device " << device
              << " is unavailable; visible device count is " << device_count << '\n';
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "This probe requires SM70, got " << properties.major << '.'
              << properties.minor << '\n';
    return EXIT_FAILURE;
  }
  int runtime_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

  std::vector<std::array<int, 4>> orders;
  std::array<int, 4> order = {0, 1, 2, 3};
  do {
    orders.push_back(order);
  } while (std::next_permutation(order.begin(), order.end()));

  __half* device_a = nullptr;
  __half* device_b = nullptr;
  float* device_c = nullptr;
  int* device_order = nullptr;
  float* device_native = nullptr;
  float* device_raw = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a),
                        kAElements * sizeof(*device_a)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b),
                        kRawBElements * sizeof(*device_b)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c),
                        kOutputElements * sizeof(*device_c)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_order),
                        kKStep * sizeof(*device_order)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_native),
                        kOutputElements * sizeof(*device_native)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_raw),
                        kOutputElements * sizeof(*device_raw)));

  std::vector<Result> results;
  results.reserve(3 * 2 * orders.size());
  std::vector<__half> host_a(kAElements);
  std::vector<__half> host_b(kRawBElements);
  std::vector<float> host_c(kOutputElements);
  std::array<float, kOutputElements> host_native{};
  std::array<float, kOutputElements> host_raw{};

  for (int input_index = 0; input_index < 3; ++input_index) {
    const InputKind input = static_cast<InputKind>(input_index);
    fill_input(input, &host_a, &host_b, &host_c);
    CUDA_CHECK(cudaMemcpy(device_a, host_a.data(),
                          host_a.size() * sizeof(host_a[0]),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, host_b.data(),
                          host_b.size() * sizeof(host_b[0]),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_c, host_c.data(),
                          host_c.size() * sizeof(host_c[0]),
                          cudaMemcpyHostToDevice));

    for (int layout_index = 0; layout_index < 2; ++layout_index) {
      const Layout layout = static_cast<Layout>(layout_index);
      for (int order_index = 0; order_index < static_cast<int>(orders.size());
           ++order_index) {
        CUDA_CHECK(cudaMemcpy(device_order, orders[order_index].data(),
                              kKStep * sizeof(int), cudaMemcpyHostToDevice));
        if (layout == Layout::kRowCol) {
          raw_hmma_probe_kernel<true><<<1, kThreads>>>(
              device_a, device_b, device_c, device_order, device_native,
              device_raw);
        } else {
          raw_hmma_probe_kernel<false><<<1, kThreads>>>(
              device_a, device_b, device_c, device_order, device_native,
              device_raw);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_native.data(), device_native,
                              kOutputElements * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_raw.data(), device_raw,
                              kOutputElements * sizeof(float),
                              cudaMemcpyDeviceToHost));
        results.push_back(
            {layout, input, order_index, compare_outputs(host_native, host_raw)});
      }
    }
  }

  CUDA_CHECK(cudaFree(device_raw));
  CUDA_CHECK(cudaFree(device_native));
  CUDA_CHECK(cudaFree(device_order));
  CUDA_CHECK(cudaFree(device_c));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_a));

  print_json(properties, device, runtime_version, validate_mapping(), orders,
             results);
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--device" && index + 1 < argc) {
      device = std::stoi(argv[++index]);
      continue;
    }
    std::cerr << "Usage: " << argv[0] << " [--device N]\n";
    return EXIT_FAILURE;
  }
  return run(device);
}
