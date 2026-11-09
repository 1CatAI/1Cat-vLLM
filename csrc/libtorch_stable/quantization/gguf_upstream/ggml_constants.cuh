// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common.cuh"

// Type properties mirror the pinned GGML type_traits table in ggml/src/ggml.c;
// the final MMQ flag mirrors the selected template instances.
// Columns: enum, block size, C++ block type, name, quantized, MMQ instance.
// Keep this Torch-free so the GGML ABI adapter and Torch bridge share one list.
#define GGUF_GGML_TYPE_TRAITS(X)                             \
  X(GGML_TYPE_F32, 1, float, "f32", 0, 0)                    \
  X(GGML_TYPE_F16, 1, ggml_fp16_t, "f16", 0, 0)              \
  X(GGML_TYPE_Q4_0, QK4_0, block_q4_0, "q4_0", 1, 1)         \
  X(GGML_TYPE_Q4_1, QK4_1, block_q4_1, "q4_1", 1, 1)         \
  X(GGML_TYPE_Q5_0, QK5_0, block_q5_0, "q5_0", 1, 1)         \
  X(GGML_TYPE_Q5_1, QK5_1, block_q5_1, "q5_1", 1, 1)         \
  X(GGML_TYPE_Q8_0, QK8_0, block_q8_0, "q8_0", 1, 1)         \
  X(GGML_TYPE_Q8_1, QK8_1, block_q8_1, "q8_1", 1, 0)         \
  X(GGML_TYPE_Q2_K, QK_K, block_q2_K, "q2_K", 1, 1)          \
  X(GGML_TYPE_Q3_K, QK_K, block_q3_K, "q3_K", 1, 1)          \
  X(GGML_TYPE_Q4_K, QK_K, block_q4_K, "q4_K", 1, 1)          \
  X(GGML_TYPE_Q5_K, QK_K, block_q5_K, "q5_K", 1, 1)          \
  X(GGML_TYPE_Q6_K, QK_K, block_q6_K, "q6_K", 1, 1)          \
  X(GGML_TYPE_Q8_K, QK_K, block_q8_K, "q8_K", 1, 0)          \
  X(GGML_TYPE_IQ2_XXS, QK_K, block_iq2_xxs, "iq2_xxs", 1, 1) \
  X(GGML_TYPE_IQ2_XS, QK_K, block_iq2_xs, "iq2_xs", 1, 1)    \
  X(GGML_TYPE_IQ3_XXS, QK_K, block_iq3_xxs, "iq3_xxs", 1, 1) \
  X(GGML_TYPE_IQ1_S, QK_K, block_iq1_s, "iq1_s", 1, 1)       \
  X(GGML_TYPE_IQ4_NL, QK4_NL, block_iq4_nl, "iq4_nl", 1, 1)  \
  X(GGML_TYPE_IQ3_S, QK_K, block_iq3_s, "iq3_s", 1, 1)       \
  X(GGML_TYPE_IQ2_S, QK_K, block_iq2_s, "iq2_s", 1, 1)       \
  X(GGML_TYPE_IQ4_XS, QK_K, block_iq4_xs, "iq4_xs", 1, 1)    \
  X(GGML_TYPE_I8, 1, int8_t, "i8", 0, 0)                     \
  X(GGML_TYPE_I16, 1, int16_t, "i16", 0, 0)                  \
  X(GGML_TYPE_I32, 1, int32_t, "i32", 0, 0)                  \
  X(GGML_TYPE_I64, 1, int64_t, "i64", 0, 0)                  \
  X(GGML_TYPE_F64, 1, double, "f64", 0, 0)                   \
  X(GGML_TYPE_IQ1_M, QK_K, block_iq1_m, "iq1_m", 1, 0)       \
  X(GGML_TYPE_BF16, 1, ggml_bf16_t, "bf16", 0, 0)            \
  X(GGML_TYPE_TQ1_0, QK_K, block_tq1_0, "tq1_0", 1, 0)       \
  X(GGML_TYPE_TQ2_0, QK_K, block_tq2_0, "tq2_0", 1, 0)       \
  X(GGML_TYPE_MXFP4, QK_MXFP4, block_mxfp4, "mxfp4", 1, 1)   \
  X(GGML_TYPE_NVFP4, QK_NVFP4, block_nvfp4, "nvfp4", 1, 1)   \
  X(GGML_TYPE_Q1_0, QK1_0, block_q1_0, "q1_0", 1, 1)         \
  X(GGML_TYPE_Q2_0, QK2_0, block_q2_0, "q2_0", 1, 1)

// Use the last column to generate only MMQ switch cases from the same list.
#define GGUF_IF_MMQ_0(action, type)
#define GGUF_IF_MMQ_1(action, type) action(type)
#define GGUF_IF_MMQ(enabled, action, type) GGUF_IF_MMQ_##enabled(action, type)

namespace gguf_constants {
inline constexpr int64_t kMatrixRowPadding = MATRIX_ROW_PADDING;
// Mirrored from the J loop in ggml-cuda/mmq.cuh::mul_mat_q_switch_J.
inline constexpr int kMmqTileColumnsMax = 128;
inline constexpr int kMmqTileStep = 8;
}  // namespace gguf_constants

static_assert(GGML_TYPE_COUNT == 43,
              "GGML types changed: update GGUF_GGML_TYPE_TRAITS");
static_assert(MATRIX_ROW_PADDING == 512,
              "Python kernel_support.py assumes 512-value row padding; "
              "update it and tests/helpers_upstream.py together with GGML");
