// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Keep the accepted 2x BM32 pair-scratch kernel in this translation unit.
// Renaming its host entry point lets this benchmark reuse the exact baseline
// without copying or weakening the reference implementation.
#define main sm70_native_bm64_allp_micro_embedded_main
#include "sm70_native_bm64_allp_micro.cu"
#undef main

namespace {

constexpr int kPipelineThreads = 256;
constexpr int kPipelineWarps = kPipelineThreads / 32;
constexpr int kPipelineOwners = 4;
constexpr int kPipelinePhysicalN = 16;
constexpr int kPipelineLogicalN = 32;
constexpr int kPipelineDFragmentsPerWarp = 8;
constexpr int kPipelineOutputFp32ValuesPerWarp =
    kPipelineDFragmentsPerWarp * AccumulatorFragment::num_elements;
constexpr int kPipelineProbabilityElements = kPanelM * kPanelN;
constexpr int kPipelineValueVectorsPerPartnerThread = 4;
constexpr int kPipelineStageVectors =
    kPipelinePhysicalN * kD * sizeof(__half) / sizeof(uint4);
constexpr int kPipelineRowStateBytes = 3 * kM64 * sizeof(float);

static_assert(kPipelineWarps == 8);
static_assert(kPipelineOwners * 2 == kPipelineWarps);
static_assert(kPipelineLogicalN == 2 * kPipelinePhysicalN);
static_assert(kPipelineStageVectors == 2 * kPipelineThreads);
static_assert(kPipelineOutputFp32ValuesPerWarp == 64);

struct alignas(16) FlashInferBm64PipelineShared {
  __half query[kM64QElements];
  __half kv_stage[kPipelinePhysicalN * kD];
  __half probability[kPipelineOwners][kPipelineProbabilityElements];
  float row_max[kM64];
  float row_sum[kM64];
  float row_exp_diff[kM64];
};

static_assert(sizeof(FlashInferBm64PipelineShared) == 43776,
              "BM64 pipeline shared layout changed unexpectedly");
static_assert(sizeof(FlashInferBm64PipelineShared) <= 48 * 1024,
              "BM64 pipeline must retain two CTA residency");

// Four partner warps cooperatively cover one N16 V tile with four uint4 loads
// per thread. The same 16-GPR payload is reused for V1 after P0 consumes V0.
struct FlashInferValuePayload {
  uint4 vector_0;
  uint4 vector_1;
  uint4 vector_2;
  uint4 vector_3;
};

static_assert(sizeof(FlashInferValuePayload) == 64,
              "value payload must remain 16 GPRs");

struct FlashInferPackedProbability {
  uint32_t word_0;
  uint32_t word_1;
  uint32_t word_2;
  uint32_t word_3;
};

static_assert(sizeof(FlashInferPackedProbability) == 16,
              "P1 must retain four packed FP16 words");

struct FlashInferQkRoleState {
  float first_row_max;
  float first_row_sum;
  float second_row_max;
  float second_row_sum;
  FlashInferPackedProbability probability_1;
};

union alignas(16) FlashInferRoleState {
  FlashInferValuePayload value_payload;
  FlashInferQkRoleState qk;
};

static_assert(sizeof(FlashInferRoleState) == sizeof(FlashInferValuePayload),
              "even and odd warp state must share the same 16-GPR budget");

__device__ __forceinline__ void flashinfer_owner_q_barrier(int owner) {
  const int barrier_id = owner + 1;
  asm volatile("bar.sync %0, 64;" : : "r"(barrier_id) : "memory");
}

__device__ __forceinline__ void flashinfer_pipeline_barrier() {
  asm volatile("bar.sync 5, 256;" ::: "memory");
}

__device__ __forceinline__ void flashinfer_stage_key_tile(
    const __half* __restrict__ key_panel, __half* __restrict__ stage) {
  const uint4* source = reinterpret_cast<const uint4*>(key_panel);
  uint4* destination = reinterpret_cast<uint4*>(stage);
  const int vector = 2 * threadIdx.x;
  destination[vector] = __ldg(source + vector);
  destination[vector + 1] = __ldg(source + vector + 1);
}

__device__ __forceinline__ void flashinfer_load_value_payload(
    const __half* __restrict__ value_panel, int vector_offset,
    FlashInferValuePayload& payload) {
  const uint4* source = reinterpret_cast<const uint4*>(value_panel);
  payload.vector_0 = __ldg(source + vector_offset);
  payload.vector_1 = __ldg(source + vector_offset + 1);
  payload.vector_2 = __ldg(source + vector_offset + 2);
  payload.vector_3 = __ldg(source + vector_offset + 3);
}

__device__ __forceinline__ void flashinfer_store_value_payload(
    const FlashInferValuePayload& payload, int vector_offset,
    __half* __restrict__ stage) {
  uint4* destination = reinterpret_cast<uint4*>(stage);
  destination[vector_offset] = payload.vector_0;
  destination[vector_offset + 1] = payload.vector_1;
  destination[vector_offset + 2] = payload.vector_2;
  destination[vector_offset + 3] = payload.vector_3;
}

__device__ __forceinline__ void flashinfer_qk_n16(
    const __half* __restrict__ shared_query,
    const __half* __restrict__ key_stage, AccumulatorFragment& accumulator) {
  MatrixAFragment query_fragment;
  QKMatrixBFragment key_fragment;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kD; k_offset += kPanelK) {
    nvcuda::wmma::load_matrix_sync(key_fragment, key_stage + k_offset, kD);
    load_swizzled_matrix_a_fragment(query_fragment, shared_query, k_offset);
    nvcuda::wmma::mma_sync(accumulator, query_fragment, key_fragment,
                           accumulator);
  }
}

__device__ __forceinline__ void flashinfer_exponentiate_probability(
    AccumulatorFragment& fragment, float first_row_max,
    float second_row_max) {
#pragma unroll
  for (int element = 0; element < fragment.num_elements; ++element) {
    const float row_max =
        (element & 2) == 0 ? first_row_max : second_row_max;
    fragment.x[element] =
        __expf(fmaxf(fragment.x[element] - row_max, -80.0f));
  }
}

__device__ __forceinline__ uint32_t flashinfer_pack_probability_pair(
    float first, float second) {
  const uint32_t low = static_cast<uint32_t>(
      __half_as_ushort(__float2half_rn(first)));
  const uint32_t high = static_cast<uint32_t>(
      __half_as_ushort(__float2half_rn(second)));
  return low | (high << 16);
}

__device__ __forceinline__ void flashinfer_pack_probability_values(
    const AccumulatorFragment& fragment, FlashInferPackedProbability& packed) {
  packed.word_0 =
      flashinfer_pack_probability_pair(fragment.x[0], fragment.x[1]);
  packed.word_1 =
      flashinfer_pack_probability_pair(fragment.x[2], fragment.x[3]);
  packed.word_2 =
      flashinfer_pack_probability_pair(fragment.x[4], fragment.x[5]);
  packed.word_3 =
      flashinfer_pack_probability_pair(fragment.x[6], fragment.x[7]);
}

__device__ __forceinline__ void flashinfer_store_probability_pair(
    __half* __restrict__ probability, int lane, int element,
    uint32_t packed_values) {
  const int first_row = accumulator_fragment_row(lane, element);
  const int first_column = accumulator_fragment_column(lane, element);
  const int second_row = accumulator_fragment_row(lane, element + 1);
  const int second_column = accumulator_fragment_column(lane, element + 1);
  probability[swizzled_matrix_a_offset(first_row, first_column)] =
      __ushort_as_half(static_cast<unsigned short>(packed_values));
  probability[swizzled_matrix_a_offset(second_row, second_column)] =
      __ushort_as_half(static_cast<unsigned short>(packed_values >> 16));
}

__device__ __forceinline__ void flashinfer_store_packed_probability(
    const FlashInferPackedProbability& packed, __half* __restrict__ probability) {
  const int lane = threadIdx.x & 31;
  flashinfer_store_probability_pair(probability, lane, 0, packed.word_0);
  flashinfer_store_probability_pair(probability, lane, 2, packed.word_1);
  flashinfer_store_probability_pair(probability, lane, 4, packed.word_2);
  flashinfer_store_probability_pair(probability, lane, 6, packed.word_3);
}

// N16 alone is not bitwise-compatible with the reference: it would round the
// first P tile using max(N0), while BM32 rounds both P tiles using max(N0,N1).
// Keep two physical N16 QK results live and apply the reference N32 recurrence.
__device__ __forceinline__ void flashinfer_prepare_n32_softmax(
    AccumulatorFragment& qk_0, AccumulatorFragment& qk_1,
    __half* __restrict__ probability, float& first_row_max,
    float& first_row_sum, float& second_row_max, float& second_row_sum,
    float& first_row_exp_diff, float& second_row_exp_diff) {
  const float panel_max_first =
      q_owner_reduce_panel_row<true>(qk_0, qk_1, 0);
  const float panel_max_second =
      q_owner_reduce_panel_row<true>(qk_0, qk_1, 2);
  const float new_max_first = fmaxf(first_row_max, panel_max_first);
  const float new_max_second = fmaxf(second_row_max, panel_max_second);
  first_row_exp_diff = __expf(first_row_max - new_max_first);
  second_row_exp_diff = __expf(second_row_max - new_max_second);

  q_owner_write_probability(qk_0, probability, 0, new_max_first,
                            new_max_second);
  flashinfer_exponentiate_probability(qk_1, new_max_first, new_max_second);
  const float panel_sum_first =
      q_owner_reduce_panel_row<false>(qk_0, qk_1, 0);
  const float panel_sum_second =
      q_owner_reduce_panel_row<false>(qk_0, qk_1, 2);
  first_row_sum = first_row_exp_diff * first_row_sum + panel_sum_first;
  second_row_sum = second_row_exp_diff * second_row_sum + panel_sum_second;
  first_row_max = new_max_first;
  second_row_max = new_max_second;
}

#define FLASHINFER_PIPELINE_D_TILES(OP) \
  OP(0)                                  \
  OP(1)                                  \
  OP(2)                                  \
  OP(3)                                  \
  OP(4)                                  \
  OP(5)                                  \
  OP(6)                                  \
  OP(7)

__device__ __forceinline__ void flashinfer_scale_output_fragments(
    AccumulatorFragment& accumulator_0, AccumulatorFragment& accumulator_1,
    AccumulatorFragment& accumulator_2, AccumulatorFragment& accumulator_3,
    AccumulatorFragment& accumulator_4, AccumulatorFragment& accumulator_5,
    AccumulatorFragment& accumulator_6, AccumulatorFragment& accumulator_7,
    float first_row_exp_diff, float second_row_exp_diff) {
#define FLASHINFER_SCALE_OUTPUT(INDEX)                                  \
  scale_accumulator_two_rows(accumulator_##INDEX, first_row_exp_diff,   \
                             second_row_exp_diff);
  FLASHINFER_PIPELINE_D_TILES(FLASHINFER_SCALE_OUTPUT)
#undef FLASHINFER_SCALE_OUTPUT
}

__device__ __forceinline__ void flashinfer_pv_n16(
    const __half* __restrict__ probability, const __half* __restrict__ value,
    int d_half, AccumulatorFragment& accumulator_0,
    AccumulatorFragment& accumulator_1, AccumulatorFragment& accumulator_2,
    AccumulatorFragment& accumulator_3, AccumulatorFragment& accumulator_4,
    AccumulatorFragment& accumulator_5, AccumulatorFragment& accumulator_6,
    AccumulatorFragment& accumulator_7) {
  MatrixAFragment probability_fragment;
  PVMatrixBFragment value_fragment;
  load_swizzled_matrix_a_fragment(probability_fragment, probability, 0);
#define FLASHINFER_UPDATE_OUTPUT(INDEX)                                 \
  nvcuda::wmma::load_matrix_sync(                                      \
      value_fragment, value + (d_half * 8 + INDEX) * kPanelN, kD);     \
  nvcuda::wmma::mma_sync(accumulator_##INDEX, probability_fragment,    \
                         value_fragment, accumulator_##INDEX);
  FLASHINFER_PIPELINE_D_TILES(FLASHINFER_UPDATE_OUTPUT)
#undef FLASHINFER_UPDATE_OUTPUT
}

__device__ __forceinline__ void flashinfer_store_output_fragments(
    const AccumulatorFragment& accumulator_0,
    const AccumulatorFragment& accumulator_1,
    const AccumulatorFragment& accumulator_2,
    const AccumulatorFragment& accumulator_3,
    const AccumulatorFragment& accumulator_4,
    const AccumulatorFragment& accumulator_5,
    const AccumulatorFragment& accumulator_6,
    const AccumulatorFragment& accumulator_7, __half* __restrict__ output,
    int d_half, float first_row_sum, float second_row_sum) {
#define FLASHINFER_STORE_OUTPUT(INDEX)                                  \
  q_owner_store_output_tile(accumulator_##INDEX, output,               \
                            (d_half * 8 + INDEX) * kPanelN,             \
                            first_row_sum, second_row_sum);
  FLASHINFER_PIPELINE_D_TILES(FLASHINFER_STORE_OUTPUT)
#undef FLASHINFER_STORE_OUTPUT
}

extern "C" __global__ __launch_bounds__(kPipelineThreads, 2)
void sm70_flashinfer_bm64_pipeline_candidate(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, __half* __restrict__ output, int groups,
    int nblocks) {
  __shared__ __align__(16) FlashInferBm64PipelineShared shared;

  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int owner = warp >> 1;
  const bool qk_warp = (warp & 1) == 0;
  const int owner_thread = (warp & 1) * 32 + lane;

  const __half* query_panel =
      query + static_cast<int64_t>(group) * kM64QElements +
      owner * kQPanelElements;
  __half* shared_query = shared.query + owner * kQPanelElements;
  __half* probability = shared.probability[owner];
  stage_swizzled_q_panel(query_panel, shared_query, owner_thread, 64);
  if (owner_thread < kPanelM) {
    const int row = owner * kPanelM + owner_thread;
    shared.row_max[row] = kNegativeInfinity;
    shared.row_sum[row] = 0.0f;
    shared.row_exp_diff[row] = 1.0f;
  }
  flashinfer_owner_q_barrier(owner);

#define FLASHINFER_DECLARE_OUTPUT(INDEX) AccumulatorFragment accumulator_##INDEX;
  FLASHINFER_PIPELINE_D_TILES(FLASHINFER_DECLARE_OUTPUT)
#undef FLASHINFER_DECLARE_OUTPUT
#define FLASHINFER_FILL_OUTPUT(INDEX) \
  nvcuda::wmma::fill_fragment(accumulator_##INDEX, 0.0f);
  FLASHINFER_PIPELINE_D_TILES(FLASHINFER_FILL_OUTPUT)
#undef FLASHINFER_FILL_OUTPUT

  FlashInferRoleState role_state;
  if (qk_warp) {
    role_state.qk.first_row_max = kNegativeInfinity;
    role_state.qk.first_row_sum = 0.0f;
    role_state.qk.second_row_max = kNegativeInfinity;
    role_state.qk.second_row_sum = 0.0f;
  }
  const __half* key_group =
      key + static_cast<int64_t>(group) * nblocks * kBlockN * kD;
  const int logical_panels = nblocks * (kBlockN / kPipelineLogicalN);

  for (int panel = 0; panel < logical_panels; ++panel) {
    const int token_0 = panel * kPipelineLogicalN;

    flashinfer_stage_key_tile(key_group + token_0 * kD, shared.kv_stage);
    flashinfer_pipeline_barrier();

    if (qk_warp) {
      AccumulatorFragment qk_0;
      AccumulatorFragment qk_1;
      flashinfer_qk_n16(shared_query, shared.kv_stage, qk_0);
      // qk_0 is intentionally scoped to the first physical N16 lifetime.
      flashinfer_pipeline_barrier();

      flashinfer_stage_key_tile(
          key_group + (token_0 + kPipelinePhysicalN) * kD, shared.kv_stage);
      flashinfer_pipeline_barrier();
      flashinfer_qk_n16(shared_query, shared.kv_stage, qk_1);
      flashinfer_pipeline_barrier();

      float first_row_exp_diff;
      float second_row_exp_diff;
      flashinfer_prepare_n32_softmax(
          qk_0, qk_1, probability, role_state.qk.first_row_max,
          role_state.qk.first_row_sum, role_state.qk.second_row_max,
          role_state.qk.second_row_sum, first_row_exp_diff,
          second_row_exp_diff);
      flashinfer_pack_probability_values(qk_1, role_state.qk.probability_1);
      if ((lane & 0xa) == 0) {
        const int first_row = accumulator_fragment_row(lane, 0);
        const int first_state_row = owner * kPanelM + first_row;
        const int second_state_row = first_state_row + 2;
        shared.row_exp_diff[first_state_row] = first_row_exp_diff;
        shared.row_exp_diff[second_state_row] = second_row_exp_diff;
        shared.row_max[first_state_row] = role_state.qk.first_row_max;
        shared.row_max[second_state_row] = role_state.qk.second_row_max;
        shared.row_sum[first_state_row] = role_state.qk.first_row_sum;
        shared.row_sum[second_state_row] = role_state.qk.second_row_sum;
      }
    } else {
      const int value_vector_offset =
          (owner * 32 + lane) * kPipelineValueVectorsPerPartnerThread;
      const __half* value_panel =
          value + (static_cast<int64_t>(group) * nblocks * kBlockN + token_0) *
                      kD;
      flashinfer_load_value_payload(value_panel, value_vector_offset,
                                    role_state.value_payload);
      flashinfer_pipeline_barrier();

      flashinfer_stage_key_tile(
          key_group + (token_0 + kPipelinePhysicalN) * kD, shared.kv_stage);
      flashinfer_pipeline_barrier();
      flashinfer_pipeline_barrier();

      flashinfer_store_value_payload(role_state.value_payload,
                                     value_vector_offset, shared.kv_stage);
    }
    flashinfer_pipeline_barrier();

    if (panel != 0) {
      const int first_row = accumulator_fragment_row(lane, 0);
      flashinfer_scale_output_fragments(
          accumulator_0, accumulator_1, accumulator_2, accumulator_3,
          accumulator_4, accumulator_5, accumulator_6, accumulator_7,
          shared.row_exp_diff[owner * kPanelM + first_row],
          shared.row_exp_diff[owner * kPanelM + first_row + 2]);
    }
    flashinfer_pv_n16(probability, shared.kv_stage, qk_warp ? 0 : 1,
                       accumulator_0, accumulator_1, accumulator_2,
                       accumulator_3, accumulator_4, accumulator_5,
                       accumulator_6, accumulator_7);
    flashinfer_pipeline_barrier();

    if (qk_warp) {
      flashinfer_store_packed_probability(role_state.qk.probability_1,
                                           probability);
    } else {
      const int value_vector_offset =
          (owner * 32 + lane) * kPipelineValueVectorsPerPartnerThread;
      const __half* value_panel = value +
          (static_cast<int64_t>(group) * nblocks * kBlockN + token_0 +
           kPipelinePhysicalN) *
              kD;
      flashinfer_load_value_payload(value_panel, value_vector_offset,
                                    role_state.value_payload);
      flashinfer_store_value_payload(role_state.value_payload,
                                     value_vector_offset, shared.kv_stage);
    }
    flashinfer_pipeline_barrier();

    flashinfer_pv_n16(probability, shared.kv_stage, qk_warp ? 0 : 1,
                       accumulator_0, accumulator_1, accumulator_2,
                       accumulator_3, accumulator_4, accumulator_5,
                       accumulator_6, accumulator_7);
    if (panel + 1 != logical_panels) {
      flashinfer_pipeline_barrier();
    }
  }

  __half* output_panel =
      output + static_cast<int64_t>(group) * kM64OutputElements +
      owner * kQPanelElements;
  const int first_row = accumulator_fragment_row(lane, 0);
  flashinfer_store_output_fragments(
      accumulator_0, accumulator_1, accumulator_2, accumulator_3,
      accumulator_4, accumulator_5, accumulator_6, accumulator_7,
      output_panel, qk_warp ? 0 : 1,
      shared.row_sum[owner * kPanelM + first_row],
      shared.row_sum[owner * kPanelM + first_row + 2]);
}

#undef FLASHINFER_PIPELINE_D_TILES

bool flashinfer_pipeline_resource_gate(const KernelResources& resources) {
  return resources.registers_per_thread <= 128 &&
         resources.local_bytes_per_thread == 0 &&
         resources.static_shared_bytes == sizeof(FlashInferBm64PipelineShared) &&
         resources.dynamic_shared_bytes == 0 &&
         resources.active_ctas_per_sm == 2;
}

void print_flashinfer_pipeline_json(
    const Args& args, const cudaDeviceProp& properties, int runtime_version,
    int sm_count, const KernelResources& baseline_resources,
    const KernelResources& candidate_resources, bool executed,
    bool exactness_available, const Exactness& exactness,
    bool timing_available, const TimingSummary& baseline_timing,
    const TimingSummary& candidate_timing, const PairSummary& pairs) {
  const bool resource_pass = bm32_resource_gate(baseline_resources) &&
                             flashinfer_pipeline_resource_gate(
                                 candidate_resources);
  const double speedup = timing_available
                             ? 100.0 * (baseline_timing.median_us -
                                        candidate_timing.median_us) /
                                   baseline_timing.median_us
                             : 0.0;
  std::cout << "{\n";
  std::cout << "  \"benchmark\": \"sm70_flashinfer_bm64_pipeline_micro\",\n";
  std::cout << "  \"target\": \"sm_70\",\n";
  std::cout << "  \"device\": {\"logical_index\": " << args.device
            << ", \"name\": ";
  print_json_string(properties.name);
  std::cout << ", \"capability\": [" << properties.major << ", "
            << properties.minor << "], \"cuda_runtime\": "
            << runtime_version << ", \"sm_count\": " << sm_count << "},\n";
  std::cout << "  \"shape\": {\"groups\": " << args.groups
            << ", \"nblocks\": " << args.nblocks
            << ", \"M\": 64, \"D\": 256, \"BN\": 128, \"N\": "
            << args.nblocks * kBlockN << "},\n";
  std::cout << "  \"input_pattern\": ";
  print_json_string(args.pattern);
  std::cout << ",\n";
  std::cout << "  \"paths\": {\n";
  std::cout << "    \"baseline\": \"2 CTA/M64 group; BM32; "
               "ALL_P=true PAIR_SCRATCH=true; reused from "
               "sm70_native_bm64_allp_micro.cu\",\n";
  std::cout << "    \"candidate\": \"1 CTA/M64 GQA-packed BM64; BN16 "
               "physical tiles; eight warps\"\n";
  std::cout << "  },\n";
  std::cout << "  \"candidate_contract\": {"
            << "\"warps\": 8, \"m16_owners\": 4, "
            << "\"qk_warps\": \"even: D0-127\", "
            << "\"partner_warps\": \"odd: D128-255\", "
            << "\"d16_fp32_output_fragments_per_warp\": 8, "
            << "\"fp32_output_values_per_warp\": "
            << kPipelineOutputFp32ValuesPerWarp << ", "
            << "\"physical_bn\": 16, \"logical_softmax_bn\": 32, "
            << "\"value_payload_gprs_per_partner\": 16, "
            << "\"q_shared_bytes\": " << kM64QElements * sizeof(__half)
            << ", \"reusable_kv_stage_bytes\": "
            << kPipelinePhysicalN * kD * sizeof(__half)
            << ", \"probability_bytes\": "
            << kPipelineOwners * kPipelineProbabilityElements * sizeof(__half)
            << ", \"row_state_bytes\": " << kPipelineRowStateBytes
            << ", \"total_shared_bytes\": "
            << sizeof(FlashInferBm64PipelineShared) << ", "
            << "\"barriers\": {\"q_ready\": \"4x bar.sync count=64\", "
            << "\"stage_epochs\": \"bar.sync id=5 count=256\"}, "
            << "\"bn16_bitwise_incompatibility_proof\": "
            << "\"BN16 rounds P(N0) with max(N0) and updates FP32 sum before "
               "N1; the BM32 reference rounds both tiles with max(N0,N1) and "
               "reduces one N32 sum, so only a two-N16 logical boundary preserves "
               "the reference recurrence\"},\n";
  std::cout << "  \"execution\": {\"resource_gate_pass\": "
            << (resource_pass ? "true" : "false")
            << ", \"kernels_executed\": "
            << (executed ? "true" : "false") << "},\n";
  std::cout << "  \"resources\": {\"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ", \"candidate\": ";
  print_resources(candidate_resources);
  std::cout << "},\n";
  std::cout << "  \"exactness\": ";
  if (exactness_available) {
    print_exactness(exactness,
                    static_cast<int64_t>(args.groups) * kM64OutputElements /
                        2);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"timing\": ";
  if (timing_available) {
    std::cout << "{\"unit\": \"us per grid launch\", \"baseline\": ";
    print_timing(baseline_timing);
    std::cout << ", \"candidate\": ";
    print_timing(candidate_timing);
    std::cout << ", \"candidate_speedup_vs_baseline_pct\": " << speedup
              << '}';
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"pairs\": ";
  if (timing_available) {
    std::cout << "{\"count\": " << pairs.count
              << ", \"candidate_faster\": " << pairs.candidate_faster
              << ", \"baseline_faster\": " << pairs.baseline_faster
              << ", \"ties\": " << pairs.ties
              << ", \"candidate_minus_baseline_median_us\": "
              << pairs.candidate_minus_baseline_median_us << '}';
  } else {
    std::cout << "null";
  }
  std::cout << "\n}\n";
}

int run_flashinfer_pipeline(const Args& args) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (args.device < 0 || args.device >= device_count) {
    std::cerr << "Requested logical CUDA device " << args.device
              << " is unavailable; visible device count is " << device_count
              << '\n';
    return EXIT_FAILURE;
  }
  if (args.groups < 1 || args.nblocks < 1 || args.warmup < 0 ||
      args.rounds < 1 || args.launches_per_sample < 1) {
    std::cerr << "groups, nblocks, rounds, and launches must be positive; "
                 "warmup cannot be negative\n";
    return EXIT_FAILURE;
  }
  if (args.pattern != "random" && args.pattern != "alternating") {
    std::cerr << "--pattern must be random or alternating\n";
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
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    args.device));
  const KernelResources baseline_resources = query_resources(
      sm70_native_bm64_allp_bm32_pair_scratch_baseline, kBaselineThreads);
  const KernelResources candidate_resources = query_resources(
      sm70_flashinfer_bm64_pipeline_candidate, kPipelineThreads);
  const Exactness no_exactness;
  const TimingSummary no_timing;
  const PairSummary no_pairs;
  if (!bm32_resource_gate(baseline_resources) ||
      !flashinfer_pipeline_resource_gate(candidate_resources)) {
    print_flashinfer_pipeline_json(
        args, properties, runtime_version, sm_count, baseline_resources,
        candidate_resources, false, false, no_exactness, false, no_timing,
        no_timing, no_pairs);
    return EXIT_FAILURE;
  }

  const size_t query_elements =
      static_cast<size_t>(args.groups) * kM64QElements;
  const size_t kv_elements = static_cast<size_t>(args.groups) * args.nblocks *
                             kBlockN * kD;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kM64OutputElements;
  std::vector<__half> host_query(query_elements);
  std::vector<__half> host_key(kv_elements);
  std::vector<__half> host_value(kv_elements);
  uint32_t random_state = 0x6d2b79f5u;
  fill_input(&host_query, args.pattern, &random_state, 1);
  fill_input(&host_key, args.pattern, &random_state, 2);
  fill_input(&host_value, args.pattern, &random_state, 3);

  __half* device_query = nullptr;
  __half* device_key = nullptr;
  __half* device_value = nullptr;
  __half* device_baseline = nullptr;
  __half* device_candidate = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_query),
                        query_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_key),
                        kv_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_value),
                        kv_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_baseline),
                        output_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_candidate),
                        output_elements * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(device_query, host_query.data(),
                        query_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_key, host_key.data(), kv_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_value, host_value.data(),
                        kv_elements * sizeof(__half), cudaMemcpyHostToDevice));

  const dim3 baseline_grid(args.groups * 2);
  const dim3 candidate_grid(args.groups);
  auto launch_baseline = [&] {
    sm70_native_bm64_allp_bm32_pair_scratch_baseline<<<baseline_grid,
                                                         kBaselineThreads>>>(
        device_query, device_key, device_value, device_baseline, args.groups,
        args.nblocks);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_candidate = [&] {
    sm70_flashinfer_bm64_pipeline_candidate<<<candidate_grid,
                                               kPipelineThreads>>>(
        device_query, device_key, device_value, device_candidate, args.groups,
        args.nblocks);
    CUDA_CHECK(cudaGetLastError());
  };
  auto free_device_buffers = [&] {
    CUDA_CHECK(cudaFree(device_candidate));
    CUDA_CHECK(cudaFree(device_baseline));
    CUDA_CHECK(cudaFree(device_value));
    CUDA_CHECK(cudaFree(device_key));
    CUDA_CHECK(cudaFree(device_query));
  };

  if (args.profile_only) {
    if (profile_kernel_selected(args, "baseline")) {
      launch_baseline();
    }
    if (profile_kernel_selected(args, "candidate")) {
      launch_candidate();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    free_device_buffers();
    print_flashinfer_pipeline_json(
        args, properties, runtime_version, sm_count, baseline_resources,
        candidate_resources, true, false, no_exactness, false, no_timing,
        no_timing, no_pairs);
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
  std::vector<__half> host_baseline(output_elements);
  std::vector<__half> host_candidate(output_elements);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), device_baseline,
                        output_elements * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_candidate.data(), device_candidate,
                        output_elements * sizeof(__half), cudaMemcpyDeviceToHost));
  const Exactness exactness = compare_outputs(host_baseline, host_candidate);
  if (!exactness.bitwise_equal) {
    free_device_buffers();
    print_flashinfer_pipeline_json(
        args, properties, runtime_version, sm_count, baseline_resources,
        candidate_resources, true, true, exactness, false, no_timing, no_timing,
        no_pairs);
    return EXIT_FAILURE;
  }
  if (args.smoke_only) {
    free_device_buffers();
    print_flashinfer_pipeline_json(
        args, properties, runtime_version, sm_count, baseline_resources,
        candidate_resources, true, true, exactness, false, no_timing, no_timing,
        no_pairs);
    return EXIT_SUCCESS;
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

  const TimingSummary baseline_timing = summarize(baseline_samples);
  const TimingSummary candidate_timing = summarize(candidate_samples);
  const PairSummary pairs =
      summarize_pairs(baseline_samples, candidate_samples);
  free_device_buffers();
  print_flashinfer_pipeline_json(
      args, properties, runtime_version, sm_count, baseline_resources,
      candidate_resources, true, true, exactness, true, baseline_timing,
      candidate_timing, pairs);
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  return run_flashinfer_pipeline(parse_args(argc, argv));
}
