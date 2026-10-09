// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Host-only policy scope. Graph replay records kernels and never borrows its
// address. A configured binding owns both immutable choices and observations.
namespace flash_v100::policy {
enum class Field {
  xqa_padded_smem_enabled,
  xqa_g6_dual_cta_enabled,
  xqa_e4m3_batch_enabled,
  xqa_e4m3_batch_optimized_enabled,
  xqa_e4m3_page800_fastpath_enabled,
  xqa_e4m3_page800_fastpath_trace_enabled,
  xqa_e5m2_g6_dual_cta_enabled,
  xqa_e5m2_g6_split_reduce_enabled,
  xqa_e5m2_partition_page_ids_enabled,
  xqa_e5m2_pair_load_enabled,
  xqa_e5m2_batch_wide_load_enabled,
  dflash2_grouped_fixed_interleaved_enabled,
  dflash2_grouped_stage_page_ids_enabled,
  xqa_e5m2_p1024_begin,
  xqa_e5m2_scalar_xqa_seq_len,
  xqa_e5m2_g6_dual_cta_trace_enabled,
  xqa_mtp5_dual_cta_enabled,
  xqa_g6_dual_cta_dense_enabled,
  xqa_g6_p1024_auto_enabled,
  xqa_g6_p1024_auto_trace_enabled,
  xqa_g6_p1024_sawtooth_enabled,
  xqa_e4m3_g6_p64_p256_auto_enabled,
  xqa_e4m3_g6_p64_p256_auto_trace_enabled,
  xqa_e4m3_g6_p256_begin,
  xqa_e4m3_g6_dual_cta_begin,
  xqa_e4m3_g6_wave_partitions_enabled,
  xqa_e4m3_g6_merged_wave_launch_enabled,
  xqa_e4m3_g6_p512_begin,
  xqa_e4m3_g6_p896_begin,
  xqa_e4m3_g6_p1664_begin,
  decode_partition_size_overridden,
  xqa_g6_qk_pipeline_enabled,
  xqa_g6_qk_pipeline_warps,
  xqa_g6_qk_pipeline_trace_enabled,
  xqa_g6_p1024_sawtooth_trace_enabled,
  xqa_g6_p1024_sawtooth_p1024_mid_seq_len,
  xqa_g6_p1024_sawtooth_p256_long_seq_len,
  xqa_g6_p1024_sawtooth_p1024_final_seq_len,
  xqa_split_reduce_enabled,
  xqa_batch_context_routing_enabled,
  xqa_batch_context_routing_trace_enabled,
  xqa_block16_layout_mode,
  xqa_block16_layout_required,
  xqa_block16_layout_trace_enabled,
  xqa_block784_index_enabled,
  xqa_block784_index_trace_enabled,
  xqa_aligned_padded_smem_enabled,
  xqa_aligned_padded_smem_trace_enabled,
  xqa_split_reduce_dim_tile,
  dense_d256_wmma_qk_default_on,
  dense_d256_low_smem_off,
  prefill_d256_bm32_all_p_default_on,
  prefill_d256_bm32_pair_scratch_default_on,
  prefill_d256_low_smem_default_on,
  prefill_d256_bm32_phase_default_on,
  prefill_contig_fast_off,
  prefill_d256_scalar_qk_off,
  prefill_d256_bm32_off,
  prefill_d256_output_stride_268_default_on,
  prefill_d256_output_stride_268_off,
  prefill_d256_software_pipeline_off,
  prefill_d256_sw_pipeline_qk_default_on,
  prefill_d256_sw_pipeline_pv_default_on,
  prefill_scalar_pv,
  e4m3_scalar_fast,
  tp2_e4m3_scalar_fast,
  count
};
inline constexpr size_t size = static_cast<size_t>(Field::count);
inline constexpr const char* names[] = {
    "VLLM_FLASH_V100_XQA_PADDED_SMEM",
    "VLLM_FLASH_V100_XQA_G6_DUAL_CTA",
    "VLLM_FLASH_V100_E4M3_BATCH_XQA",
    "VLLM_FLASH_V100_E4M3_BATCH_XQA_OPTIMIZED",
    "VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH",
    "VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH_TRACE",
    "VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA",
    "VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE",
    "VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS",
    "VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD",
    "VLLM_FLASH_V100_XQA_E5M2_BATCH_WIDE_LOAD",
    "VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED",
    "VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS",
    "VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN",
    "VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN",
    "VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA_TRACE",
    "VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA",
    "VLLM_FLASH_V100_XQA_G6_DUAL_CTA_DENSE",
    "VLLM_FLASH_V100_XQA_G6_P1024_AUTO",
    "VLLM_FLASH_V100_XQA_G6_P1024_AUTO_TRACE",
    "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH",
    "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO",
    "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO_TRACE",
    "VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN",
    "VLLM_FLASH_V100_XQA_E4M3_G6_DUAL_CTA_BEGIN",
    "VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS",
    "VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH",
    "VLLM_FLASH_V100_XQA_E4M3_G6_P512_BEGIN",
    "VLLM_FLASH_V100_XQA_E4M3_G6_P896_BEGIN",
    "VLLM_FLASH_V100_XQA_E4M3_G6_P1664_BEGIN",
    "VLLM_FLASH_V100_DECODE_PARTITION_SIZE",
    "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE",
    "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_WARPS",
    "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_TRACE",
    "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_TRACE",
    "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_MID_SEQ_LEN",
    "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P256_LONG_SEQ_LEN",
    "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_FINAL_SEQ_LEN",
    "VLLM_FLASH_V100_XQA_SPLIT_REDUCE",
    "VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING",
    "VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING_TRACE",
    "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT",
    "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_REQUIRE",
    "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_TRACE",
    "VLLM_FLASH_V100_XQA_BLOCK784_INDEX",
    "VLLM_FLASH_V100_XQA_BLOCK784_INDEX_TRACE",
    "VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM",
    "VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM_TRACE",
    "VLLM_FLASH_V100_XQA_SPLIT_REDUCE_D_TILE",
    "VLLM_FLASH_V100_DENSE_D256_WMMA_QK",
    "VLLM_FLASH_V100_DENSE_D256_LOW_SMEM",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH",
    "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE",
    "VLLM_FLASH_V100_PREFILL_CONTIG_FAST",
    "VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK",
    "VLLM_FLASH_V100_PREFILL_D256_BM32",
    "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268",
    "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268",
    "VLLM_FLASH_V100_PREFILL_D256_SOFTWARE_PIPELINE",
    "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK",
    "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV",
    "VLLM_FLASH_V100_PREFILL_SCALAR_PV",
    "VLLM_FLASH_V100_E4M3_SCALAR_FAST",
    "VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST",
};
inline int parse(Field field, const std::array<const char*, size>& raw) {
  switch (field) {
    case Field::xqa_padded_smem_enabled: {
      const char* value = raw[0];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_g6_dual_cta_enabled: {
      const char* value = raw[1];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_e4m3_batch_enabled: {
      const char* value = raw[2];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e4m3_batch_optimized_enabled: {
      const char* value = raw[3];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e4m3_page800_fastpath_enabled: {
      const char* value = raw[4];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e4m3_page800_fastpath_trace_enabled: {
      const char* value = raw[5];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_e5m2_g6_dual_cta_enabled: {
      const char* value = raw[6];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e5m2_g6_split_reduce_enabled: {
      const char* value = raw[7];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e5m2_partition_page_ids_enabled: {
      const char* value = raw[8];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e5m2_pair_load_enabled: {
      const char* value = raw[9];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e5m2_batch_wide_load_enabled: {
      const char* value = raw[10];
      return value == nullptr || value[0] != '0';
    }
    case Field::dflash2_grouped_fixed_interleaved_enabled: {
      const char* value = raw[11];
      return value == nullptr || value[0] != '0';
    }
    case Field::dflash2_grouped_stage_page_ids_enabled: {
      const char* value = raw[12];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e5m2_p1024_begin: {
      const char* value = raw[13];
      return value == nullptr ? 61633 : std::max(1, std::atoi(value));
    }
    case Field::xqa_e5m2_scalar_xqa_seq_len: {
      const char* value = raw[14];
      return value == nullptr ? 16384 : std::max(1, std::atoi(value));
    }
    case Field::xqa_e5m2_g6_dual_cta_trace_enabled: {
      const char* value = raw[15];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_mtp5_dual_cta_enabled: {
      const char* value = raw[16];
      return value == nullptr || value[0] == '1';
    }
    case Field::xqa_g6_dual_cta_dense_enabled: {
      const char* value = raw[17];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_g6_p1024_auto_enabled: {
      const char* value = raw[18];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_g6_p1024_auto_trace_enabled: {
      const char* value = raw[19];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_g6_p1024_sawtooth_enabled: {
      const char* value = raw[20];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e4m3_g6_p64_p256_auto_enabled: {
      const char* value = raw[21];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e4m3_g6_p64_p256_auto_trace_enabled: {
      const char* value = raw[22];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_e4m3_g6_p256_begin: {
      const char* value = raw[23];
      return value == nullptr ? 12288 : std::max(1, std::atoi(value));
    }
    case Field::xqa_e4m3_g6_dual_cta_begin: {
      const char* value = raw[24];
      return value == nullptr ? 32768 : std::max(1, std::atoi(value));
    }
    case Field::xqa_e4m3_g6_wave_partitions_enabled: {
      const char* value = raw[25];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e4m3_g6_merged_wave_launch_enabled: {
      const char* value = raw[26];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_e4m3_g6_p512_begin: {
      const char* value = raw[27];
      return value == nullptr ? 49152 : std::max(1, std::atoi(value));
    }
    case Field::xqa_e4m3_g6_p896_begin: {
      const char* value = raw[28];
      return value == nullptr ? 98304 : std::max(1, std::atoi(value));
    }
    case Field::xqa_e4m3_g6_p1664_begin: {
      const char* value = raw[29];
      return value == nullptr ? 196608 : std::max(1, std::atoi(value));
    }
    case Field::decode_partition_size_overridden: {
      return raw[30] != nullptr;
    }
    case Field::xqa_g6_qk_pipeline_enabled: {
      const char* value = raw[31];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_g6_qk_pipeline_warps: {
      const char* value = raw[32];
      return value != nullptr && std::atoi(value) == 6 ? 6 : 8;
    }
    case Field::xqa_g6_qk_pipeline_trace_enabled: {
      const char* value = raw[33];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_g6_p1024_sawtooth_trace_enabled: {
      const char* value = raw[34];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_g6_p1024_sawtooth_p1024_mid_seq_len: {
      const char* value = raw[35];
      return value == nullptr ? 111104 : std::max(1, std::atoi(value));
    }
    case Field::xqa_g6_p1024_sawtooth_p256_long_seq_len: {
      const char* value = raw[36];
      return value == nullptr ? 147841 : std::max(1, std::atoi(value));
    }
    case Field::xqa_g6_p1024_sawtooth_p1024_final_seq_len: {
      const char* value = raw[37];
      return value == nullptr ? 258176 : std::max(1, std::atoi(value));
    }
    case Field::xqa_split_reduce_enabled: {
      const char* value = raw[38];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_batch_context_routing_enabled: {
      const char* value = raw[39];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_batch_context_routing_trace_enabled: {
      const char* value = raw[40];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_block16_layout_mode: {
      const char* value = raw[41];
      if (value == nullptr) {
        return 0;
      }
      const int mode = std::atoi(value);
      return mode == 1 || mode == 2 ? mode : 0;
    }
    case Field::xqa_block16_layout_required: {
      const char* value = raw[42];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_block16_layout_trace_enabled: {
      const char* value = raw[43];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_block784_index_enabled: {
      const char* value = raw[44];
      return value == nullptr || value[0] != '0';
    }
    case Field::xqa_block784_index_trace_enabled: {
      const char* value = raw[45];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_aligned_padded_smem_enabled: {
      const char* value = raw[46];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_aligned_padded_smem_trace_enabled: {
      const char* value = raw[47];
      return value != nullptr && value[0] == '1';
    }
    case Field::xqa_split_reduce_dim_tile: {
      const char* value = raw[48];
      if (value == nullptr) {
        return 8;
      }
      const int dim_tile = std::atoi(value);
      return dim_tile == 8 || dim_tile == 16 || dim_tile == 32 ? dim_tile : 8;
    }
    case Field::dense_d256_wmma_qk_default_on: {
      return raw[49] == nullptr || std::strcmp(raw[49], "0") != 0;
    }
    case Field::dense_d256_low_smem_off: {
      return raw[50] != nullptr && std::strcmp(raw[50], "0") != 0;
    }
    case Field::prefill_d256_bm32_all_p_default_on: {
      return raw[51] == nullptr || std::strcmp(raw[51], "0") != 0;
    }
    case Field::prefill_d256_bm32_pair_scratch_default_on: {
      return raw[52] == nullptr || std::strcmp(raw[52], "0") != 0;
    }
    case Field::prefill_d256_low_smem_default_on: {
      return raw[53] == nullptr || std::strcmp(raw[53], "0") != 0;
    }
    case Field::prefill_d256_bm32_phase_default_on: {
      return raw[54] == nullptr || std::strcmp(raw[54], "0") != 0;
    }
    case Field::prefill_contig_fast_off: {
      return raw[55] != nullptr && std::strcmp(raw[55], "0") != 0;
    }
    case Field::prefill_d256_scalar_qk_off: {
      return raw[56] != nullptr && std::strcmp(raw[56], "0") != 0;
    }
    case Field::prefill_d256_bm32_off: {
      return raw[57] != nullptr && std::strcmp(raw[57], "0") != 0;
    }
    case Field::prefill_d256_output_stride_268_default_on: {
      return raw[58] == nullptr || std::strcmp(raw[58], "0") != 0;
    }
    case Field::prefill_d256_output_stride_268_off: {
      return raw[59] != nullptr && std::strcmp(raw[59], "0") != 0;
    }
    case Field::prefill_d256_software_pipeline_off: {
      return raw[60] != nullptr && std::strcmp(raw[60], "0") != 0;
    }
    case Field::prefill_d256_sw_pipeline_qk_default_on: {
      return raw[61] == nullptr || std::strcmp(raw[61], "0") != 0;
    }
    case Field::prefill_d256_sw_pipeline_pv_default_on: {
      return raw[62] == nullptr || std::strcmp(raw[62], "0") != 0;
    }
    case Field::prefill_scalar_pv: {
      const auto* value = raw[63];
      return value != nullptr && value[0] != '\0' && value[0] != '0';
    }
    case Field::e4m3_scalar_fast: {
      const auto* value = raw[64] ? raw[64] : raw[65];
      return value == nullptr || (value[0] == '1' && value[1] == '\0');
    }
    case Field::tp2_e4m3_scalar_fast: {
      return 0;
    }
    default:
      throw std::invalid_argument("Unknown Flash-V100 policy field");
  }
}
struct PreparedPolicy {
  std::array<int, size> values{};
  std::array<std::atomic<unsigned long long>, 16> observations{};
  explicit PreparedPolicy(
      const std::vector<std::optional<std::string>>& inputs) {
    if (inputs.size() != size)
      throw std::invalid_argument("Flash-V100 policy ABI size mismatch");
    std::array<const char*, size> raw{};
    for (size_t i = 0; i < size; ++i)
      raw[i] = inputs[i] ? inputs[i]->c_str() : nullptr;
    for (size_t i = 0; i < size; ++i)
      values[i] = parse(static_cast<Field>(i), raw);
  }
};
inline thread_local PreparedPolicy* active = nullptr;
class Scope {
 public:
  explicit Scope(PreparedPolicy& policy) : previous_(active) {
    active = &policy;
  }
  ~Scope() { active = previous_; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  PreparedPolicy* previous_;
};
inline int value(Field field) {
  if (active) return active->values[static_cast<size_t>(field)];
  // Independently retained direct-call compatibility. Configured engines never
  // enter this adapter or read the process environment during execution.
  std::array<const char*, size> raw{};
  const size_t index = static_cast<size_t>(field);
  raw[index] = std::getenv(names[index]);
  if (field == Field::e4m3_scalar_fast)
    raw[index + 1] = std::getenv(names[index + 1]);
  return parse(field, raw);
}
inline std::atomic<unsigned long long>& observation(size_t slot) {
  static std::array<std::atomic<unsigned long long>, 16> standalone{};
  return active ? active->observations.at(slot) : standalone.at(slot);
}
template <typename Result, typename... Args>
auto with_policy(Result (*operation)(Args...)) {
  return [operation](PreparedPolicy& policy, Args... args) -> Result {
    const Scope scope(policy);
    return operation(args...);
  };
}
}  // namespace flash_v100::policy
