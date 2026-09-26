# Build the Q8000/Q8192 route in SM70 FA2 builds while keeping unrelated
# CUDA targets unchanged. The cache option remains available for rollback.
set(_VLLM_SM70_79T_PREFILL_DEFAULT OFF)
if(VLLM_FLASH_ATTN_SM70)
  set(_VLLM_SM70_79T_PREFILL_DEFAULT ON)
endif()
option(VLLM_SM70_79T_PREFILL "Build the SM70 Q8000/Q8192 prefill dispatcher"
  ${_VLLM_SM70_79T_PREFILL_DEFAULT})
unset(_VLLM_SM70_79T_PREFILL_DEFAULT)
if(NOT VLLM_SM70_79T_PREFILL)
  return()
endif()
if(NOT VLLM_FLASH_ATTN_SM70 OR NOT TARGET _vllm_fa2_C)
  message(FATAL_ERROR "VLLM_SM70_79T_PREFILL requires SM70 FA2")
endif()

set(SM70_79T_DIR "${CMAKE_CURRENT_LIST_DIR}/../csrc/attention/sm70_79t")
get_target_property(_fa_sources _vllm_fa2_C SOURCES)
list(FILTER _fa_sources EXCLUDE REGEX "flash_fwd_d256_gqa_arch_sm70\\.cu$")
set_property(TARGET _vllm_fa2_C PROPERTY SOURCES "${_fa_sources}")
# Remove the old architecture's target-wide recipe. Other FA2 sources do not
# consume these macros; the new recipe belongs only to its translation unit.
get_target_property(_fa_defines _vllm_fa2_C COMPILE_DEFINITIONS)
list(FILTER _fa_defines EXCLUDE REGEX "^(PREFIX_|QK_|PV_)")
set_property(TARGET _vllm_fa2_C PROPERTY COMPILE_DEFINITIONS "${_fa_defines}")
set(_79t_defs
  PREFIX_TORCH_EXTENSION PREFIX_FULL_ENDPOINT PREFIX_TORCH_STABLE_ROWS
  PREFIX_PV_FP32_MMA_ACCUMULATE
  PREFIX_QK_CUBLAS_FP32_ACCUM
  PREFIX_TORCH_PREFIX_FP32_OUTPUT
  PREFIX_TORCH_BLOCK_N=8192
  PREFIX_QK_PRETRANSPOSE_INPUTS PREFIX_TRANSPOSED_SCORE_WORKSPACE
  PREFIX_PV_COMPUTE_SUM PREFIX_PV_UNNORMALIZED
  PREFIX_PV_DIRECT_FP16_ACCUMULATE PREFIX_QK_CUBLAS_RAW
  PREFIX_BATCHED_TRI_TAIL PREFIX_BATCHED_TAIL_TILE_TOKENS=320
  PREFIX_BATCHED_TAIL_QK_ALGO=CUBLAS_GEMM_ALGO11_TENSOR_OP
  PREFIX_BATCHED_TRI_REVERSE_PV_TASKS PREFIX_TAIL_IDLE_SM_OVERLAP
  PREFIX_TAIL_IDLE_SM_FINE_PV PREFIX_TAIL_FINE_PV_GROUP_TILES=4
  PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE
  PREFIX_BATCHED_TRI_REPAIR_FIRST_TILE PREFIX_BATCHED_TRI_REPAIR_TOKENS=64
  PREFIX_UPSTREAM_TRANSPOSED_QK PREFIX_FIXED_STATE_NO_RESET
  PREFIX_PV_FUSED_PREFIX_SUM
  QK_TB_M=128 QK_TB_N=128 QK_WARP_M=32 QK_WARP_N=64 QK_STAGES=2
  PV_TB_M=128 PV_TB_N=256 PV_TB_K=32
  PV_WARP_M=64 PV_WARP_N=64 PV_WARP_K=32)
set_source_files_properties("${SM70_79T_DIR}/prefill.cu"
  TARGET_DIRECTORY _vllm_fa2_C PROPERTIES
  COMPILE_DEFINITIONS "${_79t_defs}"
  COMPILE_OPTIONS "-gencode=arch=compute_70,code=sm_70;-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda")
set_source_files_properties("${SM70_79T_DIR}/prefill_q8192.cu"
  TARGET_DIRECTORY _vllm_fa2_C PROPERTIES
  COMPILE_DEFINITIONS "${_79t_defs}"
  COMPILE_OPTIONS "-gencode=arch=compute_70,code=sm_70;-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda")
set_source_files_properties("${SM70_79T_DIR}/legacy_tail_adapter.cu"
  TARGET_DIRECTORY _vllm_fa2_C PROPERTIES
  COMPILE_OPTIONS "-gencode=arch=compute_70,code=sm_70")
# Keep the existing aligned tail as the adapter's fallback.
set_property(SOURCE
  "${vllm-flash-attn_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_d256_splitd_sm70.cu"
  TARGET_DIRECTORY _vllm_fa2_C APPEND PROPERTY COMPILE_DEFINITIONS
  onecat_sm70_d256_dense_state_raw=onecat_sm70_d256_dense_state_legacy_raw)
target_sources(_vllm_fa2_C PRIVATE
  "${SM70_79T_DIR}/prefill.cu"
  "${SM70_79T_DIR}/prefill_q8192.cu"
  "${SM70_79T_DIR}/legacy_tail_adapter.cu"
  "${SM70_79T_DIR}/register.cpp")
target_link_libraries(_vllm_fa2_C PRIVATE CUDA::cublas)
