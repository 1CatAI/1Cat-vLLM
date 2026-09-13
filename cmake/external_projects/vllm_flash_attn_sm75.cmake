# FlashAttention-2 for Turing (sm75).
#
# The FA2 sources fetched by vllm_flash_attn.cmake build no sm75 kernels: the
# vllm-project tree targets 8.0+, the Volta fork targets 7.0 only. Turing
# therefore fell through to FlashInfer, whose paged prefill fails on sm75, or
# to TRITON_ATTN. This builds a second FA2 library from a pinned fork that
# enables the sm75 forward path (fp16-only, forward-only) and installs it as
# _vllm_fa2_C_sm75 next to the regular _vllm_fa2_C; flash_attn_interface.py
# loads the one that matches the worker's device.
#
# Why ExternalProject and not a second FetchContent: both FA2 trees define
# the target _vllm_fa2_C and override global CMake functions (see the note in
# vllm_flash_attn.cmake), so they cannot share one configure. The fork's
# CMake names the interpreter Python_EXECUTABLE, hence the translation below.

include(ExternalProject)

set(VLLM_FLASH_ATTN_SM75_COMMIT 43b9d29c9aa8e18d9351e7c643dc78ef7e7979fc)  # tag sm75-1cat-2026-09-13
set(VLLM_FLASH_ATTN_SM75_LIB _vllm_fa2_C_sm75.abi3.so)

if(DEFINED ENV{MAX_JOBS})
  set(VLLM_FLASH_ATTN_SM75_JOBS -j $ENV{MAX_JOBS})
else()
  set(VLLM_FLASH_ATTN_SM75_JOBS "")
endif()

ExternalProject_Add(vllm-flash-attn-sm75
  GIT_REPOSITORY https://github.com/Peuqui/flash-attention.git
  GIT_TAG ${VLLM_FLASH_ATTN_SM75_COMMIT}
  GIT_PROGRESS TRUE
  GIT_SUBMODULES csrc/cutlass
  GIT_SUBMODULES_RECURSE TRUE
  CMAKE_ARGS
    -DPython_EXECUTABLE=${VLLM_PYTHON_EXECUTABLE}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    -DCUDA_ARCHS=7.5
    -DFA2_ENABLED=ON
    -DFA3_ENABLED=OFF
    -DVLLM_FA2_OUTPUT_NAME=_vllm_fa2_C_sm75
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target _vllm_fa2_C ${VLLM_FLASH_ATTN_SM75_JOBS}
  BUILD_BYPRODUCTS <BINARY_DIR>/${VLLM_FLASH_ATTN_SM75_LIB}
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(vllm-flash-attn-sm75 BINARY_DIR)

# setup.py builds and installs extensions by name: target and install
# component are both called _vllm_fa2_C_sm75.
add_custom_target(_vllm_fa2_C_sm75 ALL DEPENDS vllm-flash-attn-sm75)
install(FILES ${BINARY_DIR}/${VLLM_FLASH_ATTN_SM75_LIB}
  DESTINATION vllm/vllm_flash_attn
  COMPONENT _vllm_fa2_C_sm75)
message(STATUS "vllm-flash-attn-sm75 will be built from Peuqui/flash-attention@${VLLM_FLASH_ATTN_SM75_COMMIT}")
