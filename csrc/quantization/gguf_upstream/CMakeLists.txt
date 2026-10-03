# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set(GGUF_UPSTREAM_ROOT "${CMAKE_CURRENT_LIST_DIR}/llama.cpp")
set(GGUF_BRIDGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/quantization/gguf_upstream")
# Keep the reviewed source closure explicit in the pinned TOML manifest.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import pathlib,tomllib; m=tomllib.loads(pathlib.Path(r'${CMAKE_CURRENT_LIST_DIR}/llama_cpp_upstream.toml').read_text()); print(';'.join(m['sources']))"
  OUTPUT_VARIABLE GGUF_UPSTREAM_SOURCES OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(TRANSFORM GGUF_UPSTREAM_SOURCES PREPEND "${GGUF_UPSTREAM_ROOT}/")
set(GGUF_NATIVE_SOURCES
  "${GGUF_BRIDGE_ROOT}/torch_bindings.cpp"
  "${GGUF_BRIDGE_ROOT}/runtime.cu"
  "${GGUF_BRIDGE_ROOT}/runtime_dense.cu"
  "${GGUF_BRIDGE_ROOT}/runtime_moe.cu"
  "${GGUF_BRIDGE_ROOT}/runtime_dequant_blas.cu"
  "${GGUF_BRIDGE_ROOT}/torch_context.cu"
  ${GGUF_UPSTREAM_SOURCES})
set_gencode_flags_for_srcs(SRCS "${GGUF_NATIVE_SOURCES}" CUDA_ARCHS "${CUDA_ARCHS}")
define_extension_target(_C_gguf
  DESTINATION vllm LANGUAGE CUDA
  SOURCES ${GGUF_NATIVE_SOURCES}
  COMPILE_FLAGS
    "-O3" "--use_fast_math" "--extended-lambda"
    "-U__CUDA_NO_HALF_OPERATORS__" "-U__CUDA_NO_HALF_CONVERSIONS__"
    "-U__CUDA_NO_HALF2_OPERATORS__" "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
    "-Xcompiler=-fvisibility=hidden" "-Xcompiler=-ffunction-sections"
  INCLUDE_DIRECTORIES
    "${GGUF_UPSTREAM_ROOT}/ggml/include"
    "${GGUF_UPSTREAM_ROOT}/ggml/src"
    "${GGUF_UPSTREAM_ROOT}/ggml/src/ggml-cuda"
    "${GGUF_BRIDGE_ROOT}"
  LIBRARIES CUDA::cublas USE_SABI 3 WITH_SOABI)
target_compile_definitions(_C_gguf PRIVATE USE_CUDA)
target_link_options(_C_gguf PRIVATE "-Wl,--gc-sections")

