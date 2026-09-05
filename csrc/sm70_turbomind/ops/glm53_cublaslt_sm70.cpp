// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <torch/all.h>

namespace {

void check_cublaslt(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
              " failed with cuBLAS status ", static_cast<int>(status));
}

struct MatmulResources {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t input = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t output = nullptr;

  ~MatmulResources() {
    if (input != nullptr) {
      cublasLtMatrixLayoutDestroy(input);
    }
    if (weight != nullptr) {
      cublasLtMatrixLayoutDestroy(weight);
    }
    if (output != nullptr) {
      cublasLtMatrixLayoutDestroy(output);
    }
    if (operation != nullptr) {
      cublasLtMatmulDescDestroy(operation);
    }
  }
};

void create_row_major_layout(cublasLtMatrixLayout_t* layout, int64_t rows,
                             int64_t columns) {
  check_cublaslt(
      cublasLtMatrixLayoutCreate(layout, CUDA_R_16F, rows, columns, columns),
      "cublasLtMatrixLayoutCreate");
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  check_cublaslt(
      cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &order, sizeof(order)),
      "set row-major matrix layout");
}

template <typename T>
void set_algorithm_attribute(cublasLtMatmulAlgo_t* algorithm,
                             cublasLtMatmulAlgoConfigAttributes_t attribute,
                             T value) {
  check_cublaslt(cublasLtMatmulAlgoConfigSetAttribute(algorithm, attribute,
                                                      &value, sizeof(value)),
                 "cublasLtMatmulAlgoConfigSetAttribute");
}

}  // namespace

void sm70_glm53_tp8_cublaslt_out(torch::Tensor out, torch::Tensor input,
                                 torch::Tensor weight) {
  constexpr int64_t kTokens = 8;
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weight.is_cuda(),
              "sm70_glm53_tp8_cublaslt_out: tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kFloat16,
              "sm70_glm53_tp8_cublaslt_out: tensors must be float16");
  TORCH_CHECK(
      out.is_contiguous() && input.is_contiguous() && weight.is_contiguous(),
      "sm70_glm53_tp8_cublaslt_out: tensors must be contiguous");
  TORCH_CHECK(out.dim() == 2 && input.dim() == 2 && weight.dim() == 2,
              "sm70_glm53_tp8_cublaslt_out: tensors must be rank two");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weight.get_device(),
              "sm70_glm53_tp8_cublaslt_out: tensors must share one device");

  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = weight.size(0);
  const bool in_projection = m == kTokens && k == 4096 && n == 3336 &&
                             weight.sizes() == torch::IntArrayRef({3336, 4096});
  const bool out_projection =
      m == kTokens && k == 1024 && n == 4096 &&
      weight.sizes() == torch::IntArrayRef({4096, 1024});
  TORCH_CHECK(in_projection || out_projection,
              "sm70_glm53_tp8_cublaslt_out: unsupported GLM-5.3 TP8 shape");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({m, n}),
              "sm70_glm53_tp8_cublaslt_out: output shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "sm70_glm53_tp8_cublaslt_out: requires SM70");
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();

  MatmulResources resources;
  check_cublaslt(cublasLtMatmulDescCreate(&resources.operation,
                                          CUBLAS_COMPUTE_32F, CUDA_R_32F),
                 "cublasLtMatmulDescCreate");
  cublasOperation_t transpose_a = CUBLAS_OP_N;
  cublasOperation_t transpose_b = CUBLAS_OP_T;
  check_cublaslt(cublasLtMatmulDescSetAttribute(
                     resources.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                     &transpose_a, sizeof(transpose_a)),
                 "set transpose A");
  check_cublaslt(cublasLtMatmulDescSetAttribute(
                     resources.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                     &transpose_b, sizeof(transpose_b)),
                 "set transpose B");
  create_row_major_layout(&resources.input, m, k);
  create_row_major_layout(&resources.weight, n, k);
  create_row_major_layout(&resources.output, m, n);

  cublasLtMatmulAlgo_t algorithm{};
  check_cublaslt(cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                        CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                        CUDA_R_16F, 21, &algorithm),
                 "cublasLtMatmulAlgoInit");
  set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID,
                          uint32_t{5});
  set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                          uint32_t{20});
  set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                          uint32_t{0});
  set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                          uint32_t{0});

  cublasLtMatmulHeuristicResult_t checked{};
  checked.algo = algorithm;
  check_cublaslt(
      cublasLtMatmulAlgoCheck(handle, resources.operation, resources.input,
                              resources.weight, resources.output,
                              resources.output, &checked.algo, &checked),
      "cublasLtMatmulAlgoCheck");
  TORCH_CHECK(
      checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize == 0,
      "sm70_glm53_tp8_cublaslt_out: pinned algorithm is unavailable");

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  check_cublaslt(
      cublasLtMatmul(handle, resources.operation, &alpha, input.data_ptr(),
                     resources.input, weight.data_ptr(), resources.weight,
                     &beta, out.data_ptr(), resources.output, out.data_ptr(),
                     resources.output, &checked.algo, nullptr, 0,
                     c10::cuda::getCurrentCUDAStream(input.get_device())),
      "cublasLtMatmul");
}
