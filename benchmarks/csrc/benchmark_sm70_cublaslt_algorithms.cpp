// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
              " failed with cuBLAS status ", static_cast<int>(status));
}

template <typename T>
T algo_attribute(const cublasLtMatmulAlgo_t& algo,
                 cublasLtMatmulAlgoConfigAttributes_t attribute) {
  T value{};
  size_t written = 0;
  check_cublas(cublasLtMatmulAlgoConfigGetAttribute(&algo, attribute, &value,
                                                    sizeof(value), &written),
               "cublasLtMatmulAlgoConfigGetAttribute");
  TORCH_CHECK(written == sizeof(value), "Unexpected cuBLASLt attribute size");
  return value;
}

template <typename T>
T capability_attribute(const cublasLtMatmulAlgo_t& algo,
                       cublasLtMatmulAlgoCapAttributes_t attribute) {
  T value{};
  size_t written = 0;
  check_cublas(cublasLtMatmulAlgoCapGetAttribute(&algo, attribute, &value,
                                                 sizeof(value), &written),
               "cublasLtMatmulAlgoCapGetAttribute");
  TORCH_CHECK(written == sizeof(value), "Unexpected cuBLASLt capability size");
  return value;
}

std::vector<uint32_t> capability_values(
    const cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoCapAttributes_t attribute) {
  size_t bytes = 0;
  check_cublas(
      cublasLtMatmulAlgoCapGetAttribute(&algo, attribute, nullptr, 0, &bytes),
      "query cuBLASLt capability size");
  std::vector<uint32_t> values(bytes / sizeof(uint32_t));
  if (bytes != 0) {
    size_t written = 0;
    check_cublas(cublasLtMatmulAlgoCapGetAttribute(
                     &algo, attribute, values.data(), bytes, &written),
                 "read cuBLASLt capability values");
    TORCH_CHECK(written == bytes, "Unexpected cuBLASLt capability array size");
  }
  return values;
}

template <typename T>
bool set_algorithm_attribute(cublasLtMatmulAlgo_t* algo,
                             cublasLtMatmulAlgoConfigAttributes_t attribute,
                             T value) {
  return cublasLtMatmulAlgoConfigSetAttribute(
             algo, attribute, &value, sizeof(value)) == CUBLAS_STATUS_SUCCESS;
}

class LtRunner {
 public:
  LtRunner(int64_t m, int64_t n, int64_t k, int64_t workspace_bytes,
           int requested_algorithms, bool exhaustive)
      : m_(m), n_(n), k_(k), workspace_bytes_(workspace_bytes) {
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "GEMM dimensions must be positive");
    TORCH_CHECK(workspace_bytes >= 0, "Workspace size must be non-negative");
    TORCH_CHECK(requested_algorithms > 0,
                "Requested algorithm count must be positive");

    check_cublas(
        cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate");
    cublasOperation_t transpose_a = CUBLAS_OP_N;
    cublasOperation_t transpose_b = CUBLAS_OP_T;
    check_cublas(
        cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transpose_a, sizeof(transpose_a)),
        "set transpose A");
    check_cublas(
        cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transpose_b, sizeof(transpose_b)),
        "set transpose B");

    create_row_major_layout(&a_layout_, m_, k_, k_);
    create_row_major_layout(&b_layout_, n_, k_, k_);
    create_row_major_layout(&c_layout_, m_, n_, n_);

    if (exhaustive) {
      enumerate_algorithms();
    } else {
      query_heuristic_algorithms(requested_algorithms);
    }
  }

  LtRunner(const LtRunner&) = delete;
  LtRunner& operator=(const LtRunner&) = delete;

  ~LtRunner() {
    if (a_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(a_layout_);
    }
    if (b_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(b_layout_);
    }
    if (c_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(c_layout_);
    }
    if (operation_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_);
    }
  }

  py::list algorithm_info() const {
    py::list output;
    for (size_t index = 0; index < algorithms_.size(); ++index) {
      const auto& result = algorithms_[index];
      py::dict item;
      item["index"] = index;
      item["workspace_bytes"] = result.workspaceSize;
      item["waves"] = result.wavesCount;
      item["state"] = static_cast<int>(result.state);
      item["id"] =
          algo_attribute<int32_t>(result.algo, CUBLASLT_ALGO_CONFIG_ID);
      item["tile"] =
          algo_attribute<uint32_t>(result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
      item["split_k"] =
          algo_attribute<int32_t>(result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
      item["reduction"] = algo_attribute<uint32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
      item["swizzle"] = algo_attribute<uint32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
      item["custom"] = algo_attribute<uint32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
      item["stages"] =
          algo_attribute<uint32_t>(result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
      output.append(std::move(item));
    }
    return output;
  }

  void run(int64_t algorithm_index, torch::Tensor output, torch::Tensor input,
           torch::Tensor weight, torch::Tensor workspace) const {
    TORCH_CHECK(algorithm_index >= 0 &&
                    algorithm_index < static_cast<int64_t>(algorithms_.size()),
                "Algorithm index out of range");
    check_tensor(input, "input", {m_, k_}, torch::kFloat16);
    check_tensor(weight, "weight", {n_, k_}, torch::kFloat16);
    check_tensor(output, "output", {m_, n_}, torch::kFloat16);
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous(),
                "workspace must be a contiguous CUDA tensor");
    TORCH_CHECK(workspace.scalar_type() == torch::kUInt8,
                "workspace must have uint8 dtype");
    const auto& algorithm = algorithms_[algorithm_index];
    TORCH_CHECK(
        workspace.numel() >= static_cast<int64_t>(algorithm.workspaceSize),
        "workspace tensor is too small");
    TORCH_CHECK(input.get_device() == weight.get_device() &&
                    input.get_device() == output.get_device() &&
                    input.get_device() == workspace.get_device(),
                "all tensors must be on the same CUDA device");

    void* workspace_pointer =
        algorithm.workspaceSize == 0 ? nullptr : workspace.data_ptr();
    check_cublas(
        cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), operation_,
                       &alpha_, input.data_ptr(), a_layout_, weight.data_ptr(),
                       b_layout_, &beta_, output.data_ptr(), c_layout_,
                       output.data_ptr(), c_layout_, &algorithm.algo,
                       workspace_pointer, algorithm.workspaceSize,
                       c10::cuda::getCurrentCUDAStream(input.get_device())),
        "cublasLtMatmul");
  }

 private:
  void query_heuristic_algorithms(int requested_algorithms) {
    cublasLtMatmulPreference_t preference = nullptr;
    check_cublas(cublasLtMatmulPreferenceCreate(&preference),
                 "cublasLtMatmulPreferenceCreate");
    try {
      auto maximum_workspace = static_cast<size_t>(workspace_bytes_);
      check_cublas(cublasLtMatmulPreferenceSetAttribute(
                       preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &maximum_workspace, sizeof(maximum_workspace)),
                   "set maximum workspace");
      algorithms_.resize(requested_algorithms);
      int returned_algorithms = 0;
      check_cublas(
          cublasLtMatmulAlgoGetHeuristic(
              at::cuda::getCurrentCUDABlasLtHandle(), operation_, a_layout_,
              b_layout_, c_layout_, c_layout_, preference, requested_algorithms,
              algorithms_.data(), &returned_algorithms),
          "cublasLtMatmulAlgoGetHeuristic");
      algorithms_.resize(returned_algorithms);
    } catch (...) {
      cublasLtMatmulPreferenceDestroy(preference);
      throw;
    }
    check_cublas(cublasLtMatmulPreferenceDestroy(preference),
                 "cublasLtMatmulPreferenceDestroy");
  }

  void enumerate_algorithms() {
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    std::vector<int> algorithm_ids(256);
    int returned_ids = 0;
    check_cublas(
        cublasLtMatmulAlgoGetIds(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                 CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                 static_cast<int>(algorithm_ids.size()),
                                 algorithm_ids.data(), &returned_ids),
        "cublasLtMatmulAlgoGetIds");
    algorithm_ids.resize(returned_ids);

    constexpr int32_t split_k_values[] = {2, 3, 4, 5, 6, 8, 12, 16, 24, 32};
    constexpr uint32_t reduction_values[] = {
        CUBLASLT_REDUCTION_SCHEME_INPLACE,
        CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
        CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE,
    };
    for (int algorithm_id : algorithm_ids) {
      cublasLtMatmulAlgo_t base{};
      if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                 CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                 algorithm_id,
                                 &base) != CUBLAS_STATUS_SUCCESS) {
        continue;
      }
      auto tiles = capability_values(base, CUBLASLT_ALGO_CAP_TILE_IDS);
      auto stages = capability_values(base, CUBLASLT_ALGO_CAP_STAGES_IDS);
      if (tiles.empty()) {
        tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
      }
      if (stages.empty()) {
        stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
      }
      const auto custom_max = capability_attribute<int32_t>(
          base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
      const auto swizzle_support = capability_attribute<uint32_t>(
          base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
      const auto split_k_support =
          capability_attribute<int32_t>(base, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);
      const auto reduction_mask = capability_attribute<uint32_t>(
          base, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);

      for (uint32_t tile : tiles) {
        for (uint32_t stage : stages) {
          for (int32_t custom = 0; custom <= custom_max; ++custom) {
            for (uint32_t swizzle = 0; swizzle <= swizzle_support; ++swizzle) {
              cublasLtMatmulAlgo_t configured = base;
              if (!set_algorithm_attribute(
                      &configured, CUBLASLT_ALGO_CONFIG_TILE_ID, tile) ||
                  !set_algorithm_attribute(
                      &configured, CUBLASLT_ALGO_CONFIG_STAGES_ID, stage) ||
                  !set_algorithm_attribute(&configured,
                                           CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                           static_cast<uint32_t>(custom)) ||
                  !set_algorithm_attribute(&configured,
                                           CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                           swizzle)) {
                continue;
              }
              add_checked_algorithm(configured);
              if (!split_k_support) {
                continue;
              }
              for (int32_t split_k : split_k_values) {
                for (uint32_t reduction : reduction_values) {
                  if ((reduction_mask & reduction) == 0) {
                    continue;
                  }
                  auto split_configured = configured;
                  if (!set_algorithm_attribute(&split_configured,
                                               CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                               split_k) ||
                      !set_algorithm_attribute(
                          &split_configured,
                          CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, reduction)) {
                    continue;
                  }
                  add_checked_algorithm(split_configured);
                }
              }
            }
          }
        }
      }
    }
  }

  void add_checked_algorithm(const cublasLtMatmulAlgo_t& algorithm) {
    cublasLtMatmulHeuristicResult_t result{};
    result.algo = algorithm;
    auto status = cublasLtMatmulAlgoCheck(
        at::cuda::getCurrentCUDABlasLtHandle(), operation_, a_layout_,
        b_layout_, c_layout_, c_layout_, &result.algo, &result);
    if (status == CUBLAS_STATUS_SUCCESS &&
        result.state == CUBLAS_STATUS_SUCCESS &&
        result.workspaceSize <= static_cast<size_t>(workspace_bytes_)) {
      algorithms_.push_back(result);
    }
  }

  static void create_row_major_layout(cublasLtMatrixLayout_t* layout,
                                      int64_t rows, int64_t columns,
                                      int64_t leading_dimension) {
    check_cublas(cublasLtMatrixLayoutCreate(layout, CUDA_R_16F, rows, columns,
                                            leading_dimension),
                 "cublasLtMatrixLayoutCreate");
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check_cublas(
        cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &order, sizeof(order)),
        "set row-major matrix layout");
  }

  static void check_tensor(const torch::Tensor& tensor, const char* name,
                           std::initializer_list<int64_t> expected_shape,
                           torch::ScalarType expected_type) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == expected_type, name,
                " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == at::IntArrayRef(expected_shape), name,
                " has the wrong shape");
  }

  int64_t m_;
  int64_t n_;
  int64_t k_;
  int64_t workspace_bytes_;
  float alpha_ = 1.0F;
  float beta_ = 0.0F;
  cublasLtMatmulDesc_t operation_ = nullptr;
  cublasLtMatrixLayout_t a_layout_ = nullptr;
  cublasLtMatrixLayout_t b_layout_ = nullptr;
  cublasLtMatrixLayout_t c_layout_ = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algorithms_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<LtRunner, std::shared_ptr<LtRunner>>(module, "LtRunner")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int, bool>())
      .def("algorithm_info", &LtRunner::algorithm_info)
      .def("run", &LtRunner::run);
}
