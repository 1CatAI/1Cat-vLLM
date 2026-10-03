// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Upstream declarations/registrations adapted from vllm-gguf-plugin PR #141,
// aa09d6522f29325d64d999d7d7c794f79836de07. Legacy ops stay in vllm._C.
#include <optional>
#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
bool ggml_should_use_mmvq(int64_t type, int64_t cc, int64_t batch);
int64_t ggml_dense_upstream_capabilities(Tensor W, Tensor X, int64_t type,
                                         int64_t row);
Tensor ggml_dense_mmvf(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_mmvq(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_mmq(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_mmf(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_blas(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dequantize_upstream(Tensor W, int64_t type, int64_t m, int64_t n,
                                std::optional<ScalarType> dtype);
Tensor ggml_moe_upstream(Tensor X, Tensor W, Tensor ids, int64_t type,
                         int64_t row, int64_t top_k, int64_t tokens);
Tensor ggml_moe_mmvq(Tensor X, Tensor W, Tensor ids, int64_t type, int64_t row,
                     int64_t top_k, int64_t tokens);
Tensor ggml_moe_mmq(Tensor X, Tensor W, Tensor ids, int64_t type, int64_t row,
                    int64_t top_k, int64_t tokens,
                    std::optional<Tensor> expert_ids,
                    std::optional<Tensor> padded_count);
Tensor ggml_moe_grouped_dense(Tensor X, Tensor W, Tensor ids, int64_t type,
                              int64_t row, int64_t top_k, int64_t tokens);

STABLE_TORCH_LIBRARY(_C_gguf, ops) {
  ops.def("ggml_should_use_mmvq(int type, int cc, int batch) -> bool");
  ops.def(
      "ggml_dense_upstream_capabilities(Tensor W, Tensor X, int type, SymInt "
      "row) -> int");
  ops.def(
      "ggml_dense_mmvf(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_dense_mmvq(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def("ggml_dense_mmq(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def("ggml_dense_mmf(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_dense_blas(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_dequantize_upstream(Tensor W, int type, SymInt m, SymInt n, "
      "ScalarType? dtype) -> Tensor");
  ops.def(
      "ggml_moe_upstream(Tensor X, Tensor W, Tensor ids, int type, SymInt row, "
      "SymInt top_k, SymInt tokens) -> Tensor");
  ops.def(
      "ggml_moe_mmvq(Tensor X, Tensor W, Tensor ids, int type, SymInt row, "
      "SymInt top_k, SymInt tokens) -> Tensor");
  ops.def(
      "ggml_moe_mmq(Tensor X, Tensor W, Tensor ids, int type, SymInt row, "
      "SymInt top_k, SymInt tokens, Tensor? expert_ids=None, Tensor? "
      "padded_count=None) -> Tensor");
  ops.def(
      "ggml_moe_grouped_dense(Tensor X, Tensor W, Tensor ids, int type, SymInt "
      "row, SymInt top_k, SymInt tokens) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CUDA, ops) {
  ops.impl("ggml_dense_mmvf", TORCH_BOX(&ggml_dense_mmvf));
  ops.impl("ggml_dense_mmvq", TORCH_BOX(&ggml_dense_mmvq));
  ops.impl("ggml_dense_mmq", TORCH_BOX(&ggml_dense_mmq));
  ops.impl("ggml_dense_mmf", TORCH_BOX(&ggml_dense_mmf));
  ops.impl("ggml_dense_blas", TORCH_BOX(&ggml_dense_blas));
  ops.impl("ggml_dequantize_upstream", TORCH_BOX(&ggml_dequantize_upstream));
  ops.impl("ggml_moe_upstream", TORCH_BOX(&ggml_moe_upstream));
  ops.impl("ggml_moe_mmvq", TORCH_BOX(&ggml_moe_mmvq));
  ops.impl("ggml_moe_mmq", TORCH_BOX(&ggml_moe_mmq));
  ops.impl("ggml_moe_grouped_dense", TORCH_BOX(&ggml_moe_grouped_dense));
}
STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CompositeExplicitAutograd, ops) {
  ops.impl("ggml_should_use_mmvq", TORCH_BOX(&ggml_should_use_mmvq));
  ops.impl("ggml_dense_upstream_capabilities",
           TORCH_BOX(&ggml_dense_upstream_capabilities));
}
static struct PyModuleDef _module_def = {
    PyModuleDef_HEAD_INIT, "_C_gguf", nullptr, -1, nullptr,
};
extern "C" __attribute__((visibility("default"))) PyObject* PyInit__C_gguf(
    void) {
  return PyModule_Create(&_module_def);
}
