// One-launch grouped TurboMind MXFP4 MoE GEMM: the vllm._C implementation the
// dense stage calls once per expert, here called once for all experts.
#include <torch/extension.h>

void mxfp4_moe_gemm_sm70_out_impl(
    at::Tensor out, at::Tensor sorted_input, at::Tensor expert_offsets,
    at::Tensor strided_ptrs_w, at::Tensor strided_ptrs_s, int64_t num_experts,
    int64_t k, int64_t n, int64_t group_size, at::Tensor b_group_indices,
    bool compact_grouped_rows, bool broadcast_input_rows);

void grouped(at::Tensor out, at::Tensor x, at::Tensor offsets,
             at::Tensor ptrs_w, at::Tensor ptrs_s, int64_t num_experts,
             int64_t k, int64_t n, at::Tensor expert_ids) {
  mxfp4_moe_gemm_sm70_out_impl(out, x, offsets, ptrs_w, ptrs_s, num_experts, k,
                               n, 32, expert_ids, false, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grouped", &grouped); }
