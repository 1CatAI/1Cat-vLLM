#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdexcept>
#include "fused_mha.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention-2 implementation optimized for Volta";
    m.def("fwd", &flash_attention_forward, "FlashAttention-2 Forward Pass (Volta)");
    m.def(
        "qk_scores_fwd",
        &flash_attention_qk_scores,
        "Debug FlashAttention QK score dump before softmax (Volta)");
    m.def("bwd", &flash_attention_backward, "FlashAttention-2 Backward Pass (Volta)");
    m.def(
        "decode_paged_fwd",
        &flash_attention_decode_paged,
        "FlashAttention decode over paged KV cache (Volta)");
    m.def(
        "decode_paged_wmma_fwd",
        &flash_attention_decode_paged_wmma,
        "FlashAttention single-query decode through paged-prefill WMMA order (Volta)");
    m.def(
        "decode_qk_scores_fwd",
        &flash_attention_decode_qk_scores,
        "Debug scalar paged decode QK score dump before softmax (Volta)");
    m.def(
        "decode_turboquant_paged_fwd",
        &flash_attention_turboquant_decode_paged,
        "FlashAttention decode over TurboQuant paged KV cache (Volta)");
    m.def(
        "prefill_paged_fwd",
        &flash_attention_prefill_paged,
        "FlashAttention prefill over paged KV cache (Volta)");
}
