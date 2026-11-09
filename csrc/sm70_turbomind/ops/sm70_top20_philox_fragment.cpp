// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <torch/library.h>

#include <optional>

int64_t sm70_sampler_graph_attach_branch(torch::Tensor flags,
                                         int64_t parent_handle,
                                         int64_t reference_handle,
                                         int64_t compact_handle,
                                         std::optional<torch::Tensor> counters);
int64_t sm70_sampler_graph_prepare_collective(torch::Tensor flags,
                                              int64_t graph_handle);

void sm70_sample_sorted_top20_philox_out(
    torch::Tensor sampled_token_out, torch::Tensor sparse_ids_out,
    torch::Tensor sparse_probs_out, torch::Tensor top_values,
    torch::Tensor top_indices, const std::optional<at::Generator>& generator,
    int64_t vocab_size, double top_p);

void sm70_sample_sorted_top20_philox_token_out(
    torch::Tensor sampled_token_out, torch::Tensor top_values,
    torch::Tensor top_indices, const std::optional<at::Generator>& generator,
    int64_t vocab_size, double top_p);

void sm70_sample_chunked_top20_philox_token_out(
    torch::Tensor sampled_token_out, torch::Tensor global_values,
    torch::Tensor local_indices, torch::Tensor global_positions,
    const std::optional<at::Generator>& generator, int64_t vocab_size,
    double top_p, int64_t chunk_size);

// Keep these registrations in a wheel-safe sidecar. Adding the sampler must
// not relink or perturb the speed- and quality-frozen primary vllm._C module;
// Python loads this fragment only for the explicitly admitted SM70 route.
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "sm70_sampler_graph_attach_branch(Tensor flags, int parent_handle, "
      "int reference_handle, int compact_handle, Tensor? counters=None) -> "
      "int");
  ops.impl("sm70_sampler_graph_attach_branch", torch::kCUDA,
           &sm70_sampler_graph_attach_branch);
  ops.def(
      "sm70_sampler_graph_prepare_collective(Tensor flags, int graph_handle) "
      "-> int");
  ops.impl("sm70_sampler_graph_prepare_collective", torch::kCUDA,
           &sm70_sampler_graph_prepare_collective);

  ops.def(
      "sm70_sample_sorted_top20_philox_out(Tensor(a!) sampled_token_out, "
      "Tensor(b!) sparse_ids_out, Tensor(c!) sparse_probs_out, "
      "Tensor top_values, Tensor top_indices, Generator? generator, "
      "int vocab_size, float top_p) -> ()");
  ops.impl("sm70_sample_sorted_top20_philox_out", torch::kCUDA,
           &sm70_sample_sorted_top20_philox_out);

  ops.def(
      "sm70_sample_sorted_top20_philox_token_out(Tensor(a!) "
      "sampled_token_out, Tensor top_values, Tensor top_indices, "
      "Generator? generator, int vocab_size, float top_p) -> ()");
  ops.impl("sm70_sample_sorted_top20_philox_token_out", torch::kCUDA,
           &sm70_sample_sorted_top20_philox_token_out);

  ops.def(
      "sm70_sample_chunked_top20_philox_token_out(Tensor(a!) "
      "sampled_token_out, Tensor global_values, Tensor local_indices, "
      "Tensor global_positions, Generator? generator, int vocab_size, "
      "float top_p, int chunk_size) -> ()");
  ops.impl("sm70_sample_chunked_top20_philox_token_out", torch::kCUDA,
           &sm70_sample_chunked_top20_philox_token_out);
}
