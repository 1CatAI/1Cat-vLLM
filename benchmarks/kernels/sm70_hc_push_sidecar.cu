// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Keep the CustomAllreduce object lifecycle and all methods in this DSO.
#include "../../csrc/custom_all_reduce.cu"
#include "sm70_hc_push_gather.cuh"

TORCH_LIBRARY(_C_custom_ar_flashnext, custom_ar) {
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, int rank, bool "
      "fully_connected) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, int "
      "reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("all_reduce", torch::kCUDA, &all_reduce);
  custom_ar.def(
      "sm70_tp2_all_reduce_gemma_rms_norm(int fa, Tensor inp, Tensor "
      "residual, Tensor weight, Tensor! normalized_out, Tensor! residual_out, "
      "int reg_buffer, int reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp2_all_reduce_gemma_rms_norm", torch::kCUDA,
                 &sm70_tp2_all_reduce_gemma_rms_norm);
  custom_ar.def(
      "sm70_tp4_all_reduce_gemma_rms_norm(int fa, Tensor inp, Tensor "
      "residual, Tensor weight, Tensor! normalized_out, Tensor! residual_out, "
      "int reg_buffer, int reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp4_all_reduce_gemma_rms_norm", torch::kCUDA,
                 &sm70_tp4_all_reduce_gemma_rms_norm);
  custom_ar.def(
      "sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather(int fa, Tensor inp, "
      "Tensor residual, Tensor weight, Tensor! normalized_out, Tensor! "
      "residual_out, int reg_input_buffer, int reg_output_buffer, int "
      "reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather",
                 torch::kCUDA,
                 &sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather);
  custom_ar.def(
      "all_reduce_sum2(int fa, Tensor inp_a, Tensor inp_b, Tensor! out) -> "
      "()");
  custom_ar.impl("all_reduce_sum2", torch::kCUDA, &all_reduce_sum2);
  custom_ar.def(
      "top1_argmax(int fa, Tensor input_pair, Tensor! output, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("top1_argmax", torch::kCUDA, &top1_argmax);
  custom_ar.def(
      "tile_runtime_all_reduce(int fa, Tensor inp, Tensor! out, int "
      "reg_buffer, int reg_buffer_sz_bytes, int tile_numel, int "
      "engine_blocks, int compute_iters) -> ()");
  custom_ar.impl("tile_runtime_all_reduce", torch::kCUDA,
                 &tile_runtime_all_reduce);
  custom_ar.def(
      "tile_runtime_all_reduce_engine(int fa, Tensor inp, Tensor! out, int "
      "reg_buffer, int reg_buffer_sz_bytes, int tile_numel, int "
      "producer_blocks, int reducer_blocks, int compute_iters) -> ()");
  custom_ar.impl("tile_runtime_all_reduce_engine", torch::kCUDA,
                 &tile_runtime_all_reduce_engine);
  custom_ar.def(
      "tile_runtime_wait_reduce(int fa, Tensor staging, Tensor! out, int "
      "tile_numel, int reducer_blocks) -> ()");
  custom_ar.impl("tile_runtime_wait_reduce", torch::kCUDA,
                 &tile_runtime_wait_reduce);
  custom_ar.def("dispose", &dispose);
  custom_ar.def("meta_size", &meta_size);
  custom_ar.def("sm70_tp4_push_allreduce_buffer_size",
                &sm70_tp4_push_allreduce_buffer_size);
  custom_ar.def("register_buffer", &register_buffer);
  custom_ar.def("register_sm70_tp4_push_allreduce_buffer",
                &register_sm70_tp4_push_allreduce_buffer);
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.def("register_graph_buffers", &register_graph_buffers);
  custom_ar.def("allocate_shared_buffer_and_handle",
                &allocate_shared_buffer_and_handle);
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int", &open_mem_handle);
  custom_ar.impl("open_mem_handle", torch::kCPU, &open_mem_handle);
  custom_ar.def("free_shared_buffer", &free_shared_buffer);
}
