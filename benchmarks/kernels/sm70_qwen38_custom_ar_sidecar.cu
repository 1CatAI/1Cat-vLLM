// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// The communicator lifecycle and every operation taking its opaque pointer
// must belong to the same DSO. Compile the production implementation directly.
#include "../../csrc/custom_all_reduce.cu"

TORCH_LIBRARY(_C_custom_ar_flashnext, ops) {
  ops.def("init_custom_ar", &init_custom_ar);
  ops.def("all_reduce", &all_reduce);
  ops.def("all_reduce_sum2", &all_reduce_sum2);
  ops.def(
      "sm70_qwen38_hc_down_allgather(int fa, Tensor inp, Tensor! out) -> ()",
      &sm70_qwen38_hc_down_allgather);
  ops.def(
      "sm70_qwen38_hc_gate_mix(int fa, Tensor local_gate, Tensor branches, "
      "Tensor! out) -> ()",
      &sm70_qwen38_hc_gate_mix);
  ops.def(
      "sm70_qwen38_hc_output_allgather(int fa, Tensor local_block, "
      "Tensor! out) -> ()",
      &sm70_qwen38_hc_output_allgather);
  ops.def(
      "sm70_qwen38_hc_up_mix_allgather(int fa, Tensor lora, Tensor weight, "
      "Tensor branches, Tensor! out) -> ()",
      &sm70_qwen38_hc_up_mix_allgather);
  ops.def("dispose", &dispose);
  ops.def("meta_size", &meta_size);
  ops.def("sm70_tp4_push_allreduce_buffer_size",
          &sm70_tp4_push_allreduce_buffer_size);
  ops.def("register_buffer", &register_buffer);
  ops.def("register_sm70_tp4_push_allreduce_buffer",
          &register_sm70_tp4_push_allreduce_buffer);
  ops.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  ops.def("register_graph_buffers", &register_graph_buffers);
  ops.def("allocate_shared_buffer_and_handle",
          &allocate_shared_buffer_and_handle);
  ops.def("open_mem_handle", &open_mem_handle);
  ops.def("free_shared_buffer", &free_shared_buffer);
}
