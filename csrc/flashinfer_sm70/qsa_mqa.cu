// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cmath>
#include <flashinfer/attention/sm70/qsa_mqa.cuh>

namespace {
using namespace flashinfer::attention::sm70;

void mqa(torch::Tensor q, torch::Tensor k, torch::Tensor table,
         torch::Tensor requests, torch::Tensor positions, torch::Tensor lengths,
         torch::Tensor logits, torch::Tensor visible, torch::Tensor schedule,
         int64_t ratio, double divisor, int64_t workers,
         std::optional<torch::Tensor> task_visits) {
  TORCH_CHECK(q.is_cuda(), "QSA MQA requires CUDA tensors");
  const c10::cuda::CUDAGuard guard(q.device());
  for (const auto& t :
       {k, table, requests, positions, lengths, logits, visible, schedule})
    TORCH_CHECK(t.device() == q.device(),
                "QSA MQA tensors must share a CUDA device");
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0,
              "QSA MQA native route requires SM70");
  TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(0) <= kMQAMaxRows &&
              q.size(1) > 0 && q.size(1) <= 16 && q.stride(2) == 1);
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf);
  TORCH_CHECK(k.dim() == 4 && k.size(0) > 0 && k.size(1) > 0 &&
              k.size(2) == 1 && k.size(3) == q.size(2) && k.stride(3) == 1);
  TORCH_CHECK(q.size(2) == 64 || q.size(2) == 128 || q.size(2) == 256);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                  q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 &&
                  k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0,
              "QSA MQA vector loads require aligned rows");
  TORCH_CHECK(table.dim() == 2 && table.size(0) > 0 && table.size(1) > 0 &&
              table.stride(1) == 1);
  for (const auto& t : {table, requests, lengths, visible, schedule})
    TORCH_CHECK(t.scalar_type() == at::kInt, "QSA MQA metadata must be int32");
  TORCH_CHECK(positions.scalar_type() == at::kInt ||
                  positions.scalar_type() == at::kLong,
              "QSA positions must be int32 or int64");
  for (const auto& t : {requests, positions, visible})
    TORCH_CHECK(t.dim() == 1 && t.numel() == q.size(0) && t.is_contiguous());
  TORCH_CHECK(lengths.dim() == 1 && lengths.numel() == table.size(0) &&
              lengths.is_contiguous());
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) == q.size(0) &&
              logits.size(1) > 0 && logits.size(1) <= INT32_MAX - kMQATile &&
              logits.scalar_type() == at::kFloat && logits.stride(1) == 1);
  TORCH_CHECK(ratio > 0 && ratio <= INT32_MAX && std::isfinite(divisor) &&
              divisor > 0 && std::isfinite(static_cast<float>(divisor)) &&
              static_cast<float>(divisor) > 0);
  TORCH_CHECK(workers > 0 && workers <= props->multiProcessorCount * 8);
  TORCH_CHECK(schedule.dim() == 2 && schedule.size(0) == workers + 1 &&
              schedule.size(1) == 2 && schedule.is_contiguous());
  TORCH_CHECK(k.size(0) <= INT32_MAX && k.size(1) <= INT32_MAX &&
              table.size(0) <= INT32_MAX && table.size(1) <= INT32_MAX);
  TORCH_CHECK(((logits.size(1) + kMQATile - 1) / kMQATile) * q.size(0) <=
              INT32_MAX);
  MQAParams p{};
  p.q = reinterpret_cast<const half*>(q.data_ptr());
  p.k = reinterpret_cast<const half*>(k.data_ptr());
  p.table = table.data_ptr<int32_t>();
  p.requests = requests.data_ptr<int32_t>();
  p.positions = positions.data_ptr();
  p.positions64 = positions.scalar_type() == at::kLong;
  p.lengths = lengths.data_ptr<int32_t>();
  p.visible = visible.data_ptr<int32_t>();
  p.schedule = schedule.data_ptr<int32_t>();
  p.logits = logits.data_ptr<float>();
  p.rows = q.size(0);
  p.heads = q.size(1);
  p.columns = logits.size(1);
  p.pages = k.size(0);
  p.page_size = k.size(1);
  p.table_width = table.size(1);
  p.num_requests = table.size(0);
  p.ratio = ratio;
  p.workers = workers;
  p.divisor = divisor;
  p.q_row = q.stride(0);
  p.q_head = q.stride(1);
  p.k_page = k.stride(0);
  p.k_token = k.stride(1);
  p.table_row = table.stride(0);
  p.out_row = logits.stride(0);
  if (task_visits.has_value()) {
    const auto& visits = *task_visits;
    TORCH_CHECK(visits.device() == q.device() &&
                visits.scalar_type() == at::kInt && visits.is_contiguous() &&
                visits.dim() == 2 && visits.size(0) == q.size(0) &&
                visits.size(1) == (logits.size(1) + kMQATile - 1) / kMQATile);
    p.task_visits = visits.data_ptr<int32_t>();
  }
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  PlanMQA<<<1, 32, 0, stream>>>(p);
#define SCORE(D)                                         \
  case D:                                                \
    if (task_visits.has_value())                         \
      ScoreMQA<D, true><<<workers, 128, 0, stream>>>(p); \
    else                                                 \
      ScoreMQA<D><<<workers, 128, 0, stream>>>(p);       \
    break
  switch (q.size(2)) {
    SCORE(64);
    SCORE(128);
    SCORE(256);
  }
#undef SCORE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_flashinfer_mqa_sm70, m) {
  m.def(
      "run(Tensor q, Tensor k, Tensor table, Tensor requests, Tensor "
      "positions, "
      "Tensor lengths, Tensor(a!) logits, Tensor(b!) visible, Tensor(c!) "
      "schedule, "
      "int ratio, float divisor, int workers, Tensor(d!)? task_visits=None) -> "
      "()");
}
TORCH_LIBRARY_IMPL(_C_flashinfer_mqa_sm70, CUDA, m) { m.impl("run", &mqa); }
