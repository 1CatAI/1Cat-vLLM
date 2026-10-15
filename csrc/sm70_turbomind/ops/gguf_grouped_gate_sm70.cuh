// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Sorted route groups reuse original GGUF weight/codebook decoding. Integer
// dots, group reduction, FP16 activation boundaries and Q8_1 encoding retain
// the single-route order. The original gate/up operator stays unchanged.
namespace {
template <int Type, bool BankAware>
__global__ __launch_bounds__(512, 1) void gate_up_grouped(
    Q8_1* output, const Q8_1* activation, const int* ids,
    const int* route_tokens, const int* order, const int* original_ids,
    const int* mode, const uint8_t* gate, const uint8_t* up, int m, int n,
    int k, int stride, int top_k, int experts) {
  using Dot = LatticeDot<Type, BankAware>;
  __shared__ uint32_t book[Dot::kBookWords], masks[16];
  __shared__ half intermediate[4][32];
  __shared__ int active, count, tokens[4], expert;
  extern __shared__ Q8_1 shared_x[];
  const int route = blockIdx.y, routes = m * top_k;
  if (mode[0] == 0) return;
  if (threadIdx.x == 0) {
    const int e = ids[route];
    int first = route;
    while (first > 0 && ids[first - 1] == e) --first;
    active = ((route - first) % 4 == 0) && e >= 0 && e < experts;
    count = 1;
    while (count < 4 && route + count < routes && ids[route + count] == e)
      ++count;
    for (int q = 0; q < count; ++q)
      tokens[q] = min(max(route_tokens[route + q], 0), m - 1);
    expert = e;
  }
  __syncthreads();
  if (!active) return;
  const int groups = k / 32;
  for (int i = threadIdx.x; i < count * groups * 9; i += blockDim.x) {
    const int token = tokens[i / (groups * 9)];
    reinterpret_cast<uint32_t*>(shared_x)[i] =
        reinterpret_cast<const uint32_t*>(
            activation)[int64_t(token) * groups * 9 + i % (groups * 9)];
  }
  Dot::initialize(book, masks);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int local_row = warp * 2 + lane / 16, row = blockIdx.x * 32 + local_row;
  const uint8_t* g = gate + (int64_t(expert) * n + row) * stride;
  const uint8_t* u = up + (int64_t(expert) * n + row) * stride;
  float gs[4] = {}, us[4] = {};
  for (int group = lane % 16; group < groups; group += 16) {
    if (count == 1) {
      gs[0] += Dot::dot(g, group, shared_x[group], book, masks);
      us[0] += Dot::dot(u, group, shared_x[group], book, masks);
    } else {
      const auto gd = Dot::decode(g, group, book, masks);
      const auto ud = Dot::decode(u, group, book, masks);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        if (q < count) {
          gs[q] += Dot::dot_decoded(gd, shared_x[q * groups + group]);
          us[q] += Dot::dot_decoded(ud, shared_x[q * groups + group]);
        }
      }
    }
  }
#pragma unroll
  for (int offset = 8; offset; offset >>= 1) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      gs[q] += __shfl_down_sync(0xffffffff, gs[q], offset, 16);
      us[q] += __shfl_down_sync(0xffffffff, us[q], offset, 16);
    }
  }
  if (lane % 16 == 0) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (q < count) {
        const float gv = __half2float(__float2half_rn(gs[q]));
        intermediate[q][local_row] = __hmul(
            __float2half_rn(gv / (1.f + expf(-gv))), __float2half_rn(us[q]));
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (q < count)
        vllm::sm70_gguf::quantize_q8_1_warp(
            output + int64_t(order[route + q]) * (n / 32) + blockIdx.x,
            __half2float(intermediate[q][threadIdx.x]));
    }
  }
}

template <int Type, bool BankAware>
void launch_grouped_gate(torch::Tensor out, torch::Tensor activation,
                         torch::Tensor ids, torch::Tensor gate,
                         torch::Tensor up, torch::Tensor tokens,
                         torch::Tensor order, torch::Tensor original_ids,
                         torch::Tensor mode) {
  gate_up_grouped<Type, BankAware><<<dim3(gate.size(1) / 32, ids.numel()), 512,
                                     4 * activation.size(1) * sizeof(Q8_1),
                                     at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<Q8_1*>(out.data_ptr()),
      reinterpret_cast<const Q8_1*>(activation.data_ptr()), ids.data_ptr<int>(),
      tokens.data_ptr<int>(), order.data_ptr<int>(),
      original_ids.data_ptr<int>(), mode.data_ptr<int>(),
      gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), activation.size(0),
      gate.size(1), activation.size(1) * 32, gate.size(2), ids.size(1),
      gate.size(0));
  // Sparse routing uses the retained single-route kernel, with only a device
  // admission guard. It must not inherit the grouped kernel's four-row register
  // and shared-memory footprint. Both guards are graph replay inputs.
  gate_up<Type, true, int32_t, 16, true, false, false, BankAware, 0, true>
      <<<dim3(gate.size(1) / 32, ids.numel()), 512,
         activation.size(1) * sizeof(Q8_1), at::cuda::getCurrentCUDAStream()>>>(
          out.data_ptr(), reinterpret_cast<const Q8_1*>(activation.data_ptr()),
          original_ids.data_ptr<int>(), gate.data_ptr<uint8_t>(),
          up.data_ptr<uint8_t>(), gate.size(1), activation.size(1) * 32,
          gate.size(2), ids.size(1), nullptr, nullptr, mode.data_ptr<int>());
}
}  // namespace

template <int MaxRows>
static void grouped_gate_out_impl(torch::Tensor out, torch::Tensor activation,
                                  torch::Tensor ids, torch::Tensor gate,
                                  torch::Tensor up, int64_t source_type,
                                  bool bank_aware, torch::Tensor route_tokens,
                                  torch::Tensor order,
                                  torch::Tensor original_ids,
                                  torch::Tensor mode) {
  TORCH_CHECK(source_type == 18 || source_type == 21 || source_type == 22,
              "Paired gate/up requires an original lattice format");
  const c10::cuda::CUDAGuard guard(activation.device());
  require_sm70();
  TORCH_CHECK(activation.is_cuda() && activation.scalar_type() == at::kByte &&
                  activation.is_contiguous() && activation.dim() == 3 &&
                  activation.size(0) > 0 && activation.size(0) <= MaxRows &&
                  activation.size(2) == sizeof(Q8_1) &&
                  activation.size(1) % 8 == 0,
              "Invalid paired Q8_1 activation");
  const int k = activation.size(1) * 32;
  const int bytes = source_type == 18 ? 98 : source_type == 21 ? 110 : 82;
  TORCH_CHECK(gate.device() == activation.device() &&
                  up.device() == activation.device() &&
                  gate.scalar_type() == at::kByte &&
                  up.scalar_type() == at::kByte && gate.is_contiguous() &&
                  up.is_contiguous() && gate.dim() == 3 &&
                  gate.sizes() == up.sizes() && gate.size(0) > 0 &&
                  gate.size(1) > 0 && gate.size(1) % 32 == 0 &&
                  (gate.size(2) == k / 256 * bytes ||
                   gate.size(2) == (k / 256 * bytes + 7) / 8 * 8),
              "Invalid paired original weight banks");
  TORCH_CHECK(
      ids.device() == activation.device() && ids.scalar_type() == at::kInt &&
          ids.is_contiguous() && ids.dim() == 2 &&
          ids.size(0) == activation.size(0) && ids.size(1) > 0 &&
          ids.size(1) <= 16 && route_tokens.device() == activation.device() &&
          route_tokens.scalar_type() == at::kInt &&
          route_tokens.is_contiguous() && route_tokens.numel() == ids.numel(),
      "Invalid sorted route or token mapping");
  TORCH_CHECK(order.device() == activation.device() &&
                  order.scalar_type() == at::kInt && order.is_contiguous() &&
                  order.numel() == ids.numel() &&
                  original_ids.device() == activation.device() &&
                  original_ids.scalar_type() == at::kInt &&
                  original_ids.is_contiguous() &&
                  original_ids.sizes() == ids.sizes() &&
                  mode.device() == activation.device() &&
                  mode.scalar_type() == at::kInt && mode.is_contiguous() &&
                  mode.numel() == 1,
              "Invalid grouped routing workspace");
  TORCH_CHECK(
      out.device() == activation.device() && out.scalar_type() == at::kByte &&
          out.is_contiguous() && out.dim() == 4 &&
          out.size(0) == activation.size(0) && out.size(1) == ids.size(1) &&
          out.size(2) == gate.size(1) / 32 && out.size(3) == sizeof(Q8_1),
      "Invalid paired Q8_1 output");
#define PAIR(TYPE)                                                            \
  if (bank_aware)                                                             \
    launch_grouped_gate<TYPE, true>(out, activation, ids, gate, up,           \
                                    route_tokens, order, original_ids, mode); \
  else                                                                        \
    launch_grouped_gate<TYPE, false>(out, activation, ids, gate, up,          \
                                     route_tokens, order, original_ids, mode)
  if (source_type == 18) {
    PAIR(18);
  } else if (source_type == 21) {
    PAIR(21);
  } else {
    PAIR(22);
  }
#undef PAIR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_dp4a_gate_up_grouped_sm70_out(
    torch::Tensor out, torch::Tensor activation, torch::Tensor ids,
    torch::Tensor gate, torch::Tensor up, int64_t source_type, bool bank_aware,
    torch::Tensor route_tokens, torch::Tensor order, torch::Tensor original_ids,
    torch::Tensor mode) {
  grouped_gate_out_impl<20>(out, activation, ids, gate, up, source_type,
                            bank_aware, route_tokens, order, original_ids,
                            mode);
}

void gguf_dp4a_gate_up_grouped_batch_sm70_out(
    torch::Tensor out, torch::Tensor activation, torch::Tensor ids,
    torch::Tensor gate, torch::Tensor up, int64_t source_type, bool bank_aware,
    torch::Tensor route_tokens, torch::Tensor order, torch::Tensor original_ids,
    torch::Tensor mode) {
  grouped_gate_out_impl<40>(out, activation, ids, gate, up, source_type,
                            bank_aware, route_tokens, order, original_ids,
                            mode);
}
