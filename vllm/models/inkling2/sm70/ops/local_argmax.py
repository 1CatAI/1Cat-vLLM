# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Vocabulary-parallel argmax of the Inkling2 head without the dense logits.

Greedy sampling needs one token id per row, not 200K logits per row gathered
over the TP group. Each rank reduces its own shard to (max value, global id)
and the ranks exchange only that pair:

  pack    one CTA per row: argmax of the rank's local (softcapped, fp32)
          logits, lowest index on ties; the value's 32 bits and the id are
          written as five small integers (<= 11 bits each, exact in fp16) into
          the rank's slot of a zero `[tp, rows, 8]` fp16 tensor.
  --      one tiny all-reduce (64 B per row; x + 0 == x, so the sum is the
          concatenation; the fork's push all-reduce inside CUDA graphs).
  select  per row: rebuild the tp (value, id) pairs, take the largest value,
          lowest rank (= lowest id) on ties.

The result equals `argmax` (first occurrence) of the dense path's logits:
same local logits (same fp32 GEMM), the softcap applied by the same torch
ops before the reduction, all-gather order = rank order.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 1024;
constexpr int kSlots = 8;   // fp16 values per (rank, row): 3 value chunks, 2 id chunks, 3 zeros

// grid (rows), block 1024. logits [rows, stride] fp32; only columns [0, limit) count.
__global__ void __launch_bounds__(kThreads) argmax_pack_kernel(
    const float* __restrict__ logits, long stride, int limit, int vocab_start, int rank,
    int tp, __half* __restrict__ packed, int rows) {
  __shared__ float sval[kThreads / 32];
  __shared__ int sidx[kThreads / 32];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* p = logits + (long)row * stride;
  // Ascending scan with a strict compare: the first occurrence wins inside a thread.
  float best = -INFINITY;
  int idx = 0x7fffffff;
  for (int i = tid; i < limit; i += kThreads) {
    const float v = p[i];
    if (v > best || idx == 0x7fffffff) { best = v; idx = i; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
  }
  if (lane == 0) { sval[warp] = best; sidx[warp] = idx; }
  __syncthreads();
  if (warp == 0) {
    best = sval[lane]; idx = sidx[lane];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
      if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    // An empty shard (limit <= 0) sends -inf with id 0.
    const unsigned u = __float_as_uint(best);
    const unsigned id = idx == 0x7fffffff ? 0u : (unsigned)(idx + vocab_start);
    const int e = lane;   // (rank r, slot k) = (e / 8, e % 8)
    if (e < tp * kSlots) {
      const int r = e / kSlots, k = e - r * kSlots;
      unsigned c = 0;
      if (r == rank) {
        if (k == 0) c = u & 0x7ffu;
        else if (k == 1) c = (u >> 11) & 0x7ffu;
        else if (k == 2) c = u >> 22;
        else if (k == 3) c = id & 0x7ffu;
        else if (k == 4) c = id >> 11;
      }
      packed[((long)r * rows + row) * kSlots + k] = __float2half((float)c);
    }
  }
}

// grid (ceil(rows / 32)), block 32: one thread per row.
__global__ void select_kernel(const __half* __restrict__ packed, int tp, int rows,
                              int64_t* __restrict__ ids) {
  const int row = blockIdx.x * 32 + threadIdx.x;
  if (row >= rows) return;
  float best = 0.f;
  unsigned best_id = 0;
  for (int r = 0; r < tp; ++r) {
    const __half* q = packed + ((long)r * rows + row) * kSlots;
    const unsigned u = (unsigned)__half2float(q[0]) | ((unsigned)__half2float(q[1]) << 11) |
                       ((unsigned)__half2float(q[2]) << 22);
    const unsigned id = (unsigned)__half2float(q[3]) | ((unsigned)__half2float(q[4]) << 11);
    const float v = __uint_as_float(u);
    if (r == 0 || v > best) { best = v; best_id = id; }
  }
  ids[row] = (int64_t)best_id;
}

}  // namespace

void argmax_pack(torch::Tensor logits, torch::Tensor packed, int64_t limit, int64_t vocab_start,
                 int64_t rank) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 &&
              logits.stride(1) == 1, "argmax_pack: logits");
  const int64_t rows = logits.size(0);
  TORCH_CHECK(packed.scalar_type() == torch::kFloat16 && packed.is_contiguous() &&
              packed.dim() == 3 && packed.size(1) == rows && packed.size(2) == kSlots &&
              packed.size(0) * kSlots <= 32 && rank >= 0 && rank < packed.size(0) && rows >= 1 &&
              limit <= logits.size(1) && vocab_start + logits.size(1) < (1 << 22),
              "argmax_pack: shapes");
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  argmax_pack_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(), logits.stride(0), (int)limit, (int)vocab_start, (int)rank,
      (int)packed.size(0), reinterpret_cast<__half*>(packed.data_ptr<at::Half>()), (int)rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void argmax_select(torch::Tensor packed, torch::Tensor ids) {
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kFloat16 && packed.is_contiguous() &&
              packed.dim() == 3 && packed.size(2) == kSlots && ids.scalar_type() == torch::kInt64 &&
              ids.is_contiguous() && ids.numel() == packed.size(1), "argmax_select: shapes");
  const int64_t rows = packed.size(1);
  const at::cuda::OptionalCUDAGuard guard(device_of(packed));
  select_kernel<<<(rows + 31) / 32, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(packed.data_ptr<at::Half>()), (int)packed.size(0), (int)rows,
      ids.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP = """
void argmax_pack(torch::Tensor logits, torch::Tensor packed, int64_t limit, int64_t vocab_start,
                 int64_t rank);
void argmax_select(torch::Tensor packed, torch::Tensor ids);
"""

ENABLED = os.environ.get("VLLM_INKLING2_SM70_LOCAL_ARGMAX", "0") == "1"
# (default 0: with it on, `hasattr(model, "get_top_tokens")` switches the fork's greedy fast paths on)
SLOTS = 8

_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        build = os.path.join(
            os.environ.get(
                "VLLM_INKLING2_SM70_EXT_DIR",
                os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
            ),
            "local_argmax",
        )
        os.makedirs(build, exist_ok=True)
        _ext = load_inline(
            name="inkling2_sm70_local_argmax",
            cpp_sources=[_CPP],
            cuda_sources=[_CUDA],
            functions=["argmax_pack", "argmax_select"],
            build_directory=build,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
            verbose=bool(os.environ.get("VLLM_INKLING2_SM70_EXT_VERBOSE")),
        )
    return _ext


def pack_local_argmax(
    local_logits: torch.Tensor, limit: int, vocab_start: int, rank: int, tp_size: int
) -> torch.Tensor:
    """`[tp_size, rows, 8]` fp16: this rank's slot holds the argmax (lowest index
    on ties) of `local_logits[:, :limit]` (fp32 `[rows, V_local]`) as exact
    small integers (value bits, id + vocab_start); the other slots are zero."""
    rows = local_logits.shape[0]
    logger.info_once(
        "Inkling2 SM70 shard-local argmax in use (tp=%d, local vocab %d).",
        tp_size,
        local_logits.shape[1],
    )
    packed = torch.empty(
        (tp_size, rows, SLOTS), dtype=torch.float16, device=local_logits.device
    )
    _load().argmax_pack(local_logits, packed, limit, vocab_start, rank)
    return packed


def select_top_tokens(packed: torch.Tensor) -> torch.Tensor:
    """int64 `[rows]`: the id of the largest value over the ranks of the summed
    `[tp_size, rows, 8]` tensor (lowest rank on ties)."""
    ids = torch.empty(packed.shape[1], dtype=torch.int64, device=packed.device)
    _load().argmax_select(packed, ids)
    return ids
