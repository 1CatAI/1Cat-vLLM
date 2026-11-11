# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row argmax over a large vocabulary in two short launches.

torch.argmax reduces a few long rows with one CTA per row (about 35 us for
129280 FP32 logits on an 80-SM GPU). Here each row is split into CHUNK-wide parts
reduced in parallel, then the per-part winners are combined. Ties resolve to
the lowest index, as in torch.argmax. VLLM_FAST_ARGMAX=1 enables it.
"""

import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton

ENABLED = envs.VLLM_FAST_ARGMAX == "1"
_CHUNK = 4096


@triton.jit
def _argmax_parts(
    x, stride, n, values, indices, PARTS: tl.constexpr, CHUNK: tl.constexpr
):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * CHUNK + tl.arange(0, CHUNK)
    v = tl.load(x + row * stride + col, col < n, float("-inf")).to(tl.float32)
    best = tl.max(v, 0)
    # Lowest column holding the maximum.
    index = tl.min(tl.where(v == best, col, n), 0)
    tl.store(values + row * PARTS + part, best)
    tl.store(indices + row * PARTS + part, index)


@triton.jit
def _argmax_combine(values, indices, out, parts, PARTS: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, PARTS)
    v = tl.load(values + row * PARTS + p, p < parts, float("-inf"))
    i = tl.load(indices + row * PARTS + p, p < parts, 2147483647)
    best = tl.max(v, 0)
    # Parts are in column order, so the lowest index among winners is first.
    tl.store(out + row, tl.min(tl.where(v == best, i, 2147483647), 0).to(tl.int64))


def argmax_rows(x: torch.Tensor) -> torch.Tensor:
    """argmax(dim=-1) of a 2-D tensor; the fast path for long CUDA rows."""
    if (
        not ENABLED
        or not x.is_cuda
        or x.dim() != 2
        or x.shape[1] < 8 * _CHUNK
        or x.shape[0] == 0
        or x.stride(1) != 1
        or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        return x.argmax(dim=-1)
    rows, n = x.shape
    parts = triton.cdiv(n, _CHUNK)
    width = triton.next_power_of_2(parts)
    values = torch.empty((rows, width), dtype=torch.float32, device=x.device)
    indices = torch.empty((rows, width), dtype=torch.int32, device=x.device)
    _argmax_parts[(rows, parts)](
        x, x.stride(0), n, values, indices, PARTS=width, CHUNK=_CHUNK, num_warps=8
    )
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    _argmax_combine[(rows,)](values, indices, out, parts, PARTS=width)
    return out
