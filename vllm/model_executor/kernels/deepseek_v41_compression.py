# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 ratio-2 pooling with bounded speculative partial-group snapshots."""

import torch
from torch import nn

from vllm.model_executor.layers.quantization.gguf_shared_protocol import _fail_if
from vllm.triton_utils import tl, tldevice, triton


@triton.jit(do_not_specialize=["N"])
def _begin(positions, committed_position, metadata, N, BLOCK: tl.constexpr):
    start = tl.load(positions).to(tl.int64)
    previous = tl.load(committed_position)
    pending = tl.load(metadata + 2)
    _fail_if(
        (start < 0)
        | (start + N > 1048576)
        | ((start != 0) & ((start != previous) | (pending != 0)))
    )
    i = tl.arange(0, BLOCK)
    p = tl.load(positions + i, i < N, 0)
    _fail_if(tl.sum(((i < N) & (p != start + i)).to(tl.int32), 0))
    tl.store(metadata, start)
    tl.store(metadata + 1, N)
    tl.store(metadata + 2, 1)


@triton.jit(do_not_specialize=["N"])
def _pool_c2(
    kv,
    score,
    positions,
    partial,
    candidates,
    norm,
    out,
    complete,
    N,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    start = tl.load(positions).to(tl.int64)
    valid = d < D
    current = tl.load(kv + t * D + d, valid, 0)
    current_score = tl.load(score + t * D + d, valid, 0)
    previous = tl.load(kv + (t - 1) * D + d, (t > 0) & valid, 0)
    previous_score = tl.load(score + (t - 1) * D + d, (t > 0) & valid, 0)
    previous = tl.where(t > 0, previous, tl.load(partial + d, valid, 0))
    previous_score = tl.where(t > 0, previous_score, tl.load(partial + D + d, valid, 0))
    done = (start + t + 1) % 2 == 0
    maximum = tl.maximum(previous_score, current_score)
    a = tldevice.exp(previous_score - maximum)
    b = tldevice.exp(current_score - maximum)
    # Preserve the official FP32 softmax/product/sum, then FP16 store before
    # RMSNorm. The two projections have not been rounded to FP16 beforehand.
    value = (
        (tl.div_rn(a, a + b) * previous + tl.div_rn(b, a + b) * current)
        .to(tl.float16)
        .to(tl.float32)
    )
    value = tl.where(done, value, 0)
    variance = tl.sum(tl.where(valid, value * value, 0), 0) / D
    value = value * tl.rsqrt(variance + EPS) * tl.load(norm + d, valid, 0)
    tl.store(out + t * D + d, value, valid)
    tl.store(complete + t, done)
    # Small verification batches retain every prefix. Prefill can only
    # commit its full input and therefore needs just its final partial.
    if N <= 6 or t == N - 1:
        row = tl.where(N <= 6, t + 1, 6)
        tl.store(candidates + row * 2 * D + d, tl.where(done, 0, current), valid)
        tl.store(
            candidates + (row * 2 + 1) * D + d, tl.where(done, 0, current_score), valid
        )
    if t == 0:
        old = tl.load(partial + d, valid, 0)
        old_score = tl.load(partial + D + d, valid, 0)
        tl.store(candidates + d, tl.where(start == 0, 0, old), valid)
        tl.store(candidates + D + d, tl.where(start == 0, 0, old_score), valid)


@triton.jit
def _commit_c2(
    partial,
    candidates,
    metadata,
    position,
    selector,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    count = tl.load(selector).to(tl.int64)
    length, start = tl.load(metadata + 1), tl.load(metadata)
    _fail_if(
        (tl.load(metadata + 2) != 1)
        | (count < 0)
        | (count > length)
        | ((length > 6) & (count != length))
    )
    row = tl.where(length <= 6, count, 6)
    d = tl.arange(0, BLOCK)
    for part in tl.static_range(2):
        value = tl.load(candidates + (row * 2 + part) * D + d, d < D, 0)
        tl.store(partial + part * D + d, value, d < D)
    tl.store(position, start + count)
    tl.store(metadata + 2, 0)


class V41C2State(nn.Module):
    """A ratio-2 source's incomplete group and seven candidate boundaries."""

    def __init__(self, norm_weight, eps=1e-20):
        super().__init__()
        if (
            not norm_weight.is_cuda
            or norm_weight.dtype != torch.float32
            or norm_weight.shape != (512,)
        ):
            raise ValueError("V4.1 C2 normalization requires FP32 width 512 on CUDA")
        self.register_buffer("norm_weight", norm_weight, persistent=False)
        device = norm_weight.device
        self.register_buffer(
            "partial",
            torch.zeros(2, 512, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "candidates",
            torch.empty(7, 2, 512, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "metadata",
            torch.zeros(3, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "position",
            torch.zeros(1, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.eps = eps

    def forward(self, kv, score, positions):
        n = kv.shape[0]
        if (
            kv.ndim != 2
            or kv.shape != score.shape
            or kv.shape[1] != 512
            or not 1 <= n <= 16384
            or positions.shape != (n,)
            or kv.dtype != torch.float32
            or score.dtype != torch.float32
            or positions.dtype not in (torch.int32, torch.int64)
            or any(
                t.device != self.partial.device or not t.is_contiguous()
                for t in (kv, score, positions)
            )
        ):
            raise ValueError("V4.1 C2 requires unrounded FP32 projections/scores")
        result = torch.empty_like(kv, dtype=torch.float16)
        complete = torch.empty(n, dtype=torch.bool, device=kv.device)
        _begin[(1,)](
            positions,
            self.position,
            self.metadata,
            N=n,
            BLOCK=triton.next_power_of_2(n),
        )
        _pool_c2[(n,)](
            kv,
            score,
            positions,
            self.partial,
            self.candidates,
            self.norm_weight,
            result,
            complete,
            N=n,
            D=512,
            EPS=self.eps,
            BLOCK=512,
            enable_fp_fusion=False,
        )
        return result, complete

    def commit(self, accepted_inputs):
        if (
            accepted_inputs.numel() != 1
            or accepted_inputs.dtype not in (torch.int32, torch.int64)
            or accepted_inputs.device != self.partial.device
        ):
            raise ValueError("C2 commit selector must be one CUDA integer")
        _commit_c2[(1,)](
            self.partial,
            self.candidates,
            self.metadata,
            self.position,
            accepted_inputs,
            D=512,
            BLOCK=512,
        )
