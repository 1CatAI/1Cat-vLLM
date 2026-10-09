# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hyper-connection math for DeepSeek-V4.1 (PORT_DESIGN §3.2, §4.1; reference inference/model.py:948-966 and
kernel.py:406-474 ``hc_split_sinkhorn``).

The residual stream is HC=4 copies of the hidden state, stored BF16 (software type on SM70; never FP16, A4).
All mixing math is FP32. V4.1 uses the *single-pass shift*: a sublayer's ``pre`` collapses the NEXT sublayer's
input, so each decoder block returns its FFN ``pre`` and the final collapse is ``hc_pre(h, last ffn_pre)``.

The torch implementation processes at most ``_CHUNK`` tokens at a time so the FP32 temporaries stay bounded
(~84 MiB each at 1024 tokens) during 4K-token prefill chunks. A fused Triton kernel is a P5 item.
"""

from __future__ import annotations

import torch

from .contracts import HC, HC_EPS, HC_SINKHORN_ITERS, HIDDEN, NORM_EPS, STREAM_DTYPE

MIX_HC = (2 + HC) * HC  # 24 = pre(4) + post(4) + comb(16)
_CHUNK = 1024


def _check_stream(stream: torch.Tensor) -> None:
    if stream.dim() != 3 or stream.shape[1] != HC or stream.shape[2] != HIDDEN:
        raise ValueError(f"HC stream must be [T, {HC}, {HIDDEN}], got {tuple(stream.shape)}")
    if stream.dtype != STREAM_DTYPE:
        raise TypeError(f"HC stream must be {STREAM_DTYPE}, got {stream.dtype}")


def _chunks(num_tokens: int):
    for start in range(0, num_tokens, _CHUNK):
        yield start, min(num_tokens, start + _CHUNK)


def sinkhorn_split(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                   iters: int = HC_SINKHORN_ITERS, eps: float = HC_EPS
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mixes [T,24] f32 -> pre [T,4], post [T,4], comb [T,4,4] (op order of kernel.py hc_split_sinkhorn)."""
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + eps
    post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = mixes[:, 2 * HC:].view(-1, HC, HC) * scale[2] + base[2 * HC:].view(HC, HC)
    # comb = softmax(comb, -1) + eps
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    # comb = comb / (comb.sum(-2) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """stream [T,4,5120] bf16; fn [24,20480] f32; scale [3] f32; base [24] f32.
    Returns pre [T,4] f32 (sigmoid+eps), post [T,4] f32 (2*sigmoid), comb [T,4,4] f32 (Sinkhorn, 20 iters).
    rsqrt(mean(x^2) + 1e-20) over the flattened 20480 values, FP32 GEMM (no HMMA: fn is FP32-only)."""
    _check_stream(stream)
    if fn.dtype != torch.float32 or fn.shape != (MIX_HC, HC * HIDDEN):
        raise ValueError(f"hc fn must be float32 [{MIX_HC}, {HC * HIDDEN}], got {fn.dtype} {tuple(fn.shape)}")
    num_tokens = stream.shape[0]
    mixes = torch.empty((num_tokens, MIX_HC), dtype=torch.float32, device=stream.device)
    for start, end in _chunks(num_tokens):
        x = stream[start:end].flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)
        torch.mul(torch.nn.functional.linear(x, fn), rsqrt, out=mixes[start:end])
    return sinkhorn_split(mixes, scale.float(), base.float())


def hc_pre(stream: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """sum_c pre[:,c] * stream[:,c,:] in FP32, rounded to bf16 -> [T,5120] bf16 (ref:m.py:957-960)."""
    _check_stream(stream)
    out = torch.empty((stream.shape[0], HIDDEN), dtype=STREAM_DTYPE, device=stream.device)
    for start, end in _chunks(stream.shape[0]):
        y = torch.sum(pre[start:end].unsqueeze(-1) * stream[start:end].float(), dim=1)
        out[start:end] = y.to(STREAM_DTYPE)
    return out


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """x [T,5120] f32 (sublayer output); residual [T,4,5120] bf16 -> [T,4,5120] bf16:
    y[:,c] = post[:,c]*x + sum_j comb[:,j,c]*residual[:,j]   (FIRST comb index contracts, ref:m.py:965)."""
    _check_stream(residual)
    if x.shape != (residual.shape[0], HIDDEN):
        raise ValueError(f"sublayer output must be [T, {HIDDEN}], got {tuple(x.shape)}")
    out = torch.empty_like(residual)
    for start, end in _chunks(residual.shape[0]):
        # [t, c, j] @ [t, j, d] -> [t, c, d]: comb transposed so the first comb index contracts
        mixed = torch.bmm(comb[start:end].transpose(1, 2), residual[start:end].float())
        mixed.addcmul_(post[start:end].unsqueeze(-1), x[start:end].float().unsqueeze(1))
        out[start:end] = mixed.to(STREAM_DTYPE)
    return out


def rmsnorm_to_act(x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """bf16/f32 in, FP32 variance + FP32 eps, * weight, -> fp16 [T,5120]. Zero rows -> zeros."""
    out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    weight32 = weight.float()
    for start, end in _chunks(x.shape[0]):
        x32 = x[start:end].float()
        var = x32.square().mean(-1, keepdim=True)
        out[start:end] = (weight32 * (x32 * torch.rsqrt(var + eps))).to(torch.float16)
    return out


def hc_expand(embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First stage: [T,5120] embedding -> (stream [T,4,5120] bf16 with 4 identical copies, identity pre-mix
    [T,4] f32 = one-hot copy 0) (ref:m.py:1253-1257, make_identity_pre_mix)."""
    if embedded.dim() != 2 or embedded.shape[1] != HIDDEN:
        raise ValueError(f"embedding must be [T, {HIDDEN}], got {tuple(embedded.shape)}")
    stream = embedded.to(STREAM_DTYPE).unsqueeze(1).expand(-1, HC, -1).contiguous()
    pre = torch.zeros((embedded.shape[0], HC), dtype=torch.float32, device=embedded.device)
    pre[:, 0] = 1.0
    return stream, pre


def hc_collapse(stream: torch.Tensor, pre: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
    """Final collapse on the last stage: rmsnorm(hc_pre(stream, last ffn_pre)) -> [T,5120] fp16
    (ref:m.py:1268-1269; V4.1 has no hc_head_*)."""
    return rmsnorm_to_act(hc_pre(stream, pre), norm_weight)
