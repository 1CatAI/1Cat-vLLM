# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 KV compressor (PORT_DESIGN §1 "Compressor ratios 0/1/2", §3.3; owner L-ATTN).

Only kv sources (layers 2, 8, 14: ratio 2; layer 20: ratio 1) construct one. Reference: ref:m.py:429-485.

* ratio 1: ``latent = norm(wkv x)`` for every token (no gate, no state).
* ratio 2: ``kv, score = wkv x, wgate x`` (FP32); group j = tokens (2j, 2j+1);
  ``latent_j = norm(sum_i kv_i * softmax_i(score_i))`` with a per-channel softmax over the 2 members, in FP32.
  Every real token's FP32 ``[kv | score]`` row is written to the paged state cache
  ``{prefix}.state_cache`` (sliding window 2); a token at odd position p completes its group with the
  row of p-1 read back from that cache, which may come from this step or the previous one (decode,
  chunk ending on an even position). The pending odd token is therefore never recomputed.

The latent is returned **pre-RoPE**, RMS-normed (FP32 variance, eps 1e-20), in **FP32** (PORT_DESIGN §3.3 says
fp16 -- DESIGN-CHANGE REQUEST in L-ATTN.progress.md: an FP16 rounding before the FP4 QAT flips ~0.4 % of the
compressed records vs the v100-semantic reference; consumers cast to FP16 only for GEMM operands); ``latent_pos`` is the
position of each group's FIRST token (the reference rotates a latent at ``j * ratio``). Latents are
ordered like ``DS41CompressedMetadata.latent_token_idx`` of the source's compressed cache (ascending
token index), which also gives the rows they are written to. GEMMs: FP16 operands (weights are BF16 in
the checkpoint, loaded as FP16), FP32 accumulate and FP32 output (``torch.mm(out_dtype=float32)``).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.models.deepseek_v41.common.contracts import CKV_RECORD_DIM, HIDDEN, NORM_EPS
from vllm.models.deepseek_v41.sm70.sparse import (
    STATE_ROW_DIM,
    DS41CacheLayer,
    DS41CompressedMetadata,
    DS41StateBackend,
    DS41StateMetadata,
    state_cache_spec,
    step_metadata,
)


def rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """RMSNorm with FP32 variance and FP32 eps (1e-20 underflows to 0 in FP16 -> inf -> NaN on padding
    rows); returns FP32. A zero row stays zero: rsqrt(1e-20) = 1e10 times 0."""
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * weight.float()


class FP32RMSNorm(nn.Module):
    """RMSNorm whose weight is FP32 (PORT_DESIGN §3.7: q/kv/compressor/k norms load as FP32)."""

    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fp32(x, self.weight, self.eps)


def mm_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x [N, K] fp16 @ weight [M, K]^T -> [N, M] fp32 (FP32 accumulate and output)."""
    if weight.dtype != torch.float16 or x.dtype != torch.float16:
        raise TypeError(f"mm_fp32 wants FP16 operands, got x {x.dtype} weight {weight.dtype}")
    return torch.mm(x, weight.t(), out_dtype=torch.float32)


def mm_fp32_full(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 x [N, K] @ FP16 weight [M, K]^T as an FP32 SGEMM (CUDA cores; V100 has no TF32). Used where the result
    feeds an FP4 QAT (indexer q): any rounding before the QAT flips near-tie QAT values and changes top-512 picks.
    vs the L-REF v100-semantic golden (p3_doc, layer 20, 2046 x 4096 q values): SGEMM 0 QAT mismatches; the earlier
    split-FP16 (hi + lo) GEMM left 14 near-tie flips and a 97.85 % minimum top-512 overlap. FP16 x keeps the HMMA
    path (exact products, FP32 accumulate)."""
    if x.dtype == torch.float16:
        return mm_fp32(x, weight)
    return torch.mm(x.float(), weight.float().t())


def compressed_metadata(name: str) -> DS41CompressedMetadata | None:
    md = step_metadata(name)
    if md is None:                  # dummy/profile forward without metadata (ForwardContext.is_dummy_run)
        return None
    m = md.get(name)
    if m is None:
        raise RuntimeError(f"no attention metadata for {name}")
    return cast(DS41CompressedMetadata, m)


class DeepseekV41Compressor(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, compress_ratio: int) -> None:
        super().__init__()
        if compress_ratio not in (1, 2):
            raise ValueError(f"DeepSeek-V4.1 compressor ratio must be 1 or 2, got {compress_ratio}")
        if not prefix.endswith(".compressor"):
            raise ValueError(f"compressor prefix must end with '.compressor', got {prefix!r}")
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.attn_prefix = prefix[: -len(".compressor")]       # = the compressed-KV cache layer name
        hf = vllm_config.model_config.hf_config
        self.eps = float(getattr(hf, "rms_norm_eps", NORM_EPS))
        if compress_ratio == 2:
            self.fused_wkv_wgate = MergedColumnParallelLinear(
                HIDDEN, [CKV_RECORD_DIM, CKV_RECORD_DIM], bias=False, quant_config=None, disable_tp=True,
                return_bias=False, prefix=f"{prefix}.fused_wkv_wgate")
            self.state_cache = DS41CacheLayer(
                f"{prefix}.state_cache", state_cache_spec(vllm_config.cache_config.block_size), DS41StateBackend,
                vllm_config)
        else:
            self.wkv = ReplicatedLinear(HIDDEN, CKV_RECORD_DIM, bias=False, quant_config=None,
                                        return_bias=False, prefix=f"{prefix}.wkv")
            self.state_cache = None
        self.norm = FP32RMSNorm(CKV_RECORD_DIM, self.eps)        # ``…compressor.norm.weight`` FP32

    def _weight(self) -> torch.Tensor:
        lin = self.fused_wkv_wgate if self.compress_ratio == 2 else self.wkv
        w = lin.weight
        rows = 2 * CKV_RECORD_DIM if self.compress_ratio == 2 else CKV_RECORD_DIM
        if w.dtype != torch.float16 or tuple(w.shape) != (rows, HIDDEN):
            raise RuntimeError(f"{self.prefix}: weight must be FP16 [{rows}, {HIDDEN}], got {w.dtype} "
                               f"{tuple(w.shape)}")
        return w

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        md = compressed_metadata(self.attn_prefix)
        w = self._weight()
        if md is None:            # profile / dummy run: exercise the GEMM, emit no latents
            mm_fp32(x, w)
            empty = x.new_empty((0, CKV_RECORD_DIM), dtype=torch.float32)
            return empty, positions.new_empty((0,))
        n, tok = md.num_latents, md.latent_token_idx
        if self.compress_ratio == 1:
            kv = mm_fp32(x.index_select(0, tok), w)                        # [N, 512]
        else:
            T = md.num_actual_tokens
            kv_score = mm_fp32(x[:T], w)                                   # [T, 1024] = [kv | score]
            st = self._state_metadata()
            rows = self._state_rows()
            keep = st.slot_mapping >= 0
            rows.index_copy_(0, st.slot_mapping[keep], kv_score[keep])
            prev = st.prev_slot.index_select(0, tok)
            if n and bool((prev < 0).any()):
                raise RuntimeError(f"{self.prefix}: a completing token has no state row for position p-1")
            pair = torch.stack([rows.index_select(0, prev.clamp(min=0)), kv_score.index_select(0, tok)], dim=1)
            kv_pair, score_pair = pair.split(CKV_RECORD_DIM, dim=-1)       # [N, 2, 512] each
            kv = (kv_pair * score_pair.softmax(dim=1)).sum(dim=1)          # FP32
        latent = self.norm(kv)                     # FP32: rounded only by the QAT (no extra FP16 rounding)
        assert latent.shape[0] == n
        return latent, md.latent_pos

    def _state_metadata(self) -> DS41StateMetadata:
        assert self.state_cache is not None
        md = step_metadata(self.state_cache.layer_name)
        m = md.get(self.state_cache.layer_name) if md is not None else None
        if m is None:
            raise RuntimeError(f"no metadata for {self.state_cache.layer_name}")
        return cast(DS41StateMetadata, m)

    def _state_rows(self) -> torch.Tensor:
        assert self.state_cache is not None
        rows = self.state_cache.rows()
        if rows.shape[-1] != STATE_ROW_DIM or rows.dtype != torch.float32:
            raise RuntimeError(f"{self.state_cache.layer_name}: state rows must be FP32 [*, {STATE_ROW_DIM}]")
        return rows
