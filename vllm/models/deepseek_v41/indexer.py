# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 lightning indexer (PORT_DESIGN §1 "Indexer" + "Candidate block mask", §3.3; owner L-ATTN).

Constructed on index sources (2, 8, 14, 20 = kv sources; 24, 28, 32, 36 = Reindex). Reference
ref:m.py:488-580. Replicated on every TP rank (as 1Cat V4): all 32 heads, no score all-reduce, so the
top-k is bitwise identical across ranks given identical inputs.

* ``write_keys`` (kv sources): ``k = fp4_e8m0_qdq(rope(k_norm(wk(latent))))`` -> index-K cache rows (and,
  on source 20 of a stage exporting it, ``shared.export_ik``). K comes from the PRE-RoPE compressed latent.
* ``forward``: ``q = fp4_e8m0_qdq(rope(wq_b(qr)))`` [T, 32, 128]; ``w = weights_proj(x) * 128^-1/2 * 32^-1/2``;
  ``score[t, j] = sum_h relu(q[t,h] . k[j]) * w[t,h]`` over the kv source's visible entries
  ``j < (p_t + 1) // ratio`` (FP32); layer 20 publishes candidate blocks (``common.candidate_blocks``),
  24/28/32/36 mask to them; top-``min(512, n)`` re-sorted by position -> ``shared.topk_indices[:T]``.
  When no query of the batch sees more than 512 entries the result is all visible positions and no
  score is computed (layer 20 then publishes ``CAND_ALL``: <= 16,384 positions is a no-op mask).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm.models.deepseek_v41.common.contracts import (
    CAND_ALL,
    CAND_BLOCK,
    CAND_SOURCE,
    CAND_TOPK_BLOCKS,
    CKV_RECORD_DIM,
    HIDDEN,
    IDX_DIM,
    IDX_HEADS,
    IDX_TOPK,
    Q_LORA,
    LayerTopology,
    SharedAttnBuffers,
)
from vllm.models.deepseek_v41.compressor import FP32RMSNorm, mm_fp32, mm_fp32_full
from vllm.models.deepseek_v41.sm70.indexer_kernels import (
    index_k_rope_qat_store,
    index_q_rope_qat,
    index_scores,
    topk_sorted,
)
from vllm.models.deepseek_v41.sm70.sparse import (
    DS41CacheLayer,
    DS41CompressedBackend,
    DS41CompressedMetadata,
    index_k_cache_spec,
    step_metadata,
)
from vllm.models.deepseek_v41.sm70.sparse_kernels import logical_to_rows

SCORE_BUDGET_ENV = "VLLM_DS41_ATTN_INDEXER_SCORE_MIB"


def attn_impl() -> str:
    return knobs.env_str("VLLM_DS41_ATTN_IMPL", "sm70", choices=("sm70", "torch"))


class DeepseekV41Indexer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology,
                 shared: SharedAttnBuffers) -> None:
        super().__init__()
        if not topo.owns_indexer:
            raise ValueError(f"layer {topo.layer_id} is not an index source")
        if not prefix.endswith(".indexer"):
            raise ValueError(f"indexer prefix must end with '.indexer', got {prefix!r}")
        self.prefix = prefix
        self.topo = topo
        self.shared = shared
        self.layer_id = topo.layer_id
        self.ratio = topo.compress_ratio
        attn_prefix = prefix[: -len(".indexer")]
        base = attn_prefix[: attn_prefix.rindex(f".{topo.layer_id}.")]          # "model.layers"
        self.k_cache_name = f"{base}.{topo.kv_source}.attn.indexer.k_cache"
        quant_config = vllm_config.quant_config
        self.wq_b = ReplicatedLinear(Q_LORA, IDX_HEADS * IDX_DIM, bias=False, quant_config=quant_config,
                                     return_bias=False, prefix=f"{prefix}.wq_b")
        self.weights_proj = ReplicatedLinear(HIDDEN, IDX_HEADS, bias=False, quant_config=None,
                                             return_bias=False, prefix=f"{prefix}.weights_proj")
        self.softmax_scale = IDX_DIM ** -0.5
        self.owns_k = topo.owns_compressor
        if self.owns_k:
            self.wk = ReplicatedLinear(CKV_RECORD_DIM, IDX_DIM, bias=False, quant_config=None,
                                       return_bias=False, prefix=f"{prefix}.wk")
            self.k_norm = FP32RMSNorm(IDX_DIM)
            self.k_cache = DS41CacheLayer(self.k_cache_name, index_k_cache_spec(
                vllm_config.cache_config.block_size, self.ratio), DS41CompressedBackend, vllm_config)
        # shared with (and attached by) the attention layer; kept out of the module tree on purpose
        self.__dict__["rotary_emb"] = None
        self.score_budget = knobs.env_int(SCORE_BUDGET_ENV, 128, minimum=8) << 20
        self._static_ctx = vllm_config.compilation_config.static_forward_context

    # ---------------------------------------------------------------------------------- helpers
    def _rope(self) -> torch.Tensor:
        if self.rotary_emb is None:
            raise RuntimeError(f"{self.prefix}: rotary_emb not attached by DeepseekV41Attention")
        return self.rotary_emb.cos_sin_cache

    def _metadata(self) -> DS41CompressedMetadata | None:
        md = step_metadata(self.k_cache_name)
        if md is None:                  # dummy/profile forward without metadata (ForwardContext.is_dummy_run)
            return None
        m = md.get(self.k_cache_name)
        if m is None:
            raise RuntimeError(f"no metadata for {self.k_cache_name}")
        return cast(DS41CompressedMetadata, m)

    def _key_rows(self) -> torch.Tensor:
        layer = self._static_ctx.get(self.k_cache_name)
        if layer is None:
            raise RuntimeError(f"index-K source {self.k_cache_name} is not registered on this stage")
        return cast(DS41CacheLayer, layer).rows()

    @staticmethod
    def _w(lin: nn.Module, shape: tuple[int, int]) -> torch.Tensor:
        w = lin.weight
        if w.dtype != torch.float16 or tuple(w.shape) != shape:
            raise RuntimeError(f"indexer weight must be FP16 {shape}, got {w.dtype} {tuple(w.shape)}")
        return w

    # ---------------------------------------------------------------------------------- K
    def write_keys(self, latent: torch.Tensor, latent_pos: torch.Tensor) -> None:
        if not self.owns_k:
            raise RuntimeError(f"layer {self.layer_id} does not own index keys")
        md = self._metadata()
        wk = self._w(self.wk, (IDX_DIM, CKV_RECORD_DIM))
        if md is None:
            return
        n = latent.shape[0]
        if n != md.num_latents:
            raise RuntimeError(f"{self.prefix}: {n} latents but metadata expects {md.num_latents}")
        # [N, 512] x [512, 128] in FP32 (negligible cost): no FP16 rounding of the latent before the FP4 QAT,
        # which otherwise flips ~0.7 % of index-K records vs the v100-semantic reference
        k = self.k_norm(torch.mm(latent.float(), wk.float().t()))
        export = None
        if self.layer_id == CAND_SOURCE and self.shared.export_ik is not None:
            export = self.shared.export_ik
        index_k_rope_qat_store(k, latent_pos, self._rope(), self._key_rows(), md.latent_slots, export,
                               impl=attn_impl())

    # ---------------------------------------------------------------------------------- top-k
    def forward(self, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor) -> None:
        """x [T,5120] fp16; qr [T,1280] = q_norm(wq_a x), FP32 preferred (fp16 accepted: PORT_DESIGN §3.3 says fp16 --
        DESIGN-CHANGE REQUEST in L-ATTN.progress.md); positions [T] int64. Writes shared.topk_indices[:T] (and
        shared.candidate_blocks[:T] on layer 20)."""
        md = self._metadata()
        if md is None:
            mm_fp32(x, self._w(self.weights_proj, (IDX_HEADS, HIDDEN)))
            return
        T = md.num_actual_tokens
        topk_out = self.shared.topk_indices[:T]
        cand_out = self.shared.candidate_blocks[:T]
        if T == 0:
            return
        if md.max_visible <= IDX_TOPK:
            # every visible entry is selected: [0, 1, ..., n_t - 1, -1, ...]
            ar = torch.arange(IDX_TOPK, device=x.device, dtype=torch.int64)[None, :]
            topk_out.copy_(torch.where(ar < md.num_visible[:, None], ar, torch.full_like(ar, -1)).to(torch.int32))
            if self.topo.is_candidate_source:
                cand_out.fill_(-1)
                cand_out[:, 0] = CAND_ALL
            return
        impl = attn_impl()
        # qr arrives in FP32 and the GEMM output stays FP32, so the only rounding before the FP4 QAT is the QAT
        # itself: FP16 rounding of qr or of q flips QAT values and changes near-tie top-512 picks (synthetic
        # layer 20: 60 % -> 99.5 % of tokens with the reference's exact top-512 set)
        q = mm_fp32_full(qr[:T], self._w(self.wq_b, (IDX_HEADS * IDX_DIM, Q_LORA)))
        q = index_q_rope_qat(q.view(T, IDX_HEADS, IDX_DIM), positions[:T], self._rope(), impl=impl)
        w = mm_fp32(x[:T], self._w(self.weights_proj, (IDX_HEADS, HIDDEN))) * (
            self.softmax_scale * IDX_HEADS ** -0.5)
        keys_all = self._key_rows()
        qsl = md.query_start_loc_cpu
        pos_cpu = md.positions_cpu
        for r in range(md.num_reqs):
            t0, t1 = int(qsl[r]), int(qsl[r + 1])
            if t1 <= t0:
                continue
            n = int((pos_cpu[t1 - 1] + 1) // self.ratio)
            if n == 0:
                topk_out[t0:t1] = -1
                if self.topo.is_candidate_source:
                    cand_out[t0:t1] = -1
                    cand_out[t0:t1, 0] = CAND_ALL
                continue
            j = torch.arange(n, device=x.device, dtype=torch.int64)[None, :]
            req = md.token_to_req_indices[t0:t0 + 1]
            rows = logical_to_rows(j, req, md.block_table, md.storage_block_size)[0]
            keys = keys_all.index_select(0, rows)                                    # [n, 128] fp16
            ends = md.num_visible[t0:t1]
            q_tile = max(1, min(t1 - t0, self.score_budget // (4 * n)))
            key_tile = max(1024, self.score_budget // (4 * IDX_HEADS * q_tile))
            for a in range(t0, t1, q_tile):
                b = min(t1, a + q_tile)
                scores = torch.empty((b - a, n), dtype=torch.float32, device=x.device)
                index_scores(q[a:b], w[a:b], keys, scores, key_tile=key_tile)
                e = ends[a - t0:b - t0]
                if self.topo.is_candidate_source:
                    select_candidate_blocks(scores, e, CAND_TOPK_BLOCKS, CAND_BLOCK, out=cand_out[a:b])
                if self.topo.uses_candidates:
                    apply_candidate_mask(scores, e, cand_out[a:b], CAND_BLOCK)
                else:
                    cols = torch.arange(n, device=x.device)
                    scores.masked_fill_(cols[None, :] >= e[:, None], float("-inf"))
                topk_out[a:b] = topk_sorted(scores, e, IDX_TOPK)
