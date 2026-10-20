# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark drafter for DeepSeek-V4.1 on SM70 (gpubox lane; WIP, see /rivet-shared/ds41-v100/AGENT.md "DSpark build spec").

Reference: ``inference/model.py`` 1023-1160 (DSparkAttention / DSparkBlock). Per decode step the target's aux states
(the HC stream entering layers 37-39, averaged over the 4 streams: [T, 3 x 5120]) go through ``main_proj`` + ``main_norm``
= ``main_x``; the drafter's window cache stores K/V = RoPE(kv_norm(wkv(main_x))) at the step position
(``precompute_and_store_context_kv``). The draft input is embed([bonus token, noise x 4]) expanded to the 4 HC streams,
run through 3 blocks (SWA-only, MoE 128 experts top-3, single-pass HC, identity pre-mix at block 0). Each draft query
attends NON-causally over the window entries (positions <= p, the last accepted token) and all 5 block K/V
(kv_norm(wkv(x)), ephemeral). Head: hc_pre(x, pre) -> norm -> the target's lm_head; the Markov head adds a bias per
position from the previous sampled draft token; the confidence head is optional.

The DSpark hyper-parameters are in the checkpoint's ``inference/config.json`` (not ``config.json``); serve with
``--hf-overrides '{"dspark_block_size": 5, "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [37, 38, 39],
"dspark_markov_rank": 256, "dspark_n_routed_experts": 128, "dspark_n_activated_experts": 3, "n_mtp_layers": 3}'``
and ``--speculative-config '{"method": "dspark", "num_speculative_tokens": 5}'`` (num_speculative_tokens = block size:
the proposer's query rows per request are exactly the block).

Proposer contract (vllm/v1/spec_decode/{dspark,dflash,llm_base_proposer}.py): embed_input_ids, combine_hidden_states,
precompute_and_store_context_kv, forward(input_ids, positions, inputs_embeds) -> Tensor, compute_logits, markov_embed,
markov_bias, confidence_logits, map_draft_to_target, get_draft_kv_cache_layer_names, skip_checkpoint_weight;
has_own_embed_tokens = has_own_lm_head = False.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import os as _os
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v41.common import contracts as C
from vllm.models.deepseek_v41.common.hc import hc_expand, hc_mixes, hc_post, hc_pre, rmsnorm_to_act
from vllm.models.deepseek_v41.common.topology import layer_topology
from vllm.models.deepseek_v41.sm70.sparse import (
    DS41CacheLayer,
    DS41DraftSWABackend,
    DS41SWABackend,
    DS41SWAMetadata,
    rows_of_positions,
    step_metadata,
    swa_cache_spec,
)
from vllm.models.deepseek_v41.sm70.sparse_kernels import fused_decode_attention

from .model import DeepseekV41Norm, _fp32_param, map_checkpoint_name

logger = init_logger(__name__)

BLOCK_KEYS_PAD = 16          # block K/V rows per query, padded to the fused kernel's 16-key blocks
STATIC_CAPTURE_CTX: dict | None = None    # {cache layer name: (its KV group's block table [max_reqs, max_blocks] int32 on GPU, block size)}; proposer sets
MAIN_PROJ_INPUT_SCALE = 2.0 ** -6   # exact FP16 scaling; main_norm removes it (V4 drafter does the same on SM70)


class DSparkMarkovHead(nn.Module):
    """``mtp.2.markov_head.embed`` [V, R] and ``.head`` [V, R]: bias(token) = head @ embed[token]."""

    def __init__(self, vocab: int, rank: int, prefix: str) -> None:
        super().__init__()
        self.embed = VocabParallelEmbedding(vocab, rank, prefix=f"{prefix}.embed")
        self.head = ReplicatedLinear(rank, vocab, bias=False, return_bias=False, quant_config=None,
                                     prefix=f"{prefix}.head")

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(token_ids)

    def bias(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.head(embeds)


class DSparkConfidenceHead(nn.Module):
    """``mtp.2.confidence_head.proj``: FP32 linear (H + R -> 1)."""

    def __init__(self, in_dim: int, prefix: str) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat((hidden.float(), markov.float()), dim=-1)).squeeze(-1)


class DSparkV41Attention(nn.Module):
    """The draft block's attention: the target's attention projections (wq_a/q_norm/wq_b, wkv/kv_norm, wo_a/wo_b),
    context K/V from ``main_x`` (window cache), block K/V from ``x`` (ephemeral), non-causal over window + block."""

    def __init__(self, vllm_config: VllmConfig, prefix: str, layer_id: int) -> None:
        super().__init__()
        from vllm.models.deepseek_v41.attention import DeepseekV41Attention
        from vllm.models.deepseek_v41.common.contracts import StagePlan, allocate_shared_attn_buffers

        hf = vllm_config.model_config.hf_config
        topo = layer_topology(hf, layer_id)                       # 40..42: SWA-only
        # a one-stage plan without KV sources: the drafter owns only window caches
        stage = StagePlan(pp_rank=0, pp_size=1, first_layer=layer_id, last_layer=layer_id, mirrored_kv_sources=(),
                          exports_kv_sources=(), engram_layers=())
        shared = allocate_shared_attn_buffers(vllm_config.scheduler_config.max_num_batched_tokens, stage,
                                              torch.device("cuda"))
        # reuse the target's attention module for its weights, norms, rotary cache and output projection; its
        # forward() is not used (the drafter's attention pattern differs)
        self.inner = DeepseekV41Attention(vllm_config, prefix, topo, stage, shared, aux_streams=None)
        self.inner.swa_cache._backend = DS41DraftSWABackend      # metadata flagged non-causal for the proposer
        self.prefix = prefix
        self.layer_id = layer_id
        self.n_local_heads = self.inner.n_local_heads
        self.tp_size = get_tensor_model_parallel_world_size()

    @property
    def swa_cache(self) -> DS41CacheLayer:
        return self.inner.swa_cache

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T, 5120] fp16 -> (q [T, H, 512] fp16 un-rotated, kv [T, 512] fp32 normed un-rotated)."""
        from vllm.models.deepseek_v41.attention import _fp16_weight, mm_fp32
        from vllm.models.deepseek_v41.common.contracts import HEAD_DIM, HIDDEN, Q_LORA

        inner = self.inner
        w_qkv = _fp16_weight(inner.fused_wqa_wkv, (Q_LORA + HEAD_DIM, HIDDEN), f"{self.prefix}.fused_wqa_wkv")
        w_qb = _fp16_weight(inner.wq_b, (self.n_local_heads * HEAD_DIM, Q_LORA), f"{self.prefix}.wq_b")
        qr_kv = mm_fp32(x, w_qkv)
        qr = inner.q_norm(qr_kv[:, :Q_LORA]).to(torch.float16)
        kv = inner.kv_norm(qr_kv[:, Q_LORA:])
        q = torch.mm(qr, w_qb.t()).view(x.shape[0], self.n_local_heads, HEAD_DIM)
        return q, kv

    def store_context_kv(self, main_x: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """K/V of the target's combined states at their positions -> the window cache (RoPE + FP8 QAT + insert)."""
        from vllm.models.deepseek_v41.sm70.q_rope_kv_insert import q_rope_kv_insert

        _, kv = self._qkv(main_x)
        if dump_on():
            dump_put(f"win_kv_L{self.layer_id}", kv); dump_put(f"win_main_x_L{self.layer_id}", main_x)
        q_dummy = torch.empty((main_x.shape[0], self.n_local_heads, C.HEAD_DIM), dtype=torch.float16,
                              device=main_x.device)
        q_rope_kv_insert(q_dummy, kv, positions, self.inner.rotary_emb.cos_sin_cache, self.swa_cache.rows(),
                         slot_mapping)

    def forward(self, positions: torch.Tensor, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """x [B * block_size, 5120] fp16 (draft stream, block-major); positions [B * block_size] int64."""
        from vllm.models.deepseek_v41.common.qat import fp8_block32_qdq
        from vllm.models.deepseek_v41.common.rope import apply_rope_torch

        from vllm.forward_context import get_forward_context
        md_all = get_forward_context().attn_metadata
        if not isinstance(md_all, dict):
            md_all = None             # the proposer's own dummy / profile runs carry no metadata (and no dummy flag)
        o = torch.zeros((x.shape[0], self.n_local_heads, C.HEAD_DIM), dtype=torch.float16, device=x.device)
        import os as _os
        if md_all is not None and _os.environ.get("DS41_DRAFT_FORCE_STATIC") and STATIC_CAPTURE_CTX is not None \
                and _os.environ.get("DS41_DRAFT_FORCE_STATIC_LAYER", str(self.layer_id)) == str(self.layer_id):
            md_all = None                                            # test: the static (capture) math on real eager steps
        if md_all is None:
            # No metadata: a profile/warmup run (zero attention), or the proposer's CUDA-graph CAPTURE. Under capture the
            # graph must contain the real attention, so it is built from STATIC inputs that the replay also sees: the
            # draft KV group's persistent block table (rows = request index, requests in batch order = block order),
            # the proposer's persistent positions buffer (``positions``) and the window cache. Padded rows compute
            # garbage that nothing reads. ``STATIC_CAPTURE_CTX`` is installed by the proposer before capture.
            st = STATIC_CAPTURE_CTX.get(self.swa_cache.layer_name) if STATIC_CAPTURE_CTX else None   # EACH drafter layer has its own KV group / block table
            bound = st is not None and (torch.cuda.is_current_stream_capturing() or bool(_os.environ.get("DS41_DRAFT_FORCE_STATIC")))
            if bound:
                try:                        # memory profiling captures with a stub cache: not the real capture
                    self.swa_cache.rows()
                except RuntimeError:
                    bound = False
            if not bound:
                self._qkv(x)
                return self.inner.output_projection(o, positions)
            block_table, ctx_block_size = st
            # the window cache's row geometry is the cache layer's spec (what the SWA metadata builder uses), NOT the
            # proposer's draft-group block size: with a mismatch every window row is wrong (counting acceptance 11%)
            cache_block_size = int(self.swa_cache._spec.block_size)
            T = (x.shape[0] // block_size) * block_size
            B = T // block_size
            # padded rows can form more blocks than there are requests (rows = graph size, not B x block_size):
            # those blocks read the last real table row (their output is never consumed) instead of running off it
            tok2req = torch.arange(B, device=x.device, dtype=torch.int32).clamp_(max=block_table.shape[0] - 1)
            if _os.environ.get("DS41_DRAFT_CAPTURE_LOG") and self.__dict__.get("_cap_logged", 0) < 3:
                self._cap_logged = self.__dict__.get("_cap_logged", 0) + 1
                logger.info("DSpark drafter L%d STATIC path: capturing=%s rows=%d T=%d B=%d bt=%#x %s bs=%d (ctx said %d) cache=%#x %s pos=%#x x=%#x",
                            self.layer_id, torch.cuda.is_current_stream_capturing(), x.shape[0], T, B, block_table.data_ptr(),
                            tuple(block_table.shape), cache_block_size, ctx_block_size, self.swa_cache.rows().data_ptr(),
                            tuple(self.swa_cache.rows().shape), positions.data_ptr(), x.data_ptr())
        else:
            swa_md = cast(DS41SWAMetadata, md_all[self.swa_cache.layer_name])
            T = swa_md.num_actual_tokens                             # real query rows (B blocks of block_size)
            if T % block_size:
                raise RuntimeError(f"DSpark drafter: {T} query rows is not a multiple of the block size {block_size}")
            B = T // block_size
            block_table, cache_block_size = swa_md.block_table, swa_md.block_size
            tok2req = swa_md.token_to_req_indices.view(B, block_size)[:, 0]
            if _os.environ.get("DS41_DRAFT_CAPTURE_LOG") and self.__dict__.get("_real_logged", 0) < 3:
                self._real_logged = self.__dict__.get("_real_logged", 0) + 1
                logger.info("DSpark drafter L%d REAL path: rows=%d T=%d B=%d bt=%#x %s bs=%d tok2req=%s cache=%#x pos=%#x x=%#x positions=%s",
                            self.layer_id, x.shape[0], T, B, block_table.data_ptr(), tuple(block_table.shape), cache_block_size,
                            tok2req.tolist(), self.swa_cache.rows().data_ptr(), positions.data_ptr(), x.data_ptr(), positions[:T].tolist())
        x, positions_all = x[:T], positions
        positions = positions[:T]
        q, kv = self._qkv(x)
        cos_sin = self.inner.rotary_emb.cos_sin_cache
        # block K/V: rotated at the draft positions, window-KV QAT (FP8 block-32), kept in a [T, 512] fp16 buffer
        block_kv = fp8_block32_qdq(apply_rope_torch(kv.float(), positions, cos_sin), out_dtype=torch.float16)
        q = apply_rope_torch(q, positions, cos_sin).to(torch.float16).contiguous()
        # window: for every query of a block, the cache positions p - 127 .. p where p = (first draft position) - 1
        first = positions.view(B, block_size)[:, 0]                  # [B]
        p = first - 1
        win_pos = p[:, None] - (C.WINDOW - 1) + torch.arange(C.WINDOW, device=x.device)[None, :]      # [B, 128]
        win_rows = rows_of_positions(block_table, tok2req, win_pos, cache_block_size)                   # [B, 128]
        if _os.environ.get("DS41_DRAFT_CAPTURE_LOG") and md_all is not None and STATIC_CAPTURE_CTX is not None \
                and self.__dict__.get("_cmp_logged", 0) < 2:
            # in-process comparison: what the STATIC branch would have computed from the same call
            self._cmp_logged = self.__dict__.get("_cmp_logged", 0) + 1
            sbt, _ = STATIC_CAPTURE_CTX[self.swa_cache.layer_name]
            s_tok2req = torch.arange(B, device=x.device, dtype=torch.int32)
            s_rows = rows_of_positions(sbt, s_tok2req, win_pos, int(self.swa_cache._spec.block_size))
            logger.info("DSpark drafter L%d CMP: bt same storage %s; win_rows equal %s; real rows[0,-4:] %s static %s; "
                        "tok2req real %s dtype %s; positions dtype %s contiguous %s; x contiguous %s", self.layer_id,
                        sbt.data_ptr() == block_table.data_ptr(), bool(torch.equal(win_rows, s_rows)),
                        win_rows[0, -4:].tolist(), s_rows[0, -4:].tolist(), tok2req.tolist(), tok2req.dtype,
                        positions.dtype, positions.is_contiguous(), x.is_contiguous())
        win_rows = win_rows[:, None, :].expand(B, block_size, C.WINDOW).reshape(T, C.WINDOW)
        # block rows: query i of block b sees block rows b * block_size .. + block_size - 1 (non-causal)
        brow = (torch.arange(B, device=x.device)[:, None] * block_size
                + torch.arange(BLOCK_KEYS_PAD, device=x.device)[None, :])                            # [B, 16]
        brow = torch.where(torch.arange(BLOCK_KEYS_PAD, device=x.device)[None, :] < block_size, brow,
                           torch.full_like(brow, -1))
        brow = brow[:, None, :].expand(B, block_size, BLOCK_KEYS_PAD).reshape(T, BLOCK_KEYS_PAD)
        fused_decode_attention(q, [(self.swa_cache.rows(), win_rows), (block_kv, brow)], self.inner.attn_sink,
                               self.inner.scale, o[:T])
        if _os.environ.get("DS41_DRAFT_CAPTURE_LOG") and md_all is not None and STATIC_CAPTURE_CTX is not None \
                and self.__dict__.get("_cmp2_logged", 0) < 2:
            # same call, STATIC-derived inputs end to end -> attention output difference
            self._cmp2_logged = self.__dict__.get("_cmp2_logged", 0) + 1
            sbt, _ = STATIC_CAPTURE_CTX[self.swa_cache.layer_name]
            sT = (positions_all.shape[0] // block_size) * block_size; sB = sT // block_size
            s_first = positions_all[:sT].view(sB, block_size)[:, 0]; s_p = s_first - 1
            s_win_pos = s_p[:, None] - (C.WINDOW - 1) + torch.arange(C.WINDOW, device=x.device)[None, :]
            s_rows = rows_of_positions(sbt, torch.arange(sB, device=x.device, dtype=torch.int32).clamp_(max=sbt.shape[0] - 1), s_win_pos, int(self.swa_cache._spec.block_size))
            s_rows = s_rows[:, None, :].expand(sB, block_size, C.WINDOW).reshape(sT, C.WINDOW)
            o2 = torch.zeros_like(o)
            fused_decode_attention(q, [(self.swa_cache.rows(), s_rows), (block_kv, brow)], self.inner.attn_sink, self.inner.scale, o2[:T])
            logger.info("DSpark drafter L%d CMP2: sT %d T %d; max|o-o2| %.3e; |o| %.3e; rows equal %s", self.layer_id, sT, T,
                        float((o[:T] - o2[:T]).abs().max()), float(o[:T].abs().max()), bool(torch.equal(win_rows, s_rows)))
        if dump_on():
            dump_put(f"attn_x_L{self.layer_id}", x); dump_put(f"attn_q_L{self.layer_id}", q)
            dump_put(f"attn_block_kv_L{self.layer_id}", block_kv); dump_put(f"attn_win_rows_L{self.layer_id}", win_rows)
            dump_put(f"attn_win_pos_L{self.layer_id}", win_pos)
            cache = self.swa_cache.rows(); r0 = win_rows[0]
            ok = (r0 >= 0) & (r0 < cache.shape[0])
            dump_put(f"attn_win_valid_L{self.layer_id}", ok)
            dump_put(f"attn_win_kv_rows_L{self.layer_id}", cache[r0[ok]])
            dump_put(f"attn_o_L{self.layer_id}", o[:T])
        return self.inner.output_projection(o, positions_all)


class DSparkV41Block(nn.Module):
    """One draft block: the target decoder layer's structure (HC, attention, MoE) with the drafter's attention."""

    def __init__(self, vllm_config: VllmConfig, prefix: str, layer_id: int) -> None:
        super().__init__()
        from vllm.models.deepseek_v41.sm70.moe import DeepseekV41MoE

        hf = vllm_config.model_config.hf_config
        self.layer_id = layer_id
        self.attn = DSparkV41Attention(vllm_config, f"{prefix}.attn", layer_id)
        self.ffn = DeepseekV41MoE(vllm_config, f"{prefix}.ffn", layer_id,
                                  n_routed_experts=int(hf.dspark_n_routed_experts),
                                  top_k=int(hf.dspark_n_activated_experts), spill=None, exl3=False)
        self.attn_norm = DeepseekV41Norm(C.HIDDEN)
        self.ffn_norm = DeepseekV41Norm(C.HIDDEN)
        mix, hc_dim = (2 + C.HC) * C.HC, C.HC * C.HIDDEN
        self.hc_attn_fn = _fp32_param(mix, hc_dim)
        self.hc_ffn_fn = _fp32_param(mix, hc_dim)
        self.hc_attn_base = _fp32_param(mix)
        self.hc_ffn_base = _fp32_param(mix)
        self.hc_attn_scale = _fp32_param(3)
        self.hc_ffn_scale = _fp32_param(3)

    def forward(self, stream: torch.Tensor, pre_in: torch.Tensor, positions: torch.Tensor, block_size: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
        L = self.layer_id; on = dump_on()
        if on:
            dump_put(f"blk_stream_in_L{L}", stream); dump_put(f"blk_pre_in_L{L}", pre_in)
        attn_pre, attn_post, attn_comb = hc_mixes(stream, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = rmsnorm_to_act(hc_pre(stream, pre_in), self.attn_norm.weight)
        attn_out = self.attn(positions, x, block_size)
        stream = hc_post(attn_out, stream, attn_post, attn_comb)
        if on:
            dump_put(f"blk_attn_pre_L{L}", attn_pre); dump_put(f"blk_attn_post_L{L}", attn_post); dump_put(f"blk_attn_comb_L{L}", attn_comb)
            dump_put(f"blk_x_attn_L{L}", x); dump_put(f"blk_attn_out_L{L}", attn_out); dump_put(f"blk_stream_mid_L{L}", stream)
        ffn_pre, ffn_post, ffn_comb = hc_mixes(stream, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = rmsnorm_to_act(hc_pre(stream, attn_pre), self.ffn_norm.weight)
        if _os.environ.get("DS41_DRAFT_NO_MOE") == "1":
            ffn_out = torch.zeros_like(x)        # diagnostic: bisect graph-only faults (drafter MoE off)
        else:
            ffn_out = self.ffn(x)
        stream = hc_post(ffn_out, stream, ffn_post, ffn_comb)
        if on:
            dump_put(f"blk_ffn_pre_L{L}", ffn_pre); dump_put(f"blk_ffn_post_L{L}", ffn_post); dump_put(f"blk_ffn_comb_L{L}", ffn_comb)
            dump_put(f"blk_x_ffn_L{L}", x); dump_put(f"blk_ffn_out_L{L}", ffn_out); dump_put(f"blk_stream_out_L{L}", stream)
        return stream, ffn_pre


class DSparkDeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", use_confidence_head: bool = False) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        hf = vllm_config.speculative_config.draft_model_config.hf_config
        self.hf = hf
        self.block_size = int(hf.dspark_block_size)
        self.noise_token_id = int(hf.dspark_noise_token_id)
        self.target_layer_ids = tuple(int(i) for i in hf.dspark_target_layer_ids)
        self.num_blocks = int(getattr(hf, "n_mtp_layers", 3) or 3)
        n_backbone = int(hf.num_hidden_layers)
        # replaced by the target's embedding after loading (has_own_embed_tokens = False)
        self.embed_tokens = VocabParallelEmbedding(hf.vocab_size, C.HIDDEN, prefix=maybe_prefix(prefix, "embed_tokens"))
        self.main_proj = ReplicatedLinear(C.HIDDEN * len(self.target_layer_ids), C.HIDDEN, bias=False,
                                          return_bias=False, quant_config=vllm_config.quant_config,
                                          prefix=maybe_prefix(prefix, "main_proj"))
        self.main_norm = DeepseekV41Norm(C.HIDDEN)
        self.layers = nn.ModuleList([
            DSparkV41Block(vllm_config, maybe_prefix(prefix, f"layers.{n_backbone + i}"), n_backbone + i)
            for i in range(self.num_blocks)])
        self.norm = DeepseekV41Norm(C.HIDDEN)
        self.markov_head = DSparkMarkovHead(hf.vocab_size, int(hf.dspark_markov_rank),
                                            maybe_prefix(prefix, "markov_head"))
        self.confidence_head = (DSparkConfidenceHead(C.HIDDEN + int(hf.dspark_markov_rank),
                                                     maybe_prefix(prefix, "confidence_head"))
                                if use_confidence_head else None)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux: torch.Tensor) -> torch.Tensor:
        """aux [T, 3 x 5120] fp16 -> main_x [T, 5120] fp16 (main_proj + main_norm; power-of-two prescale)."""
        projected = self.main_proj((aux * MAIN_PROJ_INPUT_SCALE).to(torch.float16))
        out = rmsnorm_to_act(projected, self.main_norm.weight)
        if dump_on():
            dump_put("aux_in", aux); dump_put("aux_projected", projected); dump_put("aux_main_x", out)
        return out

    @torch.inference_mode()
    def precompute_and_store_context_kv(self, main_x: torch.Tensor, positions: torch.Tensor,
                                        slot_mappings: dict[str, torch.Tensor] | None) -> None:
        for layer in self.layers:
            cache = layer.attn.swa_cache
            slots = None if slot_mappings is None else slot_mappings.get(cache.prefix)
            if slots is not None:
                layer.attn.store_context_kv(main_x, positions, slots)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, inputs_embeds: torch.Tensor | None = None
                ) -> torch.Tensor:
        """input_ids / positions [B * block_size] (block-major: bonus token then block_size - 1 noise tokens per
        request) -> hidden [B * block_size, 5120] fp16 (pre-norm; ``compute_logits`` applies ``norm`` + lm_head)."""
        emb = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        if dump_on():
            dump_put("emb", emb)
        stream, pre = hc_expand(emb.to(torch.float16))
        for layer in self.layers:
            stream, pre = layer(stream, pre, positions, self.block_size)
        if dump_on():
            dump_put("final_stream", stream); dump_put("final_pre", pre)
        return hc_pre(stream, pre).to(torch.float16)


_DUMP_DIR = __import__("os").environ.get("VLLM_DS41_DSPARK_DUMP", "")
_DUMP_ROUNDS = 4
_dump: dict = {"round": 0, "cur": {}}


def dump_on() -> bool:
    """True when a dump is active for this forward (dir set, rounds left, not inside a graph capture)."""
    if not _DUMP_DIR or _dump["round"] >= _DUMP_ROUNDS:
        return False
    import torch as _t
    return not _t.cuda.is_current_stream_capturing()


def dump_put(key: str, value) -> None:
    """VLLM_DS41_DSPARK_DUMP=<dir>: record per-round proposer tensors (first _DUMP_ROUNDS rounds, TP rank 0 of its stage)."""
    if not _DUMP_DIR or _dump["round"] >= _DUMP_ROUNDS:
        return
    import torch as _t
    if _t.cuda.is_current_stream_capturing():          # a D2H inside a graph capture invalidates the capture
        return
    v = value.detach().to("cpu", copy=True) if isinstance(value, _t.Tensor) else value
    _dump["cur"].setdefault(key, []).append(v)


def dump_flush(model) -> None:
    if not _DUMP_DIR or _dump["round"] >= _DUMP_ROUNDS or not _dump["cur"]:
        return
    if not any(k.startswith("attn_o_L") or k.startswith("prop_") for k in _dump["cur"]):   # dummy / profile round: discard
        _dump["cur"] = {}
        return
    import os as _os, torch as _t
    from vllm.distributed import get_tensor_model_parallel_rank
    r = get_tensor_model_parallel_rank()
    _os.makedirs(_DUMP_DIR, exist_ok=True)
    if _dump["round"] == 0:
        _t.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f"{_DUMP_DIR}/drafter_state_tp{r}.pt")
    _t.save(_dump["cur"], f"{_DUMP_DIR}/round{_dump['round']}_tp{r}.pt")
    _dump["cur"] = {}
    _dump["round"] += 1


class DSparkDeepseekV41ForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        spec = vllm_config.speculative_config
        self.config = spec.draft_model_config.hf_config
        self._confidence_required = bool(spec.dspark_confidence_threshold > 0.0)
        self._confidence_loaded = False
        self.model = DSparkDeepseekV41Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
                                            use_confidence_head=self._confidence_required)
        self.lm_head = ParallelLMHead(self.config.vocab_size, C.HIDDEN, prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # ---- proposer contract ----
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mappings=None) -> None:
        if dump_on():
            dump_put("ctx_states", context_states); dump_put("ctx_positions", context_positions)
            dump_put("ctx_slots", None if context_slot_mappings is None else {k: v.detach().cpu() for k, v in context_slot_mappings.items()})
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mappings)

    def forward(self, input_ids, positions, inputs_embeds=None) -> torch.Tensor:
        dump_put("input_ids", input_ids); dump_put("positions", positions)
        if inputs_embeds is not None:
            dump_put("inputs_embeds", inputs_embeds)
        out = self.model(input_ids, positions, inputs_embeds)
        dump_put("hidden_out", out)
        if dump_on():
            dump_put("hidden_normed", rmsnorm_to_act(out, self.model.norm.weight))
        return out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, rmsnorm_to_act(hidden_states, self.model.norm.weight))
        if logits is not None and dump_on():
            dump_put("logits_top5", torch.topk(logits.float(), 5, dim=-1).indices)
            dump_put("logits_top5_val", torch.topk(logits.float(), 5, dim=-1).values)
        if dump_on():
            dump_flush(self)
        return logits

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed_tokens(token_ids)

    def markov_bias(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(embeds)

    def confidence_logits(self, hidden_states: torch.Tensor, markov_embeds: torch.Tensor) -> torch.Tensor:
        if self.model.confidence_head is None or not self._confidence_loaded:
            raise RuntimeError("DSpark confidence head not loaded (mtp.2.confidence_head.*)")
        return self.model.confidence_head(hidden_states, markov_embeds)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def skip_checkpoint_weight(self, name: str) -> bool:
        return self._map_name(name) is None

    # ---- loading: mtp.N.* of the official checkpoint ----
    _HEAD_TENSORS = ("main_proj.", "main_norm.", "norm.", "markov_head.", "confidence_head.")

    def _map_name(self, ckpt: str) -> str | None:
        """mtp.N.X -> parameter name, or None to skip. Block tensors reuse the target's checkpoint-name mapping
        (``map_checkpoint_name``: .scale -> weight_scale(_inv), gate.bias -> e_score_correction_bias, shared w2 ->
        down_proj) on a synthetic backbone name ``layers.{40 + N}.X``; the drafter's attention module lives one level
        deeper (``attn.inner``)."""
        if ckpt == "embed.weight":                     # own copy: under PP the target embedding is on stage 1
            return "model.embed_tokens.weight"
        if not ckpt.startswith("mtp."):
            return None
        _, stage, rest = ckpt.split(".", 2)
        if rest.endswith(".bias_vl"):                  # drafts are text: no VL router bias
            return None
        if rest.startswith("confidence_head.") and self.model.confidence_head is None:
            return None                                # optional head, not instantiated (threshold 0)
        if rest.startswith(self._HEAD_TENSORS):
            name = "model." + rest
            if name.endswith(".scale"):               # main_proj FP8 scale
                name = name[: -len(".scale")] + ".weight_scale_inv"
            return name
        layer = int(self.config.num_hidden_layers) + int(stage)
        name = map_checkpoint_name(f"layers.{layer}.{rest}")
        # the drafter's ModuleList indexes its blocks 0..2 (the synthetic backbone id only steers the name mapping)
        name = name.replace(f"layers.{layer}.", f"layers.{stage}.", 1)
        return name.replace(".attn.", ".attn.inner.", 1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Loads the mtp.* tensors of the official checkpoint into the drafter (the target's embedding and lm_head
        are shared by the proposer). Mirrors DeepseekV41ForCausalLM.load_weights: stacked attention / shared-expert
        shards through their weight_loader(shard_id), routed experts through FusedMoE's expert mapping."""
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm.models.deepseek_v41.sm70.moe import DeepseekV41MoE
        from .model import _EXPERT_KEY_RE, _STACKED

        params = dict(self.named_parameters())
        n_experts = int(self.config.dspark_n_routed_experts)
        expert_mapping = {weight_name: (param_name, expert_id, shard_id) for param_name, weight_name, expert_id, shard_id
                          in DeepseekV41MoE.expert_params_mapping(n_experts)}
        loaded: set[str] = set()
        for ckpt_name, tensor in weights:
            name = self._map_name(ckpt_name)
            if name is None:
                continue
            if ".ffn.experts." in name:
                if name.endswith(".weight_scale") and tensor.dtype == torch.float8_e8m0fnu:
                    tensor = tensor.view(torch.uint8)
                match = _EXPERT_KEY_RE.search(name)
                param_name, expert_id, shard_id = expert_mapping[match.group(0)]
                mapped = name.replace(match.group(0), param_name)
                param = params[mapped]
                param.weight_loader(param, tensor, mapped, shard_id=shard_id, expert_id=expert_id)
                loaded.add(mapped)
                continue
            for param_frag, ckpt_frag, shard_id in _STACKED:
                frag = ckpt_frag.replace("attn.", "attn.inner.", 1) if ckpt_frag.startswith("attn.") else ckpt_frag
                pf = param_frag.replace("attn.", "attn.inner.", 1) if param_frag.startswith("attn.") else param_frag
                if frag in name:
                    stacked = name.replace(frag, pf)
                    if stacked in params:
                        param = params[stacked]
                        param.weight_loader(param, tensor, shard_id)
                        loaded.add(stacked)
                        break
            else:
                if name not in params:
                    raise KeyError(f"DSpark drafter: checkpoint tensor {ckpt_name!r} -> {name!r} has no parameter")
                param = params[name]
                loader = getattr(param, "weight_loader", None)
                if loader is None and name.endswith(".attn_sink"):
                    heads = C.N_HEADS // get_tensor_model_parallel_world_size()
                    from vllm.distributed import get_tensor_model_parallel_rank
                    tensor = tensor.narrow(0, heads * get_tensor_model_parallel_rank(), heads)
                (loader or default_weight_loader)(param, tensor)
                loaded.add(name)
                if "confidence_head" in name:
                    self._confidence_loaded = True
        missing = sorted(n for n in params if n not in loaded and not n.startswith("lm_head"))
        if missing:
            logger.warning("DSpark drafter: %d parameters not provided by mtp.* tensors (first: %s)", len(missing),
                           missing[:5])
        return loaded
