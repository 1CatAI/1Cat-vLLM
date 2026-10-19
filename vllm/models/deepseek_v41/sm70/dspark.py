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

Proposer contract (vllm/v1/spec_decode/{dspark,dflash,llm_base_proposer}.py): embed_input_ids, combine_hidden_states,
precompute_and_store_context_kv, forward(input_ids, positions, inputs_embeds) -> Tensor, compute_logits, markov_embed,
markov_bias, confidence_logits, map_draft_to_target, get_draft_kv_cache_layer_names, skip_checkpoint_weight;
has_own_embed_tokens = has_own_lm_head = False.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

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
        self.proj = nn.Linear(in_dim, 1, bias=True, dtype=torch.float32)

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
        q_dummy = torch.empty((main_x.shape[0], self.n_local_heads, C.HEAD_DIM), dtype=torch.float16,
                              device=main_x.device)
        q_rope_kv_insert(q_dummy, kv, positions, self.inner.rotary_emb.cos_sin_cache, self.swa_cache.rows(),
                         slot_mapping)

    def forward(self, positions: torch.Tensor, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """x [B * block_size, 5120] fp16 (draft stream, block-major); positions [B * block_size] int64."""
        from vllm.models.deepseek_v41.common.qat import fp8_block32_qdq
        from vllm.models.deepseek_v41.common.rope import apply_rope_torch

        T = x.shape[0]
        B = T // block_size
        md_all = step_metadata(self.inner.layer_name)
        q, kv = self._qkv(x)
        cos_sin = self.inner.rotary_emb.cos_sin_cache
        o = torch.empty((T, self.n_local_heads, C.HEAD_DIM), dtype=torch.float16, device=x.device)
        if md_all is None:                                           # profile / dummy run
            o.zero_()
            return self.inner.output_projection(o, positions)
        swa_md = cast(DS41SWAMetadata, md_all[self.swa_cache.layer_name])
        # block K/V: rotated at the draft positions, window-KV QAT (FP8 block-32), kept in a [T, 512] fp16 buffer
        block_kv = fp8_block32_qdq(apply_rope_torch(kv.float(), positions, cos_sin), out_dtype=torch.float16)
        q = apply_rope_torch(q, positions, cos_sin).to(torch.float16).contiguous()
        # window: for every query of a block, the cache positions p - 127 .. p where p = (first draft position) - 1
        first = positions.view(B, block_size)[:, 0]                  # [B]
        p = first - 1
        win_pos = p[:, None] - (C.WINDOW - 1) + torch.arange(C.WINDOW, device=x.device)[None, :]      # [B, 128]
        tok2req = swa_md.token_to_req_indices.view(B, block_size)[:, 0]
        win_rows = rows_of_positions(swa_md.block_table, tok2req, win_pos, swa_md.block_size)            # [B, 128]
        win_rows = win_rows[:, None, :].expand(B, block_size, C.WINDOW).reshape(T, C.WINDOW)
        # block rows: query i of block b sees block rows b * block_size .. + block_size - 1 (non-causal)
        brow = (torch.arange(B, device=x.device)[:, None] * block_size
                + torch.arange(BLOCK_KEYS_PAD, device=x.device)[None, :])                            # [B, 16]
        brow = torch.where(torch.arange(BLOCK_KEYS_PAD, device=x.device)[None, :] < block_size, brow,
                           torch.full_like(brow, -1))
        brow = brow[:, None, :].expand(B, block_size, BLOCK_KEYS_PAD).reshape(T, BLOCK_KEYS_PAD)
        fused_decode_attention(q, [(self.swa_cache.rows(), win_rows), (block_kv, brow)], self.inner.attn_sink,
                               self.inner.scale, o)
        return self.inner.output_projection(o, positions)


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
                                  top_k=int(hf.dspark_n_activated_experts), spill=None)
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
        attn_pre, attn_post, attn_comb = hc_mixes(stream, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = rmsnorm_to_act(hc_pre(stream, pre_in), self.attn_norm.weight)
        stream = hc_post(self.attn(positions, x, block_size), stream, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = hc_mixes(stream, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = rmsnorm_to_act(hc_pre(stream, attn_pre), self.ffn_norm.weight)
        stream = hc_post(self.ffn(x), stream, ffn_post, ffn_comb)
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
        return rmsnorm_to_act(projected, self.main_norm.weight)

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
        stream, pre = hc_expand(emb.to(torch.float16))
        for layer in self.layers:
            stream, pre = layer(stream, pre, positions, self.block_size)
        return hc_pre(stream, pre).to(torch.float16)


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
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mappings)

    def forward(self, input_ids, positions, inputs_embeds=None) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, rmsnorm_to_act(hidden_states, self.model.norm.weight))

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
        return not name.startswith("mtp.")

    # ---- loading: mtp.N.* of the official checkpoint ----
    _HEAD_TENSORS = ("main_proj.", "main_norm.", "norm.", "markov_head.", "confidence_head.")

    def _map_name(self, ckpt: str) -> str | None:
        """mtp.N.X -> parameter name, or None to skip. Block tensors reuse the target's checkpoint-name mapping
        (``map_checkpoint_name``: .scale -> weight_scale(_inv), gate.bias -> e_score_correction_bias, shared w2 ->
        down_proj) on a synthetic backbone name ``layers.{40 + N}.X``; the drafter's attention module lives one level
        deeper (``attn.inner``)."""
        if not ckpt.startswith("mtp."):
            return None
        _, stage, rest = ckpt.split(".", 2)
        if rest.endswith(".bias_vl"):                  # drafts are text: no VL router bias
            return None
        if rest.startswith(self._HEAD_TENSORS):
            name = "model." + rest
            if name.endswith(".scale"):               # main_proj FP8 scale
                name = name[: -len(".scale")] + ".weight_scale_inv"
            return name
        layer = int(self.config.num_hidden_layers) + int(stage)
        name = map_checkpoint_name(f"layers.{layer}.{rest}")
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
        missing = sorted(n for n in params if n not in loaded and not n.startswith(("model.embed_tokens", "lm_head")))
        if missing:
            logger.warning("DSpark drafter: %d parameters not provided by mtp.* tensors (first: %s)", len(missing),
                           missing[:5])
        return loaded
