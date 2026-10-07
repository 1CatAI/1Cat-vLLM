# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 attention for SM70 (PORT_DESIGN §3.3; owner L-ATTN). Reference: ref:m.py:613-789.

Per TP rank (TP4: 16 local heads, 2 local o-groups); ``x`` = attn_norm(hc_pre(...)) [T, 5120] fp16:

1. ``qr_kv = [wq_a | wkv] x`` (replicated, FP32 out); ``qr = q_norm(.)`` fp16, ``kv = kv_norm(.)`` FP32.
2. ``q = wq_b(qr)`` [T, 16, 512] fp16, RoPE on dims 448:512 -- **no per-head q-norm**.
3. window KV: RoPE(kv) -> ``fp8_block32_qdq`` over all 512 dims -> FP16 row of ``{P}.swa_cache``.
4. kv sources (2, 8, 14, 20): compressor -> pre-RoPE latent -> ``indexer.write_keys`` (before the cache
   write, ref:m.py:516-548) -> RoPE at the group's first position -> ``fp4_e4m3_qdq`` -> FP16 row of the
   compressed cache ``{P}`` (+ ``shared.export_ckv`` on source 20 of an exporting stage).
5. index sources: ``indexer.forward`` -> ``shared.topk_indices[:T]``; Reuse layers read what their index
   source published earlier in this forward.
6. sparse attention over <= 128 window rows + <= 512 compressed rows of the kv source (logical -> row via
   the SOURCE's block table), ``attn_sink``, running-max floor -1e30, scale 512^-1/2.
7. inverse RoPE on dims 448:512 (1Cat V4 SM70 kernel), grouped ``wo_a`` (FP16 operands, FP32 accumulate),
   ``wo_b`` row-parallel with FP32 output and an FP32 TP all-reduce -> [T, 5120] FP32.

Steps 1-6 run inside the eager-break op ``deepseek_v41_attention`` (host-dependent control flow, cache
writes); step 7 is graph-capturable. Source discovery is by name in the static forward context
(``model.layers.{kv_source}.attn`` / ``….attn.indexer.k_cache``); on a mirrored stage the mirror owns those
names, so consumer code is identical on every stage.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.contracts import (
    CAND_SOURCE,
    HEAD_DIM,
    HIDDEN,
    N_HEADS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    LayerTopology,
    SharedAttnBuffers,
    StagePlan,
)
from vllm.models.deepseek_v41.common.rope import apply_rope_torch, build_v41_rope
from vllm.models.deepseek_v41.compressor import (
    DeepseekV41Compressor,
    FP32RMSNorm,
    mm_fp32,
)
from vllm.models.deepseek_v41.indexer import DeepseekV41Indexer, attn_impl
from vllm.models.deepseek_v41.sm70.q_rope_kv_insert import q_rope_kv_insert
from vllm.models.deepseek_v41.sm70.sparse import (
    DeepseekV41SM70SparseImpl,
    DS41CacheLayer,
    DS41CompressedBackend,
    DS41CompressedMetadata,
    DS41SWABackend,
    DS41SWAMetadata,
    compressed_cache_spec,
    step_metadata,
    swa_cache_spec,
)
from vllm.models.deepseek_v41.sm70.sparse_kernels import ckv_rope_qat_store

ROPE_MARGIN = 1024          # positions past max_model_len that speculative tokens may reach


def _fp16_weight(lin: nn.Module, shape: tuple[int, int], what: str) -> torch.Tensor:
    w = lin.weight
    if w.dtype != torch.float16 or tuple(w.shape) != shape:
        raise RuntimeError(f"{what}: expected an FP16 {shape} weight (dense FP8 is dequantised at load, "
                           f"PORT_DESIGN A2), got {w.dtype} {tuple(w.shape)}")
    return w


class DeepseekV41Attention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology, stage: StagePlan,
                 shared: SharedAttnBuffers, aux_streams: list[torch.cuda.Stream] | None = None) -> None:
        super().__init__()
        hf = vllm_config.model_config.hf_config
        tp = get_tensor_model_parallel_world_size()
        if N_HEADS % tp or O_GROUPS % tp:
            raise ValueError(f"TP {tp} must divide {N_HEADS} heads and {O_GROUPS} o-groups")
        self.prefix = prefix
        self.topo = topo
        self.stage = stage
        self.shared = shared
        self.layer_id = topo.layer_id
        self.tp_size = tp
        self.n_local_heads = N_HEADS // tp
        self.n_local_groups = O_GROUPS // tp
        self.scale = HEAD_DIM ** -0.5
        self.eps = float(getattr(hf, "rms_norm_eps", 1e-20))
        block_size = vllm_config.cache_config.block_size
        quant_config = vllm_config.quant_config

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            HIDDEN, [Q_LORA, HEAD_DIM], bias=False, quant_config=quant_config, disable_tp=True,
            return_bias=False, prefix=f"{prefix}.fused_wqa_wkv")
        self.q_norm = FP32RMSNorm(Q_LORA, self.eps)
        self.kv_norm = FP32RMSNorm(HEAD_DIM, self.eps)
        self.wq_b = ColumnParallelLinear(Q_LORA, N_HEADS * HEAD_DIM, bias=False, quant_config=quant_config,
                                         return_bias=False, prefix=f"{prefix}.wq_b")
        self.wo_a = ColumnParallelLinear(N_HEADS * HEAD_DIM // O_GROUPS, O_GROUPS * O_LORA, bias=False,
                                         quant_config=quant_config, return_bias=False, prefix=f"{prefix}.wo_a")
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(O_GROUPS * O_LORA, HIDDEN, bias=False, quant_config=quant_config,
                                      reduce_results=False, return_bias=False, prefix=f"{prefix}.wo_b")
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(self.attn_sink, {"weight_loader": self._load_attn_sink})

        max_pos = int(vllm_config.model_config.max_model_len) + ROPE_MARGIN
        self.__dict__["rotary_emb"] = build_v41_rope(hf, topo.compress_ratio, max_positions=max_pos)

        cfg = vllm_config
        self.swa_cache = DS41CacheLayer(f"{prefix}.swa_cache", swa_cache_spec(block_size), DS41SWABackend, cfg)
        self.compressor: DeepseekV41Compressor | None = None
        self.kv_cache_layer: DS41CacheLayer | None = None
        if topo.owns_compressor:
            self.compressor = DeepseekV41Compressor(cfg, f"{prefix}.compressor", topo.compress_ratio)
            self.kv_cache_layer = DS41CacheLayer(prefix, compressed_cache_spec(block_size, topo.compress_ratio),
                                                 DS41CompressedBackend, cfg)
        self.indexer: DeepseekV41Indexer | None = None
        if topo.owns_indexer:
            self.indexer = DeepseekV41Indexer(cfg, f"{prefix}.indexer", topo, shared)
            self.indexer.__dict__["rotary_emb"] = self.rotary_emb
        if topo.compress_ratio > 0:
            base = prefix[: prefix.rindex(f".{topo.layer_id}.")]
            self.kv_source_name = f"{base}.{topo.kv_source}.attn"
        else:
            self.kv_source_name = None

        self.layer_name = f"{prefix}.ds41_attention"
        ctx = cfg.compilation_config.static_forward_context
        if self.layer_name in ctx:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        ctx[self.layer_name] = self
        self._static_ctx = ctx

    # ------------------------------------------------------------------------------- loading
    def _load_attn_sink(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
        if loaded.shape != (N_HEADS,):
            raise ValueError(f"{self.prefix}.attn_sink: checkpoint shape {tuple(loaded.shape)} != ({N_HEADS},)")
        r = get_tensor_model_parallel_rank()
        param.data.copy_(loaded[r * self.n_local_heads:(r + 1) * self.n_local_heads].float())

    # ------------------------------------------------------------------------------- forward
    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        o = torch.empty((T, self.n_local_heads, HEAD_DIM), dtype=torch.float16, device=x.device)
        deepseek_v41_attention(x, positions, o, self.layer_name)
        return self.output_projection(o, positions)

    def output_projection(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """inverse RoPE + grouped wo_a + wo_b (FP32 out) + FP32 TP all-reduce (ref:m.py:781-788)."""
        T = o.shape[0]
        cos_sin = self.rotary_emb.cos_sin_cache
        if attn_impl() == "torch":
            o = apply_rope_torch(o, positions[:T], cos_sin, inverse=True)
        else:
            from vllm.models.deepseek_v4.sm70.projection import sm70_inverse_rope
            o = sm70_inverse_rope(o.contiguous(), positions[:T], cos_sin, ROPE_DIM)
        g, d_in = self.n_local_groups, N_HEADS * HEAD_DIM // O_GROUPS
        wo_a = _fp16_weight(self.wo_a, (g * O_LORA, d_in), f"{self.prefix}.wo_a").view(g, O_LORA, d_in)
        og = o.reshape(T, g, d_in).transpose(0, 1)                                   # [g, T, 4096]
        z = torch.bmm(og, wo_a.transpose(1, 2), out_dtype=torch.float32)            # [g, T, 1024]
        z = z.transpose(0, 1).reshape(T, g * O_LORA).to(torch.float16)
        wo_b = _fp16_weight(self.wo_b, (HIDDEN, g * O_LORA), f"{self.prefix}.wo_b")
        out = mm_fp32(z, wo_b)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def attention_impl(self, x: torch.Tensor, positions: torch.Tensor, out: torch.Tensor) -> None:
        md_all = step_metadata(self.layer_name)
        w_qkv = _fp16_weight(self.fused_wqa_wkv, (Q_LORA + HEAD_DIM, HIDDEN), f"{self.prefix}.fused_wqa_wkv")
        w_qb = _fp16_weight(self.wq_b, (self.n_local_heads * HEAD_DIM, Q_LORA), f"{self.prefix}.wq_b")
        if md_all is None:              # dummy/profile forward without metadata (ForwardContext.is_dummy_run)
            self._profile_run(x, positions, out, w_qkv, w_qb)
            return
        swa_md = cast(DS41SWAMetadata, md_all[self.swa_cache.layer_name])
        T = swa_md.num_actual_tokens
        impl = attn_impl()
        pos = positions[:T]
        qr_kv = mm_fp32(x[:T], w_qkv)
        qr32 = self.q_norm(qr_kv[:, :Q_LORA])           # FP32; the indexer's q GEMM consumes it un-rounded
        qr = qr32.to(torch.float16)
        kv = self.kv_norm(qr_kv[:, Q_LORA:])
        q = torch.mm(qr, w_qb.t()).view(T, self.n_local_heads, HEAD_DIM)
        q_rope_kv_insert(q, kv, pos, self.rotary_emb.cos_sin_cache, self.swa_cache.rows(),
                         swa_md.slot_mapping, impl=impl)

        ckv_rows = ckv_idx = None
        if self.topo.compress_ratio > 0:
            if self.compressor is not None:
                latent, latent_pos = self.compressor(x[:T], pos)
                assert self.indexer is not None and self.kv_cache_layer is not None
                self.indexer.write_keys(latent, latent_pos)
                src_md = cast(DS41CompressedMetadata, md_all[self.kv_cache_layer.layer_name])
                export = None
                if self.layer_id == CAND_SOURCE and self.shared.export_ckv is not None:
                    export = self.shared.export_ckv
                ckv_rope_qat_store(latent, latent_pos, self.rotary_emb.cos_sin_cache, self.kv_cache_layer.rows(),
                                   src_md.latent_slots, export, impl=impl)
            if self.indexer is not None:
                self.indexer.forward(x[:T], qr32, pos)
            src_md = cast(DS41CompressedMetadata, md_all[self.kv_source_name])
            src = self._static_ctx.get(self.kv_source_name)
            if src is None:
                raise RuntimeError(f"kv source {self.kv_source_name} is not registered on this stage")
            ckv_rows = cast(DS41CacheLayer, src).rows()
            ckv_idx = DeepseekV41SM70SparseImpl.compressed_rows(self.shared.topk_indices[:T], src_md)
        DeepseekV41SM70SparseImpl.forward(q, swa_md, self.swa_cache.rows(), ckv_rows, ckv_idx, self.attn_sink,
                                          self.scale, out[:T], impl=impl)
        if out.shape[0] > T:
            out[T:].zero_()

    def _profile_run(self, x: torch.Tensor, positions: torch.Tensor, out: torch.Tensor,
                     w_qkv: torch.Tensor, w_qb: torch.Tensor) -> None:
        """Dummy/profile forward: run the projections at full size and reserve the attention workspace."""
        qr_kv = mm_fp32(x, w_qkv)
        qr = self.q_norm(qr_kv[:, :Q_LORA]).to(torch.float16)
        torch.mm(qr, w_qb.t())
        if self.compressor is not None:
            self.compressor(x, positions)
        if self.indexer is not None:
            self.indexer.forward(x, qr, positions)
        DeepseekV41SM70SparseImpl.reserve_workspace(out, self.topo.compress_ratio > 0)
        out.zero_()


@eager_break_during_capture
def deepseek_v41_attention(x: torch.Tensor, positions: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.attention_impl(x, positions, out)

