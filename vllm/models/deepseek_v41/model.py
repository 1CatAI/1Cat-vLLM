# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Flash FP16 text implementation for local SM70 TP4.

This is an independent forward. V4's compression ratios, HC head and immediate
pre-mix are not compatible. The initial implementation deliberately admits
one request and at most 16K total tokens, with prefix caching disabled.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.kernels.deepseek_v41_compression import V41C2State
from vllm.model_executor.kernels.mhc.deepseek_v41 import (
    collapse_hc,
    expand_hc,
    fp32_projection,
    hc_mixes,
)
from vllm.model_executor.kernels.mhc.deepseek_v41_fused import hc_transition
from vllm.model_executor.kernels.mhc.deepseek_v41_router import route_topk
from vllm.model_executor.layers.deepseek_v41_engram import (
    V41EngramHashState,
    apply_engram_gate,
)
from vllm.model_executor.layers.deepseek_v41_engram_io import SharedEngramIO
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.models.deepseek_v41 import dense_tm
from vllm.models.deepseek_v41.activations import swiglu
from vllm.platforms import current_platform

from .attention_ops import (
    WINDOW_RING,
    attend_paged_latents,
    gather_index_keys,
    insert_latents,
    insert_packed_rows,
    insert_window_ring,
    select_compressed_positions,
    select_packed_positions,
)
from .cache import V41LatentCache
from .contract import validate_runtime_geometry
from .norm import short_rms_norm
from .quant_rope import apply_rope, attention_qdq, rope_table

_HC_FUSED = envs.VLLM_V41_HC_FUSED == "1"
_ROUTER_FUSED = envs.VLLM_V41_ROUTER_FUSED == "1"


class V41Norm(nn.Module):
    def __init__(self, size, eps=1e-20):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(size, dtype=torch.float32), requires_grad=False
        )
        self.eps = eps

    def forward(self, x):
        if (
            x.is_cuda
            and x.dtype == torch.float16
            and x.ndim == 2
            and 1 <= x.shape[0] <= 6
            and x.shape[1] in (128, 512, 1280, 5120)
            and x.is_contiguous()
        ):
            return short_rms_norm(x, self.weight, self.eps)
        values = x.float()
        return (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
            * self.weight
        ).to(x.dtype)


class V41SharedMLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        h, intermediate = config.hidden_size, config.moe_intermediate_size
        kwargs = dict(bias=False, return_bias=False, quant_config=quant_config)
        self.gate_proj = ColumnParallelLinear(
            h, intermediate, prefix=prefix + ".gate_proj", **kwargs
        )
        self.up_proj = ColumnParallelLinear(
            h, intermediate, prefix=prefix + ".up_proj", **kwargs
        )
        self.down_proj = RowParallelLinear(
            intermediate,
            h,
            reduce_results=False,
            prefix=prefix + ".down_proj",
            **kwargs,
        )

    def forward(self, x):
        pair = dense_tm.fused(self.gate_proj, x)
        gate, up = pair if pair is not None else (self.gate_proj(x), self.up_proj(x))
        activated = swiglu(gate, up)
        return self.down_proj(activated)


class V41Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.temperature = getattr(config, "gate_temp", 1.0)

    def forward(self, x):
        logits = fp32_projection(x, self.weight)
        return logits if self.temperature == 1.0 else logits / self.temperature


class V41MoE(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        self.gate = V41Gate(c)
        self.shared_experts = V41SharedMLP(
            c, vllm_config.quant_config, prefix + ".shared_experts"
        )
        self.scale = c.routed_scaling_factor
        self.experts = FusedMoE(
            num_experts=c.n_routed_experts,
            top_k=c.num_experts_per_tok,
            hidden_size=c.hidden_size,
            intermediate_size=c.moe_intermediate_size,
            renormalize=True,
            quant_config=vllm_config.quant_config,
            prefix=prefix + ".experts",
            custom_routing_function=self._route,
            shared_experts=self.shared_experts,
            swiglu_limit=10.0,
            router_logits_dtype=torch.float32,
        )
        self.experts.expert_mapping = FusedMoE.make_expert_params_mapping(
            self, "gate_proj", "down_proj", "up_proj", c.n_routed_experts
        )

    def _route(self, hidden_states, gating_output, topk, renormalize):
        del hidden_states, renormalize
        if _ROUTER_FUSED and gating_output.is_cuda:
            routed = route_topk(gating_output.float(), self.gate.bias, topk, self.scale)
            if _SKIP_SUBST:
                stash_router(gating_output.float(), self.gate.bias, self.scale)
            if _PRED_STATS and getattr(self, "_v41_index", None) is not None:
                _score_prediction(self._v41_index, routed[1])
            return routed
        scores = F.softplus(gating_output).sqrt()
        indices = (scores + self.gate.bias).topk(topk, dim=-1).indices
        probabilities = scores.gather(1, indices)
        probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20)
        return (probabilities * self.scale).contiguous(), indices

    def forward(self, x):
        return self.experts(hidden_states=x, router_logits=self.gate(x))


class V41Engram(nn.Module):
    def __init__(self, config, quant_config, layer, prefix):
        super().__init__()
        self.channel = tuple(config.engram_layer_ids).index(layer)
        self.wkv = ColumnParallelLinear(
            6144,
            5 * config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".wkv",
        )
        self.q_weight = nn.Parameter(
            torch.empty(4, config.hidden_size, dtype=torch.float32), requires_grad=False
        )
        self.k_weight = nn.Parameter(
            torch.empty(4, config.hidden_size, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x, io, valid_tokens):
        embedding = io.runtime.lookup(self.channel, x.shape[0])
        projected = tensor_model_parallel_all_gather(self.wkv(embedding), dim=-1)
        return apply_engram_gate(
            x, projected, self.q_weight, self.k_weight, valid_tokens, inplace=True
        )


class V41Compressor(nn.Module):
    def __init__(self, c, quant_config, ratio, prefix):
        super().__init__()
        self.ratio = ratio
        dtype = torch.float32 if ratio == 2 else torch.float16
        kwargs = dict(
            bias=False, return_bias=False, quant_config=quant_config, params_dtype=dtype
        )
        self.wkv = ReplicatedLinear(
            c.hidden_size, 512, prefix=prefix + ".wkv", **kwargs
        )
        self.norm = V41Norm(512)
        self.cache = (
            None if _PACKED_KV else V41LatentCache(prefix + ".cache", ratio=ratio)
        )
        if ratio == 2:
            self.wgate = ReplicatedLinear(
                c.hidden_size, 512, prefix=prefix + ".wgate", **kwargs
            )
            self.state = V41C2State(self.norm.weight)

    def forward(self, x, positions, profile=False):
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        values, scores = self.wkv(x.float()), self.wgate(x.float())
        if profile:
            return self.norm(values.half())
        return self.state(values, scores, positions)[0]


class V41Indexer(nn.Module):
    def __init__(self, c, quant_config, source, ratio, prefix):
        super().__init__()
        kwargs = dict(bias=False, return_bias=False, quant_config=quant_config)
        self.wq_b = ColumnParallelLinear(
            c.q_lora_rank,
            c.index_n_heads * c.index_head_dim,
            prefix=prefix + ".wq_b",
            **kwargs,
        )
        self.weights_proj = ColumnParallelLinear(
            c.hidden_size, c.index_n_heads, prefix=prefix + ".weights_proj", **kwargs
        )
        self.heads = c.index_n_heads // 4
        if source:
            self.wk = ReplicatedLinear(512, 128, prefix=prefix + ".wk", **kwargs)
            self.k_norm = V41Norm(128)
            self.k_cache = (
                None
                if _PACKED_KV
                else V41LatentCache(prefix + ".k_cache", head_dim=128, ratio=ratio)
            )

    def prepare_keys(
        self, latent, positions, valid_tokens, table, ratio, store=None, layer=None
    ):
        key = self.k_norm(self.wk(latent))
        group_positions = positions - (ratio - 1)
        # Incomplete groups are never inserted; clamp their unused position
        # before the rotation so a first C2 token cannot index row -1.
        key = apply_rope(key, group_positions.clamp_min(0), table)
        if store is not None:
            # Packed FP4 rows reproduce the QDQ values exactly.
            insert_packed_rows(
                store.index[layer],
                key,
                positions,
                valid_tokens,
                ratio=ratio,
                compressed=False,
            )
            return
        key = attention_qdq(key, fp4=True, group=32)
        metadata = self.k_cache.metadata()  # type: ignore[union-attr]
        insert_latents(
            self.k_cache.kv_cache,  # type: ignore[union-attr]
            key,
            positions,
            valid_tokens,
            metadata.block_table,
            block_size=metadata.block_size,
            ratio=ratio,
        )

    def forward(
        self,
        x,
        qr,
        positions,
        valid_tokens,
        table,
        source_cache,
        ratio,
        store=None,
        source=None,
    ):
        query = self.wq_b(qr).view(x.shape[0], self.heads, 128)
        query = attention_qdq(apply_rope(query, positions, table), fp4=True, group=32)
        weights = self.weights_proj(x) * (1.0 / 64.0)
        if store is not None:
            return select_packed_positions(
                query,
                weights,
                store.index[source],
                positions,
                valid_tokens,
                ratio,
                tensor_model_parallel_all_reduce,
            )
        metadata = source_cache.metadata()
        keys = gather_index_keys(
            source_cache.kv_cache,
            metadata.block_table,
            positions,
            valid_tokens,
            block_size=metadata.block_size,
            ratio=ratio,
        )
        return select_compressed_positions(
            query,
            weights,
            keys,
            positions,
            valid_tokens,
            ratio,
            tensor_model_parallel_all_reduce,
        )


# ---- debug: cross-layer expert prediction accuracy (VLLM_V41_PRED_STATS=1) ----
_PRED_STATS = envs.VLLM_V41_PRED_STATS == "1"
_PREFETCH_HINTS = envs.VLLM_V41_PREFETCH == "1"
from vllm.model_executor.layers.quantization.gguf_shared_runtime import (  # noqa: E402
    publish_prefetch_hints,
    stash_router,
)

_SKIP_SUBST = (
    float(envs.VLLM_V41_SKIP_BELOW) > 0
    and envs.VLLM_V41_SKIP_DEFER == "1"
    and (envs.VLLM_V41_SKIP_SUBST == "1" or envs.VLLM_V41_SKIP_RENORM == "1")
)
_PRED = {}
_PRED_KS = (6, 8, 12, 16)
_PRED_COUNT = {}


def _pred_counters(device):
    if "hit" not in _PRED_COUNT:
        # [lookahead 1..2, k] hits / predicted; [lookahead] actual totals
        _PRED_COUNT["hit"] = torch.zeros(2, len(_PRED_KS), device=device)
        _PRED_COUNT["pred"] = torch.zeros(2, len(_PRED_KS), device=device)
        _PRED_COUNT["actual"] = torch.zeros(2, device=device)
    return _PRED_COUNT


def _predict_next(layers, i, inputs):
    for d in (1, 2):
        j = i + d
        if j < len(layers):
            moe = layers[j].ffn
            logits = moe.gate(inputs).float()
            score = torch.nn.functional.softplus(logits).sqrt() + moe.gate.bias
            _PRED[(j, d)] = score.topk(max(_PRED_KS), dim=-1).indices


def _score_prediction(index, ids):
    c = _pred_counters(ids.device)
    actual = torch.zeros(384, device=ids.device).scatter_(0, ids.flatten().long(), 1.0)
    for d in (1, 2):
        pred = _PRED.get((index, d))
        if pred is None or pred.shape[0] != ids.shape[0]:
            continue
        c["actual"][d - 1] += actual.sum()
        for n, k in enumerate(_PRED_KS):
            p = torch.zeros(384, device=ids.device).scatter_(
                0, pred[:, :k].flatten(), 1.0
            )
            c["hit"][d - 1, n] += (p * actual).sum()
            c["pred"][d - 1, n] += p.sum()


def prediction_report():
    if "hit" not in _PRED_COUNT:
        return None
    hit, pred, act = (_PRED_COUNT[k].cpu().tolist() for k in ("hit", "pred", "actual"))
    out = {}
    for d in (0, 1):
        for n, k in enumerate(_PRED_KS):
            out[f"ahead{d + 1}_top{k}"] = dict(
                recall=round(hit[d][n] / max(1.0, act[d]), 3),
                precision=round(hit[d][n] / max(1.0, pred[d][n]), 3),
                predicted_per_layer_over_actual=round(pred[d][n] / max(1.0, act[d]), 2),
            )
    return out


def _join_expert_writebacks():
    from vllm.model_executor.layers.quantization import gguf_shared_runtime

    gguf_shared_runtime.join_writebacks()


class _SharedAttention:
    def __init__(self):
        self.layers = {}
        self.topk = None
        # Padding-token mask of the current forward (same for every layer).
        self.valid = None
        # Layer 0's window metadata, shared by the window-ring layers.
        self.metadata = None
        # G4b packed KV store (None: paged FP16 caches).
        self.store = None


_ATT_T = {
    "on": __import__("os").environ.get("VLLM_V41_ATTN_TIMING", "0") == "1",
    "marks": [],
}


def _tmark(name, tokens):
    if _ATT_T["on"] and tokens > 256:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        _ATT_T["marks"].append((name, e))


def _treport(tokens):
    marks = _ATT_T["marks"]
    _ATT_T["marks"] = []
    if not marks or get_tensor_model_parallel_rank() != 0:
        return
    torch.accelerator.synchronize()
    total: Any = {}
    for (a, ea), (_, eb) in zip(marks, marks[1:]):
        total[a] = total.get(a, 0.0) + ea.elapsed_time(eb)
    text = " ".join(
        f"{k}={v:.0f}" for k, v in sorted(total.items(), key=lambda kv: -kv[1])
    )
    print(f"[attn timing] tokens {tokens} ms: {text}", flush=True)


# G4b: packed FP4 KV (host compressed latents + GPU hot ring, GPU index keys).
_PACKED_KV = (envs.VLLM_V41_PACKED_KV or "0") == "1"
_EMPTY_CACHE_TOKENS = int(envs.VLLM_V41_EMPTY_CACHE_TOKENS)
_WINDOW_RINGS = envs.VLLM_V41_WINDOW_RING == "1"


class V41Attention(nn.Module):
    ring_capable = True

    def __init__(self, c, quant_config, layer, shared, prefix):
        super().__init__()
        self.layer, self.shared = layer, shared
        self.ratio = c.compress_ratios[layer]
        self.heads, self.groups = c.num_attention_heads // 4, c.o_groups // 4
        self.rank = get_tensor_model_parallel_rank()
        kwargs = dict(bias=False, return_bias=False, quant_config=quant_config)
        self.wq_a = ReplicatedLinear(
            c.hidden_size, c.q_lora_rank, prefix=prefix + ".wq_a", **kwargs
        )
        self.q_norm = V41Norm(c.q_lora_rank)
        self.wq_b = ColumnParallelLinear(
            c.q_lora_rank,
            c.num_attention_heads * 512,
            prefix=prefix + ".wq_b",
            **kwargs,
        )
        self.wkv = ReplicatedLinear(
            c.hidden_size, 512, prefix=prefix + ".wkv", **kwargs
        )
        self.kv_norm = V41Norm(512)
        self.wo_a = ColumnParallelLinear(
            4096, c.o_groups * c.o_lora_rank, prefix=prefix + ".wo_a", **kwargs
        )
        self.wo_b = RowParallelLinear(
            c.o_groups * c.o_lora_rank, c.hidden_size, prefix=prefix + ".wo_b", **kwargs
        )
        self.attn_sink = nn.Parameter(
            torch.empty(self.heads, dtype=torch.float32), requires_grad=False
        )
        self.attn_sink.weight_loader = self._load_sink
        # Layer 0 keeps the paged window cache: it anchors the per-step
        # metadata. Other target layers keep their window in a private ring
        # instead of a full-length KV-manager allocation (G4).
        self.ring_window = _WINDOW_RINGS and self.ring_capable and layer > 0
        if self.ring_window:
            self.register_buffer(
                "window_ring",
                torch.zeros(
                    WINDOW_RING,
                    512,
                    dtype=torch.float16,
                    device=torch.device(
                        "cuda", torch.accelerator.current_device_index()
                    ),
                ),
                persistent=False,
            )
        else:
            self.window_cache = V41LatentCache(
                prefix + ".window_cache", sliding_window=c.sliding_window + 6
            )
        self.source = max((i for i in c.kv_source_layer_ids if i <= layer), default=-1)
        owns_kv = layer in c.kv_source_layer_ids
        if owns_kv:
            self.compressor = V41Compressor(
                c, quant_config, self.ratio, prefix + ".compressor"
            )
            shared.layers[layer] = self
        if layer in c.index_source_layer_ids:
            self.indexer = V41Indexer(
                c, quant_config, owns_kv, self.ratio, prefix + ".indexer"
            )

    def _load_sink(self, parameter, loaded):
        parameter.data.copy_(
            loaded[self.rank * self.heads : (self.rank + 1) * self.heads]
        )

    def forward(self, x, positions, table):
        tokens = x.shape[0]
        _tmark("a.proj", tokens)
        pair = dense_tm.fused(self.wq_a, x)
        q_low, kv_low = pair if pair is not None else (self.wq_a(x), self.wkv(x))
        qr = self.q_norm(q_low)
        query = apply_rope(
            self.wq_b(qr).view(tokens, self.heads, 512), positions, table
        )
        window = attention_qdq(apply_rope(self.kv_norm(kv_low), positions, table))
        metadata = (
            self.shared.metadata if self.ring_window else self.window_cache.metadata()
        )
        if metadata is None:
            # Profile computes the dense paths but has no allocated KV pages.
            if hasattr(self, "compressor"):
                self.compressor(x, positions, profile=True)
            output = torch.zeros_like(query)
        else:
            valid = self.shared.valid
            if valid is None or valid.shape[0] != tokens:
                valid = metadata.slot_mapping[:tokens] >= 0
            _tmark("a.window_insert", tokens)
            if self.ring_window:
                arguments = dict(
                    window_block=metadata.block_size,
                    window_ring=self.window_ring,
                    window_values=window,
                )
            else:
                insert_latents(
                    self.window_cache.kv_cache,
                    window,
                    positions,
                    valid,
                    metadata.block_table,
                    block_size=metadata.block_size,
                )
                arguments = dict(window_block=metadata.block_size)
            if self.ratio:
                source = self.shared.layers[self.source]
                store = self.shared.store
                if hasattr(self, "compressor") and store is not None:
                    _tmark("a.compressor", tokens)
                    latent = self.compressor(x, positions)
                    self.indexer.prepare_keys(
                        latent, positions, valid, table, self.ratio, store, self.layer
                    )
                    compact = apply_rope(
                        latent, (positions - self.ratio + 1).clamp_min(0), table
                    )
                    store.insert_compressed(self.layer, compact, positions, valid)
                    if tokens > 8:
                        store.stage(self.layer, positions)
                elif hasattr(self, "compressor"):
                    _tmark("a.compressor", tokens)
                    latent = self.compressor(x, positions)
                    self.indexer.prepare_keys(
                        latent, positions, valid, table, self.ratio
                    )
                    compact = apply_rope(
                        latent, (positions - self.ratio + 1).clamp_min(0), table
                    )
                    compact = attention_qdq(
                        compact, fp4=True, group=16, e4m3_scale=True
                    )
                    compressed_metadata = self.compressor.cache.metadata()  # type: ignore[union-attr]
                    insert_latents(
                        self.compressor.cache.kv_cache,  # type: ignore[union-attr]
                        compact,
                        positions,
                        valid,
                        compressed_metadata.block_table,
                        block_size=compressed_metadata.block_size,
                        ratio=self.ratio,
                    )
                if hasattr(self, "indexer"):
                    _tmark("a.indexer", tokens)
                    self.shared.topk = self.indexer(
                        x,
                        qr,
                        positions,
                        valid,
                        table,
                        source.indexer.k_cache,
                        self.ratio,
                        store,
                        self.source,
                    )
                    if store is not None and tokens <= 8:
                        store.fill_selected(self.source, self.shared.topk, positions)
                if store is not None:
                    arguments.update(
                        ratio=self.ratio,
                        topk=self.shared.topk,
                        packed=(store, self.source),
                    )
                else:
                    compressed_metadata = source.compressor.cache.metadata()
                    arguments.update(
                        compressed_cache=source.compressor.cache.kv_cache,
                        compressed_table=compressed_metadata.block_table,
                        compressed_block=compressed_metadata.block_size,
                        ratio=self.ratio,
                        topk=self.shared.topk,
                    )
            _tmark("a.attend", tokens)
            output = attend_paged_latents(
                query,
                positions,
                valid,
                None if self.ring_window else self.window_cache.kv_cache,
                metadata.block_table,
                self.attn_sink,
                **arguments,
            )
            if self.ring_window:
                insert_window_ring(self.window_ring, window, positions, valid)
        _tmark("a.out_proj", tokens)
        output = apply_rope(output, positions, table, inverse=True)
        return self.project_output(output)

    def project_output(self, output):
        tokens = output.shape[0]
        output = attention_qdq(output.reshape(tokens, self.groups, 4096))
        # Each O group needs only its own 1024 rows. Grouped kernels read
        # those original rows directly; larger prefill batches retain the
        # bounded GEMM path without another weight bank.
        projected = dense_tm.grouped_wo_a(self.wo_a, output)
        grouped = getattr(self.wo_a.quant_method, "apply_grouped", None)
        if projected is None and grouped is not None:
            projected = grouped(self.wo_a, output)
        if projected is None:
            projected = torch.cat(
                [
                    self.wo_a(output[:, group].contiguous())[
                        :, group * 1024 : (group + 1) * 1024
                    ]
                    for group in range(self.groups)
                ],
                dim=-1,
            )
        return self.wo_b(projected)


class V41Block(nn.Module):
    attention_cls = V41Attention
    moe_cls = V41MoE

    def __init__(self, vllm_config, layer, shared, prefix):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        self.attn = self.attention_cls(
            c, vllm_config.quant_config, layer, shared, prefix + ".attn"
        )
        self.ffn = self.moe_cls(vllm_config, prefix + ".ffn")
        self.attn_norm, self.ffn_norm = V41Norm(c.hidden_size), V41Norm(c.hidden_size)
        for branch in ("attn", "ffn"):
            for part, shape in (
                ("fn", (24, 4 * c.hidden_size)),
                ("base", (24,)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{branch}_{part}",
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )
        self.engram = (
            V41Engram(c, vllm_config.quant_config, layer, prefix + ".engram")
            if layer in c.engram_layer_ids
            else None
        )

    def forward(self, x, positions, previous_pre, table):
        pre, post, comb = hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        inputs = collapse_hc(x, previous_pre, self.attn_norm.weight)
        x = expand_hc(self.attn(inputs, positions, table), x, post, comb)
        next_pre, post, comb = hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        inputs = collapse_hc(x, pre, self.ffn_norm.weight)
        return expand_hc(self.ffn(inputs), x, post, comb), next_pre


class DeepseekV41Model(nn.Module, EagleModelMixin):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        c = self.config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.embed_tokens = VocabParallelEmbedding(
            c.vocab_size,
            c.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=prefix + ".embed_tokens",
        )
        shared = _SharedAttention()
        self.layers = nn.ModuleList(
            [
                V41Block(vllm_config, i, shared, prefix + f".layers.{i}")
                for i in range(c.num_hidden_layers)
            ]
        )
        if _PACKED_KV:
            from vllm.distributed import get_tp_group

            from .kv_store import V41PackedKV

            shared.store = V41PackedKV(
                {layer: c.compress_ratios[layer] for layer in c.kv_source_layer_ids},
                vllm_config.model_config.max_model_len,
                torch.device("cuda", torch.accelerator.current_device_index()),
                get_tp_group(),
            )
        self.norm = V41Norm(c.hidden_size)
        for name, shape, dtype in (
            ("token_map", (c.vocab_size,), torch.int32),
            ("primes", (2, 24), torch.int64),
            ("offsets", (2, 24), torch.int64),
            ("multipliers", (2, 4), torch.int64),
        ):
            self.register_parameter(
                "engram_" + name,
                nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False),
            )
        self.register_buffer("window_rope", rope_table(), persistent=False)
        self.register_buffer(
            "compressed_rope", rope_table(compressed=True), persistent=False
        )
        self.register_buffer(
            "pending_length", torch.zeros(1, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "pending_position", torch.zeros(1, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "pending_prefill", torch.ones(1, dtype=torch.bool), persistent=False
        )
        self.hash_state: V41EngramHashState | None = None
        self.engram_io: SharedEngramIO | None = None

    def prepare_postload(self):
        if self.hash_state is not None:
            return
        c = self.config
        maximum = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.hash_state = V41EngramHashState(
            self.engram_token_map,
            self.engram_primes,
            self.engram_offsets,
            self.engram_multipliers,
            pad_token_id=c.engram_pad_token_id,
            max_tokens=maximum,
            max_context=max(16384, self.vllm_config.model_config.max_model_len),
        )
        path = self.vllm_config.load_config.model_loader_extra_config["engram_path"]
        self.engram_io = SharedEngramIO(path, max_tokens=maximum)

    def forward(self, input_ids, positions, inputs_embeds=None):
        if (
            _EMPTY_CACHE_TOKENS
            and input_ids.numel() >= _EMPTY_CACHE_TOKENS
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Long prefill passes fragment the caching allocator; return the
            # previous pass's free segments so driver-free VRAM stays >= 1 GB.
            torch.accelerator.empty_cache()
        metadata = self.layers[0].attn.window_cache.metadata()
        self.layers[0].attn.shared.metadata = metadata
        valid = (
            metadata.slot_mapping[: input_ids.numel()] >= 0
            if metadata is not None
            else None
        )
        self.layers[0].attn.shared.valid = valid
        if metadata is not None:
            assert (
                self.hash_state is not None
                and self.engram_io is not None
                and self.engram_io.runtime is not None
            )
            hashes = self.hash_state(input_ids, positions, valid)
            self.engram_io.runtime.prefetch(hashes)
            self.pending_length.fill_(input_ids.numel())
            self.pending_position.copy_(positions[:1])
            self.pending_prefill.copy_(metadata.is_prefilling[:1])
        hidden = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        # FP16 cannot carry the full HC residual range of valid special
        # tokens. Keep only this stream in FP32; collapse+RMSNorm returns
        # FP16 at every dense/attention/expert input boundary.
        hidden = hidden.float()[:, None].expand(-1, 4, -1).contiguous()
        pre = torch.zeros(hidden.shape[0], 4, device=hidden.device, dtype=torch.float32)
        pre[:, 0] = 1
        auxiliary: Any = []
        if _HC_FUSED:
            result = self._forward_fused(
                hidden, positions, pre, metadata, valid, auxiliary
            )
            _join_expert_writebacks()
            return (result, auxiliary) if auxiliary else result
        for i, layer in enumerate(self.layers):
            if layer.engram is not None and metadata is not None:
                hidden = layer.engram(hidden, self.engram_io, valid)
            if i in self.aux_hidden_state_layers:
                auxiliary.append(hidden.mean(1).mul_(2.0**-6).half())
            table = self.compressed_rope if layer.attn.ratio else self.window_rope
            hidden, pre = layer(hidden, positions, pre, table)
        result = collapse_hc(hidden, pre, self.norm.weight)
        _join_expert_writebacks()
        return (result, auxiliary) if auxiliary else result

    def _forward_fused(self, hidden, positions, pre, metadata, valid, auxiliary):
        # Each sublayer boundary is one fused transition (in-place expand, the
        # next sublayer's mixes, delayed collapse). A boundary falls back to the
        # reference mixes/collapse only when Engram rewrites the residual.
        state = None
        layers = list(self.layers)
        for i, layer in enumerate(layers):
            _tmark("engram", hidden.shape[0])
            if layer.engram is not None and metadata is not None:
                hidden = layer.engram(hidden, self.engram_io, valid)
                state = None
            _tmark("hc_pre", hidden.shape[0])
            if i in self.aux_hidden_state_layers:
                auxiliary.append(hidden.mean(1).mul_(2.0**-6).half())
            if state is None:
                attn_mix = hc_mixes(
                    hidden, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
                )
                inputs = collapse_hc(hidden, pre, layer.attn_norm.weight)
            else:
                inputs, attn_mix = state
            table = self.compressed_rope if layer.attn.ratio else self.window_rope
            y = layer.attn(inputs, positions, table)
            _tmark("hc_mid", hidden.shape[0])
            inputs, ffn_mix = hc_transition(
                hidden,
                y,
                attn_mix[1],
                attn_mix[2],
                attn_mix[0],
                layer.ffn_norm.weight,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
            )
            if _PRED_STATS:
                layer.ffn._v41_index = i
                _predict_next(layers, i, inputs)
            if _PREFETCH_HINTS:
                gates = getattr(self, "_v41_gates", None)
                if gates is None:
                    gates = [(l.ffn.gate.weight, l.ffn.gate.bias) for l in layers]  # noqa: E741
                    self._v41_gates = gates
                publish_prefetch_hints(gates, i, inputs)
            _tmark("ffn(moe+shared)", hidden.shape[0])
            y = layer.ffn(inputs)
            _tmark("hc_post", hidden.shape[0])
            pre = ffn_mix[0]
            following = layers[i + 1] if i + 1 < len(layers) else None
            if following is not None and following.engram is None:
                inputs, mix = hc_transition(
                    hidden,
                    y,
                    ffn_mix[1],
                    ffn_mix[2],
                    pre,
                    following.attn_norm.weight,
                    following.hc_attn_fn,
                    following.hc_attn_scale,
                    following.hc_attn_base,
                )
                state = (inputs, mix)
            elif following is not None:
                hidden = expand_hc(y, hidden, ffn_mix[1], ffn_mix[2], inplace=True)
                state = None
            else:
                result, _ = hc_transition(
                    hidden, y, ffn_mix[1], ffn_mix[2], pre, self.norm.weight
                )
                _tmark("end", hidden.shape[0])
                if _ATT_T["on"]:
                    _treport(hidden.shape[0])
                return result
        raise AssertionError("V4.1 fused forward requires at least one layer")

    def commit(self, accepted_counts):
        assert self.hash_state is not None
        selected = torch.where(
            self.pending_prefill, self.pending_length, accepted_counts.to(torch.int64)
        )
        self.hash_state.commit(selected)
        for layer in self.layers:
            compressor = getattr(layer.attn, "compressor", None)
            if compressor is not None and compressor.ratio == 2:
                compressor.state.commit(selected)

    def restore_prefix(self, cached, prompt_ids):
        """Rewind per-sequence device state to a same-session prefix hit.

        The scheduler only admits hits that continue the last executed
        sequence at an even position inside the window rings' reach.
        """
        assert self.hash_state is not None and cached % 2 == 0
        device = self.hash_state.history.device
        tail = [prompt_ids[i] if i >= 0 else -1 for i in range(cached - 3, cached)]
        ids = torch.tensor([max(t, 0) for t in tail], dtype=torch.long, device=device)
        mapped = self.hash_state.token_map[ids].long()
        mapped = torch.where(
            torch.tensor([t >= 0 for t in tail], device=device), mapped, -1
        )
        self.hash_state.history[:3].copy_(mapped)
        self.hash_state.history[3] = cached
        self.hash_state.metadata[2] = 0
        for layer in self.layers:
            compressor = getattr(layer.attn, "compressor", None)
            if compressor is not None and compressor.ratio == 2:
                # Even positions start a new pair: the partial row is unused.
                state = compressor.state
                state.partial.zero_()
                state.position.fill_(cached)
                state.metadata[2] = 0

    def output_budget(self, max_tokens, prompt_tokens):
        # CPU request lists can contain optimistic placeholders under async
        # scheduling. Corrected input positions are authoritative on device.
        before = torch.where(
            self.pending_prefill,
            self.pending_position + self.pending_length,
            self.pending_position + 1,
        )
        before = (before - prompt_tokens).clamp_min(0)
        return (max_tokens - before).clamp_min(0)


class DeepseekV41ForCausalLM(nn.Module, SupportsEagle3):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        c = self.config = vllm_config.model_config.hf_text_config
        p, s = vllm_config.parallel_config, vllm_config.scheduler_config
        if (
            not current_platform.is_cuda()
            or not current_platform.is_device_capability(70)
            or vllm_config.model_config.dtype != torch.float16
            or get_tensor_model_parallel_world_size() != 4
            or p.pipeline_parallel_size != 1
            or p.data_parallel_size != 1
            or p.enable_expert_parallel
            or s.max_num_seqs != 1
            or (s.async_scheduling and envs.VLLM_V41_ASYNC != "1")
            or s.max_num_batched_tokens > 16384
            or vllm_config.model_config.max_model_len
            > (524288 if _PACKED_KV else 16384)
            or (
                vllm_config.cache_config.enable_prefix_caching
                and envs.VLLM_V41_SESSION_PREFIX != "1"
            )
        ):
            raise ValueError(
                "V4.1 first version requires SM70 FP16 TP4, one request, "
                "<=16K, chunks <=16384, synchronous scheduling, no prefix caching"
            )
        if (
            c.model_type != "deepseek_v41_text"
            or (
                c.hidden_size,
                c.num_hidden_layers,
                c.n_routed_experts,
                c.moe_intermediate_size,
                c.num_experts_per_tok,
            )
            != (5120, 40, 384, 2304, 6)
            or tuple(c.kv_source_layer_ids) != (2, 8, 14, 20)
            or tuple(c.index_source_layer_ids) != (2, 8, 14, 20, 24, 28, 32, 36)
            or c.hc_mult != 4
            or c.swiglu_limit != 10
            or c.index_topk != 512
            or tuple(c.engram_layer_ids) != (1, 14)
            or not (vllm_config.load_config.model_loader_extra_config or {}).get(
                "engram_path"
            )
            or vllm_config.quant_config.get_name() != "gguf"
        ):
            raise ValueError(
                "V4.1 GGUF text architecture or explicit Engram sidecar is missing"
            )
        validate_runtime_geometry(c)
        policy = vllm_config.kernel_config.sm70_gguf
        if policy.dense_storage != "original" or policy.prefill_min_m != 1:  # noqa: SIM102
            raise ValueError(
                "V4.1 requires original dense storage and prefill_min_m=1 "
                "for bounded FP16 dense GEMMs; MMVQ is outside this profile"
            )
        self.model = DeepseekV41Model(vllm_config, maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            c.vocab_size,
            c.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(c.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if intermediate_tensors is not None:
            raise ValueError("V4.1 pipeline parallelism is not admitted")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    def prepare_loaded_linear_weights(self):
        if envs.VLLM_V41_GATE_BF16 == "1":
            for module in self.modules():
                if isinstance(module, V41Gate):
                    narrow = module.weight.data.to(torch.bfloat16)
                    # Only when exact: routes stay those of the FP32 gate.
                    if torch.equal(narrow.float(), module.weight.data):
                        module.weight.data = narrow
        dense_tm.prepare(self.model)
        dense_tm.attach_lm_head(self.lm_head)
        self.model.prepare_postload()

    def commit_speculative_state(self, accepted_counts):
        self.model.commit(accepted_counts)

    def restore_prefix(self, cached, prompt_ids):
        self.model.restore_prefix(cached, prompt_ids)

    def stateful_output_budget(self, max_tokens, prompt_tokens):
        return self.model.output_budget(max_tokens, prompt_tokens)

    def get_eagle3_default_aux_hidden_state_layers(self):
        return tuple(self.config.dspark_target_layer_ids)
