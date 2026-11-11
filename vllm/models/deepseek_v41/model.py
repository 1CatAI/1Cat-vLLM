# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Flash FP16 text implementation for local SM70 TP4.

This is an independent forward. V4's compression ratios, HC head and immediate
pre-mix are not compatible. The initial implementation deliberately admits
one request and at most 16K total tokens, with prefix caching disabled.
"""

from collections.abc import Callable, Iterable
from copy import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import vllm.envs as envs
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.deepseek_v41_compression import V41C2State
from vllm.model_executor.kernels.mhc.deepseek_v41 import (
    collapse_hc,
    expand_hc,
    fp32_projection,
    hc_mixes,
)
from vllm.model_executor.kernels.mhc.deepseek_v41_fused import (
    hc_join,
    hc_kick,
    hc_transition,
)
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
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    maybe_prefix,
)
from vllm.models.deepseek_v41 import dense_tm
from vllm.models.deepseek_v41.activations import swiglu
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .attention_ops import (
    WINDOW_RING,
    attend_paged_latents,
    attend_paged_latents_batch,
    attend_ring_post,
    gather_index_keys,
    insert_latents,
    insert_latents_batch,
    insert_packed_rows,
    insert_window_ring,
    select_compressed_positions,
    select_compressed_positions_batch,
    select_packed_positions,
)
from .attn_fuse import FUSE_ATTN_POST, FUSE_ATTN_PREP, attn_post, attn_prep
from .cache import BATCH as _BATCH
from .cache import V41LatentCache
from .contract import kv_source_of, pp_split_problem, validate_runtime_geometry
from .native import _LL_AR, V41TargetQuantConfig, load_native_weights
from .native import NATIVE as _NATIVE
from .norm import set_decode_rows, short_rms_norm, short_rows
from .pp_commit import StageCommit
from .quant_rope import apply_rope, attention_qdq, rope_table

logger = init_logger(__name__)

_HC_FUSED = envs.VLLM_V41_HC_FUSED == "1"
_ROUTER_FUSED = envs.VLLM_V41_ROUTER_FUSED == "1"
# Native decode: the shared expert runs on a side stream, overlapped with the
# gate, routing and routed experts (both are short, latency-bound kernels).
_SHARED_STREAM = envs.VLLM_V41_SHARED_STREAM == "1"
_BATCH_DEBUG = envs.VLLM_V41_BATCH_DEBUG == "1"
# VLLM_V41_BATCH_ATTN_FUSED: batched decode attention + post in v41sk.attn_paged_dec.
_BATCH_ATTN_FUSED = envs.VLLM_V41_BATCH_ATTN_FUSED == "1"
_SIDE: dict[str, Any] = {}


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
            and short_rows(x.shape[0])
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
            # The overlapped native path runs the shared expert itself.
            shared_experts=(
                None if self._own_shared(vllm_config) else self.shared_experts
            ),
            swiglu_limit=10.0,
            router_logits_dtype=torch.float32,
        )
        self.experts.expert_mapping = FusedMoE.make_expert_params_mapping(
            self, "gate_proj", "down_proj", "up_proj", c.n_routed_experts
        )
        self.native = bool(getattr(vllm_config.quant_config, "v41_native", False))

    @staticmethod
    def _own_shared(vllm_config):
        return _SHARED_STREAM and bool(
            getattr(vllm_config.quant_config, "v41_native", False)
        )

    def _overlapped_partials(self, x):
        """Shared expert on a side stream; gate + routed experts on this one."""
        main = torch.cuda.current_stream()
        side = _SIDE.get(x.device)
        if side is None:
            side = _SIDE[x.device] = torch.cuda.Stream(device=x.device)
        # Route first on this stream: the latency-bound gate + top-k should not
        # share DRAM with the shared expert; the shared expert overlaps the
        # routed experts (both bandwidth-bound) instead.
        logits = self._gate_logits(x)
        hc_kick()  # pending mixes overlap the routed experts, not the gate
        side.wait_stream(main)
        event = getattr(self, "_shared_done", None)
        if event is None:
            event = self._shared_done = torch.cuda.Event()
        with torch.cuda.stream(side):
            shared = self.shared_experts(x)
            event.record(side)
        # The native route sum waits for the event and adds the shared output.
        self.experts.v41_shared_join = (shared, event)
        self.experts.v41_shared_folded = False
        routed, _ = self.experts.forward_partials(hidden_states=x, router_logits=logits)
        self.experts.v41_shared_join = None
        main.wait_stream(side)
        return routed, None if self.experts.v41_shared_folded else shared

    def _gate_logits(self, x):
        """Router logits; with an h16 gate also the routes and slot plan,
        stashed for _route (one launch instead of gate + route_plan)."""
        if self.native and x.is_cuda:
            from .native import gate_route

            fused = gate_route(self.gate, x, self.experts.top_k, self.scale)
            if fused is not None:
                self._fused_route = fused
                return fused[0]
            plain = getattr(self.gate, "v41_h16_plain", None)
            if (
                plain is not None
                and x.dtype == torch.float16
                and self.gate.temperature == 1.0
                and x.shape[0] > 8
            ):
                # Exact FP16 * 2^s products, FP32 accumulation (as h16_gemv).
                logits = torch.mm(x.contiguous(), plain, out_dtype=torch.float32)
                return logits.mul_(self.gate.v41_h16[2])
        return self.gate(x)

    def _route(self, hidden_states, gating_output, topk, renormalize):
        del hidden_states, renormalize
        fused = getattr(self, "_fused_route", None)
        if fused is not None and fused[0] is gating_output:
            self._fused_route = None
            self.experts.v41_plan = (fused[2], fused[3])
            logger.info_once("V4.1 native: gate + top-k + slot plan in one launch")
            return fused[1], fused[2]
        if self.native and _ROUTER_FUSED and gating_output.is_cuda:
            from .native import route_and_plan

            fused = route_and_plan(
                gating_output.float(), self.gate.bias, topk, self.scale
            )
            if fused is not None:
                # The expert method reuses this slot plan (same ids tensor).
                self.experts.v41_plan = (fused[1], fused[2])
                return fused[0], fused[1]
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
        if self.native:
            # Native MXFP4 experts return FP32 routed sums: combine the shared
            # expert and reduce across TP in FP32 (the drafter's contract).
            if self.experts.shared_experts is None and x.is_cuda:
                routed, shared = self._overlapped_partials(x)
            else:
                routed, shared = self.experts.forward_partials(
                    hidden_states=x, router_logits=self._gate_logits(x)
                )
            output = routed.float()
            if shared is not None:
                output = output + shared.float()
            if _LL_AR:
                from .native import tp_all_reduce

                return tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return self.experts(hidden_states=x, router_logits=self.gate(x))


def _index_all_reduce():
    if _LL_AR:
        from .native import tp_all_reduce

        return tp_all_reduce
    return tensor_model_parallel_all_reduce


def _index_all_gather():
    if _LL_AR:
        from .native import tp_all_gather

        return tp_all_gather
    return tensor_model_parallel_all_gather


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
        projected = self._h16_projection(x)
        if projected is not None:
            values, scores = projected
        else:
            values, scores = self.wkv(x.float()), self.wgate(x.float())
        if profile:
            return self.norm(values.half())
        return self.state(values, scores, positions)[0]

    def forward_batch(self, x, positions, valid, layout, slots):
        """VLLM_V41_BATCH: per-request C2 state slots (ratio 1 has no state)."""
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        projected = self._h16_projection(x)
        if projected is not None:
            values, scores = projected
        else:
            values, scores = self.wkv(x.float()), self.wgate(x.float())
        return self.state_batch(values, scores, positions, valid, layout, slots)[0]

    def _h16_projection(self, x):
        """Decode: wkv and wgate (FP32 parameters) as one exact FP16*2^s HMMA
        GEMV with FP32 accumulation; the products equal the FP32 ones."""
        state = getattr(self, "_v41_h16", None)
        if (
            state
            and len(state) == 4
            and _NATIVE
            and x.is_cuda
            and x.dtype == torch.float16
            and 8 < x.shape[0] <= 1024
        ):
            # VLLM_V41_BATCH: the same exact FP16 * 2^s products on cuBLAS
            # with FP32 accumulation and output.
            out = torch.mm(x.contiguous(), state[3], out_dtype=torch.float32)
            out.mul_(state[2])
            return out[:, :512].contiguous(), out[:, 512:].contiguous()
        if not (
            _NATIVE and x.is_cuda and x.dtype == torch.float16 and 1 <= x.shape[0] <= 8
        ):
            return None
        if not state:  # prepare_loaded_linear_weights builds it (or False)
            return None
        logger.info_once("V4.1 native: ratio-2 compressor on exact FP16 h16_gemv")
        packed, tickets, oscale = state[:3]
        out = torch.empty(x.shape[0], 1024, dtype=torch.float32, device=x.device)
        torch.ops.v41sk.h16_gemv(out, x.contiguous(), packed, tickets, 8, 5, oscale)
        return out[:, :512].contiguous(), out[:, 512:].contiguous()


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
        # Score columns of the batched decode path: the longest admitted
        # context (per-request loops stop at that request's length).
        self.max_context = max(
            16384, get_current_vllm_config().model_config.max_model_len
        )
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
        return key

    def prepare_keys_batch(self, latent, positions, valid_tokens, table, ratio, layout):
        key = self.k_norm(self.wk(latent))
        group_positions = positions - (ratio - 1)
        key = apply_rope(key, group_positions.clamp_min(0), table)
        key = attention_qdq(key, fp4=True, group=32)
        metadata = self.k_cache.metadata()  # type: ignore[union-attr]
        insert_latents_batch(
            self.k_cache.kv_cache,  # type: ignore[union-attr]
            key,
            positions,
            valid_tokens,
            metadata.block_table,
            layout,
            block_size=metadata.block_size,
            ratio=ratio,
        )
        return key

    def forward_batch(self, x, qr, positions, valid_tokens, table, source_cache, ratio):
        query = self.wq_b(qr).view(x.shape[0], self.heads, 128)
        query = attention_qdq(apply_rope(query, positions, table), fp4=True, group=32)
        weights = self.weights_proj(x) * (1.0 / 64.0)
        metadata = source_cache.metadata()
        return select_compressed_positions_batch(
            query,
            weights,
            source_cache.kv_cache,
            metadata.block_table,
            positions,
            valid_tokens,
            metadata.layout,
            ratio,
            _index_all_reduce(),
            block_size=metadata.block_size,
            max_context=self.max_context,
            all_gather=_index_all_gather(),
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
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
                _index_all_reduce(),
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
            _index_all_reduce(),
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
        # Native PP: the ratio-1 source whose new rows this stage exports to
        # the next stage's KV mirror, and the rows of the current forward.
        self.export_source = None
        self.exported = None
        # VLLM_V41_BATCH: this step's batch layout and per-row state slots.
        self.layout = None
        self.slots = None


_ATT_T: dict[str, Any] = {
    "on": envs.VLLM_V41_ATTN_TIMING == "1",
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
# Reference for the PP2 bit-identity check (M1): one stage takes the pipeline
# boundary branch before this target layer. 0: off.
_PP_BOUNDARY_SIM = int(envs.VLLM_V41_PP_BOUNDARY_SIM or 0)


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
            c.o_groups * c.o_lora_rank,
            c.hidden_size,
            prefix=prefix + ".wo_b",
            reduce_results=not _LL_AR,
            **kwargs,
        )
        self.attn_sink = nn.Parameter(
            torch.empty(self.heads, dtype=torch.float32), requires_grad=False
        )
        self.attn_sink.weight_loader = self._load_sink
        # The first layer of each pipeline stage keeps the paged window
        # cache: it anchors the per-step metadata (layer 0; layer 20 on the
        # second PP2 stage). Other target layers keep their window in a
        # private ring instead of a full-length KV-manager allocation (G4).
        # A simulated boundary layer keeps a paged cache too, as it would
        # as the first layer of the second stage.
        self.ring_window = (
            _WINDOW_RINGS
            and self.ring_capable
            and layer > v41_stage_bounds(c)[0]
            and layer != _PP_BOUNDARY_SIM
        )
        # A compressed anchor (layer 20 on the second PP2 stage, or the
        # simulated boundary) keeps its paged cache for the metadata, but
        # the packed attention path reads window keys only from a ring, so
        # it also keeps a ring, filled after every step like the others.
        self.anchor_ring = (
            _WINDOW_RINGS
            and self.ring_capable
            and not self.ring_window
            and layer > 0
            and bool(self.ratio)
        )
        if self.ring_window or self.anchor_ring:
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
        if not self.ring_window:
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
        if FUSE_ATTN_PREP:
            query, window = self._prep_fused(tokens, qr, kv_low, positions, table)
        else:
            query = apply_rope(
                self.wq_b(qr).view(tokens, self.heads, 512), positions, table
            )
            window = attention_qdq(apply_rope(self.kv_norm(kv_low), positions, table))
        if FUSE_ATTN_POST:
            ring_values = None
        metadata = (
            self.shared.metadata if self.ring_window else self.window_cache.metadata()
        )
        if metadata is None:
            # Profile computes the dense paths but has no allocated KV pages.
            if hasattr(self, "compressor"):
                self.compressor(x, positions, profile=True)
            output = torch.zeros_like(query)
        elif self.shared.layout is not None:
            output = self._attend_batch(x, qr, positions, table, query, window)
            if isinstance(output, tuple):  # fused post: QDQ'd output
                _tmark("a.out_proj", tokens)
                return self._project_quantized(output[0])
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
                    index_key = self.indexer.prepare_keys(
                        latent, positions, valid, table, self.ratio
                    )
                    compact = apply_rope(
                        latent, (positions - self.ratio + 1).clamp_min(0), table
                    )
                    compact = attention_qdq(
                        compact, fp4=True, group=16, e4m3_scale=True
                    )
                    if self.shared.export_source == self.layer:
                        # Ratio 1: one row per token, mirrored by the next stage.
                        self.shared.exported = (compact, index_key)
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
                    if self.anchor_ring:
                        arguments.update(
                            window_ring=self.window_ring, window_values=window
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
            if FUSE_ATTN_POST:
                quantized = attend_ring_post(
                    query, positions, valid, self.attn_sink, table, **arguments
                )
                if quantized is not None:
                    _tmark("a.out_proj", tokens)
                    return self._project_quantized(quantized)
            output = attend_paged_latents(
                query,
                positions,
                valid,
                None if self.ring_window else self.window_cache.kv_cache,
                metadata.block_table,
                self.attn_sink,
                **arguments,
            )
            if self.ring_window or self.anchor_ring:
                if FUSE_ATTN_POST:
                    ring_values = (window, valid)
                else:
                    insert_window_ring(self.window_ring, window, positions, valid)
        _tmark("a.out_proj", tokens)
        if FUSE_ATTN_POST:
            return self._post_fused(output, positions, table, ring_values)
        output = apply_rope(output, positions, table, inverse=True)
        return self.project_output(output)

    def _attend_batch(self, x, qr, positions, table, query, window):
        """VLLM_V41_BATCH: window insert, compressor, indexer and attention for
        several requests (paged window caches; rings are not used)."""
        if self.ring_window or self.anchor_ring or self.shared.store is not None:
            raise ValueError(
                "VLLM_V41_BATCH needs VLLM_V41_WINDOW_RING=0, no packed KV"
            )
        tokens = x.shape[0]
        layout, slots = self.shared.layout, self.shared.slots
        metadata = self.window_cache.metadata()
        valid = self.shared.valid
        if valid is None or valid.shape[0] != tokens:
            valid = metadata.slot_mapping[:tokens] >= 0
        # VLLM_V41_BATCH_ATTN_FUSED (decode-only steps, with FUSE_ATTN_POST):
        # v41sk.attn_paged_dec reads the step's window rows from `window` and
        # inserts them, the combine does the inverse RoPE + QDQ.
        fused = (
            _BATCH_ATTN_FUSED
            and FUSE_ATTN_POST
            and short_rows(tokens)
            and query.shape[1] == 16
            and positions.dtype == torch.int64
        )
        if not fused:
            insert_latents_batch(
                self.window_cache.kv_cache,
                window,
                positions,
                valid,
                metadata.block_table,
                layout,
                block_size=metadata.block_size,
            )
        arguments = dict(window_block=metadata.block_size)
        if self.ratio:
            source = self.shared.layers[self.source]
            if hasattr(self, "compressor"):
                latent = self.compressor.forward_batch(
                    x, positions, valid, layout, slots
                )
                index_key = self.indexer.prepare_keys_batch(
                    latent, positions, valid, table, self.ratio, layout
                )
                compact = apply_rope(
                    latent, (positions - self.ratio + 1).clamp_min(0), table
                )
                compact = attention_qdq(compact, fp4=True, group=16, e4m3_scale=True)
                if self.shared.export_source == self.layer:
                    self.shared.exported = (compact, index_key)
                compressed_metadata = self.compressor.cache.metadata()  # type: ignore[union-attr]
                insert_latents_batch(
                    self.compressor.cache.kv_cache,  # type: ignore[union-attr]
                    compact,
                    positions,
                    valid,
                    compressed_metadata.block_table,
                    layout,
                    block_size=compressed_metadata.block_size,
                    ratio=self.ratio,
                )
            if hasattr(self, "indexer"):
                self.shared.topk = self.indexer.forward_batch(
                    x, qr, positions, valid, table, source.indexer.k_cache, self.ratio
                )
            compressed_metadata = source.compressor.cache.metadata()
            arguments.update(
                compressed_cache=source.compressor.cache.kv_cache,
                compressed_table=compressed_metadata.block_table,
                compressed_block=compressed_metadata.block_size,
                ratio=self.ratio,
                topk=self.shared.topk,
            )
        if fused and (
            "topk" not in arguments or arguments["topk"].dtype == torch.int32
        ):
            from vllm.models.deepseek_v41.native import load_v41sk

            load_v41sk()
            result = torch.empty_like(query)
            compressed = "compressed_cache" in arguments
            torch.ops.v41sk.attn_paged_dec(
                result,
                query.contiguous(),
                window.contiguous(),
                self.window_cache.kv_cache,
                metadata.block_table,
                arguments["compressed_cache"] if compressed else window,
                arguments["compressed_table"] if compressed else metadata.block_table,
                layout.rows,
                layout.bases,
                positions,
                valid,
                arguments["topk"] if compressed else positions,
                self.attn_sink,
                512**-0.5,
                metadata.block_size,
                arguments["compressed_block"] if compressed else 1,
                self.ratio if compressed else 1,
                128,
                arguments["topk"].shape[1] if compressed else 0,
                table,
            )
            return (result.view(tokens, 2, 4096),)
        if fused:  # topk not int32: the unfused path needs the window rows
            insert_latents_batch(
                self.window_cache.kv_cache,
                window,
                positions,
                valid,
                metadata.block_table,
                layout,
                block_size=metadata.block_size,
            )
        return attend_paged_latents_batch(
            query,
            positions,
            valid,
            self.window_cache.kv_cache,
            metadata.block_table,
            self.attn_sink,
            layout,
            **arguments,
        )

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
        return self._wo_b(projected)

    def _wo_b(self, projected):
        if self.wo_b.reduce_results:
            return self.wo_b(projected)
        from .native import tp_all_reduce

        return tp_all_reduce(self.wo_b(projected))

    def _prep_fused(self, tokens, qr, kv_low, positions, table):
        # VLLM_V41_FUSE_ATTN_PREP: forward's query and window lines, one launch.
        q_high = self.wq_b(qr).view(tokens, self.heads, 512)
        hc_kick()  # pending mixes overlap the latency-bound attention kernels
        prepared = attn_prep(
            q_high, kv_low, self.kv_norm.weight, self.kv_norm.eps, positions, table
        )
        if prepared is not None:
            return prepared
        query = apply_rope(q_high, positions, table)
        window = attention_qdq(apply_rope(self.kv_norm(kv_low), positions, table))
        return query, window

    def _post_fused(self, output, positions, table, ring_values):
        # VLLM_V41_FUSE_ATTN_POST: the deferred window ring insert, the inverse
        # RoPE and project_output's QDQ in one launch, then its projections.
        values, valid = ring_values if ring_values is not None else (None, None)
        ring = self.window_ring if ring_values is not None else None
        quantized = attn_post(
            output, positions, table, self.groups, ring, values, valid
        )
        if quantized is not None:
            return self._project_quantized(quantized)
        if ring_values is not None:
            insert_window_ring(self.window_ring, values, positions, valid)
        output = apply_rope(output, positions, table, inverse=True)
        return self.project_output(output)

    def _project_quantized(self, output):
        # project_output after its attention_qdq. A copy: project_output stays
        # as it was for the knob-off path and V41DraftAttention.
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
        return self._wo_b(projected)


class V41KVMirror(nn.Module):
    """A ratio-1 KV source's caches on the stage after the source's stage.

    The caches carry the source's own prefixes, so they get the same KV specs
    and the same block ids from this stage's KV manager. ``ingest`` inserts
    the source's new rows (its post-QDQ compressed latents and index keys,
    shipped with the residual) before this stage's layers read them.
    """

    def __init__(self, c, source, shared, prefix):
        super().__init__()
        if c.compress_ratios[source] != 1:
            raise ValueError("V4.1 KV mirrors hold ratio-1 sources only")
        self.layer, self.ratio = source, 1
        self.compressor = nn.Module()
        self.compressor.cache = V41LatentCache(
            prefix + ".attn.compressor.cache", ratio=1
        )
        self.indexer = nn.Module()
        self.indexer.k_cache = V41LatentCache(
            prefix + ".attn.indexer.k_cache", head_dim=128, ratio=1
        )
        shared.layers[source] = self

    def ingest(self, compact, index_key, positions, valid):
        tokens = positions.shape[0]
        for cache, rows in (
            (self.compressor.cache, compact),
            (self.indexer.k_cache, index_key),
        ):
            metadata = cache.metadata()
            if metadata.layout is not None:
                insert_latents_batch(
                    cache.kv_cache,
                    rows[:tokens],
                    positions,
                    valid,
                    metadata.block_table,
                    metadata.layout,
                    block_size=metadata.block_size,
                    ratio=1,
                )
                continue
            insert_latents(
                cache.kv_cache,
                rows[:tokens],
                positions,
                valid,
                metadata.block_table,
                block_size=metadata.block_size,
                ratio=1,
            )


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


def v41_stage_bounds(config):
    """[start, end) of this PP stage's target layers, after the split rule.

    Every boundary must satisfy ``pp_split_problem`` (the KV20 rule): only the
    four-stream HC residual crosses a stage boundary. For 40 layers and two
    stages the only admissible split is 20.
    """
    from vllm.distributed.utils import get_pp_indices

    pp = get_pp_group()
    layers = config.num_hidden_layers
    starts = [get_pp_indices(layers, r, pp.world_size)[0] for r in range(pp.world_size)]
    for rank in range(1, pp.world_size):
        split = starts[rank]
        problem = pp_split_problem(config, split, staged_state=_NATIVE)
        source = kv_source_of(config, split)
        if (
            problem is None
            and source != split
            and not (starts[rank - 1] <= source < split)
        ):
            problem = f"KV source {source} must be on the stage just before"
        if problem is not None:
            raise ValueError(f"V4.1 pipeline split at layer {split}: {problem}")
    return get_pp_indices(layers, pp.rank_in_group, pp.world_size)


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
        # Global layer ids are kept: list index == checkpoint layer id, and
        # layers owned by the other stage are PPMissingLayer placeholders.
        self.start_layer, self.end_layer = v41_stage_bounds(c)
        self.layers = nn.ModuleList(
            [
                V41Block(vllm_config, i, shared, prefix + f".layers.{i}")
                if self.start_layer <= i < self.end_layer
                else PPMissingLayer()
                for i in range(c.num_hidden_layers)
            ]
        )
        if _PACKED_KV:
            from vllm.distributed import get_tp_group

            from .kv_store import V41PackedKV

            # One store per stage over its own KV sources ({2, 8, 14} | {20});
            # the memfd handshake is per TP group, so stages never share it.
            shared.store = V41PackedKV(
                {
                    layer: c.compress_ratios[layer]
                    for layer in c.kv_source_layer_ids
                    if self.start_layer <= layer < self.end_layer
                },
                vllm_config.model_config.max_model_len,
                torch.device("cuda", torch.accelerator.current_device_index()),
                get_tp_group(),
            )
        # Packed KV shares the same session-prefix lifetime as device state.
        self._v41_shared = shared
        self._v41_hot_refill_logged = False
        self.norm = (
            V41Norm(c.hidden_size) if get_pp_group().is_last_rank else PPMissingLayer()
        )
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
        # This stage's Engram layers (channels of the shared reader) and
        # whether it holds committed per-step state (Engram hash history,
        # ratio-2 compressors). On the GGUF route that is the first stage only
        # (pp_split_problem); on the native route any stage may. A stage with
        # state that is not the last never samples, so it commits its own
        # forwards (pp_commit.py).
        self.engram_channels = tuple(
            channel
            for channel, layer in enumerate(c.engram_layer_ids)
            if self.start_layer <= layer < self.end_layer
        )
        self.owns_state = bool(self.engram_channels) or any(
            c.compress_ratios[layer] == 2
            for layer in c.kv_source_layer_ids
            if self.start_layer <= layer < self.end_layer
        )
        self.stage_commit = (
            StageCommit()
            if self.owns_state and self.end_layer < c.num_hidden_layers
            else None
        )
        # KV mirror (native PP): a stage that starts inside the consumers of a
        # ratio-1 source keeps that source's compressed and index caches and
        # fills them from the previous stage's hand-off. The previous stage
        # exports those rows.
        source = kv_source_of(c, self.start_layer)
        self.kv_mirror = None
        if source >= 0 and source < self.start_layer:
            self.kv_mirror = V41KVMirror(
                c, source, shared, prefix + f".layers.{source}"
            )
        following = (
            kv_source_of(c, self.end_layer)
            if self.end_layer < c.num_hidden_layers
            else self.end_layer
        )
        self.export_source = (
            following if self.start_layer <= following < self.end_layer else None
        )
        shared.export_source = self.export_source
        # A stage boundary inside an index group (ratio-1 layers): the first
        # layer after it reuses the group's top-k, shipped with the residual.
        index_sources = tuple(c.index_source_layer_ids)
        ratios = tuple(c.compress_ratios)
        self.topk_import = (
            0 < self.start_layer < c.num_hidden_layers
            and self.start_layer not in index_sources
            and ratios[self.start_layer] > 0
        )
        self.topk_export = (
            self.end_layer < c.num_hidden_layers
            and self.end_layer not in index_sources
            and ratios[self.end_layer] > 0
        )
        self.register_buffer(
            "stage_commit_count", torch.zeros(1, dtype=torch.int64), persistent=False
        )
        # VLLM_V41_BATCH: request slots of the per-request state (Engram hash
        # history, C2 states), and the pending selector inputs per batch row.
        self.max_slots = vllm_config.scheduler_config.max_num_seqs if _BATCH else 0
        if _BATCH:
            device = torch.device("cuda", torch.accelerator.current_device_index())
            slots = self.max_slots
            self.req_slots = torch.zeros(slots, dtype=torch.int32, device=device)
            self._slot_hosts = [
                torch.zeros(slots, dtype=torch.int32, device="cpu").pin_memory()
                for _ in range(16)  # a ring: the host must not wait on the GPU
            ]
            self._slot_events = [torch.cuda.Event() for _ in range(16)]
            self._slot_turn = 0
            self._slot_of: dict[str, int] = {}
            self._free_slots = list(range(slots - 1, -1, -1))
            self._slot_done: dict[int, torch.cuda.Event | None] = {}
            self.batch_lengths = torch.zeros(slots, dtype=torch.int64, device=device)
            self.batch_positions = torch.zeros(slots, dtype=torch.int64, device=device)
            self.batch_prefill = torch.zeros(slots, dtype=torch.bool, device=device)
            # The state slots of the last forward's rows (its commits use them).
            self.batch_slots = torch.zeros(slots, dtype=torch.int32, device=device)
            self.batch_reqs = 0
            # Commits of a forward may run (on a side stream) after the next
            # forward rewrote the batch buffers: each armed forward keeps a
            # copy of its slots / prefill flags / lengths (VLLM_V41_PP_MICRO).
            self._snapshots = [
                (
                    torch.zeros(slots, dtype=torch.int32, device=device),
                    torch.zeros(slots, dtype=torch.bool, device=device),
                    torch.zeros(slots, dtype=torch.int64, device=device),
                    torch.zeros(slots, dtype=torch.int64, device=device),
                )
                for _ in range(64)
            ]
            # An entry is rewritten only after the deferred commit that reads it
            # (recorded on the stream that ran it) is done.
            self._snapshot_done: list[torch.cuda.Event | None] = [None] * 64
            self._snapshot_turn = 0
            # Set while a deferred round (side stream) clips and commits the
            # armed forward: its selectors come from that forward's snapshot.
            self._deferred: int | None = None
            self.hash_batch = None
        # M1: called eagerly with (hidden, pre) at a real or simulated stage
        # boundary (benchmarks/deepseek_v41/tools/pp2_m1.py).
        self.pp_boundary_tap: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        if _PP_BOUNDARY_SIM and not 0 < _PP_BOUNDARY_SIM < c.num_hidden_layers:
            raise ValueError(
                f"VLLM_V41_PP_BOUNDARY_SIM={_PP_BOUNDARY_SIM} is not a layer "
                f"boundary inside 1..{c.num_hidden_layers - 1}"
            )

    def prepare_postload(self):
        if _BATCH:
            from vllm.model_executor.kernels.deepseek_v41_compression import (
                V41C2StateBatch,
            )

            for i in range(self.start_layer, self.end_layer):
                compressor = getattr(self.layers[i].attn, "compressor", None)
                if (
                    compressor is not None
                    and compressor.ratio == 2
                    and getattr(compressor, "state_batch", None) is None
                ):
                    compressor.state_batch = V41C2StateBatch(
                        compressor.state, self.max_slots + 1
                    )
        # Engram, its hash state and the sidecar reader belong to the stages
        # that hold Engram layers; the others never open the Engram tables.
        if self.hash_state is not None or not self.engram_channels:
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
        if _BATCH:
            from vllm.model_executor.layers.deepseek_v41_engram import (
                V41EngramHashBatch,
            )

            # One extra slot (index max_slots) takes the padding rows of graph
            # batches, so they never touch a live request's state.
            self.hash_batch = V41EngramHashBatch(self.hash_state, self.max_slots + 1)
        path = self.vllm_config.load_config.model_loader_extra_config["engram_path"]
        self.engram_io = SharedEngramIO(
            path, max_tokens=maximum, channels=self.engram_channels
        )
        if _NATIVE:
            self._fold_engram_row_bias()

    def _fold_engram_row_bias(self):
        """Native rows decode to FP16 times 2^row_bias; give each projection
        the exact compensating 2^-row_bias in its UE8M0 scales."""
        c = self.config
        assert self.engram_io is not None
        for channel in self.engram_channels:
            bias = self.engram_io.specs[channel].row_bias
            wkv = self.layers[c.engram_layer_ids[channel]].engram.wkv
            scales = wkv.weight_scale_inv.data.view(torch.uint8)
            shifted = scales.to(torch.int32) - bias
            if bool((shifted < 1).any()) or bool((shifted > 254).any()):
                raise ValueError("V4.1 Engram row bias leaves the UE8M0 range")
            scales.copy_(shifted.to(torch.uint8))
            prepare = getattr(wkv.quant_method, "prepare_q8", None)
            if prepare is not None:
                prepare(wkv)  # the decode copy carries the shifted scales

    def forward(
        self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None
    ):
        tokens = positions.numel()
        self._intermediate = intermediate_tensors
        if (
            _EMPTY_CACHE_TOKENS
            and tokens >= _EMPTY_CACHE_TOKENS
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Long prefill passes fragment the caching allocator; return the
            # previous pass's free segments so driver-free VRAM stays >= 1 GB.
            torch.accelerator.empty_cache()
        # This stage's first layer anchors the per-step metadata.
        anchor = self.layers[self.start_layer].attn
        metadata = anchor.window_cache.metadata()
        anchor.shared.metadata = metadata
        valid = metadata.slot_mapping[:tokens] >= 0 if metadata is not None else None
        anchor.shared.valid = valid
        layout = metadata.layout if metadata is not None else None
        if layout is None:
            set_decode_rows(False)
        anchor.shared.layout = layout
        anchor.shared.slots = self.req_slots if layout is not None else None
        if layout is not None:
            self._forward_batch_state(input_ids, positions, valid, layout)
        elif metadata is not None:
            if self.engram_channels:
                # Engram and its hash state live on the stages that hold it.
                assert (
                    self.hash_state is not None
                    and self.engram_io is not None
                    and self.engram_io.runtime is not None
                )
                hashes = self.hash_state(input_ids, positions, valid)
                self.engram_io.runtime.prefetch(hashes)
            if self.kv_mirror is not None and intermediate_tensors is not None:
                self.kv_mirror.ingest(
                    intermediate_tensors["kv_mirror_ckv"],
                    intermediate_tensors["kv_mirror_ik"],
                    positions,
                    valid,
                )
            self._import_topk(intermediate_tensors, tokens)
            # Every stage: the first commits from these, the last clips its
            # samples with them (output_budget).
            self.pending_length.fill_(tokens)
            self.pending_position.copy_(positions[:1])
            self.pending_prefill.copy_(metadata.is_prefilling[:1])
        if intermediate_tensors is None:
            if self.start_layer != 0:
                raise ValueError("V4.1 pipeline stage needs the previous HC residual")
            hidden = (
                self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            )
            # FP16 cannot carry the full HC residual range of valid special
            # tokens. Keep only this stream in FP32; collapse+RMSNorm returns
            # FP16 at every dense/attention/expert input boundary.
            hidden = hidden.float()[:, None].expand(-1, 4, -1).contiguous()
            pre = torch.zeros(tokens, 4, device=hidden.device, dtype=torch.float32)
            pre[:, 0] = 1
        else:
            # The previous stage's expanded residual and the collapse weights
            # of its last FFN; this stage starts with the reference mixes.
            hidden, pre = intermediate_tensors["hidden"], intermediate_tensors["pre"]
        auxiliary: Any = []
        if _HC_FUSED:
            result = self._forward_fused(
                hidden, positions, pre, metadata, valid, auxiliary
            )
            _join_expert_writebacks()
        else:
            result = self._forward_reference(
                hidden, positions, pre, metadata, valid, auxiliary
            )
            _join_expert_writebacks()
        if isinstance(result, IntermediateTensors):
            return result
        return (result, auxiliary) if auxiliary else result

    def _forward_batch_state(self, input_ids, positions, valid, layout):
        """VLLM_V41_BATCH: hashes, KV mirror and per-row pending selectors."""
        starts = layout.query_start_loc_cpu[: layout.num_reqs + 1]
        set_decode_rows(bool((starts[1:] - starts[:-1]).max() <= 6))
        if self.engram_channels:
            assert self.hash_batch is not None and self.engram_io is not None
            hashes = self.hash_batch(
                input_ids, positions, valid, layout, self.req_slots
            )
            runtime = self.engram_io.runtime
            assert runtime is not None
            runtime.prefetch(hashes)
        if self.kv_mirror is not None and self._intermediate is not None:
            self.kv_mirror.ingest(
                self._intermediate["kv_mirror_ckv"],
                self._intermediate["kv_mirror_ik"],
                positions,
                valid,
            )
        self._import_topk(self._intermediate, positions.shape[0])
        reqs = layout.num_reqs
        starts = layout.query_start_loc[: reqs + 1].to(torch.int64)
        self.batch_lengths[:reqs].copy_(starts[1:] - starts[:-1])
        self.batch_positions[:reqs].copy_(
            positions[starts[:-1].clamp_max(positions.numel() - 1)]
        )
        self.batch_prefill[:reqs].copy_(layout.prefill[:reqs])
        self.batch_slots[:reqs].copy_(self.req_slots[:reqs])
        # Row 0 also fills the single-request selectors (fused clip, greedy
        # head), which only run for one-request batches.
        self.pending_length.copy_(self.batch_lengths[:1])
        self.pending_position.copy_(self.batch_positions[:1])
        self.pending_prefill.copy_(self.batch_prefill[:1])

    def _selectors(self):
        """(prefill, lengths, positions) per row of the forward being clipped
        and committed: a deferred round's snapshot, else the last forward."""
        if self._deferred is not None:
            _, prefill, lengths, positions = self._snapshots[self._deferred]
            return prefill, lengths, positions
        return self.batch_prefill, self.batch_lengths, self.batch_positions

    def v41_begin_deferred(self):
        """Runner hook: a non-last stage handles a round off the main stream."""
        armed = getattr(self, "_armed_rows", None)
        self._deferred = None if armed is None else armed[2]

    def v41_end_deferred(self):
        self._deferred = None

    def v41_set_batch(self, req_ids):
        """Runner hook before each forward: the state slot of every batch row."""
        if not _BATCH:
            return
        # Set here, not in forward: a CUDA graph replay runs no Python.
        self.batch_reqs = len(req_ids)
        if True:  # batched state checks raise here (recorded on device)
            if self.hash_batch is not None:
                self.hash_batch.check_errors()
            for i in range(self.start_layer, self.end_layer):
                state = getattr(
                    getattr(self.layers[i].attn, "compressor", None),
                    "state_batch",
                    None,
                )
                if state is not None:
                    state.check_errors()
        fresh = []
        for req_id in req_ids:
            if req_id not in self._slot_of:
                if not self._free_slots:
                    raise RuntimeError("V4.1 batch: more live requests than slots")
                slot = self._free_slots.pop()
                self._slot_of[req_id] = slot
                fresh.append(slot)
        host = self._slot_hosts[self._slot_turn]
        event = self._slot_events[self._slot_turn]
        self._slot_turn = (self._slot_turn + 1) % len(self._slot_hosts)
        event.synchronize()
        host.fill_(self.max_slots)  # padding rows: the scratch slot
        host[: len(req_ids)].copy_(
            torch.tensor([self._slot_of[r] for r in req_ids], dtype=torch.int32)
        )
        self.req_slots.copy_(host, non_blocking=True)
        event.record()
        for slot in fresh:
            done = self._slot_done.pop(slot, None)
            if done is not None:
                torch.cuda.current_stream().wait_event(done)
            self._reset_slot(slot)

    def v41_release(self, req_ids, done=None):
        """Runner hook: finished requests give their state slots back. done:
        an event after the last work already enqueued for them (a deferred
        commit of an over-scheduled step); the slot is reset only after it."""
        if not _BATCH:
            return
        for req_id in req_ids:
            slot = self._slot_of.pop(req_id, None)
            if slot is not None:
                self._free_slots.insert(0, slot)  # reuse the oldest last
                self._slot_done[slot] = done

    def _reset_slot(self, slot):
        if self.hash_batch is not None:
            self.hash_batch.reset(slot)
        for i in range(self.start_layer, self.end_layer):
            compressor = getattr(self.layers[i].attn, "compressor", None)
            state = getattr(compressor, "state_batch", None)
            if state is not None:
                state.reset(slot)

    def _stage_boundary(self, i):
        """Whether the residual leaves after layer i: at the end of a stage
        that is not the last, or before VLLM_V41_PP_BOUNDARY_SIM."""
        following = i + 1
        return following < len(self.layers) and (
            following == self.end_layer or following == _PP_BOUNDARY_SIM
        )

    def _hand_off(self, i, hidden, pre):
        """Stage boundary after layer i (expanded residual and collapse weights).

        Returns the next stage's inputs at a real boundary, else None.
        """
        assert hidden.dtype == pre.dtype == torch.float32
        tap = self.pp_boundary_tap
        if tap is not None and not torch.cuda.is_current_stream_capturing():
            tap(hidden, pre)
        if i + 1 < self.end_layer:
            return None
        _tmark("end", hidden.shape[0])
        if _ATT_T["on"]:
            _treport(hidden.shape[0])
        tensors = {"hidden": hidden, "pre": pre.contiguous()}
        if self.export_source is not None:
            tensors.update(self._exported_rows(hidden.shape[0], hidden.device))
        if self.topk_export:
            tensors["v41_topk"] = self._exported_topk(hidden.shape[0], hidden.device)
        return IntermediateTensors(tensors)

    def _exported_topk(self, tokens, device):
        """The index group's top-k for the next stage's first layers ([T, 512]
        int32, -1 padded)."""
        topk = self.layers[self.end_layer - 1].attn.shared.topk
        out = torch.full((tokens, 512), -1, dtype=torch.int32, device=device)
        if topk is not None and topk.shape[0] == tokens:
            width = min(512, topk.shape[1])
            out[:, :width].copy_(topk[:, :width])
        return out

    def _import_topk(self, intermediate, tokens):
        if self.topk_import and intermediate is not None:
            self.layers[self.start_layer].attn.shared.topk = intermediate["v41_topk"][
                :tokens
            ]

    def _exported_rows(self, tokens, device):
        """The mirrored source's new compressed/index rows for the next stage."""
        exported = self.layers[self.export_source].attn.shared.exported
        if exported is None or exported[0].shape[0] != tokens:
            # Profile and dummy forwards insert nothing.
            return {
                "kv_mirror_ckv": torch.zeros(
                    tokens, 512, device=device, dtype=torch.float16
                ),
                "kv_mirror_ik": torch.zeros(
                    tokens, 128, device=device, dtype=torch.float16
                ),
            }
        self.layers[self.export_source].attn.shared.exported = None
        return {"kv_mirror_ckv": exported[0], "kv_mirror_ik": exported[1]}

    def _forward_reference(self, hidden, positions, pre, metadata, valid, auxiliary):
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if layer.engram is not None and metadata is not None:
                hidden = layer.engram(hidden, self.engram_io, valid)
            if i in self.aux_hidden_state_layers:
                auxiliary.append(hidden.mean(1).mul_(2.0**-6).half())
            table = self.compressed_rope if layer.attn.ratio else self.window_rope
            hidden, pre = layer(hidden, positions, pre, table)
            if self._stage_boundary(i):
                handed = self._hand_off(i, hidden, pre)
                if handed is not None:
                    return handed
        return collapse_hc(hidden, pre, self.norm.weight)

    def _forward_fused(self, hidden, positions, pre, metadata, valid, auxiliary):
        # Each sublayer boundary is one fused transition (in-place expand, the
        # next sublayer's mixes, delayed collapse). A boundary falls back to the
        # reference mixes/collapse when Engram rewrites the residual, and at a
        # pipeline stage boundary (the next layer's parameters are elsewhere).
        state = None
        layers = list(self.layers)
        start, end = self.start_layer, self.end_layer
        for i in range(start, end):
            layer = layers[i]
            _tmark("engram", hidden.shape[0])
            if layer.engram is not None and metadata is not None:
                hidden = layer.engram(hidden, self.engram_io, valid)
                state = None
            _tmark("hc_pre", hidden.shape[0])
            if i in self.aux_hidden_state_layers:
                auxiliary.append(hidden.mean(1).mul_(2.0**-6).half())
            if state is None:
                hc_join()
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
                _predict_next(layers[:end], i, inputs)
            if _PREFETCH_HINTS:
                gates = getattr(self, "_v41_gates", None)
                if gates is None:
                    # Routers of this stage's layers; hint() bounds by len().
                    gates = (
                        [None] * start
                        + [
                            (l.ffn.gate.weight, l.ffn.gate.bias)  # noqa: E741
                            for l in layers[start:end]  # noqa: E741
                        ]
                    )
                    self._v41_gates = gates
                publish_prefetch_hints(gates, i, inputs)
            _tmark("ffn(moe+shared)", hidden.shape[0])
            y = layer.ffn(inputs)
            _tmark("hc_post", hidden.shape[0])
            pre = ffn_mix[0]
            boundary = self._stage_boundary(i)
            following = layers[i + 1] if i + 1 < end else None
            if boundary:
                hc_join()
                hidden = expand_hc(y, hidden, ffn_mix[1], ffn_mix[2], inplace=True)
                state = None
                handed = self._hand_off(i, hidden, pre)
                if handed is not None:
                    return handed
            elif following is not None and following.engram is None:
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
                hc_join()
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

    def _ratio2_states(self):
        for i in range(self.start_layer, self.end_layer):
            compressor = getattr(self.layers[i].attn, "compressor", None)
            if compressor is not None and compressor.ratio == 2:
                yield compressor.state

    def _commit_batch(self, selector, reqs, slots=None):
        """VLLM_V41_BATCH: selector[r] inputs of row r (-1: row skipped)."""
        selector = selector[:reqs].to(torch.int64).contiguous()
        slots = self.batch_slots[:reqs] if slots is None else slots[:reqs]
        if self.hash_batch is not None:
            self.hash_batch.commit(selector, slots)
        for i in range(self.start_layer, self.end_layer):
            compressor = getattr(self.layers[i].attn, "compressor", None)
            state = getattr(compressor, "state_batch", None)
            if state is not None:
                state.commit(selector, slots)

    def commit(self, accepted_counts):
        """Sampled-count commit (the runner, after sampling)."""
        if not self.owns_state:
            return  # this PP stage holds no committed state
        if _BATCH:
            reqs = min(accepted_counts.shape[0], self.batch_reqs)
            if reqs == 0:
                return
            counts = accepted_counts[:reqs].to(torch.int64)
            prefill = self.batch_prefill[:reqs]
            if self.stage_commit is not None:
                # Prefill rows committed after the forward; decode rows wait
                # for these counts (or the next forward's computed tokens).
                armed = getattr(self, "_armed_rows", None)
                if armed is None:
                    return
                self._armed_rows = None
                snap_slots, snap_prefill, _, _ = self._snapshots[armed[2]]
                reqs = min(reqs, len(armed[0]))
                counts = counts[:reqs]
                selector = torch.where(snap_prefill[:reqs], -1, counts)
                self._commit_batch(selector, reqs, snap_slots)
                done = torch.cuda.Event()
                done.record()
                self._snapshot_done[armed[2]] = done
                return
            selector = torch.where(prefill, self.batch_lengths[:reqs], counts)
            self._commit_batch(selector, reqs)
            return
        if self.stage_commit is not None and not self.stage_commit.take_sampled():
            return  # this stage already committed the forward (a prefill)
        self._commit(accepted_counts)

    def _commit(self, accepted_counts):
        selected = torch.where(
            self.pending_prefill, self.pending_length, accepted_counts.to(torch.int64)
        )
        if self.hash_state is not None:
            self.hash_state.commit(selected)
        for state in self._ratio2_states():
            state.commit(selected)

    def pp_after_forward(self, req_ids, num_computed, num_prompt):
        """A first PP stage that is not the last, after each forward."""
        if self.stage_commit is None:
            return
        if _BATCH:
            reqs = len(req_ids)
            if reqs == 0:
                return
            if getattr(self, "_armed_rows", None) is not None:
                raise RuntimeError(
                    "V4.1 PP batch: the previous decode forward was never committed"
                )
            prefill = self.batch_prefill[:reqs]
            self._commit_batch(
                torch.where(prefill, self.batch_lengths[:reqs], -1), reqs
            )
            if any(c >= p for c, p in zip(num_computed, num_prompt)):
                turn = self._snapshot_turn
                self._snapshot_turn = (turn + 1) % len(self._snapshots)
                done = self._snapshot_done[turn]
                if done is not None:
                    torch.cuda.current_stream().wait_event(done)
                    self._snapshot_done[turn] = None
                for target, source in zip(
                    self._snapshots[turn],
                    (
                        self.batch_slots,
                        self.batch_prefill,
                        self.batch_lengths,
                        self.batch_positions,
                    ),
                ):
                    target[:reqs].copy_(source[:reqs])
                self._armed_rows = (list(req_ids), list(num_computed), turn)
            return
        if self.stage_commit.after_forward(req_ids, num_computed, num_prompt):
            self._commit(self.pending_length)

    def pp_before_forward(self, req_ids, num_computed):
        """A first PP stage that is not the last, before each forward."""
        if self.stage_commit is None:
            return
        if _BATCH:
            armed = getattr(self, "_armed_rows", None)
            if armed is None:
                return
            self._armed_rows = None
            now = dict(zip(req_ids, num_computed))
            counts = [
                now[r] - before if r in now else -1
                for r, before in zip(armed[0], armed[1])
            ]
            reqs = len(counts)
            snap_slots, snap_prefill, _, _ = self._snapshots[armed[2]]
            selector = torch.tensor(counts, dtype=torch.int64).to(
                self.batch_prefill.device, non_blocking=True
            )
            self._commit_batch(
                torch.where(snap_prefill[:reqs], -1, selector), reqs, snap_slots
            )
            return
        count = self.stage_commit.before_forward(req_ids, num_computed)
        if count is not None:
            self._commit(self.stage_commit_count.fill_(count))

    def restore_prefix(self, cached, prompt_ids):
        """Rewind per-sequence device state to a same-session prefix hit.

        The scheduler only admits hits that continue the last executed
        sequence at an even position inside the window rings' reach, so the
        rings need no rewind. Later PP stages hold nothing else.
        """
        assert cached % 2 == 0
        if self.stage_commit is not None:
            self.stage_commit.reset()
        # 1Cat #1341: past the hot ring's size, the previous request's rows past
        # the hit (answer + rejected drafts) sit in slots readers treat as
        # current. Refill those slots from the host copy.
        store = self._v41_shared.store
        if store is not None:
            copied = store.refill_hot(cached)
            if not self._v41_hot_refill_logged:
                self._v41_hot_refill_logged = True
                logger.info(
                    "Session hit at %d refilled %d hot rows",
                    cached,
                    copied,
                )
        if not self.owns_state:
            return
        for state in self._ratio2_states():
            # Even positions start a new pair: the partial row is unused.
            state.partial.zero_()
            state.position.fill_(cached)
            state.metadata[2] = 0
        if self.hash_state is None:
            return
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

    def output_budget(self, max_tokens, prompt_tokens):
        # CPU request lists can contain optimistic placeholders under async
        # scheduling. Corrected input positions are authoritative on device.
        if _BATCH:
            # max_tokens / prompt_tokens: per batch row (device int64 tensors).
            device = self.batch_prefill.device
            if not isinstance(max_tokens, torch.Tensor):
                max_tokens = torch.tensor(
                    [max_tokens], dtype=torch.int64, device=device
                )
            if not isinstance(prompt_tokens, torch.Tensor):
                prompt_tokens = torch.tensor(
                    [prompt_tokens], dtype=torch.int64, device=device
                )
            reqs = max_tokens.shape[0]
            prefill, lengths, positions = self._selectors()
            before = torch.where(
                prefill[:reqs],
                positions[:reqs] + lengths[:reqs],
                positions[:reqs] + 1,
            )
            before = (before - prompt_tokens).clamp_min(0)
            return (max_tokens - before).clamp_min(0)
        before = torch.where(
            self.pending_prefill,
            self.pending_position + self.pending_length,
            self.pending_position + 1,
        )
        before = (before - prompt_tokens).clamp_min(0)
        return (max_tokens - before).clamp_min(0)


class DeepseekV41ForCausalLM(nn.Module, SupportsEagle3, SupportsPP):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        c = self.config = vllm_config.model_config.hf_text_config
        p, s = vllm_config.parallel_config, vllm_config.scheduler_config
        if (
            not current_platform.is_cuda()
            or not current_platform.is_device_capability(70)
            or vllm_config.model_config.dtype != torch.float16
            or get_tensor_model_parallel_world_size() != 4
            or p.pipeline_parallel_size not in ((1, 2, 4) if _NATIVE else (1, 2))
            or p.data_parallel_size != 1
            or p.enable_expert_parallel
            or (s.max_num_seqs != 1 and not _BATCH)
            or (s.async_scheduling and envs.VLLM_V41_ASYNC != "1")
            or s.max_num_batched_tokens > 16384
            or vllm_config.model_config.max_model_len
            > (524288 if _PACKED_KV else 1048576 if _NATIVE and _BATCH else 16384)
            or (
                vllm_config.cache_config.enable_prefix_caching
                and envs.VLLM_V41_SESSION_PREFIX != "1"
            )
        ):
            raise ValueError(
                "V4.1 requires SM70 FP16 TP4 (PP 1 or 2; native weights also "
                "PP4), one request unless VLLM_V41_BATCH, <=16K (native batch "
                "<=1M, packed KV <=512K), chunks <=16384, synchronous "
                "scheduling, no prefix caching"
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
            or vllm_config.quant_config.get_name() != ("fp8" if _NATIVE else "gguf")
        ):
            raise ValueError(
                "V4.1 text architecture, its weight format (GGUF, or the original "
                "FP8 checkpoint with VLLM_V41_NATIVE=1) or the explicit Engram "
                "sidecar is missing"
            )
        validate_runtime_geometry(c)
        if _NATIVE:
            quantization = getattr(
                vllm_config.model_config.hf_config, "quantization_config", None
            ) or getattr(c, "quantization_config", None)
            if (
                not quantization
                or quantization.get("weight_block_size") != [32, 32]
                or quantization.get("scale_fmt") != "ue8m0"
                or quantization.get("expert_dtype") != "fp4"
            ):
                raise ValueError(
                    "VLLM_V41_NATIVE=1 needs the original FP8 32x32 UE8M0 "
                    "checkpoint with FP4 experts"
                )
            # The module tree is unchanged; only the parameter formats differ.
            vllm_config = copy(vllm_config)
            vllm_config.quant_config = V41TargetQuantConfig.from_config(quantization)
        else:
            policy = vllm_config.kernel_config.sm70_gguf
            if policy.dense_storage != "original" or policy.prefill_min_m != 1:  # noqa: SIM102
                raise ValueError(
                    "V4.1 requires original dense storage and prefill_min_m=1 "
                    "for bounded FP16 dense GEMMs; MMVQ is outside this profile"
                )
        with set_current_vllm_config(vllm_config):
            self.model = DeepseekV41Model(vllm_config, maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                c.vocab_size,
                c.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(c.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if intermediate_tensors is None:
            return self.model(input_ids, positions, inputs_embeds)
        return self.model(
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        # The PP hand-off is the FP32 four-stream HC residual plus the
        # collapse weights of the last stage-1 FFN (``pre``), whatever the
        # model dtype: FP16 cannot carry the residual range (see forward).
        del dtype
        tensors = {
            "hidden": torch.zeros(
                (batch_size, 4, self.config.hidden_size),
                dtype=torch.float32,
                device=device,
            ),
            "pre": torch.zeros((batch_size, 4), dtype=torch.float32, device=device),
        }
        if self.model.kv_mirror is not None:
            # The KV mirror rows of the ratio-1 source (FP16 QDQ values).
            tensors["kv_mirror_ckv"] = torch.zeros(
                (batch_size, 512), dtype=torch.float16, device=device
            )
            tensors["kv_mirror_ik"] = torch.zeros(
                (batch_size, 128), dtype=torch.float16, device=device
            )
        if self.model.topk_import:
            # The index group's top-k when the stage starts inside the group.
            tensors["v41_topk"] = torch.full(
                (batch_size, 512), -1, dtype=torch.int32, device=device
            )
        return IntermediateTensors(tensors)

    def get_v41_top_tokens(self, hidden_states):
        from .greedy_head import greedy_top_tokens
        from .native import H16Head

        if (
            not isinstance(self.lm_head.quant_method, H16Head)
            or get_tensor_model_parallel_world_size() != 4
            or hidden_states.dtype != torch.float16
            or not 1 <= hidden_states.shape[0] <= 6
            or self.logits_processor.soft_cap is not None
            or self.logits_processor.scale != 1.0
            or self.lm_head.num_embeddings != self.lm_head.org_vocab_size
            or self.logits_processor.vocab_size != self.lm_head.org_vocab_size
        ):
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
        return greedy_top_tokens(self.lm_head, hidden_states)

    def compute_logits(self, hidden_states):
        if _LL_AR and self.logits_processor.use_all_gather:
            from .native import tp_all_gather

            # Vocab-parallel logits gathered on the one-shot push slots.
            # Instance override of the method (same call shape: logits -> logits).
            self.logits_processor._gather_logits = tp_all_gather  # type: ignore[method-assign,assignment]
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if _NATIVE:
            return load_native_weights(self, weights)
        return AutoWeightsLoader(self).load_weights(weights)

    def prepare_loaded_linear_weights(self):
        if envs.VLLM_V41_GATE_BF16 == "1":
            for module in self.modules():
                if isinstance(module, V41Gate):
                    narrow = module.weight.data.to(torch.bfloat16)
                    # Only when exact: routes stay those of the FP32 gate.
                    if torch.equal(narrow.float(), module.weight.data):
                        module.weight.data = narrow
        if not _NATIVE:  # the TurboMind planes are built from GGUF blocks
            dense_tm.prepare(self.model)
            if not isinstance(self.lm_head, PPMissingLayer):
                dense_tm.attach_lm_head(self.lm_head)
        else:
            from .fp8_linear import attach_q8_fused

            groups = []
            for module in self.model.modules():
                if isinstance(module, V41Attention):
                    groups.append((module.wq_a, module.wkv))
                elif isinstance(module, V41SharedMLP):
                    groups.append((module.gate_proj, module.up_proj))
            attach_q8_fused(groups)
            from .native import attach_h16_gate, attach_h16_head

            for module in self.model.modules():
                if isinstance(module, V41Gate):
                    attach_h16_gate(module)
                elif isinstance(module, V41Compressor) and module.ratio == 2:
                    # Built before graph capture (the check syncs).
                    from .native import h16_exact

                    weight = torch.cat(
                        [module.wkv.weight.data, module.wgate.weight.data], 0
                    )
                    module._v41_h16 = h16_exact(weight, keep_plain=_BATCH) or False
            if not isinstance(self.lm_head, PPMissingLayer):
                attach_h16_head(self.lm_head)
            torch.accelerator.empty_cache()
        self.model.prepare_postload()

    def commit_speculative_state(self, accepted_counts):
        self.model.commit(accepted_counts)

    def pp_after_forward(self, req_ids, num_computed, num_prompt):
        self.model.pp_after_forward(req_ids, num_computed, num_prompt)

    def pp_before_forward(self, req_ids, num_computed):
        self.model.pp_before_forward(req_ids, num_computed)

    def restore_prefix(self, cached, prompt_ids):
        self.model.restore_prefix(cached, prompt_ids)

    def stateful_output_budget(self, max_tokens, prompt_tokens):
        return self.model.output_budget(max_tokens, prompt_tokens)

    def v41_set_batch(self, req_ids):
        self.model.v41_set_batch(req_ids)

    def v41_release(self, req_ids, done=None):
        self.model.v41_release(req_ids, done)

    def v41_pending(self):
        m = self.model
        if getattr(m, "_deferred", None) is not None:
            prefill, lengths, positions = m._selectors()
            return prefill[:1], positions[:1], lengths[:1]
        return m.pending_prefill, m.pending_position, m.pending_length

    def v41_begin_deferred(self):
        self.model.v41_begin_deferred()

    def v41_end_deferred(self):
        self.model.v41_end_deferred()

    def get_eagle3_default_aux_hidden_state_layers(self):
        return tuple(self.config.dspark_target_layer_ids)
