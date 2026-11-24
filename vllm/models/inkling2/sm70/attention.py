# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 attention for SM70: relative-bias GQA with short-conv K/V,
L2-clipped q/k/r, gated output and per-token attention sinks.

Port of upstream `vllm/models/inkling2/nvidia/attention.py`:
- KV cache spec through the fork's `get_kv_cache_spec` (full or sliding
  window); fp16 KV `[num_blocks, block_size, H_kv, 2 * D]` (K and V of a
  head side by side, as the prologue kernel writes them).
- The attention prologue (q/r clip, sheared relative logits, K/V short convs,
  KV and conv-state writes) is upstream's Triton kernel; attention runs
  upstream's SM90 Triton `rel_attention` (small tiles for SM70) over every
  request, then `gate_sink` applies the output gate and the sinks.
- Projections are cuBLAS GEMMs (`F.linear`).
"""

import os
from dataclasses import dataclass
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from ..configs import Inkling2ModelConfig
from .conv_state import (
    K_STREAM,
    V_STREAM,
    Inkling2ShortConv,
    InklingConvState,
    conv_metadata,
)
from .ops import decode_attention2 as _da2
from .ops import dense_gemv as _dg
from .ops.attention import (
    gate_sink,
    max_decode_query_len,
    plan_decode_attention,
    rel_decode_attention,
)
from .ops.decode_attention import (
    decode_attention,
    kv_conv_prologue,
    plan_decode_splits,
    prepare_decode_attention,
    supports_decode_attention,
)
from .ops.prefill_attention import (
    clip_qr,
    prefill_attention,
    prepare_prefill_attention,
    supports_prefill_attention,
)
from .ops.rel_attention import MODE_OUT_LSE, rel_attention
from .ops.sconv import attention_prologue, shift_rel_proj

# One-launch CUDA decode attention (q/r clip, relative bias, split-KV, gate and
# sink) for batches of at most `_DECODE_ATTN_MAX_Q` query tokens per request.
_DECODE_ATTN = os.environ.get("VLLM_INKLING2_SM70_DECODE_ATTN", "1") == "1"
_DECODE_ATTN_MAX_Q = 8

# Second-generation decode attention (ops/decode_attention2.py): the K/V prologue and
# the attention in one short-code launch. auto (default) = sliding-window layers always,
# global layers when a request brings 2..8 tokens (verify step); 1 = every layer and length
# (best below ~25K context: global layers 18 vs 47 us at 1.3K, but 261 vs 166 us at 70K for
# plain decode); local = sliding-window layers only; 0 = the first-generation path
# (kv_conv_prologue + decode_attention).
_ATTN_V2 = os.environ.get("VLLM_INKLING2_SM70_ATTN_V2", "auto")
# VLLM_INKLING2_SM70_ATTN2_GLOBAL1=1: plain decode (1 token) on global layers also uses attention v2 (auto keeps the old
# kernel there: v2 lost at long context; with two CTAs per SM, ops/decode_attention2.py ATTN2_2CTA, it may not)
_ATTN2_GLOBAL1 = os.environ.get("VLLM_INKLING2_SM70_ATTN2_GLOBAL1", "0") == "1"
logger = init_logger(__name__)
_ATTN_V2_SEEN: set = set()

# Tensor-core CUDA flash attention (relative bias, window, gate and sink fused) for
# batches the decode kernel does not take (query length > 8): ops/prefill_attention.py.
_PREFILL_ATTN = os.environ.get("VLLM_INKLING2_SM70_PREFILL_ATTN", "1") == "1"


@dataclass
class Inkling2AttentionMetadata(AttentionMetadata):
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    query_start_loc: torch.Tensor  # [num_reqs + 1] int32
    seq_lens: torch.Tensor  # [num_reqs] int32 (includes the new tokens)
    block_table: torch.Tensor  # [num_reqs, max_blocks] int32
    slot_mapping: torch.Tensor  # [T] int64


class Inkling2AttentionMetadataBuilder(
    AttentionMetadataBuilder[Inkling2AttentionMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2AttentionMetadata:
        cm = common_attn_metadata
        return Inkling2AttentionMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Inkling2AttentionMetadata:
        return self.build(0, common_attn_metadata)


class Inkling2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "INKLING2_SM70"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("Inkling2 attention runs in the model layer")

    @staticmethod
    def get_builder_cls() -> type[Inkling2AttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)


def _kv_cache_spec(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> KVCacheSpec:
    dtype = vllm_config.model_config.dtype
    block_size = vllm_config.cache_config.block_size
    if sliding_window is not None:
        return SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=dtype,
            sliding_window=sliding_window,
        )
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=dtype,
    )


def _register(prefix: str, module: nn.Module) -> None:
    forward_context = (
        get_current_vllm_config().compilation_config.static_forward_context
    )
    if prefix in forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    forward_context[prefix] = module


class Inkling2AttentionPassCache(nn.Module, AttentionLayerBase):
    """The KV cache of one extra loop pass of a looped `Inkling2Attention`."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        _register(prefix, self)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _kv_cache_spec(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


class Inkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling2 attention for one decoder layer (see the module docstring).

    Local layers project q/k/v from one input; full layers project the query
    side (q, r, gate, sink) from `x_q`, k from `x_k` and v from `x_v`, with
    dynamic K/V convolutions and log-scaled q/r. A loop core layer runs once
    per pass, each pass with its own KV cache (`pass_caches[p - 1]`) and conv
    state (`conv_states[p]`).
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
        pass_conv_states: tuple[InklingConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # [q | r | gate | sink | k | v] per rank.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        self.q_side = sum(self.split_sizes[:4])
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim), requires_grad=False
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel), requires_grad=False
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleList(
            Inkling2AttentionPassCache(
                f"{prefix}.pass{p}",
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for p in range(1, 1 if config.loop_share_cache else len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype not in ("auto", "float16", "half"):
            raise ValueError(
                f"Inkling2 on SM70 supports an fp16 KV cache only, got {cache_dtype}"
            )
        self.max_decode_query_len = max_decode_query_len(
            self.num_heads, self.num_kv_heads
        )
        self.rel_proj_shifted: torch.Tensor | None = None
        self.use_decode_attn = _DECODE_ATTN and supports_decode_attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.d_rel,
            vllm_config.model_config.dtype,
            _DECODE_ATTN_MAX_Q,
        )
        self.use_attn_v2 = (
            self.use_decode_attn
            and (_ATTN_V2 in ("1", "auto") or (_ATTN_V2 == "local" and is_local))
            and _da2.supports(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.d_rel,
                vllm_config.model_config.dtype,
            )
        )
        self.rel_proj_t: torch.Tensor | None = None
        self.use_prefill_attn = _PREFILL_ATTN and supports_prefill_attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.d_rel,
            vllm_config.model_config.dtype,
        )
        self.kv_cache = torch.tensor([])
        _register(prefix, self)

    def inkling_post_load(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)
        if self.use_prefill_attn:
            proj = self.rel_logits_proj
            if (
                proj.dtype != torch.float16
                or not proj.is_contiguous()
                or proj.shape[0] != 16
                or proj.shape[1] % 16
            ):
                self.use_prefill_attn = False
            else:
                prepare_prefill_attention()
        if self.use_decode_attn:
            params = (self.rel_logits_proj, self.q_scale, self.r_scale)
            if any(p.dtype != torch.float16 for p in params):
                self.use_decode_attn = False
            else:
                prepare_decode_attention(self.rel_logits_proj.device)
        if self.use_attn_v2 and self.use_decode_attn:
            # (extent, d_rel): the 16 projections of one distance in one 32 B read
            self.rel_proj_t = self.rel_logits_proj.t().contiguous()
            _da2.prepare(self.rel_logits_proj.device)
        else:
            self.use_attn_v2 = False

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _kv_cache_spec(
            vllm_config,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window if self.is_local else None,
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, torch.Tensor]:
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    def _project_fast(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, taps: bool
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor | None] | None:
        """M <= 16: qkvr (+ the global-layer K/V dynamic-tap projections) in one
        fused small-M GEMV launch. None when the inputs do not qualify."""
        weight = self.qkvr.weight
        xs = (x_q, x_k, x_v)
        if not all(_dg.usable(x, weight) and _dg.aligned(x) for x in xs):
            return None
        m = x_q.shape[0]
        out = torch.empty((m, weight.shape[0]), dtype=torch.float16, device=x_q.device)
        same = x_k is x_q and x_v is x_q
        if same:
            jobs = [(weight, x_q, out)]
        else:
            kv = self.split_sizes[4]
            k_end = self.q_side + kv
            jobs = [
                (weight[: self.q_side], x_q, out[:, : self.q_side]),
                (weight[self.q_side : k_end], x_k, out[:, self.q_side : k_end]),
                (weight[k_end : k_end + kv], x_v, out[:, k_end : k_end + kv]),
            ]
        k_dyn = v_dyn = None
        if taps and self.k_sconv.dynamic:
            kp, vp = self.k_sconv.weight_proj, self.v_sconv.weight_proj
            if not (_dg.usable(x_k, kp) and _dg.usable(x_v, vp)):
                return None
            k_dyn = torch.empty(
                (m, kp.shape[0]), dtype=torch.float16, device=x_q.device
            )
            v_dyn = torch.empty(
                (m, vp.shape[0]), dtype=torch.float16, device=x_q.device
            )
            jobs += [(kp, x_k, k_dyn), (vp, x_v, v_dyn)]
        _dg.dense_gemv_multi(jobs)
        return list(out.split(self.split_sizes, dim=-1)), k_dyn, v_dyn

    def _project(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> list[torch.Tensor]:
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = nn.functional.linear(x_q, weight)
            return list(projected.split(self.split_sizes, dim=-1))
        kv_size = self.split_sizes[4]
        k_end = self.q_side + kv_size
        projected = nn.functional.linear(x_q, weight[: self.q_side])
        k = nn.functional.linear(x_k, weight[self.q_side : k_end])
        v = nn.functional.linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.split_sizes[:4], dim=-1), k, v]

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [nb, bs, H, 2D] -> K = [..., :D], V = [..., D:] (same strides).
        return kv_cache[..., : self.head_dim], kv_cache[..., self.head_dim :]

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """Return this rank's `(T, hidden)` partial output (to be
        all-reduced over TP), written to `out` if given."""
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        fast = (
            self._project_fast(x_q, x_k, x_v, taps=True)
            if _dg.ENABLED and x_q.shape[0] <= _dg.MAX_M
            else None
        )
        if fast is not None:
            parts, k_dyn, v_dyn = fast
            q, r, gate, sinks, k, v = parts
            if k_dyn is None:
                k_dyn = self.k_sconv.dynamic_taps(x_k)
                v_dyn = self.v_sconv.dynamic_taps(x_v)
        else:
            q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
            k_dyn = self.k_sconv.dynamic_taps(x_k)
            v_dyn = self.v_sconv.dynamic_taps(x_v)
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            k_dyn,
            v_dyn,
            tau,
            output,
            pass_idx,
        )
        wo = self.wo_ud.weight
        if (
            _dg.ENABLED
            and output.shape[0] <= _dg.MAX_M
            and _dg.usable(output, wo)
            and _dg.aligned(output)
            and (out is None or (out.dtype == torch.float16 and out.stride(-1) == 1))
        ):
            return _dg.dense_gemv(output, wo, out)
        result = nn.functional.linear(output, self.wo_ud.weight)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        pass_idx: int = 0,
    ) -> None:
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        attn_metadata = get_forward_context().attn_metadata
        layer_name, kv_cache = self._pass_cache(pass_idx)
        conv_state = self.conv_states[pass_idx]
        md = None
        if isinstance(attn_metadata, dict) and kv_cache.numel() > 0:
            md = cast(Inkling2AttentionMetadata, attn_metadata[layer_name])
        if (
            md is not None
            and self.use_decode_attn
            and md.max_query_len <= _DECODE_ATTN_MAX_Q
        ):
            self._decode_attention(
                positions,
                q,
                r,
                gate,
                sinks,
                k,
                v,
                k_dynamic,
                v_dynamic,
                tau,
                output,
                md,
                kv_cache,
                conv_state,
            )
            return
        if (
            md is not None
            and self.use_prefill_attn
            and md.max_query_len
            > (
                _DECODE_ATTN_MAX_Q
                if self.use_decode_attn
                else self.max_decode_query_len
            )
        ):
            self._prefill_attention(
                positions,
                q,
                r,
                gate,
                sinks,
                k,
                v,
                k_dynamic,
                v_dynamic,
                tau,
                output,
                md,
                kv_cache,
                conv_state,
            )
            return
        out = torch.zeros(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        lse = torch.zeros(
            (self.num_heads, num_tokens), dtype=torch.float32, device=q.device
        )
        if md is not None:
            conv_md = conv_metadata(conv_state)
            assert conv_md is not None and self.rel_proj_shifted is not None
            key_cache, value_cache = self._split_kv_cache(kv_cache)
            clipped_q = q.new_empty(q.shape)
            rel_logits = q.new_empty(
                (num_tokens + 128, self.num_heads, self.rel_extent + 256)
            )
            attention_prologue(
                q,
                r.view(num_tokens, self.num_heads, self.d_rel),
                k,
                v,
                self.q_scale,
                self.r_scale,
                tau,
                self.rel_proj_shifted,
                self.k_sconv.conv_weight,
                self.v_sconv.conv_weight,
                k_dynamic,
                v_dynamic,
                conv_state.kv_cache,
                positions,
                conv_md.block_table,
                conv_md.seq_idx,
                conv_md.slot_mapping,
                conv_md.query_start,
                key_cache,
                value_cache,
                md.slot_mapping,
                clipped_q,
                rel_logits,
                md.query_start_loc,
                md.seq_lens,
                clip_threshold=self.qk_clip,
                rel_threshold=self.rel_clip,
                norm_eps=self.eps,
            )
            window = self.sliding_window if self.is_local else 0
            if md.max_query_len <= self.max_decode_query_len:
                # Decode / spec-decode verify: split-KV partials + gated merge.
                rel_decode_attention(
                    clipped_q,
                    key_cache,
                    value_cache,
                    rel_logits,
                    gate,
                    sinks,
                    output,
                    block_table=md.block_table[: md.num_reqs],
                    cache_seqlens=md.seq_lens[: md.num_reqs],
                    cu_seqlens_q=md.query_start_loc[: md.num_reqs + 1],
                    max_query_len=md.max_query_len,
                    num_actual_tokens=md.num_actual_tokens,
                    softmax_scale=self.scaling,
                    window=window,
                    num_splits=plan_decode_attention(
                        self.is_local, md.num_reqs, self.num_kv_heads, self.rel_extent
                    ),
                )
                return
            rel_attention(
                clipped_q,
                key_cache,
                value_cache,
                out,
                lse,
                rel_logits,
                block_table=md.block_table[: md.num_reqs],
                cache_seqlens=md.seq_lens[: md.num_reqs],
                cu_seqlens_q=md.query_start_loc[: md.num_reqs + 1],
                max_query_len=md.max_query_len,
                softmax_scale=self.scaling,
                window=window,
                num_splits=1,
                mode=MODE_OUT_LSE,
            )
        gate_sink(out, gate, lse.t(), sinks, output)

    def _prefill_attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        md: Inkling2AttentionMetadata,
        kv_cache: torch.Tensor,
        conv_state: InklingConvState,
    ) -> None:
        """Prefill / chunked-prefill batches: K/V short convs and cache write,
        q/r clip, then one tensor-core flash-attention kernel that computes the
        relative bias and folds in the gate and sink (see
        `ops/prefill_attention.py`)."""
        num_tokens = q.shape[0]
        conv_md = conv_metadata(conv_state)
        assert conv_md is not None
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        kv_conv_prologue(
            k,
            v,
            self.k_sconv.conv_weight,
            self.v_sconv.conv_weight,
            k_dynamic,
            v_dynamic,
            conv_state.kv_cache,
            positions,
            conv_md.block_table,
            conv_md.seq_idx,
            conv_md.slot_mapping,
            conv_md.query_start,
            key_cache,
            value_cache,
            md.slot_mapping,
            clip_threshold=self.qk_clip,
            norm_eps=self.eps,
        )
        clipped_q, clipped_r = clip_qr(
            q,
            r.view(num_tokens, self.num_heads, self.d_rel),
            self.q_scale,
            self.r_scale,
            tau,
            clip_threshold=self.qk_clip,
            rel_threshold=self.rel_clip,
        )
        prefill_attention(
            clipped_q,
            clipped_r,
            self.rel_logits_proj,
            kv_cache,
            gate,
            sinks,
            output,
            block_table=md.block_table[: md.num_reqs],
            seq_lens=md.seq_lens[: md.num_reqs],
            cu_seqlens_q=md.query_start_loc[: md.num_reqs + 1],
            max_query_len=md.max_query_len,
            num_actual_tokens=md.num_actual_tokens,
            softmax_scale=self.scaling,
            window=self.sliding_window if self.is_local else 0,
        )

    def _decode_attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        md: Inkling2AttentionMetadata,
        kv_cache: torch.Tensor,
        conv_state: InklingConvState,
    ) -> None:
        """Decode / verify batches: K/V short convs and cache write, then one
        fused kernel (see `ops/decode_attention.py`)."""
        conv_md = conv_metadata(conv_state)
        assert conv_md is not None
        if self.use_attn_v2 and (
            self.is_local or _ATTN_V2 == "1" or md.max_query_len > 1 or _ATTN2_GLOBAL1
        ):
            splits = _da2.plan_splits(
                md.num_reqs, self.num_kv_heads, md.max_query_len, self.is_local
            )
            k_weight, v_weight = self.k_sconv.conv_weight, self.v_sconv.conv_weight
            if splits >= 1 and _da2.usable(
                q,
                k,
                v,
                k_weight,
                v_weight,
                k_dynamic,
                v_dynamic,
                kv_cache,
                conv_state.kv_cache,
                positions,
                conv_md.slot_mapping,
                md.slot_mapping,
            ):
                shape = (self.is_local, md.num_reqs, md.max_query_len)
                if shape not in _ATTN_V2_SEEN:
                    _ATTN_V2_SEEN.add(shape)
                    logger.info(
                        "Inkling2 SM70 decode attention v2: %s layers, %d request(s), query len <= %d, %d splits",
                        "local" if self.is_local else "global",
                        md.num_reqs,
                        md.max_query_len,
                        splits,
                    )
                _da2.decode_attention2(
                    q,
                    r.view(q.shape[0], self.num_heads, self.d_rel),
                    self.q_scale,
                    self.r_scale,
                    tau,
                    self.rel_proj_t,
                    kv_cache,
                    gate,
                    sinks,
                    output,
                    k=k,
                    v=v,
                    k_weight=k_weight,
                    v_weight=v_weight,
                    k_dynamic=k_dynamic,
                    v_dynamic=v_dynamic,
                    conv_cache=conv_state.kv_cache,
                    positions=positions,
                    conv_block_table=conv_md.block_table,
                    seq_idx=conv_md.seq_idx,
                    conv_slot_mapping=conv_md.slot_mapping,
                    query_start=conv_md.query_start,
                    kv_slot_mapping=md.slot_mapping,
                    block_table=md.block_table[: md.num_reqs],
                    seq_lens=md.seq_lens[: md.num_reqs],
                    cu_seqlens_q=md.query_start_loc[: md.num_reqs + 1],
                    max_query_len=md.max_query_len,
                    num_actual_tokens=md.num_actual_tokens,
                    softmax_scale=self.scaling,
                    window=self.sliding_window if self.is_local else 0,
                    num_splits=splits,
                    qk_clip=self.qk_clip,
                    rel_clip=self.rel_clip,
                    norm_eps=self.eps,
                    eps=1e-6,
                )
                return
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        kv_conv_prologue(
            k,
            v,
            self.k_sconv.conv_weight,
            self.v_sconv.conv_weight,
            k_dynamic,
            v_dynamic,
            conv_state.kv_cache,
            positions,
            conv_md.block_table,
            conv_md.seq_idx,
            conv_md.slot_mapping,
            conv_md.query_start,
            key_cache,
            value_cache,
            md.slot_mapping,
            clip_threshold=self.qk_clip,
            norm_eps=self.eps,
        )
        decode_attention(
            q,
            r.view(q.shape[0], self.num_heads, self.d_rel),
            self.q_scale,
            self.r_scale,
            tau,
            self.rel_logits_proj,
            kv_cache,
            gate,
            sinks,
            output,
            block_table=md.block_table[: md.num_reqs],
            seq_lens=md.seq_lens[: md.num_reqs],
            cu_seqlens_q=md.query_start_loc[: md.num_reqs + 1],
            max_query_len=md.max_query_len,
            num_actual_tokens=md.num_actual_tokens,
            softmax_scale=self.scaling,
            window=self.sliding_window if self.is_local else 0,
            num_splits=plan_decode_splits(
                self.is_local, md.num_reqs, self.num_kv_heads, md.max_query_len
            ),
            qk_clip=self.qk_clip,
            rel_clip=self.rel_clip,
            eps=1e-6,
        )
