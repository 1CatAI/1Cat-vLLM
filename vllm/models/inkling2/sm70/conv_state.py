# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 short-convolution state for SM70 (fork KV-cache API).

Port of upstream `vllm/models/inkling/nvidia/sconv_swa_attn.py` and
`vllm/models/inkling2/nvidia/sconv.py`. Each decoder layer (and each loop
pass of a core layer) owns one `InklingConvState`: a sliding-window paged cache
(window = kernel size) that keeps every stream's conv input per position,
packed per KV head as `K | V | attn-out chunk | mlp-out chunk` (padded to a
power of two). It piggybacks on the KV-cache manager as a `SlidingWindowSpec`
with `head_size_v = 0`, so prefix caching and block freeing work unchanged.

Differences from upstream: `get_kv_cache_spec` (fork) instead of
`get_kv_cache_config`; the cache dtype is the model dtype (fp16 on V100); the
fused conv+mix kernels become `short_conv` followed by `attn_res` (upstream's
own SM12x fallback).
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
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, SlidingWindowSpec

from .ops import conv_mix2 as _cm2
from .ops.attn_res import attn_res
from .ops.fused_ar_mix import (
    FusedAR,
    can_fuse,
    conv_mix_cuda,
    fused_ar_mix_level,
    setup_fused_ar,
)
from .ops.sconv import short_conv
from .ops.sconv_meta import sconv_seq_metadata

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)

# VLLM_INKLING2_SM70_SCONVMETA=1 (default): the conv-state KV groups (6 of them)
# build identical seq_idx / query_start metadata from the same query_start_loc,
# so they share one pair of buffers and the kernel runs once per distinct batch
# shape instead of once per group per step. =0 keeps one buffer pair and one
# launch per group.
_SHARE_SEQ_META = os.environ.get("VLLM_INKLING2_SM70_SCONVMETA", "1") == "1"
_SEQ_META_SHARED: dict = {}

_FUSED_AR_MIX = fused_ar_mix_level()  # VLLM_INKLING2_SM70_FUSED_AR_MIX
_FUSED_AR: FusedAR | None = None  # TP4 push all-reduce buffers of the fused kernel
# VLLM_INKLING2_SM70_CONV_MIX_V2=1 (default): ops/conv_mix2.py instead of the first fused
# kernel: same arithmetic and wire protocol, the all-reduce stays inside the kernel for
# several new tokens per request too (verify step), code-warming pass during the peer wait.
_CONV_MIX_V2 = os.environ.get("VLLM_INKLING2_SM70_CONV_MIX_V2", "1") == "1"
# With several new tokens per request (verify step): 1 = all-reduce inside conv_mix2 (grid
# barrier for the in-batch taps), 0 = the fork's all-reduce launch, then conv_mix2 without it.
_CONV_MIX_V2_VERIFY_AR = (
    os.environ.get("VLLM_INKLING2_SM70_CONV_MIX_V2_VERIFY_AR", "1") == "1"
)
_FUSED_AR_INIT = False


def _ensure_fused_ar() -> None:
    """Collective (every TP rank constructs its conv states in lockstep): set up
    the fused kernel's own IPC all-reduce buffers once. Failure => stage 1."""
    global _FUSED_AR, _FUSED_AR_INIT
    if _FUSED_AR_INIT or _FUSED_AR_MIX < 2:
        return
    _FUSED_AR_INIT = True
    tp = get_tensor_model_parallel_world_size()
    if tp != 4:
        return
    try:
        from vllm.distributed import get_tp_group

        _FUSED_AR = setup_fused_ar(
            get_tensor_model_parallel_rank(), tp, get_tp_group().cpu_group
        )
    except Exception as e:  # noqa: BLE001 - unsupported topology: keep stage 1
        import logging

        logging.getLogger(__name__).warning("fused all-reduce disabled: %r", e)
        _FUSED_AR = None


@dataclass
class InklingSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks] physical blocks per req
    slot_mapping: torch.Tensor  # [T] int64 flat slot of each token (-1 => skip)
    seq_idx: torch.Tensor  # [T] int32 token -> batch request
    query_start: torch.Tensor  # [T] int32 first x-row of each token's request
    max_query_len: int = 0  # 1 => pure decode batch (no token has an in-batch tap)


class InklingSconvMetadataBuilder(AttentionMetadataBuilder[InklingSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        shared = (
            _SEQ_META_SHARED.setdefault((str(device), max_num_tokens), {})
            if _SHARE_SEQ_META
            else {}
        )
        if "seq_idx" not in shared:
            shared["seq_idx"] = torch.empty(
                max_num_tokens, dtype=torch.int32, device=device
            )
            shared["query_start"] = torch.empty(
                max_num_tokens, dtype=torch.int32, device=device
            )
            shared["key"] = None
        self._shared_meta = shared if _SHARE_SEQ_META else None
        self.seq_idx_buffer = shared["seq_idx"]
        self.query_start_buffer = shared["query_start"]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingSconvMetadata:
        cm = common_attn_metadata
        num_actual_tokens = int(cm.query_start_loc_cpu[cm.num_reqs])
        num_padded_tokens = cm.slot_mapping.shape[0]
        shared = self._shared_meta
        key = None
        if shared is not None:
            key = (
                num_actual_tokens,
                num_padded_tokens,
                tuple(cm.query_start_loc_cpu[: cm.num_reqs + 1].tolist()),
            )
        if shared is None or shared["key"] != key:
            sconv_seq_metadata(
                cm.query_start_loc,
                cm.num_reqs,
                num_actual_tokens,
                self.seq_idx_buffer,
                self.query_start_buffer,
                num_padded_tokens,
            )
            if shared is not None:
                shared["key"] = key
        return InklingSconvMetadata(
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
            max_query_len=int(cm.max_query_len),
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> InklingSconvMetadata:
        return self.build(0, common_attn_metadata)


class InklingSconvBackend(AttentionBackend):
    """Metadata-only backend: the convs run out-of-band in the model."""

    @staticmethod
    def get_name() -> str:
        return "INKLING_SCONV_SWA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("InklingSconvBackend has no attention impl")

    @staticmethod
    def get_builder_cls() -> type[InklingSconvMetadataBuilder]:
        return InklingSconvMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # [num_blocks, H, N, Dpack]: head-major pages, as the conv kernels read.
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)


class InklingConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer (per loop pass) owner of one conv-state cache."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
        replicate_hidden: bool = True,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= num_kv_heads, (
            f"sconv cache supports tp_size <= num_kv_heads ({num_kv_heads})"
        )
        self.num_kv_heads = num_kv_heads // tp_size
        hidden_per_head = hidden_size // (
            self.num_kv_heads if replicate_hidden else num_kv_heads
        )
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),
            (head_dim, head_dim),
            (2 * head_dim, hidden_per_head),
            (2 * head_dim + hidden_per_head, hidden_per_head),
        )
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        _ensure_fused_ar()

    def forward(self): ...

    @property
    def cache_block_size(self) -> int:
        if self.kv_cache.numel() > 0:
            return self.kv_cache.shape[2]
        return self.block_size

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingSconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> SlidingWindowSpec:
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,  # all 4 streams packed into head_size
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )


def conv_metadata(owner: InklingConvState) -> InklingSconvMetadata | None:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or owner.kv_cache.numel() == 0:
        return None
    return cast(InklingSconvMetadata | None, attn_metadata.get(owner.prefix))


class Inkling2ShortConv(nn.Module):
    """Causal depthwise conv (+ residual, scaled) over one conv-state stream.

    Static: `weight` of shape `(dim, 1, W)`. Dynamic (global-layer K/V):
    per-token taps `static_w (W, dim) + weight_proj(dynamic_input)` per head.
    The K/V convs only hold weights; the attention prologue kernel runs them.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        owner: InklingConvState,
        stream: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.stream = stream
        self.dynamic = dynamic_input_size is not None
        self.tp_rank = get_tensor_model_parallel_rank()
        if dynamic_input_size is not None:
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, dim), requires_grad=False
            )
            self.weight_proj = nn.Parameter(
                torch.empty(num_heads * kernel_size, dynamic_input_size),
                requires_grad=False,
            )
            set_weight_attrs(self.static_w, {"weight_loader": self._load_dim1})
            set_weight_attrs(self.weight_proj, {"weight_loader": self._load_dim0})
        else:
            self.weight = nn.Parameter(
                torch.empty(dim, 1, kernel_size), requires_grad=False
            )
            set_weight_attrs(self.weight, {"weight_loader": self._load_dim0})

    def _load(self, param: nn.Parameter, loaded: torch.Tensor, dim: int) -> None:
        size = param.shape[dim]
        if loaded.shape[dim] != size:
            loaded = loaded.narrow(dim, self.tp_rank * size, size)
        param.data.copy_(loaded)

    def _load_dim0(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 0)

    def _load_dim1(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 1)

    @property
    def conv_weight(self) -> torch.Tensor:
        """Static `(dim, W)` taps, or `(W, dim)` static part of dynamic taps."""
        return self.static_w if self.dynamic else self.weight.squeeze(1)

    def dynamic_taps(self, x: torch.Tensor) -> torch.Tensor | None:
        """Per-token tap deltas `(T, num_heads * W)` projected from `x`."""
        if not self.dynamic:
            return None
        return nn.functional.linear(x, self.weight_proj)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        owner: InklingConvState | None = None,
    ) -> torch.Tensor:
        """Static conv of `x`; with `addend`, returns `(residual + addend)
        + conv(x)` accumulated in fp32 (written to `out` if given)."""
        assert not self.dynamic
        if owner is None:
            owner = self.owner
        m = conv_metadata(owner)
        if m is None:
            # No conv state (profile run): skip the conv, keep the residual sum.
            if addend is not None:
                acc = (
                    addend.float()
                    if residual is None
                    else residual.float() + addend.float()
                )
                x = (acc + x.to(addend.dtype).float()).to(addend.dtype)
            return x if out is None else out.copy_(x)
        offset, width = owner.stream_ranges[self.stream]
        return short_conv(
            x,
            self.conv_weight,
            owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            addend,
            residual,
            out,
        )

    def conv_mix(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
        stack: torch.Tensor,
        last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: nn.Module | None,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        owner: InklingConvState | None = None,
        reduce: bool = False,
    ) -> torch.Tensor:
        """With `reduce`, `x` is the TP partial sum and is all-reduced first
        (inside the fused kernel at TP4 decode). Convolve `x` into `out` like `forward`, then return the AttnRes
        mixes of `[*stack, (last +) out]` with `queries` through `norm` (the
        fp32 mixes without it). Upstream's SM12x fallback of the fused
        conv_mix kernel."""
        if _FUSED_AR_MIX >= 1 and not self.dynamic and can_fuse(x, stack, queries):
            ow = self.owner if owner is None else owner
            m = conv_metadata(ow)
            if m is not None and out is not None:
                offset, width = ow.stream_ranges[self.stream]
                ar = None
                if reduce:
                    if _FUSED_AR is not None and (
                        m.max_query_len == 1
                        or (_CONV_MIX_V2 and _CONV_MIX_V2_VERIFY_AR)
                    ):
                        ar = _FUSED_AR  # all-reduce inside the kernel (decode)
                    else:
                        x = tensor_model_parallel_all_reduce(x)
                if _CONV_MIX_V2:
                    return _cm2.conv_mix2_cuda(
                        x,
                        self.conv_weight,
                        ow.kv_cache,
                        positions,
                        m.block_table,
                        m.seq_idx,
                        m.slot_mapping,
                        m.query_start,
                        offset,
                        width,
                        out,
                        residual,
                        addend,
                        stack,
                        last,
                        queries,
                        None if norm is None else norm.weight,
                        0.0 if norm is None else norm.eps,
                        ar=ar,
                        inbatch=m.max_query_len != 1,
                    )
                return conv_mix_cuda(
                    x,
                    self.conv_weight,
                    ow.kv_cache,
                    positions,
                    m.block_table,
                    m.seq_idx,
                    m.slot_mapping,
                    m.query_start,
                    offset,
                    width,
                    out,
                    residual,
                    addend,
                    stack,
                    last,
                    queries,
                    None if norm is None else norm.weight,
                    0.0 if norm is None else norm.eps,
                    ar=ar,
                )
        if reduce:
            x = tensor_model_parallel_all_reduce(x)
        self.forward(
            x, positions, addend=addend, residual=residual, out=out, owner=owner
        )
        return attn_res(
            stack,
            out if last is None else last,
            None if last is None else out,
            queries,
            None if norm is None else norm.weight,
            0.0 if norm is None else norm.eps,
            True,
        )
