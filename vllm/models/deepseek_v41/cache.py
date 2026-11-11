# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 FP16 latent cache registration, with physical compression geometry."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from torch import nn

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v4.compressor import CompressorBackend
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


@dataclass
class V41CacheMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    is_prefilling: torch.Tensor
    block_size: int
    causal: bool = True
    # VLLM_V41_BATCH: the batch layout shared by every group of one step.
    # rows[t] is token t's request row (its block-table row), bases[t] the
    # index of that request's first token in this forward, prefill[r] the
    # per-request prefill flag; None for the single-request route.
    layout: "V41BatchLayout | None" = None


@dataclass
class V41BatchLayout:
    num_reqs: int
    num_tokens: int
    rows: torch.Tensor  # int32 [max_tokens]
    bases: torch.Tensor  # int32 [max_tokens]
    prefill: torch.Tensor  # bool [max_reqs]
    query_start_loc: torch.Tensor  # int32 [max_reqs + 1]
    query_start_loc_cpu: torch.Tensor


# Several requests per step (default off: the model admits exactly one).
BATCH = envs.VLLM_V41_BATCH == "1"


_HOST_RING = 16


class _LayoutBuffers:
    """Persistent device buffers of the batch layout (graph-stable addresses)."""

    def __init__(self, device, max_tokens, max_reqs):
        self.rows = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.bases = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.prefill = torch.zeros(max_reqs, dtype=torch.bool, device=device)
        # A ring of pinned host copies: a build's non-blocking upload must
        # finish before its host buffer is written again. Two were not enough:
        # the drafter builds again in the same step (and with PP the host runs
        # a step ahead of a stream waiting on the previous round), so the
        # synchronize below blocked the host on the GPU (py-spy, 2026-11-09).
        self.hosts = [
            (
                torch.zeros(max_tokens, dtype=torch.int32, device="cpu").pin_memory(),
                torch.zeros(max_tokens, dtype=torch.int32, device="cpu").pin_memory(),
                torch.zeros(max_reqs, dtype=torch.bool, device="cpu").pin_memory(),
                torch.cuda.Event(),
            )
            for _ in range(_HOST_RING)
        ]
        self.turn = 0
        self.captured: list = []
        self.token = None
        self.key = None
        self.layout = None

    def build(self, common):
        """Fill the buffers once per step (every group gets the same common)."""
        if self.token is common:
            return self.layout
        # Every KV group of a step gets its own common metadata, sharing the
        # step's query_start_loc_cpu and is_prefilling objects: build once per
        # step (each rebuild cost ~200 us of host time, 8 per step on the last
        # stage). The held references keep the ids from being reused.
        key = self.key
        if (
            key is not None
            and key[0] is common.query_start_loc_cpu
            and key[1] is common.is_prefilling
            and key[2] == (common.num_reqs, common.num_actual_tokens)
            and not common.is_dummy_batch
        ):
            self.token = common
            return self.layout
        starts = common.query_start_loc_cpu.numpy().astype(np.int64)
        reqs = common.num_reqs
        lens = np.diff(starts[: reqs + 1])
        tokens = int(starts[reqs])
        rows = np.repeat(np.arange(reqs, dtype=np.int32), lens)
        bases = np.repeat(starts[:reqs].astype(np.int32), lens)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            # A graph replays its upload from these host buffers: they are the
            # graph's own and never written again (no ring event either).
            rows_host = torch.zeros_like(self.hosts[0][0]).pin_memory()
            bases_host = torch.zeros_like(self.hosts[0][1]).pin_memory()
            flags = torch.zeros_like(self.hosts[0][2]).pin_memory()
            uploaded = None
            self.captured.append((rows_host, bases_host, flags))
        else:
            rows_host, bases_host, flags, uploaded = self.hosts[self.turn]
            self.turn = (self.turn + 1) % len(self.hosts)
            uploaded.synchronize()
        # Padding tokens (graph sizes above num_actual_tokens) keep row 0 and
        # base 0; their slot mapping is -1, so no kernel stores from them.
        rows_host.zero_()
        bases_host.zero_()
        rows_host[:tokens].copy_(torch.from_numpy(rows))
        bases_host[:tokens].copy_(torch.from_numpy(bases))
        self.rows.copy_(rows_host, non_blocking=True)
        self.bases.copy_(bases_host, non_blocking=True)
        flags.zero_()
        source = common.is_prefilling
        on_device = source is not None and source.device.type != "cpu"
        if common.is_dummy_batch:
            pass
        elif source is None:
            flags[:reqs] = True
        elif not on_device:
            flags[:reqs].copy_(source[:reqs])
        self.prefill.copy_(flags, non_blocking=True)
        if uploaded is not None:
            uploaded.record()
        if on_device and not common.is_dummy_batch:
            self.prefill[:reqs].copy_(source[:reqs])
        self.token = common
        self.key = (
            common.query_start_loc_cpu,
            common.is_prefilling,
            (common.num_reqs, common.num_actual_tokens),
        )
        self.layout = V41BatchLayout(
            reqs,
            tokens,
            self.rows,
            self.bases,
            self.prefill,
            common.query_start_loc,
            common.query_start_loc_cpu,
        )
        return self.layout


class V41CacheBuilder(AttentionMetadataBuilder):
    # The model admits one request, and exact M1..6 graphs retain the same
    # block/slot buffers. The device flag also supports a small prefill at
    # the same shape; no mixed-request geometry is admitted by build().
    # VLLM_V41_BATCH admits several requests through a shared batch layout.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    uses_physical_block_table = True

    _shared_flags: ClassVar[dict] = {}
    _layouts: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Every V4.1 group reads the same single-request prefill state. One
        # device flag (and its last host value) is shared by all builders.
        shared = V41CacheBuilder._shared_flags.get(self.device)
        if shared is None:
            shared = [torch.ones(1, dtype=torch.bool, device=self.device), None, None]
            V41CacheBuilder._shared_flags[self.device] = shared
        self._shared = shared
        self.prefill_flag = shared[0]
        self._layout = None
        if BATCH:
            layout = V41CacheBuilder._layouts.get(self.device)
            if layout is None:
                scheduler = self.vllm_config.scheduler_config
                layout = _LayoutBuffers(
                    self.device,
                    scheduler.max_num_batched_tokens,
                    scheduler.max_num_seqs,
                )
                V41CacheBuilder._layouts[self.device] = layout
            self._layout = layout

    def _set_flag(self, value, token):
        shared = self._shared
        if shared[1] is not None and shared[2] == value:
            return
        shared[0].fill_(value)
        shared[1], shared[2] = token, value

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        del common_prefix_len, fast_build
        common = common_attn_metadata
        if self._layout is not None:
            layout = self._layout.build(common)
            return V41CacheMetadata(
                common.block_table_tensor,
                common.slot_mapping,
                layout.prefill,
                self.kv_cache_spec.block_size,
                causal=common.causal,
                layout=layout,
            )
        if common.num_reqs != 1:
            raise ValueError("V4.1 FP16 currently supports exactly one request")
        # The common flag is normally CPU-backed. A graph must read one
        # stable device address that the builder updates before every replay,
        # rather than capturing a host copy or a dummy batch's constant flag.
        source = common.is_prefilling
        if common.is_dummy_batch:
            self._set_flag(False, common)
        elif source is None:
            self._set_flag(True, common)
        elif source.device.type == "cpu":
            self._set_flag(bool(source[0]), common)
        else:
            self.prefill_flag.copy_(source[:1])
            self._shared[1] = None
        return V41CacheMetadata(
            common.block_table_tensor,
            common.slot_mapping,
            self.prefill_flag,
            self.kv_cache_spec.block_size,
            causal=common.causal,
        )


class V41CacheBackend(CompressorBackend):
    supported_dtypes = [torch.float16]

    @staticmethod
    def get_name():
        return "DEEPSEEK_V41_FP16_LATENTS"

    @staticmethod
    def get_builder_cls():
        return V41CacheBuilder

    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 512]


class V41LatentCache(nn.Module, AttentionLayerBase):
    """One K=V vector per physical row, after the official FP4/FP8 QDQ."""

    def __init__(self, prefix, head_dim=512, ratio=1, sliding_window=None):
        super().__init__()
        self.prefix, self.head_dim, self.ratio = prefix, head_dim, ratio
        self.sliding_window = sliding_window
        self.kv_cache = torch.empty(0, dtype=torch.float16)
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context or head_dim not in (128, 512) or ratio not in (1, 2):
            raise ValueError("Invalid or duplicate V4.1 latent cache")
        context[prefix] = self

    def get_attn_backend(self):
        return V41CacheBackend

    def get_kv_cache_spec(self, vllm_config):
        block = vllm_config.cache_config.block_size
        if block % self.ratio:
            raise ValueError("V4.1 cache block must cover complete compression groups")
        kwargs = dict(
            block_size=block,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.float16,
            compress_ratio=self.ratio,
            # One page size across window, compressed and index pools. The
            # engine exposes padded physical-page strides to the gather ops.
            alignment=block * 512 * 2,
        )
        if self.sliding_window is not None:
            return SlidingWindowMLASpec(sliding_window=self.sliding_window, **kwargs)
        return MLAAttentionSpec(**kwargs)

    def metadata(self):
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return None
        return metadata[self.prefix]
