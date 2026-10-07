# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 SM70 sparse MLA: KV-cache specs, cache layers, backends + metadata builders, and the
sparse attention impl over FP16 records (PORT_DESIGN §3.3, A3; owner L-ATTN).

Cache layers per attention layer ``P = model.layers.{i}.attn`` (§3.3); ``B`` = ``cache_config.block_size``
(the SWA backend prefers 256):

=====================================  ==============================================  ===========
name                                   spec                                            row
=====================================  ==============================================  ===========
``P.swa_cache`` (every layer)          SlidingWindowMLASpec(B/2 tok/block, window 128)  512 x fp16
``P`` (kv sources + mirror)            MLAAttentionSpec(B tok/block, ratio r)          512 x fp16
``P.indexer.k_cache`` (idem)           MLAAttentionSpec(B tok/block, ratio r)          128 x fp16
``P.compressor.state_cache`` (r = 2)   SlidingWindowMLASpec(B/8 tok/block, window 2)   1024 x fp32
=====================================  ==============================================  ===========

Block sizes: every block id of vLLM's single pool reserves one page in every tensor of the stage
(``kv_cache_utils._deepseek_v41_tensor_plan``), so the windowed groups use the largest blocks whose
page equals the ratio-2 compressed page (B/2 x 1 KiB = B/8 x 4 KiB = 128 KiB at B = 256): fewer
block ids per prefill chunk and no extra tensor on stages holding ratio-2 sources. Sweep
(``scratch/ds41/attn/kv_sweep.py``, 4096-token chunks, 256K context): PP3 pool 509 / 764 / 626 MiB per
stage vs 928 / 1161 / 571 MiB with 64-token SWA and 8-token state blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import (
    CKV_CACHE_DTYPE_STR,
    CKV_RECORD_DIM,
    HEAD_DIM,
    IDX_TOPK,
    IK_CACHE_DTYPE_STR,
    IK_RECORD_DIM,
    KV_MODEL_VERSION,
    KV_RECORD_DTYPE,
    N_HEADS,
    SWA_RECORD_DIM,
    WINDOW,
)
from vllm.models.deepseek_v41.sm70.sparse_kernels import (
    logical_to_rows,
    sparse_attention,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

STATE_ROW_DIM = 2 * CKV_RECORD_DIM   # FP32 [kv 512 | score 512] of one token (ratio-2 pairing)
STATE_WINDOW = 2
PREFERRED_BLOCK_SIZE = 256     # MLA (compressed / index-K) tokens per block


def swa_block_size(block_size: int) -> int:
    _check_block(block_size)
    return block_size // 2


def state_block_size(block_size: int) -> int:
    _check_block(block_size)
    return block_size // 8


def swa_cache_spec(block_size: int) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=swa_block_size(block_size), num_kv_heads=1, head_size=SWA_RECORD_DIM, dtype=KV_RECORD_DTYPE,
        sliding_window=WINDOW, compress_ratio=1, model_version=KV_MODEL_VERSION)


def compressed_cache_spec(block_size: int, compress_ratio: int) -> MLAAttentionSpec:
    _check_mla_block(block_size, compress_ratio)
    return MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=CKV_RECORD_DIM, dtype=KV_RECORD_DTYPE,
        compress_ratio=compress_ratio, cache_dtype_str=CKV_CACHE_DTYPE_STR, model_version=KV_MODEL_VERSION)


def index_k_cache_spec(block_size: int, compress_ratio: int) -> MLAAttentionSpec:
    _check_mla_block(block_size, compress_ratio)
    return MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=IK_RECORD_DIM, dtype=KV_RECORD_DTYPE,
        compress_ratio=compress_ratio, cache_dtype_str=IK_CACHE_DTYPE_STR, model_version=KV_MODEL_VERSION)


def state_cache_spec(block_size: int) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=state_block_size(block_size), num_kv_heads=1, head_size=STATE_ROW_DIM, dtype=torch.float32,
        sliding_window=STATE_WINDOW, compress_ratio=1, model_version=KV_MODEL_VERSION)


def _check_block(block_size: int) -> None:
    if block_size < 64 or block_size % 8:
        raise ValueError(f"DeepSeek-V4.1 needs cache block_size >= 64 and a multiple of 8, got {block_size}")


def _check_mla_block(block_size: int, compress_ratio: int) -> None:
    _check_block(block_size)
    if compress_ratio not in (1, 2):
        raise ValueError(f"DeepSeek-V4.1 compressed caches have ratio 1 or 2, got {compress_ratio}")
    if block_size <= 0 or block_size % compress_ratio:
        raise ValueError(f"block_size {block_size} must be a positive multiple of compress_ratio {compress_ratio}")


# =====================================================================================
# metadata (built once per KV-cache group and step; read by every layer of the group)
# =====================================================================================


@dataclass
class DS41BatchMetadata:
    """Fields every V4.1 cache metadata carries (token order = the runner's flattened order)."""
    num_reqs: int
    num_actual_tokens: int                 # T: rows 0..T-1 are real tokens
    query_start_loc: torch.Tensor          # [R+1] int32 (GPU)
    query_start_loc_cpu: np.ndarray        # [R+1] int64
    seq_lens_cpu: np.ndarray               # [R] int64 (exact: synchronous scheduling, no async spec decode)
    token_to_req_indices: torch.Tensor     # [T] int32 (GPU)
    positions: torch.Tensor                # [T] int64 (GPU), recomputed from seq_lens/query_start_loc
    positions_cpu: np.ndarray              # [T] int64
    block_table: torch.Tensor              # [R, max_blocks] int32: physical block ids of this group
    block_size: int                        # tokens per block of this group
    slot_mapping: torch.Tensor             # [T] int64: row written by token t in this cache, -1 = none


@dataclass
class DS41SWAMetadata(DS41BatchMetadata):
    window_slots: torch.Tensor = None      # [T, WINDOW] int64: rows of positions p-127..p (ascending), -1 pad


@dataclass
class DS41CompressedMetadata(DS41BatchMetadata):
    compress_ratio: int = 1
    storage_block_size: int = 0            # compressed rows per block = block_size // compress_ratio
    num_latents: int = 0                   # N: groups completed this step (ratio 1: N = T)
    latent_token_idx: torch.Tensor = None  # [N] int64: token completing each group, ascending
    latent_pos: torch.Tensor = None        # [N] int64: position of each group's FIRST token
    latent_slots: torch.Tensor = None      # [N] int64: compressed row of each latent
    num_visible: torch.Tensor = None       # [T] int64: compressed entries visible to token t = (p+1)//r
    max_visible: int = 0                   # max over the batch (host int, for workspace sizing)


@dataclass
class DS41StateMetadata(DS41BatchMetadata):
    prev_slot: torch.Tensor = None         # [T] int64: state row of position p-1 when p completes a pair, else -1


def step_metadata(site: str) -> dict[str, object] | None:
    """The per-layer attention metadata of the current forward, or None for a dummy/profile forward without
    metadata (the caller then runs its projections only and writes no cache).

    Decided on ``ForwardContext.is_dummy_run`` (PORT_DESIGN §9 AM-2), never on the metadata's type alone:
    a list (micro-batching / DBO) raises NotImplementedError, and None on a real step raises -- either would
    otherwise zero the attention output and skip every cache write without a trace (rule 9)."""
    ctx = get_forward_context()
    md = ctx.attn_metadata
    if isinstance(md, dict):
        return md
    if isinstance(md, list):
        raise NotImplementedError(f"{site}: DeepSeek-V4.1 attention got per-ubatch (list) attention metadata; "
                                  "micro-batching / DBO is not supported")
    if md is None:
        if ctx.is_dummy_run:
            return None
        raise RuntimeError(f"{site}: attn_metadata is None on a forward that is not a dummy/profile run "
                           "(ForwardContext.is_dummy_run=False): refusing to skip attention and cache writes")
    raise TypeError(f"{site}: unexpected attn_metadata type {type(md).__name__}")


def _exact_seq_lens_cpu(cm: CommonAttentionMetadata) -> np.ndarray:
    seq = cm.seq_lens_cpu_upper_bound
    if seq is None:
        seq = cm._seq_lens_cpu
    if seq is None:
        raise RuntimeError(
            "DeepSeek-V4.1 attention metadata needs host-side sequence lengths "
            "(CommonAttentionMetadata.seq_lens_cpu_upper_bound) to size the compressor output without a "
            "device sync; none were provided")
    return np.asarray(seq.numpy() if isinstance(seq, torch.Tensor) else seq, dtype=np.int64)


def _require_exact_host_lengths(vllm_config: VllmConfig) -> None:
    spec = getattr(vllm_config, "speculative_config", None)
    sched = getattr(vllm_config, "scheduler_config", None)
    if spec is not None and getattr(sched, "async_scheduling", False):
        raise NotImplementedError(
            "DeepSeek-V4.1 SM70 attention (P2) relies on exact host sequence lengths; async scheduling with "
            "speculative decoding makes them upper bounds. Run with --no-async-scheduling (PORT_DESIGN A9).")


class _DS41BuilderBase(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # V4.1 attention is an eager break whose control flow depends on host metadata (number of
    # compressed latents, workspace tiles): it must not be captured inside FULL cudagraphs (P5 item).
    uses_physical_block_table: ClassVar[bool] = True

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig,
                 device: torch.device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        _require_exact_host_lengths(vllm_config)
        self._init_reorder_batch_threshold(1)

    def _check_block_geometry(self) -> None:
        kbs = getattr(self, "kernel_block_size", None)
        if kbs is not None and kbs != self.kv_cache_spec.block_size:
            raise RuntimeError(
                f"DeepSeek-V4.1 cache group {self.layer_names[:2]}... got kernel block size {kbs} != block size "
                f"{self.kv_cache_spec.block_size}; the V4.1 builders index physical blocks")

    def _common(self, cm: CommonAttentionMetadata) -> dict:
        self._check_block_geometry()
        T = int(cm.num_actual_tokens)
        R = int(cm.num_reqs)
        qsl_cpu = np.asarray(cm.query_start_loc_cpu[: R + 1].numpy(), dtype=np.int64)
        seq_cpu = _exact_seq_lens_cpu(cm)[:R]
        qlen = np.diff(qsl_cpu)
        starts = seq_cpu - qlen
        if (starts < 0).any():
            raise RuntimeError(f"inconsistent batch: seq_lens {seq_cpu} shorter than query lens {qlen}")
        pos_cpu = np.repeat(starts - qsl_cpu[:-1], qlen) + np.arange(int(qsl_cpu[-1]), dtype=np.int64)
        if pos_cpu.shape[0] != T:
            raise RuntimeError(f"token count mismatch: query_start_loc covers {pos_cpu.shape[0]} tokens, "
                               f"num_actual_tokens={T}")
        tok2req = cm.token_to_req_indices(self._tok2req_buf)[:T]
        positions = torch.from_numpy(pos_cpu).to(self.device, non_blocking=True)
        return dict(
            num_reqs=R, num_actual_tokens=T, query_start_loc=cm.query_start_loc[: R + 1],
            query_start_loc_cpu=qsl_cpu, seq_lens_cpu=seq_cpu, token_to_req_indices=tok2req,
            positions=positions, positions_cpu=pos_cpu, block_table=cm.block_table_tensor[:R],
            block_size=int(self.kv_cache_spec.block_size), slot_mapping=cm.slot_mapping[:T].to(torch.int64))

    @property
    def _tok2req_buf(self) -> torch.Tensor:
        buf = getattr(self, "_tok2req", None)
        if buf is None:
            n = self.vllm_config.scheduler_config.max_num_batched_tokens
            buf = torch.zeros(n, dtype=torch.int32, device=self.device)
            self._tok2req = buf
        return buf


def rows_of_positions(block_table: torch.Tensor, tok2req: torch.Tensor, pos: torch.Tensor,
                      block_size: int) -> torch.Tensor:
    """Row (= block * block_size + offset) of position ``pos`` [T, K] for each token's request; pos < 0 -> -1."""
    valid = pos >= 0
    p = pos.clamp(min=0)
    blk = block_table[tok2req.to(torch.long)[:, None].expand_as(p), (p // block_size)].to(torch.int64)
    return torch.where(valid, blk * block_size + p % block_size, torch.full_like(p, -1))


class DS41SWAMetadataBuilder(_DS41BuilderBase):
    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> DS41SWAMetadata:
        c = self._common(common_attn_metadata)
        T = c["num_actual_tokens"]
        pos = c["positions"][:, None] - (WINDOW - 1) + torch.arange(WINDOW, device=self.device)[None, :]
        real = (c["slot_mapping"] >= 0)[:, None]
        pos = torch.where(real, pos, torch.full_like(pos, -1))
        win = rows_of_positions(c["block_table"], c["token_to_req_indices"], pos, c["block_size"])
        assert win.shape == (T, WINDOW)
        return DS41SWAMetadata(**c, window_slots=win)


class DS41CompressedMetadataBuilder(_DS41BuilderBase):
    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> DS41CompressedMetadata:
        c = self._common(common_attn_metadata)
        spec = self.kv_cache_spec
        r = int(spec.compress_ratio)
        storage = int(spec.storage_block_size)
        T = c["num_actual_tokens"]
        cm = common_attn_metadata
        slots = get_compressed_slot_mapping(
            T, c["query_start_loc"], cm.seq_lens[: c["num_reqs"]], c["block_table"].clamp(min=0), storage, r)
        pos_cpu = c["positions_cpu"]
        # every real token of the step is in 0..T-1; a group completes at the token with (p + 1) % r == 0
        tok_cpu = np.flatnonzero((pos_cpu + 1) % r == 0).astype(np.int64)
        n = int(tok_cpu.shape[0])
        tok = torch.from_numpy(tok_cpu).to(self.device, non_blocking=True)
        lpos = torch.from_numpy(pos_cpu[tok_cpu] - (r - 1)).to(self.device, non_blocking=True)
        num_visible = torch.div(c["positions"] + 1, r, rounding_mode="floor")
        num_visible = torch.where(c["slot_mapping"] >= 0, num_visible, torch.zeros_like(num_visible))
        c["slot_mapping"] = slots.to(torch.int64)
        return DS41CompressedMetadata(
            **c, compress_ratio=r, storage_block_size=storage, num_latents=n, latent_token_idx=tok,
            latent_pos=lpos, latent_slots=c["slot_mapping"].index_select(0, tok), num_visible=num_visible,
            max_visible=int(((pos_cpu + 1) // r).max()) if T else 0)


class DS41StateMetadataBuilder(_DS41BuilderBase):
    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> DS41StateMetadata:
        c = self._common(common_attn_metadata)
        p = c["positions"]
        completes = ((p + 1) % 2 == 0) & (c["slot_mapping"] >= 0)
        prev = torch.where(completes, p - 1, torch.full_like(p, -1))
        prev_slot = rows_of_positions(c["block_table"], c["token_to_req_indices"], prev[:, None],
                                      c["block_size"])[:, 0]
        return DS41StateMetadata(**c, prev_slot=prev_slot)


# =====================================================================================
# backends and cache layers
# =====================================================================================


class _DS41BackendBase(AttentionBackend):
    HEAD_SIZES: ClassVar[tuple[int, ...]] = ()

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("DeepSeek-V4.1 cache backends carry metadata only; attention runs in "
                                  "DeepseekV41Attention")

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int,
                           cache_dtype_str: str = "auto") -> tuple[int, ...]:
        if num_kv_heads != 1:
            raise ValueError(f"DeepSeek-V4.1 caches are MQA, got num_kv_heads={num_kv_heads}")
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension: bool = False) -> tuple[int, ...]:
        return (0, 1, 2, 3) if include_num_layers_dimension else (0, 1, 2)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return list(cls.HEAD_SIZES)

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # every V4.1 cache backend: Platform.update_block_size_for_backend() asks whichever V4.1 cache layer it
        # finds first (SWA, compressed or state), and the specs were built with this size at model init
        return PREFERRED_BLOCK_SIZE


class DS41SWABackend(_DS41BackendBase):
    HEAD_SIZES = (SWA_RECORD_DIM,)

    @staticmethod
    def get_name() -> str:
        return "DS41_SM70_SWA"

    @staticmethod
    def get_builder_cls() -> type[DS41SWAMetadataBuilder]:
        return DS41SWAMetadataBuilder


class DS41CompressedBackend(_DS41BackendBase):
    HEAD_SIZES = (CKV_RECORD_DIM, IK_RECORD_DIM)

    @staticmethod
    def get_name() -> str:
        return "DS41_SM70_COMPRESSED"

    @staticmethod
    def get_builder_cls() -> type[DS41CompressedMetadataBuilder]:
        return DS41CompressedMetadataBuilder


class DS41StateBackend(_DS41BackendBase):
    HEAD_SIZES = (STATE_ROW_DIM,)

    @staticmethod
    def get_name() -> str:
        return "DS41_SM70_COMPRESSOR_STATE"

    @staticmethod
    def get_builder_cls() -> type[DS41StateMetadataBuilder]:
        return DS41StateMetadataBuilder


class DS41CacheLayer(nn.Module, AttentionLayerBase):
    """A named KV-cache holder in the static forward context (no forward of its own)."""

    def __init__(self, name: str, spec: KVCacheSpec, backend: type[AttentionBackend],
                 vllm_config: VllmConfig | None = None) -> None:
        super().__init__()
        self.layer_name = name
        self.prefix = name
        self._spec = spec
        self._backend = backend
        self.kv_cache = torch.tensor([])
        cfg = vllm_config if vllm_config is not None else get_current_vllm_config()
        ctx = cfg.compilation_config.static_forward_context
        if name in ctx:
            raise ValueError(f"Duplicate layer name: {name}")
        ctx[name] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return self._spec

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self._backend

    def rows(self) -> torch.Tensor:
        """The bound cache as [num_rows, row_dim] (row = physical block * storage_block_size + offset)."""
        kv = self.kv_cache
        if kv.numel() == 0:
            raise RuntimeError(f"KV cache of {self.layer_name} is not bound")
        dim = self._spec.head_size
        if kv.shape[-1] != dim or kv.dtype != self._spec.dtype or not kv.is_contiguous():
            raise RuntimeError(f"{self.layer_name}: unexpected cache tensor {tuple(kv.shape)} {kv.dtype} "
                               f"(contiguous={kv.is_contiguous()}), want [..., {dim}] {self._spec.dtype}")
        return kv.view(-1, dim)

    def forward(self) -> None:  # pragma: no cover - never called
        raise RuntimeError(f"{self.layer_name} is a cache holder")


# =====================================================================================
# sparse attention over FP16 records (decode + chunked prefill share one gather path)
# =====================================================================================

TILE_ENV = "VLLM_DS41_ATTN_TILE"


class DeepseekV41SM70SparseImpl:
    """Sparse MLA over the window rows (SWA cache) and the kv source's compressed rows.

    Decode and (chunked) prefill tokens are handled alike: every query gathers its own <= 128 window rows
    (``DS41SWAMetadata.window_slots``, written before the read) and <= 512 compressed rows (the published
    top-k, logical positions mapped through the SOURCE's block table). Queries are processed in tiles of
    ``VLLM_DS41_ATTN_TILE`` tokens (default 256: gather 256 x 640 x 1 KiB = 160 MiB) so the workspace does not
    grow with context (the V4 path gathered whole per-request caches)."""

    @staticmethod
    def tile() -> int:
        return knobs.env_int(TILE_ENV, 256, minimum=1)

    @staticmethod
    def compressed_rows(topk: torch.Tensor, src_md: DS41CompressedMetadata) -> torch.Tensor:
        return logical_to_rows(topk, src_md.token_to_req_indices, src_md.block_table, src_md.storage_block_size)

    @classmethod
    def forward(cls, q: torch.Tensor, swa_md: DS41SWAMetadata, swa_rows: torch.Tensor,
                ckv_rows: torch.Tensor | None, ckv_idx: torch.Tensor | None, attn_sink: torch.Tensor,
                scale: float, out: torch.Tensor, impl: str = "sm70") -> None:
        sources = [(swa_rows, swa_md.window_slots)]
        if ckv_rows is not None:
            assert ckv_idx is not None
            sources.append((ckv_rows, ckv_idx))
        sparse_attention(q, sources, attn_sink, scale, out, impl=impl, tile=cls.tile())

    @classmethod
    def reserve_workspace(cls, out: torch.Tensor, compressed: bool) -> None:
        """Profile run: allocate the largest per-tile buffers once so the memory profiler sees them."""
        q_tile = min(cls.tile(), max(1, out.shape[0]))
        k = WINDOW + (IDX_TOPK if compressed else 0)
        h = out.shape[1] if out.dim() == 3 else N_HEADS
        bufs = [torch.empty((q_tile, k, HEAD_DIM), dtype=torch.float16, device=out.device),
                torch.empty((q_tile, h, k), dtype=torch.float32, device=out.device),
                torch.empty((q_tile, h, k), dtype=torch.float32, device=out.device),
                torch.empty((q_tile, h, HEAD_DIM), dtype=torch.float32, device=out.device)]
        del bufs
