# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-local replica of kv source 20 for PP3 (PORT_DESIGN A7, §3.3, §3.5; owner L-ATTN, consumer half).

Stage 3 ([28-39]) reads source 20's compressed KV and index-K records (Reuse/Reindex layers) and its
candidate blocks (Reindex 28/32/36), but layer 20 runs on stage 2. The mirror registers two cache layers
under the SOURCE's names -- ``model.layers.20.attn`` (compressed KV) and ``model.layers.20.attn.indexer.k_cache``
(index-K) -- with specs equal to the source's, so ``get_kv_cache_configs`` merges them into the source's KV
cache group: the single scheduler-side KV manager hands both stages the same block ids and slot mappings,
and prefix-cache hits / preemption act on both copies together.

Exactness: layer 20 has compress ratio 1, so it emits one latent and one index key per scheduled token, in
token order; ``shared.export_ckv[:T]`` / ``shared.export_ik[:T]`` hold the exact FP16 records the source wrote
(copied by the same kernels). ``ingest`` scatters them to the slots of this stage's metadata for the same
names and copies the candidate rows into ``shared.candidate_blocks[:T]``. Nothing is recomputed.

Payload rows: exactly one per forward token (``positions.shape[0]``, padding included), never more -- a payload
that does not match the forward's token count raises (a mis-sliced boundary would otherwise be ignored).

Cross-stage check (PORT_DESIGN §3.5 item 4), debug knob ``VLLM_DS41_ATTN_MIRROR_CHECK=1``: the exporting stage
sends ``kv20_crc`` [T] int32 = ``kv20_crc(export_ckv[:T], export_ik[:T])`` (the packer is L-CORE's; key name
``PP_KEY_KV20_CRC``); ``ingest(..., crc=...)`` verifies the received rows against it (transport) and re-reads the
written slots against it (placement). With the knob on and no ``crc`` supplied, ingest raises.
"""

from __future__ import annotations

import re
from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import (
    CAND_SOURCE,
    CAND_TOPK_BLOCKS,
    CKV_RECORD_DIM,
    IK_RECORD_DIM,
    KV_RECORD_DTYPE,
    SharedAttnBuffers,
)
from vllm.models.deepseek_v41.sm70.sparse import (
    DS41CacheLayer,
    DS41CompressedBackend,
    DS41CompressedMetadata,
    compressed_cache_spec,
    index_k_cache_spec,
    step_metadata,
)

MIRROR_CHECK_ENV = "VLLM_DS41_ATTN_MIRROR_CHECK"
# PP payload key of the debug CRC (PORT_DESIGN §3.5 item 4). Not in the frozen contracts.py (CORE-owned): L-CORE
# adds it to its send/receive schema only when the knob is on.
PP_KEY_KV20_CRC = "kv20_crc"
_ATTN_LAYER_NAME = re.compile(r"(?P<base>.+)\.(?P<layer>\d+)\.attn\.ds41_attention")


def source_names(source_layer: int, layers_prefix: str) -> tuple[str, str]:
    """Names of the source's compressed-KV and index-K cache layers under ``layers_prefix`` (e.g. "model.layers")."""
    p = f"{layers_prefix}.{source_layer}.attn"
    return p, f"{p}.indexer.k_cache"


def layers_prefix_of_stage(static_forward_context: dict[str, object]) -> str:
    """The decoder-layer prefix shared by every DeepseekV41Attention registered on this stage (the consumers
    derive kv-source names from their own prefix the same way). Fails loud on none or on several."""
    bases = {m.group("base") for name in static_forward_context
             if (m := _ATTN_LAYER_NAME.fullmatch(name)) is not None}
    if len(bases) != 1:
        raise RuntimeError("DeepseekV41KVSourceMirror: cannot derive the decoder-layer prefix from the registered "
                           f"attention layers (found {sorted(bases)}); build the stage's layers first or pass "
                           "layers_prefix")
    return bases.pop()


def kv20_crc(ckv: torch.Tensor, ik: torch.Tensor) -> torch.Tensor:
    """[T] int32: per-row sum of the int16 bit patterns of ckv [T,512] and ik [T,128] (|sum| <= 640 * 32768 < 2^31)."""
    if ckv.dtype != KV_RECORD_DTYPE or ik.dtype != KV_RECORD_DTYPE or ckv.shape[0] != ik.shape[0]:
        raise ValueError(f"kv20_crc needs fp16 rows of equal count, got {ckv.dtype} {tuple(ckv.shape)} / "
                         f"{ik.dtype} {tuple(ik.shape)}")
    return (ckv.view(torch.int16).sum(dim=-1, dtype=torch.int32)
            + ik.view(torch.int16).sum(dim=-1, dtype=torch.int32))


class DeepseekV41KVSourceMirror(nn.Module):
    def __init__(self, vllm_config: VllmConfig, source_layer: int, shared: SharedAttnBuffers, *,
                 layers_prefix: str | None = None) -> None:
        """Registers in the static forward context, under the SOURCE's names, two AttentionLayerBase children:
        f"{layers_prefix}.{source_layer}.attn" (compressed-KV spec) and
        f"{layers_prefix}.{source_layer}.attn.indexer.k_cache" (index-K spec), whose get_kv_cache_spec() returns
        specs equal (dataclass ==) to the source's on the earlier stage. ``layers_prefix`` defaults to the prefix
        of this stage's registered DeepseekV41Attention layers (build them first), e.g. "model.layers"."""
        super().__init__()
        if source_layer != CAND_SOURCE:
            raise ValueError(f"v1 mirrors only kv source {CAND_SOURCE}, got {source_layer}")
        ratio = int(vllm_config.model_config.hf_config.compress_ratios[source_layer])
        if ratio != 1:
            raise ValueError(f"the mirror needs a ratio-1 source (one record per token), source {source_layer} "
                             f"has ratio {ratio}")
        block_size = vllm_config.cache_config.block_size
        self.source_layer = source_layer
        self.shared = shared
        ctx = vllm_config.compilation_config.static_forward_context
        self.layers_prefix = layers_prefix if layers_prefix is not None else layers_prefix_of_stage(ctx)
        ckv_name, ik_name = source_names(source_layer, self.layers_prefix)
        self.ckv_cache = DS41CacheLayer(ckv_name, compressed_cache_spec(block_size, ratio), DS41CompressedBackend,
                                        vllm_config)
        self.ik_cache = DS41CacheLayer(ik_name, index_k_cache_spec(block_size, ratio), DS41CompressedBackend,
                                       vllm_config)
        self.check = knobs.env_bool(MIRROR_CHECK_ENV, False)
        self.layer_name = f"{ckv_name}.ds41_mirror"
        if self.layer_name in ctx:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        ctx[self.layer_name] = self

    def ingest(self, positions: torch.Tensor, ckv: torch.Tensor, ik: torch.Tensor, cand: torch.Tensor,
               crc: torch.Tensor | None = None) -> None:
        """ckv [T,512] fp16, ik [T,128] fp16 (exact records as written by the source), cand [T,2048] int32, with
        T == positions.shape[0] (the forward's token count). Scatter ckv/ik to the slots of
        attn_metadata[source prefix].slot_mapping; copy cand to shared.candidate_blocks[:T]. ``crc`` [T] int32
        (payload key kv20_crc) is required when VLLM_DS41_ATTN_MIRROR_CHECK=1. Eager-break op."""
        if ckv.dtype != KV_RECORD_DTYPE or ckv.shape[-1] != CKV_RECORD_DIM:
            raise ValueError(f"mirror ckv must be fp16 [T, {CKV_RECORD_DIM}], got {ckv.dtype} {tuple(ckv.shape)}")
        if ik.dtype != KV_RECORD_DTYPE or ik.shape[-1] != IK_RECORD_DIM:
            raise ValueError(f"mirror ik must be fp16 [T, {IK_RECORD_DIM}], got {ik.dtype} {tuple(ik.shape)}")
        if cand.dtype != torch.int32 or cand.shape[-1] != CAND_TOPK_BLOCKS:
            raise ValueError(f"mirror cand must be int32 [T, {CAND_TOPK_BLOCKS}], got {cand.dtype} {tuple(cand.shape)}")
        T = positions.shape[0]
        if ckv.shape[0] != T or ik.shape[0] != T or cand.shape[0] != T:
            raise ValueError(f"mirror payload has {ckv.shape[0]}/{ik.shape[0]}/{cand.shape[0]} (ckv/ik/cand) rows for a "
                             f"forward of {T} tokens: the boundary must carry exactly one row per forward token")
        if crc is not None and (crc.dtype != torch.int32 or tuple(crc.shape) != (T,)):
            raise ValueError(f"mirror {PP_KEY_KV20_CRC} must be int32 [{T}], got {crc.dtype} {tuple(crc.shape)}")
        if self.check and crc is None:
            raise RuntimeError(f"{MIRROR_CHECK_ENV}=1 but the payload carries no {PP_KEY_KV20_CRC}: the exporting "
                               f"stage must send kv20_crc(export_ckv[:T], export_ik[:T])")
        ds41_mirror_ingest(positions, ckv, ik, cand, crc, self.layer_name)

    def ingest_impl(self, positions: torch.Tensor, ckv: torch.Tensor, ik: torch.Tensor, cand: torch.Tensor,
                    crc: torch.Tensor | None) -> None:
        md_all = step_metadata(self.layer_name)
        if md_all is None:              # dummy/profile forward without metadata: nothing to replicate
            return
        m_ckv = cast(DS41CompressedMetadata, md_all[self.ckv_cache.layer_name])
        m_ik = cast(DS41CompressedMetadata, md_all[self.ik_cache.layer_name])
        n = m_ckv.num_latents
        if m_ik.num_latents != n or m_ckv.compress_ratio != 1:
            raise RuntimeError(f"mirror metadata mismatch: ckv {n} latents (ratio {m_ckv.compress_ratio}), "
                               f"ik {m_ik.num_latents}")
        if n > ckv.shape[0]:
            raise RuntimeError(f"mirror metadata has {n} latents for a payload of {ckv.shape[0]} rows")
        ckv_rows, ik_rows = self.ckv_cache.rows(), self.ik_cache.rows()
        for rows, src, slots in ((ckv_rows, ckv, m_ckv.latent_slots), (ik_rows, ik, m_ik.latent_slots)):
            keep = slots >= 0
            rows.index_copy_(0, slots[keep], src[:n][keep])
        self.shared.candidate_blocks[:n].copy_(cand[:n])
        if self.check:
            if crc is None:
                raise RuntimeError(f"{MIRROR_CHECK_ENV}=1 but no {PP_KEY_KV20_CRC} reached ingest_impl")
            sent = crc[:n]
            if not torch.equal(kv20_crc(ckv[:n], ik[:n]), sent):
                bad = int((kv20_crc(ckv[:n], ik[:n]) != sent).sum())
                raise RuntimeError(f"KV mirror: {bad}/{n} payload rows do not match the exporter's {PP_KEY_KV20_CRC}")
            keep = (m_ckv.latent_slots >= 0) & (m_ik.latent_slots >= 0)
            if not torch.equal(m_ckv.latent_slots >= 0, m_ik.latent_slots >= 0):
                raise RuntimeError("KV mirror: ckv and ik slot mappings disagree on which tokens are written")
            got = kv20_crc(ckv_rows.index_select(0, m_ckv.latent_slots[keep]),
                           ik_rows.index_select(0, m_ik.latent_slots[keep]))
            if not torch.equal(got, sent[keep]):
                raise RuntimeError(f"KV mirror: {int((got != sent[keep]).sum())} written rows differ from the "
                                   f"exporter's {PP_KEY_KV20_CRC} after ingest")


@eager_break_during_capture
def ds41_mirror_ingest(positions: torch.Tensor, ckv: torch.Tensor, ik: torch.Tensor, cand: torch.Tensor,
                       crc: torch.Tensor | None, layer_name: str) -> None:
    forward_context: ForwardContext = get_forward_context()
    forward_context.no_compile_layers[layer_name].ingest_impl(positions, ckv, ik, cand, crc)
