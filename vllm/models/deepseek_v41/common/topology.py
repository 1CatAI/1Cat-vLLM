# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer topology and pipeline stage plans for DeepSeek-V4.1 (PORT_DESIGN §3.1, §5.1).

Every layer >= 2 with a non-zero compress ratio reads compressed KV from the latest kv source
(2, 8, 14, 20) and top-k indices from the latest index source (2, 8, 14, 20, 24, 28, 32, 36). A pipeline cut
must not separate a Reuse layer from its index source, and the only kv source a later stage may read across
a cut is 20 (ratio 1: one record per token, shipped in the PP payload and mirrored, §3.5). In v1 the mirror
must sit on the stage immediately after the stage that holds source 20 (the mirror does not re-export).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contracts import (
    CAND_SOURCE,
    COMPRESS_ROPE_THETA,
    ENGRAM_LAYERS,
    INDEX_SOURCES,
    KV_SOURCES,
    LEGAL_STAGE_STARTS,
    ROPE_THETA,
    LayerTopology,
    StagePlan,
)

N_BACKBONE_LAYERS = 40
MIRRORABLE_KV_SOURCES = (CAND_SOURCE,)


def _check_config(hf_config: Any) -> None:
    """The contract constants are the model's; a config that disagrees is a different model."""
    pairs = (
        ("kv_source_layer_ids", list(KV_SOURCES)),
        ("index_source_layer_ids", list(INDEX_SOURCES)),
        ("engram_layer_ids", list(ENGRAM_LAYERS)),
        ("candidate_source_layer_id", CAND_SOURCE),
        ("num_hidden_layers", N_BACKBONE_LAYERS),
    )
    for name, expected in pairs:
        value = getattr(hf_config, name, None)
        if isinstance(value, tuple):
            value = list(value)
        if value != expected:
            raise ValueError(f"DeepSeek-V4.1 topology: hf_config.{name}={value!r}, contract expects {expected!r}")
    ratios = getattr(hf_config, "compress_ratios", None)
    if ratios is None or len(ratios) < N_BACKBONE_LAYERS:
        raise ValueError(f"DeepSeek-V4.1 topology: compress_ratios must list >= {N_BACKBONE_LAYERS} layers, "
                         f"got {ratios!r}")


def _latest(sources: Sequence[int], layer_id: int) -> int | None:
    candidates = [s for s in sources if s <= layer_id]
    return max(candidates) if candidates else None


def layer_topology(hf_config: Any, layer_id: int) -> LayerTopology:
    """Topology of backbone layer ``layer_id`` (0..39) or DSpark block ``layer_id`` (40..42, SWA only)."""
    _check_config(hf_config)
    ratios = list(hf_config.compress_ratios)
    if not 0 <= layer_id < len(ratios):
        raise ValueError(f"layer {layer_id} outside compress_ratios (0..{len(ratios) - 1})")
    ratio = int(ratios[layer_id])
    if ratio not in (0, 1, 2):
        raise ValueError(f"layer {layer_id}: compress_ratio {ratio} not in (0, 1, 2)")

    if layer_id >= N_BACKBONE_LAYERS:  # DSpark draft blocks: SWA only, no compressor / indexer / Engram
        if ratio != 0:
            raise ValueError(f"DSpark layer {layer_id} must have compress_ratio 0, got {ratio}")
        return LayerTopology(layer_id=layer_id, compress_ratio=0, mode="swa", kv_source=None, index_source=None,
                             owns_compressor=False, owns_indexer=False, is_candidate_source=False,
                             uses_candidates=False, has_engram=False, rope_theta=ROPE_THETA, yarn=False)

    owns_compressor = layer_id in KV_SOURCES
    owns_indexer = layer_id in INDEX_SOURCES
    if ratio == 0:
        if owns_compressor or owns_indexer:
            raise ValueError(f"layer {layer_id} is a kv/index source but has compress_ratio 0")
        mode = "swa"
        kv_source = index_source = None
    else:
        kv_source = _latest(KV_SOURCES, layer_id)
        index_source = _latest(INDEX_SOURCES, layer_id)
        if kv_source is None or index_source is None:
            raise ValueError(f"layer {layer_id} has compress_ratio {ratio} but no kv/index source at or before it")
        if ratio != int(ratios[kv_source]):
            raise ValueError(f"layer {layer_id} (ratio {ratio}) reads kv source {kv_source} whose ratio is "
                             f"{ratios[kv_source]}")
        mode = "full" if owns_compressor else ("reindex" if owns_indexer else "reuse")
    return LayerTopology(
        layer_id=layer_id,
        compress_ratio=ratio,
        mode=mode,
        kv_source=kv_source,
        index_source=index_source,
        owns_compressor=owns_compressor,
        owns_indexer=owns_indexer,
        is_candidate_source=layer_id == CAND_SOURCE,
        uses_candidates=owns_indexer and CAND_SOURCE < layer_id,
        has_engram=layer_id in ENGRAM_LAYERS,
        rope_theta=COMPRESS_ROPE_THETA if ratio > 0 else ROPE_THETA,
        yarn=ratio > 0,
    )


def stage_bounds(partition: Sequence[int], num_layers: int = N_BACKBONE_LAYERS) -> list[tuple[int, int]]:
    """[(first, last)] inclusive per stage from per-stage layer counts; validates the partition."""
    counts = [int(c) for c in partition]
    if not counts or any(c <= 0 for c in counts) or sum(counts) != num_layers:
        raise ValueError(f"pipeline partition {list(partition)} must be positive counts summing to {num_layers}")
    bounds = []
    first = 0
    for count in counts:
        bounds.append((first, first + count - 1))
        first += count
    return bounds


def validate_stage_cuts(hf_config: Any, partition: Sequence[int]) -> list[tuple[int, int]]:
    """Check every stage of ``partition``; raise ValueError naming the first offending layer."""
    _check_config(hf_config)
    bounds = stage_bounds(partition)
    stage_of = {layer: s for s, (first, last) in enumerate(bounds) for layer in range(first, last + 1)}
    for s, (first, last) in enumerate(bounds):
        if s > 0 and first not in LEGAL_STAGE_STARTS:
            raise ValueError(f"illegal pipeline cut: stage {s} starts at layer {first}, which is not a legal "
                             f"stage start {LEGAL_STAGE_STARTS} (PORT_DESIGN §5.1)")
        for layer in range(first, last + 1):
            topo = layer_topology(hf_config, layer)
            if topo.index_source is not None and topo.index_source < first:
                raise ValueError(f"illegal pipeline cut: layer {layer} ({topo.mode}) reuses the top-k of index "
                                 f"source {topo.index_source}, which lies on earlier stage "
                                 f"{stage_of[topo.index_source]} (stage {s} starts at {first})")
            if topo.kv_source is not None and topo.kv_source < first:
                if topo.kv_source not in MIRRORABLE_KV_SOURCES:
                    raise ValueError(f"illegal pipeline cut: layer {layer} reads kv source {topo.kv_source} on "
                                     f"earlier stage {stage_of[topo.kv_source]}; only {MIRRORABLE_KV_SOURCES} "
                                     "can be mirrored")
                if stage_of[topo.kv_source] != s - 1:
                    raise ValueError(f"illegal pipeline cut: layer {layer} reads kv source {topo.kv_source} from "
                                     f"stage {stage_of[topo.kv_source]}, not the immediately preceding stage "
                                     f"{s - 1} (v1 mirrors do not re-export)")
    return bounds


def make_stage_plan(hf_config: Any, pp_rank: int, pp_size: int, partition: list[int]) -> StagePlan:
    """StagePlan of ``pp_rank`` for ``partition`` (VLLM_PP_LAYER_PARTITION counts over the 40 backbone layers)."""
    if not 0 <= pp_rank < pp_size or len(partition) != pp_size:
        raise ValueError(f"pp_rank {pp_rank} / pp_size {pp_size} do not match partition {partition}")
    bounds = validate_stage_cuts(hf_config, partition)
    first, last = bounds[pp_rank]

    def referenced_earlier_kv(stage: int) -> tuple[int, ...]:
        s_first, s_last = bounds[stage]
        refs = {layer_topology(hf_config, layer).kv_source for layer in range(s_first, s_last + 1)}
        return tuple(sorted(src for src in refs if src is not None and src < s_first))

    mirrored = referenced_earlier_kv(pp_rank)
    exports = tuple(sorted({src for later in range(pp_rank + 1, pp_size) for src in referenced_earlier_kv(later)
                            if first <= src <= last}))
    return StagePlan(
        pp_rank=pp_rank,
        pp_size=pp_size,
        first_layer=first,
        last_layer=last,
        mirrored_kv_sources=mirrored,
        exports_kv_sources=exports,
        engram_layers=tuple(layer for layer in ENGRAM_LAYERS if first <= layer <= last),
    )
