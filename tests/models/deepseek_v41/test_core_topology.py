# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2 item 3: layer_topology for all 43 layer ids (PORT_DESIGN §3.1 table) and make_stage_plan legality.

The expected values are transcribed from the §3.1 table independently of topology.py; legality of every 2- and
3-stage partition (and a 4-stage sample) is checked against a dependency oracle built from that table."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v41.common.contracts import LEGAL_STAGE_STARTS, LayerTopology, StagePlan
from vllm.models.deepseek_v41.common.topology import layer_topology, make_stage_plan

RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]


def _config(**overrides) -> SimpleNamespace:
    base = dict(compress_ratios=list(RATIOS), num_hidden_layers=40, kv_source_layer_ids=[2, 8, 14, 20],
                index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36], engram_layer_ids=[1, 14],
                candidate_source_layer_id=20)
    base.update(overrides)
    return SimpleNamespace(**base)


def _expected(layer: int) -> tuple:
    """(ratio, mode, kv_source, index_source, engram, cand_source, uses_cand) straight from the §3.1 table."""
    if layer in (0,) or layer >= 40:
        return (0, "swa", None, None, False, False, False)
    if layer == 1:
        return (0, "swa", None, None, True, False, False)
    if layer in (2, 8, 14):
        return (2, "full", layer, layer, layer == 14, False, False)
    if 3 <= layer <= 19:
        src = 2 if layer < 8 else (8 if layer < 14 else 14)
        return (2, "reuse", src, src, False, False, False)
    if layer == 20:
        return (1, "full", 20, 20, False, True, False)
    if 21 <= layer <= 23:
        return (1, "reuse", 20, 20, False, False, False)
    if layer in (24, 28, 32, 36):
        return (1, "reindex", 20, layer, False, False, True)
    idx = max(s for s in (24, 28, 32, 36) if s <= layer)
    return (1, "reuse", 20, idx, False, False, False)


@pytest.mark.parametrize("layer", range(43))
def test_layer_topology_table(layer: int) -> None:
    topo = layer_topology(_config(), layer)
    ratio, mode, kv, idx, engram, cand_src, uses_cand = _expected(layer)
    assert isinstance(topo, LayerTopology) and topo.layer_id == layer
    assert (topo.compress_ratio, topo.mode, topo.kv_source, topo.index_source) == (ratio, mode, kv, idx)
    assert topo.has_engram is engram and topo.is_candidate_source is cand_src and topo.uses_candidates is uses_cand
    assert topo.owns_compressor is (layer in (2, 8, 14, 20))
    assert topo.owns_indexer is (layer in (2, 8, 14, 20, 24, 28, 32, 36))
    assert topo.rope_theta == (160000.0 if ratio > 0 else 10000.0) and topo.yarn is (ratio > 0)


def test_layer_topology_on_official_config(ds41_text_config) -> None:
    for layer in range(43):
        assert layer_topology(ds41_text_config, layer) == layer_topology(_config(), layer)


@pytest.mark.parametrize("bad", [
    dict(kv_source_layer_ids=[2, 8, 14, 21]),
    dict(index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32]),
    dict(engram_layer_ids=[1]),
    dict(num_hidden_layers=43),
    dict(compress_ratios=[0, 0] + [2] * 18 + [4] * 20 + [0, 0, 0]),
    dict(compress_ratios=[0, 0, 0] + [2] * 17 + [1] * 20 + [0, 0, 0]),   # source 2 with ratio 0
    dict(compress_ratios=[0, 0] + [2] * 19 + [1] * 19 + [0, 0, 0]),      # layer 20 ratio 2 vs consumers 1
    dict(compress_ratios=list(RATIOS[:40]) + [0, 0, 1]),                  # DSpark layer with a ratio
])
def test_layer_topology_rejects_foreign_configs(bad: dict) -> None:
    cfg = _config(**bad)
    with pytest.raises(ValueError):
        for layer in range(len(cfg.compress_ratios)):
            layer_topology(cfg, layer)


# ---- legality oracle from the table: what each layer needs from earlier layers in the same step ----
def _oracle_legal(cuts: tuple[int, ...]) -> bool:
    starts = (0,) + cuts
    stage_of = {}
    for s, first in enumerate(starts):
        end = starts[s + 1] if s + 1 < len(starts) else 40
        for layer in range(first, end):
            stage_of[layer] = s
    for layer in range(40):
        _, mode, kv, idx, *_ = _expected(layer)
        if mode == "swa":
            continue
        s = stage_of[layer]
        if stage_of[idx] != s:          # top-k indices never cross a stage boundary
            return False
        if stage_of[kv] != s and not (kv == 20 and stage_of[kv] == s - 1):
            return False                # only source 20, from the adjacent stage, can be mirrored
    return True


def _partition(cuts: tuple[int, ...]) -> list[int]:
    edges = (0,) + cuts + (40,)
    return [b - a for a, b in zip(edges, edges[1:])]


def test_legal_stage_starts_match_oracle() -> None:
    assert {b for b in range(1, 40) if _oracle_legal((b,))} == set(LEGAL_STAGE_STARTS)


@pytest.mark.parametrize("n_cuts", [1, 2])
def test_all_partitions_match_oracle(n_cuts: int) -> None:
    cfg = _config()
    for cuts in itertools.combinations(range(1, 40), n_cuts):
        partition = _partition(cuts)
        legal = _oracle_legal(cuts)
        for rank in range(len(partition)):
            if legal:
                plan = make_stage_plan(cfg, rank, len(partition), partition)
                assert plan.first_layer == ((0,) + cuts)[rank]
            else:
                with pytest.raises(ValueError):
                    make_stage_plan(cfg, rank, len(partition), partition)


def test_four_stage_sample_matches_oracle() -> None:
    cfg = _config()
    for cuts in itertools.combinations((1, 2, 8, 13, 14, 20, 21, 24, 28, 32, 36), 3):
        partition = _partition(cuts)
        if _oracle_legal(cuts):
            make_stage_plan(cfg, 0, 4, partition)
        else:
            with pytest.raises(ValueError):
                make_stage_plan(cfg, 0, 4, partition)


def test_pp2_cut_20() -> None:
    cfg = _config()
    s0 = make_stage_plan(cfg, 0, 2, [20, 20])
    s1 = make_stage_plan(cfg, 1, 2, [20, 20])
    assert s0 == StagePlan(0, 2, 0, 19, (), (), (1, 14))
    assert s1 == StagePlan(1, 2, 20, 39, (), (), ())


def test_pp3_cuts_14_28_mirror_source_20() -> None:
    cfg = _config()
    plans = [make_stage_plan(cfg, r, 3, [14, 14, 12]) for r in range(3)]
    assert plans[0] == StagePlan(0, 3, 0, 13, (), (), (1,))
    assert plans[1] == StagePlan(1, 3, 14, 27, (), (20,), (14,))
    assert plans[2] == StagePlan(2, 3, 28, 39, (20,), (), ())


def test_single_stage() -> None:
    assert make_stage_plan(_config(), 0, 1, [40]) == StagePlan(0, 1, 0, 39, (), (), (1, 14))


def test_d3_split_rejected_naming_layer_13() -> None:
    with pytest.raises(ValueError, match=r"layer 13\b"):
        make_stage_plan(_config(), 0, 3, [13, 14, 13])


@pytest.mark.parametrize("partition", [[20, 19], [21, 20], [0, 40], [20, 20, 0]])
def test_bad_partitions(partition: list[int]) -> None:
    with pytest.raises(ValueError):
        make_stage_plan(_config(), 0, len(partition), partition)


def test_reuse_cut_names_index_source() -> None:
    # cut at 24 then 26: layer 26 (reuse) needs layer 24's top-k from the previous stage
    with pytest.raises(ValueError, match="layer 26"):
        make_stage_plan(_config(), 0, 3, [24, 2, 14])
