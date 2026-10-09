# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN §7.3 extras: candidate-mask path beyond 16,384 tokens vs the FP32 reference (official weights of
20, 21, 24), prefix-cache hit at a block boundary, and preemption + recompute (NaN-poisoned freed blocks)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import CAND_ALL

from .test_attn_harness import (
    RefState,
    layer_inputs,
    load_attn_weights,
    ref_attention,
    ref_config,
    rel_rms,
    synthetic_attn_weights,
    topology,
)
from .test_attn_layers import DEV, Stage, _overlap, dist_env  # noqa: F401  (fixture)

pytestmark = pytest.mark.sm70


@pytest.mark.weights
def test_long_context_candidates_real_weights(dist_env) -> None:    # noqa: F811
    cfg = ref_config()
    layers = (20, 21, 24)
    S = 20000
    weights = {i: load_attn_weights(i, DEV) for i in layers}
    st = Stage(cfg, layers, weights, num_blocks=400, max_model_len=65536)
    inputs = {i: {"a": layer_inputs(weights[i], S, seed=11 + i, device=DEV)} for i in layers}
    outs = {i: {"a": torch.full((S, 5120), float("nan"), device=DEV)} for i in layers}
    pos = 0
    for n in [4096] * 4 + [3612, 1, 1, 1, 1]:
        st.step([("a", pos, n)], inputs, outs)
        pos += n
    assert pos == S
    rows = torch.cat([torch.arange(600, S, 97), torch.arange(S - 40, S)]).unique().to(DEV)
    ref_state = RefState()
    report: dict = {}
    for i in layers:
        rec: dict = {}
        ref_out = ref_attention(cfg, topology(cfg, i), weights[i], inputs[i]["a"], ref_state, record=rec,
                                q_rows=rows, score_rows=64)
        m = report.setdefault(i, {})
        m["out"] = rel_rms(outs[i]["a"][rows], ref_out[rows])
        ptk = torch.cat([st.rec[("topk", i, "a")][k] for k in sorted(st.rec[("topk", i, "a")])]).long()
        m["topk_mean"], m["topk_min"] = _overlap(ptk[rows], rec["idx.topk"][rows])
        same = (ptk[rows] == rec["idx.topk"][rows]).all(dim=1)
        m["frac_same_topk"] = float(same.float().mean())
        m["out_same_topk"] = rel_rms(outs[i]["a"][rows][same], ref_out[rows][same])
        if i == 20:
            pc = torch.cat([st.rec[("cand", 20, "a")][k] for k in sorted(st.rec[("cand", 20, "a")])])[rows]
            long_rows = (rows + 1) > 16384 + 8
            assert bool((pc[long_rows, 0] != CAND_ALL).all()) and bool((pc[~long_rows, 0] == CAND_ALL).all())
            ov = []
            for r_i, r in enumerate(rows.tolist()):
                if not long_rows[r_i]:
                    continue
                port_blocks = set(pc[r_i].tolist()) - {-1}
                ref_keep = ref_state.cand[r][: (r + 1)]
                ref_blocks = set(torch.nonzero(ref_keep.view(-1)).view(-1).div(8, rounding_mode="floor").tolist())
                ov.append(len(port_blocks & ref_blocks) / len(ref_blocks))
            m["cand_mean"], m["cand_min"], m["cand_rows"] = float(np.mean(ov)), float(np.min(ov)), len(ov)
    print(report)
    for i, m in report.items():
        # Port and reference get the same FP16 inputs here. Rows whose top-512 set equals the reference's are at the
        # §4.5 sparse-attention gate (measured <= 9.2e-4). The other ~3 % of rows differ by 1-6 near-tie indexer picks
        # among up to 20K entries (FP4 QAT discontinuities); a swapped key can carry a large ATTENTION weight even
        # though its INDEXER score was a near-tie, so those rows dominate the composite (measured 7.5e-3..1.23e-2).
        # The composite is not a §4.5 metric; it is bounded at 2e-2 and the share of such rows is gated.
        assert m["out_same_topk"] <= 2e-3, (i, m)
        assert m["frac_same_topk"] >= 0.95, (i, m)                     # measured >= 96.7 %
        assert m["out"] <= 2e-2, (i, m)
        assert m["topk_mean"] >= 0.995 and m["topk_min"] >= 0.98, (i, m)  # §4.5 indexer gate (measured min 98.8 %)
    c = report[20]
    assert c["cand_rows"] > 20 and c["cand_mean"] >= 0.999 and c["cand_min"] >= 0.99, c   # measured 99.999 / 99.95 %


def _compare(cfg, layers, weights, inputs, outs, req, lo, hi, gate=1e-2):
    ref_state = RefState()
    for i in layers:
        ref = ref_attention(cfg, topology(cfg, i), weights[i], inputs[i][req], ref_state)
        err = rel_rms(outs[i][req][lo:hi], ref[lo:hi])
        assert err <= gate, f"layer {i} req {req} rows [{lo},{hi}): rel-RMS {err}"


def test_prefix_cache_hit_at_block_boundary(dist_env) -> None:    # noqa: F811
    """B reuses A's blocks for positions < 1024 (a multiple of every group's block size: 256/128/32) and computes
    only from 1024 on; its outputs must match the reference over B's full sequence."""
    cfg = ref_config()
    layers = (2, 3, 20, 21)
    weights = {i: synthetic_attn_weights(cfg, topology(cfg, i), DEV, seed=40 + i) for i in layers}
    st = Stage(cfg, layers, weights, num_blocks=256)
    S, L = 1500, 1024
    inputs = {i: {"a": layer_inputs(weights[i], S, seed=i, device=DEV)} for i in layers}
    for i in layers:
        tail = layer_inputs(weights[i], S, seed=500 + i, device=DEV)
        inputs[i]["b"] = torch.cat([inputs[i]["a"][:L], tail[L:]])
    outs = {i: {r: torch.full((S, 5120), float("nan"), device=DEV) for r in "ab"} for i in layers}
    pos = 0
    for n in (700, 324, 476):
        st.step([("a", pos, n)], inputs, outs)
        pos += n
    st.sim.share_prefix("b", "a", L)
    pos = L
    for n in (301, 1, 1, 173):
        st.step([("b", pos, n)], inputs, outs)
        pos += n
    assert pos == S
    _compare(cfg, layers, weights, inputs, outs, "b", L, S)


def test_preemption_and_recompute(dist_env) -> None:    # noqa: F811
    """A runs to 701 (odd end: a pending ratio-2 state), is preempted (blocks freed and NaN-poisoned), then
    recomputed from scratch and continued: every output must match the reference and stay finite."""
    cfg = ref_config()
    layers = (2, 3, 20, 21)
    weights = {i: synthetic_attn_weights(cfg, topology(cfg, i), DEV, seed=60 + i) for i in layers}
    st = Stage(cfg, layers, weights, num_blocks=256)
    S = 1300
    inputs = {i: {"a": layer_inputs(weights[i], S, seed=70 + i, device=DEV)} for i in layers}
    outs = {i: {"a": torch.full((S, 5120), float("nan"), device=DEV)} for i in layers}
    pos = 0
    for n in (511, 1, 189):
        st.step([("a", pos, n)], inputs, outs)
        pos += n
    st.sim.release("a")                    # preemption: everything recomputed
    for i in layers:
        outs[i]["a"].fill_(float("nan"))
    pos = 0
    for n in (701, 1, 598):
        st.step([("a", pos, n)], inputs, outs)
        pos += n
    _compare(cfg, layers, weights, inputs, outs, "a", 0, S)
