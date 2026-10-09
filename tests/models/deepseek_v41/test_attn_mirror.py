# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN: PP3 kv-source mirror (PORT_DESIGN §3.5, A7). Stage 2 runs layers 20, 21 and exports source 20's records
and candidate rows; stage 3 ingests them into the mirror (registered under layer 20's names, same block ids) and runs
Reindex 28 + Reuse 29. Gates: the mirror's rows equal the source's rows bit for bit after every step (prefill chunks
and decodes), and stage-3 outputs equal a single stage running 20, 21, 28, 29 bit for bit -- including a context
beyond 16,384 tokens, where real candidate blocks (not CAND_ALL) cross the boundary."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import CAND_ALL, StagePlan

from vllm.models.deepseek_v41.kv_mirror import kv20_crc

from .test_attn_harness import layer_inputs, ref_config, synthetic_attn_weights, topology
from .test_attn_layers import DEV, Stage, dist_env  # noqa: F401  (fixture)

pytestmark = pytest.mark.sm70

SRC = ("model.layers.20.attn", "model.layers.20.attn.indexer.k_cache")


def _run(S: int, chunks: list[int], check_env: bool, monkeypatch) -> None:
    if check_env:
        monkeypatch.setenv("VLLM_DS41_ATTN_MIRROR_CHECK", "1")
    cfg = ref_config()
    layers = (20, 21, 28, 29)
    weights = {i: synthetic_attn_weights(cfg, topology(cfg, i), DEV, seed=100 + i) for i in layers}
    nb = 64 + S // 64
    single = Stage(cfg, layers, weights, num_blocks=nb, max_model_len=max(65536, S + 1))
    s2 = Stage(cfg, (20, 21), weights, stage=StagePlan(1, 3, 14, 27, (), (20,), (14,)), num_blocks=nb,
               max_model_len=max(65536, S + 1))
    s3 = Stage(cfg, (28, 29), weights, stage=StagePlan(2, 3, 28, 39, (20,), (), ()), mirror=True, num_blocks=nb,
               max_model_len=max(65536, S + 1))
    for name in SRC:
        s3.sim.link(name, s2.sim)
    inputs = {i: {"a": layer_inputs(weights[i], S, seed=i, device=DEV)} for i in layers}
    out_1 = {i: {"a": torch.full((S, 5120), float("nan"), device=DEV)} for i in layers}
    out_pp = {i: {"a": torch.full((S, 5120), float("nan"), device=DEV)} for i in layers}
    pos, saw_real_candidates = 0, False
    for n in chunks:
        batch = [("a", pos, n)]
        single.step(batch, inputs, out_1)
        exported: list = []
        s2.step(batch, inputs, out_pp, export=exported)
        ckv, ik, cand, _ = exported[0]
        saw_real_candidates |= bool((cand[:, 0] != CAND_ALL).any())
        s3.step(batch, inputs, out_pp, payload=(ckv, ik, cand, kv20_crc(ckv, ik) if check_env else None))
        pos += n
        # mirror rows == source rows, bit for bit, for every compressed entry written so far (ratio 1)
        for name in SRC:
            a = s2.sim.rows_at(name, "a", np.arange(pos))
            b = s3.sim.rows_at(name, "a", np.arange(pos))
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name} differs after {pos} tokens"
    assert pos == S
    for i in layers:
        assert torch.equal(out_1[i]["a"], out_pp[i]["a"]), f"layer {i}: PP3 output != single-stage output"
    if S > 16384 + 8:
        assert saw_real_candidates, "context > 16,384 must ship real candidate blocks"


def test_mirror_short(dist_env, monkeypatch) -> None:    # noqa: F811
    _run(1300, [511, 1, 1, 400, 1, 255, 1, 128, 2], check_env=True, monkeypatch=monkeypatch)


def test_mirror_long_context_candidates(dist_env, monkeypatch) -> None:    # noqa: F811
    S = 4096 * 4 + 900
    _run(S, [4096, 4096, 4096, 4096, 899, 1], check_env=False, monkeypatch=monkeypatch)
