# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN vs L-REF golden tensors (PORT_DESIGN §3.8, §4.5; mode v100-semantic = the primary comparator).

Per-op tests feed the golden's inputs to one port op and compare with the golden output:
  * q path        attn.x  -> attn.qr, attn.q (dense-linear gate 2e-3); attn.x in FP32 -> port window-KV kernel ->
                  attn.kv_win bitwise (0 mismatches); FP16-rounded attn.x (the port's real input) -> <= 3 % flips
  * indexer       idx.q, idx.w, cache.ik -> idx.score (rel-RMS <= 1e-3, -inf pattern exact) -> idx.topk (same set)
  * candidates    idx.score (layer 20) -> cand.blocks (set equality)
  * sparse attn   attn.q + window/compressed records + attn.topk_all -> attn.o (<= 2e-3)
  * o-proj        attn.o -> attn.out (dense-linear gate 2e-3)
a long-context indexer test on p4_long (24.6K tokens, layers 20/24: real candidate blocks beyond 16,384), and a
golden-input end-to-end run: attn.x of the prefill (in chunks) and of decode1..4 through DeepseekV41Attention on the
simulated paged cache, gated in three links (see test_golden_layer_end_to_end):
  golden ~ ref32 (own FP32 reference fed FP32 attn.x)            -- validates the reference
  ref16  (same reference fed the FP16-rounded attn.x the port gets) -- input-rounding floor, not gated vs golden
  port   ~ ref16 at the §4.5 per-op gates                          -- the port's own error.

Goldens: $DS41_GOLDEN_DIR (default /mnt/nvme2/scratch/ds41/golden), final cases preferred over provisional/. Layers
not captured in the available batch are skipped (the PROV-2 batch has 0, 1, 2, 3, 14; later batches add 20, 21, 24).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks
from vllm.models.deepseek_v41.common.contracts import CAND_ALL
from vllm.models.deepseek_v41.common.rope import build_v41_rope
from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores, topk_sorted
from vllm.models.deepseek_v41.sm70.q_rope_kv_insert import q_rope_kv_insert
from vllm.models.deepseek_v41.sm70.sparse_kernels import sparse_attention

from .test_attn_harness import (
    RefState,
    load_attn_weights,
    ref_attention,
    ref_config,
    rel_rms,
    topology,
)
from .test_attn_layers import DEV, Stage, _overlap, dist_env  # noqa: F401  (fixture)

pytestmark = [pytest.mark.sm70, pytest.mark.weights]

ROOT = Path(os.environ.get("DS41_GOLDEN_DIR", "/mnt/nvme2/scratch/ds41/golden"))
MODE = "v100-semantic"
PROMPTS = ("p1_legal", "p2_code", "p3_doc")
GROUPS = ((0,), (1,), (2, 3), (14,), (20, 21, 24))
PHASES = ("prefill", "decode1", "decode2", "decode3", "decode4")


def _case_dir(prompt: str, phase: str, mode: str = MODE) -> Path | None:
    for base in (ROOT, ROOT / "provisional"):
        d = base / f"{prompt}__{mode}__{phase}"
        if (d / "manifest.json").is_file():
            return d
    return None


def _ownk_cases(prompt: str) -> list[Case] | None:
    """Prefill + decode1..4 with the decode steps from L-REF's own-K goldens (PORT_DESIGN AM-9: index sources score
    against their OWN index-K on every decode step, as the port does); None when the own-K batch is absent."""
    pre = _case_dir(prompt, "prefill")
    dec = [_case_dir(prompt, ph, MODE + "+ownk") for ph in PHASES[1:]]
    if pre is None or any(d is None for d in dec):
        return None
    return [Case(pre)] + [Case(d) for d in dec if d is not None]


class Case:
    def __init__(self, d: Path) -> None:
        from safetensors import safe_open
        self.dir = d
        self.manifest = json.loads((d / "manifest.json").read_text())
        self._files: dict = {}
        self._open = safe_open
        self.captured = set(self.manifest["capture_layers"])

    def get(self, layer: int, name: str) -> torch.Tensor:
        f = self.dir / f"L{layer:02d}.safetensors"
        if f not in self._files:
            self._files[f] = self._open(str(f), framework="pt", device="cpu")
        return self._files[f].get_tensor(name).to(DEV)

    def has(self, layer: int, name: str) -> bool:
        return any(t["layer"] == layer and t["op"] == name for t in self.manifest["tensors"])

    @property
    def start(self) -> int:
        return int(self.manifest["start_pos"])

    @property
    def num_tokens(self) -> int:
        return int(self.manifest["num_tokens"])


def _cases(prompt: str) -> list[Case]:
    out = []
    for ph in PHASES:
        d = _case_dir(prompt, ph)
        if d is None:
            break
        out.append(Case(d))
    return out


def _require(prompt: str, layers: tuple[int, ...]) -> list[Case]:
    cases = _cases(prompt)
    if not cases:
        pytest.skip(f"no {MODE} golden for {prompt} under {ROOT}")
    if not set(layers) <= cases[0].captured:
        pytest.skip(f"golden batch at {cases[0].dir.parent} did not capture layers {layers}")
    return cases


@pytest.fixture(scope="module")
def weights_cache():
    return {}


def _weights(cache: dict, layer: int) -> dict:
    if layer not in cache:
        cache[layer] = load_attn_weights(layer, DEV)
    return cache[layer]


# ------------------------------------------------------------------------------------------------ per-op tests
@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("layer", [0, 1, 2, 3, 14, 20, 21, 24])
def test_golden_per_op(dist_env, weights_cache, prompt: str, layer: int) -> None:    # noqa: F811
    cases = _require(prompt, (layer,))
    cfg = ref_config()
    topo = topology(cfg, layer)
    w = _weights(weights_cache, layer)
    st = Stage(cfg, (layer,), {layer: w})
    attn = st.attn[layer]
    report: dict = {}
    from vllm.models.deepseek_v41.common.rope import apply_rope_torch
    for case in cases:
        def g(n: str, _case=case) -> torch.Tensor:
            return _case.get(layer, n)
        T, start = case.num_tokens, case.start
        pos = torch.arange(start, start + T, device=DEV)
        subset = case.manifest.get("row_subset") or {}
        rows = torch.arange(T, device=DEV)
        if "attn.q" in subset.get("tensors", ()):
            rows = torch.tensor(subset["rows"], device=DEV)
        # ---- q path fed attn.x
        x16 = g("attn.x").half()
        qr_kv = torch.mm(x16, attn.fused_wqa_wkv.weight.t(), out_dtype=torch.float32)
        qr32 = attn.q_norm(qr_kv[:, :1280])
        q = torch.mm(qr32.half()[rows], attn.wq_b.weight.t()).view(rows.numel(), 64, 512)
        q = apply_rope_torch(q, pos[rows], attn.rotary_emb.cos_sin_cache)
        gq = g("attn.q")
        report.setdefault("qr", []).append(rel_rms(qr32, g("attn.qr")))
        report.setdefault("q", []).append(rel_rms(q, gq))
        # ---- window-KV records through the port kernel (RoPE + FP8 QAT -> FP16 rows). The golden computes the kv
        # projection from FP32 attn.x: fed the same FP32 x the kernel must reproduce every record bit for bit;
        # fed the FP16-rounded x of the real forward, QAT flips of up to one FP8 step remain (reported + bounded).
        gkw = g("attn.kv_win")
        x32 = g("attn.x")
        w_kv = attn.fused_wqa_wkv.weight[1280:]
        for tag, kv_pre in (("kv_win_mismatch_x32", x32 @ w_kv.float().t()),
                            ("kv_win_mismatch_x16", qr_kv[:, 1280:])):
            kv = attn.kv_norm(kv_pre)
            kv_rows = torch.empty(T, 512, dtype=torch.float16, device=DEV)
            q_rope_kv_insert(torch.zeros(T, 1, 512, dtype=torch.float16, device=DEV), kv, pos,
                             attn.rotary_emb.cos_sin_cache, kv_rows, torch.arange(T, device=DEV), impl="sm70")
            report.setdefault(tag, []).append(float((kv_rows.float() != gkw).float().mean()))
        # ---- o-proj fed the golden attention output
        go = g("attn.o")
        out = attn.output_projection(go.half().contiguous(), pos[rows])
        report.setdefault("oproj", []).append(rel_rms(out, g("attn.out")[rows]))
        # ---- sparse attention fed golden q, records and the full index list
        tk = g("attn.topk_all").long()
        win_rows = g("attn.kv_win") if case.manifest["phase"] == "prefill" else g("cache.win")
        srcs = [win_rows]
        if topo.compress_ratio:
            srcs.append(case.get(topo.kv_source, "cache.ckv"))
        kvs = torch.cat(srcs).half()
        o = torch.empty(rows.numel(), 64, 512, dtype=torch.float16, device=DEV)
        sparse_attention(gq.half().contiguous(), [(kvs, tk)], attn.attn_sink, 512 ** -0.5, o, impl="sm70")
        report.setdefault("sparse", []).append(rel_rms(o, go))
        # ---- indexer: scores, top-k and candidates fed golden q, w, K
        if topo.owns_indexer and case.has(layer, "idx.score"):
            gs = g("idx.score")
            n = gs.shape[1]
            keys = case.get(topo.kv_source, "cache.ik")[:n].half().contiguous()
            sc = torch.empty(T, n, device=DEV)
            index_scores(g("idx.q").half().contiguous(), g("idx.w"), keys, sc)
            ends = torch.div(pos + 1, topo.compress_ratio, rounding_mode="floor")
            beyond = torch.arange(n, device=DEV)[None, :] >= ends[:, None]
            # golden prompts are < 16,384 tokens: the candidate mask is a no-op, -inf == unreachable exactly
            assert torch.equal(torch.isinf(gs), beyond), "golden -inf pattern != causal bound"
            stale = (topo.owns_compressor and topo.compress_ratio == 2 and case.manifest["phase"] != "prefill"
                     and int(pos[-1] + 1) % 2 != 0)
            if stale:
                # Official Indexer.forward re-points shared_attn.index_k only when the layer emits a latent; on a
                # decode step that completes no pair a ratio-2 source scores against the K cache published last
                # (another layer's). The port uses its own index-K (PORT_DESIGN §3.3, upstream vLLM). Identify the
                # golden's K source instead of gating on it.
                hit = []
                for src in sorted(case.captured):
                    if case.has(src, "cache.ik") and case.get(src, "cache.ik").shape[0] >= n:
                        alt = torch.empty(T, n, device=DEV)
                        index_scores(g("idx.q").half().contiguous(), g("idx.w"),
                                     case.get(src, "cache.ik")[:n].half().contiguous(), alt)
                        if rel_rms(alt, gs) < 1e-6:
                            hit.append(src)
                report.setdefault("stale_index_k_golden_uses_layer", []).append(hit)
                assert hit, "golden scores at a stale-index_k step match no captured index-K cache"
            else:
                report.setdefault("score", []).append(rel_rms(sc[~beyond], gs[~beyond]))
            if not stale:
                # per-op indexer: golden attn.x (weights_proj), attn.qr, K -> port q GEMM/RoPE/QAT, scores, top-k
                from vllm.models.deepseek_v41.compressor import mm_fp32, mm_fp32_full
                from vllm.models.deepseek_v41.sm70.indexer_kernels import (
                    index_q_rope_qat,
                )
                ix = attn.indexer
                pq = mm_fp32_full(g("attn.qr"), ix.wq_b.weight).view(T, 32, 128)
                pq = index_q_rope_qat(pq, pos, attn.rotary_emb.cos_sin_cache)
                pw = mm_fp32(g("attn.x").half(), ix.weights_proj.weight) * (128 ** -0.5 * 32 ** -0.5)
                ps = torch.empty(T, n, device=DEV)
                index_scores(pq, pw, keys, ps)
                ps.masked_fill_(beyond, float("-inf"))
                report.setdefault("idx_q_mismatch", []).append(float((pq.float() != g("idx.q")).float().mean()))
                report.setdefault("score_port_q", []).append(rel_rms(ps[~beyond], gs[~beyond]))
                mean, mn = _overlap(topk_sorted(ps, ends, 512).long(), g("idx.topk").long())
                report.setdefault("topk_port_mean", []).append(mean)
                report.setdefault("topk_port_min", []).append(mn)
            got = topk_sorted(gs.clone(), ends, 512).long()
            ref = g("idx.topk").long()
            same = [set(a) - {-1} == set(b) - {-1} for a, b in zip(got.tolist(), ref.tolist())]
            report.setdefault("topk_same_set_from_golden_scores", []).append(float(np.mean(same)))
            if topo.is_candidate_source and case.has(layer, "cand.blocks"):
                cand = torch.empty(T, 2048, dtype=torch.int32, device=DEV)
                select_candidate_blocks(gs.clone(), ends, 2048, 8, cand)
                gc = g("cand.blocks").long()
                for r in range(T):
                    want = set(gc[r].tolist()) - {-1}
                    if cand[r, 0].item() == CAND_ALL:
                        assert want == set(range((int(ends[r]) + 7) // 8)), f"row {r}: CAND_ALL vs golden"
                    else:
                        assert set(cand[r].tolist()) - {-1} == want, f"row {r}"
                report.setdefault("cand_rows", []).append(T)
    print(prompt, layer, {k: [v if isinstance(v, list) else round(v, 6) for v in vals] for k, vals in report.items()})
    assert max(report["q"]) <= 2e-3 and max(report["qr"]) <= 2e-3, report
    assert max(report["oproj"]) <= 2e-3, report
    assert max(report["sparse"]) <= 2e-3, report
    assert max(report["kv_win_mismatch_x32"]) == 0.0, report       # bitwise window-KV records (MC-ATTN F4)
    assert max(report["kv_win_mismatch_x16"]) <= 0.03, report      # measured <= 2.2 % (FP16 input rounding)
    if "idx_q_mismatch" in report:
        assert max(report["idx_q_mismatch"]) == 0.0, report         # FP4 q from golden attn.qr (FP32 SGEMM): bitwise
    if "score" in report:
        assert max(report["score"]) <= 1e-3, report
        assert min(report["topk_same_set_from_golden_scores"]) >= 0.99, report
    if "score_port_q" in report:
        assert max(report["score_port_q"]) <= 1e-3, report
        assert min(report["topk_port_mean"]) >= 0.995 and min(report["topk_port_min"]) >= 0.98, report


# ------------------------------------------------------------------------------------------- long context (> 16K)
LONG_PROMPT = "p4_long"


@pytest.mark.parametrize("phase", ("prefill", "decode1"))
def test_golden_long_candidates_and_indexer(dist_env, weights_cache, phase: str) -> None:    # noqa: F811
    """p4_long (24,600-token prompt, layers 20 + 24 captured; MC-ATTN F5): the candidate path beyond 16,384 tokens.
    Layer 20 (candidate source): scores from golden q/w/K, candidate blocks from golden scores (100 % set equality,
    §4.5). Layer 24 (Reindex): the port's candidate mask on golden scores == golden cand.mask, top-512 from the
    masked golden scores == golden; the port's own indexer (golden attn.qr / attn.x, layer-20 K) at the §4.5
    indexer gates. Both layers also run the port indexer end to end from golden inputs: layer-20 candidate blocks
    from PORT scores vs golden (reported + >= 99.9 % mean overlap)."""
    from vllm.models.deepseek_v41.common.candidate_blocks import apply_candidate_mask
    from vllm.models.deepseek_v41.compressor import mm_fp32, mm_fp32_full
    from vllm.models.deepseek_v41.sm70.indexer_kernels import index_q_rope_qat

    d = _case_dir(LONG_PROMPT, phase)
    if d is None:
        pytest.skip(f"no {MODE} golden for {LONG_PROMPT}/{phase} under {ROOT}")
    case = Case(d)
    if not {20, 24} <= case.captured:
        pytest.skip(f"{d} did not capture layers 20 and 24")
    cfg = ref_config()
    for i in (20, 24):
        assert topology(cfg, i).compress_ratio == 1
    T, start = case.num_tokens, case.start
    subset = case.manifest.get("row_subset")
    rows = torch.tensor(subset["rows"], device=DEV) if subset else torch.arange(T, device=DEV)
    pos = start + rows
    ends = pos + 1                                                  # ratio 1: one entry per token
    R = rows.numel()
    g20 = lambda n: case.get(20, n)                                 # noqa: E731
    g24 = lambda n: case.get(24, n)                                 # noqa: E731
    s20, s24 = g20("idx.score"), g24("idx.score")
    n = s20.shape[1]
    vis = torch.arange(n, device=DEV)[None, :] < ends[:, None]
    assert torch.equal(torch.isneginf(s20), ~vis) and torch.equal(torch.isneginf(s24), ~vis)
    keys = g20("cache.ik")[:n].half().contiguous()
    rope = build_v41_rope(cfg, 1, max_positions=start + T + 1, device=DEV).cos_sin_cache
    report: dict = {"rows": R}

    # (1) layer-20 scores from golden q / w / K
    sc = torch.empty(R, n, device=DEV)
    index_scores(g20("idx.q").half().contiguous(), g20("idx.w"), keys, sc)
    report["l20_score"] = rel_rms(sc[vis], s20[vis])

    # (2) candidate blocks from golden layer-20 scores: set equality per row (CAND_ALL rows: every visible block)
    cand = torch.empty(R, 2048, dtype=torch.int32, device=DEV)
    select_candidate_blocks(s20.clone(), ends, 2048, 8, cand)
    gcb = g20("cand.blocks").long()
    real = 0
    for r in range(R):
        want = set(gcb[r].tolist()) - {-1}
        if cand[r, 0].item() == CAND_ALL:
            assert want == set(range((int(ends[r]) + 7) // 8)), f"row {r}: CAND_ALL vs golden"
        else:
            real += 1
            assert set(cand[r].tolist()) - {-1} == want, f"row {r}: candidate blocks differ"
    report["cand_real_rows"] = real
    assert real > 0, "p4_long must exercise real candidate blocks (rows beyond 16,384 + 8 tokens)"

    # (3) layer-24 mask from those candidates == golden cand.mask (& causal); finite scores untouched
    m24 = s24.clone()
    apply_candidate_mask(m24, ends, cand, 8)
    assert torch.equal(torch.isneginf(m24), ~(g20("cand.mask").bool() & vis)), "candidate mask != golden"
    fin = torch.isfinite(m24)
    assert torch.equal(m24[fin], s24[fin])
    gtk24 = g24("idx.topk").long()
    same = [set(a) - {-1} == set(b) - {-1} for a, b in zip(topk_sorted(m24, ends, 512).long().tolist(),
                                                           gtk24.tolist())]
    report["l24_topk_same_set_from_golden_scores"] = float(np.mean(same))
    assert report["l24_topk_same_set_from_golden_scores"] >= 0.999, report

    # (4) the port's indexer from golden inputs (attn.qr FP32, attn.x, layer-20 K): q, scores, candidates, top-k
    for i, g in ((20, g20), (24, g24)):
        ix = Stage(cfg, (i,), {i: _weights(weights_cache, i)}).attn[i].indexer
        pq = index_q_rope_qat(mm_fp32_full(g("attn.qr"), ix.wq_b.weight).view(R, 32, 128), pos, rope)
        report[f"l{i}_idx_q_mismatch"] = float((pq.float() != g("idx.q")).float().mean())
        pw = mm_fp32(g("attn.x").half(), ix.weights_proj.weight) * (128 ** -0.5 * 32 ** -0.5)
        ps = torch.empty(R, n, device=DEV)
        index_scores(pq, pw, keys, ps)
        ps.masked_fill_(~vis, float("-inf"))
        report[f"l{i}_score_port_q"] = rel_rms(ps[vis], g("idx.score")[vis])
        if i == 20:
            pc = torch.empty(R, 2048, dtype=torch.int32, device=DEV)
            select_candidate_blocks(ps.clone(), ends, 2048, 8, pc)
            ov = []
            for r in range(R):
                want = set(gcb[r].tolist()) - {-1}
                got = (set(range((int(ends[r]) + 7) // 8)) if pc[r, 0].item() == CAND_ALL
                       else set(pc[r].tolist()) - {-1})
                ov.append(len(got & want) / len(want))
            report["l20_cand_port_mean"], report["l20_cand_port_min"] = float(np.mean(ov)), float(np.min(ov))
            port_cand = pc
        else:
            apply_candidate_mask(ps, ends, port_cand, 8)
        mean, mn = _overlap(topk_sorted(ps, ends, 512).long(), g("idx.topk").long())
        report[f"l{i}_topk_port_mean"], report[f"l{i}_topk_port_min"] = mean, mn
    print(LONG_PROMPT, phase, {k: round(v, 7) if isinstance(v, float) else v for k, v in report.items()})
    assert report["l20_score"] <= 1e-3, report
    for i in (20, 24):
        assert report[f"l{i}_idx_q_mismatch"] == 0.0, report       # FP32 SGEMM q: bitwise vs golden
        assert report[f"l{i}_score_port_q"] <= 1e-3, report
        assert report[f"l{i}_topk_port_mean"] >= 0.995 and report[f"l{i}_topk_port_min"] >= 0.98, report
    assert report["l20_cand_port_mean"] >= 0.999, report


# ------------------------------------------------------------------------------------------------ end to end
@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("group", GROUPS, ids=lambda g: "L" + "-".join(map(str, g)))
def test_golden_layer_end_to_end(dist_env, weights_cache, prompt: str, group) -> None:    # noqa: F811
    _require(prompt, group)
    ownk = _ownk_cases(prompt)
    cases = ownk if ownk is not None else _cases(prompt)
    cfg = ref_config()
    weights = {i: _weights(weights_cache, i) for i in group}
    st = Stage(cfg, group, weights)
    pre = cases[0]
    T0 = pre.num_tokens
    inputs = {i: {"a": torch.cat([c.get(i, "attn.x") for c in cases]).half()} for i in group}
    S = inputs[group[0]]["a"].shape[0]
    outs = {i: {"a": torch.full((S, 5120), float("nan"), device=DEV)} for i in group}
    chunks, done = [], 0
    for n in (min(T0, 37), min(T0, 511)):
        if done + n <= T0 and n > 0:
            chunks.append(n)
            done += n
    if done < T0:
        chunks.append(T0 - done)
    chunks += [1] * (S - T0)
    pos = 0
    for n in chunks:
        st.step([("a", pos, n)], inputs, outs)
        pos += n
    # Gated links (MC-ATTN F3 + final goldens; evidence /mnt/nvme2/scratch/ds41/attn/f3f4/):
    #   golden ~ ref32: the harness's FP32 reference fed the golden's FP32 attn.x reproduces the golden (<= 1e-3).
    #     Decode steps come from L-REF's own-K goldens (AM-9). Without them, ratio-2 decode steps completing no pair
    #     are excluded (STALE rows: the official reference scores them against another layer's index-K).
    #   ref16: the same reference fed the FP16-rounded attn.x the port receives (§4.1 FP16 sublayer inputs). The
    #     FP16 rounding flips FP4 x E4M3 compressed records (0.06-0.24 %) and FP8 window records (~0.8 %); one
    #     flipped compressed record among the few an early query sees moves its output by percent: final p1_legal
    #     layer 14 ref16 vs golden = 2.18e-2, but 2.07e-3 when ref16 uses ref32's compressed records. So the
    #     input-rounding floor is gated through that substitution (<= 6e-3), not on the raw distance.
    #   port ~ ref16: same input, same FP32-faithful math; near-tie FP4 flips of the compressed records (port vs
    #     ref16 <= 1e-4 of values) dominate the raw distance (final p3_doc layer 14: 4.7e-3 from 6 flipped values
    #     out of 2.1 M). Gated: records <= 1e-4 flips; output given the PORT's records at §4.5 (<= 2e-3); raw
    #     output <= 1e-2; top-512 at §4.5; latent <= 1e-3; vs golden, the port is within 2e-4 of the reference fed
    #     the same FP16 input and the port's records (measured excess <= 5e-5).
    ref32_state, ref16_state, ref16_c32_state, ref16_cport_state = RefState(), RefState(), RefState(), RefState()
    report: dict = {}
    for i in group:
        topo = topology(cfg, i)
        m = report.setdefault(i, {})
        x32 = torch.cat([c.get(i, "attn.x") for c in cases])
        rec32: dict = {}
        rec16: dict = {}
        ref32 = ref_attention(cfg, topo, weights[i], x32, ref32_state, record=rec32)
        ref16 = ref_attention(cfg, topo, weights[i], x32.half(), ref16_state, record=rec16)
        port_ckv = None
        if topo.owns_compressor:
            n_c = rec16["attn.ckv"].shape[0]
            port_ckv = st.sim.rows_at(f"model.layers.{i}.attn", "a", np.arange(n_c)).float()
            m["ckv_flips_port_ref16"] = float((port_ckv != rec16["attn.ckv"]).float().mean())
            m["ckv_flips_ref16_ref32"] = float((rec16["attn.ckv"] != rec32["attn.ckv"]).float().mean())
        ref16_c32 = ref_attention(cfg, topo, weights[i], x32.half(), ref16_c32_state,
                                  ckv_records=rec32["attn.ckv"] if topo.owns_compressor else None)
        ref16_cport = ref_attention(cfg, topo, weights[i], x32.half(), ref16_cport_state, ckv_records=port_ckv)
        golden_out = torch.cat([c.get(i, "attn.out") for c in cases])
        port_out = outs[i]["a"]
        keep = torch.ones(S, dtype=torch.bool, device=DEV)
        if ownk is None and topo.compress_ratio == 2:
            for k, c in enumerate(cases[1:]):
                if (c.start + 1) % 2 != 0:
                    keep[T0 + k] = False
        m["ownk_decode"] = ownk is not None
        m["stale_rows_excluded"] = int((~keep).sum())
        m["out_ref32_golden"] = rel_rms(ref32[keep], golden_out[keep])
        m["out_ref16_golden"] = rel_rms(ref16[keep], golden_out[keep])
        m["out_ref16_c32_golden"] = rel_rms(ref16_c32[keep], golden_out[keep])
        m["out_port_golden"] = rel_rms(port_out[keep], golden_out[keep])
        m["out_port_ref16"] = rel_rms(port_out, ref16)
        m["out_port_ref16_given_port_records"] = rel_rms(port_out, ref16_cport)
        m["out_ref16_cport_golden"] = rel_rms(ref16_cport[keep], golden_out[keep])
        if topo.compress_ratio:
            src = i if cases[0].has(i, "idx.topk") else topo.index_source
            ptk = torch.cat([st.rec[("topk", i, "a")][k] for k in sorted(st.rec[("topk", i, "a")])]).long()
            kc = keep.cpu()
            m["topk_port_ref16"] = _overlap(ptk, rec16["idx.topk"].long().cpu())
            if all(c.has(src, "idx.topk") for c in cases):
                gtk = torch.cat([c.get(src, "idx.topk") for c in cases]).long().cpu()
                m["topk_ref16_golden"] = _overlap(rec16["idx.topk"].long().cpu()[kc], gtk[kc])
                m["topk_port_golden"] = _overlap(ptk.cpu()[kc], gtk[kc])
        if topo.owns_compressor and cases[0].has(i, "attn.latent"):
            lat = st.rec[("latent", i, "a")]
            gl = torch.cat([c.get(i, "attn.latent") for c in cases if c.has(i, "attn.latent")])
            port = torch.stack([lat[p] for p in sorted(lat)])[: gl.shape[0]]
            m["latent_port_golden"] = rel_rms(port, gl)
            m["latent_port_ref16"] = rel_rms(port, rec16["attn.latent"][: gl.shape[0]])
    print(prompt, group, report)
    for i, m in report.items():
        assert m["out_ref32_golden"] <= 1e-3, (i, m)                 # the reference is faithful to the golden
        assert m["out_ref16_c32_golden"] <= 6e-3, (i, m)             # input-rounding floor, records held fixed
        assert m["out_port_ref16_given_port_records"] <= 2e-3, (i, m)  # §4.5: the port's own arithmetic
        assert m["out_port_ref16"] <= 1e-2, (i, m)                   # composite incl. near-tie record flips
        if "ckv_flips_port_ref16" in m:
            assert m["ckv_flips_port_ref16"] <= 1e-4, (i, m)
        # vs golden the port adds no more than 2e-4 to the FP32 reference fed the same FP16 input and the same
        # compressed records (its near-tie record choices are gated above by ckv_flips_port_ref16)
        assert m["out_port_golden"] <= m["out_ref16_cport_golden"] + 2e-4, (i, m)
        if "topk_port_ref16" in m:
            mean, mn = m["topk_port_ref16"]
            assert mean >= 0.995 and mn >= 0.98, (i, m)               # §4.5 indexer gate
        if "topk_port_golden" in m:
            # no worse than the FP32 reference given the same FP16 input, within 2 of 512 picks
            assert m["topk_port_golden"][0] >= 0.995, (i, m)
            assert m["topk_port_golden"][1] >= m["topk_ref16_golden"][1] - 2 / 512, (i, m)
        if "latent_port_golden" in m:
            assert m["latent_port_golden"] <= 1e-3 and m["latent_port_ref16"] <= 1e-3, (i, m)
