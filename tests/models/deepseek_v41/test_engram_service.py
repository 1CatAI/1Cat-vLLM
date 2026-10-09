# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngramHostService on synthetic tables of the real layout (lane L-ENGRAM; PORT_DESIGN §3.6, §7.3).

Checks runner-order assembly, padding rows = zero, dedup, chunked-prefill continuation, decode steps, double-buffered
steps without host syncs, async sampled fill, spec-decode rollback, resume, finished requests, dummy binds, admission
prefetch, TP sub-table sharding and O_DIRECT mode -- every staged row byte-compared with the table it came from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import EngramBatchLayout, EngramReqStep, EngramStepPlan

from .test_engram_synth import SyntheticEngram, build_synthetic_engram, tokenizer_path

pytestmark = pytest.mark.sm70


@pytest.fixture(scope="module")
def syn(tmp_path_factory: pytest.TempPathFactory) -> SyntheticEngram:
    return build_synthetic_engram(tmp_path_factory.mktemp("engram_syn"))


@pytest.fixture(autouse=True)
def _knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "0")


def make_service(syn: SyntheticEngram, tp_rank: int = 0, tp_size: int = 4, layers=(1, 14), max_tokens: int = 512,
                 io_threads: int = 3):
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    return EngramHostService(syn.hf_config, layers, tp_rank, tp_size, str(syn.row_dir), tokenizer_path(),
                             max_tokens, torch.device("cuda"), io_threads=io_threads)


class Oracle:
    """Expected rows straight from the synthetic tables, using an independent history per request."""

    def __init__(self, svc, syn: SyntheticEngram) -> None:
        from vllm.models.deepseek_v41.common.engram import EngramHasher

        self.svc, self.syn = svc, syn
        self.hasher = EngramHasher(svc.layout, svc.hasher.token_map, svc.layers, svc.subtables)
        self.hist: dict[str, list[int]] = {}

    def set(self, rid: str, start: int, ids) -> None:
        h = self.hist.setdefault(rid, [])
        del h[start:]
        h.extend(int(x) for x in ids)

    def rows(self, rid: str, start: int, n: int) -> np.ndarray:
        """[n, L, S, 264] expected bytes."""
        comp = self.hasher.compress(np.array(self.hist[rid]))
        ids = self.hasher.hash_positions(comp, np.arange(start, start + n))
        out = np.zeros(ids.shape + (264,), np.uint8)
        for li, lid in enumerate(self.svc.layers):
            out[:, li, :, :256] = self.syn.weights[lid][ids[:, li]]
            out[:, li, :, 256:] = self.syn.scales[lid][ids[:, li]]
        return out


def run_step(svc, step_id: int, reqs: list[EngramReqStep], runner_order: list[str], t_pad: int,
             sampled_fill=None, finished=frozenset()) -> dict[int, np.ndarray]:
    svc.begin_step(EngramStepPlan(step_id=step_id, reqs=tuple(reqs), finished_req_ids=frozenset(finished)))
    n_by = {r.req_id: r.num_tokens for r in reqs}
    qsl = np.cumsum([0] + [n_by[r] for r in runner_order]).astype(np.int32)
    svc.bind_batch(EngramBatchLayout(step_id=step_id, req_order=tuple(runner_order), query_start_loc=qsl,
                                     num_tokens=int(qsl[-1]), num_tokens_padded=t_pad, sampled_fill=sampled_fill))
    return {lid: svc.wait_rows(lid).cpu().numpy() for lid in svc.layers}


def expect(oracle: Oracle, got: dict[int, np.ndarray], segments: list[tuple[str, int, int]], t_pad: int) -> None:
    """segments in runner order: (req, start_pos, n)."""
    exp = np.concatenate([oracle.rows(rid, s, n) for rid, s, n in segments], axis=0)
    t = exp.shape[0]
    for li, lid in enumerate(oracle.svc.layers):
        assert got[lid].shape == (t_pad, oracle.svc.n_sub, 264)
        np.testing.assert_array_equal(got[lid][:t], exp[:, li])
        assert not got[lid][t:].any(), "padding rows must be all-zero"


def test_prefill_decode_runner_order_padding(syn: SyntheticEngram) -> None:
    svc = make_service(syn, tp_rank=1)
    try:
        orc = Oracle(svc, syn)
        rng = np.random.default_rng(0)
        pa = rng.integers(3, 129000, size=300)
        pb = rng.integers(3, 129000, size=50)
        pb[10:20] = pb[0]                      # repeated n-grams -> dedup inside a step
        orc.set("A", 0, pa)
        orc.set("B", 0, pb)
        # step 0: A prefill chunk [0, 200) (token_ids given), B whole prompt (token_ids None, prompt known)
        got = run_step(svc, 0, [EngramReqStep("A", 0, 200, pa[:200].astype(np.int32), pa.astype(np.int32)),
                                EngramReqStep("B", 0, 50, None, pb.astype(np.int32))],
                       ["B", "A"], t_pad=256)
        expect(orc, got, [("B", 0, 50), ("A", 0, 200)], 256)
        st = svc.stats()
        assert st["unique_rows"] < st["lookups"], "dedup did not remove the repeated n-grams"
        assert st["prefetch_rows"] > 0, "admission prefetch of A's remaining prompt not issued"
        # step 1: A continuation [200, 300) without token ids (known prompt), B first decode token
        tok_b = 777
        orc.set("B", 50, [tok_b])
        got = run_step(svc, 1, [EngramReqStep("A", 200, 100, None, None),
                                EngramReqStep("B", 50, 1, np.array([tok_b], np.int32), None)],
                       ["B", "A"], t_pad=104)
        expect(orc, got, [("B", 50, 1), ("A", 200, 100)], 104)
        # steps 2..7: decode both, no host synchronisation between steps (double-buffered slots)
        pos = {"A": 300, "B": 51}
        outs = []
        for k in range(2, 8):
            toks = {r: int(rng.integers(3, 129000)) for r in ("A", "B")}
            for r, tk in toks.items():
                orc.set(r, pos[r], [tk])
            svc.begin_step(EngramStepPlan(k, tuple(EngramReqStep(r, pos[r], 1, np.array([toks[r]], np.int32), None)
                                                   for r in ("A", "B")), frozenset()))
            svc.bind_batch(EngramBatchLayout(k, ("A", "B"), np.array([0, 1, 2], np.int32), 2, 8, None))
            outs.append(({lid: svc.wait_rows(lid).clone() for lid in svc.layers}, dict(pos)))
            for r in pos:
                pos[r] += 1
        torch.cuda.synchronize()
        for got_t, p in outs:
            expect(orc, {lid: v.cpu().numpy() for lid, v in got_t.items()}, [("A", p["A"], 1), ("B", p["B"], 1)], 8)
    finally:
        svc.shutdown()


def test_async_fill_rollback_resume_finish(syn: SyntheticEngram) -> None:
    svc = make_service(syn, tp_rank=3, layers=(14,))
    try:
        orc = Oracle(svc, syn)
        rng = np.random.default_rng(1)
        p = rng.integers(3, 129000, size=40)
        orc.set("R", 0, p)
        got = run_step(svc, 10, [EngramReqStep("R", 0, 40, p.astype(np.int32), p.astype(np.int32))], ["R"], 40)
        expect(orc, got, [("R", 0, 40)], 40)
        # async PP: the sampled token is unknown at begin_step, arrives through sampled_fill
        fill = torch.tensor([4242], dtype=torch.int32).pin_memory()
        ev = torch.cuda.Event()
        ev.record()
        orc.set("R", 40, [4242])
        got = run_step(svc, 11, [EngramReqStep("R", 40, 1, None, None)], ["R"], 1, sampled_fill=(fill, ev))
        expect(orc, got, [("R", 40, 1)], 1)
        # spec decode: 1 sampled + 2 drafts, then a rollback of the 2 rejected drafts
        d = np.array([11, 12, 13], np.int32)
        orc.set("R", 41, d)
        got = run_step(svc, 12, [EngramReqStep("R", 41, 3, d, None)], ["R"], 4)
        expect(orc, got, [("R", 41, 3)], 4)
        d2 = np.array([99, 98], np.int32)
        orc.set("R", 42, d2)                    # positions 42.. rewritten
        got = run_step(svc, 13, [EngramReqStep("R", 42, 2, d2, None)], ["R"], 2)
        expect(orc, got, [("R", 42, 2)], 2)
        # resume after preemption: full token list as prompt_token_ids, recompute from position 30
        full = np.array(orc.hist["R"], np.int32)
        got = run_step(svc, 14, [EngramReqStep("R", 30, 14, None, full)], ["R"], 16)
        expect(orc, got, [("R", 30, 14)], 16)
        # finished -> history dropped; reusing the id without a prompt fails loudly
        with pytest.raises(KeyError, match="no token history"):
            svc.begin_step(EngramStepPlan(15, (EngramReqStep("R", 44, 1, np.array([5], np.int32), None),),
                                          frozenset({"R"})))
        # a gap in the history is refused
        with pytest.raises(ValueError, match="without token ids"):
            svc.begin_step(EngramStepPlan(16, (EngramReqStep("S", 5, 2, None, np.arange(3, 6, dtype=np.int32)),),
                                          frozenset()))
    finally:
        svc.shutdown()


def test_sampled_fill_shape_and_draft_rollback(syn: SyntheticEngram) -> None:
    """sampled_fill must carry one id per request in runner order; a sampled token written after rejected drafts
    replaces them (no stale draft is ever hashed as a known token)."""
    svc = make_service(syn, tp_rank=1, layers=(1,))
    try:
        orc = Oracle(svc, syn)
        p = np.random.default_rng(6).integers(3, 129000, size=20)
        q = np.random.default_rng(7).integers(3, 129000, size=12)
        orc.set("A", 0, p)
        orc.set("B", 0, q)
        run_step(svc, 0, [EngramReqStep("A", 0, 20, p.astype(np.int32), p.astype(np.int32)),
                          EngramReqStep("B", 0, 12, q.astype(np.int32), q.astype(np.int32))], ["A", "B"], 32)
        drafts = np.array([7, 8, 9], np.int32)            # spec step: 1 sampled + 2 drafts for A
        orc.set("A", 20, drafts)
        run_step(svc, 1, [EngramReqStep("A", 20, 3, drafts, None)], ["A"], 3)
        # next step: only the first draft was accepted; the token at 21 comes from the sampler (unknown at begin)
        ev = torch.cuda.Event()
        ev.record()
        bad = (torch.tensor([555], dtype=torch.int32).pin_memory(), ev)
        svc.begin_step(EngramStepPlan(2, (EngramReqStep("A", 21, 1, None, None),
                                          EngramReqStep("B", 12, 1, None, None)), frozenset()))
        with pytest.raises(ValueError, match="one per"):
            svc.bind_batch(EngramBatchLayout(2, ("B", "A"), np.array([0, 1, 2], np.int32), 2, 2, bad))
        fill = (torch.tensor([[31], [555]], dtype=torch.int32).pin_memory(), ev)    # [num_reqs, 1] like recv
        orc.set("B", 12, [31])
        orc.set("A", 21, [555])
        got = run_step(svc, 3, [EngramReqStep("A", 21, 1, None, None), EngramReqStep("B", 12, 1, None, None)],
                       ["B", "A"], 2, sampled_fill=fill)
        expect(orc, got, [("B", 12, 1), ("A", 21, 1)], 2)
        # without ids, a position past the prompt cannot be treated as known (drafts may have been rejected)
        with pytest.raises(ValueError, match="only prompt positions"):
            svc.begin_step(EngramStepPlan(4, (EngramReqStep("A", 20, 2, None, None),), frozenset()))
    finally:
        svc.shutdown()


def test_dummy_rows_and_unbound_real_step(syn: SyntheticEngram) -> None:
    """AM-2: dummy forwards get zero rows in the static buffer (no host I/O, counted); a real step without bound rows
    raises; bind_batch refuses dummy layouts; a bind for a step begin_step never saw raises."""
    svc = make_service(syn, tp_rank=0)
    try:
        buf = svc.rows_buffer(1, 8)
        buf.fill_(7)
        rows = svc.dummy_rows(1, 8)
        assert rows.data_ptr() == buf.data_ptr() and not rows.any()
        svc.dummy_rows(14, 8)
        assert svc.stats()["dummy_steps"] == 1 and svc.stats()["lookups"] == 0
        with pytest.raises(RuntimeError, match="REAL step"):
            svc.wait_rows(1)
        with pytest.raises(ValueError, match="is_dummy_run"):
            svc.bind_batch(EngramBatchLayout(-1, (), np.zeros(1, np.int32), 0, 8, None))
        p = np.arange(100, 120, dtype=np.int32)
        svc.begin_step(EngramStepPlan(0, (EngramReqStep("Z", 0, 20, p, p),), frozenset()))
        with pytest.raises(RuntimeError, match="REAL step"):
            svc.wait_rows(1)                      # begun but not bound
        with pytest.raises(RuntimeError, match="begin_step saw"):
            svc.bind_batch(EngramBatchLayout(5, ("Z",), np.array([0, 20], np.int32), 20, 20, None))
    finally:
        svc.shutdown()


def test_tp_sharding_covers_every_subtable(syn: SyntheticEngram) -> None:
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    layout = EngramLayout.from_hf_config(syn.hf_config)
    for tp in (1, 2, 4, 8):
        subs = [layout.subtables_for_rank(r, tp) for r in range(tp)]
        assert sorted(s for sub in subs for s in sub) == list(range(24))
    for r, sub in enumerate(layout.subtables_for_rank(r, 4) for r in range(4)):
        assert [s // 8 for s in sub] == [0, 0, 1, 1, 2, 2], "TP4: two heads of every n-gram order per rank"
        assert sub == (r, r + 4, r + 8, r + 12, r + 16, r + 20)
    with pytest.raises(ValueError):
        layout.subtables_for_rank(0, 5)
    # four rank services together stage exactly the 24 sub-table rows of a full (TP1) service
    p = np.random.default_rng(2).integers(3, 129000, size=64).astype(np.int32)
    full = make_service(syn, tp_rank=0, tp_size=1, max_tokens=64)
    try:
        ref = run_step(full, 0, [EngramReqStep("Q", 0, 64, p, p)], ["Q"], 64)
    finally:
        full.shutdown()
    for r in range(4):
        svc = make_service(syn, tp_rank=r, tp_size=4, max_tokens=64)
        try:
            got = run_step(svc, 0, [EngramReqStep("Q", 0, 64, p, p)], ["Q"], 64)
            for lid in (1, 14):
                np.testing.assert_array_equal(got[lid], ref[lid][:, list(svc.subtables)])
        finally:
            svc.shutdown()


def test_o_direct_mode_identical(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    p = np.random.default_rng(3).integers(3, 129000, size=100).astype(np.int32)
    outs = []
    for odirect in ("0", "1"):
        monkeypatch.setenv("VLLM_DS41_ENGRAM_O_DIRECT", odirect)
        svc = make_service(syn, tp_rank=2)
        try:
            outs.append(run_step(svc, 0, [EngramReqStep("Q", 0, 100, p, p)], ["Q"], 128))
            assert svc.o_direct == (odirect == "1")
        finally:
            svc.shutdown()
    for lid in (1, 14):
        np.testing.assert_array_equal(outs[0][lid], outs[1][lid])


def test_unverified_shard_refused(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "1")
    with pytest.raises(RuntimeError, match="sha256-ok"):
        make_service(syn)


def test_scale_range_and_row_bias(tmp_path: Path) -> None:
    """Scales outside FP16's exact window get a power-of-two row bias; an impossible span is refused."""
    from vllm.models.deepseek_v41.common.engram import row_bias_for_exponents

    assert row_bias_for_exponents(127 - 10, 127 - 6) == 0
    assert row_bias_for_exponents(127 - 20, 127 - 12) == 5
    assert row_bias_for_exponents(127 + 1, 127 + 3) == 0      # 448 * 2^3 still fits FP16
    assert row_bias_for_exponents(127 + 6, 127 + 9) == -2     # 448 * 2^9 does not: shift down by 2
    with pytest.raises(ValueError):
        row_bias_for_exponents(127 - 30, 127 + 0)
    syn2 = build_synthetic_engram(tmp_path / "lowexp", seed=5, exp_range=(-20, -12))
    svc = make_service(syn2, layers=(1,), max_tokens=16)
    try:
        assert svc.row_bias(1) == 5 and svc.scale_exponent_range[1] == (-20, -12)
    finally:
        svc.shutdown()


def test_io_threads_knob_overrides_ctor(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_IO_THREADS", "1")
    svc = make_service(syn, io_threads=4)
    try:
        assert svc.io_threads == 1 and svc._reader.num_threads() == 1
    finally:
        svc.shutdown()
    monkeypatch.setenv("VLLM_DS41_ENGRAM_IO_THREADS", "0")
    with pytest.raises(ValueError, match="minimum"):
        make_service(syn)


def test_service_builds_under_cuda_default_device(syn: SyntheticEngram) -> None:
    """L-INTEG: vLLM's model loader constructs the model under `with torch.device("cuda")`; host staging must stay
    pinned CPU memory (failed with "cannot pin 'torch.cuda.ByteTensor'" at the first engine bring-up)."""
    with torch.device("cuda"):
        svc = make_service(syn)
    try:
        assert svc._host_rows.device.type == "cpu" and svc._host_rows.is_pinned()
        assert svc._host_idx.device.type == "cpu" and svc._host_idx.is_pinned()
    finally:
        svc.shutdown()


def test_failed_ctor_releases_scale_registration(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    """L-INTEG: a ctor that raises after registering its scale tables (here: io_threads knob 0) must unregister them;
    otherwise the next service fails with "cudaHostRegister ... failed" once the address is recycled."""
    import gc

    monkeypatch.setenv("VLLM_DS41_ENGRAM_IO_THREADS", "0")
    for _ in range(3):
        with pytest.raises(ValueError, match="minimum"):
            make_service(syn)
        gc.collect()
    monkeypatch.delenv("VLLM_DS41_ENGRAM_IO_THREADS")
    svc = make_service(syn)
    svc.shutdown()
