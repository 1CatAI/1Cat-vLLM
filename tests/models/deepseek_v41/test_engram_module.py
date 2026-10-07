# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV41Engram vs an FP32 transcription of the official Engram (lane L-ENGRAM; PORT_DESIGN §4.1, §4.5, §7.3).

The transcription follows ref:m.py:296-365 literally (embedding dequant -> BF16, wkv dequant, kv with FP64
accumulation rounded to FP32, the gate in FP32, output to BF16) and is fed the rows read straight from the tables.
Covered: torch and sm70 impls, prefill sub-chunks, padding rows, TP4 row-parallel wkv through the real forward with
a simulated all-reduce, the wkv loader (bias exactness, refusals), real weights of layers 1 and 14 once the shards
are verified, and L-REF golden tensors when they exist.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import EngramBatchLayout, EngramReqStep, EngramStepPlan

from .test_engram_synth import SyntheticEngram, build_synthetic_engram, random_e4m3_bytes, tokenizer_path

pytestmark = pytest.mark.sm70

GOLDEN_DIR = Path(os.environ.get("DS41_GOLDEN_DIR", "/mnt/nvme2/scratch/ds41/golden"))
ENGRAM_DIR = Path(os.environ.get("DS41_ENGRAM_DIR", "/home/mermiges/ds41-engram"))
E4M3 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float64)


# ------------------------------------------------------------------------------ transcription of the reference
def ref_engram(stream: torch.Tensor, rows: np.ndarray, wkv_w: np.ndarray, wkv_s: np.ndarray, q: torch.Tensor,
               k: torch.Tensor, round_kv_bf16: bool = False) -> torch.Tensor:
    """stream [T,4,5120] bf16 (cpu); rows [T,24,264] uint8 (ALL sub-tables, order-major); wkv [25600,6144] u8 E4M3 +
    [800,192] u8 UE8M0; q/k [4,5120] bf16. Returns [T,4,5120] bf16 (ref:m.py:309-365, FP32 math, FP64 GEMM sum)."""
    T = rows.shape[0]
    vals = E4M3[torch.from_numpy(rows[..., :256]).long()].float()
    sc = torch.pow(2.0, torch.from_numpy(rows[..., 256:]).float() - 127)
    emb = (vals.view(T, 24, 8, 32) * sc.unsqueeze(-1)).view(T, 24 * 256).to(torch.bfloat16)  # ref casts to bf16
    W = E4M3[torch.from_numpy(wkv_w).long()]
    S = torch.pow(2.0, torch.from_numpy(wkv_s).double() - 127)
    W = W * S.repeat_interleave(32, 0).repeat_interleave(32, 1)
    kv = (emb.double() @ W.T).float()
    if round_kv_bf16:
        kv = kv.to(torch.bfloat16).float()
    key, value = kv.split([4 * 5120, 5120], dim=-1)
    key = key.unflatten(-1, (4, 5120))
    weight = q.float() * k.float()
    h = stream.float()
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
    dot = (h * weight * key).sum(-1) * rstd * 5120 ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(torch.bfloat16)


def compare(out: torch.Tensor, ref: torch.Tensor, stream_in: torch.Tensor, tag: str,
            gate: bool = True) -> dict[str, float]:
    o, r = out.float().cpu(), ref.float().cpu()
    assert torch.isfinite(o).all(), f"{tag}: non-finite output"
    rel = ((o - r).norm() / r.norm()).item()
    maxabs = (o - r).abs().max().item()
    ulp = (out.cpu().view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    d_ref = r - stream_in.float()
    d_rel = ((o - r).norm() / d_ref.norm()).item() if d_ref.norm() > 0 else 0.0
    m = dict(rel_rms=rel, max_abs=maxabs, max_ref=r.abs().max().item(), ulp_max=int(ulp.max()),
             frac_ulp_gt0=(ulp > 0).float().mean().item(), delta_rel_rms=d_rel)
    if not gate:
        return m
    # PORT_DESIGN §4.5 module gate (rel-RMS <= 1e-3, max-abs <= 1e-2 x max) + a tighter look at the Engram delta
    assert rel <= 1e-3 and maxabs <= 1e-2 * m["max_ref"], f"{tag}: {m}"
    assert d_rel <= 1e-2, f"{tag}: Engram delta off: {m}"
    return m


# ------------------------------------------------------------------------------ synthetic setup
@pytest.fixture(scope="module")
def syn(tmp_path_factory: pytest.TempPathFactory) -> SyntheticEngram:
    return build_synthetic_engram(tmp_path_factory.mktemp("engram_mod"), seed=7)


@pytest.fixture(autouse=True)
def _knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "0")


def synthetic_wkv(seed: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """FP8 wkv with the measured exponent span -18..-8 (23 % of real blocks are below 2^-14) and bf16 q/k."""
    rng = np.random.default_rng(seed)
    w = random_e4m3_bytes(rng, 25600 * 6144).reshape(25600, 6144)
    s = rng.integers(127 - 18, 127 - 8 + 1, size=(800, 192), dtype=np.uint8)
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(4, 5120, generator=g) * 0.5 + 1).to(torch.bfloat16)
    k = (torch.randn(4, 5120, generator=g) * 0.5 + 1).to(torch.bfloat16)
    return w, s, q, k


def make_module(syn: SyntheticEngram, layer: int, tp_rank: int, tp_size: int, wkv, max_tokens: int = 512):
    from vllm.models.deepseek_v41.common.engram import DeepseekV41Engram
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    svc = EngramHostService(syn.hf_config, (layer,), tp_rank, tp_size, str(syn.row_dir), tokenizer_path(),
                            max_tokens, torch.device("cuda"), io_threads=2)
    with torch.device("cuda"):
        mod = DeepseekV41Engram(None, f"model.layers.{layer}.engram", layer, svc)
    w, s, q, k = wkv
    got = mod.load_weights([("wkv.weight", torch.from_numpy(w).view(torch.float8_e4m3fn)), ("q_weight", q),
                            ("wkv.scale", torch.from_numpy(s).view(torch.float8_e8m0fnu)), ("k_weight", k)])
    # AM-1: consumed names + completed parameters (vLLM checks every named parameter was loaded)
    assert got == {"wkv.weight", "wkv.scale", "q_weight", "k_weight", "wkv_r", "qk"}
    assert {n for n, _ in mod.named_parameters()} <= got and mod.weights_loaded()
    return mod, svc


def bind(svc, step: int, ids: np.ndarray, t_pad: int) -> None:
    svc.begin_step(EngramStepPlan(step, (EngramReqStep("r", 0, len(ids), ids, ids),), frozenset()))
    svc.bind_batch(EngramBatchLayout(step, ("r",), np.array([0, len(ids)], np.int32), len(ids), t_pad, None))


def table_rows(syn: SyntheticEngram, svc, layer: int, ids: np.ndarray) -> np.ndarray:
    """All 24 sub-table rows [T, 24, 264] for the token ids, straight from the tables."""
    from vllm.models.deepseek_v41.common.engram import EngramHasher

    h = EngramHasher(svc.layout, svc.hasher.token_map, (layer,), tuple(range(24)))
    rid = h.hash_positions(h.compress(ids), np.arange(len(ids)))[:, 0]
    return np.concatenate([syn.weights[layer][rid], syn.scales[layer][rid]], axis=-1)


def random_stream(T: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mag = torch.exp(torch.randn(T, 1, 1, generator=g))
    return (torch.randn(T, 4, 5120, generator=g) * mag).to(torch.bfloat16)


# ------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("impl", ["torch", "sm70"])
def test_module_vs_reference_tp1(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch, impl: str) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", impl)
    monkeypatch.setenv("VLLM_DS41_ENGRAM_PREFILL_CHUNK", "64")    # 200 tokens -> 4 sub-chunks
    wkv = synthetic_wkv(11)
    for layer in (1, 14):
        mod, svc = make_module(syn, layer, 0, 1, wkv)
        try:
            ids = np.random.default_rng(layer).integers(3, 129000, size=200).astype(np.int32)
            bind(svc, 0, ids, 208)
            stream = random_stream(208, layer)
            stream[200:] = 0
            out = mod(stream.cuda(), torch.arange(208, device="cuda")).cpu()
            ref = ref_engram(stream[:200], table_rows(syn, svc, layer, ids), *wkv)
            m = compare(out[:200], ref, stream[:200], f"{impl} L{layer}")
            assert torch.equal(out[200:], stream[200:]), "padding rows must pass through unchanged"
            print(f"{impl} layer {layer}: {m}")
        finally:
            svc.shutdown()


def test_module_tp4_row_parallel_matches_tp1(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    """Four ranks (6 sub-tables each) through the real forward; the all-reduce is simulated in two passes."""
    import vllm.distributed as dist

    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", "sm70")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_PREFILL_CHUNK", "48")
    wkv = synthetic_wkv(12)
    ids = np.random.default_rng(5).integers(3, 129000, size=100).astype(np.int32)
    stream = random_stream(100, 9)
    ranks = [make_module(syn, 14, r, 4, wkv) for r in range(4)]
    try:
        for _, svc in ranks:
            bind(svc, 0, ids, 100)
        partials: list[list[torch.Tensor]] = [[] for _ in range(4)]
        for r, (mod, _) in enumerate(ranks):
            monkeypatch.setattr(dist, "tensor_model_parallel_all_reduce", lambda x, r=r: partials[r].append(x.clone()) or x)
            mod(stream.cuda(), torch.arange(100, device="cuda"))
        sums = [sum(partials[r][c] for r in range(4)) for c in range(len(partials[0]))]
        outs = []
        for mod, _ in ranks:
            it = iter(sums)
            monkeypatch.setattr(dist, "tensor_model_parallel_all_reduce", lambda x, it=it: next(it))
            outs.append(mod(stream.cuda(), torch.arange(100, device="cuda")).cpu())
        for o in outs[1:]:
            assert torch.equal(o, outs[0]), "TP ranks disagree after the all-reduce"
        ref = ref_engram(stream, table_rows(syn, ranks[0][1], 14, ids), *wkv)
        print("tp4:", compare(outs[0], ref, stream, "tp4"))
        assert sorted(s for _, svc in ranks for s in svc.subtables) == list(range(24))
    finally:
        for _, svc in ranks:
            svc.shutdown()


def test_tp4_ranks_with_different_row_biases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rank 0's sub-tables need a row bias (scale exponents -20..-12) while the others do not: each rank removes its
    own bias before the all-reduce, so TP4 == the reference (and TP1)."""
    import vllm.distributed as dist

    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", "sm70")
    syn = build_synthetic_engram(tmp_path / "mixed", seed=21)
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    layout = EngramLayout.from_hf_config(syn.hf_config)
    rng = np.random.default_rng(22)
    for s in layout.subtables_for_rank(0, 4):
        lo, n = layout.offsets[1][s], layout.primes[1][s]
        syn.scales[14][lo:lo + n] = rng.integers(127 - 20, 127 - 12 + 1, size=(n, 8), dtype=np.uint8)
    from .test_engram_synth import REAL_DATA_START, write_synthetic_shard

    write_synthetic_shard(syn.row_dir / "model-00048-of-00048.safetensors", 14, syn.weights[14], syn.scales[14],
                          REAL_DATA_START[14])
    wkv = synthetic_wkv(23)
    ids = np.random.default_rng(24).integers(3, 129000, size=64).astype(np.int32)
    stream = random_stream(64, 25)
    ranks = [make_module(syn, 14, r, 4, wkv) for r in range(4)]
    try:
        assert [svc.row_bias(14) for _, svc in ranks] == [5, 0, 0, 0]
        for _, svc in ranks:
            bind(svc, 0, ids, 64)
        partials: list[list[torch.Tensor]] = [[] for _ in range(4)]
        for r, (mod, _) in enumerate(ranks):
            monkeypatch.setattr(dist, "tensor_model_parallel_all_reduce", lambda x, r=r: partials[r].append(x.clone()) or x)
            mod(stream.cuda(), torch.arange(64, device="cuda"))
        sums = iter([sum(partials[r][c] for r in range(4)) for c in range(len(partials[0]))])
        monkeypatch.setattr(dist, "tensor_model_parallel_all_reduce", lambda x: next(sums))
        out = ranks[2][0](stream.cuda(), torch.arange(64, device="cuda")).cpu()
        ref = ref_engram(stream, table_rows(syn, ranks[0][1], 14, ids), *wkv)
        print("tp4 mixed bias:", compare(out, ref, stream, "tp4-mixed-bias"))
    finally:
        for _, svc in ranks:
            svc.shutdown()


def test_wkv_loader_exactness_and_refusals(syn: SyntheticEngram) -> None:
    from vllm.models.deepseek_v41.common.engram import DeepseekV41Engram
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    svc = EngramHostService(syn.hf_config, (1,), 2, 4, str(syn.row_dir), tokenizer_path(), 16,
                            torch.device("cuda"), io_threads=1)
    try:
        w, s, q, k = synthetic_wkv(13)
        with torch.device("cuda"):
            mod = DeepseekV41Engram(None, "model.layers.1.engram", 1, svc)
        with pytest.raises(ValueError, match="skip_checkpoint_weight"):
            mod.load_checkpoint_tensor("embed.weight", torch.zeros(1))
        with pytest.raises(KeyError):
            mod.load_checkpoint_tensor("bogus", torch.zeros(1))
        mod.load_checkpoint_tensor("wkv.weight", torch.from_numpy(w))
        mod.load_checkpoint_tensor("wkv.scale", torch.from_numpy(s))
        with pytest.raises(RuntimeError, match="twice"):
            mod.load_checkpoint_tensor("wkv.weight", torch.from_numpy(w))
        with torch.device("cuda"):
            half = DeepseekV41Engram(None, "model.layers.1.engram", 1, svc)
        with pytest.raises(ValueError, match="not provided"):
            half.load_weights([("q_weight", q), ("k_weight", k)])        # AM-1: a half-loaded Engram never runs
        # wkv_r == exact dequant of this rank's columns x 2^10 (sub-tables 2, 6, 10, 14, 18, 22)
        cols = np.concatenate([np.arange(sub * 256, sub * 256 + 256) for sub in svc.subtables])
        Wd = E4M3[torch.from_numpy(w[:, cols]).long()] * torch.pow(
            2.0, torch.from_numpy(s[:, cols // 32]).double() - 127).repeat_interleave(32, 0)
        assert torch.equal(mod.wkv_r.double().cpu(), Wd * 1024)
        assert (mod.wkv_r.abs() < 2.0**-14).any(), "test data must exercise FP16 subnormals"
        # an exponent that the 2^10 bias cannot bring into FP16's exact range is refused, loudly
        with torch.device("cuda"):
            bad = DeepseekV41Engram(None, "model.layers.1.engram", 1, svc)
        s_bad = s.copy()
        s_bad[3, svc.subtables[0] * 8] = 127 - 30
        bad.load_checkpoint_tensor("wkv.weight", torch.from_numpy(w))
        with pytest.raises(ValueError, match="not exact in FP16"):
            bad.load_checkpoint_tensor("wkv.scale", torch.from_numpy(s_bad))
    finally:
        svc.shutdown()


def _fc(dummy: bool):
    from vllm.forward_context import ForwardContext, override_forward_context

    return override_forward_context(ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={},
                                                   is_dummy_run=dummy))


def test_dummy_forward_is_exact_zero_and_unbound_real_raises(syn: SyntheticEngram,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """AM-2: is_dummy_run -> stream unchanged bit for bit (no host I/O); a real forward without bound rows raises;
    a FULL-mode style capture of the wait raises (it would replay zero rows forever)."""
    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", "sm70")
    mod, svc = make_module(syn, 1, 0, 1, synthetic_wkv(14), max_tokens=8)
    try:
        stream = random_stream(8, 3).cuda()
        before = stream.clone()
        with _fc(True):
            mod(stream, torch.arange(8, device="cuda"))
        torch.cuda.synchronize()
        assert torch.equal(stream, before) and svc.stats()["dummy_steps"] == 1 and svc.stats()["lookups"] == 0
        for ctx in (_fc(False), contextlib.nullcontext()):
            with ctx, pytest.raises(RuntimeError, match="REAL step"):
                mod(stream, torch.arange(8, device="cuda"))
        g = torch.cuda.CUDAGraph()
        with _fc(True), pytest.raises(RuntimeError, match="captured into a CUDA graph"):
            with torch.cuda.graph(g):
                mod(stream, torch.arange(8, device="cuda"))
    finally:
        svc.shutdown()


def test_breakable_capture_dummy_then_replay_real(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture under 1Cat's breakable CUDA graph with a dummy forward context, replay on a real bound step:
    identical to the eager forward (the wait is an eager segment; decode/GEMM/gate are graph segments)."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.models.deepseek_v41.common import engram as E

    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", "sm70")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    E._eager_wait.cache_clear()
    try:
        mod, svc = make_module(syn, 14, 0, 1, synthetic_wkv(15), max_tokens=16)
        try:
            T, t_pad = 6, 8
            ids = np.random.default_rng(4).integers(3, 129000, size=T).astype(np.int32)
            real = random_stream(t_pad, 5).cuda()
            real[T:] = 0
            static = torch.zeros_like(real)
            pos = torch.arange(t_pad, device="cuda")
            cs = torch.cuda.Stream()
            with torch.cuda.stream(cs):
                with _fc(True):
                    mod(static, pos)                               # warm-up (Triton compile) outside capture
                    cap = BreakableCUDAGraphCapture()
                    with cap:
                        mod(static, pos)
            assert cap.num_eager_breaks == 1 and cap.num_graphs == 2
            bind(svc, 0, ids, t_pad)
            static.copy_(real)
            torch.cuda.synchronize()
            with torch.cuda.stream(cs), _fc(False):
                cap.replay()
            torch.cuda.synchronize()
            bind(svc, 1, ids, t_pad)
            eager = real.clone()
            with _fc(False):
                mod(eager, pos)
            torch.cuda.synchronize()
            assert not torch.equal(eager[:T], real[:T]), "Engram contributed nothing"
            assert torch.equal(static, eager), "graph replay differs from the eager forward"
        finally:
            svc.shutdown()
    finally:
        E._eager_wait.cache_clear()


# ------------------------------------------------------------------------------ real weights (shards 47/48)
def _verified_engram_shard(name: str) -> Path:
    p = ENGRAM_DIR / name
    if not (p.is_file() and Path(str(p) + ".sha256-ok").is_file()):
        pytest.skip(f"{p} not downloaded + sha256-verified yet")
    import subprocess

    out = subprocess.run(["bash", "/mnt/nvme2/models/_dl-logs/verify_shard.sh", name], capture_output=True,
                         text=True, timeout=900)
    if out.returncode != 0 or not out.stdout.strip().startswith("OK"):
        pytest.fail(f"verify_shard.sh {name}: {out.stdout} {out.stderr}")
    return p


@pytest.mark.weights
@pytest.mark.parametrize("layer,shard", [(1, "model-00047-of-00048.safetensors"),
                                         (14, "model-00048-of-00048.safetensors")])
def test_module_real_weights_vs_reference(monkeypatch: pytest.MonkeyPatch, layer: int, shard: str) -> None:
    """Real layers 1/14: official tables, wkv, q/k; real token ids (official tokenizer, real text); TP1 and TP4."""
    import json
    import types

    from safetensors import safe_open
    from transformers import PreTrainedTokenizerFast

    from vllm.models.deepseek_v41.common.engram import DeepseekV41Engram
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    path = _verified_engram_shard(shard)
    ref_dir = Path(tokenizer_path()).parent
    hf = types.SimpleNamespace(**json.load(open(ref_dir / "config.json")))
    tok = PreTrainedTokenizerFast(tokenizer_file=str(ref_dir / "tokenizer.json"))
    text = open("/mnt/hdd/v100-research/docs/models/deepseek-v4.1-flash/ENGRAM.md", encoding="utf-8").read()
    ids = np.array([0] + tok.encode(text, add_special_tokens=False)[:383], np.int32)
    with safe_open(str(path), framework="pt") as f:
        pre = f"layers.{layer}.engram."
        w = f.get_tensor(pre + "wkv.weight")
        s = f.get_tensor(pre + "wkv.scale")
        q = f.get_tensor(pre + "q_weight")
        k = f.get_tensor(pre + "k_weight")
        table_w = f.get_slice(pre + "embed.weight")
        table_s = f.get_slice(pre + "embed.scale")
        svc0 = EngramHostService(hf, (layer,), 0, 1, str(ENGRAM_DIR), str(ref_dir), 384, torch.device("cuda"))
        try:
            from vllm.models.deepseek_v41.common.engram import EngramHasher

            hasher = EngramHasher(svc0.layout, svc0.hasher.token_map, (layer,), tuple(range(24)))
            rid = hasher.hash_positions(hasher.compress(ids), np.arange(len(ids)))[:, 0]
            rows = np.zeros((len(ids), 24, 264), np.uint8)
            for t in range(len(ids)):
                for j in range(24):
                    r = int(rid[t, j])
                    rows[t, j, :256] = table_w[r:r + 1].view(torch.uint8).numpy()[0]
                    rows[t, j, 256:] = table_s[r:r + 1].view(torch.uint8).numpy()[0]
        finally:
            svc0.shutdown()
    w8, s8 = w.view(torch.uint8).numpy(), s.view(torch.uint8).numpy()
    g = torch.Generator().manual_seed(layer)
    stream = (torch.randn(len(ids), 4, 5120, generator=g) * 0.05).to(torch.bfloat16)   # bf16 stream, real-ish scale
    ref = ref_engram(stream, rows, w8, s8, q, k)
    ref_bf16kv = ref_engram(stream, rows, w8, s8, q, k, round_kv_bf16=True)
    print(f"L{layer} reference with bf16-rounded kv vs FP32 kv: "
          f"{compare(ref_bf16kv, ref, stream, 'ref-bf16kv', gate=False)}")
    for impl in ("torch", "sm70"):
        monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", impl)
        svc = EngramHostService(hf, (layer,), 0, 1, str(ENGRAM_DIR), str(ref_dir), 384, torch.device("cuda"))
        try:
            with torch.device("cuda"):
                mod = DeepseekV41Engram(None, f"model.layers.{layer}.engram", layer, svc)
            mod.load_weights([("wkv.weight", w), ("wkv.scale", s), ("q_weight", q), ("k_weight", k)])
            bind(svc, 0, ids, len(ids))
            got_rows = svc.wait_rows(layer).cpu().numpy()
            np.testing.assert_array_equal(got_rows, rows[:, list(svc.subtables)])
            out = mod(stream.cuda(), torch.arange(len(ids), device="cuda")).cpu()
            print(f"L{layer} {impl}: {compare(out, ref, stream, f'real L{layer} {impl}')}")
        finally:
            svc.shutdown()


# ------------------------------------------------------------------------------ golden (L-REF) when present
def _golden_cases() -> list[Path]:
    """Final (non-provisional) L-REF cases with Engram applied, v100-semantic mode (the primary comparator)."""
    import json

    out = []
    if GOLDEN_DIR.is_dir():
        for m in sorted(GOLDEN_DIR.glob("*/manifest.json")):
            man = json.load(open(m))
            # "greedy" cases (L-REF final batch, 03:22Z) hold only final-logit tensors over prompt + continuation
            # (no token_ids/start_pos, no per-layer files): the Engram layer goldens live in prefill/decode cases.
            if (man.get("engram_applied") and not man.get("provisional") and man.get("mode") == "v100-semantic"
                    and man.get("phase") != "greedy"):
                out.append(m.parent)
    return out


@pytest.mark.weights
@pytest.mark.parametrize("case", _golden_cases() or [None], ids=lambda c: c.name if c else "none")
def test_engram_vs_golden(case: Path | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per captured Engram layer: hash ids and dequantised rows bitwise; key/value (rel-RMS <= 1e-5) and gate
    (|diff| <= 1e-5) from the module's own decode + FP16 GEMM path; module output stream (see below)."""
    if case is None:
        pytest.skip(f"no final L-REF golden case with engram_applied under {GOLDEN_DIR} yet")
    import json
    import types

    from safetensors import safe_open
    from safetensors.torch import load_file

    from vllm.models.deepseek_v41.common.engram import (
        DeepseekV41Engram, EngramHasher, EngramLayout, decode_rows_torch, load_compressed_token_map)
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "1")
    man = json.load(open(case / "manifest.json"))
    ref_dir = Path(tokenizer_path()).parent
    hf = types.SimpleNamespace(**json.load(open(ref_dir / "config.json")))
    layout = EngramLayout.from_hf_config(hf)
    tmap = load_compressed_token_map(str(ref_dir), layout.compressed_vocab_size)
    hist = np.asarray(list(man.get("history_token_ids") or []) + list(man["token_ids"]), np.int64)
    start, n = int(man["start_pos"]), int(man["num_tokens"])
    assert hist.shape[0] == start + n, (hist.shape, start, n)
    pos = np.arange(start, start + n)
    found = 0
    for layer, shard in ((1, "model-00047-of-00048.safetensors"), (14, "model-00048-of-00048.safetensors")):
        f = case / f"L{layer:02d}.safetensors"
        if not f.is_file():
            continue
        g = load_file(str(f))
        if "engram.hash" not in g:
            continue
        found += 1
        h = EngramHasher(layout, tmap, (layer,), tuple(range(24)))
        np.testing.assert_array_equal(h.hash_positions(h.compress(hist), pos)[:, 0], g["engram.hash"].numpy())
        path = _verified_engram_shard(shard)
        svc = EngramHostService(hf, (layer,), 0, 1, str(ENGRAM_DIR), str(ref_dir), max(n, 8), torch.device("cuda"))
        try:
            with torch.device("cuda"):
                mod = DeepseekV41Engram(None, f"model.layers.{layer}.engram", layer, svc)
            with safe_open(str(path), framework="pt") as fh:
                mod.load_weights([(name, fh.get_tensor(f"layers.{layer}.engram.{name}"))
                                  for name in ("wkv.weight", "wkv.scale", "q_weight", "k_weight")])
            ids32 = hist.astype(np.int32)
            svc.begin_step(EngramStepPlan(0, (EngramReqStep("g", start, n, ids32[start:], ids32),), frozenset()))
            svc.bind_batch(EngramBatchLayout(0, ("g",), np.array([0, n], np.int32), n, n, None))
            rows = svc.wait_rows(layer)
            dec = decode_rows_torch(rows, svc.row_bias(layer)).float().cpu().view(n, 24, 256)
            assert torch.equal(dec, g["engram.rows"].float()), "dequantised rows differ from the golden"
            bias = svc.row_bias(layer)
            kv = torch.mm(decode_rows_torch(rows, bias), mod.wkv_r.t(), out_dtype=torch.float32) * 2.0 ** -(10 + bias)
            key, value = kv[:, :20480].view(n, 4, 5120).cpu(), kv[:, 20480:].cpu()
            for name, ours in (("engram.key", key), ("engram.value", value)):
                ref = g[name].float()
                assert ((ours - ref).norm() / ref.norm()).item() <= 1e-5, name
            h32 = g["stream_in"].float()
            qk = mod.qk.cpu()
            rstd = torch.rsqrt(h32.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
            dot = (h32 * qk * key).sum(-1) * rstd * 5120 ** -0.5
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
            assert (gate - g["engram.gate"].float()).abs().max().item() <= 1e-5
            # The port's stream is BF16 (A4); the v100-semantic golden's is FP32. Rounding the input and the output
            # to BF16 alone costs ~2e-3 rel-RMS, so the module is gated against the reference gate math applied to
            # the same BF16 input with the GOLDEN key/value (<= 1e-3), and the FP32-golden metric is reported.
            h16 = h32.to(torch.bfloat16)
            out = mod(h16.cuda(), torch.from_numpy(pos).cuda()).float().cpu()
            gk, gv = g["engram.key"].float(), g["engram.value"].float()
            hb = h16.float()
            rstd_b = torch.rsqrt(hb.square().mean(-1) + 1e-20) * torch.rsqrt(gk.square().mean(-1) + 1e-20)
            dot_b = (hb * qk * gk).sum(-1) * rstd_b * 5120 ** -0.5
            gate_b = torch.sigmoid(torch.copysign(dot_b.abs().clamp_min(1e-6).sqrt(), dot_b))
            ref_b = (hb + gate_b.unsqueeze(-1) * gv.unsqueeze(-2)).to(torch.bfloat16).float()
            rel_b = ((out - ref_b).norm() / ref_b.norm()).item()
            gold = g["engram.out"].float()
            rel32 = ((out - gold).norm() / gold.norm()).item()
            d_gold = gold - h32
            d_rel = ((out - hb - d_gold).norm() / d_gold.norm()).item()
            print(f"{case.name} L{layer}: stream rel-RMS vs golden-kv/BF16-input ref {rel_b:.2e}; vs FP32 golden "
                  f"{rel32:.2e} (BF16 storage floor); Engram delta vs golden {d_rel:.2e}")
            assert rel_b <= 1e-3 and (out - ref_b).abs().max().item() <= 1e-2 * ref_b.abs().max().item()
        finally:
            svc.shutdown()
    if not found:
        pytest.skip(f"{case}: no engram.hash tensors in this golden case")
