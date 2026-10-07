# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-MOE P5: DeepSeek-V4.1 dense FP8 (E4M3, 32 x 32 UE8M0 blocks, PORT_DESIGN §1 first row) kept FP8 on SM70 —
the decode GEMV, TurboMind group-32 HMMA GEMM and dequant + cuBLAS paths of ``sm70/fp8_g32.py`` — against the exact
FP32 dequantisation and the P2 FP16 dequant fallback.

* decode exactness: one-hot rows read back every weight through each path; dequant is bitwise the fallback's weight;
* products at the V4.1 dense shapes (TP4-local and replicated), M in {1, 2, 3, 8, 64, 512, 4096}: FP32
  accumulation, FP16 output at the FP16 rounding floor (rel-RMS ~2e-4) or FP32 output at accumulation-order level;
* inexact scales are refused (Engram exponents, non powers of two); the group-128 op path is unchanged.

Bench (not collected): ``python -m tests.models.deepseek_v41.test_moe_fp8g32 --out <dir>`` times every path against
the FP16 cuBLAS fallback as CUDA-graph replays with weight rotation (V100 L2 = 6 MB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

from vllm.models.deepseek_v41.sm70 import fp8_g32

pytestmark = pytest.mark.sm70

# (name, N, K): TP4-local shapes of one decoder layer plus the replicated ones
SHAPES = [
    ("wq_a+wkv", 1792, 5120),      # fused wq_a (q_lora 1280) + wkv (latent 512), replicated
    ("wq_b_tp4", 8192, 1280),      # 16 of 64 heads x 512
    ("indexer_wq_b", 4096, 1280),  # 32 heads x 128, replicated
    ("wo_a_group", 1024, 4096),    # one o_lora group (2 per rank at TP4)
    ("wo_b_tp4", 5120, 2048),      # row-parallel slice, FP32 output
    ("shared_w13_tp4", 1152, 5120),
    ("shared_w2_tp4", 5120, 576),  # FP32 output
    ("shared_w2_tp8", 5120, 288),
]
TOKENS = [1, 2, 3, 8, 64, 512, 4096]
PATHS = ("gemv", "turbomind", "dequant")


@pytest.fixture(autouse=True)
def _fp32_cublas_reductions():
    """PORT_DESIGN §4.1: the model sets ``allow_fp16_reduced_precision_reduction = False`` at init (torch's default
    True lets cuBLAS split-K reduce in FP16: rel-RMS 3.6e-4 instead of 2.1e-4 at M=512)."""
    matmul = torch.backends.cuda.matmul
    saved = matmul.allow_fp16_reduced_precision_reduction
    matmul.allow_fp16_reduced_precision_reduction = False
    yield
    matmul.allow_fp16_reduced_precision_reduction = saved


def _rand_fp8(n: int, k: int, seed: int, exp_lo: int = -13, exp_hi: int = -6):
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
    codes[(codes & 0x7F) == 0x7F] = 0x3C  # no NaN codes
    e = torch.randint(exp_lo, exp_hi + 1, (n // 32, k // 32), device="cuda", generator=g)
    return codes.view(torch.float8_e4m3fn), (e + 127).to(torch.uint8)


def _rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = a.float() - b.float()
    return ((d.pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt()).item(),
            (d.abs().max() / b.float().abs().max()).item())


def _run(path: str, x: torch.Tensor, w: fp8_g32.Fp8G32Weight, out_dtype: torch.dtype) -> torch.Tensor:
    if path == "gemv":
        return fp8_g32.fp8_g32_gemv(x, w, out_dtype=out_dtype)
    if path == "turbomind":
        return fp8_g32.fp8_g32_turbomind(x, w)
    return fp8_g32.fp8_g32_dequant_mm(x, w, out_dtype=out_dtype)


@pytest.mark.parametrize("n,k", [(1152, 5120), (512, 576)])
def test_every_path_decodes_exactly(n, k):
    """x = one-hot rows: out[j] = W[:, j], so every decoded weight is checked bitwise (subnormals included)."""
    w, s = _rand_fp8(n, k, seed=n + k, exp_lo=-14, exp_hi=-6)
    ref = fp8_g32.dequant_fp8_g32_reference(w, s)
    assert torch.equal(ref.half().float(), ref)  # E4M3 x 2^e is an FP16 value
    tw = fp8_g32.prepare_fp8_g32(w, s)
    torch.testing.assert_close(fp8_g32.dequant_fp8_g32(tw), ref.t().half(), rtol=0, atol=0)
    for beg in range(0, k, 8):
        x = torch.zeros(8, k, dtype=torch.float16, device="cuda")
        x[torch.arange(8), beg + torch.arange(8)] = 1.0
        expect = ref[:, beg:beg + 8].t()
        torch.testing.assert_close(fp8_g32.fp8_g32_gemv(x, tw), expect, rtol=0, atol=0)
        if beg % 1024 == 0:
            torch.testing.assert_close(fp8_g32.fp8_g32_turbomind(x, tw), expect.half(), rtol=0, atol=0)


@pytest.mark.parametrize("name,n,k", SHAPES)
def test_paths_match_fp32_dequant(name, n, k):
    w, s = _rand_fp8(n, k, seed=n * 7 + k)
    ref = fp8_g32.dequant_fp8_g32_reference(w, s)
    tw = fp8_g32.prepare_fp8_g32(w, s)
    assert tw.nbytes == n * k + n * (k // 32) * 2
    for m in TOKENS:
        g = torch.Generator(device="cuda").manual_seed(m)
        x = torch.randn(m, k, device="cuda", generator=g).half()
        exact = x.float() @ ref.t()
        fb_rms, _ = _rel(x @ ref.half().t(), exact)  # the P2 FP16 dequant fallback (cuBLAS)
        exact64 = x.double() @ ref.double().t()
        fb32_rms, _ = _rel(torch.mm(x, ref.half().t(), out_dtype=torch.float32), exact64)
        for path in PATHS:
            if path == "gemv" and m > fp8_g32.GEMV_MAX_M:
                continue
            out = _run(path, x, tw, torch.float16)
            rel_rms, rel_max = _rel(out, exact)
            assert rel_rms < 3e-4 and rel_max < 1e-3, (name, m, path, rel_rms, rel_max)  # FP16 output floor
            assert rel_rms < 1.5 * fb_rms + 1e-6, (name, m, path, rel_rms, fb_rms)
            if path != "turbomind":
                out32 = _run(path, x, tw, torch.float32)
                assert out32.dtype == torch.float32
                # FP32 accumulation-order level: no worse than the FP32-output fallback (both ~sqrt(K) * 2^-24)
                rel32 = _rel(out32, exact64)[0]
                assert rel32 < 2.0 * fb32_rms + 2e-7, (name, m, path, rel32, fb32_rms)
        assert fp8_g32.fp8_g32_linear(x, tw).dtype == torch.float16


def test_policy_and_alpha():
    assert [fp8_g32.fp8_g32_path(m) for m in (1, 2, 3, 64, 65)] == ["gemv", "gemv", "turbomind", "turbomind",
                                                                     "dequant"]
    assert [fp8_g32.fp8_g32_path(m, torch.float32) for m in (1, 8, 9)] == ["gemv", "gemv", "dequant"]
    with pytest.raises(TypeError):
        fp8_g32.fp8_g32_path(1, torch.bfloat16)
    w, s = _rand_fp8(256, 1024, seed=5)
    tw = fp8_g32.prepare_fp8_g32(w, s)
    x = torch.randn(5, 1024, device="cuda").half()
    base = fp8_g32.fp8_g32_gemv(x, tw)
    torch.testing.assert_close(fp8_g32.fp8_g32_gemv(x, tw, alpha=2.0**-4), base * 2.0**-4, rtol=0, atol=0)
    with pytest.raises(ValueError, match="at most 8"):
        fp8_g32.fp8_g32_gemv(torch.zeros(9, 1024, dtype=torch.float16, device="cuda"), tw)


def test_out_buffer_and_empty_batch():
    w, s = _rand_fp8(256, 1024, seed=1)
    tw = fp8_g32.prepare_fp8_g32(w, s)
    for m in (2, 5, 100):
        x = torch.randn(m, 1024, device="cuda").half()
        out = torch.full((m, 256), float("nan"), dtype=torch.float16, device="cuda")
        assert fp8_g32.fp8_g32_linear(x, tw, out=out) is out
        torch.testing.assert_close(out, fp8_g32.fp8_g32_linear(x, tw), rtol=0, atol=0)
    with pytest.raises(TypeError, match="out must be"):
        fp8_g32.fp8_g32_linear(x, tw, out=torch.empty(100, 256, device="cuda"))
    for dt in (torch.float16, torch.float32):
        assert fp8_g32.fp8_g32_linear(x[:0], tw, out_dtype=dt).shape == (0, 256)


def test_refuses_inexact_scales_and_shapes():
    w, s = _rand_fp8(256, 1024, seed=2)
    s_low = s.clone()
    s_low[0, 0] = 127 - 18  # Engram wkv-like exponent: not an FP16 normal
    with pytest.raises(ValueError, match="exact FP16"):
        fp8_g32.prepare_fp8_g32(w, s_low)
    with pytest.raises(ValueError, match="powers of two"):
        fp8_g32.prepare_fp8_g32(w, torch.full((8, 32), 0.75, device="cuda"))
    with pytest.raises(ValueError, match="scale shape"):
        fp8_g32.prepare_fp8_g32(w, s[:, :16])
    with pytest.raises(ValueError, match="N % 32"):
        fp8_g32.prepare_fp8_g32(w[:200], s[:7])
    tw = fp8_g32.prepare_fp8_g32(w, s)
    with pytest.raises(TypeError, match="fp16"):
        fp8_g32.fp8_g32_linear(torch.randn(2, 1024, device="cuda"), tw)


@pytest.mark.parametrize("exp", [-14, 7])
def test_exponent_window_edges_are_exact_on_every_path(exp):
    """At the window edges every path returns E4M3 x 2^e exactly: 448 x 2^7 = 57344 (largest finite), and the
    smallest subnormal 2^-9 x 2^-14."""
    n, k = 64, 256
    codes = torch.full((n, k), 0x01, dtype=torch.uint8, device="cuda")      # 2^-9
    codes[:, ::2] = 0x7E                                                    # 448
    codes[1::2] |= 0x80                                                     # negative rows
    w = codes.view(torch.float8_e4m3fn)
    s = torch.full((n // 32, k // 32), 127 + exp, dtype=torch.uint8, device="cuda")
    ref = fp8_g32.dequant_fp8_g32_reference(w, s)
    assert torch.equal(ref.half().float(), ref) and torch.isfinite(ref.half()).all()
    tw = fp8_g32.prepare_fp8_g32(w, s)
    torch.testing.assert_close(fp8_g32.dequant_fp8_g32(tw), ref.t().half(), rtol=0, atol=0)
    x = torch.zeros(8, k, dtype=torch.float16, device="cuda")
    x[torch.arange(8), torch.arange(8)] = 1.0
    expect = ref[:, :8].t()
    for dt in (torch.float32, torch.float16):
        torch.testing.assert_close(fp8_g32.fp8_g32_gemv(x, tw, out_dtype=dt), expect.to(dt), rtol=0, atol=0)
    torch.testing.assert_close(fp8_g32.fp8_g32_turbomind(x, tw), expect.half(), rtol=0, atol=0)


def test_refuses_overflowing_exponent():
    """e = 8: 448 x 2^8 = 114688 is not an FP16 value (FP16 paths would give inf/NaN, the FP32 GEMV 114688)."""
    w, s = _rand_fp8(64, 256, seed=6)
    s_hi = s.clone()
    s_hi[1, 3] = 127 + 8
    with pytest.raises(ValueError, match=r"outside the exact FP16 window \[-14, 7\]"):
        fp8_g32.prepare_fp8_g32(w, s_hi)


@pytest.mark.parametrize("code", [0x7F, 0xFF])
def test_refuses_nan_codes(code):
    """E4M3fn NaN bytes would decode as +-480 on the g32 paths but NaN in the FP16 fallback: refuse them."""
    w, s = _rand_fp8(64, 256, seed=7)
    bad = w.view(torch.uint8).clone()
    bad[5, 17] = code
    assert torch.isnan(bad.view(torch.float8_e4m3fn).float()[5, 17])
    with pytest.raises(ValueError, match="1 E4M3 NaN codes"):
        fp8_g32.prepare_fp8_g32(bad.view(torch.float8_e4m3fn), s)
    with pytest.raises(ValueError, match="NaN codes"):
        fp8_g32.prepare_fp8_g32(bad, s)  # uint8 bytes, same check


def test_group128_path_unchanged():
    """DeepSeek-V4 / generic 128 x 128 block FP8 keeps running through the same ops."""
    from vllm import _sm70_ops as sm70_ops

    n, k = 1024, 2048
    w, _ = _rand_fp8(n, k, seed=3)
    g = torch.Generator(device="cuda").manual_seed(4)
    scales = torch.exp2(torch.randint(-12, -6, (n // 128, k // 128), device="cuda", generator=g).float())
    ref = w.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    tm_w, tm_s, meta = sm70_ops.fp8_sm70_prepare(w, scales, 128)
    for m in (1, 16, 300):
        x = torch.randn(m, k, device="cuda").half()
        out = torch.empty(m, n, dtype=torch.float16, device="cuda")
        sm70_ops.fp8_gemm_sm70_out(out, x, tm_w, tm_s, 128, int(meta[0]), int(meta[1]), False)
        rel_rms, _ = _rel(out, x.float() @ ref.t())
        assert rel_rms < 3e-4, (m, rel_rms)
    dense = torch.empty(k, n, dtype=torch.float16, device="cuda")
    sm70_ops.fp8_sm70_dequantize_out(dense, tm_w, tm_s, 128)
    torch.testing.assert_close(dense, ref.t().half(), rtol=0, atol=0)


# ---------------------------------------------------------------------------------------------- bench (CLI)


def bench(args: argparse.Namespace) -> dict:
    from tests.models.deepseek_v41.test_moe_bench import graph_us, roofline_us

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # as the model runs (§4.1)

    rows = []
    for name, n, k in SHAPES:
        copies = max(2, (24 << 20) // (n * k) + 1)  # rotate > 4x L2 of FP8 bytes
        raw = [_rand_fp8(n, k, seed=i) for i in range(copies)]
        tws = [fp8_g32.prepare_fp8_g32(w, s) for w, s in raw]
        fbs = [fp8_g32.dequant_fp8_g32_reference(w, s).half() for w, s in raw]
        for m in args.tokens:
            x = torch.randn(m, k, device="cuda").half()
            it = {"w": 0}

            def nxt(pool: list):
                it["w"] = (it["w"] + 1) % len(pool)
                return pool[it["w"]]

            row = {"name": name, "n": n, "k": k, "m": m,
                   "fp8_roofline_us": round(roofline_us(n * k * 1.0625 + 2 * m * (n + k), args.bw), 2),
                   "fp16_roofline_us": round(roofline_us(n * k * 2 + 2 * m * (n + k), args.bw), 2)}
            out16 = torch.empty(m, n, dtype=torch.float16, device="cuda")
            out32 = torch.empty(m, n, dtype=torch.float32, device="cuda")
            row["fp16_cublas_us"] = round(graph_us(lambda: torch.mm(x, nxt(fbs).t(), out=out16)), 2)
            row["fp16_cublas_f32out_us"] = round(graph_us(
                lambda: torch.mm(x, nxt(fbs).t(), out_dtype=torch.float32, out=out32)), 2)
            if m <= fp8_g32.GEMV_MAX_M:
                row["gemv_us"] = round(graph_us(lambda: fp8_g32.fp8_g32_gemv(x, nxt(tws), out=out32)), 2)
            row["turbomind_us"] = round(graph_us(lambda: fp8_g32.fp8_g32_turbomind(x, nxt(tws), out=out16)), 2)
            row["dequant_us"] = round(graph_us(lambda: fp8_g32.fp8_g32_dequant_mm(x, nxt(tws), out=out16)), 2)
            row["policy"] = fp8_g32.fp8_g32_path(m)
            row["policy_us"] = round(graph_us(lambda: fp8_g32.fp8_g32_linear(x, nxt(tws), out=out16)), 2)
            row["policy_f32"] = fp8_g32.fp8_g32_path(m, torch.float32)
            row["policy_f32_us"] = round(graph_us(
                lambda: fp8_g32.fp8_g32_linear(x, nxt(tws), out_dtype=torch.float32, out=out32)), 2)
            rows.append(row)
            print(json.dumps(row), flush=True)
        del tws, fbs, raw
        torch.cuda.empty_cache()
    return {"bw_gbs": args.bw, "rows": rows}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 3, 4, 8, 16, 32, 64, 128, 512, 4096])
    ap.add_argument("--bw", type=float, default=800.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    report = bench(args)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "fp8_g32_bench.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
    sys.exit(0)
