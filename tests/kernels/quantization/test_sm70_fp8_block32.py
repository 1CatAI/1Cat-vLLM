# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only E4M3 with 32 x 32 power-of-two block scales on SM70.

* exact decode: one-hot rows read every weight back through each kernel,
  over all 254 finite E4M3 codes and the extremes of the admitted scale
  window (subnormal products at 2^-14, 57344 at 2^7);
* products against the exact FP32/FP64 dequantization at FP16-output and
  FP32-accumulation-order level, for every dispatch regime;
* refusal of scales outside the exact window (E4M3 max 448 x 2^8 would
  overflow FP16), of NaN codes and of bad shapes;
* the group-128 TurboMind ops are unchanged.
"""

import math

import pytest
import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.utils import sm70_fp8_block32 as b32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an SM70 GPU",
)

# (N, K): the tuned GEMV shapes (DeepSeek-V4.1 dense projections at TP4) and
# shapes that take the generic GEMV configuration.
SHAPES = [
    (1792, 5120),
    (8192, 1280),
    (4096, 1280),
    (1024, 4096),
    (5120, 2048),
    (1152, 5120),
    (5120, 576),
    (5120, 288),
    (2048, 7168),
    (7168, 2048),
    (96, 1056),
]
# One M per dispatch regime of fp8_block32_path, plus their boundaries.
TOKENS = [1, 2, 3, 4, 5, 8, 16, 17, 32, 33, 64, 65, 128, 129, 512, 2048]
PATHS = ("gemv", "turbomind", "dequant")


@pytest.fixture(autouse=True)
def _fp32_cublas_reductions():
    # torch's default lets cuBLAS reduce split-K partials in FP16, which would
    # make the FP16 reference itself noisier than the kernels under test.
    matmul = torch.backends.cuda.matmul
    saved = matmul.allow_fp16_reduced_precision_reduction
    matmul.allow_fp16_reduced_precision_reduction = False
    yield
    matmul.allow_fp16_reduced_precision_reduction = saved


def _rand_fp8(n: int, k: int, seed: int, exp_lo: int = -13, exp_hi: int = -6):
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
    codes[(codes & 0x7F) == 0x7F] = 0x3C  # a checkpoint never stores NaN codes
    e = torch.randint(
        exp_lo, exp_hi + 1, (n // 32, k // 32), device="cuda", generator=g
    )
    return codes.view(torch.float8_e4m3fn), (e + 127).to(torch.uint8)


def _rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = a.double() - b.double()
    ref = b.double()
    return (
        (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
        (d.abs().max() / ref.abs().max()).item(),
    )


def _run(path: str, x: torch.Tensor, w: b32.Fp8Block32Weight, out_dtype: torch.dtype):
    if path == "gemv":
        return b32.fp8_block32_gemv(x, w, out_dtype=out_dtype)
    if path == "turbomind":
        return b32.fp8_block32_turbomind(x, w)
    return b32.fp8_block32_dequant_mm(x, w, out_dtype=out_dtype)


def _all_codes_extreme_scales(n: int, k: int):
    """Every finite E4M3 code in every 32-column window, and block exponents
    alternating between the window ends and the interior."""
    finite = torch.tensor(
        [c for c in range(256) if (c & 0x7F) != 0x7F], dtype=torch.uint8, device="cuda"
    )
    idx = torch.arange(n * k, device="cuda").view(n, k)
    codes = finite[(idx * 37 + idx // k) % finite.numel()]
    lo, hi = b32.SCALE_EXP_MIN, b32.SCALE_EXP_MAX
    exps = torch.tensor([lo, hi, -6, 0, lo + 1, hi - 1], device="cuda")
    bidx = torch.arange((n // 32) * (k // 32), device="cuda").view(n // 32, k // 32)
    return codes.view(torch.float8_e4m3fn), (exps[bidx % exps.numel()] + 127).to(
        torch.uint8
    )


@pytest.mark.parametrize("n,k", [(1152, 5120), (512, 576), (96, 1056)])
@pytest.mark.parametrize("extreme", [False, True])
def test_every_path_decodes_exactly(n, k, extreme):
    """x = one-hot rows, so out[i] = W[:, j_i]: every decoded weight is checked
    bitwise, subnormal products and the largest admitted values included."""
    if extreme:
        w, s = _all_codes_extreme_scales(n, k)
    else:
        w, s = _rand_fp8(n, k, seed=n + k, exp_lo=-14, exp_hi=-6)
    ref = b32.dequant_fp8_block32_reference(w, s)
    assert torch.equal(ref.half().float(), ref)  # every product is an FP16 value
    assert torch.isfinite(ref.half()).all()
    tw = b32.prepare_fp8_block32(w, s)
    assert tw.nbytes == n * k + n * (k // 32) * 2
    torch.testing.assert_close(
        b32.dequant_fp8_block32(tw), ref.t().half(), rtol=0, atol=0
    )
    for beg in range(0, k, 8):
        x = torch.zeros(8, k, dtype=torch.float16, device="cuda")
        x[torch.arange(8), beg + torch.arange(8)] = 1.0
        expect = ref[:, beg : beg + 8].t()
        torch.testing.assert_close(b32.fp8_block32_gemv(x, tw), expect, rtol=0, atol=0)
        torch.testing.assert_close(
            b32.fp8_block32_gemv(x[:2], tw, out_dtype=torch.float16),
            expect[:2].half(),
            rtol=0,
            atol=0,
        )
        if beg % 256 == 0:
            torch.testing.assert_close(
                b32.fp8_block32_turbomind(x, tw), expect.half(), rtol=0, atol=0
            )


@pytest.mark.parametrize("n,k", SHAPES)
def test_paths_match_exact_dequantization(n, k):
    w, s = _rand_fp8(n, k, seed=n * 7 + k)
    ref = b32.dequant_fp8_block32_reference(w, s)
    ref16 = ref.half()
    tw = b32.prepare_fp8_block32(w, s)
    for m in TOKENS:
        g = torch.Generator(device="cuda").manual_seed(m)
        x = torch.randn(m, k, device="cuda", generator=g).half()
        exact = x.double() @ ref.double().t()
        # The load-time FP16 dequantization (cuBLAS): the accuracy to match.
        fb16_rms, _ = _rel(x @ ref16.t(), exact)
        fb32_rms, _ = _rel(torch.mm(x, ref16.t(), out_dtype=torch.float32), exact)
        for path in PATHS:
            if path == "gemv" and m > b32.GEMV_MAX_M:
                continue
            out = _run(path, x, tw, torch.float16)
            assert out.dtype == torch.float16 and out.shape == (m, n)
            rel_rms, rel_max = _rel(out, exact)
            # FP16 output rounding floor (2^-11 relative, ~3e-4 RMS) ...
            assert rel_rms < 3e-4 and rel_max < 1e-3, (n, k, m, path, rel_rms, rel_max)
            # ... and no worse than the FP16 dequantized weight with cuBLAS.
            assert rel_rms < 1.5 * fb16_rms + 1e-6, (n, k, m, path, rel_rms, fb16_rms)
            if path != "turbomind":
                out32 = _run(path, x, tw, torch.float32)
                assert out32.dtype == torch.float32
                # FP32 accumulation-order level: like cuBLAS with FP32 output,
                # or within the sqrt(K) * 2^-24 order-of-summation bound where
                # cuBLAS happens to pick a luckier order on a tiny sample.
                rel32 = _rel(out32, exact)[0]
                bound = max(2.0 * fb32_rms, 2.0 * math.sqrt(k) * 2.0**-24)
                assert rel32 < bound, (n, k, m, path, rel32, fb32_rms)
        policy = b32.fp8_block32_linear(x, tw)
        torch.testing.assert_close(
            policy,
            _run(b32.fp8_block32_path(m, n, k), x, tw, torch.float16),
            rtol=0,
            atol=0,
        )


def test_dispatch_regimes():
    """GEMV for small M (up to 4 rows when N <= 1152), TurboMind for mid M
    (longer for wide and deep products), dequant + cuBLAS above; FP32 output:
    GEMV up to 8 rows."""

    def paths(n, k, ms, dtype=torch.float16):
        return [b32.fp8_block32_path(m, n, k, dtype) for m in ms]

    assert paths(1152, 5120, (4, 5, 16, 17)) == [
        "gemv",
        "turbomind",
        "turbomind",
        "dequant",
    ]
    assert paths(1792, 5120, (2, 3, 32, 33)) == [
        "gemv",
        "turbomind",
        "turbomind",
        "dequant",
    ]
    assert paths(5120, 576, (2, 3, 64, 65)) == [
        "gemv",
        "turbomind",
        "turbomind",
        "dequant",
    ]
    assert paths(8192, 1280, (2, 3, 128, 129)) == [
        "gemv",
        "turbomind",
        "turbomind",
        "dequant",
    ]
    assert paths(4096, 1280, (1, 8, 9), torch.float32) == ["gemv", "gemv", "dequant"]
    with pytest.raises(TypeError):
        b32.fp8_block32_path(1, 4096, 1280, torch.bfloat16)


def test_alpha_out_buffer_and_empty_batch():
    w, s = _rand_fp8(256, 1024, seed=5)
    tw = b32.prepare_fp8_block32(w, s)
    x = torch.randn(5, 1024, device="cuda").half()
    base = b32.fp8_block32_gemv(x, tw)
    torch.testing.assert_close(
        b32.fp8_block32_gemv(x, tw, alpha=2.0**-4), base * 2.0**-4, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="at most 8"):
        b32.fp8_block32_gemv(
            torch.zeros(9, 1024, dtype=torch.float16, device="cuda"), tw
        )
    for m in (2, 5, 100):
        x = torch.randn(m, 1024, device="cuda").half()
        out = torch.full((m, 256), float("nan"), dtype=torch.float16, device="cuda")
        assert b32.fp8_block32_linear(x, tw, out=out) is out
        torch.testing.assert_close(out, b32.fp8_block32_linear(x, tw), rtol=0, atol=0)
    with pytest.raises(TypeError, match="out must be"):
        b32.fp8_block32_linear(x, tw, out=torch.empty(100, 256, device="cuda"))
    for dt in (torch.float16, torch.float32):
        assert b32.fp8_block32_linear(x[:0], tw, out_dtype=dt).shape == (0, 256)
    with pytest.raises(TypeError, match="fp16"):
        b32.fp8_block32_linear(torch.randn(2, 1024, device="cuda"), tw)


def test_custom_op_matches_policy():
    w, s = _rand_fp8(512, 2048, seed=6)
    tw = b32.prepare_fp8_block32(w, s)
    for m in (1, 3, 40, 300):
        x = torch.randn(m, 2048, device="cuda").half()
        out = torch.empty(m, 512, dtype=torch.float16, device="cuda")
        torch.ops.vllm.sm70_fp8_block32_linear(
            out, x, tw.tm_weight, tw.tm_scales, tw.k_ld, tw.q_ld
        )
        torch.testing.assert_close(out, b32.fp8_block32_linear(x, tw), rtol=0, atol=0)


@pytest.mark.parametrize(
    "exponent", [b32.SCALE_EXP_MIN - 1, b32.SCALE_EXP_MAX + 1, -18, 15]
)
def test_refuses_scales_outside_the_exact_window(exponent):
    w, s = _rand_fp8(256, 1024, seed=2)
    s[1, 3] = 127 + exponent
    assert "exact FP16 window" in b32.block32_scale_ineligibility(tuple(w.shape), s)
    with pytest.raises(ValueError, match="exact FP16 window"):
        b32.prepare_fp8_block32(w, s)


@pytest.mark.parametrize("code", [0x7F, 0xFF])
def test_refuses_nan_codes(code):
    """E4M3 NaN bytes would decode to finite +-480 in the GEMV and TurboMind
    converters, so prepare refuses them instead of serving a wrong value."""
    w, s = _rand_fp8(256, 1024, seed=8)
    w.view(torch.uint8)[17, 333] = code
    with pytest.raises(ValueError, match="NaN codes"):
        b32.prepare_fp8_block32(w, s)


def test_refuses_inexact_scales_and_shapes():
    w, s = _rand_fp8(256, 1024, seed=2)
    assert b32.block32_scale_ineligibility(tuple(w.shape), s) is None
    fp32 = torch.exp2(s.float() - 127.0)
    assert b32.block32_scale_ineligibility(tuple(w.shape), fp32) is None
    fp32[0, 0] *= 0.75
    with pytest.raises(ValueError, match="powers of two"):
        b32.prepare_fp8_block32(w, fp32)
    with pytest.raises(ValueError, match="scale shape"):
        b32.prepare_fp8_block32(w, s[:, :16])
    with pytest.raises(ValueError, match="N % 32"):
        b32.prepare_fp8_block32(w[:200], s[:7])
    assert "dtype" in b32.block32_scale_ineligibility((256, 1024), s.half())
    with pytest.raises(TypeError, match="E4M3"):
        b32.prepare_fp8_block32(w.view(torch.uint8).view(torch.float8_e5m2), s)


def test_group128_ops_unchanged():
    """128 x 128 block FP8 keeps running through the same ops and tiles."""
    n, k = 1024, 2048
    w, _ = _rand_fp8(n, k, seed=3)
    g = torch.Generator(device="cuda").manual_seed(4)
    scales = torch.exp2(
        torch.randint(-12, -6, (n // 128, k // 128), device="cuda", generator=g).float()
    )
    ref = w.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    tm_w, tm_s, meta = sm70_ops.fp8_sm70_prepare(w, scales, 128)
    for m in (1, 16, 300):
        x = torch.randn(m, k, device="cuda").half()
        out = torch.empty(m, n, dtype=torch.float16, device="cuda")
        sm70_ops.fp8_gemm_sm70_out(
            out, x, tm_w, tm_s, 128, int(meta[0]), int(meta[1]), False
        )
        rel_rms, _ = _rel(out, x.double() @ ref.double().t())
        assert rel_rms < 3e-4, (m, rel_rms)
    dense = torch.empty(k, n, dtype=torch.float16, device="cuda")
    sm70_ops.fp8_sm70_dequantize_out(dense, tm_w, tm_s, 128)
    torch.testing.assert_close(dense, ref.t().half(), rtol=0, atol=0)


def test_group32_refuses_unqualified_variants():
    w, s = _rand_fp8(256, 1024, seed=7)
    scales = torch.exp2(s.float() - 127.0)
    with pytest.raises(RuntimeError, match="gated-SiLU"):
        sm70_ops.fp8_sm70_prepare(w, scales, 32, True)
    tw = b32.prepare_fp8_block32(w, s)
    x = torch.randn(4, 1024, device="cuda").half()
    out = torch.empty(4, 256, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="group_size 128, or 32"):
        sm70_ops.fp8_gemm_sm70_out(
            out, x, tw.tm_weight, tw.tm_scales, 32, tw.k_ld, tw.q_ld, True
        )
    with pytest.raises(RuntimeError, match="only group_size 128 or 32"):
        sm70_ops.fp8_sm70_prepare(w, torch.exp2(s.float()[:, ::2] - 127.0), 64)
