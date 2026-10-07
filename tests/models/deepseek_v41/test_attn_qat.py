# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN: QAT twins are bit-exact (PORT_DESIGN §4.2, §4.5 row 1: 0 mismatches).

Three implementations must agree bit for bit:
  * ``_ref_*``: a numpy transcription of ``inference/kernel.py`` (act_quant / fp4_quant_kernel with
    ``inplace=True``) written independently here -- the grids are enumerated explicitly and the nearest
    value is searched with ties to the even mantissa;
  * ``qat.*_torch`` (float8_e4m3fn casts + E2M1 threshold table);
  * ``qat.*`` Triton (integer RNE on the FP32 bit pattern).
Inputs: random blocks over 2^-30..2^12, every finite FP16 value under several block scales, exact
rounding ties (incl. ties of ``x / s`` for non-power-of-two E4M3 scales), zeros / -0 / FP16
subnormals / +-65504, BF16 and FP32 inputs, non-finite blocks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common import qat

pytestmark = pytest.mark.sm70

F32 = np.float32


# ----------------------------------------------------------------------------- transcription
def _e4m3_grid() -> tuple[np.ndarray, np.ndarray]:
    vals, par = [0.0], [0]
    for m in range(1, 8):                      # subnormals m * 2^-9
        vals.append(m * 2.0**-9)
        par.append(m & 1)
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:             # 0x7F is NaN in E4M3FN
                continue
            vals.append((1 + m / 8) * 2.0 ** (e - 7))
            par.append(m & 1)
    order = np.argsort(vals)
    return np.asarray(vals, np.float64)[order], np.asarray(par)[order]


E4M3_GRID, E4M3_PAR = _e4m3_grid()
E2M1_GRID = np.asarray([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)
E2M1_PAR = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
assert E4M3_GRID[-1] == 448.0 and len(E4M3_GRID) == 127


def _round_grid(y: np.ndarray, grid: np.ndarray, par: np.ndarray) -> np.ndarray:
    a = np.minimum(np.abs(y.astype(np.float64)), grid[-1])
    hi = np.clip(np.searchsorted(grid, a, side="left"), 1, len(grid) - 1)
    lo = hi - 1
    dlo, dhi = a - grid[lo], grid[hi] - a
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (par[hi] == 0))
    mag = np.where(pick_hi, grid[hi], grid[lo]).astype(F32)
    out = np.copysign(mag, y).astype(F32)
    return np.where(np.isnan(y), y, out).astype(F32)


def _ceil_log2(v: np.ndarray) -> np.ndarray:
    bits = v.astype(F32).view(np.uint32)
    return ((bits >> 23) & 0xFF).astype(np.int64) - 127 + ((bits & 0x7FFFFF) != 0)


def _pow2(k: np.ndarray) -> np.ndarray:
    return np.ldexp(F32(1.0), k).astype(F32)


def _ref_qdq(x32: np.ndarray, block: int, kind: str) -> np.ndarray:
    """Transcription of kernel.py act_quant_kernel / fp4_quant_kernel (inplace=True), FP32 result."""
    xb = x32.astype(F32).reshape(*x32.shape[:-1], -1, block)
    with np.errstate(invalid="ignore", over="ignore"):
        amax = np.abs(xb).max(axis=-1)
        finite = np.isfinite(amax)
        amax = np.where(finite, amax, F32(1.0)).astype(F32)
        if kind == "fp8":
            amax = np.maximum(amax, F32(1e-4))
            s = _pow2(_ceil_log2((amax * F32(1.0 / 448.0)).astype(F32)))
            y = _round_grid(np.clip((xb / s[..., None]).astype(F32), F32(-448), F32(448)), E4M3_GRID, E4M3_PAR)
        elif kind == "e8m0":
            amax = np.maximum(amax, F32(6 * 2.0**-126))
            s = _pow2(_ceil_log2((amax * F32(1.0 / 6.0)).astype(F32)))
            y = _round_grid(np.clip((xb / s[..., None]).astype(F32), F32(-6), F32(6)), E2M1_GRID, E2M1_PAR)
        else:  # e4m3 scales, block 16
            amax = np.maximum(amax, F32(6 * 2.0**-9))
            s = _round_grid(np.minimum((amax / F32(6.0)).astype(F32), F32(448)), E4M3_GRID, E4M3_PAR)
            y = _round_grid(np.clip((xb / s[..., None]).astype(F32), F32(-6), F32(6)), E2M1_GRID, E2M1_PAR)
        y = (y * s[..., None]).astype(F32)
        y = np.where(finite[..., None], y, F32(np.nan))
    return y.reshape(x32.shape)


KINDS = {
    "fp8": (qat.fp8_block32_qdq, qat.fp8_block32_qdq_torch, 32),
    "e8m0": (qat.fp4_e8m0_qdq, qat.fp4_e8m0_qdq_torch, 32),
    "e4m3": (qat.fp4_e4m3_qdq, qat.fp4_e4m3_qdq_torch, 16),
}


def _bits_equal(a: np.ndarray, b: np.ndarray) -> tuple[int, str]:
    """Bitwise equality treating any NaN == any NaN. Returns (mismatches, first example)."""
    a = np.ascontiguousarray(a.astype(F32)).reshape(-1)
    b = np.ascontiguousarray(b.astype(F32)).reshape(-1)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = (a.view(np.uint32) != b.view(np.uint32)) & ~both_nan
    n = int(diff.sum())
    ex = ""
    if n:
        i = int(np.flatnonzero(diff)[0])
        ex = f"idx {i}: {a[i]!r} ({a.view(np.uint32)[i]:#010x}) vs {b[i]!r} ({b.view(np.uint32)[i]:#010x})"
    return n, ex


def _check(x: torch.Tensor, kind: str) -> None:
    fn, fn_torch, block = KINDS[kind]
    x = x.cuda()
    ref = _ref_qdq(x.float().cpu().numpy(), block, kind)
    tri = fn(x, out_dtype=torch.float32, impl="triton").cpu().numpy()
    tor = fn_torch(x, out_dtype=torch.float32).cpu().numpy()
    n1, e1 = _bits_equal(tri, ref)
    n2, e2 = _bits_equal(tor, ref)
    assert n1 == 0, f"{kind} triton vs transcription: {n1} mismatches; {e1}"
    assert n2 == 0, f"{kind} torch twin vs transcription: {n2} mismatches; {e2}"
    # FP16 records: the cast of the exact FP32 value, identical in both impls
    tri16 = fn(x, out_dtype=torch.float16, impl="triton")
    tor16 = fn_torch(x, out_dtype=torch.float16)
    assert torch.equal(tri16.view(torch.int16), tor16.view(torch.int16)) or (
        torch.isnan(tri16) | torch.isnan(tor16) | (tri16.view(torch.int16) == tor16.view(torch.int16))).all()
    ref16 = torch.from_numpy(ref).to(torch.float16).cuda()
    same = (tri16.view(torch.int16) == ref16.view(torch.int16)) | (torch.isnan(tri16) & torch.isnan(ref16))
    assert bool(same.all()), f"{kind} fp16 records differ from the cast transcription"


# ----------------------------------------------------------------------------- inputs
def _random_blocks(rows: int, dim: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dim, generator=g, dtype=torch.float64)
    scale = torch.pow(2.0, torch.randint(-30, 13, (rows, dim // 16), generator=g).double())
    x = x * scale.repeat_interleave(16, dim=1)
    if dtype == torch.float16:
        x = x.clamp(-65504, 65504)
    return x.to(dtype)


def _all_fp16_values() -> torch.Tensor:
    v = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16)
    return v[torch.isfinite(v)]


def _anchor_blocks(values: torch.Tensor, anchor: float, block: int) -> torch.Tensor:
    """Every value with |v| <= anchor, packed (block-1) per block next to the anchor that fixes amax."""
    v = values[values.float().abs() <= anchor]
    per = block - 1
    pad = (-v.numel()) % per
    v = torch.cat([v, torch.zeros(pad, dtype=v.dtype)]).view(-1, per)
    a = torch.full((v.shape[0], 1), anchor, dtype=v.dtype)
    blocks = torch.cat([a, v], dim=1)
    per_row = 512 // block
    extra = (-blocks.shape[0]) % per_row
    if extra:
        filler = torch.zeros(extra, block, dtype=v.dtype)
        filler[:, 0] = anchor
        blocks = torch.cat([blocks, filler])
    return blocks.reshape(-1, 512)


@pytest.mark.parametrize("kind", ["fp8", "e8m0", "e4m3"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_random_blocks(kind: str, dtype: torch.dtype) -> None:
    _check(_random_blocks(512, 512, dtype, seed=hash((kind, str(dtype))) % 2**31), kind)


@pytest.mark.parametrize("kind", ["fp8", "e8m0", "e4m3"])
def test_every_fp16_value(kind: str) -> None:
    block = KINDS[kind][2]
    vals = _all_fp16_values()
    anchors = [2.0**-24, 2.0**-14, 1e-4, 0.01171875, 0.1, 1.0, 6.75, 448.0, 2.0**12, 65504.0]
    x = torch.cat([_anchor_blocks(vals, a, block) for a in anchors]).to(torch.float16)
    _check(x, kind)


def _tie_blocks(kind: str) -> torch.Tensor:
    rows = []
    if kind == "fp8":
        mids = (E4M3_GRID[:-1] + E4M3_GRID[1:]) / 2
        for k in range(-20, 8):                       # s = 2^k, anchor = 448 * 2^k
            s = 2.0**k
            vals = np.concatenate([mids, -mids]) * s
            for i in range(0, len(vals), 31):
                chunk = vals[i:i + 31]
                rows.append(np.concatenate([[448.0 * s], chunk, np.zeros(31 - len(chunk))]))
        x = np.asarray(rows).reshape(-1, 32)
    else:
        ties = np.asarray([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5, 0.1, 6.0])
        if kind == "e8m0":
            scales = [2.0**k for k in range(-40, 10)]
            block = 32
        else:   # non-power-of-two E4M3 scales: amax = 6 s gives amax / 6 == s exactly
            scales = [m * 2.0**e for e in range(-8, 8) for m in (1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875)]
            scales += [k * 2.0**-9 for k in range(1, 8)]
            block = 16
        for s in scales:
            vals = np.concatenate([ties, -ties]) * s
            vals = np.concatenate([vals, np.zeros(block - 1 - len(vals) % (block - 1))])
            for i in range(0, len(vals), block - 1):
                rows.append(np.concatenate([[6.0 * s], vals[i:i + block - 1]]))
        x = np.asarray(rows).reshape(-1, block)
    x32 = x.astype(F32)
    width = 512 if kind == "fp8" else (256 if kind == "e8m0" else 128)
    pad = (-x32.size) % width
    return torch.from_numpy(np.concatenate([x32.reshape(-1), np.zeros(pad, F32)]).reshape(-1, width))


@pytest.mark.parametrize("kind", ["fp8", "e8m0", "e4m3"])
def test_rounding_ties(kind: str) -> None:
    x = _tie_blocks(kind)
    _check(x, kind)
    if kind != "e8m0":      # e8m0 ties span 2^-40..2^9 scales: below FP16 range on purpose
        _check(x.to(torch.float16), kind)


@pytest.mark.parametrize("kind", ["fp8", "e8m0", "e4m3"])
def test_adversarial_blocks(kind: str) -> None:
    block = KINDS[kind][2]
    width = 512
    sub = torch.arange(1, 1024, dtype=torch.int16).view(torch.float16)        # FP16 subnormals
    rows = [
        torch.zeros(width),
        -torch.zeros(width),
        torch.full((width,), 65504.0),
        torch.full((width,), -65504.0),
        torch.tensor([65504.0, -65504.0] * (width // 2)),
        sub[:width].float(),
        -sub[:width].float(),
        torch.tensor([2.0**-24] + [0.0] * (width - 1)),
        torch.tensor([-(2.0**-24)] * width),
        torch.tensor([1e-4, -1e-4] * (width // 2)),
        torch.tensor([447.9, 448.0, 449.0, 480.0, 500.0, -470.0, 6.0, 6.1] * (width // 8)),
    ]
    _check(torch.stack(rows).to(torch.float16), kind)
    # FP32 inputs below FP16 range, FP32 subnormals, huge values
    tiny = torch.tensor([1e-38, -1e-40, 1e-45, 3e-39] * (width // 4), dtype=torch.float32)
    huge = torch.tensor([3e38, -1e30, 1.0, 2.0**100] * (width // 4), dtype=torch.float32)
    _check(torch.stack([tiny, huge]), kind)


@pytest.mark.parametrize("kind", ["fp8", "e8m0", "e4m3"])
def test_non_finite_blocks_become_nan(kind: str) -> None:
    fn, fn_torch, block = KINDS[kind]
    x = torch.randn(4, 512, dtype=torch.float32, device="cuda")
    x[0, 3] = float("inf")
    x[1, 100] = float("nan")
    x[2, 200] = -float("inf")
    for out in (fn(x, out_dtype=torch.float32, impl="triton"), fn_torch(x, out_dtype=torch.float32)):
        for r, c in ((0, 3), (1, 100), (2, 200)):
            b = c // block
            assert torch.isnan(out[r, b * block:(b + 1) * block]).all()
        assert torch.isfinite(out[3]).all()
        assert torch.isfinite(out[0, 64:]).all()


def test_fp16_exact_ranges() -> None:
    """FP16 holds every compressed-KV value, and window-KV values of blocks with exponent e >= -15."""
    x = _random_blocks(1024, 512, torch.float32, seed=7)
    ckv = qat.fp4_e4m3_qdq(x.cuda(), out_dtype=torch.float32, impl="triton")
    assert torch.equal(ckv.to(torch.float16).float(), ckv)
    win = qat.fp8_block32_qdq(x.cuda(), out_dtype=torch.float32, impl="triton")
    amax = x.cuda().view(1024, 16, 32).abs().amax(-1)
    # exact bound: block exponent e = ceil_log2(amax / 448) >= -15  <=>  amax > 448 * 2^-16 = 0.0068359375
    keep = (amax > 448.0 * 2.0**-16).repeat_interleave(32, dim=1)
    assert torch.equal(win.to(torch.float16).float()[keep], win[keep])


def test_impl_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(qat.IMPL_ENV, "torch")
    assert qat.default_impl() == "torch"
    monkeypatch.setenv(qat.IMPL_ENV, "fast")
    with pytest.raises(ValueError):
        qat.default_impl()
    with pytest.raises(ValueError):
        qat.fp8_block32_qdq(torch.ones(2, 32), impl="triton")


# ----------------------------------------------------------------------------- vs L-REF twins (§4.2)
LREF_QAT = "/mnt/hdd/v100-research/tools/ds41_ref/ref_qat.py"


def _lref_qat():
    """L-REF's ref_qat.py loaded read-only by path (no sys.path change; it imports only torch)."""
    import importlib.util
    import os
    import sys

    if not os.path.exists(LREF_QAT):
        pytest.skip(f"L-REF twins not present at {LREF_QAT}")
    spec = importlib.util.spec_from_file_location("ds41_lref_ref_qat", LREF_QAT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # strictly read-only on L-REF's tree
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


def _lref_inputs(it: int, rows: int, width: int, gen: torch.Generator) -> torch.Tensor:
    dev = "cuda"
    if it % 3 == 0:     # log-uniform magnitudes with a per-row exponent spread (block amax over many exponents)
        mag = torch.exp2(torch.empty(rows, width, device=dev).uniform_(-12, 4, generator=gen))
        x = mag * torch.exp2(torch.randint(-30, 20, (rows, 1), device=dev, generator=gen).float())
        x[torch.rand(rows, width, device=dev, generator=gen) < 0.05] = 0.0
    elif it % 3 == 1:   # exact E2M1 / E4M3 grid points and midpoints times powers of two
        g = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.5, 6.0, 1.0625, 1.1875], device=dev)
        x = g[torch.randint(0, len(g), (rows, width), device=dev, generator=gen)]
        x = x * torch.exp2(torch.randint(-20, 10, (rows, 1), device=dev, generator=gen).float())
    else:               # every-bit-pattern FP16 inputs (what the model feeds), non-finite zeroed
        x = torch.randint(-32768, 32767, (rows, width), device=dev, dtype=torch.int16, generator=gen)
        x = x.view(torch.float16)
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    sign = torch.where(torch.rand(rows, width, device=x.device, generator=gen) < 0.5, -1.0, 1.0)
    return (x.float() * sign).contiguous()


@pytest.mark.parametrize("kind", ["fp8_block32", "fp4_e8m0", "fp4_e4m3"])
def test_bitwise_vs_lref_twins(kind: str) -> None:
    """PORT_DESIGN §4.2 / §4.5 row 1: port Triton and torch twins == L-REF ref_qat bit for bit (sign of zero
    included), on 12 x 1024 x 512 values per kind."""
    ref = getattr(_lref_qat(), f"{kind}_qdq")
    port = getattr(qat, f"{kind}_qdq")
    gen = torch.Generator(device="cuda").manual_seed(4321)
    n = mism_triton = mism_torch = 0
    for it in range(12):
        x = _lref_inputs(it, 1024, 512, gen)
        want = ref(x).contiguous().view(torch.int32)
        mism_triton += int((port(x, out_dtype=torch.float32, impl="triton").view(torch.int32) != want).sum())
        mism_torch += int((port(x, out_dtype=torch.float32, impl="torch").view(torch.int32) != want).sum())
        n += x.numel()
    assert (mism_triton, mism_torch) == (0, 0), f"{kind}: {mism_triton}/{mism_torch} of {n} values differ"
