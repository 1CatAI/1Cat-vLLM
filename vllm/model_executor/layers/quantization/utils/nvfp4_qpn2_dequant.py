# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The dequantization of the QPN2 prepack is derived from dnv2003/v100-skinny
# (MIT), where it serves the same purpose on Turing.
"""Dense fp16 GEMM on the QPN2-packed NVFP4 layout for large M.

The QPN2 prepack (``nvfp4_qpn2_prepare_sm70``) is the only resident weight
layout of the pre-Ampere-but-not-Volta NVFP4 linear path. Decode runs the
QPN2 kernels on it directly; prefill dequantizes one layer at a time into a
transient fp16 ``[N, K]`` buffer and runs ``torch.matmul`` on the fp16 tensor
cores. Marlin's FP4 GEMM reaches only about 27 TFLOPS on Turing, cuBLAS fp16
does better, and keeping a second weight layout for prefill would double the
weight memory.

Layout (see ``nvfp4_qpn2_prepack_codes_kernel``): codes are
``[tiles = N/32][groups = K/16][lane = 32][8 bytes]``; a lane owns row
``n = tile * 32 + col(lane)`` with
``col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0) * 4``. Its 8
bytes hold the 16 nibbles of the group's 16 k in the order
``(0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)``: byte ``b`` holds
``korder[2b]`` in its low nibble and ``korder[2b + 1]`` in its high nibble.
Scales are one fp8-e4m3fn byte per ``(tile, group, lane)``.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# e2m1 magnitudes indexed by the low three code bits; bit 3 is the sign.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_KORDER = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)

# Bounded transient workspace: one fp16 [N, K] buffer per distinct shape and
# device, reused by every layer of that shape.
_dense_workspaces: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def clear_nvfp4_qpn2_dense_workspaces() -> None:
    _dense_workspaces.clear()


def _lane_to_col() -> torch.Tensor:
    lane = torch.arange(32)
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4


def nvfp4_qpn2_dequant_reference(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
) -> torch.Tensor:
    """Pure-torch inverse of the QPN2 prepack (slow; the spec for the kernel)."""
    tiles, groups = n // 32, k // 16
    device = codes.device
    qc = codes.view(tiles, groups, 32, 8)
    qs = scales.view(torch.uint8).view(tiles, groups, 32)
    nib = torch.stack([qc & 0xF, qc >> 4], dim=-1).view(tiles, groups, 32, 16)
    korder = torch.tensor(_KORDER, device=device)
    inverse = torch.empty(16, dtype=torch.long, device=device)
    inverse[korder] = torch.arange(16, device=device)
    nib = nib[..., inverse]
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, device=device, dtype=torch.float32)
    values = magnitudes[(nib & 7).long()] * torch.where(nib & 8 > 0, -1.0, 1.0)
    scale = qs.view(torch.float8_e4m3fn).to(torch.float32)
    values = values * scale.unsqueeze(-1) * global_scale
    col = _lane_to_col().to(device)
    out = torch.empty(n, k, dtype=torch.float32, device=device)
    rows = torch.arange(tiles, device=device).view(tiles, 1) * 32 + col.view(1, 32)
    out[rows.view(-1)] = values.permute(0, 2, 1, 3).reshape(tiles * 32, k)
    return out.to(torch.float16)


@triton.jit
def _e2m1_value(code):
    """e2m1 nibble -> float: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign."""
    mag = code & 7
    m_f = mag.to(tl.float32)
    val = tl.where(
        mag < 4, m_f * 0.5, tl.where(mag < 6, m_f - 2.0, tl.where(mag == 6, 4.0, 6.0))
    )
    return tl.where((code & 8) > 0, -val, val)


@triton.jit
def _e4m3_value(b):
    """fp8-e4m3fn byte -> float in integer arithmetic (no fp8 hardware here).

    Bias 7; exponent 0 is subnormal (mantissa / 8 * 2^-6). The NaN code 0x7f
    does not occur in NVFP4 block scales.
    """
    sign = tl.where((b & 0x80) > 0, -1.0, 1.0)
    exp = ((b >> 3) & 0xF).to(tl.float32)
    mant = (b & 7).to(tl.float32) / 8.0
    normal = (1.0 + mant) * tl.exp2(exp - 7.0)
    subnormal = mant * tl.exp2(-6.0)
    return sign * tl.where(exp == 0, subnormal, normal)


@triton.jit
def _nvfp4_qpn2_dequant_kernel(
    codes32_ptr,
    scales_ptr,
    out_ptr,
    global_scale,
    groups,
    K,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    # One program: one tile (32 rows) x GROUPS_PER_BLOCK groups. A lane's
    # 8-byte payload is read as two 32-bit words (64-bit shifts are slow on
    # Volta and Turing); nibble j sits at bit 4 * (j & 7) of word j >> 3 and
    # holds k offset korder[j]. Scales are read once per (lane, group).
    tile = tl.program_id(0)
    gblock = tl.program_id(1)
    lane = tl.arange(0, 32)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).to(tl.int32) * 4
    row = tile * 32 + col
    g = gblock * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)
    valid = g < groups
    lane_base = (tile * groups + g) * 32
    idx = lane_base[None, :] + lane[:, None]
    w0 = tl.load(codes32_ptr + idx * 2, mask=valid[None, :], other=0)
    w1 = tl.load(codes32_ptr + idx * 2 + 1, mask=valid[None, :], other=0)
    sc = _e4m3_value(tl.load(scales_ptr + idx, mask=valid[None, :], other=0))
    sc = sc * global_scale
    # Produce the 8 k of each word in natural k order so the stores are
    # contiguous 16-byte runs: k offset p (0..7) lives in nibble
    # j = (p & 1) * 4 + (p >> 1), the inverse of korder 0, 2, 4, 6, 1, 3, 5, 7.
    p = tl.arange(0, 8)
    shift = (4 * ((p & 1) * 4 + (p >> 1)))[None, None, :]
    v0 = _e2m1_value((w0[:, :, None] >> shift) & 0xF) * sc[:, :, None]
    v1 = _e2m1_value((w1[:, :, None] >> shift) & 0xF) * sc[:, :, None]
    base_idx = row[:, None, None] * K + g[None, :, None] * 16 + p[None, None, :]
    m3 = valid[None, :, None]
    tl.store(out_ptr + base_idx, v0.to(tl.float16), mask=m3)
    tl.store(out_ptr + base_idx + 8, v1.to(tl.float16), mask=m3)


def nvfp4_qpn2_dequant(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense fp16 ``[n, k]`` weight from the QPN2 prepack; ``out`` may be reused."""
    groups = k // 16
    tiles = n // 32
    if out is None:
        out = torch.empty(n, k, dtype=torch.float16, device=codes.device)
    # 64 groups per program with 8 warps was the fastest of the swept
    # configurations on Volta (34816 x 5120: 1.83 ms vs 2.19 ms for 32 / 4).
    groups_per_block = 64
    grid = (tiles, triton.cdiv(groups, groups_per_block))
    codes32 = codes.view(torch.int32)
    # The scales are fp8-e4m3fn bytes; Triton on these devices reads them as
    # uint8 and converts in integer arithmetic.
    _nvfp4_qpn2_dequant_kernel[grid](
        codes32,
        scales.view(torch.uint8),
        out,
        global_scale,
        groups,
        k,
        GROUPS_PER_BLOCK=groups_per_block,
        num_warps=8,
    )
    return out


def _nvfp4_qpn2_dense_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
) -> torch.Tensor:
    key = (n, k, x.device)
    workspace = _dense_workspaces.get(key)
    if workspace is None:
        workspace = torch.empty(n, k, dtype=torch.float16, device=x.device)
        _dense_workspaces[key] = workspace
    weight = nvfp4_qpn2_dequant(codes, scales, global_scale, n, k, out=workspace)
    return torch.nn.functional.linear(x, weight)


def _nvfp4_qpn2_dense_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="nvfp4_qpn2_dense_linear",
    op_func=_nvfp4_qpn2_dense_linear,
    mutates_args=[],
    fake_impl=_nvfp4_qpn2_dense_linear_fake,
)


def nvfp4_qpn2_dense_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
) -> torch.Tensor:
    """``x @ W.T`` for fp16 ``x`` of ``[M, k]`` and the QPN2 prepack of ``[n, k]``."""
    return torch.ops.vllm.nvfp4_qpn2_dense_linear(x, codes, scales, global_scale, n, k)


# Mirror of ``kQpn2DispatchMaxRows`` in nvfp4_qpn2_sm70.cu.
QPN2_DISPATCH_MAX_ROWS = 32


def _nvfp4_qpn2_dispatch_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
    split_k: int,
    accumulator_chains: int,
) -> torch.Tensor:
    # The split on M happens here at run time, inside one opaque op: a Python
    # branch in the model's forward would be traced once by torch.compile at
    # the warm-up M and keep the dense path in the decode graph.
    if x.shape[0] <= QPN2_DISPATCH_MAX_ROWS:
        from vllm import _sm70_ops as sm70_ops

        out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
        sm70_ops.nvfp4_qpn2_gemm_sm70_out(
            out, x, codes, scales, global_scale, split_k, accumulator_chains
        )
        return out
    return _nvfp4_qpn2_dense_linear(x, codes, scales, global_scale, n, k)


def _nvfp4_qpn2_dispatch_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
    split_k: int,
    accumulator_chains: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="nvfp4_qpn2_dispatch_linear",
    op_func=_nvfp4_qpn2_dispatch_linear,
    mutates_args=[],
    fake_impl=_nvfp4_qpn2_dispatch_linear_fake,
)


def nvfp4_qpn2_dispatch_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
    split_k: int,
    accumulator_chains: int,
) -> torch.Tensor:
    """QPN2 kernels for M <= 32, dequantization plus cuBLAS above.

    The split is decided at run time inside the op.
    """
    return torch.ops.vllm.nvfp4_qpn2_dispatch_linear(
        x, codes, scales, global_scale, n, k, split_k, accumulator_chains
    )
