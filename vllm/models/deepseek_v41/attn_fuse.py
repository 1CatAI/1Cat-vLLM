# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise one-launch folds of V4.1 decode attention's small kernels.

VLLM_V41_FUSE_ATTN_PREP replaces, after wq_b,

    query  = apply_rope(q, positions, table)                 # _rotate, (T, 16)
    window = attention_qdq(apply_rope(kv_norm(kv_low), ...)) # _short_rms, (T,)
                                                             # _rotate, (T, 1)
                                                             # _qdq, (4T,)

with one (T, 17) launch: programs 0..15 rotate a query head; program 16 runs
the window chain for its token. VLLM_V41_FUSE_ATTN_POST replaces

    insert_window_ring(ring, window, positions, valid)       # _ring_insert, (T,)
    output = apply_rope(output, ..., inverse=True)           # _rotate, (T, 16)
    attention_qdq(output.reshape(T, 2, 4096))                # _qdq, (64T,)

with one (T, 16 + ring) launch: each head's program rotates and quantizes its
512 values (four whole 128-groups); the extra program copies the ring row.

Each stage is the original kernel's body with the original arithmetic. The
FP16 intermediates the old launches wrote (normalized kv, rotated rows) are
written and re-read the same way, through a scratch row and a CTA barrier, so
every rounding happens where it did before. Calls the fused kernels do not
cover return None and the caller runs the original kernels.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
    fp32_to_fp8_e4m3fn_bits,
)
from vllm.triton_utils import tl, triton

from .attention_ops import WINDOW_RING
from .quant_rope import _ceil_power_two

logger = init_logger(__name__)

FUSE_ATTN_PREP = envs.VLLM_V41_FUSE_ATTN_PREP == "1"
FUSE_ATTN_POST = envs.VLLM_V41_FUSE_ATTN_POST == "1"
_LOGGED: set = set()

D = 512  # q/kv head and latent width
GROUP = 128  # FP8 QDQ group


def _log_once(name, message):
    if name not in _LOGGED:
        _LOGGED.add(name)
        logger.info(message)


@triton.jit
def _rotate_row(
    x,
    positions,
    table,
    out,
    LENGTH,
    token,
    head,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # quant_rope._rotate's body for one (token, head).
    col = tl.arange(0, BLOCK)
    base = (token * HEADS + head) * D
    nonrotary = col < D - 64
    tl.store(out + base + col, tl.load(x + base + col, nonrotary, 0), nonrotary)
    i = tl.arange(0, 32)
    position = tl.load(positions + token).to(tl.int64)
    if position < LENGTH:
        c, s = (
            tl.load(table + position * 64 + i),
            tl.load(table + position * 64 + 32 + i),
        )
    else:
        angle = position.to(tl.float32) * tl.load(table + LENGTH * 64 + i)
        c, s = tl.cos(angle), tl.sin(angle)
    if INVERSE:
        s = -s
    a = tl.load(x + base + D - 64 + 2 * i).to(tl.float32)
    b = tl.load(x + base + D - 64 + 2 * i + 1).to(tl.float32)
    tl.store(out + base + D - 64 + 2 * i, a * c - b * s)
    tl.store(out + base + D - 64 + 2 * i + 1, a * s + b * c)


@triton.jit
def _qdq_fp8_rows(x, out, base, GROUPS: tl.constexpr, GROUP: tl.constexpr):
    # quant_rope._qdq's FP8 branch for GROUPS consecutive groups; the group
    # max is order-independent, the rest is elementwise.
    offs = base + tl.arange(0, GROUPS)[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    value = tl.load(x + offs).to(tl.float32)
    amax = tl.max(tl.abs(value), 1)
    amax = tl.maximum(amax, 1e-4)
    scale = _ceil_power_two(amax * (1.0 / 448.0))[:, None]
    quant = fp8_e4m3fn_bits_to_fp32_bitcast(
        fp32_to_fp8_e4m3fn_bits(tl.div_rn(value, scale))
    )
    tl.store(out + offs, quant * scale)


@triton.jit
def _attn_prep(
    q,
    kv,
    kv_weight,
    positions,
    table,
    q_out,
    normed,
    roped,
    window,
    LENGTH,
    HEADS: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    if head < HEADS:
        _rotate_row(
            q, positions, table, q_out, LENGTH, token, head, HEADS, H, False, BLOCK
        )
    else:
        # norm._short_rms for this token's kv row.
        at = tl.arange(0, BLOCK)
        value = tl.load(kv + token * H + at, at < H, 0).to(tl.float32)
        variance = tl.sum(value * value, 0) / H
        result = value * tl.rsqrt(variance + EPS) * tl.load(kv_weight + at, at < H, 0)
        tl.store(normed + token * H + at, result, at < H)
        tl.debug_barrier()
        _rotate_row(
            normed, positions, table, roped, LENGTH, token, 0, 1, H, False, BLOCK
        )
        tl.debug_barrier()
        _qdq_fp8_rows(roped, window, token * H, H // GROUP, GROUP)


@triton.jit
def _attn_post(
    x,
    positions,
    table,
    roped,
    out,
    values,
    valid_tokens,
    ring,
    LENGTH,
    HEADS: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP: tl.constexpr,
    RING: tl.constexpr,
    DO_RING: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    if head < HEADS:
        _rotate_row(
            x, positions, table, roped, LENGTH, token, head, HEADS, H, True, BLOCK
        )
        tl.debug_barrier()
        _qdq_fp8_rows(roped, out, (token * HEADS + head) * H, H // GROUP, GROUP)
    elif DO_RING:
        # attention_ops._ring_insert with FIRST = 0.
        position = tl.load(positions + token).to(tl.int64)
        valid = tl.load(valid_tokens + token) & (position >= 0)
        col = tl.arange(0, 512)
        data = tl.load(values + token * 512 + col, valid, 0)
        tl.store(ring + (position % RING) * 512 + col, data, valid)


def _rope_ok(x, positions, table):
    # apply_rope's accepted inputs (3-D form, width 512).
    return (
        x.is_cuda
        and x.dtype == torch.float16
        and x.ndim == 3
        and x.shape[-1] == D
        and x.is_contiguous()
        and positions.shape == (x.shape[0],)
        and positions.dtype in (torch.int32, torch.int64)
        and positions.device == x.device
        and table.device == x.device
        and table.ndim == 2
        and table.shape[1] == 64
        and table.dtype == torch.float32
    )


def attn_prep(q, kv, kv_weight, eps, positions, table):
    """(query, window) as apply_rope(q) and attention_qdq(apply_rope(kv_norm(kv)))
    compute them, in one launch; None if this call is not decode-shaped."""
    tokens = q.shape[0]
    if not (
        1 <= tokens <= 6  # V41Norm's short_rms_norm route
        and _rope_ok(q, positions, table)
        and kv.is_cuda
        and kv.dtype == torch.float16
        and kv.shape == (tokens, D)
        and kv.is_contiguous()
        and kv_weight.shape == (D,)
        and kv_weight.dtype == torch.float32
        and kv_weight.device == kv.device
        and kv_weight.is_contiguous()
        and kv.device == q.device
    ):
        return None
    _log_once(
        "prep",
        "VLLM_V41_FUSE_ATTN_PREP=1: query RoPE + kv_norm/RoPE/FP8 QDQ in one "
        "launch per layer.",
    )
    return _prep_launch(q, kv, kv_weight, eps, positions, table)


def _prep_launch(q, kv, kv_weight, eps, positions, table):
    tokens, heads = q.shape[0], q.shape[1]
    q_out = torch.empty_like(q)
    normed = torch.empty_like(kv)
    roped = torch.empty_like(kv)
    window = torch.empty_like(kv)
    _attn_prep[(tokens, heads + 1)](
        q,
        kv,
        kv_weight,
        positions,
        table,
        q_out,
        normed,
        roped,
        window,
        table.shape[0] - 1,
        HEADS=heads,
        H=D,
        EPS=eps,
        BLOCK=D,
        GROUP=GROUP,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q_out, window


def attn_post(output, positions, table, groups, ring=None, values=None, valid=None):
    """attention_qdq(apply_rope(output, inverse=True).reshape(T, groups, 4096)),
    after insert_window_ring(ring, values, positions, valid) when ring is given,
    in one launch; None if this call is not decode-shaped."""
    tokens = output.shape[0]
    do_ring = ring is not None
    if not (
        1 <= tokens <= 8
        and _rope_ok(output, positions, table)
        and output.shape[1] * D == groups * 4096  # project_output's reshape
    ):
        return None
    if do_ring and not (
        tokens <= WINDOW_RING
        and ring.shape == (WINDOW_RING, 512)
        and ring.dtype == torch.float16
        and ring.is_contiguous()
        and ring.device == output.device
        and values.shape == (tokens, 512)
        and values.dtype == torch.float16
        and values.is_contiguous()
        and values.device == output.device
        and valid.shape == (tokens,)
        and valid.dtype == torch.bool
        and valid.is_contiguous()
        and valid.device == output.device
    ):
        return None
    _log_once(
        "post",
        "VLLM_V41_FUSE_ATTN_POST=1: window ring insert + inverse RoPE + output "
        "FP8 QDQ in one launch per layer.",
    )
    return _post_launch(output, positions, table, groups, ring, values, valid)


def _post_launch(output, positions, table, groups, ring, values, valid):
    tokens, heads = output.shape[0], output.shape[1]
    do_ring = ring is not None
    roped = torch.empty_like(output)
    out = torch.empty(tokens, groups, 4096, dtype=torch.float16, device=output.device)
    _attn_post[(tokens, heads + int(do_ring))](
        output,
        positions,
        table,
        roped,
        out,
        values if do_ring else output,
        valid if do_ring else positions,
        ring if do_ring else output,
        table.shape[0] - 1,
        HEADS=heads,
        H=D,
        BLOCK=D,
        GROUP=GROUP,
        RING=WINDOW_RING,
        DO_RING=do_ring,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
