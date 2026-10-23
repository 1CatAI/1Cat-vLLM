# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped skinny MoE projection as a Triton GEMM on M8N8K4 tensor cores.

Same contract as the hand-written skinny QPN MoE op for E8M0-scaled FP4
expert weights packed in the M8N8K4 fragment order: ``y[slot] = x[row] @
W[gids[g]].T`` for the slots of every expert group (``perm``, ``gids``,
``goff``), ``row = slot`` (x slot-major) or ``slot // topk``. The hand kernel
decodes every weight tile again for each 8-row block; here one program takes
BLOCK_M rows of one expert group and BLOCK_N packed columns, so a decoded tile
serves BLOCK_M rows. That pays off once a prefill chunk gives the experts more
than 13 rows on average. Programs map to row blocks, not to groups, so a
popular expert spreads over as many programs as it has blocks instead of
leaving one program as the straggler; routing is skewed, so each group takes
the row tile that fits its size (one launch per tile size).

E2M1 codes widen to fp16 by a shift (value * 2^-14, exact); the E8M0 block
scale, the expert's global scale and the 2^14 re-bias multiply the decoded
weights in fp16 as in the hand kernel (exact power-of-two products), and the
dot accumulates in FP32.
"""

import torch

from vllm.triton_utils import tl, triton

# More rows per expert on average than this take the Triton kernel, the hand
# kernel the rest. Measured against the hand kernel on DeepSeek-V4-Flash layer 5
# (W13 + W2, Zipf skew 0 / 0.8 / 1.2): Triton is 0.3-1.5 % slower at 12.9 rows,
# 0.7-5.9 % faster at 13.8 and 16-23 % faster at 23.9.
TRITON_MIN_ROWS_PER_EXPERT = 13
# Groups by size: (more than, at most) rows, and the tile (BLOCK_M, BLOCK_N,
# BLOCK_K, warps) for W13 (token-major x) and W2 (slot-major x). Measured on the
# M8N8K4 cards with a DeepSeek-V4-Flash layer (256 experts, top-6) under uniform and
# Zipf-skewed routing; the bounds sit on the tile heights, so a group decodes
# its weight tile once per row block it fills.
_TIERS = (
    (0, 16, {False: (16, 128, 32, 4), True: (16, 128, 64, 4)}),
    (16, 32, {False: (32, 128, 32, 4), True: (32, 128, 64, 4)}),
    (32, 1 << 30, {False: (64, 128, 32, 4), True: (64, 128, 128, 4)}),
)


@triton.jit
def _e2m1x2_to_f16(raw):
    # Both E2M1 codes of each byte to fp16 (value * 2^-14, exact for every code,
    # subnormal 0.5 included): sign to bit 15, exponent and mantissa to bits
    # 11..9. One shift pair moves two bytes 16 bits apart at once, as the hand
    # kernel's decoder does: the halves come out in byte order 0, 2, 1, 3 (low
    # nibbles first, then high), which the k order of the caller absorbs.
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b32 s, m;
        shl.b32 s, $4, 12;
        shl.b32 m, $4, 9;
        and.b32 s, s, 0x80008000;
        and.b32 m, m, 0x0E000E00;
        or.b32 $0, s, m;
        shl.b32 s, $4, 4;
        shl.b32 m, $4, 1;
        and.b32 s, s, 0x80008000;
        and.b32 m, m, 0x0E000E00;
        or.b32 $1, s, m;
        shl.b32 s, $4, 8;
        shl.b32 m, $4, 5;
        and.b32 s, s, 0x80008000;
        and.b32 m, m, 0x0E000E00;
        or.b32 $2, s, m;
        and.b32 s, $4, 0x80008000;
        shr.b32 m, $4, 3;
        and.b32 m, m, 0x0E000E00;
        or.b32 $3, s, m;
        }""",
        constraints="=r,=r,=r,=r,r",
        args=[raw],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=4,
    )


@triton.jit
def _lanes_to_columns(acc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # lane = column bits 0-1 | bits 3-4 << 2 | bit 2 << 4: as dimensions (bit 2,
    # bits 3-4, bits 0-1) of every 32 columns, swapped to (bits 3-4, bit 2,
    # bits 0-1).
    acc = tl.reshape(acc, (BLOCK_M, BLOCK_N // 32, 2, 4, 4))
    return tl.reshape(tl.permute(acc, (0, 1, 3, 2, 4)), (BLOCK_M, BLOCK_N))


@triton.jit
def _skinny_moe_e8m0_m8n8k4_kernel(
    x_ptr,
    codes_ptr,
    scales_ptr,
    global_scales_ptr,
    perm_ptr,
    gids_ptr,
    goff_ptr,
    y_ptr,
    N,
    K,
    TOP_K,
    x_stride,
    y_stride,
    GROUPS,
    LOW,
    HIGH,
    GROUPS_P2: tl.constexpr,
    X_SLOT_MAJOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 32 == 0, "whole E8M0 scale groups per K step")
    # This program's row block among the groups of its tier (more than LOW, at
    # most HIGH rows), each cut into BLOCK_M-row blocks in group order. Every
    # program derives it from goff itself; a separate list build cost about 80
    # small kernels per layer and chunk (1.6 s of a 150k-token DeepSeek-V4
    # prefill on one pipeline stage).
    g = tl.arange(0, GROUPS_P2)
    g_ok = g < GROUPS
    sizes = tl.load(goff_ptr + g + 1, mask=g_ok, other=0) - tl.load(
        goff_ptr + g, mask=g_ok, other=0
    )
    sizes = tl.where((sizes > LOW) & (sizes <= HIGH), sizes, 0)
    block_end = tl.cumsum((sizes + BLOCK_M - 1) // BLOCK_M, axis=0)
    passed = block_end <= tl.program_id(0)
    group = tl.sum(passed.to(tl.int32), axis=0)
    if group >= GROUPS:
        return
    first_block = tl.max(tl.where(passed, block_end, 0), axis=0)
    row_start = (tl.program_id(0) - first_block) * BLOCK_M
    begin = tl.load(goff_ptr + group)
    count = tl.load(goff_ptr + group + 1) - begin
    expert = tl.load(gids_ptr + group)
    # The expert's global scale with the decoder's 2^14 re-bias, in fp16 as the
    # hand kernel holds it.
    rescale = (tl.load(global_scales_ptr + expert) * 16384.0).to(tl.float16)
    lanes = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    tiles = N // 32
    # Per expert and 32-column tile: K/16 groups of 32 lanes x 8 code bytes and
    # K/32 scale rows of 32 bytes.
    tile = (expert * tiles + lanes // 32).to(tl.int64)
    code_columns = tile * (K // 16) * 256 + (lanes % 32) * 8
    scale_columns = tile * (K // 32) * 32 + lanes % 32
    q = tl.arange(0, BLOCK_K // 2)
    code_k = (q // 8) * 256 + q % 8
    scale_groups = tl.arange(0, BLOCK_K // 32)
    offs_k = tl.arange(0, BLOCK_K)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    index = row_start + tl.arange(0, BLOCK_M)
    row_ok = index < count
    slot = tl.load(perm_ptr + begin + index, mask=row_ok, other=0)
    x_row = slot if X_SLOT_MAJOR else slot // TOP_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        raw = tl.load(
            codes_ptr + (k // 16) * 256 + code_k[:, None] + code_columns[None, :]
        )
        scale_bytes = tl.load(
            scales_ptr + (k // 32 + scale_groups[:, None]) * 32 + scale_columns[None, :]
        )
        # Rows past the group stay undefined: they are never stored.
        x = tl.load(
            x_ptr + x_row[:, None] * x_stride + (k + offs_k)[None, :],
            mask=row_ok[:, None],
        )
        # The dot runs over k in the decoder's order: all low nibbles of the
        # step's bytes, then all high ones (p = nibble * BLOCK_K/2 + q), with
        # bytes 1 and 2 of every four swapped, so the weight needs no
        # repacking and x is read in that order. Position q = 8g + 4a + 2b + c
        # holds byte 8g + 4a + 2c + b, whose nibble carries logical k
        # 16g + 8a + 4b + 2 * nibble + c.
        low, high = _e2m1x2_to_f16(raw)
        w = tl.reshape(
            tl.permute(tl.join(low, high), (2, 0, 1)),
            (2, BLOCK_K // 32, 16, BLOCK_N),
        )
        # E8M0 2^(b - 127) is the fp16 exponent field b - 112 (the rebase
        # keeps it in 1..30), times the global factor: two exact power-of-two
        # products in fp16, as in the hand kernel. Byte q is in group q // 16.
        scale = ((scale_bytes.to(tl.int16) - 112) << 10).to(
            tl.float16, bitcast=True
        ) * rescale
        w = tl.reshape(w * scale[None, :, None, :], (BLOCK_K, BLOCK_N))
        x = tl.reshape(
            tl.permute(
                tl.reshape(x, (BLOCK_M, BLOCK_K // 16, 2, 2, 2, 2)),
                (0, 4, 1, 2, 3, 5),
            ),
            (BLOCK_M, BLOCK_K),
        )
        acc += tl.dot(x, w)
    tl.store(
        y_ptr + slot[:, None].to(tl.int64) * y_stride + columns[None, :],
        _lanes_to_columns(acc, BLOCK_M, BLOCK_N).to(tl.float16),
        mask=row_ok[:, None],
    )


def skinny_moe_e8m0_m8n8k4_triton(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scales: torch.Tensor,
    perm: torch.Tensor,
    gids: torch.Tensor,
    goff: torch.Tensor,
    top_k: int,
    y: torch.Tensor,
    x_slot_major: bool,
) -> None:
    """The hand-written skinny QPN MoE op for E8M0 scales in the M8N8K4 order,
    on tensor cores (needs the ``triton_sm7x_tensor_cores`` Triton)."""
    n, k = y.shape[1], x.shape[1]
    slots = y.shape[0]
    # At most one group per expert.
    groups = min(goff.numel() - 1, global_scales.numel())
    for low, high, tiles in _TIERS:
        block_m, block_n, block_k, warps = tiles[x_slot_major]
        if n % block_n or k % block_k:
            raise ValueError(
                f"Triton skinny MoE tiles N by {block_n} and K by {block_k}, "
                f"got N={n}, K={k}"
            )
        # Row blocks of the tier, bounded without reading the group sizes: one
        # per group (each has more than `low` rows), plus slots / block_m for
        # groups that take several.
        capacity = min(groups, slots // (low + 1))
        if high > block_m:
            capacity += slots // block_m
        _skinny_moe_e8m0_m8n8k4_kernel[(capacity, n // block_n)](
            x,
            codes,
            scales,
            global_scales,
            perm,
            gids,
            goff,
            y,
            n,
            k,
            top_k,
            x.stride(0),
            y.stride(0),
            groups,
            low,
            high,
            GROUPS_P2=triton.next_power_of_2(groups),
            X_SLOT_MAJOR=x_slot_major,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=1,
        )
