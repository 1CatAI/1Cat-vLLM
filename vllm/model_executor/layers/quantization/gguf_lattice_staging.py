# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expand original IQ blocks into reusable TurboMind operand storage.

Layouts and scale formulas match gguf_lattice_transcode.py and TurboMind's
SM70 Converter. Integer packets, signs and FP16 coefficients are preserved;
this does not dequantize or requantize the weights.
GGUF layouts follow gguf-py/llama.cpp (MIT), as in the existing transcode layer.
"""

from weakref import WeakValueDictionary

import torch

from vllm.triton_utils import tl, triton

_POOLS: WeakValueDictionary = WeakValueDictionary()


class GGUFLatticeStaging(torch.nn.Module):
    """Two reusable projection slots, independent of layer count or IQ type."""

    def __init__(self, experts: int, n: int, k: int, device):
        super().__init__()
        if experts <= 0 or n <= 0 or n % 32 or k <= 0 or k % 256:
            raise ValueError("IQ staging slots require complete SM70 tiles")
        self.experts, self.n, self.k = experts, n, k
        self.handle = id(self)
        self.register_buffer(
            "codes",
            torch.empty((2, experts, k, n // 16), dtype=torch.int32, device=device),
            persistent=False,
        )
        _POOLS[self.handle] = self
        self.register_buffer(
            "metadata",
            torch.empty((2, experts, k // 32, n * 8), dtype=torch.uint8, device=device),
            persistent=False,
        )

    def slot(self, shard: str, source_type: int):
        if shard not in ("w1", "w3") or source_type not in (18, 21, 22):
            raise ValueError("IQ staging slots are gate/up for IQ3_XXS, IQ3_S, IQ2_S")
        index = 0 if shard == "w1" else 1
        group = 16 if source_type == 22 else 32
        dtype = torch.int32 if source_type == 22 else torch.int64
        return self.codes[index], self.metadata[index].view(dtype).reshape(
            self.experts, self.k // group, self.n
        )

    def stage(self, raw_gate, raw_up, source_type):
        for shard, raw in (("w1", raw_gate), ("w3", raw_up)):
            codes, stats = self.slot(shard, source_type)
            stage_lattice(raw, codes, stats, source_type)


def stage_expert_pool(handle: int, raw_gate, raw_up, source_type) -> None:
    """Private scratch is written and consumed inside one opaque FFN call.

    The owning model keeps the pool alive. Forward calls must be serialized;
    this policy is not admitted with dual batch overlap. Exposing these scratch
    views as mutable compiler inputs would clone large banks and invalidate
    the C++ pointer tables, so they never cross the opaque boundary.
    """
    pool = _POOLS[handle]
    pool.stage(raw_gate, raw_up, source_type)


@triton.jit
def _stage_packets(
    raw,
    packets,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    TYPE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    B: tl.constexpr,
):
    p = tl.program_id(0) * B + tl.arange(0, B)
    valid = p < E * N * K // 8
    expert = p // (N * K // 8)
    row = (p // (32 * (K // 8)) % (N // 32)) * 32 + p % 32
    octet = (p // 32) % (K // 8)
    block = raw + (expert * N + row) * STRIDE + octet // 32 * BLOCK_BYTES
    local = octet % 32
    if TYPE == 22:
        low = tl.load(block + 2 + local, valid, other=0).to(tl.uint16)
        high = tl.load(block + 34 + local, valid, other=0).to(tl.uint16)
    else:
        low = tl.load(block + 2 + 2 * local, valid, other=0).to(tl.uint16)
        high = tl.load(block + 3 + 2 * local, valid, other=0).to(tl.uint16)
    tl.store(packets + p, low | (high << 8), valid)


@triton.jit
def _stage_metadata(
    raw,
    stats,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    TYPE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    GROUP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < E * N * K // GROUP
    expert = i // (N * K // GROUP)
    row = i % N
    group = i // N % (K // GROUP)
    local = group % (256 // GROUP)
    block = raw + (expert * N + row) * STRIDE + group // (256 // GROUP) * BLOCK_BYTES
    bits = tl.load(block, valid, other=0).to(tl.uint16)
    bits |= tl.load(block + 1, valid, other=0).to(tl.uint16) << 8
    d = bits.to(tl.float16, bitcast=True).to(tl.float32)
    if TYPE == 18:
        meta = tl.full((B,), 0, tl.uint32)
        for j in tl.static_range(4):
            meta |= tl.load(block + 66 + 4 * local + j, valid, other=0).to(
                tl.uint32
            ) << (8 * j)
        scale = (d * (0.5 + (meta >> 28).to(tl.float32))) * 0.5
        packed = scale.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint64)
        for j in tl.static_range(4):
            sign = (meta >> (7 * j)) & 127
            parity = sign ^ (sign >> 4)
            parity ^= parity >> 2
            parity ^= parity >> 1
            sign |= (parity & 1) << 7
            packed |= sign.to(tl.uint64) << (16 + 8 * j)
    elif TYPE == 21:
        nibble = (
            tl.load(block + 106 + local // 2, valid, other=0).to(tl.uint32)
            >> (4 * (local % 2))
        ) & 15
        scale = d * (1 + 2 * nibble).to(tl.float32)
        packed = scale.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint64)
        for j in tl.static_range(4):
            sign = tl.load(block + 74 + 4 * local + j, valid, other=0)
            packed |= sign.to(tl.uint64) << (16 + 8 * j)
        high = tl.load(block + 66 + local, valid, other=0)
        packed |= high.to(tl.uint64) << 48
    else:
        nibble = (
            tl.load(block + 74 + local // 2, valid, other=0).to(tl.uint32)
            >> (4 * (local % 2))
        ) & 15
        scale = (d * (0.5 + nibble.to(tl.float32))) * 0.25
        packed = scale.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        high = (
            tl.load(block + 66 + local // 2, valid, other=0).to(tl.uint32)
            >> (4 * (local % 2))
        ) & 15
        packed |= high << 16
    tl.store(stats + i, packed, valid)


def stage_lattice(raw, codes, stats, source_type: int) -> None:
    """Write a prepared [E,K,N/16] U2 bank and transposed group metadata."""
    if source_type not in (18, 21, 22):
        raise ValueError("IQ staging supports IQ3_XXS, IQ3_S and IQ2_S")
    e, n, stride = raw.shape
    k = codes.shape[1]
    group = 16 if source_type == 22 else 32
    expected_stats = torch.int32 if source_type == 22 else torch.int64
    block_bytes = {18: 98, 21: 110, 22: 82}[source_type]
    if not (
        raw.dtype == torch.uint8
        and codes.dtype == torch.int32
        and stats.dtype == expected_stats
        and raw.device == codes.device == stats.device
        and raw.is_contiguous()
        and codes.is_contiguous()
        and stats.is_contiguous()
        and n % 32 == 0
        and k % 256 == 0
        and stride >= k // 256 * block_bytes
        and codes.shape == (e, k, n // 16)
        and stats.shape == (e, k // group, n)
    ):
        raise ValueError("IQ staging requires aligned original and canonical banks")
    _stage_packets[(triton.cdiv(e * n * k // 8, 256),)](
        raw, codes.view(torch.int16), e, n, k, stride, source_type, block_bytes, 256
    )
    _stage_metadata[(triton.cdiv(e * n * k // group, 256),)](
        raw,
        stats,
        e,
        n,
        k,
        stride,
        source_type,
        block_bytes,
        group,
        256,
        enable_fp_fusion=False,
    )
