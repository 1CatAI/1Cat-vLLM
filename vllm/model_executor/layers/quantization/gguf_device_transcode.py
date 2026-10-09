# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side GGUF expert transcoding for whole expert banks.

These functions reproduce the NumPy codecs in ``gguf_lattice_transcode``,
``gguf_lut_transcode`` and ``gguf_transcode`` bit for bit, but operate on
``[rows, bytes]`` uint8 tensors on the target device so a whole expert bank is
decoded in a few kernels instead of one host pass per expert. Scale products
use the same FP32 operation order and the same round-to-nearest FP16
conversion. Only the formats listed in ``DEVICE_TYPES`` are handled here;
everything else keeps the host codec.
"""

import gguf
import numpy as np
import torch

from vllm.transformers_utils.gguf_tensor_reader import quant_size

DEVICE_LATTICE_TYPES = frozenset((18, 21, 22))
DEVICE_LUT4_TYPES = frozenset((20, 23))
DEVICE_AFFINE_TYPES = frozenset((42,))
DEVICE_TYPES = DEVICE_LATTICE_TYPES | DEVICE_LUT4_TYPES | DEVICE_AFFINE_TYPES

_KSIGNS: dict[torch.device, torch.Tensor] = {}


def _ksigns(device: torch.device) -> torch.Tensor:
    table = _KSIGNS.get(device)
    if table is None:
        table = torch.from_numpy(
            np.frombuffer(gguf.quants.IQ2_XXS.ksigns, dtype=np.uint8).copy()
        ).to(device)
        _KSIGNS[device] = table
    return table


def _f16(blocks: torch.Tensor, start: int) -> torch.Tensor:
    """Little-endian FP16 at byte ``start`` of every block, as FP32 [count,1]."""
    return blocks[:, start : start + 2].contiguous().view(torch.float16).float()


def _u32(blocks: torch.Tensor, start: int, words: int) -> torch.Tensor:
    """Little-endian uint32 words as int64 [count, words]."""
    b = blocks[:, start : start + 4 * words].reshape(-1, words, 4).long()
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def _nibbles(data: torch.Tensor) -> torch.Tensor:
    low = data & 15
    high = data >> 4
    return torch.stack((low, high), dim=-1).reshape(data.shape[0], -1)


def _fp16(value: torch.Tensor, name: str) -> torch.Tensor:
    result = value.half()
    if bool((torch.isfinite(value) & ~torch.isfinite(result)).any()):
        raise ValueError(f"GGUF {name} overflows canonical FP16 coefficients")
    return result


def _blocks(data: torch.Tensor, weight_type: int):
    block, size = quant_size(weight_type)
    if data.dtype != torch.uint8 or data.ndim != 2 or data.shape[1] % size:
        raise ValueError("GGUF projection needs complete packed rows [N,bytes]")
    rows, width = data.shape
    return data.contiguous().reshape(-1, size), rows, width // size * block


def lattice_storage(data: torch.Tensor, weight_type: int):
    """IQ2_S/IQ3_XXS/IQ3_S rows -> (U2 codes [N,K], packed metadata [N,K/g]).

    Equivalent to ``transcode_lattice(...).mma884_storage()`` with the
    metadata already reinterpreted as the signed type the native prepare
    operator receives.
    """
    if weight_type not in DEVICE_LATTICE_TYPES:
        raise ValueError(f"GGUF type {weight_type} has no device lattice codec")
    blocks, n, k = _blocks(data, weight_type)
    count = blocks.shape[0]
    d = _f16(blocks, 0)
    if weight_type == 18:
        group, grid_width = 32, 4
        indices = blocks[:, 2:66].long()
        meta = _u32(blocks, 66, 8)
        scales = d * (0.5 + (meta >> 28).float())
        scales = scales * 0.5
        sign_indices = (
            meta[..., None] >> torch.tensor([0, 7, 14, 21], device=data.device)
        ) & 127
        signs = _ksigns(data.device)[sign_indices.reshape(count, 32)]
    elif weight_type == 22:
        group, grid_width = 16, 8
        qh = blocks[:, 66:74].long()
        high = (
            (qh[..., None] >> torch.tensor([0, 2, 4, 6], device=data.device)) & 3
        ).reshape(count, 32)
        indices = blocks[:, 2:34].long() | (high << 8)
        signs = blocks[:, 34:66]
        scales = d * (0.5 + _nibbles(blocks[:, 74:]).float())
        scales = scales * 0.25
    else:
        group, grid_width = 32, 4
        qh = blocks[:, 66:74].long()
        high = ((qh[..., None] >> torch.arange(8, device=data.device)) & 1).reshape(
            count, 64
        )
        indices = blocks[:, 2:66].long() | (high << 8)
        signs = blocks[:, 74:106]
        scales = d * (1 + 2 * _nibbles(blocks[:, 106:])).float()
    scales = _fp16(scales.reshape(n, k // group), "lattice scale")
    indices = indices.reshape(n, k // grid_width)
    signs = signs.reshape(n, k // 8).long()
    scale_bits = scales.view(torch.int16).long() & 0xFFFF
    if grid_width == 8:
        packets = (indices & 255) | (signs << 8)
        high = (indices >> 8).reshape(n, k // group, -1)
        shifts = 16 + 2 * torch.arange(high.shape[-1], device=data.device)
        metadata = scale_bits | (high << shifts).sum(-1)
        # uint32 metadata reinterpreted as int32.
        metadata = (metadata - ((metadata >> 31) & 1) * (1 << 32)).to(torch.int32)
    else:
        pairs = indices.reshape(n, k // 8, 2)
        packets = (pairs[..., 0] & 255) | ((pairs[..., 1] & 255) << 8)
        sign_groups = signs.reshape(n, k // group, -1)
        sign_shifts = 16 + 8 * torch.arange(sign_groups.shape[-1], device=data.device)
        high = (indices >> 8).reshape(n, k // group, -1)
        high_shifts = 48 + torch.arange(high.shape[-1], device=data.device)
        metadata = (
            scale_bits
            | (sign_groups << sign_shifts).sum(-1)
            | (high << high_shifts).sum(-1)
        )
    shifts = torch.tensor([0, 8, 2, 10, 4, 12, 6, 14], device=data.device)
    codes = ((packets[..., None] >> shifts) & 3).to(torch.uint8).reshape(n, k)
    return codes, metadata.contiguous()


def lut4_codes(data: torch.Tensor, weight_type: int):
    """IQ4_NL/IQ4_XS rows -> (nibble codes [N,K], FP16 scales [N,K/32])."""
    if weight_type not in DEVICE_LUT4_TYPES:
        raise ValueError(f"GGUF type {weight_type} has no device LUT4 codec")
    blocks, n, k = _blocks(data, weight_type)
    count = blocks.shape[0]
    group = 32
    if weight_type == 20:
        scales = _f16(blocks, 0)
        payload = blocks[:, 2:]
    else:
        d = _f16(blocks, 0)
        hi = blocks[:, 2:3].long() | (blocks[:, 3:4].long() << 8)
        hi = (hi >> (2 * torch.arange(8, device=data.device))) & 3
        lo = _nibbles(blocks[:, 4:8]).long()
        scale_codes = lo | (hi << 4)
        scales = d * (scale_codes - 32).float()
        payload = blocks[:, 8:]
    lanes = payload.reshape(count, -1, 1, group // 2)
    codes = torch.cat((lanes & 15, lanes >> 4), dim=2).reshape(n, k)
    scales = scales.reshape(n, k // group)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("GGUF LUT4 scale overflows finite canonical coefficients")
    converted = _fp16(scales, "LUT4 scale")
    if bool(((scales != 0) & (converted == 0)).any()):
        raise ValueError("GGUF LUT4 scale underflows canonical FP16 coefficients")
    return codes.contiguous(), converted.contiguous()


def affine_codes(data: torch.Tensor, weight_type: int):
    """Q2_0 rows -> (U2 codes [N,K], FP16 scales, FP16 mins), group 32."""
    if weight_type not in DEVICE_AFFINE_TYPES:
        raise ValueError(f"GGUF type {weight_type} has no device affine codec")
    blocks, n, k = _blocks(data, weight_type)
    block, _ = quant_size(weight_type)
    group = 32
    d = _f16(blocks, 0)
    shifts = torch.arange(0, 8, 2, device=data.device)
    codes = (blocks[:, 2:, None] >> shifts.to(torch.uint8)) & 3
    scales = d.repeat_interleave(block // group, dim=1)
    mins = -scales
    return (
        codes.reshape(n, k).contiguous(),
        _fp16(scales.reshape(n, k // group), "scale").contiguous(),
        _fp16(mins.reshape(n, k // group), "min").contiguous(),
    )
