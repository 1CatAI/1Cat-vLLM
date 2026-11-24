# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversion of Inkling2 GGUF (llama.cpp / Unsloth) expert weights into the
official checkpoint layout consumed by `Inkling2Sm70Experts`.

GGML NVFP4 and the official NVFP4 experts hold the same numbers: E2M1 values in
16-element blocks with an E4M3 block scale, times a per-expert global scale.
Only the byte arrangement differs, so the conversion is exact:

* GGML row: super-blocks of 64 elements, 36 bytes each: 4 UE4M3 sub-block
  scales, then 4 x 8 bytes; in sub-block ``s`` byte ``j`` holds element ``j``
  (low nibble) and element ``j + 8`` (high nibble). The per-expert global scale
  is the companion ``*.scale`` tensor (F32 ``[E]``).
* Official: ``w*_weight`` uint8 ``[E, N, K // 2]`` with elements ``2i`` (low
  nibble) and ``2i + 1`` (high nibble) per byte, ``w*_weight_scale`` E4M3
  ``[E, N, K // 16]``, and a global scale ``amax / outer_scale_denominator``.
"""

import torch

NVFP4_BLOCK = 16
GGML_NVFP4_SUPER = 64
GGML_NVFP4_SUPER_BYTES = 36

E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip


def ggml_nvfp4_to_official(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GGML NVFP4 rows ``uint8 [..., rows, nsb * 36]`` ->
    (packed ``uint8 [..., rows, nsb * 32]``, scales ``float8_e4m3fn
    [..., rows, nsb * 4]``) in the official layout."""
    if raw.dtype != torch.uint8:
        raise TypeError(f"expected uint8 GGML NVFP4 rows, got {raw.dtype}")
    if raw.shape[-1] % GGML_NVFP4_SUPER_BYTES:
        raise ValueError(f"row of {raw.shape[-1]} bytes is not whole NVFP4 blocks")
    lead = raw.shape[:-1]
    nsb = raw.shape[-1] // GGML_NVFP4_SUPER_BYTES
    blk = raw.reshape(*lead, nsb, GGML_NVFP4_SUPER_BYTES)
    # UE4M3 scales are non-negative E4M3 values with identical bits.
    scales = blk[..., :4].reshape(*lead, nsb * 4).contiguous()
    qs = blk[..., 4:].reshape(*lead, nsb, 4, 8)
    elems = torch.cat((qs & 0x0F, qs >> 4), dim=-1)  # [..., nsb, 4, 16] in order
    elems = elems.reshape(*lead, nsb * GGML_NVFP4_SUPER)
    packed = elems[..., 0::2] | (elems[..., 1::2] << 4)
    return packed.contiguous(), scales.view(torch.float8_e4m3fn)


def dequant_ggml_nvfp4(raw: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Reference dequantisation of GGML NVFP4 expert rows ``[E, N, nsb * 36]``
    with per-expert ``global_scale [E]`` -> float32 ``[E, N, nsb * 64]``."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=raw.device)
    lead = raw.shape[:-1]
    nsb = raw.shape[-1] // GGML_NVFP4_SUPER_BYTES
    blk = raw.reshape(*lead, nsb, GGML_NVFP4_SUPER_BYTES)
    scales = blk[..., :4].contiguous().view(torch.float8_e4m3fn).float()
    qs = blk[..., 4:].reshape(*lead, nsb, 4, 8)
    nib = torch.cat((qs & 0x0F, qs >> 4), dim=-1).long()
    vals = lut[nib] * scales[..., None]
    out = vals.reshape(*lead, nsb * GGML_NVFP4_SUPER)
    return out * global_scale.float().reshape(-1, *([1] * (out.dim() - 1)))


def dequant_official_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Reference dequantisation of the official layout -> float32
    ``[E, N, K]``, following `Inkling2Sm70Experts.process_weights_after_loading`
    (``scale * amax / outer_scale_denominator`` per 16-element block)."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    elems = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2).long()
    block = scales.float().repeat_interleave(NVFP4_BLOCK, dim=-1)
    out = lut[elems] * block
    return out * global_scale.float().reshape(-1, *([1] * (out.dim() - 1)))
