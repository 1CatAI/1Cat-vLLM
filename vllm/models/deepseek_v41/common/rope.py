# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 rotary embedding (PORT_DESIGN §1 "RoPE / YaRN" row; owner L-ATTN).

Selection follows the reference ``Attention.__init__`` (ref:m.py:680-698) and upstream
``deepseek_v41/common/rope.py``: it keys on ``compress_ratio > 0`` -- *not* on ``> 1`` as the V4
code does (V4.1 layers 20-39 have ratio 1 and must get YaRN):

* ``compress_ratio > 0`` (layers 2-39): theta = ``compress_rope_theta`` (160000) + YaRN with
  ``factor`` 16 over ``original_max_position_embeddings`` 65536, ``beta_fast`` 32 / ``beta_slow`` 1,
  no mscale (cos/sin are not rescaled);
* ``compress_ratio == 0`` (layers 0, 1, DSpark): theta = ``rope_theta`` (10000), no YaRN.

RoPE rotates the LAST ``qk_rope_head_dim`` (64) dims of a vector as interleaved pairs
``(x[2i], x[2i+1])`` (GPT-J style, ``is_neox_style=False``). The inverse rotation (attention output,
ref:m.py:781) conjugates it. The frequencies are computed with the reference formula
(ref:m.py:369-389) in FP32 on the target device; the cache layout ``[positions, rope_dim]`` =
``[cos(32) | sin(32)]`` FP32 is the one 1Cat's V4 kernels read (``cos_sin_cache``), so the V4
inverse-RoPE kernel can be reused unchanged. The cache length is capped by ``max_positions``
(normally ``max_model_len``) instead of YaRN's ``factor x original`` = 1,048,576 rows (256 MiB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class V41RopeParams:
    rope_dim: int
    theta: float
    yarn: bool
    factor: float
    original_max_position: int
    beta_fast: float
    beta_slow: float


def _rope_dict(config) -> dict:
    for name in ("rope_parameters", "rope_scaling"):
        value = getattr(config, name, None)
        if isinstance(value, dict) and value:
            return value
    raise ValueError("DeepSeek-V4.1 config carries neither rope_parameters nor rope_scaling")


def v41_rope_params(config, compress_ratio: int) -> V41RopeParams:
    """RoPE parameters of a layer with ``compress_ratio`` (0, 1 or 2)."""
    if compress_ratio not in (0, 1, 2):
        raise ValueError(f"DeepSeek-V4.1 compress_ratio must be 0, 1 or 2, got {compress_ratio}")
    rope_dim = int(config.qk_rope_head_dim)
    if compress_ratio == 0:
        return V41RopeParams(rope_dim, float(config.rope_theta), False, 1.0, 0, 0.0, 0.0)
    rp = _rope_dict(config)
    if rp.get("rope_type", rp.get("type")) != "yarn":
        raise ValueError(f"DeepSeek-V4.1 compressed layers need YaRN rope, got {rp!r}")
    missing = [k for k in ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow") if k not in rp]
    if missing:
        raise ValueError(f"DeepSeek-V4.1 rope parameters lack {missing}: {rp!r}")
    return V41RopeParams(
        rope_dim=rope_dim,
        theta=float(config.compress_rope_theta),
        yarn=True,
        factor=float(rp["factor"]),
        original_max_position=int(rp["original_max_position_embeddings"]),
        beta_fast=float(rp["beta_fast"]),
        beta_slow=float(rp["beta_slow"]),
    )


def compute_inv_freq(params: V41RopeParams, device: torch.device | str | None = None) -> torch.Tensor:
    """[rope_dim // 2] FP32 frequencies, exactly ref ``precompute_freqs_cis`` (m.py:376-386)."""
    dim, base = params.rope_dim, params.theta
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if params.yarn:
        def corrected_dim(rotations: float) -> float:
            return dim * math.log(params.original_max_position / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(params.beta_fast)), 0)
        high = min(math.ceil(corrected_dim(params.beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / params.factor * (1 - smooth) + freqs * smooth
    return freqs


def compute_cos_sin_cache(params: V41RopeParams, max_positions: int,
                          device: torch.device | str | None = None) -> torch.Tensor:
    """[max_positions, rope_dim] FP32 = [cos | sin] of position x frequency (ref polar(1, outer))."""
    freqs = compute_inv_freq(params, device)
    angles = torch.outer(torch.arange(max_positions, device=device), freqs)
    cis = torch.polar(torch.ones_like(angles), angles)     # the reference's op (CPU cos() differs by 1 ulp)
    return torch.cat([cis.real, cis.imag], dim=-1).contiguous()


def apply_rope_torch(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor,
                     inverse: bool = False, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Rotate the last ``rope_dim`` dims of ``x`` [T, ..., D] as interleaved pairs at ``positions`` [T].
    FP32 math, returns a new tensor in ``out_dtype`` (default ``x.dtype``); the first D - rope_dim dims
    are copied unchanged (converted to ``out_dtype``)."""
    rd = cos_sin_cache.shape[-1]
    half = rd // 2
    cs = cos_sin_cache.index_select(0, positions.to(torch.long))
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [half]
    cos = cs[:, :half].reshape(shape)
    sin = cs[:, half:].reshape(shape)
    if inverse:
        sin = -sin
    rot = x[..., -rd:].float().unflatten(-1, (half, 2))
    a, b = rot[..., 0], rot[..., 1]
    out_rot = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)
    out = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    out[..., :-rd] = x[..., :-rd]
    out[..., -rd:] = out_rot
    return out


class DeepseekV41RotaryEmbedding(nn.Module):
    """FP32 cos/sin cache + torch rotation; ``cos_sin_cache`` is what the SM70 kernels consume."""

    is_neox_style = False

    def __init__(self, params: V41RopeParams, max_positions: int,
                 device: torch.device | str | None = None) -> None:
        super().__init__()
        if max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {max_positions}")
        self.params = params
        self.rotary_dim = params.rope_dim
        self.max_positions = max_positions
        self.register_buffer("cos_sin_cache", compute_cos_sin_cache(params, max_positions, device),
                             persistent=False)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor, inverse: bool = False,
               out_dtype: torch.dtype | None = None) -> torch.Tensor:
        return apply_rope_torch(x, positions, self.cos_sin_cache, inverse=inverse, out_dtype=out_dtype)

    def extra_repr(self) -> str:
        p = self.params
        return (f"rope_dim={p.rope_dim}, theta={p.theta}, yarn={p.yarn}, factor={p.factor}, "
                f"original_max_position={p.original_max_position}, max_positions={self.max_positions}")


_CACHE: dict[tuple, DeepseekV41RotaryEmbedding] = {}


def build_v41_rope(config, compress_ratio: int, *, max_positions: int | None = None,
                   device: torch.device | str | None = None) -> DeepseekV41RotaryEmbedding:
    """Shared rope module for a layer of ``compress_ratio``; one instance per (params, length, device).

    ``max_positions`` defaults to ``config.max_position_embeddings``; the model passes
    ``max_model_len`` (+ speculative headroom) so the cache does not cost 256 MiB per instance.
    """
    params = v41_rope_params(config, compress_ratio)
    n = int(max_positions if max_positions is not None else config.max_position_embeddings)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    key = (params, n, str(torch.device(device)))
    rope = _CACHE.get(key)
    if rope is None:
        rope = DeepseekV41RotaryEmbedding(params, n, device)
        _CACHE[key] = rope
    return rope
