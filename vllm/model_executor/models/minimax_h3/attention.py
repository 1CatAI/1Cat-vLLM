# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense, non-causal H3 attention with explicit backend selection."""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

attention_backend = ContextVar("h3_attention_backend", default="FLASH_ATTN_V100")


@dataclass(frozen=True, slots=True)
class VideoTokenSpan:
    start: int
    latent_grid: tuple[int, int, int]
    role: Literal["reference", "target"]

    @property
    def length(self):
        t, h, w = self.latent_grid
        return t * h * w


@dataclass(frozen=True, slots=True)
class VideoTokenLayout:
    prefix_len: int | None = None
    latent_grid: tuple[int, int, int] | None = None
    used_len: int | None = None
    video_spans: tuple[VideoTokenSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class PackedPaddingMetadata:
    q_length: int
    kv_length: int
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor


@dataclass
class AttentionMetadata:
    attn_mask: torch.Tensor | None = None
    packed_padding: PackedPaddingMetadata | None = None
    video_layout: VideoTokenLayout | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def chunked_attention_reference(q, k, v, *, scale, chunk_size=128):
    """FP32 reference with O(chunk_size * sequence) score storage."""
    qh, kh, vh = (x.transpose(1, 2).float() for x in (q, k, v))
    result = torch.empty_like(q)
    for start in range(0, q.shape[1], chunk_size):
        scores = (qh[:, :, start : start + chunk_size] @ kh.transpose(-2, -1)) * scale
        result[:, start : start + chunk_size] = (
            (scores.softmax(-1) @ vh).transpose(1, 2).to(q.dtype)
        )
    return result


class Attention(nn.Module):
    supports_prefix_kv_slicing = True
    use_ring = False

    def __init__(
        self,
        *,
        num_heads,
        num_kv_heads,
        head_size,
        softmax_scale,
        causal=False,
        qkv_layout="BSND",
        **kwargs,
    ):
        super().__init__()
        if causal or qkv_layout != "BSND" or num_heads != num_kv_heads:
            raise ValueError("H3 DiT requires non-causal BSND MHA")
        self.backend = attention_backend.get()
        self.scale = softmax_scale
        self.head_size = head_size
        self.query_tile = 64

    @property
    def attn_backend(self):
        return self

    def get_name(self):
        return self.backend

    @staticmethod
    def supports_multi_doc_packed_varlen():
        return False

    @staticmethod
    def supports_packed_mask_free():
        return True

    def forward(self, q, k, v, metadata):
        if metadata.attn_mask is not None:
            raise ValueError("H3 accepts validated suffix padding only")
        used = metadata.extra.get("valid_kv_length", q.shape[1])
        if not 0 < used <= q.shape[1] or k.shape != v.shape:
            raise ValueError("invalid packed H3 attention lengths")
        q_valid, k_valid, v_valid = (x[:, :used].contiguous() for x in (q, k, v))
        if self.backend == "FLASH_ATTN_V100":
            from .cuda_ops import flashattn_extension

            if self.query_tile == 64:
                attended = flashattn_extension().forward(
                    q_valid, k_valid, v_valid, self.scale
                )
            else:
                attended = flashattn_extension().forward(
                    q_valid, k_valid, v_valid, self.scale, 0, self.query_tile
                )
        elif self.backend == "FLASHINFER_SM70":
            from .cuda_ops import flashinfer_extension

            attended = flashinfer_extension().forward(
                q_valid, k_valid, v_valid, self.scale
            )
        elif self.backend == "TORCH_SDPA":
            attended = chunked_attention_reference(
                q_valid, k_valid, v_valid, scale=self.scale
            )
        else:
            raise ValueError(f"unknown H3 backend {self.backend}")
        if used == q.shape[1]:
            return attended
        result = torch.zeros_like(q)
        result[:, :used] = attended
        return result
