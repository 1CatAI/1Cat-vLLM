# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mutable per-layer Flash-V100 workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

if TYPE_CHECKING:
    from vllm.v1.attention.backends.flash_v100 import impl as _impl


def _reset_decode_cache(self: _impl.FlashAttnV100Impl) -> None:
    self._decode_cache_k = None
    self._decode_cache_v = None
    self._decode_cache_len = 0
    self._decode_cache_capacity = 0


def _ensure_decode_cache_capacity(
    self: _impl.FlashAttnV100Impl,
    required_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        self._decode_cache_k is not None
        and self._decode_cache_v is not None
        and self._decode_cache_capacity >= required_len
        and self._decode_cache_k.shape[1] == num_kv_heads
        and self._decode_cache_k.shape[2] == head_dim
        and self._decode_cache_k.dtype == dtype
        and self._decode_cache_k.device == device
    ):
        return

    new_capacity = max(required_len, max(16, self._decode_cache_capacity * 2))
    new_k = torch.empty(
        (new_capacity, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    new_v = torch.empty(
        (new_capacity, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )

    if (
        self._decode_cache_k is not None
        and self._decode_cache_v is not None
        and self._decode_cache_len > 0
    ):
        new_k[: self._decode_cache_len].copy_(
            self._decode_cache_k[: self._decode_cache_len]
        )
        new_v[: self._decode_cache_len].copy_(
            self._decode_cache_v[: self._decode_cache_len]
        )

    self._decode_cache_k = new_k
    self._decode_cache_v = new_v
    self._decode_cache_capacity = new_capacity


def _get_decode_kv_single_seq(
    self: _impl.FlashAttnV100Impl,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    seq_lens_cpu: torch.Tensor,
    block_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(seq_lens_cpu[0])
    q_len = int(attn_metadata.num_actual_tokens)
    num_kv_heads = key.shape[1]

    cache_hit = (
        self._decode_cache_k is not None
        and self._decode_cache_v is not None
        and seq_len > self._decode_cache_len
        and seq_len - q_len == self._decode_cache_len
    )

    if not cache_hit:
        k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            total_tokens=seq_len,
        )
        self._ensure_decode_cache_capacity(
            seq_len,
            num_kv_heads,
            head_dim,
            k_cont.dtype,
            k_cont.device,
        )
        assert self._decode_cache_k is not None
        assert self._decode_cache_v is not None
        self._decode_cache_k[:seq_len].copy_(k_cont)
        self._decode_cache_v[:seq_len].copy_(v_cont)
        self._decode_cache_len = seq_len
        return (
            self._decode_cache_k[:seq_len],
            self._decode_cache_v[:seq_len],
        )

    self._ensure_decode_cache_capacity(
        seq_len,
        num_kv_heads,
        head_dim,
        key.dtype,
        key.device,
    )
    assert self._decode_cache_k is not None
    assert self._decode_cache_v is not None
    self._decode_cache_k[self._decode_cache_len : seq_len].copy_(key[:q_len])
    self._decode_cache_v[self._decode_cache_len : seq_len].copy_(value[:q_len])
    self._decode_cache_len = seq_len
    return (
        self._decode_cache_k[:seq_len],
        self._decode_cache_v[:seq_len],
    )
