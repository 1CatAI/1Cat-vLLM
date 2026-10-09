# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 mutable attention workspaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata


@dataclass
class DecodeCache:
    """Per-layer dense decode cache, invalidated at the original prefill sites."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    length: int = 0
    capacity: int = 0

    def invalidate(self) -> None:
        self.key = None
        self.value = None
        self.length = 0
        self.capacity = 0

    def ensure_capacity(
        self,
        required_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            self.key is not None
            and self.value is not None
            and self.capacity >= required_len
            and self.key.shape[1] == num_kv_heads
            and self.key.shape[2] == head_dim
            and self.key.dtype == dtype
            and self.key.device == device
        ):
            return

        new_capacity = max(required_len, max(16, self.capacity * 2))
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

        if self.key is not None and self.value is not None and self.length > 0:
            new_k[: self.length].copy_(self.key[: self.length])
            new_v[: self.length].copy_(self.value[: self.length])

        self.key = new_k
        self.value = new_v
        self.capacity = new_capacity

    def get_kv_single_seq(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        block_size: int,
        head_dim: int,
        *,
        extract: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(seq_lens_cpu[0])
        q_len = int(attn_metadata.num_actual_tokens)
        num_kv_heads = key.shape[1]

        cache_hit = (
            self.key is not None
            and self.value is not None
            and seq_len > self.length
            and seq_len - q_len == self.length
        )

        if not cache_hit:
            k_cont, v_cont = extract(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=seq_len,
            )
            self.ensure_capacity(
                seq_len,
                num_kv_heads,
                head_dim,
                k_cont.dtype,
                k_cont.device,
            )
            assert self.key is not None
            assert self.value is not None
            self.key[:seq_len].copy_(k_cont)
            self.value[:seq_len].copy_(v_cont)
            self.length = seq_len
            return (
                self.key[:seq_len],
                self.value[:seq_len],
            )

        self.ensure_capacity(
            seq_len,
            num_kv_heads,
            head_dim,
            key.dtype,
            key.device,
        )
        assert self.key is not None
        assert self.value is not None
        self.key[self.length : seq_len].copy_(key[:q_len])
        self.value[self.length : seq_len].copy_(value[:q_len])
        self.length = seq_len
        return (
            self.key[:seq_len],
            self.value[:seq_len],
        )


@dataclass
class V100Workspace:
    """Mutable state owned by one attention layer, separate from its policy."""

    decode_cache: DecodeCache = field(default_factory=DecodeCache)
