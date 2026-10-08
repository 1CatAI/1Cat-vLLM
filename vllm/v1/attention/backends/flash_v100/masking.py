# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-format-independent sparse prefill and DFlash visibility masks."""

from __future__ import annotations

import torch

import vllm.envs as envs


def _cdiv_int(a: int, b: int) -> int:
    return (a + b - 1) // b


def _build_bfla_block_mask_for_seq(
    q_seq: torch.Tensor,
    key_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
    mask_block_n: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Build [1, Hkv, q_tiles, kv_tiles] sparse prefill mask."""
    if q_seq.ndim != 4 or q_seq.shape[0] != 1:
        return None
    if key_cache.dtype != torch.float16 or q_seq.dtype != torch.float16:
        return None
    if mask_block_n <= 0:
        return None

    pool_mode = envs.VLLM_FLASH_V100_BFLA_POOL.lower()
    flat_group_tokens = 64
    use_flat64 = pool_mode == "flat64"
    if use_flat64 and mask_block_n % flat_group_tokens != 0:
        return None

    q_len = int(q_seq.shape[1])
    num_query_heads = int(q_seq.shape[2])
    head_dim = int(q_seq.shape[3])
    num_kv_heads = int(key_cache.shape[2])
    if q_len <= 1 or seq_len < q_len:
        return None
    if num_query_heads % num_kv_heads != 0:
        return None

    q_blocks = _cdiv_int(q_len, mask_block_n)
    kv_tiles = _cdiv_int(seq_len, mask_block_n)
    if q_blocks <= 0 or kv_tiles <= 0:
        return None

    def pool_blocks(x: torch.Tensor) -> torch.Tensor:
        if use_flat64:
            groups = mask_block_n // flat_group_tokens
            return (
                x.view(
                    x.shape[0],
                    groups,
                    flat_group_tokens,
                    x.shape[2],
                    x.shape[3],
                )
                .permute(3, 0, 1, 2, 4)
                .reshape(
                    x.shape[2],
                    x.shape[0],
                    groups,
                    flat_group_tokens * x.shape[3],
                )
            )
        if pool_mode == "center":
            return x[:, min(mask_block_n // 2, x.shape[1] - 1)].permute(1, 0, 2)
        if pool_mode == "maxabs":
            idx = torch.argmax(x.abs(), dim=1, keepdim=True)
            return torch.gather(x, 1, idx).squeeze(1).permute(1, 0, 2)
        return x.mean(dim=1).permute(1, 0, 2)

    q_req = q_seq.squeeze(0)
    q_pad = torch.zeros(
        (q_blocks * mask_block_n, num_query_heads, head_dim),
        device=q_seq.device,
        dtype=q_seq.dtype,
    )
    q_pad[:q_len].copy_(q_req)
    q_low = pool_blocks(q_pad.view(q_blocks, mask_block_n, num_query_heads, head_dim))

    num_pages = _cdiv_int(seq_len, block_size)
    pages = block_table_row[:num_pages].to(torch.long)
    k_req = key_cache.index_select(0, pages).reshape(-1, num_kv_heads, head_dim)
    k_req = k_req[:seq_len]
    k_pad = torch.zeros(
        (kv_tiles * mask_block_n, num_kv_heads, head_dim),
        device=q_seq.device,
        dtype=key_cache.dtype,
    )
    k_pad[:seq_len].copy_(k_req)
    k_low = pool_blocks(k_pad.view(kv_tiles, mask_block_n, num_kv_heads, head_dim))

    num_queries_per_kv = num_query_heads // num_kv_heads
    keep_per_kv = torch.zeros(
        (num_kv_heads, q_blocks, kv_tiles),
        device=q_seq.device,
        dtype=torch.bool,
    )
    context_len = seq_len - q_len
    q_block_end = (
        context_len
        + (torch.arange(q_blocks, device=q_seq.device) + 1) * mask_block_n
        - 1
    )
    q_block_end = torch.clamp(q_block_end, max=seq_len - 1)
    k_block_start = torch.arange(kv_tiles, device=q_seq.device) * mask_block_n
    causal = k_block_start[None, :] <= q_block_end[:, None]

    threshold = float(envs.VLLM_FLASH_V100_BFLA_THRESHOLD)
    keep_mass = float(envs.VLLM_FLASH_V100_BFLA_KEEP_MASS)
    keep_ratio = float(envs.VLLM_FLASH_V100_BFLA_KEEP_RATIO)
    min_keep_blocks = int(envs.VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS)
    for kv_h in range(num_kv_heads):
        q_h0 = kv_h * num_queries_per_kv
        q_h1 = q_h0 + num_queries_per_kv
        if use_flat64:
            group_scores = torch.einsum(
                "hqgf,krf->hqkgr", q_low[q_h0:q_h1], k_low[kv_h]
            )
            scores = group_scores.amax(dim=(-1, -2))
        else:
            scores = torch.einsum("hqd,kd->hqk", q_low[q_h0:q_h1], k_low[kv_h])
        scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
        probs = torch.softmax(scores.float() * softmax_scale, dim=-1)
        keep = (probs > threshold).any(dim=0)

        if keep_mass >= 1.0:
            keep |= causal
        elif keep_mass > 0:
            sorted_probs, sorted_idx = torch.sort(
                probs.float(), dim=-1, descending=True
            )
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mass_keep_sorted = cumsum <= keep_mass
            mass_keep_sorted[..., 0] = True
            first_over = torch.argmax(
                (cumsum >= keep_mass).to(torch.int32), dim=-1, keepdim=True
            )
            mass_keep_sorted.scatter_(-1, first_over, True)
            mass_keep = torch.zeros_like(probs, dtype=torch.bool)
            mass_keep.scatter_(-1, sorted_idx, mass_keep_sorted)
            keep |= mass_keep.any(dim=0)

        if keep_ratio > 0 or min_keep_blocks > 0:
            topk = max(min_keep_blocks, int(kv_tiles * keep_ratio))
            topk = max(1, min(topk, kv_tiles))
            _, topk_idx = torch.topk(scores.float(), k=topk, dim=-1)
            topk_keep = torch.zeros_like(scores, dtype=torch.bool)
            topk_keep.scatter_(-1, topk_idx, True)
            keep |= topk_keep.any(dim=0)
        keep_per_kv[kv_h] = keep

    keep_per_kv &= causal[None, :, :]
    q_tile_abs = (
        context_len + torch.arange(q_blocks, device=q_seq.device) * mask_block_n
    ) // mask_block_n
    k_idx = torch.arange(kv_tiles, device=q_seq.device)
    local_blocks = max(0, int(envs.VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS))
    local = (k_idx[None, :] <= q_tile_abs[:, None]) & (
        k_idx[None, :] >= q_tile_abs[:, None] - local_blocks
    )
    keep_per_kv |= local[None, :, :]
    keep_per_kv[:, :, 0] = True

    spec_stride = int(envs.VLLM_FLASH_V100_BFLA_SPEC_STRIDE)
    if spec_stride > 0:
        dropped = causal[None, :, :] & ~keep_per_kv
        q_idx = torch.arange(q_blocks, device=q_seq.device, dtype=torch.int64)[:, None]
        k_idx_i64 = torch.arange(kv_tiles, device=q_seq.device, dtype=torch.int64)[
            None, :
        ]
        stride_keep = (
            (q_idx * 131 + k_idx_i64 * 17 + int(envs.VLLM_FLASH_V100_BFLA_SPEC_SEED))
            % spec_stride
        ) == 0
        keep_per_kv |= dropped & stride_keep[None, :, :]

    spec_prob = float(envs.VLLM_FLASH_V100_BFLA_SPEC_PROB)
    if spec_prob > 0:
        prob = max(0.0, min(spec_prob, 1.0))
        dropped = causal[None, :, :] & ~keep_per_kv
        if prob >= 1.0:
            keep_per_kv |= dropped
        else:
            q_idx = torch.arange(q_blocks, device=q_seq.device, dtype=torch.int64)[
                None, :, None
            ]
            k_idx_i64 = torch.arange(kv_tiles, device=q_seq.device, dtype=torch.int64)[
                None, None, :
            ]
            h_idx = torch.arange(num_kv_heads, device=q_seq.device, dtype=torch.int64)[
                :, None, None
            ]
            hashed = (
                (q_idx + 1) * 1103515245
                + (k_idx_i64 + 1) * 12345
                + (h_idx + 1) * 2654435761
                + int(envs.VLLM_FLASH_V100_BFLA_SPEC_SEED)
            ) & 0x7FFFFFFF
            random_keep = (hashed % 1000000) < int(prob * 1000000)
            keep_per_kv |= dropped & random_keep

    return keep_per_kv.to(torch.int32).unsqueeze(0).contiguous()


def _build_ddtree_visibility_mask(
    *,
    q_len: int,
    seq_len: int,
    prefix_len: int,
    tree_len: int,
    parent_row: torch.Tensor | None,
    device: torch.device,
    window_size: tuple[int, int],
) -> torch.Tensor:
    visible = torch.zeros((q_len, seq_len), dtype=torch.bool, device=device)
    if q_len <= 0 or seq_len <= 0:
        return visible

    for q_offset in range(q_len):
        logical_q_idx = prefix_len + q_offset
        if q_offset == 0 or tree_len <= 0:
            visible[q_offset, : min(logical_q_idx + 1, seq_len)] = True
            continue

        if q_offset > tree_len or parent_row is None:
            visible[q_offset, : min(logical_q_idx + 1, seq_len)] = True
            continue

        visible[q_offset, : min(prefix_len, seq_len)] = True
        if prefix_len < seq_len:
            visible[q_offset, prefix_len] = True
        if logical_q_idx < seq_len:
            visible[q_offset, logical_q_idx] = True

        ancestor = q_offset
        max_slots = int(parent_row.shape[0])
        for _ in range(max_slots):
            if ancestor < 0 or ancestor >= max_slots:
                break
            parent = int(parent_row[ancestor].item())
            parent = 0 if parent < 0 else parent
            parent_pos = prefix_len + parent
            if 0 <= parent_pos < seq_len:
                visible[q_offset, parent_pos] = True
            if parent <= 0:
                break
            ancestor = parent

    left, right = window_size
    if left >= 0 or right >= 0:
        q_pos = torch.arange(q_len, device=device) + prefix_len
        k_pos = torch.arange(seq_len, device=device)
        if left >= 0:
            visible &= k_pos.unsqueeze(0) >= q_pos.unsqueeze(1) - left
        if right >= 0:
            visible &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + right
    return visible
