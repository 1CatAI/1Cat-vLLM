# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe DDTree branch verifier attention for Flash-V100.

This is a narrow correction kernel for the DDTree small-query verifier path.
It reads vLLM's paged KV cache directly and applies the DDTree ancestor mask
for the verifier rows.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _ddtree_paged_attention_kernel(
    query,
    key,
    value,
    key_cache,
    value_cache,
    output,
    block_table,
    parent_ids,
    query_start_loc,
    seq_lens,
    max_q_len: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    scale: tl.constexpr,
    sliding_left: tl.constexpr,
    sliding_right: tl.constexpr,
    parent_stride0: tl.constexpr,
    parent_stride1: tl.constexpr,
    block_stride0: tl.constexpr,
    block_stride1: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_cur_stride_t: tl.constexpr,
    k_cur_stride_h: tl.constexpr,
    k_cur_stride_d: tl.constexpr,
    v_cur_stride_t: tl.constexpr,
    v_cur_stride_h: tl.constexpr,
    v_cur_stride_d: tl.constexpr,
    o_stride_t: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req_i = tl.program_id(0)
    row = tl.program_id(1)
    q_head = tl.program_id(2)
    kv_head = q_head // num_queries_per_kv

    q_start = tl.load(query_start_loc + req_i).to(tl.int32)
    q_end = tl.load(query_start_loc + req_i + 1).to(tl.int32)
    q_len = q_end - q_start
    seq_len = tl.load(seq_lens + req_i).to(tl.int32)
    prefix_len = seq_len - q_len
    active = (row < q_len) & (seq_len > 0)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_size
    q_ptrs = (
        query
        + (q_start + row) * q_stride_t
        + q_head * q_stride_h
        + offs_d * q_stride_d
    )
    q_vec = tl.load(q_ptrs, mask=d_mask & active, other=0.0).to(tl.float32)

    m_i = tl.full((), -3.4028234663852886e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    q_abs = prefix_len + row

    kv_start = 0
    while kv_start < seq_len:
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        n_mask = (offs_n < seq_len) & active
        local = offs_n - prefix_len

        visible = (offs_n < prefix_len) & n_mask

        cur = row
        for _ in range(max_q_len):
            cur_valid = active & (cur >= 0) & (cur < max_q_len) & (cur < q_len)
            visible = visible | (
                (local == cur)
                & (local >= 0)
                & (local < q_len)
                & n_mask
                & cur_valid
            )
            safe_cur = tl.maximum(tl.minimum(cur, max_q_len - 1), 0)
            parent_ptr = (
                parent_ids
                + req_i * parent_stride0
                + safe_cur * parent_stride1
            )
            parent = tl.load(parent_ptr, mask=cur_valid, other=0).to(tl.int32)
            # In vLLM DDTree metadata, -1 means "root child". The verifier has
            # a synthetic prefix/root row at local slot 0, so root children must
            # include local 0 instead of terminating the walk.
            cur = tl.where(parent < 0, 0, parent)

        if sliding_left >= 0:
            visible = visible & (offs_n >= (q_abs - sliding_left))
        if sliding_right >= 0:
            visible = visible & (offs_n <= (q_abs + sliding_right))

        block_offsets = offs_n - (offs_n // block_size) * block_size
        block_ptrs = (
            block_table
            + req_i * block_stride0
            + (offs_n // block_size) * block_stride1
        )
        block_ids = tl.load(block_ptrs, mask=n_mask, other=0).to(tl.int64)

        prefix_mask = n_mask & (offs_n < prefix_len)
        k_cache_ptrs = (
            key_cache
            + block_ids[:, None] * k_stride_b
            + block_offsets[:, None] * k_stride_t
            + kv_head * k_stride_h
            + offs_d[None, :] * k_stride_d
        )
        v_cache_ptrs = (
            value_cache
            + block_ids[:, None] * v_stride_b
            + block_offsets[:, None] * v_stride_t
            + kv_head * v_stride_h
            + offs_d[None, :] * v_stride_d
        )
        cache_kv_mask = prefix_mask[:, None] & d_mask[None, :]
        k_cache_vals = tl.load(k_cache_ptrs, mask=cache_kv_mask, other=0.0)
        v_cache_vals = tl.load(v_cache_ptrs, mask=cache_kv_mask, other=0.0)

        current_mask = n_mask & (local >= 0) & (local < q_len)
        safe_local = tl.maximum(tl.minimum(local, q_len - 1), 0)
        k_cur_ptrs = (
            key
            + (q_start + safe_local[:, None]) * k_cur_stride_t
            + kv_head * k_cur_stride_h
            + offs_d[None, :] * k_cur_stride_d
        )
        v_cur_ptrs = (
            value
            + (q_start + safe_local[:, None]) * v_cur_stride_t
            + kv_head * v_cur_stride_h
            + offs_d[None, :] * v_cur_stride_d
        )
        current_kv_mask = current_mask[:, None] & d_mask[None, :]
        k_cur_vals = tl.load(k_cur_ptrs, mask=current_kv_mask, other=0.0)
        v_cur_vals = tl.load(v_cur_ptrs, mask=current_kv_mask, other=0.0)

        k = tl.where(current_mask[:, None], k_cur_vals, k_cache_vals).to(tl.float32)
        v = tl.where(current_mask[:, None], v_cur_vals, v_cache_vals).to(tl.float32)

        scores = tl.sum(k * q_vec[None, :], axis=1) * scale
        scores = tl.where(visible, scores, -3.4028234663852886e38)
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        p = tl.where(visible, p, 0.0)
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new
        kv_start += BLOCK_N

    out = tl.where(l_i > 0.0, acc / l_i, 0.0)
    o_ptrs = (
        output
        + (q_start + row) * o_stride_t
        + q_head * o_stride_h
        + offs_d * o_stride_d
    )
    tl.store(o_ptrs, out, mask=d_mask & active)


def _validate_paged_key_value_cache(tensor: torch.Tensor) -> None:
    if tensor.dim() != 4:
        raise ValueError(
            "expected paged KV cache [blocks, block, heads, dim], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.stride(-1) <= 0:
        raise ValueError(f"unsupported KV cache stride {tensor.stride()}")


def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def ddtree_branch_attention_correction(
    *,
    impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    attn_metadata,
    parent_ids: torch.Tensor,
    window_size: tuple[int, int],
) -> bool:
    """Overwrite DDTree verifier rows with exact ancestor-mask attention."""

    if impl.alibi_slopes is not None:
        raise ValueError("DDTree Triton branch attention does not support ALiBi")
    if getattr(impl, "logits_soft_cap", 0):
        raise ValueError("DDTree Triton branch attention does not support softcap")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported query dtype {query.dtype}")
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported key dtype {key.dtype}")
    if value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported value dtype {value.dtype}")
    if key_cache.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported key cache dtype {key_cache.dtype}")
    if value_cache.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported value cache dtype {value_cache.dtype}")
    if parent_ids is None or parent_ids.ndim != 2 or parent_ids.shape[1] <= 0:
        return True
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "DDTree Triton branch attention expects current key/value "
            f"[tokens, heads, dim], got {tuple(key.shape)} and {tuple(value.shape)}"
        )

    _validate_paged_key_value_cache(key_cache)
    _validate_paged_key_value_cache(value_cache)

    if parent_ids.device != query.device or parent_ids.dtype != torch.int32:
        if _is_cuda_graph_capturing(query):
            raise ValueError(
                "DDTree Triton branch attention requires CUDA int32 parent_ids "
                "during graph capture"
            )
        parent_ids = parent_ids.to(device=query.device, dtype=torch.int32)

    query_start_loc = attn_metadata.query_start_loc
    seq_lens = attn_metadata.seq_lens
    block_table = attn_metadata.block_table
    block_size = int(key_cache.shape[1])
    max_q_len = int(parent_ids.shape[1])
    num_reqs = min(int(parent_ids.shape[0]), int(query_start_loc.shape[0]) - 1)
    if num_reqs <= 0 or max_q_len <= 0:
        return True

    max_query_len = int(getattr(attn_metadata, "max_query_len", max_q_len) or 1)
    if max_query_len > max_q_len:
        raise ValueError(
            "DDTree parent metadata does not cover max_query_len: "
            f"{max_q_len} < {max_query_len}"
        )

    num_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[2])
    if int(key.shape[1]) != num_kv_heads or int(value.shape[1]) != num_kv_heads:
        raise ValueError(
            "DDTree Triton branch attention current KV head count mismatch: "
            f"key={tuple(key.shape)}, value={tuple(value.shape)}, "
            f"cache_kv_heads={num_kv_heads}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "DDTree Triton branch attention requires Q heads divisible by KV "
            f"heads, got {num_heads} and {num_kv_heads}"
        )
    head_size = int(query.shape[2])
    if int(key.shape[2]) != head_size or int(value.shape[2]) != head_size:
        raise ValueError(
            "DDTree Triton branch attention current KV head dim mismatch: "
            f"query={tuple(query.shape)}, key={tuple(key.shape)}, "
            f"value={tuple(value.shape)}"
        )
    block_d = triton.next_power_of_2(head_size)
    if block_d > 256:
        raise ValueError(f"unsupported rounded head size {block_d}")

    sliding_left, sliding_right = window_size
    grid = (num_reqs, max_q_len, num_heads)
    _ddtree_paged_attention_kernel[grid](
        query,
        key,
        value,
        key_cache,
        value_cache,
        output,
        block_table,
        parent_ids,
        query_start_loc,
        seq_lens,
        max_q_len=max_q_len,
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_queries_per_kv=num_heads // num_kv_heads,
        scale=float(impl.scale),
        sliding_left=int(sliding_left),
        sliding_right=int(sliding_right),
        parent_stride0=parent_ids.stride(0),
        parent_stride1=parent_ids.stride(1),
        block_stride0=block_table.stride(0),
        block_stride1=block_table.stride(1),
        k_stride_b=key_cache.stride(0),
        k_stride_t=key_cache.stride(1),
        k_stride_h=key_cache.stride(2),
        k_stride_d=key_cache.stride(3),
        v_stride_b=value_cache.stride(0),
        v_stride_t=value_cache.stride(1),
        v_stride_h=value_cache.stride(2),
        v_stride_d=value_cache.stride(3),
        q_stride_t=query.stride(0),
        q_stride_h=query.stride(1),
        q_stride_d=query.stride(2),
        k_cur_stride_t=key.stride(0),
        k_cur_stride_h=key.stride(1),
        k_cur_stride_d=key.stride(2),
        v_cur_stride_t=value.stride(0),
        v_cur_stride_h=value.stride(1),
        v_cur_stride_d=value.stride(2),
        o_stride_t=output.stride(0),
        o_stride_h=output.stride(1),
        o_stride_d=output.stride(2),
        BLOCK_N=64,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return True
