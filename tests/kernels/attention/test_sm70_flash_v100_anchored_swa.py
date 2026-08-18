# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 Flash-V100 anchored decode-window mask kernel tests.

Validates the masked (second) kernel instantiations of the paged decode and
paged prefill kernels against a plain torch reference:
``causal AND (kv < anchor_len OR q_abs - kv < window)``, element-wise.

Requires a CUDA device and the flash_attn_v100 extension built with the
anchored kernel arguments; skipped otherwise.
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
flash_ops = None
if cuda_available:
    try:
        from flash_attn_v100 import (
            flash_attn_decode_paged,
            flash_attn_prefill_paged,
        )

        flash_ops = (flash_attn_decode_paged, flash_attn_prefill_paged)
    except ImportError:
        flash_ops = None

pytestmark = pytest.mark.skipif(
    not cuda_available or flash_ops is None,
    reason="requires CUDA and the flash_attn_v100 extension",
)


def _anchored_mask(
    q_positions: torch.Tensor,
    kv_len: int,
    anchor_len: int,
    window: int,
    device: str,
) -> torch.Tensor:
    """Boolean keep-mask [num_q, kv_len]; True = attend."""
    kv = torch.arange(kv_len, device=device)[None, :]
    q = q_positions[:, None]
    causal = kv <= q
    in_prefix = kv < anchor_len
    in_window = (q - kv) < window
    return causal & (in_prefix | in_window)


def _ref_attention(
    q: torch.Tensor,  # [num_q, H, D]
    k: torch.Tensor,  # [kv_len, H_kv, D]
    v: torch.Tensor,
    mask: torch.Tensor,  # [num_q, kv_len]
    scale: float,
) -> torch.Tensor:
    group = q.shape[1] // k.shape[1]
    k_exp = k.repeat_interleave(group, dim=1)
    v_exp = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k_exp.float()) * scale
    scores = scores.masked_fill(~mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v_exp.float())


def _build_paged_kv(
    kv_len: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
):
    torch.manual_seed(seed)
    num_blocks = (kv_len + block_size - 1) // block_size + 2
    k_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)
    k_lin = k_cache.reshape(-1, num_kv_heads, head_dim)[:kv_len]
    v_lin = v_cache.reshape(-1, num_kv_heads, head_dim)[:kv_len]
    return k_cache, v_cache, block_table, k_lin, v_lin


@pytest.mark.parametrize(
    "head_dim,num_heads,num_kv_heads",
    [(128, 8, 8), (128, 8, 4), (64, 4, 4)],
)
# kv_len=1100 spans three 512-token partitions with the middle partition
# entirely inside the masked gap (exercises the neutral-stats path).
@pytest.mark.parametrize(
    "anchor,window,kv_len", [(32, 16, 96), (16, 8, 40), (32, 16, 1100)]
)
def test_decode_paged_anchored_matches_reference(
    head_dim, num_heads, num_kv_heads, anchor, window, kv_len
):
    device = "cuda"
    dtype = torch.float16
    block_size = 16
    scale = head_dim**-0.5

    k_cache, v_cache, block_table, k_lin, v_lin = _build_paged_kv(
        kv_len, block_size, num_kv_heads, head_dim, device, dtype, seed=0
    )
    q = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    anchor_lens = torch.tensor([anchor], dtype=torch.int32, device=device)

    out = torch.empty_like(q)
    flash_ops[0](
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        softmax_scale=scale,
        out=out,
        anchor_lens=anchor_lens,
        anchored_window=window,
    )

    q_pos = torch.tensor([kv_len - 1], device=device)
    mask = _anchored_mask(q_pos, kv_len, anchor, window, device)
    assert not mask[0].all(), "test must exercise a non-trivial gap"
    ref = _ref_attention(q[0].unsqueeze(0), k_lin, v_lin, mask, scale)
    torch.testing.assert_close(
        out[0].float(), ref[0], atol=2e-2, rtol=2e-2
    )


def test_decode_paged_anchored_off_path_unchanged():
    """Without anchor arguments the decode op matches its own baseline."""
    device = "cuda"
    dtype = torch.float16
    head_dim, num_heads, num_kv_heads = 128, 8, 8
    kv_len, block_size = 96, 16
    scale = head_dim**-0.5

    k_cache, v_cache, block_table, k_lin, v_lin = _build_paged_kv(
        kv_len, block_size, num_kv_heads, head_dim, device, dtype, seed=1
    )
    q = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)

    out = torch.empty_like(q)
    flash_ops[0](
        q, k_cache, v_cache, block_table, seq_lens, softmax_scale=scale, out=out
    )
    q_pos = torch.tensor([kv_len - 1], device=device)
    causal = _anchored_mask(q_pos, kv_len, kv_len, kv_len, device)
    assert causal[0].all()
    ref = _ref_attention(q[0].unsqueeze(0), k_lin, v_lin, causal, scale)
    torch.testing.assert_close(out[0].float(), ref[0], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_dim,num_heads,num_kv_heads", [(128, 8, 4)])
@pytest.mark.parametrize(
    "anchor,window,kv_len,q_len", [(32, 16, 96, 8), (16, 8, 48, 48)]
)
def test_prefill_paged_anchored_matches_reference(
    head_dim, num_heads, num_kv_heads, anchor, window, kv_len, q_len
):
    """Masked paged prefill vs torch reference.

    Covers both the chunked-continuation shape (q_len < kv_len, queries in
    the decode region) and the full-sequence shape (q_len == kv_len, where
    prompt-region queries must stay purely causal).
    """
    device = "cuda"
    dtype = torch.float16
    block_size = 16
    scale = head_dim**-0.5

    k_cache, v_cache, block_table, k_lin, v_lin = _build_paged_kv(
        kv_len, block_size, num_kv_heads, head_dim, device, dtype, seed=2
    )
    q = torch.randn(1, q_len, num_heads, head_dim, device=device, dtype=dtype)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    anchor_lens = torch.tensor([anchor], dtype=torch.int32, device=device)

    out = flash_ops[1](
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        softmax_scale=scale,
        causal=True,
        anchor_lens=anchor_lens,
        anchored_window=window,
    )

    # Query absolute positions: the last q_len positions of the sequence.
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
    mask = _anchored_mask(q_pos, kv_len, anchor, window, device)
    ref = _ref_attention(q[0], k_lin, v_lin, mask, scale)
    torch.testing.assert_close(
        out[0].float(), ref.float(), atol=2e-2, rtol=2e-2
    )
