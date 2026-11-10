# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN operand preparation shared by execution and warmup.

Separate materialized gating retains its FP32 g and caller-selected beta
precision. The fused post-convolution path retains its normalization and
exponential-gate order; these are distinct numerical contracts.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fla.ops import fused_post_conv_prep
from vllm.model_executor.layers.fla.ops.gdn_stages import GdnHeadContract
from vllm.triton_utils import tl, triton


@triton.jit
def _sm70_pack_qwen_gdn_qkv_kernel(
    mixed_qkv,
    packed,
    input_row_stride: tl.int64,
    num_rows: tl.constexpr,
    q_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Bitwise row-slice copy into [all Q][all K][all V] storage."""
    row = tl.program_id(0)
    input_row = mixed_qkv + row * input_row_stride

    q_cols = tl.arange(0, BLOCK_Q)
    q_mask = q_cols < q_dim
    q = tl.load(input_row + q_cols, mask=q_mask)
    tl.store(packed + row * q_dim + q_cols, q, mask=q_mask)

    k_cols = tl.arange(0, BLOCK_K)
    k_mask = k_cols < k_dim
    k = tl.load(input_row + q_dim + k_cols, mask=k_mask)
    q_numel = num_rows * q_dim
    tl.store(packed + q_numel + row * k_dim + k_cols, k, mask=k_mask)

    v_cols = tl.arange(0, BLOCK_V)
    v_mask = v_cols < v_dim
    v = tl.load(input_row + q_dim + k_dim + v_cols, mask=v_mask)
    qk_numel = q_numel + num_rows * k_dim
    tl.store(packed + qk_numel + row * v_dim + v_cols, v, mask=v_mask)


def _sm70_pack_qwen_gdn_qkv(
    mixed_qkv: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
) -> torch.Tensor:
    """Materialize the recurrent Q/K/V layout with one copy launch."""
    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must be rank two")
    if mixed_qkv.stride(1) != 1:
        raise ValueError("mixed_qkv must be contiguous by row")
    if min(q_dim, k_dim, v_dim) <= 0:
        raise ValueError("Q/K/V widths must be positive")
    if mixed_qkv.shape[1] < q_dim + k_dim + v_dim:
        raise ValueError("mixed_qkv is narrower than the Q/K/V widths")

    num_rows = mixed_qkv.shape[0]
    packed = torch.empty(
        num_rows * (q_dim + k_dim + v_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    _sm70_pack_qwen_gdn_qkv_kernel[(num_rows,)](
        mixed_qkv,
        packed,
        mixed_qkv.stride(0),
        num_rows=num_rows,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        BLOCK_Q=triton.next_power_of_2(q_dim),
        BLOCK_K=triton.next_power_of_2(k_dim),
        BLOCK_V=triton.next_power_of_2(v_dim),
        num_warps=8,
        num_stages=1,
    )
    return packed


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    *,
    beta_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid(), materialized in beta_dtype when provided.

    Speculative recurrent updates request FP32 beta to match the fused packed
    decode transition. Other callers retain the historical b.dtype output.
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    if beta_dtype is None:
        beta_dtype = b.dtype
    beta_output = torch.empty(1, batch, num_heads, dtype=beta_dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


def unpack_mixed_qkv(contract: GdnHeadContract, mixed_qkv, *, fused_pack=False):
    """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

    The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
    followed by ``.contiguous()`` on each tensor.  This version flattens
    all three splits into a single buffer via ``torch.cat`` so that
    torch.compile emits one Triton copy kernel instead of three separate
    contiguous() calls.
    """
    if mixed_qkv is None:
        return None, None, None

    seq_len = mixed_qkv.shape[0]
    q_dim = contract.num_k_heads * contract.head_k_dim // contract.tp_size
    k_dim = contract.num_k_heads * contract.head_k_dim // contract.tp_size
    v_dim = contract.num_v_heads * contract.head_v_dim // contract.tp_size

    if (
        fused_pack
        and seq_len == 8
        and mixed_qkv.is_cuda
        and mixed_qkv.dtype == torch.float16
        and mixed_qkv.stride(1) == 1
    ):
        fused = _sm70_pack_qwen_gdn_qkv(mixed_qkv, q_dim, k_dim, v_dim)
    else:
        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

    q_size = seq_len * q_dim
    k_size = seq_len * k_dim

    q_contig = fused[0:q_size]
    k_contig = fused[q_size : q_size + k_size]
    v_contig = fused[q_size + k_size :]

    query = q_contig.view(1, seq_len, -1, contract.head_k_dim)
    key = k_contig.view(1, seq_len, -1, contract.head_k_dim)
    value = v_contig.view(1, seq_len, -1, contract.head_v_dim)

    return query, key, value


def prepare_prefill(
    contract: GdnHeadContract,
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    *,
    legacy=False,
    fused_pack=False,
    gate_is_exp=False,
):
    """Return canonical [1,T,H,D] Q/K/V and [1,T,H] gates.

    Legacy preparation normalizes in the recurrence. The fused preparation
    normalizes here, retaining the selected log/exponential gate encoding.
    """
    if legacy:
        q, k, v = unpack_mixed_qkv(contract, mixed_qkv, fused_pack=fused_pack)
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        return q, k, v, g, beta
    prepared = fused_post_conv_prep(
        conv_output=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=contract.local_k_heads,
        head_k_dim=contract.head_k_dim,
        head_v_dim=contract.head_v_dim,
        apply_l2norm=True,
        output_g_exp=gate_is_exp,
    )
    return tuple(tensor.unsqueeze(0) for tensor in prepared)


@dataclass(frozen=True)
class GdnPreparation:
    """Static operand policy, shared by recurrent unpacking and prefill."""

    heads: GdnHeadContract
    legacy: bool = False
    fused_pack: bool = False
    gate_is_exp: bool = False

    def unpack(self, mixed_qkv):
        return unpack_mixed_qkv(self.heads, mixed_qkv, fused_pack=self.fused_pack)

    def prefill(self, mixed_qkv, a, b, A_log, dt_bias):
        return prepare_prefill(
            self.heads,
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            legacy=self.legacy,
            fused_pack=self.fused_pack,
            gate_is_exp=self.gate_is_exp,
        )
