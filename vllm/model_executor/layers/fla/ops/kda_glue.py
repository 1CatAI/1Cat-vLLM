# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue kernels of the KDA layer between conv, recurrence and o_proj.

In decode and spec-verify steps a KDA layer launched 15 tiny kernels around the
recurrent kernel: four ``.contiguous()`` copies of q, k, v and beta before it
and the eager ``FusedRMSNormGated.forward_native`` chain (11 kernels for an
fp32 input, 12 for fp16 or bf16) after it. Both are replaced here by one launch
each. The arithmetic is the eager one, op for op, so the results are
bit-identical:

* the packing is a pure copy;
* the norm reproduces torch's fp32 operation order and rounding: ``x * x``
  stored before summing (no FMA contraction), PyTorch's CUDA reduction order
  for a 128-wide row (four sequential values per lane, then a
  halving tree over the 32 lanes), the exact ``mean`` scale 1/128, IEEE
  ``expf`` and division for the sigmoid, and one round-to-nearest cast.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

_NORM_DIM = 128
_NORM_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_NORM_LANES = 32
# Rows per program and warps per program of the norm kernel. Four rows share a
# program (the rows are one contiguous load). Up to the rows of a few dozen
# tokens (64 heads each) the kernel is latency-bound and four warps are fastest;
# from 4096 rows on it is bandwidth-bound and two warps match rms_norm_gated
# on the two GPU generations it was measured on.
_NORM_ROWS_PER_PROGRAM = 4
_NORM_LARGE_ROWS = 4096
_PACK_BLOCK = 1024


@triton.jit
def _pair_sum(partial, ROWS: tl.constexpr, GROUPS: tl.constexpr):
    # Adds the upper half of every row to its lower half: [ROWS, 2 * GROUPS] ->
    # [ROWS, GROUPS]. Every output adds exactly two values, so it does not
    # depend on the order the compiler chooses.
    return tl.sum(tl.reshape(partial, [ROWS, 2, GROUPS]), axis=1)


@triton.jit
def _kda_gated_rmsnorm_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    num_rows,
    EPS: tl.constexpr,
    DIM: tl.constexpr,
    LANES: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_mask = (rows < num_rows)[:, None]
    offsets = tl.arange(0, DIM)
    index = rows[:, None] * DIM + offsets[None, :]
    # One contiguous load of the ROWS rows; they are read once.
    x = tl.load(x_ptr + index, mask=row_mask, other=0.0).to(tl.float32)

    # x.float().pow(2).mean(-1): per lane ((v0 + v1) + v2) + v3 over its four
    # consecutive values, then the halving tree over the lanes. The two splits
    # take apart the four values of every lane, (v0, v2) and (v1, v3) first.
    values = tl.reshape(x, [ROWS, LANES, 2, 2])
    even, odd = tl.split(values)
    v0, v2 = tl.split(even)
    v1, v3 = tl.split(odd)
    lane_sum = v0 * v0
    lane_sum = lane_sum + v1 * v1
    lane_sum = lane_sum + v2 * v2
    lane_sum = lane_sum + v3 * v3
    # Halving tree over the lanes (lane i + lane i + 16, then 8, 4, 2, 1), the
    # order torch's CUDA reduction produces (test_kda_glue.py compares the
    # result with torch.mean's). LANES is 32, so five levels.
    tree = _pair_sum(lane_sum, ROWS, 16)
    tree = _pair_sum(tree, ROWS, 8)
    tree = _pair_sum(tree, ROWS, 4)
    tree = _pair_sum(tree, ROWS, 2)
    total = _pair_sum(tree, ROWS, 1)
    rstd = tl.rsqrt(total * (1.0 / DIM) + EPS)

    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    gate = tl.load(gate_ptr + index, mask=row_mask, other=0.0).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-gate))
    out = ((x * rstd) * weight[None, :]) * sigmoid
    tl.store(out_ptr + index, out.to(out_ptr.dtype.element_ty), mask=row_mask)


def kda_gated_rmsnorm_supported(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor
) -> bool:
    """Whether :func:`kda_gated_rmsnorm` covers these operands.

    The kernel is written for rows of exactly ``_NORM_DIM`` values, an input
    and a gate of one of ``_NORM_DTYPES`` with the same number of elements,
    and a weight of one row, also of one of ``_NORM_DTYPES``.
    """
    return (
        x.dtype in _NORM_DTYPES
        and gate.dtype in _NORM_DTYPES
        and x.shape[-1] == _NORM_DIM
        and x.numel() > 0
        and gate.numel() == x.numel()
        and weight.dtype in _NORM_DTYPES
        and weight.shape == (_NORM_DIM,)
    )


def kda_gated_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``FusedRMSNormGated(activation="sigmoid").forward_native`` as one kernel.

    Equals ``(x.float() * rsqrt(x.float().pow(2).mean(-1) + eps) * weight.float()
    * sigmoid(gate.float())).to(out_dtype)`` evaluated eagerly in that order,
    for an fp32, fp16 or bf16 ``x``.
    """
    assert kda_gated_rmsnorm_supported(x, gate, weight)
    assert x.is_contiguous() and gate.is_contiguous() and weight.is_contiguous()
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    num_rows = x.numel() // _NORM_DIM
    num_warps = 4 if num_rows < _NORM_LARGE_ROWS else 2
    # FMA contraction would fuse x * x into the sum and the final products.
    _kda_gated_rmsnorm_kernel[(triton.cdiv(num_rows, _NORM_ROWS_PER_PROGRAM),)](
        x,
        gate,
        weight,
        out,
        num_rows,
        EPS=eps,
        DIM=_NORM_DIM,
        LANES=_NORM_LANES,
        ROWS=_NORM_ROWS_PER_PROGRAM,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _kda_pack_recurrent_inputs_kernel(
    qkv_ptr,
    beta_ptr,
    packed_ptr,
    beta_out_ptr,
    stride_qkv,
    stride_beta,
    num_tokens,
    PROJECTION: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < 3 * PROJECTION
    value = tl.load(qkv_ptr + token * stride_qkv + offsets, mask=mask)
    part = offsets // PROJECTION
    destination = (
        part * (num_tokens * PROJECTION)
        + token * PROJECTION
        + (offsets - part * PROJECTION)
    )
    tl.store(packed_ptr + destination, value, mask=mask)
    if tl.program_id(1) == 0:
        head = tl.arange(0, BLOCK)
        head_mask = head < HEADS
        beta = tl.load(beta_ptr + token * stride_beta + head, mask=head_mask)
        tl.store(beta_out_ptr + token * HEADS + head, beta, mask=head_mask)


def kda_pack_recurrent_inputs(
    qkv: torch.Tensor, beta: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contiguous ``q, k, v`` of shape (1, T, H, D) and ``beta`` (1, T, H).

    One launch instead of the four ``.contiguous()`` copies that
    ``fused_recurrent_kda`` makes of the strided q|k|v split and of beta.
    ``qkv`` is (T, 3 * H * D) and ``beta`` (T, H), both with unit column stride.
    """
    tokens = qkv.size(0)
    projection = num_heads * head_dim
    assert qkv.shape[1] == 3 * projection and qkv.stride(1) == 1
    assert beta.shape == (tokens, num_heads) and beta.stride(1) == 1
    assert num_heads <= _PACK_BLOCK
    packed = torch.empty((3, tokens, projection), dtype=qkv.dtype, device=qkv.device)
    beta_out = torch.empty((tokens, num_heads), dtype=beta.dtype, device=beta.device)
    _kda_pack_recurrent_inputs_kernel[
        (tokens, triton.cdiv(3 * projection, _PACK_BLOCK))
    ](
        qkv,
        beta,
        packed,
        beta_out,
        qkv.stride(0),
        beta.stride(0),
        tokens,
        PROJECTION=projection,
        HEADS=num_heads,
        BLOCK=_PACK_BLOCK,
    )
    q, k, v = (part.view(1, tokens, num_heads, head_dim) for part in packed)
    return q, k, v, beta_out.view(1, tokens, num_heads)
