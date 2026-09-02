# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

import vllm._sm70_ops as sm70_ops
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@triton.jit
def _sm70_mhc_sqrsum_staging_kernel(
    x_ptr,
    sqrsum_ptr,
    x_stride_t,
    sqrsum_stride_s,
    sqrsum_stride_t,
    K_PER_SPLIT: tl.constexpr,
    SQRSUM_MULT: tl.constexpr,
):
    token_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    offsets = split_idx * K_PER_SPLIT + tl.arange(0, K_PER_SPLIT)
    x = tl.load(x_ptr + token_idx * x_stride_t + offsets).to(tl.float32)
    sqrsum = tl.sum(x * x, axis=0) * SQRSUM_MULT
    tl.store(
        sqrsum_ptr + split_idx * sqrsum_stride_s + token_idx * sqrsum_stride_t,
        sqrsum,
    )


@triton.jit
def _sm70_mhc_dot_staging_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    x_stride_t,
    fn_stride_n,
    out_stride_s,
    out_stride_t,
    K_PER_SPLIT: tl.constexpr,
):
    token_idx = tl.program_id(0)
    output_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    offsets = split_idx * K_PER_SPLIT + tl.arange(0, K_PER_SPLIT)
    x = tl.load(x_ptr + token_idx * x_stride_t + offsets).to(tl.float32)
    weight = tl.load(fn_ptr + output_idx * fn_stride_n + offsets).to(tl.float32)
    dot = tl.sum(x * weight, axis=0)
    tl.store(
        out_ptr + split_idx * out_stride_s + token_idx * out_stride_t + output_idx,
        dot,
    )


def sm70_mhc_prenorm_staging(
    x: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    *,
    sqrsum_mult: int = 1,
) -> None:
    """Compute exact-FP32 mHC logits and norm partials on V100."""
    if x.dtype != torch.float16 or fn.dtype != torch.float32:
        raise TypeError("SM70 mHC prenorm requires FP16 input and FP32 weights")
    if x.ndim != 2 or fn.ndim != 2 or not x.is_contiguous():
        raise ValueError("SM70 mHC prenorm requires contiguous 2D tensors")
    if fn.shape[1] != x.shape[1] or fn.shape[0] != gemm_out_mul.shape[2]:
        raise ValueError("SM70 mHC prenorm staging shapes do not match")
    if (
        gemm_out_mul.shape[:2]
        != (
            gemm_out_sqrsum.shape[0],
            x.shape[0],
        )
        or gemm_out_sqrsum.shape[1] != x.shape[0]
    ):
        raise ValueError("SM70 mHC prenorm output shapes do not match")
    n_splits = gemm_out_mul.shape[0]
    if x.shape[1] % n_splits != 0:
        raise ValueError("SM70 mHC prenorm K dimension must divide n_splits")
    k_per_split = x.shape[1] // n_splits
    if k_per_split & (k_per_split - 1):
        raise ValueError("SM70 mHC prenorm split size must be a power of two")

    _sm70_mhc_sqrsum_staging_kernel[(x.shape[0], n_splits)](
        x,
        gemm_out_sqrsum,
        x.stride(0),
        gemm_out_sqrsum.stride(0),
        gemm_out_sqrsum.stride(1),
        K_PER_SPLIT=k_per_split,
        SQRSUM_MULT=sqrsum_mult,
        num_warps=8,
    )
    _sm70_mhc_dot_staging_kernel[(x.shape[0], fn.shape[0], n_splits)](
        x,
        fn,
        gemm_out_mul,
        x.stride(0),
        fn.stride(0),
        gemm_out_mul.stride(0),
        gemm_out_mul.stride(1),
        K_PER_SPLIT=k_per_split,
        num_warps=8,
    )


@triton.jit
def _sm70_mhc_post_kernel(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    out_ptr,
    x_stride_t,
    residual_stride_t,
    residual_stride_m,
    residual_stride_h,
    post_stride_t,
    post_stride_m,
    comb_stride_t,
    comb_stride_i,
    comb_stride_j,
    out_stride_t,
    out_stride_m,
    out_stride_h,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x = tl.load(
        x_ptr + token_idx * x_stride_t + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    for output_stream in tl.static_range(0, HC_MULT):
        value = (
            tl.load(
                post_mix_ptr + token_idx * post_stride_t + output_stream * post_stride_m
            )
            * x
        )
        for input_stream in tl.static_range(0, HC_MULT):
            residual = tl.load(
                residual_ptr
                + token_idx * residual_stride_t
                + input_stream * residual_stride_m
                + hidden_offsets * residual_stride_h,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            mix = tl.load(
                comb_mix_ptr
                + token_idx * comb_stride_t
                + input_stream * comb_stride_i
                + output_stream * comb_stride_j
            )
            value += mix * residual
        tl.store(
            out_ptr
            + token_idx * out_stride_t
            + output_stream * out_stride_m
            + hidden_offsets * out_stride_h,
            value,
            mask=hidden_mask,
        )


def sm70_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the mHC post mapping without TileLang's SM70 BF16 headers."""
    num_tokens, hc_mult, hidden_size = residual.shape
    if residual.dtype != torch.float16 or x.dtype != torch.float16:
        raise TypeError("SM70 mHC post requires FP16 activations")
    if hc_mult != 4 or hidden_size != 4096:
        raise ValueError("SM70 mHC post requires hidden_size=4096 and hc_mult=4")
    post_layer_mix = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix = comb_res_mix.view(num_tokens, hc_mult, hc_mult)
    if out is None:
        out = torch.empty_like(residual)
    elif out.shape != residual.shape or out.dtype != residual.dtype:
        raise ValueError("SM70 mHC post output must match the residual")
    block_h = 512
    _sm70_mhc_post_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        x.view(num_tokens, hidden_size),
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        x.stride(0),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        post_layer_mix.stride(0),
        post_layer_mix.stride(1),
        comb_res_mix.stride(0),
        comb_res_mix.stride(1),
        comb_res_mix.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


@triton.jit
def _sm70_mhc_pre_norm_kernel(
    gemm_out_mul_ptr,
    gemm_out_sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    norm_weight_ptr,
    gemm_mul_stride_s,
    gemm_mul_stride_t,
    gemm_sq_stride_s,
    gemm_sq_stride_t,
    residual_stride_t,
    residual_stride_m,
    residual_stride_h,
    post_stride_t,
    comb_stride_t,
    output_stride_t,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    norm_eps: tl.constexpr,
    N_SPLITS: tl.constexpr,
    SINKHORN_REPEAT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """SM70 mHC pre/Sinkhorn/RMSNorm fed by FP32 staging reductions."""
    token_idx = tl.program_id(0)
    sqrsum = 0.0
    for split_idx in tl.static_range(0, N_SPLITS):
        sqrsum += tl.load(
            gemm_out_sqrsum_ptr
            + split_idx * gemm_sq_stride_s
            + token_idx * gemm_sq_stride_t
        )
    input_rsqrt = tl.rsqrt(sqrsum / (HC_MULT * HIDDEN_SIZE) + rms_eps)

    hc_offsets = tl.arange(0, HC_MULT)
    post_logits = tl.zeros((HC_MULT,), dtype=tl.float32)
    for split_idx in tl.static_range(0, N_SPLITS):
        post_logits += tl.load(
            gemm_out_mul_ptr
            + split_idx * gemm_mul_stride_s
            + token_idx * gemm_mul_stride_t
            + HC_MULT
            + hc_offsets
        )
    post_mix = (
        tl.sigmoid(
            post_logits * input_rsqrt * tl.load(hc_scale_ptr + 1)
            + tl.load(hc_base_ptr + HC_MULT + hc_offsets)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + token_idx * post_stride_t + hc_offsets, post_mix)

    comb_offsets = tl.arange(0, 16)
    comb_logits = tl.zeros((16,), dtype=tl.float32)
    for split_idx in tl.static_range(0, N_SPLITS):
        comb_logits += tl.load(
            gemm_out_mul_ptr
            + split_idx * gemm_mul_stride_s
            + token_idx * gemm_mul_stride_t
            + 2 * HC_MULT
            + comb_offsets
        )
    comb_logits = comb_logits * input_rsqrt * tl.load(hc_scale_ptr + 2) + tl.load(
        hc_base_ptr + 2 * HC_MULT + comb_offsets
    )
    comb = tl.zeros((16,), dtype=tl.float32)
    for row_idx in tl.static_range(0, HC_MULT):
        row_mask = comb_offsets // HC_MULT == row_idx
        row_max = tl.max(tl.where(row_mask, comb_logits, -float("inf")), axis=0)
        row_exp = tl.exp(comb_logits - row_max)
        row_sum = tl.sum(tl.where(row_mask, row_exp, 0.0), axis=0)
        comb = tl.where(row_mask, row_exp / row_sum + hc_sinkhorn_eps, comb)

    for col_idx in tl.static_range(0, HC_MULT):
        col_mask = comb_offsets % HC_MULT == col_idx
        col_sum = tl.sum(tl.where(col_mask, comb, 0.0), axis=0)
        comb = tl.where(col_mask, comb / (col_sum + hc_sinkhorn_eps), comb)

    for _ in tl.static_range(0, SINKHORN_REPEAT - 1):
        for row_idx in tl.static_range(0, HC_MULT):
            row_mask = comb_offsets // HC_MULT == row_idx
            row_sum = tl.sum(tl.where(row_mask, comb, 0.0), axis=0)
            comb = tl.where(row_mask, comb / (row_sum + hc_sinkhorn_eps), comb)
        for col_idx in tl.static_range(0, HC_MULT):
            col_mask = comb_offsets % HC_MULT == col_idx
            col_sum = tl.sum(tl.where(col_mask, comb, 0.0), axis=0)
            comb = tl.where(col_mask, comb / (col_sum + hc_sinkhorn_eps), comb)
    tl.store(comb_mix_ptr + token_idx * comb_stride_t + comb_offsets, comb)

    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    layer_input = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for stream_idx in tl.static_range(0, HC_MULT):
        pre_logit = 0.0
        for split_idx in tl.static_range(0, N_SPLITS):
            pre_logit += tl.load(
                gemm_out_mul_ptr
                + split_idx * gemm_mul_stride_s
                + token_idx * gemm_mul_stride_t
                + stream_idx
            )
        pre_mix = (
            tl.sigmoid(
                pre_logit * input_rsqrt * tl.load(hc_scale_ptr)
                + tl.load(hc_base_ptr + stream_idx)
            )
            + hc_pre_eps
        )
        residual = tl.load(
            residual_ptr
            + token_idx * residual_stride_t
            + stream_idx * residual_stride_m
            + hidden_offsets * residual_stride_h,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        layer_input += pre_mix * residual

    variance = tl.sum(layer_input * layer_input, axis=0) / HIDDEN_SIZE
    norm_scale = tl.rsqrt(variance + norm_eps)
    norm_weight = tl.load(
        norm_weight_ptr + hidden_offsets, mask=hidden_mask, other=0.0
    ).to(tl.float32)
    # Match the upstream fused kernel's FP16 staging before RMSNorm scaling.
    layer_input = layer_input.to(tl.float16).to(tl.float32)
    layer_input = layer_input * norm_scale * norm_weight
    tl.store(
        layer_input_ptr + token_idx * output_stride_t + hidden_offsets,
        layer_input,
        mask=hidden_mask,
    )


def sm70_mhc_pre_norm_from_staging(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> None:
    """Launch the SM70 mHC final stage without BF16 codegen."""
    num_tokens, hc_mult, hidden_size = residual.shape
    if hidden_size != 4096 or hc_mult != 4:
        raise ValueError(
            "SM70 mHC staging kernel requires hidden_size=4096 and hc_mult=4"
        )
    if num_tokens < 1:
        raise ValueError("SM70 mHC staging kernel requires at least one token")
    if norm_weight.dtype != torch.float16:
        raise TypeError("SM70 mHC staging kernel requires FP16 norm weights")

    use_native_verify = num_tokens == 8 and envs.VLLM_SM70_GLM53_MHC_NATIVE_VERIFY
    if num_tokens == 1 or use_native_verify:
        if not hasattr(torch.ops._C, "sm70_glm_mhc_pre_norm_out"):
            raise RuntimeError(
                "SM70 GLM mHC decode requires the native CUDA final-stage op. "
                "Rebuild vLLM from source with CUDA arch 7.0."
            )
        if use_native_verify:
            logger.info_once("SM70 GLM mHC native CUDA q8 final stage enabled.")
        else:
            logger.info_once("SM70 GLM mHC native CUDA decode final stage enabled.")
        sm70_ops.sm70_glm_mhc_pre_norm_out(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
        )
        return

    _sm70_mhc_pre_norm_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        gemm_out_mul.stride(0),
        gemm_out_mul.stride(1),
        gemm_out_sqrsum.stride(0),
        gemm_out_sqrsum.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        post_mix.stride(0),
        comb_mix.stride(0),
        layer_input.stride(0),
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        norm_eps=norm_eps,
        N_SPLITS=gemm_out_mul.shape[0],
        SINKHORN_REPEAT=sinkhorn_repeat,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )


@triton.jit
def _rmsnorm_nw_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    D,
    eps,
    RBLOCK: tl.constexpr,
):
    """Weight-free RMSNorm Triton kernel: out = x * rsqrt(mean(x², -1) + eps)."""
    row = tl.program_id(0)
    cols = tl.arange(0, RBLOCK)
    mask = cols < D

    x = tl.load(
        x_ptr + row * stride_row + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 0) / D
    rstd = tl.rsqrt(var + eps)

    out = (x * rstd).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * D + cols, out, mask=mask, eviction_policy="evict_first")


def rmsnorm_nw(x: Tensor, eps: float) -> Tensor:
    """Weight-free RMSNorm over the last dimension.

    Treats *x* as ``[num_rows, D]`` where ``num_rows = product(shape[:-1])``.
    Returns a contiguous tensor with the same shape and dtype as *x*.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    num_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    RBLOCK = triton.next_power_of_2(D)

    _rmsnorm_nw_kernel[(num_rows,)](
        x_2d,
        out,
        x_2d.stride(0),
        D,
        eps,
        RBLOCK=RBLOCK,
        num_warps=1 if RBLOCK <= 512 else (4 if RBLOCK <= 4096 else 8),
    )
    return out.view(orig_shape)


@triton.jit
def _hc_head_reduce_store_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        out_ptr + token_idx * out_stride_t + offsets * out_stride_h,
        acc,
        mask=mask,
    )


def hc_head_reduce_triton_kernel(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> None:
    x_flat = x.flatten(-2)
    x_normed = rmsnorm_nw(x_flat, norm_eps)
    mixes = F.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    hidden_size = x.shape[-1]
    hc_mult = x.shape[-2]
    block_h = 1024
    _hc_head_reduce_store_kernel[(x.shape[0], (hidden_size + block_h - 1) // block_h)](
        pre,
        x,
        out,
        hidden_size,
        hc_mult,
        pre.stride(0),
        pre.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Fill pre-allocated `out` (T, H) in-place with the hc_head result."""
    if hs_flat.shape[0] == 0:
        return

    hc_head_reduce_triton_kernel(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_eps,
    )
    return


direct_register_custom_op(
    op_name="hc_head_triton",
    op_func=_hc_head_triton,
    mutates_args=["out"],
)
