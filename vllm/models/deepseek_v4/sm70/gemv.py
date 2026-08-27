# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-token FP16 GEMV for fixed DeepSeek V4 SM70 projections."""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_SUPPORTED_N = frozenset((64, 256, 512, 1024, 2048))
_K = 4096
_BLOCK_K = 1024
_NUM_WARPS = 4
_FUSED_AUX_ROWS = (2048, 512, 64)
_FUSED_AUX_N = sum(_FUSED_AUX_ROWS)


@triton.jit
def _sm70_dsv4_fp16_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = 0.0
    for block_start in tl.static_range(0, K, BLOCK_K):
        x = tl.load(x_ptr + block_start + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + row * K + block_start + offsets).to(tl.float32)
        acc += tl.sum(x * weight, axis=0)
    tl.store(out_ptr + row, acc)


@triton.jit
def _sm70_dsv4_fused_fp16_aux_gemv_kernel(
    x_ptr,
    weight_ptr,
    compressor_out_ptr,
    indexer_compressor_out_ptr,
    indexer_weights_out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPRESSOR_N: tl.constexpr,
    INDEXER_COMPRESSOR_N: tl.constexpr,
):
    """Compute the three C4 auxiliary GEMVs in one exact-FP16 launch."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = 0.0
    for block_start in tl.static_range(0, K, BLOCK_K):
        x = tl.load(x_ptr + block_start + offsets).to(tl.float32)
        weight = tl.load(
            weight_ptr + row * K + block_start + offsets
        ).to(tl.float32)
        acc += tl.sum(x * weight, axis=0)
    tl.store(compressor_out_ptr + row, acc, mask=row < COMPRESSOR_N)
    tl.store(
        indexer_compressor_out_ptr + row - COMPRESSOR_N,
        acc,
        mask=(row >= COMPRESSOR_N)
        & (row < COMPRESSOR_N + INDEXER_COMPRESSOR_N),
    )
    tl.store(
        indexer_weights_out_ptr + row - COMPRESSOR_N - INDEXER_COMPRESSOR_N,
        acc,
        mask=row >= COMPRESSOR_N + INDEXER_COMPRESSOR_N,
    )


def has_sm70_dsv4_fused_fp16_aux_weight_contract(
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    """Check the exact C4 auxiliary row order used by the fused kernel."""
    if tuple(tuple(weight.shape) for weight in weights) != tuple(
        (rows, _K) for rows in _FUSED_AUX_ROWS
    ):
        return False
    device = weights[0].device
    return all(
        weight.dtype == torch.float16
        and weight.ndim == 2
        and weight.shape[1] == _K
        and weight.device == device
        and weight.is_contiguous()
        for weight in weights
    )


@torch.no_grad()
def prepare_sm70_dsv4_fused_fp16_aux_weight(
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor | None:
    if (
        not envs.VLLM_SM70_DSV4_FP16_GEMV
        or not envs.VLLM_SM70_DSV4_FUSED_FP16_AUX_GEMV
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability((7, 0))
        or not weights[0].is_cuda
        or not has_sm70_dsv4_fused_fp16_aux_weight_contract(weights)
    ):
        return None
    return torch.cat(weights, dim=0).contiguous()


def can_use_sm70_dsv4_fused_fp16_aux_gemv(
    x: torch.Tensor,
    fused_weight: torch.Tensor | None,
) -> bool:
    return (
        envs.VLLM_SM70_DSV4_FP16_GEMV
        and envs.VLLM_SM70_DSV4_FUSED_FP16_AUX_GEMV
        and current_platform.is_cuda()
        and current_platform.is_device_capability((7, 0))
        and x.dtype == torch.float16
        and x.ndim == 2
        and x.shape == (1, _K)
        and x.is_contiguous()
        and fused_weight is not None
        and fused_weight.dtype == torch.float16
        and fused_weight.device == x.device
        and fused_weight.shape == (_FUSED_AUX_N, _K)
        and fused_weight.is_contiguous()
    )


def maybe_sm70_dsv4_fused_fp16_aux_gemv(
    x: torch.Tensor,
    fused_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not can_use_sm70_dsv4_fused_fp16_aux_gemv(x, fused_weight):
        return None
    assert fused_weight is not None
    compressor_out = torch.empty(
        (1, _FUSED_AUX_ROWS[0]), device=x.device, dtype=torch.float32
    )
    indexer_compressor_out = torch.empty(
        (1, _FUSED_AUX_ROWS[1]), device=x.device, dtype=torch.float32
    )
    indexer_weights_out = torch.empty(
        (1, _FUSED_AUX_ROWS[2]), device=x.device, dtype=torch.float16
    )
    _sm70_dsv4_fused_fp16_aux_gemv_kernel[(_FUSED_AUX_N,)](
        x,
        fused_weight,
        compressor_out,
        indexer_compressor_out,
        indexer_weights_out,
        K=_K,
        BLOCK_K=_BLOCK_K,
        COMPRESSOR_N=_FUSED_AUX_ROWS[0],
        INDEXER_COMPRESSOR_N=_FUSED_AUX_ROWS[1],
        num_warps=_NUM_WARPS,
    )
    logger.info_once(
        "DeepSeek V4 SM70 exact fused-FP16 C4 auxiliary GEMV enabled."
    )
    return compressor_out, indexer_compressor_out, indexer_weights_out


def can_use_sm70_dsv4_fp16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> bool:
    return (
        envs.VLLM_SM70_DSV4_FP16_GEMV
        and current_platform.is_cuda()
        and current_platform.is_device_capability((7, 0))
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and output_dtype in (torch.float16, torch.float32)
        and x.ndim == 2
        and x.shape == (1, _K)
        and weight.ndim == 2
        and weight.shape[0] in _SUPPORTED_N
        and weight.shape[1] == _K
        and (output_dtype == torch.float32 or weight.shape[0] == 64)
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def maybe_sm70_dsv4_fp16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor | None:
    if not can_use_sm70_dsv4_fp16_gemv(x, weight, output_dtype):
        return None

    logger.info_once(
        "DeepSeek V4 SM70 fixed-shape FP16 GEMV enabled for batch-one decode."
    )
    out = torch.empty((1, weight.shape[0]), device=x.device, dtype=output_dtype)
    _sm70_dsv4_fp16_gemv_kernel[(weight.shape[0],)](
        x,
        weight,
        out,
        K=_K,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
    )
    return out
