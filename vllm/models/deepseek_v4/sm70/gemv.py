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
