# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        row_out + block,
        (x * rrms * weight).to(row_out.dtype.element_ty),
        mask=mask,
    )


def fused_q_kv_rmsnorm(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.ndim == 2 and kv.ndim == 2
    assert q.shape[0] == kv.shape[0], f"token dim mismatch: q={q.shape}, kv={kv.shape}"
    assert q.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv)
    if q.shape[0] == 0:
        return q_out, kv_out

    block_size = triton.next_power_of_2(max(q.shape[1], kv.shape[1]))
    _fused_q_kv_rmsnorm_kernel[(q.shape[0], 2)](
        q,
        q_out,
        q_weight,
        q.stride(0),
        q_out.stride(0),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        eps,
        Q_SIZE=q.shape[1],
        KV_SIZE=kv.shape[1],
        BLOCK_SIZE=block_size,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return q_out, kv_out
