# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 normalization providers; preserve each reduction and cast boundary."""

import torch

from vllm import ir
from vllm.config.execution_policy import graph_policy
from vllm.config.sm70_dflash2 import sm70_dflash2_enabled
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@torch.compiler.assume_constant_result
def _sm70_gated_norm_device_supported(device_id: int | None) -> bool:
    # Capability is static for the device guarded by the compiled tensor input.
    # Do not trace the platform's cached NVML/PyTorch capability query.
    return current_platform.is_device_capability(70, device_id=device_id)


def _sm70_gated_norm_shape_supported(
    x: torch.Tensor, z: torch.Tensor | None, weight: torch.Tensor
) -> bool:
    return bool(
        z is not None
        and x.ndim == 2
        and 1 <= x.shape[0] <= 192
        and x.shape[1] == 128
        and z.shape == x.shape
        and weight.shape == (128,)
        and x.dtype == z.dtype == weight.dtype == torch.float16
        and x.device == z.device == weight.device
        and x.is_contiguous()
        and z.is_contiguous()
        and weight.is_contiguous()
    )


def _sm70_rmsnorm_gated_exact_impl(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, silu: bool
) -> torch.Tensor:
    logger.info_once(
        "SM70 exact native gated RMSNorm fusion enabled "
        "(N128, sigmoid/SiLU, M1 and batch)."
    )
    out = torch.empty_like(x)
    torch.ops._C.sm70_rmsnorm_gated_exact_out(out, x, z, weight, eps, silu)
    return out


def _sm70_rmsnorm_gated_exact_fake(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, silu: bool
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="sm70_rmsnorm_gated_exact",
    op_func=_sm70_rmsnorm_gated_exact_impl,
    fake_impl=_sm70_rmsnorm_gated_exact_fake,
)


@triton.jit
def _sm70_dflash2_fixed_gemma_rms_kernel(
    x,
    residual,
    weight,
    normalized_out,
    residual_out,
    HAS_RESIDUAL: tl.constexpr,
    epsilon: tl.constexpr,
):
    # Pin both the reduction extent and warp count. Inductor's 2048/8192
    # autotune changes FP32 reduction order, including between TP ranks.
    row = tl.program_id(0)
    cols = tl.arange(0, 8192)
    mask = cols < 5120
    values = tl.load(x + row * 5120 + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        values += tl.load(residual + row * 5120 + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(residual_out + row * 5120 + cols, values, mask=mask)
    # Preserve the masked square and residual materialization of the pinned
    # Inductor reduction. Removing this boundary changes FMA contraction for
    # sums of two FP16 inputs even with an identical reduction tile.
    variance = tl.sum(tl.where(mask, values * values, 0.0), axis=0) / 5120.0
    inverse_rms = tl.rsqrt(variance + epsilon)
    if HAS_RESIDUAL:
        values = tl.load(residual_out + row * 5120 + cols, mask=mask, other=0.0)
    gemma_weight = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    tl.store(
        normalized_out + row * 5120 + cols,
        values * inverse_rms * gemma_weight,
        mask=mask,
    )


def _sm70_dflash2_fixed_gemma_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    normalized_out = torch.empty_like(x)
    residual_out = (
        torch.empty_like(x, dtype=torch.float32) if residual is not None else None
    )
    _sm70_dflash2_fixed_gemma_rms_kernel[(x.shape[0],)](
        x,
        residual,
        weight,
        normalized_out,
        residual_out,
        HAS_RESIDUAL=residual is not None,
        epsilon=variance_epsilon,
        num_warps=16,
        num_stages=1,
        enable_fp_fusion=True,
    )
    if residual_out is None:
        return normalized_out
    return normalized_out, residual_out


def _use_sm70_dflash2_fixed_gemma_rms(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    policy=None,
    graph=None,
) -> bool:
    return bool(
        sm70_dflash2_enabled("fixed_gemma_rms", policy)
        and (graph if graph is not None else graph_policy()).compile_graph
        and _sm70_gemma_long_prefill_available()
        and x.is_cuda
        and x.dtype == torch.float16
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == 5120
        and x.is_contiguous()
        and weight.device == x.device
        and weight.dtype == torch.float16
        and weight.shape == (5120,)
        and weight.is_contiguous()
        and (
            residual is None
            or (
                residual.dtype == torch.float16
                and residual.device == x.device
                and residual.shape == x.shape
                and residual.is_contiguous()
            )
        )
    )


@triton.jit
def _sm70_dflash2_gemma_fused_add_rms_kernel(
    x,
    residual,
    weight,
    normalized_out,
    residual_out,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    epsilon,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    values = tl.load(x + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    values += tl.load(residual + row * hidden_size + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(residual_out + row * hidden_size + cols, values, mask=mask)

    variance = tl.sum(tl.where(mask, values * values, 0.0), axis=0)
    inverse_rms = tl.rsqrt(variance / hidden_size + epsilon)
    gemma_weight = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    tl.store(
        normalized_out + row * hidden_size + cols,
        values * inverse_rms * gemma_weight,
        mask=mask,
    )


def _sm70_dflash2_gemma_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    *,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    _sm70_dflash2_gemma_fused_add_rms_kernel[(x.shape[0],)](
        x,
        residual,
        weight,
        normalized_out,
        residual_out,
        hidden_size=x.shape[1],
        BLOCK_SIZE=triton.next_power_of_2(x.shape[1]),
        epsilon=variance_epsilon,
        num_warps=num_warps,
        num_stages=1,
    )
    return normalized_out, residual_out


def _sm70_dflash2_gemma_fused_add_rms_boundary(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _sm70_dflash2_gemma_fused_add_rms_norm(x, residual, weight, variance_epsilon)


def _sm70_dflash2_gemma_fused_add_rms_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)


direct_register_custom_op(
    op_name="sm70_dflash2_gemma_fused_add_rms_norm",
    op_func=_sm70_dflash2_gemma_fused_add_rms_boundary,
    fake_impl=_sm70_dflash2_gemma_fused_add_rms_fake,
    mutates_args=[],
)


def _use_sm70_dflash2_gemma_fused_add_rms(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    policy=None,
    graph=None,
) -> bool:
    # Keep the dynamic token dimension out of this Python predicate. AOT traces
    # the target once at a large warmup shape; a decode-only row bound would be
    # constant-folded there and would leave the M=8 replay on the decomposed path.
    return bool(
        sm70_dflash2_enabled("fused_gemma_rms", policy)
        and (graph if graph is not None else graph_policy()).compile_graph
        # Keep the cached platform query out of the AOT fullgraph. The helper
        # below is explicitly constant-foldable by Dynamo.
        and _sm70_gemma_long_prefill_available()
        and residual is not None
        and x.is_cuda
        and x.dtype == torch.float16
        and residual.dtype == torch.float32
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == 5120
        and residual.shape == x.shape
        and residual.device == x.device
        and weight.device == x.device
        and weight.ndim == 1
        and weight.numel() == 5120
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


@torch.compiler.assume_constant_result
def _sm70_gemma_long_prefill_available() -> bool:
    return current_platform.is_device_capability(70)


def _sm70_gemma_rms_norm_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    orig_dtype = x.dtype
    gemma_weight = weight.float() + 1.0
    out = ir.ops.rms_norm(x, gemma_weight, variance_epsilon)
    return out.to(orig_dtype)


def _sm70_gemma_rms_norm_eager_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    return _sm70_gemma_rms_norm_eager(x, weight, variance_epsilon)


def _sm70_gemma_fused_add_rms_norm_eager(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    gemma_weight = weight.float() + 1.0
    x = x.float() + residual.float() if orig_dtype == torch.float16 else x + residual
    residual_out = x
    out = ir.ops.rms_norm(x, gemma_weight, variance_epsilon)
    return out.to(orig_dtype), residual_out


def _sm70_gemma_fused_add_rms_norm_eager_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _sm70_gemma_fused_add_rms_norm_eager(
        x,
        residual,
        weight,
        variance_epsilon,
    )


def _sm70_gemma_long_prefill_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops

    # A long-prefill example can select this custom op while torch.compile is
    # tracing a dynamic graph that is later reused for small CUDA Graph capture
    # sizes. Preserve the normal exact path for those runtime shapes instead of
    # dispatching the long-prefill kernel below its numerical contract.
    if x.shape[0] < 256:
        return _sm70_gemma_fused_add_rms_norm_eager(
            x,
            residual,
            weight,
            variance_epsilon,
        )

    normalized_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    ops.sm70_gemma_long_prefill_fused_add_rms_norm(
        normalized_out,
        residual_out,
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return normalized_out, residual_out


def _sm70_gemma_long_prefill_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, variance_epsilon
    return torch.empty_like(x), torch.empty_like(residual)


direct_register_custom_op(
    op_name="sm70_gemma_rms_norm_eager",
    op_func=_sm70_gemma_rms_norm_eager,
    mutates_args=[],
    fake_impl=_sm70_gemma_rms_norm_eager_fake,
)

direct_register_custom_op(
    op_name="sm70_gemma_fused_add_rms_norm_eager",
    op_func=_sm70_gemma_fused_add_rms_norm_eager,
    mutates_args=[],
    fake_impl=_sm70_gemma_fused_add_rms_norm_eager_fake,
)

direct_register_custom_op(
    op_name="sm70_gemma_long_prefill_fused_add_rms_norm",
    op_func=_sm70_gemma_long_prefill_fused_add_rms_norm,
    mutates_args=[],
    fake_impl=_sm70_gemma_long_prefill_fused_add_rms_norm_fake,
)


def use_long_prefill_fused(
    policy,
    weight,
    x: torch.Tensor,
    residual: torch.Tensor | None,
) -> bool:
    return (
        policy.gemma_long_prefill_fused
        and residual is not None
        and x.is_cuda
        and _sm70_gemma_long_prefill_available()
        and x.dtype == torch.float16
        and residual.dtype == torch.float32
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim == 2
        and x.shape[0] >= 256
        and x.shape[1] == 5120
        and residual.shape == x.shape
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


def maybe_gemma_norm(x, residual, weight, epsilon, *, policy, dflash, graph):
    """Common ordered dispatch; dynamic shape/dtype tests stay at the call site.

    Fixed reduction, fused FP32 residual and mixed-dtype long-prefill retain
    distinct arithmetic. None means the caller must use its native/eager tail.
    """
    if _use_sm70_dflash2_fixed_gemma_rms(
        x,
        residual,
        weight,
        dflash,
        graph,
    ):
        return _sm70_dflash2_fixed_gemma_rms_norm(x, residual, weight, epsilon)
    if _use_sm70_dflash2_gemma_fused_add_rms(
        x,
        residual,
        weight,
        dflash,
        graph,
    ):
        assert residual is not None
        return torch.ops.vllm.sm70_dflash2_gemma_fused_add_rms_norm(
            x,
            residual,
            weight,
            epsilon,
        )
    if use_long_prefill_fused(policy, weight, x, residual):
        assert residual is not None
        if not torch.compiler.is_compiling():
            logger.info_once(
                "SM70 exact mixed-dtype Gemma RMSNorm long-prefill path active."
            )
        return torch.ops.vllm.sm70_gemma_long_prefill_fused_add_rms_norm(
            x,
            residual,
            weight,
            epsilon,
        )
    return None
