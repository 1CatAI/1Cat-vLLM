# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape-checked checkpoint-FP16 Qwen4Exp decode routes for SM70.

Admission depends on the individual projection rather than a model/TP profile.
GEMV covers single-token
decode and draft; a separately gated packed GDN input kernel covers M2..16 batch
decode. All prefill and unsupported shapes retain the ordinary unquantized
linear path.
"""

from __future__ import annotations

import os
from types import MethodType
from typing import NamedTuple

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class _GemvPlan(NamedTuple):
    block_k: int
    num_warps: int
    load_policy: int


_HC_DOWN_SUFFIX = ".input_mix_weight_down_block_inject"
_GDN_QKVZ_SUFFIX = ".linear_attn.in_proj_qkvz"
_GDN_BA_SUFFIX = ".linear_attn.in_proj_ba"
_GDN_OUT_SUFFIX = ".linear_attn.out_proj"
_QSA_QKV_SUFFIX = ".self_attn.qkv_proj"
_QSA_OUT_SUFFIX = ".self_attn.o_proj"
_QSA_INDEX_SUFFIX = ".self_attn.indexer.index_qk_proj"
_ROUTER_SUFFIX = ".mlp.gate"
_SHARED_UP_SUFFIX = ".mlp.shared_expert.gate_up_proj"

# Plans are cold-cache CUDA Graph winners on real checkpoint weights. Keep the
# role in the key: GDN and QSA can share a physical shape while remaining
# independently auditable.
_ROLE_PLANS: tuple[tuple[str, tuple[int, int], _GemvPlan], ...] = (
    (_HC_DOWN_SUFFIX, (336, 10240), _GemvPlan(512, 4, 1)),
    (_GDN_QKVZ_SUFFIX, (4096, 2560), _GemvPlan(512, 2, 0)),
    (_GDN_BA_SUFFIX, (24, 2560), _GemvPlan(512, 4, 0)),
    (_GDN_OUT_SUFFIX, (2560, 1536), _GemvPlan(512, 4, 1)),
    (_QSA_QKV_SUFFIX, (3584, 2560), _GemvPlan(512, 2, 0)),
    (_QSA_OUT_SUFFIX, (2560, 1536), _GemvPlan(512, 4, 0)),
    (_QSA_INDEX_SUFFIX, (640, 2560), _GemvPlan(512, 2, 0)),
    (_ROUTER_SUFFIX, (512, 2560), _GemvPlan(1024, 8, 0)),
)

# Retain the legacy two-argument custom-op behavior for external callers.
# Model layers pass their role explicitly: same-shape GDN/QSA plans differ.
_SHAPE_PLANS = {shape: plan for _, shape, plan in _ROLE_PLANS}


@triton.jit
def _qwen38_gdn_projection_split_kernel(
    qkvz,
    ba,
    qkv,
    z,
    b,
    a,
    QKV: tl.constexpr,
    Z: tl.constexpr,
    B: tl.constexpr,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
    COPY_QKV: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(qkvz + row * (QKV + Z) + col, col < QKV + Z, other=0)
    if COPY_QKV:
        tl.store(qkv + row * QKV + col, value, col < QKV)
    tl.store(z + row * Z + col - QKV, value, (col >= QKV) & (col < QKV + Z))
    if tl.program_id(1) == 0:
        tl.static_assert(B + A <= BLOCK)
        gate_col = tl.arange(0, BLOCK)
        gate = tl.load(ba + row * (B + A) + gate_col, gate_col < B + A, other=0)
        tl.store(b + row * B + gate_col, gate, gate_col < B)
        tl.store(a + row * A + gate_col - B, gate, (gate_col >= B) & (gate_col < B + A))


def _split_gdn_projection_outputs(qkvz, ba):
    m = qkvz.shape[0]
    out = tuple(qkvz.new_empty((m, n)) for n in (2560, 1536, 12, 12))
    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
        qkvz,
        ba,
        *out,
        QKV=2560,
        Z=1536,
        B=12,
        A=12,
        BLOCK=256,
        COPY_QKV=True,
        num_warps=4,
        num_stages=1,
    )
    return out


def _split_gdn_projection_tails(qkvz, ba, z_out):
    """Copy z/b/a together while preserving the input QKV view and stride."""
    m = qkvz.shape[0]
    qkv = qkvz[:, :2560]
    b, a = (qkvz.new_empty((m, 12)) for _ in range(2))
    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
        qkvz,
        ba,
        qkv,
        z_out,
        b,
        a,
        QKV=2560,
        Z=1536,
        B=12,
        A=12,
        BLOCK=256,
        COPY_QKV=False,
        num_warps=4,
        num_stages=1,
    )
    return qkv, b, a


def _can_fuse_gdn_projection_split(qkvz: torch.Tensor, ba: torch.Tensor) -> bool:
    # This copy-only operation does not change either GEMM. Keep the existing
    # M1 path and all unsupported layouts; no maximum batch/sequence binding.
    return bool(
        envs.VLLM_SM70_GDN_BATCH_SPLIT_COPY
        and _is_packed_row_major(qkvz)
        and _is_packed_row_major(ba)
        and qkvz.shape[0] > 1
        and qkvz.shape[1] == 4096
        and ba.shape == (qkvz.shape[0], 24)
        and qkvz.dtype == ba.dtype == torch.float16
        and qkvz.is_cuda
        and ba.is_cuda
        and qkvz.device == ba.device
        and current_platform.is_device_capability(70)
    )


@triton.jit
def _qwen38_fp16_row_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOAD_POLICY: tl.constexpr,
    N: tl.constexpr = 0,
):
    row = tl.program_id(0)
    token = tl.program_id(1)
    x_ptr += token * K
    out_ptr += token * N
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        if LOAD_POLICY == 1:
            x = tl.load(
                x_ptr + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            x = tl.load(x_ptr + indices, mask=mask, other=0.0)
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
            )
        acc += x.to(tl.float32) * weight.to(tl.float32)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def _qwen38_fp16_gdn_input_kernel(
    x_ptr,
    qkvz_weight_ptr,
    ba_weight_ptr,
    qkv_out_ptr,
    z_out_ptr,
    b_out_ptr,
    a_out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    is_qkvz = row < 4096
    ba_row = row - 4096
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        qkvz_weight = tl.load(
            qkvz_weight_ptr + row * K + indices,
            mask=is_qkvz & mask,
            other=0.0,
        )
        ba_weight = tl.load(
            ba_weight_ptr + ba_row * K + indices,
            mask=(~is_qkvz) & mask,
            other=0.0,
        )
        weight = tl.where(is_qkvz, qkvz_weight, ba_weight)
        acc += x.to(tl.float32) * weight.to(tl.float32)

    value = tl.sum(acc, axis=0)
    is_qkv = is_qkvz & (row < 2560)
    is_z = is_qkvz & (row >= 2560)
    is_b = (~is_qkvz) & (ba_row < 12)
    is_a = (~is_qkvz) & (ba_row >= 12)
    tl.store(qkv_out_ptr + row, value, mask=is_qkv)
    tl.store(z_out_ptr + row - 2560, value, mask=is_z)
    tl.store(b_out_ptr + ba_row, value, mask=is_b)
    tl.store(a_out_ptr + ba_row - 12, value, mask=is_a)


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return bool(
        not envs.VLLM_BATCH_INVARIANT
        and x.shape[0] == 1
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
        and current_platform.is_device_capability(70)
    )


def _pack_router_batch_weight(weight: torch.Tensor) -> torch.Tensor:
    """Keep four K640 partitions contiguous for each N8 output tile."""
    if weight.dtype != torch.float16 or weight.shape != (512, 2560):
        raise ValueError("Batch router packing requires FP16 [512, 2560]")
    return (
        weight.detach()
        .reshape(64, 8, 4, 40, 2, 8)
        .permute(0, 3, 4, 2, 1, 5)
        .contiguous()
    )


def _shared_batch_runtime_ok(x: torch.Tensor) -> bool:
    return bool(
        envs.VLLM_SM70_MTP_SHARED_BATCH
        and not envs.VLLM_BATCH_INVARIANT
        and x.ndim == 2
        and x.shape[0] in (5, 10)
        and x.shape[1] == 2560
        and x.is_cuda
        and x.dtype == torch.float16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
    )


def _qwen38_sm70_shared_up(
    x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor | None
) -> torch.Tensor:
    if _shared_batch_runtime_ok(x) and packed is not None:
        out = x.new_empty((x.shape[0], 160))
        reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        if reduced:
            partial = x.new_empty((8, x.shape[0], 320))
            torch.ops._C.qwen38_shared_up_batch_sm70_out(out, partial, x, packed)
        else:
            partial = x.new_empty((8, x.shape[0], 320), dtype=torch.float32)
            torch.ops._C.qwen38_shared_up_batch_fp32_sm70_out(out, partial, x, packed)
        logger.info_once(
            "SM70 MTP4 shared-expert batch projection enabled (partials=%s).",
            partial.dtype,
        )
        return out
    gate_up = torch.nn.functional.linear(x, weight)
    out = gate_up.new_empty((*gate_up.shape[:-1], 160))
    torch.ops._C.silu_and_mul(out, gate_up)
    return out


def _qwen38_sm70_shared_up_fake(
    x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor | None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 160))


direct_register_custom_op(
    op_name="qwen38_sm70_shared_up",
    op_func=_qwen38_sm70_shared_up,
    fake_impl=_qwen38_sm70_shared_up_fake,
)


def _qwen38_sm70_shared_gate_mul(
    logits: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    if _shared_batch_runtime_ok(source):
        out = torch.empty_like(source)
        torch.ops._C.qwen38_shared_gate_mul_sm70_out(out, logits, source)
        return out
    return torch.sigmoid(logits) * source


def _qwen38_sm70_shared_gate_mul_fake(
    logits: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(source)


direct_register_custom_op(
    op_name="qwen38_sm70_shared_gate_mul",
    op_func=_qwen38_sm70_shared_gate_mul,
    fake_impl=_qwen38_sm70_shared_gate_mul_fake,
)


def _forward_shared_batch_silu(layer, x):
    if not use_sm70_decode_graph_semantics():
        return None
    return torch.ops.vllm.qwen38_sm70_shared_up(
        x, layer.weight, getattr(layer, "_sm70_mtp_shared_packed", None)
    )


def _router_batch_runtime_ok(x, packed) -> bool:
    return bool(
        envs.VLLM_SM70_MTP_ROUTER_BATCH
        and not envs.VLLM_BATCH_INVARIANT
        and x.ndim == 2
        and x.shape[0] in (5, 10)
        and x.shape[1] == 2560
        and x.is_cuda
        and x.dtype == torch.float16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and packed is not None
        and packed.shape == (64, 40, 2, 4, 8, 8)
        and packed.device == x.device
        and packed.dtype == x.dtype
        and packed.is_contiguous()
        and packed.data_ptr() % 16 == 0
        # This kernel keeps all four partials and their ordered sum in FP32.
        # The cuBLAS reduced-precision switch does not describe its arithmetic.
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
    )


def _qwen38_sm70_fp16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    role: str = "",
    packed_router: torch.Tensor | None = None,
    dense_batch: bool = False,
    output_codes: torch.Tensor | None = None,
    output_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    if dense_batch and _can_use_dense_batch(x, weight, role):
        out = x.new_empty((x.shape[0], weight.shape[0]))
        if output_codes is not None and output_scales is not None:
            from vllm import _sm70_ops as ops

            ops.fp8_qpn8_gemm_sm70_out(
                out, x, output_codes, output_scales, 12, 2, True, False
            )
            logger.info_once("SM70 QPN8 batch output projection candidate enabled.")
            return out
        torch.ops._C.qwen38_dense_batch_sm70_out(out, x, weight)
        logger.info_once("SM70 Qwen3.8 exact small-batch dense projections enabled.")
        return out
    if role.endswith(_ROUTER_SUFFIX) and _router_batch_runtime_ok(x, packed_router):
        out = x.new_empty((x.shape[0], 512))
        torch.ops._C.qwen38_router_batch_sm70_out(out, x, packed_router)
        logger.info_once("SM70 MTP4 batch router with ordered FP32 splits enabled.")
        return out
    shape = (weight.shape[0], weight.shape[1])
    plan = _plan_for(role, shape) if role else _SHAPE_PLANS.get(shape)
    small_ba = bool(
        role.endswith(_GDN_BA_SUFFIX)
        and not envs.VLLM_BATCH_INVARIANT
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
        and x.ndim == 2
        and 2 <= x.shape[0] <= 32
        and x.shape[1] == 2560
        and 0 < weight.shape[0] <= 32
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == weight.dtype == torch.float16
        and x.is_cuda
        and weight.device == x.device
        and current_platform.is_device_capability(70)
    )
    if plan is None or not (_runtime_ok(x, weight) or small_ba):
        return torch.nn.functional.linear(x, weight)

    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _qwen38_fp16_row_gemv_kernel[(weight.shape[0], x.shape[0])](
        x,
        weight,
        out,
        K=weight.shape[1],
        N=weight.shape[0],
        BLOCK_K=plan.block_k,
        LOAD_POLICY=plan.load_policy,
        num_warps=plan.num_warps,
    )
    if small_ba:
        logger.info_once(
            "SM70 checkpoint-FP16 batch a/b row GEMV enabled (M=%d, N=%d).",
            x.shape[0],
            weight.shape[0],
        )
    else:
        logger.info_once("SM70 Qwen3.8 checkpoint-FP16 M=1 GEMV route enabled.")
    return out


def _qwen38_sm70_fp16_gemv_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    role: str = "",
    packed_router: torch.Tensor | None = None,
    dense_batch: bool = False,
    output_codes: torch.Tensor | None = None,
    output_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gemv",
    op_func=_qwen38_sm70_fp16_gemv,
    fake_impl=_qwen38_sm70_fp16_gemv_fake,
)


def _dense_batch_limit(role: str, shape: tuple[int, ...]) -> int:
    # Reject measured regressions: router M8/M16 and output M16. The C1
    # reduction is a different tree and must keep its existing route.
    if role.endswith(_ROUTER_SUFFIX) and shape == (512, 2560):
        return 4
    if role.endswith((_GDN_OUT_SUFFIX, _QSA_OUT_SUFFIX)) and shape == (2560, 1536):
        return 8
    return 0


def _can_use_dense_batch(x: torch.Tensor, weight: torch.Tensor, role: str) -> bool:
    return bool(
        envs.VLLM_SM70_QWEN38_BATCH_FASTPATH
        and not envs.VLLM_BATCH_INVARIANT
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
        # With reduced-precision reductions allowed, cuBLAS uses FP16 partials
        # for the small output projections. Retain that baseline schedule;
        # the native FP32 kernel differs at M2/M4/M5 in the operator oracle.
        and not (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            and tuple(weight.shape) == (2560, 1536)
            and x.shape[0] < 8
        )
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and 2 <= x.shape[0] <= _dense_batch_limit(role, tuple(weight.shape))
        and x.shape[1] == weight.shape[1]
        and x.is_cuda
        and weight.is_cuda
        and x.dtype == weight.dtype == torch.float16
        and x.device == weight.device
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and current_platform.is_device_capability(70)
    )


def _pack_gdn_input_weight(weight: torch.Tensor) -> torch.Tensor:
    """Copy FP16 bits into N32/K16 tiles; keep originals for M1/prefill."""
    if weight.dtype != torch.float16 or weight.shape not in ((4096, 2560), (24, 2560)):
        raise ValueError("Unsupported packed GDN input weight")
    weight = weight.detach()
    if weight.shape[0] == 24:
        weight = torch.cat((weight, weight.new_zeros((8, 2560))))
    return weight.reshape(-1, 32, 160, 2, 8).permute(0, 2, 3, 1, 4).contiguous()


def _can_use_packed_gdn_input(x, packed_qkvz, packed_ba) -> bool:
    return bool(
        (envs.VLLM_SM70_QWEN38_GDN_INPUT_BATCH or envs.VLLM_SM70_QWEN38_BATCH_FASTPATH)
        and not envs.VLLM_BATCH_INVARIANT
        # The packed MMA preserves the original FP32 accumulation contract.
        # Let cuBLAS honor an explicit request for FP16 accumulation.
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
        and x.ndim == 2
        and 2 <= x.shape[0] <= 16
        and x.shape[1] == 2560
        and x.is_cuda
        and x.dtype == torch.float16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and packed_qkvz is not None
        and packed_ba is not None
        and packed_qkvz.shape == (128, 160, 2, 32, 8)
        and packed_ba.shape == (1, 160, 2, 32, 8)
        and all(
            w.device == x.device
            and w.dtype == x.dtype
            and w.is_contiguous()
            and w.data_ptr() % 16 == 0
            for w in (packed_qkvz, packed_ba)
        )
        and current_platform.is_device_capability(70)
    )


def _qwen38_sm70_fp16_gdn_input(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    packed_qkvz: torch.Tensor | None = None,
    packed_ba: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _can_use_packed_gdn_input(x, packed_qkvz, packed_ba):
        qkv, z, b, a = (x.new_empty((x.shape[0], n)) for n in (2560, 1536, 12, 12))
        torch.ops._C.qwen38_gdn_input_batch_sm70_out(
            qkv, z, b, a, x, packed_qkvz, packed_ba
        )
        logger.info_once("SM70 Qwen3.8 checkpoint-FP16 batched GDN input enabled.")
        return qkv, z, b, a
    if not (
        qkvz_weight.shape == (4096, 2560)
        and ba_weight.shape == (24, 2560)
        and _runtime_ok(x, qkvz_weight)
        and _runtime_ok(x, ba_weight)
    ):
        qkvz = torch.nn.functional.linear(x, qkvz_weight)
        ba = torch.nn.functional.linear(x, ba_weight)
        if _can_fuse_gdn_projection_split(qkvz, ba):
            logger.info_once("SM70 GDN batched projection split-copy fusion enabled.")
            return _split_gdn_projection_outputs(qkvz, ba)
        return (
            qkvz[..., :2560].contiguous(),
            qkvz[..., 2560:].contiguous(),
            ba[..., :12].contiguous(),
            ba[..., 12:].contiguous(),
        )

    qkv = x.new_empty((1, 2560))
    z = x.new_empty((1, 1536))
    b = x.new_empty((1, 12))
    a = x.new_empty((1, 12))
    _qwen38_fp16_gdn_input_kernel[(4096 + 24,)](
        x,
        qkvz_weight,
        ba_weight,
        qkv,
        z,
        b,
        a,
        K=2560,
        BLOCK_K=512,
        num_warps=2,
    )
    logger.info_once("SM70 Qwen3.8 checkpoint-FP16 fused GDN input route enabled.")
    return qkv, z, b, a


def _qwen38_sm70_fp16_gdn_input_fake(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    packed_qkvz: torch.Tensor | None = None,
    packed_ba: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del qkvz_weight, ba_weight, packed_qkvz, packed_ba
    batch_shape = x.shape[:-1]
    return (
        x.new_empty((*batch_shape, 2560)),
        x.new_empty((*batch_shape, 1536)),
        x.new_empty((*batch_shape, 12)),
        x.new_empty((*batch_shape, 12)),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gdn_input",
    op_func=_qwen38_sm70_fp16_gdn_input,
    fake_impl=_qwen38_sm70_fp16_gdn_input_fake,
)


def _mtp_batch_packing_allowed(layer: nn.Module, role: str) -> bool:
    # Match the router/shared runtime precision guards without changing their
    # forward hooks or fallbacks. The worker sets this policy before loading;
    # configure it before preparing weights, not during graph replay.
    matmul = torch.backends.cuda.matmul
    reason = None
    # Router always uses FP32 partials; shared up selects their dtype from
    # the worker policy. Older binaries retain the vendor FP32 fallback.
    if matmul.allow_fp16_accumulation:
        reason = "fp16_accumulation_enabled"
    elif (
        role == "shared"
        and not matmul.allow_fp16_reduced_precision_reduction
        and not hasattr(torch.ops._C, "qwen38_shared_up_batch_fp32_sm70_out")
    ):
        reason = "fp32_shared_operator_missing"
    # The loaded-worker report already collects _sm70_*_reason attributes.
    setattr(layer, f"_sm70_mtp_{role}_batch_reason", reason)
    if reason is not None:
        logger.info_once(
            "Skipping SM70 MTP %s packed weights due to precision policy: %s.",
            role,
            reason,
            scope="process",
        )
    return reason is None


class Qwen38SM70FP16LinearMethod(UnquantizedLinearMethod):
    """Prepare admitted FP16 projections; use row GEMV for single tokens."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if (
            getattr(layer, "_sm70_qwen38_dense_batch", False)
            and layer.weight.is_cuda
            and not hasattr(torch.ops._C, "qwen38_dense_batch_sm70_out")
        ):
            raise RuntimeError(
                "Rebuild the SM70 extension for batched dense projections"
            )
        if getattr(
            layer, "_sm70_mtp_prepare_shared_batch", False
        ) and _mtp_batch_packing_allowed(layer, "shared"):
            weight = layer.weight
            if weight.is_cuda and weight.dtype == torch.float16:
                if not hasattr(torch.ops._C, "qwen38_shared_up_batch_sm70_out"):
                    raise RuntimeError("Rebuild the SM70 extension for shared expert")
                layer.register_buffer(
                    "_sm70_mtp_shared_packed",
                    weight.detach()
                    .reshape(10, 32, 160, 2, 8)
                    .permute(0, 2, 3, 1, 4)
                    .contiguous(),
                    persistent=False,
                )
        if getattr(layer, "_sm70_qwen38_hc_batch_role", None) is not None:
            from .sm70_fp16_hc import _prepare_hc_batch_weight

            _prepare_hc_batch_weight(layer)
        if getattr(
            layer, "_sm70_mtp_prepare_router_batch", False
        ) and _mtp_batch_packing_allowed(layer, "router"):
            weight = layer.weight
            if weight.is_cuda and weight.dtype == torch.float16:
                if not hasattr(torch.ops._C, "qwen38_router_batch_sm70_out"):
                    raise RuntimeError("Rebuild the SM70 extension for batch router")
                layer.register_buffer(
                    "_sm70_mtp_router_packed",
                    _pack_router_batch_weight(weight),
                    persistent=False,
                )
        if not getattr(layer, "_sm70_qwen38_prepare_gdn_batch", False):
            return
        weight = layer.weight
        if not weight.is_cuda or weight.dtype != torch.float16:
            return  # The PLE-only/meta loader never executes this GPU route.
        if not hasattr(torch.ops._C, "qwen38_gdn_input_batch_sm70_out"):
            raise RuntimeError("Rebuild the SM70 extension for batched GDN input")
        layer.register_buffer(
            "_sm70_qwen38_gdn_packed", _pack_gdn_input_weight(weight), persistent=False
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The first dynamic compile sees prefill (M > 1). Keep the M=1
        # decision inside the opaque op so decode does not inherit a baked-in
        # prefill branch.
        if bias is None and use_sm70_decode_graph_semantics():
            return torch.ops.vllm.qwen38_sm70_fp16_gemv(
                x,
                layer.weight,
                getattr(layer, "prefix", ""),
                getattr(layer, "_sm70_mtp_router_packed", None),
                getattr(layer, "_sm70_qwen38_dense_batch", False),
                getattr(layer, "_sm70_qwen38_output_codes", None),
                getattr(layer, "_sm70_qwen38_output_scales", None),
            )
        return super().apply(layer, x, bias)


def _plan_for(prefix: str, shape: tuple[int, int]) -> _GemvPlan | None:
    for suffix, expected_shape, plan in _ROLE_PLANS:
        if prefix.endswith(suffix) and shape == expected_shape:
            return plan
    # The row kernel masks K tails and accumulates in FP32; the tuned plans
    # above are preferred, but different TP shards/model widths are legal.
    if min(shape) > 0 and any(prefix.endswith(role) for role, _, _ in _ROLE_PLANS):
        return _GemvPlan(512, 4, 0)
    return None


def _exact_runtime_contract(vllm_config=None) -> bool:
    try:
        config = vllm_config or get_current_vllm_config()
        from vllm.config.vllm import _is_sm70_qwen38_decode_compile_contract

        return _is_sm70_qwen38_decode_compile_contract(
            config.model_config, config.speculative_config, config.parallel_config
        )
    except (AssertionError, AttributeError, RuntimeError):
        return False


def _batch_runtime_contract(vllm_config=None) -> bool:
    if not _exact_runtime_contract(vllm_config) or envs.VLLM_BATCH_INVARIANT:
        return False
    config = vllm_config or get_current_vllm_config()
    from vllm.model_executor.models.config import sm70_flash_next_batch_qualified

    # Verifier and draft projections obey the same local shape/layout checks.
    # Speculation is not an operator capability. HC retains its separate
    # split-K numerical policy, selected in its loader.
    return sm70_flash_next_batch_qualified(config) and not getattr(
        config.parallel_config, "use_ubatching", False
    )


def enable_qwen38_sm70_fp16_gemv(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Replace admitted unquantized methods before checkpoint loading."""
    capability_ok = current_platform.is_device_capability((7, 0))
    contract_ok = _exact_runtime_contract(vllm_config)
    if (
        capability_ok
        and contract_ok
        and dtype == torch.float16
        and (vllm_config or get_current_vllm_config()).speculative_config is None
    ):
        # The no-MTP control and candidate both use FP32 accumulation/reduction.
        # Never recover throughput by truncating intermediate GEMM sums.
        # MTP has separate explicit numerical-policy admission; preserve it.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        logger.info_once("SM70 Qwen3.8 FP32 GEMM accumulation/reductions required.")
    if not envs.VLLM_SM70_QWEN38_FP16_GEMV:
        return
    if (
        envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
        or dtype != torch.float16
        or not capability_ok
        or not contract_ok
    ):
        logger.warning_once(
            "Qwen3.8 checkpoint-FP16 GEMV opt-in rejected: "
            "online_qpn8=%s dtype=%s sm70=%s exact_contract=%s.",
            envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8,
            dtype,
            capability_ok,
            contract_ok,
        )
        return

    replaced = 0
    batch_allowed = bool(
        envs.VLLM_SM70_QWEN38_BATCH_FASTPATH and _batch_runtime_contract(vllm_config)
    )
    for child in module.modules():
        if not (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        shape = (int(weight.shape[0]), int(weight.shape[1]))
        prefix = str(getattr(child, "prefix", ""))
        shared_batch = False
        if envs.VLLM_SM70_MTP_SHARED_BATCH:
            from .sm70_fp16_hc import _mtp_batch_runtime_contract

            shared_batch = (
                prefix.endswith(_SHARED_UP_SUFFIX)
                and shape == (320, 2560)
                and _mtp_batch_runtime_contract(vllm_config)
            )
        if _plan_for(prefix, shape) is None and not shared_batch:
            continue
        child.quant_method = Qwen38SM70FP16LinearMethod()
        child._sm70_qwen38_dense_batch = bool(
            batch_allowed and _dense_batch_limit(prefix, shape)
        )
        if shared_batch:
            child._sm70_mtp_prepare_shared_batch = True
            child.forward_fused_silu_and_mul = MethodType(
                _forward_shared_batch_silu, child
            )
        if envs.VLLM_SM70_MTP_ROUTER_BATCH and str(
            getattr(child, "prefix", "")
        ).endswith(_ROUTER_SUFFIX):
            from .sm70_fp16_hc import _mtp_batch_runtime_contract

            if _mtp_batch_runtime_contract(vllm_config):
                child._sm70_mtp_prepare_router_batch = True
        replaced += 1

    fused_gdn_inputs = 0
    if envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16:
        for child in module.modules():
            qkvz = getattr(child, "in_proj_qkvz", None)
            ba = getattr(child, "in_proj_ba", None)
            qkvz_weight = getattr(qkvz, "weight", None)
            ba_weight = getattr(ba, "weight", None)
            if not (
                isinstance(qkvz_weight, torch.Tensor)
                and qkvz_weight.shape == (4096, 2560)
                and isinstance(ba_weight, torch.Tensor)
                and ba_weight.shape == (24, 2560)
                and isinstance(
                    getattr(qkvz, "quant_method", None),
                    Qwen38SM70FP16LinearMethod,
                )
                and isinstance(
                    getattr(ba, "quant_method", None),
                    Qwen38SM70FP16LinearMethod,
                )
                and not bool(getattr(child, "gqa_interleaved_layout", True))
                and not bool(getattr(child, "disable_tp_for_ba_proj", True))
            ):
                continue
            child.sm70_qwen38_fp16_fused_input = True
            # Automatic promotion covers ordinary decode and MTP. Retain an
            # explicit legacy opt-in for other proposers; they need paired
            # quality before this default can be widened.
            batch_qualified = _batch_runtime_contract(vllm_config)
            explicit_gdn_batch = "VLLM_SM70_QWEN38_GDN_INPUT_BATCH" in os.environ
            if (
                envs.VLLM_SM70_QWEN38_GDN_INPUT_BATCH
                and (batch_qualified or explicit_gdn_batch)
            ) or (envs.VLLM_SM70_QWEN38_BATCH_FASTPATH and batch_qualified):
                assert qkvz is not None and ba is not None
                qkvz._sm70_qwen38_prepare_gdn_batch = True
                ba._sm70_qwen38_prepare_gdn_batch = True
            fused_gdn_inputs += 1

    if replaced:
        logger.info_once(
            "Prepared %d Qwen3.8 checkpoint-FP16 SM70 M=1 GEMV projections.",
            replaced,
        )
    else:
        logger.warning_once(
            "Qwen3.8 checkpoint-FP16 GEMV opt-in matched the runtime but no "
            "target projections were found."
        )
    if fused_gdn_inputs:
        logger.info_once(
            "Prepared %d Qwen3.8 fused checkpoint-FP16 GDN inputs.",
            fused_gdn_inputs,
        )
    elif envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16:
        logger.warning_once(
            "Qwen3.8 fused checkpoint-FP16 GDN input opt-in found no targets."
        )


__all__ = [
    "Qwen38SM70FP16LinearMethod",
    "_qwen38_fp16_gdn_input_kernel",
    "enable_qwen38_sm70_fp16_gemv",
]
