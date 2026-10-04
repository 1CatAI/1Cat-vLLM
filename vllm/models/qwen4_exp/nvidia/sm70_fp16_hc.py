# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in fused checkpoint-FP16 HyperConnection decode route for SM70."""

from __future__ import annotations

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.fp16_gemv_silu import Sm70Fp16GemvSiluKernel
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .sm70_fp16_gemv import (
    Qwen38SM70FP16LinearMethod,
    _batch_runtime_contract,
    _exact_runtime_contract,
)

logger = init_logger(__name__)

_HC_COUNT = 4
_HC_DIM = 2560
_HC_RANK = 320
_HC_HIDDEN = _HC_COUNT * _HC_DIM


def _mtp_batch_runtime_contract(vllm_config=None) -> bool:
    if not _exact_runtime_contract(vllm_config) or envs.VLLM_BATCH_INVARIANT:
        return False
    config = vllm_config or get_current_vllm_config()
    speculative = config.speculative_config
    return bool(
        speculative is not None
        and getattr(speculative, "method", None) == "mtp"
        # M=5/10 arithmetic is checked at dispatch, independently of draft width.
        and not getattr(config.parallel_config, "use_ubatching", False)
    )


def _pack_hc_batch_weight(
    weight: torch.Tensor, role: str, rank: int | None
) -> torch.Tensor:
    """Lossless batch packs; keep checkpoint layout for M1 and prefill."""
    if weight.dtype != torch.float16 or (rank is not None and not 0 <= rank < 4):
        raise ValueError("HC batch packing requires FP16 and a TP4 rank")
    weight = weight.detach()
    if role == "down" and weight.shape == (336, 10240):
        if rank is None:
            padded = weight.new_zeros((352, 10240))
            padded[:336].copy_(weight)
            return padded.reshape(11, 32, 640, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
        padded = weight.new_zeros((96, 10240))
        padded[:88].copy_(weight[rank * 80 : rank * 80 + 88])
        return padded.reshape(3, 32, 640, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
    if role == "up" and weight.shape == (10240, 320):
        shard = weight.reshape(4, 2560, 320)
        if rank is not None:
            shard = shard[:, rank * 640 : (rank + 1) * 640]
        return (
            shard.contiguous()
            .reshape(4, 320 if rank is None else 80, 8, 20, 2, 8)
            .permute(1, 3, 4, 0, 2, 5)
            .contiguous()
        )
    raise ValueError("Unsupported HC batch weight role/geometry")


def _prepare_hc_batch_weight(layer: nn.Module) -> None:
    from vllm import _custom_ops as ops
    from vllm.distributed.parallel_state import get_tp_group

    weight = layer.weight
    if not weight.is_cuda or weight.dtype != torch.float16:
        return
    if not ops.supports_sm70_qwen38_hc_batch():
        raise RuntimeError("Rebuild the SM70 extension for batched HC")
    rank: int | None = get_tp_group().rank_in_group
    ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    if (
        not (ca is not None and not ca.disabled and ca.fully_connected)
        and not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        and ops.supports_sm70_qwen38_hc_replicated()
    ):
        rank = None
    layer.register_buffer(
        "_sm70_qwen38_hc_batch_packed",
        _pack_hc_batch_weight(weight, layer._sm70_qwen38_hc_batch_role, rank),
        persistent=False,
    )


def _batch_runtime_ok(x, packed_down, packed_up, concurrent_batch=False) -> bool:
    if x.ndim != 2:
        return False
    reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    admitted = (
        envs.VLLM_SM70_QWEN38_BATCH_FASTPATH and not reduced and 2 <= x.shape[0] <= 16
        if concurrent_batch
        else envs.VLLM_SM70_MTP_HC_BATCH and reduced and x.shape[0] in (5, 10)
    )

    return bool(
        admitted
        and not envs.VLLM_BATCH_INVARIANT
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
        and x.shape[1] == _HC_HIDDEN
        and x.is_cuda
        and x.dtype == torch.float16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and packed_down is not None
        and packed_up is not None
        and packed_down.shape == (3, 640, 2, 32, 8)
        and packed_up.shape == (80, 20, 2, 4, 8, 8)
        and all(
            w.device == x.device
            and w.dtype == x.dtype
            and w.is_contiguous()
            and w.data_ptr() % 16 == 0
            for w in (packed_down, packed_up)
        )
    )


def _replicated_runtime_ok(x, down, up, concurrent_batch):
    return bool(
        concurrent_batch
        and envs.VLLM_SM70_QWEN38_BATCH_FASTPATH
        and not envs.VLLM_BATCH_INVARIANT
        and not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
        and x.ndim == 2
        and 2 <= x.shape[0] <= 16
        and x.shape[1] == _HC_HIDDEN
        and x.is_cuda
        and x.dtype == torch.float16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and down is not None
        and up is not None
        and down.shape == (11, 640, 2, 32, 8)
        and up.shape == (320, 20, 2, 4, 8, 8)
        and all(
            w.device == x.device
            and w.dtype == x.dtype
            and w.is_contiguous()
            and w.data_ptr() % 16 == 0
            for w in (down, up)
        )
    )


@triton.jit
def _qwen38_hc_down_silu_inject_kernel(
    x_ptr,
    weight_ptr,
    lora_ptr,
    injection_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RANK_VALUE: tl.constexpr,
    HC_COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
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
        acc += x.to(tl.float32) * weight.to(tl.float32)

    # Preserve the baseline GEMV -> FP16 -> SiLU boundary.
    value = tl.sum(acc, axis=0).to(tl.float16).to(tl.float32)
    is_lora = row < RANK_VALUE
    scaled = value / HC_COUNT
    tl.store(lora_ptr + row, scaled * tl.sigmoid(scaled), mask=is_lora)
    tl.store(injection_ptr + row - RANK_VALUE, value, mask=~is_lora)


@triton.jit
def _qwen38_hc_down_local_shard_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    TP_RANK: tl.constexpr,
):
    """Compute this TP rank's 80 low-rank rows and one injection row."""
    row = tl.program_id(0)
    active = row < 81
    checkpoint_row = tl.where(row < 80, TP_RANK * 80 + row, 320 + TP_RANK)
    offsets = tl.arange(0, 256)
    acc = tl.zeros((256,), dtype=tl.float32)
    for block_start in tl.static_range(0, 10240, 256):
        indices = block_start + offsets
        x = tl.load(
            x_ptr + indices,
            mask=active,
            other=0.0,
            eviction_policy="evict_last",
        )
        weight = tl.load(
            weight_ptr + checkpoint_row * 10240 + indices,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += x.to(tl.float32) * weight.to(tl.float32)

    # Match the replicated projection's FP16 materialization before SiLU.
    value = tl.sum(acc, axis=0).to(tl.float16).to(tl.float32)
    scaled = value / 4
    value = tl.where(row < 80, scaled * tl.sigmoid(scaled), value)
    tl.store(output_ptr + row, value, mask=active)
    # Keep the 88-element communication packet aligned to 16 bytes. Padding
    # is canonical zero and is discarded after the rank-ordered gather.
    tl.store(output_ptr + row, 0.0, mask=~active)


@triton.jit
def _qwen38_hc_up_local_gate_kernel(
    lora_ptr,
    weight_ptr,
    gate_ptr,
    TP_RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute the 2560 gate rows owned by this TP rank."""
    hidden = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, 512)
    hidden_mask = hidden < 2560
    k_mask = offsets < 320
    lora = tl.load(
        lora_ptr + offsets,
        mask=k_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    checkpoint_row = TP_RANK * 2560 + hidden
    weight = tl.load(
        weight_ptr + checkpoint_row[:, None] * 320 + offsets[None, :],
        mask=hidden_mask[:, None] & k_mask[None, :],
        other=0.0,
        eviction_policy="evict_first",
    )
    gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
    # The communication kernel applies the original FP16 gate boundary,
    # sigmoid, rank-ordered FP32 FMA, and final FP16 materialization.
    tl.store(gate_ptr + hidden, gate, mask=hidden_mask)


@triton.jit
def _qwen38_hc_up_hidden_shard_kernel(
    lora_ptr,
    weight_ptr,
    branches_ptr,
    out_ptr,
    TP_RANK: tl.constexpr,
):
    """Mix all four branches locally for two of this rank's 640 hidden rows."""
    rows = tl.arange(0, 8)
    hidden = tl.program_id(0) * 2 + rows // 4
    checkpoint_row = (rows % 4) * 2560 + TP_RANK * 640 + hidden
    offsets = tl.arange(0, 512)
    lora = tl.load(lora_ptr + offsets, offsets < 320, 0).to(tl.float32)
    weight = tl.load(
        weight_ptr + checkpoint_row[:, None] * 320 + offsets[None, :],
        offsets[None, :] < 320,
        0,
    )
    # Keep the existing two-K-warp reduction, FP16 gate boundary, and
    # branch-ordered FP32 FMA. Only row ownership changes; weights are neither
    # repacked nor duplicated, and prefill keeps its original layout.
    gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
    gate = gate.to(tl.float16).to(tl.float32).reshape((2, 4))
    branches = tl.load(branches_ptr + checkpoint_row).to(tl.float32).reshape((2, 4))
    result = tl.full((2,), 0, tl.float32)
    for branch in tl.static_range(4):
        index = tl.full((2, 1), branch, tl.int32)
        g = tl.gather(gate, index, 1).reshape((2,))
        x = tl.gather(branches, index, 1).reshape((2,))
        result = tl.fma(tl.sigmoid(g), x, result)
    tl.store(out_ptr + tl.program_id(0) * 2 + tl.arange(0, 2), result / 4)


@triton.jit
def _qwen38_hc_up_gate_mix_kernel(
    lora_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    K: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    hidden = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    lora = tl.load(
        lora_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    result = 0.0
    for stream in tl.static_range(HC_COUNT):
        row = stream * HC_DIMENSION + hidden
        weight = tl.load(
            weight_ptr + row * K + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        # Preserve the baseline GEMV -> FP16 gate -> sigmoid boundary.
        gate = tl.sum(lora * weight.to(tl.float32), axis=0)
        gate = gate.to(tl.float16).to(tl.float32)
        branch = tl.load(x_ptr + stream * HC_DIMENSION + hidden).to(tl.float32)
        result += tl.sigmoid(gate) * branch
    tl.store(out_ptr + hidden, result / HC_COUNT)


@triton.jit
def _qwen38_hc_up_gate_mix_row4_kernel(
    lora_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    K: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Reuse the low-rank input across four bitwise-equivalent output rows."""
    hidden = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, BLOCK_K)
    hidden_mask = hidden < HC_DIMENSION
    k_mask = offsets < K
    lora = tl.load(
        lora_ptr + offsets,
        mask=k_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    result = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for stream in tl.static_range(HC_COUNT):
        row = stream * HC_DIMENSION + hidden
        weight = tl.load(
            weight_ptr + row[:, None] * K + offsets[None, :],
            mask=hidden_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        # Keep the established FP32 reduction and FP16 gate boundary. Row
        # tiling changes only work assignment and shares the lora read.
        gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
        gate = gate.to(tl.float16).to(tl.float32)
        branch = tl.load(
            x_ptr + stream * HC_DIMENSION + hidden,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        result += tl.sigmoid(gate) * branch
    tl.store(out_ptr + hidden, result / HC_COUNT, mask=hidden_mask)


def _runtime_ok(
    x: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor
) -> bool:
    return bool(
        x.ndim == 2
        and x.shape == (1, _HC_HIDDEN)
        and down_weight.shape == (_HC_RANK + _HC_COUNT + 12, _HC_HIDDEN)
        and up_weight.shape == (_HC_HIDDEN, _HC_RANK)
        and x.dtype == torch.float16
        and down_weight.dtype == torch.float16
        and up_weight.dtype == torch.float16
        and x.is_cuda
        and down_weight.is_cuda
        and up_weight.is_cuda
        and x.is_contiguous()
        and down_weight.is_contiguous()
        and up_weight.is_contiguous()
        and x.device == down_weight.device == up_weight.device
    )


def _qwen38_sm70_fp16_fused_hc(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    packed_down: torch.Tensor | None = None,
    packed_up: torch.Tensor | None = None,
    concurrent_batch: bool = False,
    reassociated_down: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _replicated_runtime_ok(x, packed_down, packed_up, concurrent_batch):
        from vllm import _custom_ops as ops

        if ops.supports_sm70_qwen38_hc_replicated():
            m = x.shape[0]
            partials = x.new_empty((20, m, 352), dtype=torch.float32)
            lora, block, injection = (x.new_empty((m, n)) for n in (320, 2560, 4))
            ops.sm70_qwen38_hc_replicated(
                x, packed_down, packed_up, partials, lora, block, injection
            )
            logger.info_once(
                "SM70 batched HC uses replicated FP32 MMA without TP transport."
            )
            return block, injection
    if _batch_runtime_ok(x, packed_down, packed_up, concurrent_batch):
        from vllm.distributed.parallel_state import get_tp_group

        custom_ar = getattr(get_tp_group().device_communicator, "ca_comm", None)
        if custom_ar is not None and custom_ar.can_sm70_qwen38_hc_batch(x):
            m = x.shape[0]
            partials = x.new_empty((20, m, 96), dtype=torch.float32)
            lora, local_block, block, injection = (
                x.new_empty((m, n)) for n in (320, 640, 2560, 4)
            )
            custom_ar.sm70_qwen38_hc_batch(
                x,
                packed_down,
                packed_up,
                partials,
                lora,
                local_block,
                block,
                injection,
                round_down_partials=not concurrent_batch,
                cooperative=not concurrent_batch and envs.VLLM_SM70_MTP_HC_COOPERATIVE,
                full_unroll=not concurrent_batch and envs.VLLM_SM70_MTP_HC_FULL_UNROLL,
                fused_chain=concurrent_batch,
            )
            logger.info_once(
                "SM70 TP4 batched HC enabled (FP32 concurrent partials=%s).",
                concurrent_batch,
            )
            return block, injection
        # Local MMA does not require a fully-connected IPC topology. For the
        # FP32-partial policy, transport disjoint Half output rows through the
        # TP group's admitted collective (including the verified-edge ring).
        # Each element has one contributor, preserving its Half value without
        # narrowing a sum of live contributions.
        from vllm import _custom_ops as ops

        group = get_tp_group()
        # The local+collective screen regressed M10. Keep it as a reference;
        # automatic non-mesh preparation uses the replicated FP32 route.
        if concurrent_batch and group.world_size == 4 and x.shape[0] == 5:
            if ops.supports_sm70_qwen38_hc_local():
                m = x.shape[0]
                partials = x.new_empty((20, m, 96), dtype=torch.float32)
                local_down = x.new_empty((m, 336))
                ops.sm70_qwen38_hc_down_local(
                    x, packed_down, partials, local_down, group.rank_in_group
                )
                gathered = group.all_reduce(local_down)
                lora = gathered[:, :320].contiguous()
                local_block = x.new_empty((m, 2560))
                ops.sm70_qwen38_hc_up_local(
                    lora, packed_up, x, local_block, group.rank_in_group
                )
                block = group.all_reduce(local_block)
                logger.info_once(
                    "SM70 TP4 batch HC uses local FP32 MMA and the admitted "
                    "TP collective; fully-connected IPC is unavailable."
                )
                return block, gathered[:, 320:324].contiguous()
            logger.warning_once(
                "SM70 TP4 local batch HC unavailable: packaged local operators "
                "missing; retaining dense HC projections."
            )
    if not _runtime_ok(x, down_weight, up_weight):
        # Preserve the ordinary projection and FP16 materialization boundaries
        # for prefill and any unsupported runtime shape. This fallback lives
        # inside the opaque op so a prefill-first dynamic compile cannot bake
        # the M > 1 decision into subsequent decode graphs.
        down_and_injection = torch.nn.functional.linear(x, down_weight)
        lora = torch.ops.vllm.qwen4_exp_hc_silu(
            down_and_injection[..., :_HC_RANK], _HC_COUNT
        )
        injection = down_and_injection[..., _HC_RANK : _HC_RANK + _HC_COUNT]
        gate = torch.nn.functional.linear(lora, up_weight)
        block = torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, _HC_COUNT)
        return block, injection
    try:
        from vllm.distributed.parallel_state import get_tp_group

        device_communicator = get_tp_group().device_communicator
        custom_ar = getattr(device_communicator, "ca_comm", None)
    except (AssertionError, AttributeError, RuntimeError, ValueError):
        custom_ar = None

    if custom_ar is not None and custom_ar.can_sm70_qwen38_hc_shard(x):
        tp_rank = int(custom_ar.rank)
        local_down = x.new_empty((1, 88))
        gathered_down = x.new_empty((1, 336))
        block = x.new_empty((1, _HC_DIM))
        if reassociated_down and Sm70Fp16GemvSiluKernel.can_implement(
            x, down_weight, local_down, 81, 80, tp_rank * 80, 320 + tp_rank, 4.0
        ):
            Sm70Fp16GemvSiluKernel.apply_out(
                x,
                down_weight,
                local_down,
                81,
                80,
                prefix_start=tp_rank * 80,
                suffix_start=320 + tp_rank,
                divisor=4.0,
            )
        else:
            _qwen38_hc_down_local_shard_kernel[(88,)](
                x,
                down_weight,
                local_down,
                TP_RANK=tp_rank,
                num_warps=4,
            )
        custom_ar.sm70_qwen38_hc_down_allgather(local_down, gathered_down)
        if custom_ar.supports_sm70_qwen38_hc_up_mix_allgather():
            custom_ar.sm70_qwen38_hc_up_mix_allgather(
                gathered_down, up_weight, x, block
            )
            logger.info_once(
                "SM70 Qwen3.8 exact TP4 fused FP16 HC up/mix/gather enabled."
            )
            return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]
        if custom_ar.supports_sm70_qwen38_hc_output_allgather():
            local_block = x.new_empty((1, _HC_DIM // _HC_COUNT))
            _qwen38_hc_up_hidden_shard_kernel[(320,)](
                gathered_down,
                up_weight,
                x,
                local_block,
                TP_RANK=tp_rank,
                num_warps=8,
            )
            custom_ar.sm70_qwen38_hc_output_allgather(local_block, block)
            logger.info_once(
                "SM70 Qwen3.8 exact TP4 hidden-sharded FP16 HC route enabled."
            )
            return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]

        # An older wheel/sidecar can still use the established gate-sharded
        # route. Never pass its opaque communicator to a different DSO.
        local_gate = x.new_empty((1, _HC_DIM))
        _qwen38_hc_up_local_gate_kernel[(triton.cdiv(_HC_DIM, 8),)](
            gathered_down,
            up_weight,
            local_gate,
            TP_RANK=tp_rank,
            BLOCK_N=8,
            num_warps=8,
        )
        custom_ar.sm70_qwen38_hc_gate_mix(local_gate, x, block)
        logger.info_once(
            "SM70 Qwen3.8 exact TP4-sharded checkpoint-FP16 HC route enabled."
        )
        return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]

    lora = x.new_empty((1, _HC_RANK))
    injection = x.new_empty((1, _HC_COUNT))
    block = x.new_empty((1, _HC_DIM))
    if reassociated_down:
        projected = x.new_empty((1, _HC_RANK + _HC_COUNT))
        Sm70Fp16GemvSiluKernel.apply_out(
            x,
            down_weight,
            projected,
            _HC_RANK + _HC_COUNT,
            _HC_RANK,
            suffix_start=_HC_RANK,
            divisor=float(_HC_COUNT),
        )
        lora, injection = projected[:, :_HC_RANK], projected[:, _HC_RANK:]
    else:
        _qwen38_hc_down_silu_inject_kernel[(_HC_RANK + _HC_COUNT,)](
            x,
            down_weight,
            lora,
            injection,
            K=_HC_HIDDEN,
            BLOCK_K=256,
            RANK_VALUE=_HC_RANK,
            HC_COUNT=_HC_COUNT,
            num_warps=4,
        )
    _qwen38_hc_up_gate_mix_row4_kernel[(triton.cdiv(_HC_DIM, 4),)](
        lora,
        up_weight,
        x,
        block,
        K=_HC_RANK,
        HC_DIMENSION=_HC_DIM,
        HC_COUNT=_HC_COUNT,
        BLOCK_N=4,
        BLOCK_K=512,
        num_warps=8,
    )
    logger.info_once("SM70 Qwen3.8 fused checkpoint-FP16 HC M=1 route enabled.")
    return block, injection


def _qwen38_sm70_fp16_fused_hc_fake(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    packed_down: torch.Tensor | None = None,
    packed_up: torch.Tensor | None = None,
    concurrent_batch: bool = False,
    reassociated_down: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del down_weight, up_weight, packed_down, packed_up
    return (
        x.new_empty((*x.shape[:-1], _HC_DIM)),
        x.new_empty((*x.shape[:-1], _HC_COUNT)),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_fused_hc",
    op_func=_qwen38_sm70_fp16_fused_hc,
    fake_impl=_qwen38_sm70_fp16_fused_hc_fake,
)


def maybe_apply_qwen38_sm70_fp16_fused_hc(
    down_layer: nn.Module,
    up_layer: nn.Module,
    x: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not enabled or not use_sm70_decode_graph_semantics():
        return None
    # LoRA wrappers and quantized methods must retain their own forward path.
    # Enabling unrelated FP16 projections must never bypass adapter updates.
    if not all(
        isinstance(layer, LinearBase)
        and type(layer.quant_method)
        in (UnquantizedLinearMethod, Qwen38SM70FP16LinearMethod)
        for layer in (down_layer, up_layer)
    ):
        return None
    down_weight = getattr(down_layer, "weight", None)
    up_weight = getattr(up_layer, "weight", None)
    if down_weight is None or up_weight is None:
        return None
    if down_weight.dtype != torch.float16 or up_weight.dtype != torch.float16:
        return None
    if down_weight.shape != (
        _HC_RANK + _HC_COUNT + 12,
        _HC_HIDDEN,
    ) or up_weight.shape != (_HC_HIDDEN, _HC_RANK):
        return None
    return torch.ops.vllm.qwen38_sm70_fp16_fused_hc(
        x,
        down_weight,
        up_weight,
        getattr(down_layer, "_sm70_qwen38_hc_batch_packed", None),
        getattr(up_layer, "_sm70_qwen38_hc_batch_packed", None),
        getattr(down_layer, "_sm70_qwen38_hc_batch_concurrent", False),
        getattr(down_layer, "_sm70_fp16_gemv_silu_ranges", False),
    )


def enable_qwen38_sm70_fp16_fused_hc(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Mark exact base-model HC modules for the fused M=1 route."""
    if (
        not envs.VLLM_SM70_QWEN38_FUSED_HC_FP16
        or envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
        or dtype != torch.float16
        or not current_platform.is_device_capability((7, 0))
        or not _exact_runtime_contract(vllm_config)
    ):
        return

    enabled_count = 0
    tp4 = (
        vllm_config or get_current_vllm_config()
    ).parallel_config.tensor_parallel_size == 4
    for child in module.modules():
        if not (
            getattr(child, "use_combine", False)
            and getattr(child, "lora_rank", None) == _HC_RANK
            and getattr(child, "hc_count", None) == _HC_COUNT
            and getattr(child, "hidden_size", None) == _HC_DIM
            and hasattr(child, "input_mix_weight_down_block_inject")
            and hasattr(child, "input_mix_weight_up")
        ):
            continue
        child._sm70_qwen38_fp16_fused_hc = True
        # The generic operator changes FP32 reduction association. Enable it
        # for the quality-equivalent no-speculative policy; verifier/draft and
        # batch-invariant execution retain their existing reductions.
        config = vllm_config or get_current_vllm_config()
        child.input_mix_weight_down_block_inject._sm70_fp16_gemv_silu_ranges = (
            config.speculative_config is None and not envs.VLLM_BATCH_INVARIANT
        )
        fp32_partials = (
            not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        )
        concurrent_batch = bool(
            envs.VLLM_SM70_QWEN38_BATCH_FASTPATH
            and _batch_runtime_contract(vllm_config)
            and tp4
            # The default worker disables FP16 partial reductions. Reuse the
            # batch operator's FP32-partial schedule for that MTP policy rather
            # than falling back to two full, replicated projections. The
            # legacy FP16-partial policy retains its own qualified schedule.
            and (
                not _mtp_batch_runtime_contract(vllm_config)
                or (envs.VLLM_SM70_MTP_HC_BATCH and fp32_partials)
            )
        )
        # Only HC's packed collective owns exactly four TP shards. Local router
        # and shared-expert kernels use the independent MTP admission above.
        if tp4 and (
            concurrent_batch
            or (
                envs.VLLM_SM70_MTP_HC_BATCH and _mtp_batch_runtime_contract(vllm_config)
            )
        ):
            for role, layer in (
                ("down", child.input_mix_weight_down_block_inject),
                ("up", child.input_mix_weight_up),
            ):
                if type(layer.quant_method) is UnquantizedLinearMethod:
                    layer.quant_method = Qwen38SM70FP16LinearMethod()
                if not isinstance(layer.quant_method, Qwen38SM70FP16LinearMethod):
                    raise RuntimeError("Batched HC requires checkpoint-FP16 linears")
                layer._sm70_qwen38_hc_batch_role = role
                layer._sm70_qwen38_hc_batch_concurrent = concurrent_batch
        enabled_count += 1

    if enabled_count:
        logger.info_once(
            "Prepared %d Qwen3.8 SM70 fused checkpoint-FP16 HC modules.",
            enabled_count,
        )


__all__ = [
    "_qwen38_hc_down_local_shard_kernel",
    "_qwen38_hc_up_local_gate_kernel",
    "enable_qwen38_sm70_fp16_fused_hc",
    "maybe_apply_qwen38_sm70_fp16_fused_hc",
]
