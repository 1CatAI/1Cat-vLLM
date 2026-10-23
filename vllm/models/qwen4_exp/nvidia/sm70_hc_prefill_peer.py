# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded TP4 peer reduction fused with row-owned HC residual/RMSNorm.

Existing CUDA IPC allocation and NCCL ordering are reused. Full-mesh SM70
prefill rows read the four physical TP peers and accumulate in FP32. MoE
local sums use scaled FP16 only if every element's reconstruction error is
at most 2**-14; other rows retain FP32 sums. The four-rank transport error
before the original FP16 boundary is therefore at most 2**-12, apart from
FP32 addition association. Attention partials retain their original FP16
storage. The normal sharded HC stages handle all unqualified inputs.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.runtime_resources import current_runtime_resources, runtime_resources_for
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_MAX_ROWS = 8192
_PARTIAL_ABS_LIMIT = 2**-14


@triton.jit
def _prepare_row(
    a,
    b,
    ptr,
    HALF_BYTES: tl.constexpr,
    FLAG_BYTES: tl.constexpr,
    LIMIT: tl.constexpr,
    SUM2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 4)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    offset = row * 2560 + col
    source = tl.load(a + offset, col < 2560, 0).to(tl.float32)
    if SUM2:
        source += tl.load(b + offset, col < 2560, 0).to(tl.float32)
        scaled = (source * 0.125).to(tl.float16)
    else:
        scaled = source.to(tl.float16)
    error = tl.abs(source - scaled.to(tl.float32) * (8 if SUM2 else 1))
    bad = (error > LIMIT) | (source != source) | (tl.abs(source) == float("inf"))
    unsafe = tl.sum(tl.sum(bad.to(tl.int32), 1), 0) > 0
    base = ptr.to(tl.pointer_type(tl.uint8))
    flags = (base + FLAG_BYTES).to(tl.pointer_type(tl.int32))
    tl.store(flags + row, unsafe.to(tl.int32))
    if unsafe:
        fallback = (base + HALF_BYTES).to(tl.pointer_type(tl.float32))
        tl.store(fallback + offset, source, col < 2560)
    else:
        tl.store(ptr.to(tl.pointer_type(tl.float16)) + offset, scaled, col < 2560)


@triton.jit
def _read_partial(
    ptr,
    global_row,
    col,
    HALF_BYTES: tl.constexpr,
    FLAG_BYTES: tl.constexpr,
    SUM2: tl.constexpr,
):
    base = ptr.to(tl.pointer_type(tl.uint8))
    unsafe = tl.load((base + FLAG_BYTES).to(tl.pointer_type(tl.int32)) + global_row)
    offset = global_row * 2560 + col
    if unsafe:
        value = tl.load(
            (base + HALF_BYTES).to(tl.pointer_type(tl.float32)) + offset, col < 2560, 0
        )
    else:
        value = tl.load(ptr.to(tl.pointer_type(tl.float16)) + offset, col < 2560, 0).to(
            tl.float32
        ) * (8 if SUM2 else 1)
    return value


@triton.jit(do_not_specialize=["LOCAL"])
def _peer_combine_norm(
    p0,
    p1,
    p2,
    p3,
    hidden,
    inj,
    weight,
    updated,
    normalized,
    RANK: tl.constexpr,
    LOCAL,
    HALF_BYTES: tl.constexpr,
    FLAG_BYTES: tl.constexpr,
    SUM2: tl.constexpr,
    NORMALIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 4)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    global_row = RANK * LOCAL + row
    partial = _read_partial(p0, global_row, col, HALF_BYTES, FLAG_BYTES, SUM2)
    partial += _read_partial(p1, global_row, col, HALF_BYTES, FLAG_BYTES, SUM2)
    partial += _read_partial(p2, global_row, col, HALF_BYTES, FLAG_BYTES, SUM2)
    partial += _read_partial(p3, global_row, col, HALF_BYTES, FLAG_BYTES, SUM2)
    block = partial.to(tl.float16).to(tl.float32)
    for stream in tl.static_range(4):
        position = row * 10240 + stream * 2560 + col
        residual = tl.load(hidden + position, col < 2560, 0).to(tl.float32)
        factor = 2 * tl.sigmoid(tl.load(inj + row * 4 + stream).to(tl.float32) / 4)
        value = (residual + factor * block).to(tl.float16)
        tl.store(updated + position, value, col < 2560)
        if NORMALIZE:
            fp32 = value.to(tl.float32)
            variance = tl.sum(tl.sum(fp32 * fp32, 1), 0) / 2560
            norm = fp32 * tl.rsqrt(variance + EPS)
            w = tl.load(weight + stream * 2560 + col, col < 2560, 0).to(tl.float32)
            norm += norm * w
            tl.store(normalized + position, norm, col < 2560)


class HcPrefillPeerRuntime:
    """One engine's reusable IPC banks; NCCL fences every reader/writer turn."""

    def __init__(self, group, device: torch.device, max_rows: int):
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.group = group
        self.device = device
        self.rank = group.rank_in_group
        self.max_rows = max_rows
        self.half_bytes = max_rows * 2560 * 2
        self.flag_bytes = max_rows * 2560 * 6
        self.total_bytes = self.flag_bytes + max_rows * 4
        self.nccl = group.device_communicator.pynccl_comm
        self.pointers = CustomAllreduce.create_shared_buffer(
            self.total_bytes, group.cpu_group
        )
        self.barrier_in = torch.zeros(1, dtype=torch.int32, device=device)
        self.barrier_out = torch.empty_like(self.barrier_in)

    def can_use(self, hidden, first, second, injection, weight) -> bool:
        return bool(
            self.pointers
            and 4096 <= first.shape[0] <= self.max_rows
            and first.shape == (4 * hidden.shape[0], 2560)
            and hidden.shape[1] == 10240
            and injection.shape == (hidden.shape[0], 4)
            and weight.numel() == 10240
            and all(
                t.dtype == torch.float16
                and t.device == self.device
                and t.is_contiguous()
                for t in (hidden, first, injection, weight)
            )
            and (
                second is None
                or (
                    second.shape == first.shape
                    and second.dtype == first.dtype
                    and second.device == first.device
                    and second.is_contiguous()
                )
            )
        )

    def combine_norm(self, hidden, first, second, injection, weight, eps, normalize):
        sum2 = second is not None
        rows = first.shape[0]
        updated = torch.empty_like(hidden)
        normalized = torch.empty_like(hidden) if normalize else hidden.new_empty((0,))
        _prepare_row[(rows,)](
            first,
            first if second is None else second,
            self.pointers[self.rank],
            self.half_bytes,
            self.flag_bytes,
            _PARTIAL_ABS_LIMIT,
            sum2,
            1024,
            num_warps=8,
        )
        self.nccl.all_reduce(self.barrier_in, self.barrier_out)
        _peer_combine_norm[(hidden.shape[0],)](
            *self.pointers,
            hidden,
            injection,
            weight,
            updated,
            normalized,
            self.rank,
            hidden.shape[0],
            self.half_bytes,
            self.flag_bytes,
            sum2,
            normalize,
            eps,
            1024,
            num_warps=8,
        )
        # The next module may reuse this bank only after all peer reads finish.
        self.nccl.all_reduce(self.barrier_in, self.barrier_out)
        return updated, normalized

    def close(self) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        if not self.pointers:
            return
        torch.accelerator.synchronize(self.device)
        pointers, self.pointers = self.pointers, []
        for rank, pointer in enumerate(pointers):
            if rank != self.rank:
                torch.ops._C.sm70_ring_close_mem_handle(pointer)
        CustomAllreduce.free_shared_buffer(pointers, rank=self.rank)


def prepare_hc_prefill_peer_runtime(vllm_config, device) -> None:
    """Bind after model qualification; keep missing transport on the old stages."""
    policy = vllm_config.kernel_config.layer_execution
    if not (policy.hc_prefill_sequence_parallel and policy.hc_prefill_peer_norm):
        return
    resources = runtime_resources_for(vllm_config)
    if "hc_prefill_peer" in resources:
        return
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    communicator = group.device_communicator
    ca = getattr(communicator, "ca_comm", None)
    nccl = getattr(communicator, "pynccl_comm", None)
    max_rows = min(_MAX_ROWS, vllm_config.scheduler_config.max_num_batched_tokens)
    reason = (
        "requires_tp4"
        if group.world_size != 4
        else "requires_full_nvlink_mesh"
        if ca is None or not getattr(ca, "fully_connected", False)
        else "nccl_unavailable"
        if nccl is None or nccl.disabled
        else "ipc_close_operator_missing"
        if not hasattr(torch.ops._C, "sm70_ring_close_mem_handle")
        else "query_capacity_below_4096"
        if max_rows < 4096
        else None
    )
    reasons = [None] * group.world_size
    dist.all_gather_object(reasons, reason, group=group.cpu_group)
    rejected = next((value for value in reasons if value is not None), None)
    status = {
        "enabled": rejected is None,
        "reason": rejected,
        "max_query_rows": max_rows,
    }
    vllm_config.kernel_config.collective_kernel_selections["hc_prefill_peer"] = status
    if rejected is not None:
        logger.info("HC prefill peer reduction retains original stages: %s", rejected)
        return
    resources["hc_prefill_peer"] = HcPrefillPeerRuntime(group, device, max_rows)
    logger.info("HC prefill TP4 peer reduction enabled: %s", status)


def _combine_norm(
    hidden: torch.Tensor,
    payload: torch.Tensor,
    injection: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    moe_payload: bool,
    partial: bool,
    normalize: bool,
    second: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed import get_tp_group

    from .ops.hc import hc_combine, hc_combine_norm

    group = get_tp_group()
    rows = payload.shape[0] // (2 if moe_payload else 1)
    first = payload[:rows]
    if moe_payload:
        if second is not None:
            raise ValueError("HC payload cannot be both packed and separate")
        second = payload[rows:]
    if second is not None and second.shape != first.shape:
        raise ValueError("HC partial contributions must have matching shapes")
    padding = 4 * hidden.shape[0] - rows
    if not 0 <= padding <= 3:
        raise ValueError("HC sequence payload must match the padded row ownership")
    if padding:
        first = torch.nn.functional.pad(first, (0, 0, 0, padding))
        if second is not None:
            second = torch.nn.functional.pad(second, (0, 0, 0, padding))
    resources = current_runtime_resources()
    owner = resources.get("hc_prefill_peer") if resources is not None else None
    if (
        partial
        and owner is not None
        and owner.group is group
        and owner.can_use(hidden, first, second, injection, weight)
    ):
        return owner.combine_norm(
            hidden, first, second, injection, weight, eps, normalize
        )
    if partial:
        # Attention can supply one partial, while MoE supplies two. Both
        # use the qualified FP32 transport before the original FP16 combine.
        source = first.float() if second is None else first.float() + second.float()
        reduced = group.reduce_scatter(source, dim=0).to(first.dtype)
    else:
        reduced = first.reshape(4, -1, first.shape[-1])[group.rank_in_group]
    if normalize:
        return hc_combine_norm(hidden, reduced, injection, weight, eps, 4)
    return hc_combine(hidden, reduced, injection, 4), hidden.new_empty((0,))


def _combine_norm_fake(
    hidden,
    payload,
    injection,
    weight,
    eps,
    moe_payload,
    partial,
    normalize,
    second=None,
):
    return torch.empty_like(hidden), (
        torch.empty_like(hidden) if normalize else hidden.new_empty((0,))
    )


direct_register_custom_op(
    "qwen38_hc_sequence_combine_norm",
    _combine_norm,
    fake_impl=_combine_norm_fake,
)
hc_sequence_combine_norm = torch.ops.vllm.qwen38_hc_sequence_combine_norm


def _shard_rows(value: torch.Tensor) -> torch.Tensor:
    """Pad only the HC carrier, then return an owned physical-rank row span."""
    from vllm.distributed import get_tp_group

    rows = (value.shape[0] + 3) // 4
    padding = 4 * rows - value.shape[0]
    if padding:
        value = torch.nn.functional.pad(value, (0, 0, 0, padding))
    return value.reshape(4, rows, value.shape[-1])[get_tp_group().rank_in_group].clone()


def _shard_rows_fake(value: torch.Tensor) -> torch.Tensor:
    return value.new_empty(((value.shape[0] + 3) // 4, value.shape[-1]))


direct_register_custom_op(
    "qwen38_hc_sequence_shard_rows",
    _shard_rows,
    fake_impl=_shard_rows_fake,
)
hc_sequence_shard_rows = torch.ops.vllm.qwen38_hc_sequence_shard_rows


def _gather_rows_into(output: torch.Tensor, local: torch.Tensor, tokens: int) -> None:
    """Commit HC rows into their existing owner without a full temporary.

    Only complete, contiguous TP4 receive spans use the existing PyNCCL
    transport directly. A partial final rank preserves the owner's unused
    capacity (at most three rows); insufficient padded capacity or unavailable
    transport keeps the original gather/copy fallback.
    """
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if (
        group.world_size != 4
        or output.ndim != 2
        or local.ndim != 2
        or output.shape[1] != local.shape[1]
        or output.dtype != local.dtype
        or output.device != local.device
        or not 0 <= tokens <= output.shape[0]
        or local.shape[0] != (tokens + 3) // 4
    ):
        raise ValueError("HC row commit requires matching TP4 rows and capacity")
    if tokens == 0:
        return
    padded = 4 * local.shape[0]
    communicator = getattr(group.device_communicator, "pynccl_comm", None)
    if (
        communicator is None
        or communicator.disabled
        or output.shape[0] < padded
        or not output.is_contiguous()
        or not local.is_contiguous()
    ):
        output[:tokens].copy_(group.all_gather(local, dim=0)[:tokens])
        return
    tail = output[tokens:padded].clone() if tokens < padded else None
    communicator.all_gather(output[:padded], local)
    if tail is not None:
        output[tokens:padded].copy_(tail)


def _gather_rows_into_fake(
    output: torch.Tensor, local: torch.Tensor, tokens: int
) -> None:
    return


direct_register_custom_op(
    "qwen38_hc_sequence_gather_into",
    _gather_rows_into,
    mutates_args=["output"],
    fake_impl=_gather_rows_into_fake,
)
hc_sequence_gather_into = torch.ops.vllm.qwen38_hc_sequence_gather_into
