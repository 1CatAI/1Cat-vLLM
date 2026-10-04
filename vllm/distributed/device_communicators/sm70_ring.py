# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small collectives using only verified direct NVLink peers."""

import itertools
from typing import Any

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.config.kernel import Sm70RingConfig
from vllm.platforms import current_platform


def find_peer_order(direct: list[list[bool]]) -> tuple[int, ...] | None:
    if len(direct) != 4 or any(len(row) != 4 for row in direct):
        return None
    if all(direct[i][j] for i in range(4) for j in range(4) if i != j):
        return None  # Preserve the existing fully connected transport.
    for tail in itertools.permutations(range(1, 4)):
        order = (0, *tail)
        if all(direct[order[i]][order[i ^ bit]] for i in range(4) for bit in (1, 2)):
            return order
    return None


class Sm70RingCommunicator:
    def __init__(self, group, device, name: str, requested: bool):
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.pointers: list[int] | None = None
        self.status: dict[str, Any] = {
            "enabled": False,
            "reason": None,
            "scope": "collective_capability",
        }
        cfg = get_current_vllm_config_or_none()
        self.policy = cfg.kernel_config.sm70_ring if cfg else Sm70RingConfig()
        reason = None
        if not requested or not self.policy.enabled:
            reason = "disabled_by_configuration"
        elif envs.VLLM_BATCH_INVARIANT:
            reason = "batch_invariant_requires_existing_reduction_order"
        elif dist.get_world_size(group) != 4:
            reason = "requires_four_ranks"
        elif (
            not current_platform.is_cuda()
            or not current_platform.is_device_capability(70)
        ):
            reason = "requires_sm70_cuda"
        else:
            missing = [
                op
                for op in (
                    "sm70_ring_atomic_allreduce_out",
                    "sm70_ring_native_peer_atomics",
                    "sm70_ring_close_mem_handle",
                )
                if not hasattr(torch.ops._C, op)
            ]
            if missing:
                reason = "operator_missing:" + ",".join(missing)
        if dist.get_world_size(group) == 4:
            reasons = [None] * 4
            dist.all_gather_object(reasons, reason, group=group)
            reason = next((r for r in reasons if r is not None), None)
        if reason is None:
            reason = self._prepare()
        self.status.update(
            enabled=reason is None, reason=reason, max_bytes=self.policy.max_bytes
        )
        if cfg:
            cfg.kernel_config.collective_kernel_selections[name] = self.status

    def _prepare(self):
        import pynvml

        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        local = self.device.index
        if local is None:
            local = torch.accelerator.current_device_index()
        self.device_index = local
        self.device = torch.device("cuda", local)
        uuid = current_platform.get_device_uuid(local)
        uuids = [""] * 4
        dist.all_gather_object(uuids, uuid, group=self.group)
        visible = {
            current_platform.get_device_uuid(i): i
            for i in range(torch.accelerator.device_count())
        }
        local_visibility = all(u in visible for u in uuids)
        visibility = [False] * 4
        dist.all_gather_object(visibility, local_visibility, group=self.group)
        if not all(visibility):
            return "requires_local_visible_peer_group"
        pynvml.nvmlInit()
        try:
            handles = [pynvml.nvmlDeviceGetHandleByUUID(u) for u in uuids]
            direct = [
                [
                    i == j
                    or pynvml.nvmlDeviceGetP2PStatus(
                        handles[i], handles[j], pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    == pynvml.NVML_P2P_STATUS_OK
                    for j in range(4)
                ]
                for i in range(4)
            ]
        finally:
            pynvml.nvmlShutdown()
        order = find_peer_order(direct)
        if order is None:
            return "requires_direct_nvlink_ring_without_full_mesh"
        logical = order.index(self.rank)
        neighbors = {order[logical ^ 1], order[logical ^ 2]}
        # CUDA device indices are local to each worker; exchange physical UUIDs
        # and match them against the worker's visible CUDA devices.
        supported = True
        for peer in neighbors:
            dst = visible.get(uuids[peer])
            if dst is None:
                supported = False
                break
            supported &= torch.ops._C.sm70_ring_native_peer_atomics(local, dst)
        admitted = [None] * 4
        dist.all_gather_object(admitted, supported, group=self.group)
        if not all(admitted):
            return "native_peer_atomics_unavailable"
        self.order = order
        self.capacity = (self.policy.max_bytes + 3) // 4
        with torch.accelerator.device_index(local):
            self.pointers = CustomAllreduce.create_shared_buffer(
                2 * 2 * self.capacity * 8,
                group=self.group,
                peer_ranks=neighbors | {self.rank},
            )
            self.addresses = torch.tensor(
                self.pointers, dtype=torch.int64, device=self.device
            )
            self.counters = torch.zeros(
                self.capacity, dtype=torch.int32, device=self.device
            )
        self.status.update(
            rank_order=list(order), direct_nvlink=direct, dtype="float16"
        )
        return None

    def rejection_reason(self, tensor):
        if not self.status["enabled"]:
            return self.status["reason"]
        if tensor.dtype != torch.float16:
            return "fp16_input_required_for_exact_packet_encoding"
        if tensor.device != self.device or not tensor.is_contiguous():
            return "requires_contiguous_local_cuda_input"
        if not 0 < tensor.numel() * tensor.element_size() <= self.policy.max_bytes:
            return "payload_outside_calibrated_byte_range"
        return None

    def all_reduce(self, tensor):
        reason = self.rejection_reason(tensor)
        self.status["last_rejection_reason"] = reason
        if reason is not None:
            return None
        output = torch.empty_like(tensor)
        torch.ops._C.sm70_ring_atomic_allreduce_out(
            output,
            tensor,
            self.addresses,
            self.counters,
            list(self.order),
            self.rank,
            self.capacity,
            False,
        )
        return output

    def close(self):
        if self.pointers is not None:
            with torch.accelerator.device_index(self.device_index):
                torch.accelerator.synchronize()
                dist.barrier(group=self.group)
                for rank, pointer in enumerate(self.pointers):
                    if rank != self.rank and pointer:
                        torch.ops._C.sm70_ring_close_mem_handle(pointer)
                dist.barrier(group=self.group)
                ops.free_shared_buffer(self.pointers[self.rank])
            self.pointers = None
