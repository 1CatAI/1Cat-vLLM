# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable LL HC buffers using two verified direct NVLink peers."""

from typing import Any

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.platforms import current_platform

from .custom_all_reduce import CustomAllreduce
from .sm70_ring import find_peer_order


class Sm70HcLLCommunicator:
    def __init__(self, group, device, name):
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.pointers: list[int] | None = None
        self.status: dict[str, Any] = {
            "enabled": False,
            "reason": None,
            "scope": "hc_operator_capability",
        }
        cfg = get_current_vllm_config_or_none()
        reason = None
        if cfg is None or not cfg.kernel_config.hc_ll_shard:
            reason = "disabled_by_kernel_config"
        elif dist.get_world_size(group) != 4:
            reason = "requires_tp4"
        elif (
            not current_platform.is_cuda()
            or not current_platform.is_device_capability(70)
        ):
            reason = "requires_sm70_cuda"
        elif not all(
            hasattr(torch.ops._C, op)
            for op in (
                "sm70_hc_ll_down_out",
                "sm70_hc_ll_up_out",
                "sm70_ring_native_peer_atomics",
            )
        ):
            reason = "packaged_hc_ll_operator_missing"
        if dist.get_world_size(group) == 4:
            reasons = [None] * 4
            dist.all_gather_object(reasons, reason, group=group)
            reason = next((r for r in reasons if r is not None), None)
        if reason is None:
            reason = self._prepare()
        self.status.update(enabled=reason is None, reason=reason, min_m=1, max_m=20)
        if cfg:
            cfg.kernel_config.collective_kernel_selections[
                "hc_ll:" + (name or "<unnamed>")
            ] = self.status

    def _prepare(self):
        import pynvml

        local = self.device.index
        if local is None:
            local = torch.accelerator.current_device_index()
        self.device = torch.device("cuda", local)
        self.device_index = local
        uuids = [""] * 4
        dist.all_gather_object(
            uuids, current_platform.get_device_uuid(local), group=self.group
        )
        visible = {
            current_platform.get_device_uuid(i): i
            for i in range(torch.accelerator.device_count())
        }
        available = [False] * 4
        dist.all_gather_object(
            available, all(u in visible for u in uuids), group=self.group
        )
        if not all(available):
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
        order = (
            (0, 1, 2, 3)
            if all(direct[i][j] for i in range(4) for j in range(4))
            else find_peer_order(direct)
        )
        if order is None:
            return "requires_two_direct_nvlink_peers_per_rank"
        self.logical_rank = order.index(self.rank)
        neighbors = {order[self.logical_rank ^ 1], order[self.logical_rank ^ 2]}
        supported = all(
            torch.ops._C.sm70_ring_native_peer_atomics(local, visible[uuids[p]])
            for p in neighbors
        )
        admitted = [False] * 4
        dist.all_gather_object(admitted, supported, group=self.group)
        if not all(admitted):
            return "direct_cuda_peer_access_unavailable"
        self.order = order
        self.down_bytes = 2 * 24 * 336 * 4
        self.up_bytes = 2 * 24 * 2560 * 4
        with torch.accelerator.device_index(local):
            self.pointers = CustomAllreduce.create_shared_buffer(
                self.down_bytes + self.up_bytes,
                group=self.group,
                peer_ranks=neighbors | {self.rank},
            )
            self.down_pointers = [self.pointers[r] for r in order]
            self.up_pointers = [
                p + self.down_bytes if p else 0 for p in self.down_pointers
            ]
            self.partial = torch.empty(
                40 * 24 * 96, device=self.device, dtype=torch.float32
            )
            self.down_counter = torch.zeros(10, device=self.device, dtype=torch.int32)
            self.up_counter = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.down_seq = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.up_seq = torch.zeros(1, device=self.device, dtype=torch.int32)
            torch.accelerator.synchronize()
        dist.barrier(group=self.group)
        self.status.update(rank_order=list(order), direct_nvlink=direct)
        return None

    def apply(self, x, down, up):
        if not self.status["enabled"] or not 1 <= x.shape[0] <= 20:
            return None
        m = x.shape[0]
        lora = x.new_empty((m, 320))
        output = x.new_empty((m, 2560))
        injection = x.new_empty((m, 4))
        torch.ops._C.sm70_hc_ll_down_out(
            x,
            down,
            self.partial,
            self.down_counter,
            self.down_pointers,
            self.down_seq,
            self.logical_rank,
            1,
        )
        torch.ops._C.sm70_hc_ll_up_out(
            self.down_pointers[self.logical_rank],
            up,
            x,
            self.up_counter,
            self.up_pointers,
            self.up_seq,
            self.down_seq,
            self.logical_rank,
            output,
            lora,
            injection,
            5,
        )
        return output, injection

    def close(self):
        if self.pointers is None:
            return
        with torch.accelerator.device_index(self.device_index):
            torch.accelerator.synchronize()
            dist.barrier(group=self.group)
            for rank, pointer in enumerate(self.pointers):
                if rank != self.rank and pointer:
                    torch.ops._C.sm70_ring_close_mem_handle(pointer)
            dist.barrier(group=self.group)
            ops.free_shared_buffer(self.pointers[self.rank])
        self.pointers = None
