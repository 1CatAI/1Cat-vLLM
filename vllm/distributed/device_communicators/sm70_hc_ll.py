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
        self.optimized_loads = bool(cfg and cfg.kernel_config.hc_ll_optimized_loads)
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
        self.status.update(
            enabled=reason is None,
            reason=reason,
            min_m=1,
            max_m=64,
            optimized_loads=reason is None and self.optimized_loads,
            load_optimization_reason=(
                reason
                if reason is not None
                else (None if self.optimized_loads else "disabled_by_kernel_config")
            ),
        )
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
        full_mesh = all(direct[i][j] for i in range(4) for j in range(4))
        order = (0, 1, 2, 3) if full_mesh else find_peer_order(direct)
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
        p3 = order[self.logical_rank ^ 3]
        direct_p3 = full_mesh and bool(
            torch.ops._C.sm70_ring_native_peer_atomics(local, visible[uuids[p3]])
        )
        admitted_p3 = [False] * 4
        dist.all_gather_object(admitted_p3, direct_p3, group=self.group)
        self.full_mesh = all(admitted_p3)
        if self.full_mesh:
            neighbors.add(p3)
        self.order = order
        self.phys_order = [int(r) for r in order]
        self.down_bytes = 2 * 64 * 336 * 4
        self.up_bytes = 2 * 64 * 2560 * 4
        self.ple_bytes = 2 * 64 * 12800 * 4
        self.fc_bytes = 2 * 64 * 10240 * 4
        self.ar_bytes = (2 * 4 * 64 * 2560 * 4) if self.full_mesh else 0
        total_bytes = (
            self.down_bytes
            + self.up_bytes
            + self.ple_bytes
            + self.fc_bytes
            + self.ar_bytes
        )
        ple_offset = self.down_bytes + self.up_bytes
        fc_offset = ple_offset + self.ple_bytes
        ar_offset = fc_offset + self.fc_bytes
        with torch.accelerator.device_index(local):
            self.pointers = CustomAllreduce.create_shared_buffer(
                total_bytes,
                group=self.group,
                peer_ranks=neighbors | {self.rank},
            )
            self.down_pointers = [self.pointers[r] for r in order]
            self.up_pointers = [
                p + self.down_bytes if p else 0 for p in self.down_pointers
            ]
            self.ple_pointers = [p + ple_offset if p else 0 for p in self.down_pointers]
            self.fc_pointers = [p + fc_offset if p else 0 for p in self.down_pointers]
            self.ar_pointers = (
                [p + ar_offset if p else 0 for p in self.down_pointers]
                if self.full_mesh
                else []
            )
            self.partial = torch.empty(
                40 * 64 * 96, device=self.device, dtype=torch.float32
            )
            self.down_counter = torch.zeros(16, device=self.device, dtype=torch.int32)
            self.up_counter = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.down_seq = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.up_seq = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.ple_counter = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.ple_dirty_m = torch.zeros(2, device=self.device, dtype=torch.int32)
            self.ple_seq = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.fc_counter = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.fc_dirty_m = torch.zeros(2, device=self.device, dtype=torch.int32)
            self.fc_seq = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.ar_counter = torch.zeros(1, device=self.device, dtype=torch.int32)
            self.ar_dirty_m = torch.zeros(2, device=self.device, dtype=torch.int32)
            self.ar_seq = torch.zeros(1, device=self.device, dtype=torch.int32)
            self._empty_lora = torch.empty(0, device=self.device, dtype=torch.float16)
            self._empty_scale = torch.empty(0, device=self.device, dtype=torch.float32)
            torch.accelerator.synchronize()
        dist.barrier(group=self.group)
        self.status.update(
            rank_order=list(order), direct_nvlink=direct, full_mesh=self.full_mesh
        )
        return None

    def apply(self, x, down, up):
        if not self.status["enabled"] or not 1 <= x.shape[0] <= 64:
            return None
        m = x.shape[0]
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
            self.optimized_loads,
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
            self._empty_lora,
            injection,
            5,
            self.optimized_loads,
        )
        return output, injection

    def apply_ple(
        self,
        embeddings: torch.Tensor,
        scale: torch.Tensor | None,
        packed_kv: torch.Tensor,
        hidden_states: torch.Tensor,
        norm_key_w: torch.Tensor,
        norm_query_w: torch.Tensor,
        norm_conv_w: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        m = embeddings.shape[0]
        if (
            not self.status["enabled"]
            or not 1 <= m <= 64
            or not hasattr(torch.ops._C, "qwen38_ple_kv_proj_tp4_sm70_out")
            or not hasattr(torch.ops._C, "qwen38_ple_gate_norm_sm70_out")
        ):
            return None
        gated_value = hidden_states.new_empty((m, 10240))
        normalized = hidden_states.new_empty((m, 10240))
        torch.ops._C.qwen38_ple_kv_proj_tp4_sm70_out(
            embeddings,
            scale if scale is not None else self._empty_scale,
            packed_kv,
            self.ple_counter,
            self.ple_dirty_m,
            self.ple_pointers,
            self.ple_seq,
            self.logical_rank,
        )
        torch.ops._C.qwen38_ple_gate_norm_sm70_out(
            gated_value,
            normalized,
            hidden_states,
            self._empty_lora,
            self._empty_lora,
            self.ple_pointers[self.logical_rank],
            self.ple_seq,
            norm_key_w,
            norm_query_w,
            norm_conv_w,
            eps,
        )
        return gated_value, normalized

    @property
    def can_add_all_gather_fc(self) -> bool:
        return bool(
            self.status["enabled"]
            and hasattr(torch.ops._C, "sm70_hc_ll_add_all_gather_fc_out")
        )

    def add_all_gather_fc(
        self, el: torch.Tensor, hl: torch.Tensor
    ) -> torch.Tensor | None:
        m = el.shape[0]
        if (
            not self.can_add_all_gather_fc
            or not 1 <= m <= 64
            or not el.is_contiguous()
            or not hl.is_contiguous()
            or el.shape != (m, 640)
            or hl.shape != (m, 4, 640)
        ):
            return None
        out = el.new_empty((m, 10240))
        torch.ops._C.sm70_hc_ll_add_all_gather_fc_out(
            out,
            el,
            hl,
            self.fc_counter,
            self.fc_dirty_m,
            self.fc_pointers,
            self.fc_seq,
            self.logical_rank,
            self.phys_order,
        )
        return out

    @property
    def can_allreduce(self) -> bool:
        return bool(
            self.status["enabled"]
            and getattr(self, "full_mesh", False)
            and hasattr(torch.ops._C, "sm70_hc_ll_allreduce_combine_norm_out")
        )

    def allreduce_combine_norm(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor | None,
        res: torch.Tensor,
        inj: torch.Tensor,
        nw: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        m = p0.shape[0]
        if (
            not self.can_allreduce
            or not 1 <= m <= 64
            or p0.shape != (m, 2560)
            or res.shape != (m, 10240)
            or inj.shape != (m, 4)
            or not p0.is_contiguous()
            or not res.is_contiguous()
            or not inj.is_contiguous()
            or p0.data_ptr() % 16 != 0
            or res.data_ptr() % 16 != 0
            or (
                p1 is not None
                and (
                    p1.shape != (m, 2560)
                    or not p1.is_contiguous()
                    or p1.data_ptr() % 16 != 0
                )
            )
        ):
            return None
        res_out = res.new_empty((m, 10240))
        norm_out = res.new_empty((m, 10240))
        torch.ops._C.sm70_hc_ll_allreduce_combine_norm_out(
            res_out,
            norm_out,
            p0,
            p1 if p1 is not None else self._empty_lora,
            res,
            inj,
            nw,
            float(eps),
            self.ar_counter,
            self.ar_dirty_m,
            self.ar_seq,
            self.ar_pointers,
            self.logical_rank,
        )
        return res_out, norm_out

    def allreduce_combine(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor | None,
        res: torch.Tensor,
        inj: torch.Tensor,
    ) -> torch.Tensor | None:
        m = p0.shape[0]
        if (
            not self.can_allreduce
            or not 1 <= m <= 64
            or p0.shape != (m, 2560)
            or res.shape != (m, 10240)
            or inj.shape != (m, 4)
            or not p0.is_contiguous()
            or not res.is_contiguous()
            or not inj.is_contiguous()
            or p0.data_ptr() % 16 != 0
            or res.data_ptr() % 16 != 0
            or (
                p1 is not None
                and (
                    p1.shape != (m, 2560)
                    or not p1.is_contiguous()
                    or p1.data_ptr() % 16 != 0
                )
            )
        ):
            return None
        res_out = res.new_empty((m, 10240))
        torch.ops._C.sm70_hc_ll_allreduce_combine_norm_out(
            res_out,
            self._empty_lora,
            p0,
            p1 if p1 is not None else self._empty_lora,
            res,
            inj,
            self._empty_lora,
            1e-6,
            self.ar_counter,
            self.ar_dirty_m,
            self.ar_seq,
            self.ar_pointers,
            self.logical_rank,
        )
        return res_out

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
