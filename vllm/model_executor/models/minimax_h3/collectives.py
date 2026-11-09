# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline ownership and request accounting for explicit residual collectives."""

import time

import torch
import torch.distributed as dist

from vllm.model_executor.layers.sm70_collectives import SM70ExactRowReductionPlan


class H3ResidualReduction:
    def __init__(self, group, *, memory_budget_bytes):
        self.group = group
        self.memory_budget_bytes = memory_budget_bytes
        self.plan: SM70ExactRowReductionPlan | None = None
        self.begin_request()

    def begin_request(self):
        self.peer_calls = 0
        self.native_calls = 0
        self.setup_seconds = 0.0
        self.raw_peak_bytes = self.plan.raw_ipc_bytes if self.plan is not None else 0
        self.fallback_reason = None
        self._rejected_shape = None

    def _setup_unavailable(self, value):
        """Agree before allocating IPC scratch; one slow/full rank affects all."""
        device = value.device
        free, total = torch.accelerator.get_memory_info(device)
        reusable = torch.accelerator.memory_reserved(
            device
        ) - torch.accelerator.memory_allocated(device)
        required = SM70ExactRowReductionPlan.required_memory_bytes(value.shape)
        reserve = max(512 * 1024**2, total // 20)
        if free < required + reserve <= free + reusable:
            # Raw CUDA IPC allocations cannot reuse PyTorch's cached blocks.
            # Release only this worker's inactive allocator cache, once at setup.
            torch.accelerator.empty_cache()
            free, total = torch.accelerator.get_memory_info(device)
        reason = None
        if required + reserve > free:
            reason = "insufficient free memory for residual communication setup"
        elif any(
            peer != device.index
            and not torch.cuda.can_device_access_peer(device.index, peer)
            for peer in range(self.group.world_size)
        ):
            reason = "peer access unavailable for this GPU group"
        reasons = [None] * self.group.world_size
        with torch.inference_mode(False):
            dist.all_gather_object(reasons, reason, group=self.group.cpu_group)
        return next((item for item in reasons if item is not None), None)

    def reduce(self, value):
        shape = tuple(value.shape)
        if self.plan is not None and self.plan.shape != shape:
            self.plan.close()
            self.plan = None
        if self.group.world_size != 4:
            self.fallback_reason = "peer execution requires TP4"
        elif (
            SM70ExactRowReductionPlan.required_memory_bytes(shape)
            > self.memory_budget_bytes
        ):
            self.fallback_reason = "shape exceeds residual communication budget"
        else:
            if self.plan is None and self._rejected_shape != shape:
                self.fallback_reason = self._setup_unavailable(value)
                if self.fallback_reason:
                    self._rejected_shape = shape
            if self._rejected_shape == shape:
                return self._native(value)
            if self.plan is None:
                started = time.perf_counter()
                self.plan = SM70ExactRowReductionPlan(
                    self.group, shape, memory_budget_bytes=self.memory_budget_bytes
                )
                self.setup_seconds += time.perf_counter() - started
            self.raw_peak_bytes = max(self.raw_peak_bytes, self.plan.raw_ipc_bytes)
            self.peer_calls += 1
            return self.plan.reduce(value)
        return self._native(value)

    def _native(self, value):
        self.native_calls += 1
        rows = value.shape[0] // self.group.world_size
        return self.group.all_reduce(value).narrow(
            0, self.group.rank_in_group * rows, rows
        )

    def snapshot(self):
        return {
            "configured_backend": "peer",
            "peer_calls": self.peer_calls,
            "native_calls": self.native_calls,
            "fallback_reason": self.fallback_reason,
            "setup_seconds": self.setup_seconds,
            "raw_ipc_peak_bytes": self.raw_peak_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
        }

    def close(self):
        if self.plan is not None:
            self.plan.close()
            self.plan = None
