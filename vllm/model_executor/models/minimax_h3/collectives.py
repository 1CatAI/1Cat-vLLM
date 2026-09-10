# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline ownership and request accounting for explicit residual collectives."""

import time

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
            if self.plan is None:
                started = time.perf_counter()
                self.plan = SM70ExactRowReductionPlan(
                    self.group, shape, memory_budget_bytes=self.memory_budget_bytes
                )
                self.setup_seconds += time.perf_counter() - started
            self.raw_peak_bytes = max(self.raw_peak_bytes, self.plan.raw_ipc_bytes)
            self.peer_calls += 1
            return self.plan.reduce(value)
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
