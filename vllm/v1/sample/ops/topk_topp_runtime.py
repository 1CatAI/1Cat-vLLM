# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling policy and scratch storage bound before out-of-forward sampling."""

from dataclasses import dataclass, field

from vllm.config.execution_policy import LayerExecutionPolicy, layer_policy
from vllm.runtime_resources import current_runtime_resources, runtime_resources_for


@dataclass
class TopKTopPRuntime:
    policy: LayerExecutionPolicy
    buffers: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)

    def close(self):
        # The graph branch uses the reference path before borrowing these buffers.
        self.buffers.clear()
        self.tables.clear()


_STANDALONE_BUFFERS: dict = {}
_STANDALONE_TABLES: dict = {}


def bind_topk_topp_runtime(config=None):
    resources = (
        runtime_resources_for(config)
        if config is not None
        else current_runtime_resources()
    )
    if resources is None:
        # Standalone compatibility keeps its historical allocation cache, while
        # its next explicit call may still take a fresh legacy policy snapshot.
        return TopKTopPRuntime(layer_policy(), _STANDALONE_BUFFERS, _STANDALONE_TABLES)
    runtime = resources.get("topk_topp")
    if runtime is None:
        runtime = resources["topk_topp"] = TopKTopPRuntime(layer_policy(config))
    return runtime
