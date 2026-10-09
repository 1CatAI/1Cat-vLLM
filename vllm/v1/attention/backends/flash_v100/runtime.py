# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialize the independent attention package from resolved engine owners."""

from vllm.runtime_resources import current_runtime_resources


def prepare_attention_runtime():
    """Called at operator loading, before forward or CUDA graph capture."""
    resources = current_runtime_resources()
    if resources is None:
        return None
    if "flash_v100" not in resources:
        from flash_attn_v100.runtime import AttentionRuntime, PythonPolicy

        from vllm.config.execution_policy import flash_v100_policy, graph_policy
        from vllm.config.sm70_runtime import capture_runtime_trace

        options = flash_v100_policy().options
        # Partial standalone engine fixtures can omit the ordered default pass.
        # Real workers transfer these projections with their configuration.
        if not options.native_inputs:
            options.finalize(graph_policy(), capture_runtime_trace().flash_v100)
        resources["flash_v100"] = AttentionRuntime(
            PythonPolicy(**options.python_policy), options.native_inputs
        )
    return resources["flash_v100"]


def bind_attention_operation(operation):
    if operation is None:
        return None
    resources = current_runtime_resources()
    if resources is None:
        return operation
    runtime = resources.get("flash_v100")
    if runtime is None:
        raise RuntimeError("Flash-V100 resources must be bound before execution")
    return runtime.bind(operation)
