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


class PrefillRuntime:
    """Engine-owned FA2 buffers; the kernel's physical-device gate stays shared."""

    def __init__(self, options, native):
        import torch

        probe = getattr(native, "sm70_prefill_policy_abi", None)
        if probe is None or probe() != 1:
            raise RuntimeError(
                "Flash-V100 79T requires FA2 prefill policy ABI 1; rebuild FA2 "
                "to bind explicit per-engine policy and workspaces."
            )
        self.owner = torch.classes._vllm_fa2_C.Sm70PrefillRuntime(
            list(options.prefill_native_effective)
        )
        # Resolve bound methods once. No native strings or env parsing in forward.
        self.operations = {8000: self.owner.q8000, 8192: self.owner.q8192}

    def close(self):
        self.owner.close()

    def explain(self):
        counts = self.owner.observations()
        return {
            "evidence": "host dispatches: capture included; replay excluded",
            "q8000": counts[0],
            "q8192": counts[1],
        }


def prepare_prefill_runtime(native):
    resources = current_runtime_resources()
    if resources is None or "sm70_prefill" in resources:
        return
    from vllm.config.execution_policy import flash_v100_policy

    options = flash_v100_policy().options
    if (
        not options.fa2_d256_prefill
        or not options.prefill_d256_gqa_arch_128k_experimental
        or options.prefill_d256_gqa_v37
    ):
        return
    if not hasattr(native, "sm70_d256_gqa_architecture_fwd"):
        return  # Missing optional operator retains the existing dense fallback.
    resources["sm70_prefill"] = PrefillRuntime(options, native)


def bind_prefill_operation(operation, query_tokens):
    if operation is None:
        return None
    resources = current_runtime_resources()
    if resources is None:
        return operation  # Independent no-configuration compatibility caller.
    owner = resources.get("sm70_prefill")
    if owner is not None:
        return owner.operations[query_tokens]
    from vllm.config.execution_policy import flash_v100_policy

    options = flash_v100_policy().options
    if (
        options.fa2_d256_prefill
        and options.prefill_d256_gqa_arch_128k_experimental
        and not options.prefill_d256_gqa_v37
    ):
        raise RuntimeError("SM70 prefill resources must be bound before execution")
    return None  # Qualified engine execution never uses the legacy native entry.
