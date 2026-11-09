# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit host policy and workspace ownership for the independent package.

This package does not depend on an inference engine. An engine prepares one
runtime after worker configuration transfer and passes it to its bound calls.
The old functions without a runtime retain their independent compatibility API.
"""

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace


@dataclass(frozen=True)
class PythonPolicy:
    dynamic_partitions: bool
    staged_pv: bool
    share_workspace: bool
    partition_size: int | None
    partition_error: str | None
    scalar_fast: bool
    batch_xqa: bool
    padded_smem: bool
    dual_cta: bool
    partition_cause: str | None = None

    def partition(self) -> int | None:
        if self.partition_error is not None:
            if self.partition_cause is not None:
                raise ValueError(self.partition_error) from ValueError(
                    self.partition_cause
                )
            raise ValueError(self.partition_error)
        return self.partition_size


class AttentionRuntime:
    """One owner for native observations, plans and all package workspaces."""

    def __init__(self, policy: PythonPolicy, native_inputs):
        from .flash_attn_interface import flash_attn_v100_cuda as extension

        if getattr(extension, "policy_abi_version", 0) != 1 or getattr(
            extension, "policy_fields", 0
        ) != len(native_inputs):
            raise RuntimeError(
                "Engine Flash-V100 policy requires a rebuilt extension with "
                "policy ABI version 1; the loaded binary cannot bind its policy."
            )
        self.policy = policy
        self.native_policy = extension.PreparedPolicy(native_inputs)
        self.native = SimpleNamespace()
        for name in dir(extension):
            if name.endswith("_configured"):
                setattr(
                    self.native,
                    name.removesuffix("_configured"),
                    partial(getattr(extension, name), self.native_policy),
                )
        self.decode_plan_cache = {}
        self.decode_workspace_cache = {}
        self.xqa_staged_rescale_workspace_cache = {}
        self.turboquant_decode_workspace_cache = {}
        self.prefill_splitkv3_workspace_cache = {}
        self.grouped_verify_workspace_cache = {}
        self.bindings = {}

    def bind(self, operation):
        if operation is None:
            return None
        if operation not in self.bindings:
            bound = partial(operation, _runtime=self)
            bound.__dict__.update(getattr(operation, "__dict__", {}))
            self.bindings[operation] = bound
        return self.bindings[operation]

    def observations(self):
        """Host dispatch/capture counts, never a claim about graph replay hits."""
        return tuple(self.native_policy.observations)

    def close(self):
        """Release only this engine's scratch after its graphs are destroyed."""
        self.decode_plan_cache.clear()
        self.decode_workspace_cache.clear()
        self.xqa_staged_rescale_workspace_cache.clear()
        self.turboquant_decode_workspace_cache.clear()
        self.prefill_splitkv3_workspace_cache.clear()
        self.grouped_verify_workspace_cache.clear()
        self.bindings.clear()
        self.native.__dict__.clear()
        self.native_policy = None
