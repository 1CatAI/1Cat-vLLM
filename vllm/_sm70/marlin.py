# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind Marlin policy at its existing workspace-preparation checkpoint."""

from vllm.runtime_resources import current_runtime_resources


def bind_marlin_workspace(device) -> None:
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None or device.type != "cuda":
        return
    resources = current_runtime_resources()
    assert resources is not None
    if "sm70_marlin_binding" not in resources:
        from vllm._sm70.policy import NativeBindings

        policy = config.kernel_config.sm70_marlin
        policy.resolve("marlin")
        # The old operator schema borrows this slot from the existing native
        # runtime scope. AOT contains neither mutable owner nor pointer values.
        resources["sm70_marlin_binding"] = NativeBindings(policy.values)
