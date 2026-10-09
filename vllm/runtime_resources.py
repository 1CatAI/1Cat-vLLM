# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local resources carried by an engine and its forward contexts.

Resources are bound after configuration transfer. The configuration's declared
fields stay serializable and hashable; this private runtime map is not a policy
input. A forward borrows the same owners, never a copy of their mutable state.
"""

from typing import Any


def runtime_resources_for(config) -> dict[str, Any]:
    resources = getattr(config, "_runtime_resources", None)
    if resources is None:
        from vllm.config.execution_policy import POLICY_OWNERS

        policies = {}
        for path in POLICY_OWNERS:
            owner, name = path.split(".")
            policy = getattr(getattr(config, owner, None), name, None)
            if policy is not None:
                policies[path] = policy
        resources = {"execution_policies": policies}
        config._runtime_resources = resources
    return resources


def current_runtime_resources() -> dict[str, Any] | None:
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if is_forward_context_available():
        return get_forward_context().runtime_resources
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return runtime_resources_for(config) if config is not None else None
