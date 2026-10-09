# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Initialization-only compatibility for runner lifecycle policy."""

import os
from dataclasses import fields

from pydantic import Field

from vllm.config.utils import config


def resolve_legacy_fields(policy, aliases: dict[str, str]) -> None:
    from vllm import envs

    for field in fields(policy):
        name = field.name
        if name not in aliases:
            continue
        legacy = aliases[name]
        if getattr(policy, name) is not None:
            policy.sources[name] = "typed"
        else:
            setattr(policy, name, envs.environment_variables[legacy]())
            policy.sources[name] = legacy if legacy in os.environ else "default"
            if legacy == "VLLM_SM70_MTP_PROFILE" and "VLLM_SM70_DEBUG" in os.environ:
                policy.sources[name] = "VLLM_SM70_DEBUG"


@config
class Sm70RuntimeConfig:
    """Warmup policy; does not alter the compiled model computation."""

    auxiliary_warmup: bool | None = None
    """Warm eligible helper kernels before the first request; legacy default on."""
    mtp_concurrency_warmup: bool | None = None
    """Include alternate MTP warmup batch sizes; legacy default off."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance, excluded from compiled computation."""

    def __post_init__(self) -> None:
        resolve_legacy_fields(
            self,
            {
                "auxiliary_warmup": "VLLM_SM70_AUX_KERNEL_WARMUP",
                "mtp_concurrency_warmup": "VLLM_SM70_MTP_CONCURRENCY_WARMUP",
            },
        )


@config
class StepProfilerConfig:
    """Diagnostic-only policy; CUDA eligibility belongs to the consumer."""

    enabled: bool | None = None
    """Collect eligible speculative step timings; legacy default off."""
    interval: int | None = Field(default=None, ge=1)
    """Report every this many calls, also reporting the first call."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance, excluded from compiled computation."""

    def __post_init__(self) -> None:
        resolve_legacy_fields(
            self,
            {
                "enabled": "VLLM_SM70_MTP_PROFILE",
                "interval": "VLLM_SM70_MTP_PROFILE_INTERVAL",
            },
        )


def capture_runtime_config() -> Sm70RuntimeConfig:
    """Standalone warmup compatibility; engine consumers pass their own policy."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return config.kernel_config.sm70_runtime if config else Sm70RuntimeConfig()
