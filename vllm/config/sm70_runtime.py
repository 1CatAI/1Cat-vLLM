# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Initialization-only compatibility for runner lifecycle policy."""

import os
from dataclasses import fields
from typing import ClassVar

from pydantic import Field

from vllm.config.utils import config


def resolve_legacy_fields(
    policy,
    aliases: dict[str, str],
    *,
    inactive_defaults: dict[str, int | float] | None = None,
) -> None:
    from vllm import envs

    for field in fields(policy):
        name = field.name
        if name not in aliases:
            continue
        legacy = aliases[name]
        if getattr(policy, name) is not None:
            policy.sources[name] = "typed"
        else:
            source = legacy if legacy in os.environ else "default"
            try:
                value = envs.environment_variables[legacy]()
            except ValueError:
                if inactive_defaults is None or name not in inactive_defaults:
                    raise
                value = inactive_defaults[name]
                source = f"inactive default (ignored {legacy})"
            setattr(policy, name, value)
            policy.sources[name] = source
            if legacy == "VLLM_SM70_MTP_PROFILE" and "VLLM_SM70_DEBUG" in os.environ:
                policy.sources[name] = "VLLM_SM70_DEBUG"


@config
class Sm70RuntimeConfig:
    """Warmup policy; does not alter the compiled model computation."""

    staged_input: bool | None = None
    """Opt into the existing single-request asynchronous input staging gate."""

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
                "staged_input": "VLLM_SM70_ASYNC_STAGED_INPUT_PREP",
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


@config
class RuntimeTraceConfig:
    """Captured runner diagnostics; never part of compiled computation."""

    async_cpu: bool | None = None
    """Trace asynchronous runner preparation/execute/sample CPU stages."""
    async_every: int | None = Field(default=None, ge=1)
    """Asynchronous trace interval, default 16 with legacy clamping."""
    events: bool | None = None
    """Trace synchronization events and matching NVTX ranges."""
    event_every: int | None = Field(default=None, ge=1)
    """Report the first four slow events and then every this many events."""
    event_threshold_ms: float | None = None
    """Event duration reporting threshold, retaining the legacy float parser."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Source of each initialized diagnostic option."""

    def __post_init__(self) -> None:
        from vllm import envs

        resolve_legacy_fields(
            self,
            {
                "async_cpu": "VLLM_SM70_ASYNC_CPU_TRACE",
                "events": "VLLM_SM70_DECODE_EVENT_TRACE",
            },
        )
        # Disabled legacy diagnostics never parsed their numeric options.
        # Retain valid captured values, but do not reject an unused malformed
        # interval/threshold. The deferred worker override still enables its
        # original interval parser during initialization.
        inactive: dict[str, int | float] = {}
        worker_profile = (
            envs.environment_variables["VLLM_DFLASH_DDTREE_WORKER_PROFILE"]() == "1"
        )
        if not self.async_cpu and not worker_profile:
            inactive["async_every"] = 16
        if not self.events:
            inactive.update(event_every=16, event_threshold_ms=1.0)
        resolve_legacy_fields(
            self,
            {
                "async_every": "VLLM_SM70_ASYNC_CPU_TRACE_EVERY",
                "event_every": "VLLM_SM70_DECODE_EVENT_TRACE_EVERY",
                "event_threshold_ms": "VLLM_SM70_DECODE_EVENT_TRACE_THRESHOLD_MS",
            },
            inactive_defaults=inactive,
        )
        if "VLLM_SM70_DEBUG" in os.environ and self.sources["events"] != "typed":
            self.sources["events"] = "VLLM_SM70_DEBUG"


@config
class SpecDecodeTraceConfig:
    """Target-logit diagnostics for all existing speculative methods."""

    legacy_fields: ClassVar[dict[str, str]] = {
        "target_logits": "VLLM_DFLASH_DEBUG_TARGET_LOGITS",
        "target_min_position": "VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION",
    }

    target_logits: bool | None = None
    """Legacy target-logit trace, enabled only by the exact string '1'."""
    target_min_position: int | None = None
    """First traced target position, default 8; not a computation policy."""
    legacy_min_position: str | None = Field(default=None, init=False, repr=False)
    """Captured raw threshold; parse only when a speculative consumer initializes."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Sources of initialized diagnostic fields."""

    def __post_init__(self) -> None:
        from vllm import envs

        alias = self.legacy_fields["target_logits"]
        if self.target_logits is None:
            self.target_logits = envs.environment_variables[alias]() == "1"
            self.sources["target_logits"] = alias if alias in os.environ else "default"
        else:
            self.sources["target_logits"] = "typed"
        alias = self.legacy_fields["target_min_position"]
        if self.target_min_position is None:
            # The legacy threshold was not parsed when tracing was disabled.
            raw = envs.environment_variables[alias]() if self.target_logits else None
            self.legacy_min_position = raw
            self.sources["target_min_position"] = (
                alias if alias in os.environ and raw is not None else "default"
            )
        else:
            self.sources["target_min_position"] = "typed"

    def resolve(self) -> "SpecDecodeTraceConfig":
        if self.target_min_position is None:
            self.target_min_position = int(
                self.legacy_min_position
                if self.legacy_min_position is not None
                else "8"
            )
        return self
