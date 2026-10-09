# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Initialized GDN computation policy and legacy input provenance."""

import os
from dataclasses import fields

from pydantic import Field

from vllm.config.gdn_projection import GdnProjectionConfig
from vllm.config.gdn_schedule import GdnScheduleConfig
from vllm.config.gdn_state import GdnStateConfig
from vllm.config.utils import config, hash_factors

GDN_LEGACY_FIELDS = {
    "packed_recurrent_decode": "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
    "mixed_qkv_decode": "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV",
    "flashqla_decode": "VLLM_SM70_GDN_DECODE_FLASHQLA",
    "legacy_prefill_prep": "VLLM_SM70_GDN_LEGACY_PREFILL_PREP",
}
GDN_TEXT_FLAGS = {
    "original_prefill": ("VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL", True),
    "indexed_prefill": ("VLLM_SM70_FLASHQLA_INDEXED_PREFILL", False),
    "direct_prefill_output": ("VLLM_SM70_FLASHQLA_DIRECT_OUTPUT", True),
    "decode_warmup": ("VLLM_SM70_FLASHQLA_DECODE_WARMUP", False),
}


@config
class GdnConfig:
    """Static policy; tensor admission and dynamic token dispatch stay in providers.

    Resolve after the existing model/platform default checkpoints, before loading
    GDN layers. A configuration without GDN does not read its legacy controls or
    contribute these options to graph hashes.
    """

    projection: GdnProjectionConfig = Field(default_factory=GdnProjectionConfig)
    """Projection, layout and opaque-boundary policy captured before compilation."""

    state: GdnStateConfig = Field(default_factory=GdnStateConfig)
    """State indices, metadata preparation and speculative boundary policy."""

    schedule: GdnScheduleConfig = Field(default_factory=GdnScheduleConfig)
    """Per-engine FLA tuning, shared by all recurrence and prefill stages."""

    prefill_backend: str | None = None
    """Requested backend; None preserves the additional_config compatibility key."""
    packed_recurrent_decode: bool | None = None
    """Use the existing packed recurrent decode candidate."""
    mixed_qkv_decode: bool | None = None
    """Admit the existing fused sigmoid mixed-QKV recurrence."""
    flashqla_decode: bool | None = None
    """Admit the existing FlashQLA global-state decode provider."""
    legacy_prefill_prep: bool | None = None
    """Keep separate prefill packing/gating as an explicit diagnostic algorithm."""
    original_prefill: bool | None = None
    """Use original TileLang FlashQLA prefill, including the historical alias."""
    indexed_prefill: bool | None = None
    """Allow original prefill to address and update the resident state directly."""
    direct_prefill_output: bool | None = None
    """Supply the existing preallocated output when its tensor contract matches."""
    decode_warmup: bool | None = None
    """Warm the optional FlashQLA decode provider; not a computation hash input."""
    native_verify: bool | None = None
    """Existing sequential CUDA verifier; legacy KernelConfig flag remains valid."""
    resolved: bool = Field(default=False, init=False)
    """Whether an active engine has captured this configuration."""
    active_prefill_backend: str | None = Field(default=None, init=False)
    """Existing selector's bound result, captured before model compilation."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance; excluded from computation hashes."""

    def apply_platform_defaults(self) -> None:
        """Called at the original baseline checkpoint, without process writes."""
        for name in ("packed_recurrent_decode", "flashqla_decode"):
            if (
                getattr(self, name) is None
                and GDN_LEGACY_FIELDS[name] not in os.environ
            ):
                setattr(self, name, True)
                self.sources[name] = "platform baseline"
        # Schedule defaults are already true. Their parser intentionally keeps
        # the original exact-'1' semantics and explicit legacy overrides.

    def resolve(self, *, additional_config=None, native_verify: bool = False) -> None:
        if self.resolved:
            return
        from vllm import envs

        if self.prefill_backend is None:
            supplied = (
                isinstance(additional_config, dict)
                and "gdn_prefill_backend" in additional_config
            )
            value = (
                additional_config.get("gdn_prefill_backend", "auto")
                if isinstance(additional_config, dict)
                else "auto"
            )
            self.prefill_backend = str(value).strip().lower()
            self.sources["prefill_backend"] = (
                "additional_config.gdn_prefill_backend" if supplied else "default"
            )
        else:
            self.prefill_backend = self.prefill_backend.strip().lower()
            self.sources["prefill_backend"] = "typed"
        for field, legacy in GDN_LEGACY_FIELDS.items():
            if getattr(self, field) is None:
                setattr(self, field, envs.environment_variables[legacy]())
                self.sources[field] = legacy if legacy in os.environ else "default"
            else:
                self.sources.setdefault(field, "typed")
        for field, (legacy, default) in GDN_TEXT_FLAGS.items():
            if getattr(self, field) is not None:
                self.sources[field] = "typed"
                continue
            raw = os.environ.get(legacy)
            source = legacy if raw is not None else "default"
            if raw is None and field == "original_prefill":
                raw = os.environ.get("FLASH_QLA_SM70_USE_ORIGINAL_TILELANG")
                if raw is not None:
                    source = "FLASH_QLA_SM70_USE_ORIGINAL_TILELANG"
            value = (
                default
                if raw is None
                else raw.strip().lower() not in ("0", "false", "no", "off", "")
            )
            setattr(self, field, value)
            self.sources[field] = source
        if self.native_verify is None:
            self.native_verify = native_verify
            self.sources["native_verify"] = (
                "KernelConfig.sm70_gdn_verify" if native_verify else "default"
            )
        else:
            self.sources["native_verify"] = "typed"
        self.projection.resolve()
        self.state.resolve()
        self.schedule.resolve()
        self.resolved = True

    def compute_hash(self) -> str:
        return hash_factors(self.graph_options())

    def graph_options(self) -> dict:
        if not self.resolved:
            return {}
        options = {
            field.name: (
                {
                    field: getattr(self.projection, field)
                    for field in self.projection.aliases
                }
                if field.name == "projection"
                else self.schedule.graph_options()
                if field.name == "schedule"
                else self.state.graph_options()
                if field.name == "state"
                else getattr(self, field.name)
            )
            for field in fields(self)
            if field.name
            not in {"resolved", "sources", "decode_warmup", "active_prefill_backend"}
        }
        if self.active_prefill_backend is not None:
            options["prefill_backend"] = self.active_prefill_backend
            if self.active_prefill_backend != "flashqla_sm70":
                options.pop("original_prefill")
                options.pop("indexed_prefill")
            # Only the native prefill and FlashQLA's dtype fallback can consume
            # FLA chunk tuning. Recurrent schedules remain active in every plan.
            if self.active_prefill_backend in ("flashinfer", "cutedsl"):
                options["schedule"] = {
                    name: value
                    for name, value in self.schedule.graph_options().items()
                    if name.startswith(("recurrent_", "sigmoid_"))
                }
        return options


def resolve_gdn_config(vllm_config) -> GdnConfig:
    """Bind once at GDN initialization, after existing engine defaults."""
    kernel = getattr(vllm_config, "kernel_config", None)
    policy = getattr(kernel, "gdn", None)
    if policy is None:
        policy = GdnConfig()
    policy.resolve(
        additional_config=getattr(vllm_config, "additional_config", None),
        native_verify=bool(getattr(kernel, "sm70_gdn_verify", False)),
    )
    return policy


@config
class GdnProfileConfig:
    """Prefill synchronization/timing diagnostics; never a graph hash input."""

    enabled: bool | None = None
    """Enable the existing per-stage prefill timing log."""
    max_logs: int | None = None
    """Engine-wide log budget; preserve zero and negative legacy limits."""
    max_per_stage: int | None = None
    """Per-layer/stage log budget."""
    resolved: bool = Field(default=False, init=False)
    """Whether initialization captured the diagnostic policy."""

    def resolve(self) -> None:
        if self.resolved:
            return
        if self.enabled is None:
            raw = os.environ.get("VLLM_SM70_GDN_PREFILL_PROFILE", "")
            self.enabled = raw.strip().lower() in ("1", "true", "yes", "on")
        for name, legacy, default in (
            ("max_logs", "VLLM_SM70_GDN_PREFILL_PROFILE_MAX_LOGS", 256),
            ("max_per_stage", "VLLM_SM70_GDN_PREFILL_PROFILE_MAX_PER_STAGE", 2),
        ):
            if getattr(self, name) is None:
                value = (
                    int(os.environ.get(legacy, str(default)))
                    if self.enabled
                    else default
                )
                setattr(self, name, value)
        self.resolved = True
