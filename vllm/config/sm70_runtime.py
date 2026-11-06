# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Initialization-only compatibility for runner lifecycle policy."""

import os
from typing import ClassVar

import torch
from pydantic import Field

from vllm.config.diagnostic_dump import (
    DUMP_BINDINGS,
    TensorDiagnosticsConfig,
    TensorDumpConfig,
)
from vllm.config.diagnostic_sampling import SamplingDiagnosticsConfig
from vllm.config.flash_v100 import FlashV100Diagnostics
from vllm.config.sm70_dflash2 import DFlashDiagnosticsConfig
from vllm.config.turboquant_runtime import TurboQuantDiagnostics
from vllm.config.utils import config


def resolve_legacy_fields(
    policy, aliases, *, inactive_defaults=None, reader=None, deferred_errors=None
):
    from vllm.config.utils import resolve_legacy_fields as resolve

    resolve(
        policy,
        aliases,
        inactive_defaults=inactive_defaults,
        deferred_errors=deferred_errors,
        reader=reader,
        source_overrides={"VLLM_SM70_MTP_PROFILE": "VLLM_SM70_DEBUG"}
        if "VLLM_SM70_DEBUG" in os.environ
        else None,
    )


@config
class Sm70RuntimeConfig:
    """Warmup policy; does not alter the compiled model computation."""

    runtime_aliases: ClassVar[dict[str, str]] = {
        "awq_warmup_max_m": "VLLM_SM70_AWQ_WARMUP_MAX_M",
        "staged_input": "VLLM_SM70_ASYNC_STAGED_INPUT_PREP",
        "auxiliary_warmup": "VLLM_SM70_AUX_KERNEL_WARMUP",
        "mtp_concurrency_warmup": "VLLM_SM70_MTP_CONCURRENCY_WARMUP",
    }

    legacy_output_token_repair: bool | None = None
    """Retain the async output-history rollback for speculative and ordinary runs."""
    input_aliases: ClassVar[dict[str, str]] = {
        "legacy_output_token_repair": "VLLM_SM70_MTP_LEGACY_OUTPUT_TOKEN_REPAIR",
    }

    awq_warmup: bool | None = None
    """Run the existing quantized-kernel warmup at the original checkpoint."""
    awq_warmup_max_moe_tokens: int | None = None
    """Largest MoE warmup shape, clamped by the consumer's decode sizes."""
    fp8_coordinated_tuning: bool | None = None
    """Share the existing authoritative tensor-parallel tuning plan."""
    gemm_lut_path: str | None = None
    """Optional tuning-cache template; device/rank expansion remains worker-local."""
    errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Deferred errors for options skipped by the original warmup gates."""

    warmup_aliases: ClassVar[dict[str, str]] = {
        "awq_warmup": "VLLM_SM70_AWQ_WARMUP",
        "awq_warmup_max_moe_tokens": "VLLM_SM70_AWQ_WARMUP_MAX_MOE_TOKENS",
        "fp8_coordinated_tuning": "VLLM_SM70_FP8_COORDINATED_TUNING",
        "gemm_lut_path": "VLLM_SM70_GEMM_LUT_PATH",
    }

    awq_warmup_max_m: int | None = None
    """Largest dense AWQ warmup shape; platform default applies at engine init."""

    staged_input: bool | None = None
    """Opt into the existing single-request asynchronous input staging gate."""

    auxiliary_warmup: bool | None = None
    """Warm eligible helper kernels before the first request; legacy default on."""
    mtp_concurrency_warmup: bool | None = None
    """Include alternate MTP warmup batch sizes; legacy default off."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance, excluded from compiled computation."""

    def value(self, field):
        if field in self.errors:
            raise ValueError(self.errors[field])
        return getattr(self, field)

    def __post_init__(self) -> None:
        resolve_legacy_fields(
            self,
            {
                field: alias
                for field, alias in (self.warmup_aliases | self.input_aliases).items()
                if field not in self.sources
            },
            deferred_errors=self.errors,
        )
        resolve_legacy_fields(
            self,
            self.runtime_aliases,
        )

    def compile_ignored_aliases(self):
        return set(
            (self.warmup_aliases | self.input_aliases | self.runtime_aliases).values()
        )


def bind_output_token_repair(policy=None):
    """Bind a narrow deferred value; old standalone batches capture only their flag."""
    from functools import partial

    if policy is not None:
        return partial(policy.value, "legacy_output_token_repair")
    from vllm.config.legacy_inputs import LegacyInputs

    inputs = LegacyInputs()
    alias = Sm70RuntimeConfig.input_aliases["legacy_output_token_repair"]
    inputs.capture((alias,))
    return partial(inputs.value, alias)


@config
class StepProfilerConfig:
    """Diagnostic-only policy; CUDA eligibility belongs to the consumer."""

    aliases: ClassVar[dict[str, str]] = {
        "enabled": "VLLM_SM70_MTP_PROFILE",
        "interval": "VLLM_SM70_MTP_PROFILE_INTERVAL",
    }

    enabled: bool | None = None
    """Collect eligible speculative step timings; legacy default off."""
    interval: int | None = Field(default=None, ge=1)
    """Report every this many calls, also reporting the first call."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance, excluded from compiled computation."""

    def __post_init__(self) -> None:
        resolve_legacy_fields(
            self,
            self.aliases,
        )

    def compile_ignored_aliases(self):
        return set(self.aliases.values())


def capture_runtime_config() -> Sm70RuntimeConfig:
    """Standalone warmup compatibility; engine consumers pass their own policy."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return config.kernel_config.sm70_runtime if config else Sm70RuntimeConfig()


@config
class RuntimeTraceConfig:
    """Captured runner diagnostics; never part of compiled computation."""

    interval_aliases: ClassVar[dict[str, str]] = {
        "async_every": "VLLM_SM70_ASYNC_CPU_TRACE_EVERY",
        "event_every": "VLLM_SM70_DECODE_EVENT_TRACE_EVERY",
        "event_threshold_ms": "VLLM_SM70_DECODE_EVENT_TRACE_THRESHOLD_MS",
    }

    event_aliases: ClassVar[dict[str, str]] = {
        "async_cpu": "VLLM_SM70_ASYNC_CPU_TRACE",
        "events": "VLLM_SM70_DECODE_EVENT_TRACE",
    }

    turboquant: TurboQuantDiagnostics = Field(default_factory=TurboQuantDiagnostics)
    """Packed-cache compare policy; counters and outputs share engine diagnostics."""

    flash_v100: FlashV100Diagnostics = Field(default_factory=FlashV100Diagnostics)
    """Attention comparison, native trace and route observations."""

    dflash: DFlashDiagnosticsConfig = Field(default_factory=DFlashDiagnosticsConfig)
    """Family-specific trace inputs; qualification belongs to its adapter."""

    sampling: SamplingDiagnosticsConfig = Field(
        default_factory=SamplingDiagnosticsConfig
    )
    """Sampler and proposer diagnostic policy, independent of RNG policy."""

    dumps: TensorDiagnosticsConfig = Field(default_factory=TensorDiagnosticsConfig)
    """Shared tensor-diagnostic policy; mutable observations are engine-owned."""

    layer_aliases: ClassVar[dict[str, str]] = {
        "spec_target_profiler_step": "VLLM_SM70_SPEC_TARGET_FORWARD_PROFILER_STEP",
        "shared_gate_replaced_notice": "VLLM_SM70_SHARED_GATE_MAX_M",
        "gdn_empty_output_notice": "VLLM_SM70_GDN_EMPTY_CORE_OUT",
        "gdn_legacy_fused_notice": "VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING",
        "require_profile_acceleration": "VLLM_SM70_REQUIRE_PROFILE_ACCELERATION",
        "gdn_route_debug": "VLLM_SM70_GDN_DECODE_FLASHQLA_ROUTE_DEBUG",
        "gdn_mixed_compare": "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV_COMPARE",
        "sync_before_compile": "VLLM_SM70_SYNC_BEFORE_COMPILE_GRAPH_FORWARD",
        "spec_target_nvtx": "VLLM_SM70_SPEC_TARGET_FORWARD_NVTX",
        "cg_dispatch": "VLLM_SM70_CG_DISPATCH_DEBUG",
        "tp_allreduce": "VLLM_TP_ALLREDUCE_TRACE",
        "dense_debug": "VLLM_SM70_F16_DENSE_DEBUG",
        "qwen_next_trace": "VLLM_QWEN3_NEXT_SM70_TRACE",
        "unquant_debug": "VLLM_SM70_UNQUANT_DEBUG",
        "profile_trace": "VLLM_SM70_PROFILE_TRACE",
        "qwen_mlp_internals": "VLLM_SM70_DUMP_QWEN_MLP_INTERNALS",
        "mtp_load": "VLLM_DEBUG_MTP_LOAD",
        "mtp_load_verbose": "VLLM_DEBUG_MTP_LOAD_VERBOSE",
        "greedy_token_trace": "VLLM_SM70_GREEDY_TOKEN_FASTPATH_TRACE",
    }

    legacy_layer_aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "profile_trace": (
            "VLLM_SM70_PROFILE_TRACE",
            "VLLM_SM70_DECODE_TILE_PROFILE",
            "VLLM_SM70_DEBUG",
        ),
        "events": ("VLLM_SM70_DECODE_EVENT_TRACE", "VLLM_SM70_DEBUG"),
        "spec_target_profiler_step": (
            "VLLM_SM70_SPEC_TARGET_FORWARD_PROFILER_STEP",
            "VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP",
        ),
        "spec_target_nvtx": (
            "VLLM_SM70_SPEC_TARGET_FORWARD_NVTX",
            "VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX",
        ),
    }

    spec_target_profiler_step: int | None = None
    """DSpark/legacy tree target step profiler; zero preserves disabled state."""

    shared_gate_replaced_notice: bool | None = None
    """Record the explicit obsolete M-limit input without parsing its value."""

    gdn_empty_output_notice: bool | None = None
    """Explain the retained paused empty-output experiment; allocation stays zeroed."""
    gdn_legacy_fused_notice: bool | None = None
    """Explain the replaced coarse gate when its legacy name was explicitly set."""

    require_profile_acceleration: bool | None = None
    """Fail initialization when required profile capabilities are unavailable."""

    gdn_route_debug: bool | None = None
    """Bounded GDN decode-admission reports."""
    gdn_mixed_compare: bool | None = None
    """Retain fused mixed-QKV comparison observations."""
    sync_before_compile: bool | None = None
    """Existing opt-in synchronization before graph forwarding."""
    spec_target_nvtx: bool | None = None
    """Existing speculative target-forward NVTX diagnostic."""
    cg_dispatch: bool | None = None
    """Bounded runner graph-dispatch diagnostics; no computation changes."""

    tp_allreduce: bool | None = None
    """Explain each collective route once per owning communicator or layer."""

    dense_debug: bool | None = None
    """Explain prepared dense FP16 projections."""
    qwen_next_trace: bool | None = None
    """Retain Qwen3Next projection route diagnostics."""
    unquant_debug: bool | None = None
    """Explain unquantized fallback projections."""
    profile_trace: bool | None = None
    """Trace retained model/layer route events without changing computation."""
    qwen_mlp_internals: bool | None = None
    """Admit the retained intermediate MLP tensor observation points."""
    mtp_load: bool | None = None
    """Report draft weight preparation at the original loading checkpoints."""
    mtp_load_verbose: bool | None = None
    """Include detailed draft parameter names only when loading trace is enabled."""
    greedy_token_trace: bool | None = None
    """Explain local greedy-token fastpath admission."""
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

    errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Deferred parse errors from model-qualified diagnostic inputs."""

    def dump_channels(self):
        return {
            **{name: getattr(self.dumps, name) for name in DUMP_BINDINGS},
            **self.dflash.dump_channels(),
            **self.turboquant.dump_channels(),
        }

    @staticmethod
    def legacy_dump_channel(name):
        if name in DUMP_BINDINGS:
            policy = TensorDumpConfig()
            policy.resolve(name)
            return policy
        return DFlashDiagnosticsConfig.legacy_dump_channel(name)

    def value(self, field: str):
        if field in self.errors:
            raise ValueError(self.errors[field])
        return getattr(self, field)

    def __post_init__(self) -> None:
        if self.sources:
            return
        self.flash_v100.resolve()
        self.turboquant.resolve()
        from vllm import envs

        def read_flag(name):
            if name == "VLLM_SM70_SPEC_TARGET_FORWARD_PROFILER_STEP":
                primary, fallback = self.legacy_layer_aliases[
                    "spec_target_profiler_step"
                ]
                raw = os.getenv(primary, os.getenv(fallback, "0"))
                try:
                    return max(0, int(raw))
                except ValueError:
                    return 0
            if name == "VLLM_SM70_SPEC_TARGET_FORWARD_NVTX":
                primary, fallback = self.legacy_layer_aliases["spec_target_nvtx"]
                return os.getenv(fallback, "0") == "1" or os.getenv(primary, "0") == "1"
            if name in (
                "VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING",
                "VLLM_SM70_SHARED_GATE_MAX_M",
            ):
                return name in os.environ
            if name == "VLLM_SM70_DUMP_QWEN_MLP_INTERNALS":
                return os.getenv(name) == "1"
            try:
                value = envs.environment_variables[name]()
            except ValueError as exc:
                field = next(
                    (
                        field
                        for field in (
                            "gdn_route_debug",
                            "gdn_empty_output_notice",
                            "gdn_mixed_compare",
                            "mtp_load",
                            "mtp_load_verbose",
                            "cg_dispatch",
                        )
                        if self.layer_aliases[field] == name
                    ),
                    None,
                )
                if field is None:
                    raise
                self.errors[field] = str(exc)
                return False
            return (
                value == "1" if name == "VLLM_SM70_SPEC_TARGET_FORWARD_NVTX" else value
            )

        resolve_legacy_fields(
            self,
            {
                "async_cpu": self.event_aliases["async_cpu"],
                **self.layer_aliases,
                "events": self.event_aliases["events"],
            },
            reader=read_flag,
        )
        for field, aliases in self.legacy_layer_aliases.items():
            if self.sources[field] != "typed":
                present = [alias for alias in aliases if alias in os.environ]
                if field == "spec_target_profiler_step":
                    present = present[:1]
                elif field in ("events", "profile_trace") and aliases[-1] in present:
                    present = [aliases[-1]]
                self.sources[field] = "+".join(present) if present else "default"
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
            self.interval_aliases,
            inactive_defaults=inactive,
        )

    def compile_ignored_aliases(self):
        return set(
            (self.layer_aliases | self.event_aliases | self.interval_aliases).values()
        ) | {name for aliases in self.legacy_layer_aliases.values() for name in aliases}


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
    min_position_error: str | None = Field(default=None, init=False)
    """Deferred malformed threshold, parsed once with its compatibility input."""
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
            raw = envs.environment_variables[alias]()
            self.legacy_min_position = raw
            try:
                self.target_min_position = int(raw if raw is not None else "8")
            except ValueError as exc:
                self.target_min_position = 8
                self.min_position_error = str(exc)
            self.sources["target_min_position"] = (
                alias if alias in os.environ and raw is not None else "default"
            )
        else:
            self.sources["target_min_position"] = "typed"

    def resolve(self, *, required=False) -> "SpecDecodeTraceConfig":
        if self.min_position_error is not None and (self.target_logits or required):
            raise ValueError(self.min_position_error)
        return self

    def compile_ignored_aliases(self):
        return set(self.legacy_fields.values())


def capture_runtime_trace():
    """Borrow the engine diagnostic owner during initialization or a forward."""
    from vllm.runtime_resources import current_runtime_resources

    resources = current_runtime_resources()
    if resources is not None and resources.get("runtime_trace") is not None:
        return resources["runtime_trace"]
    return _standalone_runtime_trace()


@torch.compiler.assume_constant_result
def _standalone_runtime_trace():
    return RuntimeTraceConfig()


def capture_spec_decode_trace():
    from vllm.runtime_resources import current_runtime_resources

    resources = current_runtime_resources()
    if resources is not None and resources.get("spec_decode_trace") is not None:
        return resources["spec_decode_trace"]
    return SpecDecodeTraceConfig()


def target_trace_min_position() -> int:
    return capture_spec_decode_trace().resolve(required=True).target_min_position
