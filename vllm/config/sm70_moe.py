# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-engine MoE policy and the sole adapter for migrated legacy switches."""

from typing import Any, ClassVar, Literal

from pydantic import Field

from vllm.config.diagnostic_dump import TensorDumpConfig
from vllm.config.execution_policy_base import ExecutionPolicy
from vllm.config.sm70_native import Sm70NativeConfig
from vllm.config.utils import config, hash_factors

MoEFormat = Literal["awq", "fp8"]
W13Mode = Literal["dense", "indexed", "compact"]
W2Mode = Literal["dense", "indexed"]
ReduceMode = Literal["unpermute", "weighted"]

# These are independent decisions; historical combined flags are normalized
# below to modes rather than carried into every execution branch.
COMMON_ALIASES = {
    "single_token_w13": (
        "VLLM_SM70_MOE_SINGLE_TOKEN_COMPACT_W13_FASTPATH",
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH",
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W13_FASTPATH",
    ),
    "single_token_w2": (
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH",
        "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH",
    ),
    "single_token_reduce": (
        "VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH",
        "VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH",
    ),
}
ALIASES = {
    "awq": {
        "batched": "VLLM_SM70_AWQ_MOE_BATCHED_GEMM",
        "qpn_m1": "VLLM_SM70_AWQ_QWEN38_QPN_M1",
        "strict_w13": "VLLM_SM70_AWQ_MOE_BATCHED_SINGLE_TOKEN_DENSE_W13",
        "exact_w2": "VLLM_SM70_AWQ_MOE_BATCHED_EXACT_W2",
        "active_exact_w2": "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2",
        "max_batched_tokens": "VLLM_SM70_AWQ_MOE_BATCHED_DECODE_MAX_TOKENS",
        "legacy_compact": "VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT",
        "persistent_tokens": "VLLM_SM70_AWQ_MOE_PERSISTENT_MAX_TOKENS",
        "compact_metadata": "VLLM_SM70_AWQ_MOE_COMPACT_METADATA",
        "active_grouped_decode": "VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE",
        "indexed_prefill": "VLLM_SM70_AWQ_QWEN38_MOE_INDEXED_PREFILL",
        "w2_chunk_tokens": "VLLM_SM70_AWQ_QWEN38_MOE_W2_CHUNK_TOKENS",
        "layer_allowlist": "VLLM_SM70_AWQ_MOE_BATCHED_LAYER_ALLOWLIST",
        "layer_denylist": "VLLM_SM70_AWQ_MOE_BATCHED_LAYER_DENYLIST",
    },
    "fp8": {
        "batched": "VLLM_SM70_FP8_MOE_BATCHED_GEMM",
        "w13_per_expert": "VLLM_SM70_FP8_MOE_BATCHED_W13_PER_EXPERT_DISPATCH",
        "w2_per_expert": "VLLM_SM70_FP8_MOE_BATCHED_W2_PER_EXPERT_DISPATCH",
        "legacy_compact": "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT",
        "permute_scratch": "VLLM_SM70_FP8_MOE_PERMUTE_WITH_SCRATCH",
    },
}


# These aliases are also consumed below the current native ABI. An explicit
# conflicting Python request must not silently execute a different native
# algorithm. Delivery 3 replaces this compatibility guard with native policy
# arguments; keep it until that ABI is available, without process-env mutation.
NATIVE_LEGACY_ALIASES = frozenset(
    {
        "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2",
        "VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE",
        "VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH",
        "VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH",
        "VLLM_SM70_MXFP4_MOE_GROUPED_M8",
        "VLLM_SM70_MXFP4_MOE_GROUPED_M8_EXPERT_ROWS",
        "VLLM_SM70_MXFP4_MOE_GROUPED_VERIFIER",
        "VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS",
        "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL",
        "VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL",
    }
)


def _validate_native_legacy_request(policy, field: str, name: str) -> None:
    from vllm import envs
    from vllm._sm70.policy import native_policy_abi_available

    if native_policy_abi_available():
        return
    if name in NATIVE_LEGACY_ALIASES and getattr(policy, field) != getattr(envs, name):
        raise ValueError(
            f"sm70_moe.{field} conflicts with {name}: the current native ABI "
            "also consumes this legacy switch. Use matching values until the "
            "native policy-argument ABI is installed."
        )


# Diagnostics share one compatibility declaration with compile-cache filtering.
AWQ_DUMP_ALIASES = {
    "dump_buffers": ("VLLM_SM70_DUMP_AWQ_MOE_BUFFERS", None),
    "dump_dir": ("VLLM_SM70_DUMP_QWEN_LAYER_DIR", None),
    "dump_layers": ("VLLM_SM70_DUMP_QWEN_LAYER_IDS", "0,1"),
    "dump_labels": ("VLLM_SM70_DUMP_AWQ_MOE_LABELS", ""),
}
AWQ_COMPARE_ALIASES = {
    "compare_dir": "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_DIR",
    "compare_enable_file": "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_ENABLE_FILE",
    "compare_layers": "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_LAYER_IDS",
    "compare_steps": "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_STEPS",
    "compare_max_reports": "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_MAX_REPORTS",
}
FP8_COMPARE_ALIASES = {
    "compact_compare": ("VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE", "0"),
    "compact_compare_reports": (
        "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE_REPORTS",
        "16",
    ),
    "strict_compare_fail": ("VLLM_SM70_FP8_MOE_COMPACT_STRICT_COMPARE_FAIL", "0"),
}


@config
class Sm70MoEDiagnostics:
    """Observations do not change numerical policy or the graph fingerprint."""

    dump_buffers: bool | None = None
    """Enable the historical AWQ buffer observer."""
    dump_dir: str | None = None
    """Destination for the existing Qwen layer dump operator."""
    dump_layers: str | None = None
    """Comma-separated layer IDs/ranges, or all, for buffer dumps."""
    dump_labels: str | None = None
    """Optional comma-separated historical AWQ observation labels."""
    compare_dir: str | None = None
    """Directory for AWQ dense-reference JSONL records."""
    compare_enable_file: str | None = None
    """Optional live enable-file gate for AWQ comparisons."""
    compare_layers: str | None = None
    """Layer selection for the AWQ dense-reference observer."""
    compare_steps: str | None = None
    """Decode-step selection for the AWQ dense-reference observer."""
    compare_max_reports: int | None = None
    """Maximum AWQ records per layer; nonpositive is unlimited."""
    compact_compare: bool | None = None
    """Enable the original FP8 compact reference comparison."""
    compact_compare_reports: int | None = None
    """Maximum FP8 compact comparison reports."""
    strict_compare_fail: bool | None = None
    """Retained FP8 legacy no-op warning; never changes arithmetic."""

    dump_policy: TensorDumpConfig | None = Field(default=None, init=False)
    """Canonical observer projection; compatibility fields above retain old access."""
    compare_policy: TensorDumpConfig | None = Field(default=None, init=False)
    """Canonical comparison policy shared with the engine diagnostic owner."""

    def bind(self, family: MoEFormat, dumps) -> None:
        bindings = MOE_DIAGNOSTIC_BINDINGS[family]
        for old, (channel, field) in bindings.items():
            setattr(self, old, getattr(getattr(dumps, channel), field))
        self.dump_policy = dumps.awq_buffers if family == "awq" else None
        self.compare_policy = getattr(dumps, f"{family}_compare")

    def resolve(self, family: MoEFormat) -> None:
        if self.compare_policy is None:
            # Independent no-engine compatibility entry, resolved once per owner.
            from vllm.config.diagnostic_dump import TensorDiagnosticsConfig

            dumps = TensorDiagnosticsConfig()
            _project_moe_diagnostic_inputs(self, family, dumps)
            dumps.project_shared_fields()
            self.bind(family, dumps)
        # Preserve the qualified format initialization error checkpoint.
        fields = (
            ("max_dumps",)
            if family == "awq"
            else ("enabled", "max_dumps", "strict_fail")
        )
        assert self.compare_policy is not None
        for field in fields:
            self.compare_policy.value(field)


MOE_DIAGNOSTIC_BINDINGS = {
    "awq": {
        "dump_buffers": ("awq_buffers", "enabled"),
        "dump_dir": ("qwen_layer", "directory"),
        "dump_layers": ("qwen_layer", "layers"),
        "dump_labels": ("awq_buffers", "labels"),
        "compare_dir": ("awq_compare", "directory"),
        "compare_enable_file": ("awq_compare", "enable_file"),
        "compare_layers": ("awq_compare", "layers"),
        "compare_steps": ("awq_compare", "steps"),
        "compare_max_reports": ("awq_compare", "max_dumps"),
    },
    "fp8": {
        "compact_compare": ("fp8_compare", "enabled"),
        "compact_compare_reports": ("fp8_compare", "max_dumps"),
        "strict_compare_fail": ("fp8_compare", "strict_fail"),
    },
}


def _project_moe_diagnostic_inputs(legacy, family, dumps) -> None:
    if legacy.compare_policy is not None:
        return
    for old, (channel, field) in MOE_DIAGNOSTIC_BINDINGS[family].items():
        value = getattr(legacy, old)
        policy = getattr(dumps, channel)
        if value is not None and policy.sources.get(field) != "typed":
            setattr(policy, field, value)
            policy.sources[field] = f"kernel_config.sm70_moe.{family}.diagnostics.{old}"
            policy.filter_errors.pop(field, None)
            policy.parse_filters(channel)


def bind_moe_diagnostics(kernel, trace) -> None:
    """Resolve B typed aliases before workers, with observability overrides first."""
    for family in ("awq", "fp8"):
        _project_moe_diagnostic_inputs(
            getattr(kernel.sm70_moe, family).diagnostics, family, trace.dumps
        )
    trace.dumps.project_shared_fields()
    for family in ("awq", "fp8"):
        getattr(kernel.sm70_moe, family).diagnostics.bind(family, trace.dumps)


@config
class Sm70MoEFormatConfig:
    """None uses the original default/alias; explicit fields take precedence.

    Capability rejection stays in the selector. A requested indexed kernel
    that is absent therefore retains the old dense fallback, not a new error.
    """

    native: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """Captured native selector and tuning policy, shared with kernel bindings."""
    qpn_m1: bool | None = None
    """AWQ native-g32 Qwen QPN M1 request; original strict admission applies."""
    batched: bool | None = None
    """Request the existing grouped TurboMind route."""
    single_token_w13: tuple[W13Mode, ...] | None = None
    """Candidate set normalized to compact, indexed, dense priority."""
    single_token_w2: W2Mode | None = None
    """Indexed or dense active-expert W2 request."""
    single_token_reduce: ReduceMode | None = None
    """Weighted native reduction or original unpermute fallback."""
    legacy_compact: bool | None = None
    """Retain the separate monolithic single-token experiment."""
    w13_per_expert: bool | None = None
    """FP8 grouped W13 uses the per-expert dispatch binding."""
    w2_per_expert: bool | None = None
    """FP8 grouped W2 uses the per-expert dispatch binding."""
    permute_scratch: bool | None = None
    """FP8 permutation uses the existing caller-owned scratch route."""
    strict_w13: bool | None = None
    """AWQ strict dense-stage override with legacy single-token precedence."""
    exact_w2: bool | None = None
    """AWQ grouped W13 with full dense W2."""
    active_exact_w2: bool | None = None
    """AWQ active W2 up to 128 slots, then full dense fallback."""
    max_batched_tokens: int | None = None
    """AWQ grouped decode ceiling; nonpositive leaves it unbounded."""
    persistent_tokens: int | None = None
    """AWQ resident capacity request, bounded by the old ceiling."""
    compact_metadata: bool | None = None
    """AWQ compact scale/zero preparation-layout request."""
    active_grouped_decode: bool | None = None
    """AWQ qualified Qwen3.8 M2..8 grouped-active request."""
    indexed_prefill: bool | None = None
    """AWQ qualified indexed-input W13 prefill request."""
    w2_chunk_tokens: int | None = None
    """AWQ indexed-prefill W2 chunk size; 0, 4096 or 6144."""
    layer_allowlist: str | None = None
    """AWQ grouped layer allowlist; None retains all layers."""
    layer_denylist: str | None = None
    """AWQ grouped layer denylist, applied after the allowlist."""
    compact_exact_layout: bool | None = None
    """FP8 legacy compact exact-layout variant."""
    compact_native_unpermute: bool | None = None
    """FP8 legacy compact native-unpermute variant."""
    compact_decomposed: bool | None = None
    """FP8 legacy compact decomposed reference variant."""
    diagnostics: Sm70MoEDiagnostics = Field(default_factory=Sm70MoEDiagnostics)
    """Observation-only options excluded from calculation hashes."""
    resolved: bool = Field(default=False, init=False)
    """Whether this engine has captured its legacy compatibility inputs."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Field-to-source attribution; excluded from calculation hashes."""
    explicit_fields: tuple[str, ...] = Field(default=(), init=False)
    """Explicit requests, retained for the existing fail-closed gates."""

    def resolve(self, family: MoEFormat) -> None:
        from vllm import envs

        if self.resolved:
            return
        supported = set(ALIASES[family]) | set(COMMON_ALIASES)
        if family == "fp8":
            supported |= {
                "compact_exact_layout",
                "compact_native_unpermute",
                "compact_decomposed",
            }
        metadata = {"resolved", "sources", "explicit_fields", "diagnostics", "native"}
        for field, supplied_value in vars(self).items():
            if field not in supported | metadata and supplied_value is not None:
                raise ValueError(f"sm70_moe.{family}.{field} is not supported")
        self.diagnostics.resolve(family)
        if family == "fp8":
            import os

            for field, suffix, default in (
                ("compact_exact_layout", "EXACT_LAYOUT", "1"),
                ("compact_native_unpermute", "NATIVE_UNPERMUTE", "0"),
                ("compact_decomposed", "DECOMPOSED", "0"),
            ):
                if getattr(self, field) is None:
                    name = "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_" + suffix
                    setattr(self, field, bool(int(os.getenv(name, default))))
                    self.sources[field] = name if name in os.environ else "default"
                else:
                    self.sources[field] = "configuration"
        explicit = []
        for field, name in ALIASES[family].items():
            if getattr(self, field) is not None:
                _validate_native_legacy_request(self, field, name)
                self.sources[field] = "configuration"
                explicit.append(field)
            else:
                setattr(self, field, getattr(envs, name))
                self.sources[field] = name if envs.is_set(name) else "default"
                if envs.is_set(name):
                    explicit.append(field)

        for field, names in COMMON_ALIASES.items():
            if getattr(self, field) is not None:
                self.sources[field] = "configuration"
                explicit.append(field)
                continue
            if field == "single_token_w13":
                value: Any = tuple(
                    mode
                    for mode, requested in (
                        ("compact", getattr(envs, names[0])),
                        ("indexed", any(getattr(envs, name) for name in names[1:])),
                        ("dense", True),
                    )
                    if requested
                )
            elif field == "single_token_w2":
                if family == "fp8":
                    names = (
                        "VLLM_SM70_FP8_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH",
                        *names,
                    )
                value = (
                    "indexed" if any(getattr(envs, name) for name in names) else "dense"
                )
            else:
                value = (
                    "weighted"
                    if any(getattr(envs, name) for name in names)
                    else "unpermute"
                )
            setattr(self, field, value)
            enabled_aliases = [name for name in names if envs.is_set(name)]
            self.sources[field] = ",".join(enabled_aliases) or "default"
            if enabled_aliases:
                explicit.append(field)

        defaults = {
            "strict_w13": False,
            "exact_w2": False,
            "active_exact_w2": False,
            "max_batched_tokens": 0,
            "persistent_tokens": 32,
            "w13_per_expert": family == "awq",
            "w2_per_expert": family == "awq",
            "permute_scratch": True,
            "compact_metadata": False,
            "active_grouped_decode": False,
            "indexed_prefill": False,
            "w2_chunk_tokens": 0,
        }
        for field, default_value in defaults.items():
            if getattr(self, field) is None:
                setattr(self, field, default_value)
        if self.single_token_w13 is not None:
            if not self.single_token_w13 or len(set(self.single_token_w13)) != len(
                self.single_token_w13
            ):
                raise ValueError(
                    "single_token_w13 must be nonempty and have no duplicates"
                )
            priority: tuple[W13Mode, ...] = ("compact", "indexed", "dense")
            self.single_token_w13 = tuple(
                mode for mode in priority if mode in self.single_token_w13
            )
        self.explicit_fields = tuple(explicit)
        overrides = {
            alias: getattr(self, field)
            for field, alias in ALIASES[family].items()
            if self.sources.get(field) == "configuration"
        }
        if self.sources.get("single_token_reduce") == "configuration":
            overrides["VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH"] = (
                self.single_token_reduce == "weighted"
            )
        self.native.resolve(family, overrides)
        self.resolved = True

    def hash_options(self) -> dict[str, Any]:
        return {
            name: (self.native.hash_options() if name == "native" else value)
            for name, value in vars(self).items()
            if name not in {"resolved", "sources", "explicit_fields", "diagnostics"}
        }


NVFP4_ALIASES = {
    "qpn_m1": "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE",
    "qpn_batch": "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE",
    "qpn_dynamic": "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE",
    "qpn_mtp5": "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE",
    "fused_batch_w13": "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W13",
    "fused_batch_w2": "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2",
    "w2_direct_reduce": "VLLM_SM70_NVFP4_QWEN38_MOE_W2_DIRECT_REDUCE",
    "indexed_prefill": "VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL",
    "fused_swiglu_prefill": "VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL",
    "fast_prefill": "VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL",
    "raw_scale": "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE",
    "grouped_prefill": "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL",
    "grouped_decode": "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE",
    "grouped_mtp5": "VLLM_SM70_NVFP4_MOE_GROUPED_MTP5",
    "grouped_expert_rows": "VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS",
    "glm53_fused_permute": "VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8",
    "glm53_qpn_w13": "VLLM_SM70_GLM53_MOE_QPN_W13_Q8",
}

MXFP4_ALIASES = {
    "qpn_m1": "VLLM_SM70_MXFP4_MOE_QPN_M1_DECODE",
    "direct_top6": "VLLM_SM70_MXFP4_MOE_DIRECT_TOP6_DECODE",
    "direct_order": "VLLM_SM70_MXFP4_MOE_DIRECT_ORDER_DECODE",
    "active_experts": "VLLM_SM70_MXFP4_MOE_ACTIVE_EXPERT_B1",
    "active_expert_max_tokens": "VLLM_SM70_MXFP4_MOE_ACTIVE_EXPERT_MAX_TOKENS",
    "grouped_m8": "VLLM_SM70_MXFP4_MOE_GROUPED_M8",
    "grouped_verifier": "VLLM_SM70_MXFP4_MOE_GROUPED_VERIFIER",
    "grouped_expert_rows": "VLLM_SM70_MXFP4_MOE_GROUPED_M8_EXPERT_ROWS",
    "single_token_fastpath": "VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH",
    "single_token_permute": "VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH",
}


@config
class Sm70MoELegacyConfig:
    """Initialization-only alias adapter shared by the native FP4 formats."""

    native: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """Captured native selector and tuning policy, shared with kernel bindings."""
    resolved: bool = Field(default=False, init=False)
    """Whether this engine has captured compatibility inputs."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Origin of each value, excluded from graph hashes."""
    explicit_fields: tuple[str, ...] = Field(default=(), init=False)
    """Explicit requests preserve the old missing-operator error behavior."""

    def _resolve(self, aliases: dict[str, str], family: str) -> None:
        from vllm import envs

        if self.resolved:
            return
        explicit = []
        for field, name in aliases.items():
            if getattr(self, field) is not None:
                _validate_native_legacy_request(self, field, name)
                self.sources[field] = "configuration"
                explicit.append(field)
            else:
                setattr(self, field, getattr(envs, name))
                self.sources[field] = name if envs.is_set(name) else "default"
                if envs.is_set(name):
                    explicit.append(field)
        self.explicit_fields = tuple(explicit)
        self.native.resolve(
            family,
            {
                alias: getattr(self, field)
                for field, alias in aliases.items()
                if self.sources.get(field) == "configuration"
            },
        )
        self.resolved = True

    def hash_options(self) -> dict[str, Any]:
        return {
            name: (self.native.hash_options() if name == "native" else value)
            for name, value in vars(self).items()
            if name not in {"resolved", "sources", "explicit_fields", "route_debug"}
        }


@config
class Sm70NvFp4MoEConfig(Sm70MoELegacyConfig):
    """Native NVFP4 requests; selectors retain existing shape/capability gates."""

    qpn_m1: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE."""
    qpn_batch: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE."""
    qpn_dynamic: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE."""
    qpn_mtp5: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE."""
    fused_batch_w13: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W13."""
    fused_batch_w2: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2."""
    w2_direct_reduce: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_W2_DIRECT_REDUCE."""
    indexed_prefill: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL."""
    fused_swiglu_prefill: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL."""
    fast_prefill: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL."""
    raw_scale: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE."""
    grouped_prefill: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL."""
    grouped_decode: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_MOE_GROUPED_DECODE."""
    grouped_mtp5: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_MOE_GROUPED_MTP5."""
    grouped_expert_rows: bool | None = None
    """Initialization override for VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS."""
    glm53_fused_permute: bool | None = None
    """Initialization override for VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8."""
    glm53_qpn_w13: bool | None = None
    """Initialization override for VLLM_SM70_GLM53_MOE_QPN_W13_Q8."""
    route_debug: bool | None = None
    """Historical Qwen route diagnostic, excluded from the calculation hash."""

    def resolve(self) -> None:
        if self.route_debug is None:
            import os

            self.route_debug = os.getenv("VLLM_SM70_QWEN38_QPN_ROUTE_DEBUG") == "1"
        self._resolve(NVFP4_ALIASES, "nvfp4")


@config
class Sm70MxFp4MoEConfig(Sm70MoELegacyConfig):
    """Native MXFP4 requests; selectors retain existing shape/capability gates."""

    qpn_m1: bool | None = None
    """Initialization override for VLLM_SM70_MXFP4_MOE_QPN_M1_DECODE."""
    direct_top6: bool | None = None
    """Initialization override for VLLM_SM70_MXFP4_MOE_DIRECT_TOP6_DECODE."""
    direct_order: bool | None = None
    """Initialization override for VLLM_SM70_MXFP4_MOE_DIRECT_ORDER_DECODE."""
    active_experts: bool | None = None
    """Initialization override for VLLM_SM70_MXFP4_MOE_ACTIVE_EXPERT_B1."""
    active_expert_max_tokens: int | None = None
    """Initialization override for VLLM_SM70_MXFP4_MOE_ACTIVE_EXPERT_MAX_TOKENS."""
    grouped_m8: bool | None = None
    """Initialization override for VLLM_SM70_MXFP4_MOE_GROUPED_M8."""
    grouped_verifier: bool | None = None
    """Initialization override for VLLM_SM70_MXFP4_MOE_GROUPED_VERIFIER."""
    grouped_expert_rows: bool | None = None
    """Initialization override for VLLM_SM70_MXFP4_MOE_GROUPED_M8_EXPERT_ROWS."""
    single_token_fastpath: bool | None = None
    """Initialization override for VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH."""
    single_token_permute: bool | None = None
    """Initialization override for VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH."""

    def resolve(self) -> None:
        self._resolve(MXFP4_ALIASES, "mxfp4")


@config
class Sm70UnquantizedMoEConfig(ExecutionPolicy):
    """One initialized policy for warmup and unquantized execution."""

    legacy_tiles: bool | None = None
    """Retain the existing 0.0.3 SM70 tile selection."""
    functional: bool | None = None
    """Retain the functional expert implementation at its layout gates."""
    disable_inplace: bool | None = None
    """Keep the explicit override disabling in-place output."""
    mtp_tuned: bool | None = None
    """Retain exact-shape MTP tiles in warmup and execution."""
    mtp_fp16_exact: bool | None = None
    """Retain the exact FP16 MTP native provider."""

    aliases: ClassVar[dict[str, str]] = {
        "legacy_tiles": "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_CONFIG",
        "functional": "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_FUNCTIONAL",
        "disable_inplace": "VLLM_SM70_DISABLE_UNQUANTIZED_MOE_INPLACE",
        "mtp_tuned": "VLLM_SM70_MTP_MOE_TUNED_CONFIG",
        "mtp_fp16_exact": "VLLM_SM70_MTP_MOE_FP16_EXACT",
    }

    errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Captured parser failures; dynamic admission retains the original ordering."""

    def resolve(self):
        from vllm.config.utils import resolve_legacy_fields

        resolve_legacy_fields(
            self,
            {
                field: alias
                for field, alias in self.aliases.items()
                if field not in self.sources
            },
            deferred_errors=self.errors,
        )

    def value(self, field):
        if field in self.errors:
            raise ValueError(self.errors[field])
        return getattr(self, field)


@config
class Sm70MoEConfig:
    """Resolve only loaded families; unused options do not salt graph caches."""

    unquantized: Sm70UnquantizedMoEConfig = Field(
        default_factory=Sm70UnquantizedMoEConfig
    )
    """Shared policy for native/Triton unquantized execution and its warmup."""

    awq: Sm70MoEFormatConfig = Field(default_factory=Sm70MoEFormatConfig)
    """AWQ options, captured only if an AWQ MoE layer is initialized."""
    fp8: Sm70MoEFormatConfig = Field(default_factory=Sm70MoEFormatConfig)
    """FP8 options, captured only if an FP8 MoE layer is initialized."""

    nvfp4: Sm70NvFp4MoEConfig = Field(default_factory=Sm70NvFp4MoEConfig)
    """NVFP4 options captured only when a native NVFP4 layer initializes."""
    mxfp4: Sm70MxFp4MoEConfig = Field(default_factory=Sm70MxFp4MoEConfig)
    """MXFP4 options captured only when a native MXFP4 layer initializes."""

    @property
    def resolved(self) -> bool:
        return bool(self.unquantized.sources) or any(
            getattr(self, family).resolved
            for family in ("awq", "fp8", "nvfp4", "mxfp4")
        )

    def compute_hash(self) -> str:
        return hash_factors(
            {
                **{
                    family: getattr(self, family).hash_options()
                    for family in ("awq", "fp8", "nvfp4", "mxfp4")
                    if getattr(self, family).resolved
                },
                **(
                    {"unquantized": self.unquantized.compute_hash()}
                    if self.unquantized.active and self.unquantized.sources
                    else {}
                ),
            }
        )


def capture_sm70_moe_config(family: MoEFormat) -> Sm70MoEFormatConfig:
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    policy = (
        getattr(cfg.kernel_config.sm70_moe, family) if cfg else Sm70MoEFormatConfig()
    )
    policy.resolve(family)
    return policy


def capture_nvfp4_moe_config() -> Sm70NvFp4MoEConfig:
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    policy = cfg.kernel_config.sm70_moe.nvfp4 if cfg else Sm70NvFp4MoEConfig()
    policy.resolve()
    return policy


def capture_mxfp4_moe_config() -> Sm70MxFp4MoEConfig:
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    policy = cfg.kernel_config.sm70_moe.mxfp4 if cfg else Sm70MxFp4MoEConfig()
    policy.resolve()
    return policy


def unquantized_moe_policy(cfg=None) -> Sm70UnquantizedMoEConfig:
    from vllm.config.execution_policy import capture_execution_policy

    return capture_execution_policy(
        "kernel_config.sm70_moe.unquantized", Sm70UnquantizedMoEConfig, cfg
    )
