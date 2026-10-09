# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-engine MoE policy and the sole adapter for migrated legacy switches."""

from typing import Any, Literal

from pydantic import Field

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

    def resolve(self, family: MoEFormat) -> None:
        import os

        from vllm import envs

        if family == "awq":
            raw = {
                "dump_buffers": ("VLLM_SM70_DUMP_AWQ_MOE_BUFFERS", None),
                "dump_dir": ("VLLM_SM70_DUMP_QWEN_LAYER_DIR", None),
                "dump_layers": ("VLLM_SM70_DUMP_QWEN_LAYER_IDS", "0,1"),
                "dump_labels": ("VLLM_SM70_DUMP_AWQ_MOE_LABELS", ""),
            }
            for field, (name, default) in raw.items():
                if getattr(self, field) is None:
                    value = os.getenv(name, default)
                    setattr(
                        self, field, value == "1" if field == "dump_buffers" else value
                    )
            registered = {
                "compare_dir": "DIR",
                "compare_enable_file": "ENABLE_FILE",
                "compare_layers": "LAYER_IDS",
                "compare_steps": "STEPS",
                "compare_max_reports": "MAX_REPORTS",
            }
            for field, suffix in registered.items():
                if getattr(self, field) is None:
                    setattr(
                        self,
                        field,
                        getattr(envs, "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_" + suffix),
                    )
        else:
            for field, suffix, default in (
                ("compact_compare", "COMPARE", "0"),
                ("compact_compare_reports", "COMPARE_REPORTS", "16"),
            ):
                if getattr(self, field) is None:
                    numeric_value = int(
                        os.getenv(
                            "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_" + suffix,
                            default,
                        )
                    )
                    setattr(
                        self,
                        field,
                        bool(numeric_value)
                        if field == "compact_compare"
                        else numeric_value,
                    )
            if self.strict_compare_fail is None:
                self.strict_compare_fail = (
                    envs.VLLM_SM70_FP8_MOE_COMPACT_STRICT_COMPARE_FAIL
                )


@config
class Sm70MoEFormatConfig:
    """None uses the original default/alias; explicit fields take precedence.

    Capability rejection stays in the selector. A requested indexed kernel
    that is absent therefore retains the old dense fallback, not a new error.
    """

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
        metadata = {"resolved", "sources", "explicit_fields", "diagnostics"}
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
        self.resolved = True

    def hash_options(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in vars(self).items()
            if name not in {"resolved", "sources", "explicit_fields", "diagnostics"}
        }


@config
class Sm70MoEConfig:
    """Resolve only loaded families; unused options do not salt graph caches."""

    awq: Sm70MoEFormatConfig = Field(default_factory=Sm70MoEFormatConfig)
    """AWQ options, captured only if an AWQ MoE layer is initialized."""
    fp8: Sm70MoEFormatConfig = Field(default_factory=Sm70MoEFormatConfig)
    """FP8 options, captured only if an FP8 MoE layer is initialized."""

    @property
    def resolved(self) -> bool:
        return self.awq.resolved or self.fp8.resolved

    def compute_hash(self) -> str:
        return hash_factors(
            {
                family: getattr(self, family).hash_options()
                for family in ("awq", "fp8")
                if getattr(self, family).resolved
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
