# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 pipeline policy; operator admission remains beside each operator."""

import os
from typing import ClassVar

from pydantic import Field

from vllm import envs
from vllm.config.execution_policy import ExecutionPolicy, read_execution_legacy
from vllm.config.utils import config
from vllm.envs_metadata import EnvVar
from vllm.logger import init_logger

logger = init_logger(__name__)

# Retain the qualified pipeline schedule, including FP32 logits and dense tie order.
SM70_DFLASH2_VERIFIER_DEFAULTS = {
    "VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY": "1",
    "VLLM_SM70_DFLASH2_FUSED_GDN_COMBINED_SPLIT": "1",
    "VLLM_SM70_DFLASH2_CONTEXT_PIPELINE": "1",
    "VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH": "1",
    "VLLM_SM70_DFLASH2_QUANT_LM_HEAD": "1",
    "VLLM_SM70_DFLASH2_FP32_LOGITS": "1",
    "VLLM_SM70_FP8_QPN8": "1",
    "VLLM_SM70_DFLASH2_QPN8_RERANK": "1",
    "VLLM_SM70_DFLASH2_VERIFY_FASTPATH": "1",
    "VLLM_SM70_DFLASH2_FUSED_GDN_METADATA": "1",
    "VLLM_SM70_DFLASH2_FUSED_GDN_NORM": "1",
    "VLLM_SM70_DFLASH2_FUSED_GDN_SPLIT": "1",
    "VLLM_SM70_DFLASH2_FUSED_GEMMA_RMS": "1",
    "VLLM_SM70_DFLASH2_FIXED_GEMMA_RMS": "1",
    "VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA": "1",
    "VLLM_SM70_DFLASH2_GROUPED_SMALLQ_METADATA": "1",
    "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION": "1",
    "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC": "1",
}

SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS = {
    "VLLM_SM70_DFLASH2_VERIFY_FASTPATH": "1",
    "VLLM_SM70_DFLASH2_FUSED_GDN_METADATA": "1",
    "VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA": "1",
    "VLLM_SM70_DFLASH2_GROUPED_SMALLQ_METADATA": "1",
    "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC": "1",
    "VLLM_SM70_DFLASH2_BF16_EMULATION": "1",
    "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE": "0.9",
    "VLLM_SM70_DFLASH2_PROPOSAL_TOP_P": "0.95",
    # The sparse target sampler is not part of the retained GLM quality route.
    "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION": "0",
    "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY": "1",
    "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_MIN_MODEL_LEN": "1",
    "VLLM_SM70_GLM53_TP8_CUBLASLT": "1",
    "VLLM_SM70_GLM53_TP8_FUSED_FG_B": "1",
    "VLLM_SM70_GLM53_MHC_NATIVE_VERIFY": "1",
    "VLLM_SM70_GLM53_MHC_FUSED_POST_DOT_Q8": "1",
    "VLLM_SM70_GLM_MHC_PRE_THREADS": "1024",
    "VLLM_SM70_GLM53_MOE_QPN_W13_Q8": "0",
    "VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS": "1",
    "VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR": "1",
    "VLLM_SM70_TP8_HIERARCHICAL_PUSH_AR": "1",
    "VLLM_USE_AOT_COMPILE": "0",
}


SM70_DFLASH2_LEGACY_FIELDS = {
    "VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY": "fused_gdn_verify",
    "VLLM_SM70_DFLASH2_TP2_GDN_BV2": "tp2_gdn_bv2",
    "VLLM_SM70_DFLASH2_FUSED_QKV_PACK": "fused_qkv_pack",
    "VLLM_SM70_DFLASH2_FUSED_GDN_COMBINED_SPLIT": "fused_gdn_combined_split",
    "VLLM_SM70_DFLASH2_CONTEXT_PIPELINE": "context_pipeline",
    "VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH": "context_kv_graph",
    "VLLM_SM70_DFLASH2_QUANT_LM_HEAD": "quant_lm_head",
    "VLLM_SM70_DFLASH2_FP32_LOGITS": "fp32_logits",
    "VLLM_SM70_FP8_QPN8": "target_fp8_qpn8",
    "VLLM_SM70_DFLASH2_QPN8_RERANK": "qpn8_rerank",
    "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW": "qpn8_rerank_shadow",
    "VLLM_SM70_DFLASH2_VERIFY_FASTPATH": "verify_fastpath",
    "VLLM_SM70_DFLASH2_FUSED_GDN_METADATA": "fused_gdn_metadata",
    "VLLM_SM70_DFLASH2_FUSED_GDN_NORM": "fused_gdn_norm",
    "VLLM_SM70_DFLASH2_FUSED_GDN_SPLIT": "fused_gdn_split",
    "VLLM_SM70_DFLASH2_FUSED_GEMMA_RMS": "fused_gemma_rms",
    "VLLM_SM70_DFLASH2_FIXED_GEMMA_RMS": "fixed_gemma_rms",
    "VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA": "fused_smallq_metadata",
    "VLLM_SM70_DFLASH2_GROUPED_SMALLQ_METADATA": "grouped_smallq_metadata",
    "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION": "sparse_target_rejection",
    "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC": "sharded_context_fc",
}


@config
class DFlashLookupPolicy(ExecutionPolicy):
    """Captured only when lookup assistance can run; does not own request state."""

    adaptive: bool | None = None
    """Adapt lookup width using the retained acceptance policy."""

    nstrong: int | None = Field(default=None, ge=1)
    """Strong-match threshold; legacy minimum one."""

    agree: int | None = Field(default=None, ge=0)
    """Required agreement count; legacy minimum zero."""

    nmin_tail: int | None = Field(default=None, ge=1)
    """Minimum tail match; legacy minimum one."""

    long_min: int | None = Field(default=None, ge=1)
    """Minimum long-context match; legacy minimum one."""

    search: int | None = Field(default=None, ge=1)
    """Maximum lookup search length; legacy minimum one."""

    entry_streak: int | None = Field(default=None, ge=1)
    """Accepted streak needed to enter long lookup mode."""

    sticky: int | None = Field(default=None, ge=0)
    """Number of sticky lookup steps; legacy minimum zero."""

    cheap_context: int | None = Field(default=None, ge=0)
    """Context threshold for the existing cheap lookup mode."""

    aliases: ClassVar[dict[str, str]] = {
        "adaptive": "VLLM_DFLASH2_LOOKUP_ADAPTIVE",
        "nstrong": "VLLM_DFLASH2_LOOKUP_NSTRONG",
        "agree": "VLLM_DFLASH2_LOOKUP_AGREE",
        "nmin_tail": "VLLM_DFLASH2_LOOKUP_NMIN_TAIL",
        "long_min": "VLLM_DFLASH2_LOOKUP_LONG_MIN",
        "search": "VLLM_DFLASH2_LOOKUP_SEARCH",
        "entry_streak": "VLLM_DFLASH2_LOOKUP_ENTRY_STREAK",
        "sticky": "VLLM_DFLASH2_LOOKUP_STICKY",
        "cheap_context": "VLLM_DFLASH2_LOOKUP_CHEAP_CONTEXT",
    }


@config
class Sm70DFlash2Config:
    """Per-engine verifier decisions. None selects the qualified model policy.

    Operator dtype/layout/shape/native checks still govern every dispatch.
    Explicit settings take precedence over legacy aliases during compatibility.
    """

    draft_window_split: bool = True
    """Use the qualified FP16 single-request split window on 832-token pages."""

    fused_gdn_verify: bool | None = None
    """Policy for fused gdn verify; None retains automatic qualification."""

    fused_gdn_combined_split: bool | None = None
    """Policy for fused gdn combined split; None retains automatic qualification."""

    context_pipeline: bool | None = None
    """Policy for context pipeline; None retains automatic qualification."""

    context_kv_graph: bool | None = None
    """Policy for context kv graph; None retains automatic qualification."""

    quant_lm_head: bool | None = None
    """Policy for quant lm head; None retains automatic qualification."""

    fp32_logits: bool | None = None
    """Policy for fp32 logits; None retains automatic qualification."""

    target_fp8_qpn8: bool | None = None
    """Policy for target fp8 qpn8; None retains automatic qualification."""

    qpn8_rerank: bool | None = None
    """Policy for qpn8 rerank; None retains automatic qualification."""

    qpn8_rerank_shadow: bool | None = None
    """Eager coverage audit returning dense logits; changes execution semantics."""

    verify_fastpath: bool | None = None
    """Policy for verify fastpath; None retains automatic qualification."""

    fused_gdn_metadata: bool | None = None
    """Policy for fused gdn metadata; None retains automatic qualification."""

    fused_gdn_norm: bool | None = None
    """Policy for fused gdn norm; None retains automatic qualification."""

    fused_gdn_split: bool | None = None
    """Policy for fused gdn split; None retains automatic qualification."""

    fused_gemma_rms: bool | None = None
    """Policy for fused gemma rms; None retains automatic qualification."""

    fixed_gemma_rms: bool | None = None
    """Policy for fixed gemma rms; None retains automatic qualification."""

    fused_smallq_metadata: bool | None = None
    """Policy for fused smallq metadata; None retains automatic qualification."""

    grouped_smallq_metadata: bool | None = None
    """Policy for grouped smallq metadata; None retains automatic qualification."""

    sparse_target_rejection: bool | None = None
    """Policy for sparse target rejection; None retains automatic qualification."""

    sharded_context_fc: bool | None = None
    """Policy for sharded context fc; None retains automatic qualification."""

    tp2_gdn_bv2: bool | None = None
    """Retain the TP2 GDN verifier tile candidate and its original admission."""
    fused_qkv_pack: bool | None = None
    """Retain the admitted post-convolution QKV pack candidate."""
    lookup: DFlashLookupPolicy = Field(default_factory=DFlashLookupPolicy)
    """Optional ngram lookup tuning; initialization binds enabled assistance."""

    bf16_emulation: bool | None = None
    """Preserve the draft's BF16 emulation contract on FP16-only devices."""
    proposal_temperature_scale: float | None = None
    """Multiplier for probabilistic proposal temperature; positive."""
    proposal_top_p: float | None = None
    """Nucleus probability for draft proposals, in (0, 1]."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization sources, excluded from graph options."""

    qualified: bool = Field(default=False, init=False, repr=False)
    """Whether the retained complete-model validation boundary matches."""

    resolved: bool = Field(default=False, init=False, repr=False)
    """Whether the per-engine policy has been resolved."""

    explicit_fields: tuple[str, ...] = Field(default=(), init=False, repr=False)
    """Explicit configuration or legacy settings, used by mixed-format defaults."""

    def resolve(self, *, qualified: bool) -> None:
        if "VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER" in os.environ:
            logger.warning_once(
                "VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER is deprecated and ignored: "
                "dense tie ordering is mandatory after retiring the failed "
                "candidate-order experiment. No replacement switch is needed. "
                "The alias remains for one full released compatibility cycle."
            )
        if self.resolved:
            return
        explicit = []
        for name, field in SM70_DFLASH2_LEGACY_FIELDS.items():
            configured = getattr(self, field)
            self.sources.setdefault(
                field,
                "typed"
                if configured is not None
                else name
                if name in os.environ
                else "qualified_model"
                if qualified and name in SM70_DFLASH2_VERIFIER_DEFAULTS
                else "default",
            )
            if (
                configured is not None
                and not self.sources.get(field, "").startswith("default:")
            ) or name in os.environ:
                explicit.append(field)
            if name in os.environ:
                variable = envs.environment_variables[name]
                if isinstance(variable, EnvVar) and variable.metadata.deprecated:
                    variable.warn_if_deprecated()
                else:
                    logger.warning_once(
                        "%s is deprecated; use speculative_config.sm70_dflash2.%s. "
                        "Explicit configuration takes precedence. The alias remains "
                        "for one full released compatibility cycle.",
                        name,
                        field,
                    )
            if configured is None:
                configured = (
                    bool(int(SM70_DFLASH2_VERIFIER_DEFAULTS[name]))
                    if qualified
                    and name in SM70_DFLASH2_VERIFIER_DEFAULTS
                    and name not in os.environ
                    else envs.environment_variables[name]()
                )
            setattr(self, field, configured)
        from vllm.config.sm70_runtime import resolve_legacy_fields

        resolve_legacy_fields(
            self,
            {
                "bf16_emulation": "VLLM_SM70_DFLASH2_BF16_EMULATION",
                "proposal_temperature_scale": (
                    "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE"
                ),
                "proposal_top_p": "VLLM_SM70_DFLASH2_PROPOSAL_TOP_P",
            },
            reader=read_execution_legacy,
        )
        self.explicit_fields = tuple(explicit)
        self.qualified = qualified
        self.resolved = True

    def resolve_lookup(self, spec) -> None:
        from vllm.config.speculative import get_dflash_model_draft_tokens

        if spec.ngram_assist and (
            get_dflash_model_draft_tokens(spec) < spec.num_speculative_tokens
        ):
            self.lookup.resolve()

    def native_overrides(self) -> dict[str, bool | None]:
        """Bridge explicit/model defaults to B's FP16 native policy ABI.

        Unchanged legacy native inputs retain their own historical parser.
        """
        return {
            alias: getattr(self, field)
            for field, alias in (
                ("qpn8_rerank", "VLLM_SM70_DFLASH2_QPN8_RERANK"),
                ("qpn8_rerank_shadow", "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW"),
            )
            if self.sources.get(field) in ("typed", "qualified_model")
            or self.sources.get(field, "").startswith("default:")
        }

    def graph_options(self) -> dict:
        result = {
            field: getattr(self, field)
            for field in (
                *SM70_DFLASH2_LEGACY_FIELDS.values(),
                "draft_window_split",
                "bf16_emulation",
            )
        }

        if self.lookup.sources:
            result["lookup"] = {
                field: getattr(self.lookup, field) for field in self.lookup.aliases
            }
        return result


def capture_sm70_dflash2_config(vllm_config=None) -> Sm70DFlash2Config | None:
    """Capture during initialization, then pass the object with its owning layer."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    spec = getattr(vllm_config, "speculative_config", None)
    return getattr(spec, "sm70_dflash2", None)


def sm70_dflash2_enabled(field: str, policy: Sm70DFlash2Config | None) -> bool:
    if policy is not None and policy.resolved:
        return bool(getattr(policy, field))
    # Direct operator tests and unconfigured callers retain the legacy contract.
    name = next(
        name for name, value in SM70_DFLASH2_LEGACY_FIELDS.items() if value == field
    )
    return bool(getattr(envs, name))


def dflash2_bf16_emulation(policy: Sm70DFlash2Config | None) -> bool:
    if policy is not None and policy.resolved:
        return bool(policy.bf16_emulation)
    return read_execution_legacy("VLLM_SM70_DFLASH2_BF16_EMULATION")


def resolved_sm70_dflash2_config():
    """Bind a standalone policy once when a layer has no speculative owner."""
    policy = capture_sm70_dflash2_config()
    if policy is None:
        policy = Sm70DFlash2Config()
        policy.resolve(qualified=False)
    return policy


@config
class DFlashDiagnosticsConfig(ExecutionPolicy):
    proposal_stages: bool | None = None
    """Retain GLM proposal/target nonfinite stage observations."""
    target_layer_trace: bool | None = None
    """Retain armed GLM target-layer trace observations per engine."""

    corruption: bool | None = None
    """Existing family-specific corruption diagnostic admission."""
    draft_logits: bool | None = None
    """Existing family-specific draft-logit dumps."""
    profile: bool | None = None
    """Existing lookup timing observation points."""
    interval: int | None = None
    """Existing lookup timing log interval."""

    errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Captured parse failures; unused family controls retain short-circuiting."""

    aliases: ClassVar[dict[str, str]] = {
        "proposal_stages": "VLLM_DFLASH_DEBUG_PROPOSAL_STAGES",
        "target_layer_trace": "VLLM_DFLASH_DEBUG_TARGET_LAYER_TRACE",
        "corruption": "VLLM_DFLASH_DEBUG_CORRUPTION",
        "draft_logits": "VLLM_DFLASH_DUMP_DRAFT_LOGITS",
        "profile": "VLLM_DFLASH_PROFILE",
        "interval": "VLLM_DFLASH_PROFILE_LOG_INTERVAL",
    }

    def __post_init__(self) -> None:
        from vllm.config.sm70_runtime import resolve_legacy_fields

        pending = {
            field: alias
            for field, alias in self.aliases.items()
            if field not in self.sources
        }

        def read(name):
            try:
                return envs.environment_variables[name]()
            except ValueError as exc:
                field = next(
                    field for field, alias in self.aliases.items() if alias == name
                )
                self.errors[field] = str(exc)
                return 32 if field == "interval" else False

        resolve_legacy_fields(self, pending, reader=read)

    def value(self, field: str):
        if field in self.errors:
            raise ValueError(self.errors[field])
        return getattr(self, field)


def proposer_diagnostic_flag(method: str, field: str, trace=None) -> bool:
    """Retain family-before-generic precedence and its short-circuit parser."""
    family = method in ("dflash", "dflash_ddtree", "dspark")
    if trace is None:
        from vllm.config.diagnostic_sampling import SamplingDiagnosticsConfig

        return bool(
            family
            and envs.environment_variables[DFlashDiagnosticsConfig.aliases[field]]()
            or envs.environment_variables[SamplingDiagnosticsConfig.aliases[field]]()
        )
    return bool(family and trace.dflash.value(field) or trace.sampling.value(field))


def proposer_diagnostic_flags(method: str, trace) -> tuple[bool, bool]:
    """Bind the historical family qualification once at proposer initialization."""
    return (
        proposer_diagnostic_flag(method, "corruption", trace),
        proposer_diagnostic_flag(method, "draft_logits", trace),
    )
