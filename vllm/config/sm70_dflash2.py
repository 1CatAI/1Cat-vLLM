# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 pipeline policy; operator admission remains beside each operator."""

import os

from pydantic import Field

from vllm import envs
from vllm.config.execution_policy import read_execution_legacy
from vllm.config.utils import config
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
    "VLLM_SM70_DFLASH2_FUSED_GDN_COMBINED_SPLIT": "fused_gdn_combined_split",
    "VLLM_SM70_DFLASH2_CONTEXT_PIPELINE": "context_pipeline",
    "VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH": "context_kv_graph",
    "VLLM_SM70_DFLASH2_QUANT_LM_HEAD": "quant_lm_head",
    "VLLM_SM70_DFLASH2_FP32_LOGITS": "fp32_logits",
    "VLLM_SM70_FP8_QPN8": "target_fp8_qpn8",
    "VLLM_SM70_DFLASH2_QPN8_RERANK": "qpn8_rerank",
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
            if (
                configured is not None
                and not self.sources.get(field, "").startswith("default:")
            ) or name in os.environ:
                explicit.append(field)
            if name in os.environ:
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
                    if qualified and name not in os.environ
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

    def graph_options(self) -> dict[str, bool | None]:
        return {
            field: getattr(self, field)
            for field in (
                *SM70_DFLASH2_LEGACY_FIELDS.values(),
                "draft_window_split",
                "bf16_emulation",
            )
        }


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
