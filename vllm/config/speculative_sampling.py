# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-engine speculative sampling inputs; RNG and admission remain in callers."""

import os
from typing import ClassVar

from pydantic import Field

from vllm.config.execution_policy import ExecutionPolicy
from vllm.config.sm70_runtime import resolve_legacy_fields
from vllm.config.utils import config, hash_factors


@config
class SpeculativeSamplingPolicy(ExecutionPolicy):
    sync_accept_counts: bool | None = None
    """Synchronize acceptance counts at the retained async scheduling boundary."""

    token_matching: bool | None = None
    """Retain stochastic token matching with the existing safety checks."""

    combine_bonus: bool | None = None
    """Admit the existing combined bonus sampling path."""

    draft_temperature_scale: float | None = None
    """Temperature multiplier, validated at the original stochastic checkpoint."""

    draft_top_p_override: str | float | None = None
    """Raw optional nucleus override; empty retains the legacy skip semantics."""

    draft_apply_top_p: bool | None = None
    """Apply top-p alongside top-k on the dense proposal path."""

    draft_sparse_topk: bool | None = None
    """Admit the retained sparse top-k draft proposal."""

    legacy_qwen_step_idx: bool | None = None
    """Use the existing legacy Qwen speculative step index."""

    exact_draft_seq_lens_cpu: bool | None = None
    """Keep exact draft CPU sequence-length shadows at the original checkpoints."""

    ranking_path: str | None = None
    """Explicit ranking asset; model-qualified defaults remain in the adapter."""

    shortlist_size: int | None = None
    """Static shortlist size, preserving existing model qualification."""

    dynamic_tail_size: int | None = None
    """Requested dynamic tail; zero retains the adapter default behavior."""

    full_refresh_interval: int | None = None
    """Existing full-vocabulary refresh interval."""

    fused_proposal_enabled: bool | None = None
    """Retain the fused proposal candidate."""

    gpu_lru_enabled: bool | None = None
    """Retain device-managed draft tail replacement."""

    prefill_topk: int | None = None
    """Existing prefill candidate count."""

    dynamic_vocab_default: bool | None = None
    """Allow the existing model-qualified automatic vocabulary policy."""

    aliases: ClassVar[dict[str, str]] = {
        "sync_accept_counts": "VLLM_SM70_MTP_SYNC_ACCEPT_COUNTS",
        "token_matching": "VLLM_MTP_STOCHASTIC_TOKEN_MATCHING",
        "combine_bonus": "VLLM_SM70_REJECTION_COMBINE_BONUS",
        "draft_temperature_scale": "VLLM_SM70_MTP_PROB_DRAFT_TEMPERATURE_SCALE",
        "draft_top_p_override": "VLLM_SM70_MTP_PROB_DRAFT_TOP_P_OVERRIDE",
        "draft_apply_top_p": "VLLM_SM70_MTP_PROB_DRAFT_APPLY_TOP_P",
        "draft_sparse_topk": "VLLM_SM70_MTP_PROB_DRAFT_SPARSE_TOPK",
        "legacy_qwen_step_idx": "VLLM_SM70_MTP_LEGACY_QWEN_STEP_IDX",
        "exact_draft_seq_lens_cpu": "VLLM_SM70_MTP_EXACT_DRAFT_SEQ_LENS_CPU",
        "ranking_path": "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING",
        "shortlist_size": "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE",
        "dynamic_tail_size": "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE",
        "full_refresh_interval": (
            "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FULL_REFRESH_INTERVAL"
        ),
        "fused_proposal_enabled": "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FUSED_PROPOSAL",
        "gpu_lru_enabled": "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_GPU_LRU",
        "prefill_topk": "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_PREFILL_TOPK",
        "dynamic_vocab_default": "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT",
    }

    fused_apply_top_p: bool = Field(default=False, init=False)
    """The fused proposal historically uses the registered integer parser."""
    fused_apply_top_p_error: str | None = Field(default=None, init=False)
    """A legacy parse failure is raised only when the fused path consumes it."""

    def resolve_fields(self, names) -> None:
        from vllm import envs

        pending = {
            field: self.aliases[field] for field in names if field not in self.sources
        }
        if "draft_apply_top_p" in pending:
            if self.draft_apply_top_p is not None:
                self.fused_apply_top_p = self.draft_apply_top_p
            else:
                try:
                    self.fused_apply_top_p = envs.environment_variables[
                        pending["draft_apply_top_p"]
                    ]()
                except ValueError as exc:
                    self.fused_apply_top_p_error = str(exc)

        def reader(name):
            if name == "VLLM_MTP_STOCHASTIC_TOKEN_MATCHING":
                return os.getenv(name, "0") == "1"
            if name == "VLLM_SM70_REJECTION_COMBINE_BONUS":
                return os.getenv(name, "1") != "0"
            if name == "VLLM_SM70_MTP_PROB_DRAFT_APPLY_TOP_P":
                return os.getenv(name, "0") == "1"
            return envs.environment_variables[name]()

        resolve_legacy_fields(self, pending, reader=reader)

    def resolve(
        self, *, draft: bool = True, vocab: bool = True, vocab_default: bool = True
    ) -> None:
        names = {
            "sync_accept_counts",
            "token_matching",
            "combine_bonus",
            "dynamic_tail_size",
            "legacy_qwen_step_idx",
            "exact_draft_seq_lens_cpu",
        }
        if draft:
            names.update(field for field in self.aliases if field.startswith("draft_"))
        if vocab:
            names.update(
                field
                for field in self.aliases
                if field
                not in {
                    "sync_accept_counts",
                    "token_matching",
                    "combine_bonus",
                    "legacy_qwen_step_idx",
                    "exact_draft_seq_lens_cpu",
                }
                and not field.startswith("draft_")
            )
        if not vocab_default:
            names.discard("dynamic_vocab_default")
        self.resolve_fields(names)

    def use_fused_top_p(self) -> bool:
        if self.fused_apply_top_p_error is not None:
            raise ValueError(self.fused_apply_top_p_error)
        return self.fused_apply_top_p

    def compute_hash(self, *, draft: bool = True, vocab: bool = True) -> str:
        inactive: set[str] = set()
        if not draft:
            inactive.update(
                field for field in self.aliases if field.startswith("draft_")
            )
        if not vocab:
            inactive.update(
                (
                    "ranking_path",
                    "shortlist_size",
                    "full_refresh_interval",
                    "fused_proposal_enabled",
                    "gpu_lru_enabled",
                    "prefill_topk",
                    "dynamic_vocab_default",
                    "legacy_qwen_step_idx",
                )
            )
        options = {
            field: getattr(self, field)
            for field in self.aliases
            if field in self.sources and field not in inactive
        }
        if "draft_apply_top_p" in options:
            options.update(
                fused_apply_top_p=self.fused_apply_top_p,
                fused_apply_top_p_error=self.fused_apply_top_p_error,
            )
        return hash_factors(options)


def resolve_sampling_policy(spec=None, *, draft=False, vocab=False):
    """Initialization binding; standalone constructors retain their own snapshot."""
    policy = getattr(spec, "sampling_policy", None)
    if policy is None:
        policy = SpeculativeSamplingPolicy()
    policy.resolve(draft=draft, vocab=vocab)
    return policy
