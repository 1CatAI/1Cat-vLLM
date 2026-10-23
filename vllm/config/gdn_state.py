# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialized state-routing policy; tensor state is owned by the worker."""

import os
from dataclasses import asdict
from typing import ClassVar

from pydantic import Field

from vllm.config.utils import config

GDN_STATE_FIELDS = {
    "spec_core": "VLLM_SM70_QWEN_GDN_SPEC_CORE_OP",
    "shared_mtp_metadata": "VLLM_SM70_MTP4_SHARED_GDN_METADATA",
    "fused_mtp_metadata": "VLLM_SM70_MTP4_FUSED_GDN_METADATA",
    "legacy_mixed_decode_routing": "VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING",
    "legacy_non_spec_slot0": "VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0",
}


@config
class GdnStateConfig:
    spec_core: bool | None = None
    """Retain the explicit speculative-core graph boundary."""
    shared_mtp_metadata: bool | None = None
    """Share native MTP request classification across cache groups."""
    fused_mtp_metadata: bool | None = None
    """Prepare admitted MTP state rows with the existing grouped kernel."""
    legacy_mixed_decode_routing: bool | None = None
    """Retained legacy classification of mixed decode batches."""
    legacy_non_spec_slot0: bool | None = None
    """Retained non-speculative committed-slot compatibility mode."""
    resolved: bool = Field(default=False, init=False)
    """Worker serialization retains captured decisions."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance, excluded from computation hashes."""

    def resolve(self) -> None:
        if self.resolved:
            return
        from vllm import envs

        for field, name in GDN_STATE_FIELDS.items():
            if getattr(self, field) is None:
                setattr(self, field, envs.environment_variables[name]())
                self.sources[field] = name if name in os.environ else "default"
            else:
                self.sources[field] = "typed"
        self.resolved = True

    def graph_options(self) -> dict:
        return {
            k: v for k, v in asdict(self).items() if k not in {"sources", "resolved"}
        }


@config
class GdnStateTraceConfig:
    aliases: ClassVar[dict[str, str]] = {
        "table_dir": "VLLM_SM70_DUMP_GDN_STATE_TABLE_DIR",
        "table_seqs": "VLLM_SM70_DUMP_GDN_STATE_TABLE_SEQS",
        "table_start": "VLLM_SM70_DUMP_GDN_STATE_TABLE_START_SEQ",
        "table_end": "VLLM_SM70_DUMP_GDN_STATE_TABLE_END_SEQ",
        "table_limit": "VLLM_SM70_DUMP_GDN_STATE_TABLE_MAX_DUMPS",
        "debug_state_table": "VLLM_DFLASH_DEBUG_STATE_TABLE",
        "metadata_profile": "VLLM_DFLASH_DDTREE_METADATA_PROFILE",
        "assert_standard_boundary": "VLLM_SM70_QWEN_GDN_ASSERT_NO_ACTIVE_SPEC_STANDARD",
        "assert_contract": "VLLM_SM70_GDN_STATE_CONTRACT_ASSERT",
        "metadata_shadow": "VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW",
        "sync_assert": "VLLM_SM70_DFLASH2_GDN_SYNC_ASSERT",
    }

    table_dir: str | None = None
    """Optional state-table dump directory."""
    table_seqs: str | None = None
    """Optional legacy sequence/range filter for state-table dumps."""
    table_start: int | None = None
    """Minimum sequence length to dump; zero disables this bound."""
    table_end: int | None = None
    """Maximum sequence length to dump; zero disables this bound."""
    table_limit: int | None = None
    """Maximum table dumps per engine process."""
    debug_state_table: bool | None = None
    """Retain DFlash's existing debug table observation point."""
    metadata_profile: bool | None = None
    """Retain metadata stage timing and log labels for the active builder."""
    assert_standard_boundary: bool | None = None
    """Reject active speculative rows reaching the standard core boundary."""
    assert_contract: bool | None = None
    """Check accepted counts, slots and align-mode state indices."""
    metadata_shadow: bool | None = None
    """Compare grouped state metadata with the original tensor preparation."""
    sync_assert: bool | None = None
    """Retain the optional device fence checking shared query ranges."""
    resolved: bool = Field(default=False, init=False)
    """Whether diagnostic compatibility inputs have been captured."""

    def resolve(self) -> None:
        if self.resolved:
            return
        from vllm import envs

        if self.table_dir is None:
            self.table_dir = os.getenv(self.aliases["table_dir"])
        if self.table_seqs is None and self.table_dir:
            self.table_seqs = os.getenv(self.aliases["table_seqs"])
        for field, name, default in (
            ("table_start", self.aliases["table_start"], 0),
            ("table_end", self.aliases["table_end"], 0),
            ("table_limit", self.aliases["table_limit"], 32),
        ):
            if getattr(self, field) is None:
                # Disabled diagnostics never parse unused numeric inputs.
                setattr(
                    self,
                    field,
                    int(os.getenv(name, str(default))) if self.table_dir else default,
                )
        if self.debug_state_table is None:
            self.debug_state_table = envs.environment_variables[
                self.aliases["debug_state_table"]
            ]()
        if self.metadata_profile is None:
            self.metadata_profile = (
                os.getenv(self.aliases["metadata_profile"], "0") == "1"
            )
        if self.assert_standard_boundary is None:
            self.assert_standard_boundary = (
                os.getenv(self.aliases["assert_standard_boundary"]) == "1"
            )
        if self.assert_contract is None:
            self.assert_contract = os.getenv(self.aliases["assert_contract"]) == "1"
        for field, name in (
            ("metadata_shadow", self.aliases["metadata_shadow"]),
            ("sync_assert", self.aliases["sync_assert"]),
        ):
            if getattr(self, field) is None:
                setattr(self, field, envs.environment_variables[name]())
        self.resolved = True

    def compile_ignored_aliases(self):
        return set(self.aliases.values())


def resolve_state_trace(vllm_config) -> GdnStateTraceConfig:
    observability = getattr(vllm_config, "observability_config", None)
    policy = getattr(observability, "gdn_state", None)
    if policy is None:
        policy = GdnStateTraceConfig()
    policy.resolve()
    return policy
