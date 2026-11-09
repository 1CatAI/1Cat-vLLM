# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Captured sampler/proposer diagnostics shared by both runner generations."""

import os
from typing import ClassVar

from pydantic import Field

from vllm.config.diagnostic_dump import parse_int_filter
from vllm.config.utils import config


@config
class SamplingDiagnosticsConfig:
    alignment: bool | None = None
    """Collect the existing target/draft alignment records."""
    alignment_limit: int | None = None
    """Maximum saved alignment records per engine."""
    alignment_steps: str | None = None
    """Retained strict nonnegative step/range filter."""
    alignment_directory: str | None = None
    """Alignment output directory, default /tmp."""
    alignment_tag: str | None = None
    """Existing filename tag, sanitized once during initialization."""
    corruption: bool | None = None
    """Ordinary speculative corruption diagnostics."""
    draft_logits: bool | None = None
    """Ordinary draft-logit dumps."""
    rejection_profile: bool | None = None
    """Retain rejection sampler GPU stage timings."""
    rejection_interval: int | None = None
    """Rejection timing aggregation interval, default 20 with legacy clamping."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization sources; no diagnostic field contributes to graph hashing."""
    errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Captured parse errors for flags hidden by a family override."""
    selected_steps: frozenset[int] | None = Field(default=None, init=False)
    """Parsed alignment filter; invalid legacy filters select no steps."""
    safe_tag: str = Field(default="", init=False)
    """Sanitized filename component."""

    aliases: ClassVar[dict[str, str]] = {
        "alignment": "VLLM_SPEC_DUMP_ALIGNMENT",
        "alignment_limit": "VLLM_SPEC_DUMP_ALIGNMENT_LIMIT",
        "alignment_steps": "VLLM_SPEC_DUMP_ALIGNMENT_STEPS",
        "alignment_directory": "VLLM_SPEC_DUMP_ALIGNMENT_DIR",
        "alignment_tag": "VLLM_SPEC_DUMP_ALIGNMENT_TAG",
        "corruption": "VLLM_SPEC_DEBUG_CORRUPTION",
        "draft_logits": "VLLM_SPEC_DUMP_DRAFT_LOGITS",
        "rejection_profile": "VLLM_SM70_REJECTION_PROFILE",
        "rejection_interval": "VLLM_SM70_REJECTION_PROFILE_INTERVAL",
    }

    def __post_init__(self) -> None:
        if self.sources:
            return
        from vllm import envs
        from vllm.config.sm70_runtime import resolve_legacy_fields

        def read(name):
            if name == "VLLM_SM70_REJECTION_PROFILE":
                return legacy_rejection_profile("enabled")
            if name == "VLLM_SM70_REJECTION_PROFILE_INTERVAL":
                return legacy_rejection_profile("interval")
            if name == "VLLM_SPEC_DUMP_ALIGNMENT_DIR":
                return os.getenv(name, "/tmp")
            if name == "VLLM_SPEC_DUMP_ALIGNMENT_TAG":
                return os.getenv(name, "")
            try:
                return envs.environment_variables[name]()
            except ValueError as exc:
                for field in self.aliases:
                    if self.aliases[field] == name:
                        self.errors[field] = str(exc)
                        return False
                raise

        resolve_legacy_fields(self, self.aliases, reader=read)
        try:
            self.selected_steps = parse_int_filter(self.alignment_steps, strict=True)
        except ValueError:
            self.selected_steps = frozenset()
        self.safe_tag = "".join(
            ch if ch.isalnum() or ch in "._-" else "_"
            for ch in (self.alignment_tag or "")
        )

    def value(self, field: str):
        if field in self.errors:
            raise ValueError(self.errors[field])
        return getattr(self, field)


def legacy_rejection_profile(field: str):
    """Independent old helper API; engine callers supply SamplingDiagnosticsConfig."""
    if field == "enabled":
        return os.getenv("VLLM_SM70_REJECTION_PROFILE", "0") == "1"
    try:
        return max(1, int(os.getenv("VLLM_SM70_REJECTION_PROFILE_INTERVAL", "20")))
    except ValueError:
        return 20
