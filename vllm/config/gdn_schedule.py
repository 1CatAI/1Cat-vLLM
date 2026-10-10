# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Captured FLA launch controls, shared by prefill and recurrent providers."""

import os
from typing import Annotated

from pydantic import Field

from vllm.config.utils import config

GDN_SCHEDULE_FIELDS = {
    "kda_prefill_enabled": ("VLLM_SM70_KDA_PREFILL_SCHEDULE", True, "exact"),
    "recurrent_enabled": ("VLLM_SM70_FLA_RECURRENT_SCHEDULE", True, "exact"),
    "recurrent_bv": ("VLLM_SM70_FLA_BV", None, "positive"),
    "recurrent_warps": ("VLLM_SM70_FLA_WARPS", None, "positive"),
    "recurrent_stages": ("VLLM_SM70_FLA_STAGES", None, "positive"),
    "recurrent_target_waves": ("VLLM_SM70_FLA_TARGET_WAVES", 2, "positive"),
    "recurrent_bv_candidates": ("VLLM_SM70_FLA_BV_CANDIDATES", [32, 16, 8], "list"),
    "sigmoid_enabled": ("VLLM_SM70_FUSED_SIGMOID_GATING_SCHED", True, "exact"),
    "sigmoid_bv": ("VLLM_SM70_FUSED_SIGMOID_GATING_BV", None, "positive"),
    "sigmoid_warps": ("VLLM_SM70_FUSED_SIGMOID_GATING_WARPS", None, "positive"),
    "sigmoid_stages": ("VLLM_SM70_FUSED_SIGMOID_GATING_STAGES", None, "positive"),
    "kkt_enabled": ("VLLM_SM70_GDN_KKT_SCHEDULE", True, "exact"),
    "kkt_bk": ("VLLM_SM70_GDN_KKT_BK", [32, 64], "list"),
    "kkt_warps": ("VLLM_SM70_GDN_KKT_WARPS", [4], "list"),
    "kkt_stages": ("VLLM_SM70_GDN_KKT_STAGES", [2], "list"),
    "delta_h_enabled": ("VLLM_SM70_GDN_DELTA_H_SCHEDULE", True, "exact"),
    "delta_h_bv": ("VLLM_SM70_GDN_DELTA_H_BV", [16], "list"),
    "delta_h_warps": ("VLLM_SM70_GDN_DELTA_H_WARPS", [8], "list"),
    "delta_h_stages": ("VLLM_SM70_GDN_DELTA_H_STAGES", [1], "list"),
    "chunk_o_enabled": ("VLLM_SM70_GDN_CHUNK_O_SCHEDULE", True, "exact"),
    "chunk_o_bk": ("VLLM_SM70_GDN_CHUNK_O_BK", [32, 64], "list"),
    "chunk_o_bv": ("VLLM_SM70_GDN_CHUNK_O_BV", [32, 64], "list"),
    "chunk_o_warps": ("VLLM_SM70_GDN_CHUNK_O_WARPS", [4, 8], "list"),
    "chunk_o_stages": ("VLLM_SM70_GDN_CHUNK_O_STAGES", [2], "list"),
}


def _positive(raw):
    try:
        value = int(raw)
    except (ValueError, TypeError):
        return None
    return value if value > 0 else None


@config
class GdnScheduleConfig:
    """Explicit launch choices override the captured historical environment."""

    kda_prefill_enabled: bool | None = None
    """Retain the KDA prefill launch search space on the qualified device."""
    recurrent_enabled: bool | None = None
    """Launch control formerly provided by VLLM_SM70_FLA_RECURRENT_SCHEDULE."""
    recurrent_bv: int | None = Field(default=None, gt=0)
    """Launch control formerly provided by VLLM_SM70_FLA_BV."""
    recurrent_warps: int | None = Field(default=None, gt=0)
    """Launch control formerly provided by VLLM_SM70_FLA_WARPS."""
    recurrent_stages: int | None = Field(default=None, gt=0)
    """Launch control formerly provided by VLLM_SM70_FLA_STAGES."""
    recurrent_target_waves: int | None = Field(default=None, gt=0)
    """Launch control formerly provided by VLLM_SM70_FLA_TARGET_WAVES."""
    recurrent_bv_candidates: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_FLA_BV_CANDIDATES."""
    sigmoid_enabled: bool | None = None
    """Launch control formerly provided by VLLM_SM70_FUSED_SIGMOID_GATING_SCHED."""
    sigmoid_bv: int | None = Field(default=None, gt=0)
    """Launch control formerly provided by VLLM_SM70_FUSED_SIGMOID_GATING_BV."""
    sigmoid_warps: int | None = Field(default=None, gt=0)
    """Launch control formerly provided by VLLM_SM70_FUSED_SIGMOID_GATING_WARPS."""
    sigmoid_stages: int | None = Field(default=None, gt=0)
    """Launch control formerly provided by VLLM_SM70_FUSED_SIGMOID_GATING_STAGES."""
    kkt_enabled: bool | None = None
    """Launch control formerly provided by VLLM_SM70_GDN_KKT_SCHEDULE."""
    kkt_bk: list[Annotated[int, Field(gt=0)]] | None = Field(default=None, min_length=1)
    """Launch control formerly provided by VLLM_SM70_GDN_KKT_BK."""
    kkt_warps: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_KKT_WARPS."""
    kkt_stages: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_KKT_STAGES."""
    delta_h_enabled: bool | None = None
    """Launch control formerly provided by VLLM_SM70_GDN_DELTA_H_SCHEDULE."""
    delta_h_bv: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_DELTA_H_BV."""
    delta_h_warps: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_DELTA_H_WARPS."""
    delta_h_stages: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_DELTA_H_STAGES."""
    chunk_o_enabled: bool | None = None
    """Launch control formerly provided by VLLM_SM70_GDN_CHUNK_O_SCHEDULE."""
    chunk_o_bk: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_CHUNK_O_BK."""
    chunk_o_bv: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_CHUNK_O_BV."""
    chunk_o_warps: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_CHUNK_O_WARPS."""
    chunk_o_stages: list[Annotated[int, Field(gt=0)]] | None = Field(
        default=None, min_length=1
    )
    """Launch control formerly provided by VLLM_SM70_GDN_CHUNK_O_STAGES."""
    recurrent_override: bool = Field(default=False, init=False)
    """Retain the legacy override admission, even for ignored invalid values."""
    sigmoid_override: bool = Field(default=False, init=False)
    """Retain the legacy sigmoid override admission."""
    resolved: bool = Field(default=False, init=False)
    """Whether the engine has captured launch controls."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Provenance, excluded from computation hashes."""

    def resolve(self):
        if self.resolved:
            return
        for name, (alias, default, parser) in GDN_SCHEDULE_FIELDS.items():
            value = getattr(self, name)
            raw = os.environ.get(alias)
            if not name.endswith("_enabled") and (
                value is not None or raw not in (None, "")
            ):
                if name.startswith("recurrent_"):
                    self.recurrent_override = True
                elif name.startswith("sigmoid_"):
                    self.sigmoid_override = True
            if value is not None:
                self.sources[name] = "typed"
                continue
            self.sources[name] = alias if raw is not None else "default"
            if parser == "exact":
                value = default if raw is None else raw == "1"
            elif parser == "positive":
                value = _positive(raw) or default
            else:
                assert isinstance(default, list)
                value = [_positive(token.strip()) for token in (raw or "").split(",")]
                value = [item for item in value if item is not None] or list(default)
            setattr(self, name, value)
        self.resolved = True

    active_family: str | None = Field(default=None, init=False)
    """Model-declared schedule consumers; None retains independent API hashing."""

    def graph_options(self):
        options = {
            **{name: getattr(self, name) for name in GDN_SCHEDULE_FIELDS},
            "recurrent_override": self.recurrent_override,
            "sigmoid_override": self.sigmoid_override,
        }
        if self.active_family == "gdn":
            options.pop("kda_prefill_enabled")
        elif self.active_family == "kda":
            options = {
                k: v for k, v in options.items() if k.startswith(("kda_", "delta_h_"))
            }
        return options


def resolve_schedule(schedule=None):
    """Use an initialized owner; only independent no-config calls adapt env."""
    if schedule is not None:
        return schedule
    from vllm.runtime_resources import current_runtime_resources

    resources = current_runtime_resources()
    if resources is not None and resources.get("fla_schedule") is not None:
        return resources["fla_schedule"]
    schedule = GdnScheduleConfig()
    schedule.resolve()
    return schedule
