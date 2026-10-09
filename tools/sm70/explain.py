# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explain an existing selection using the codec's own binding declarations."""

from dataclasses import asdict
from typing import Any

from vllm.config.sm70_native import NATIVE_FIELDS, UNSET
from vllm.model_executor.layers.fused_moe.sm70.declarations import (
    FP4_STAGE_BINDINGS,
    STAGE_BINDINGS,
    fp4_binding_mode,
    native_binding,
)


def configuration_report(policy) -> list[dict[str, Any]]:
    records = []
    sources = getattr(policy, "sources", {})
    native = getattr(policy, "native", policy)
    if native is not policy:
        for field, value in vars(policy).items():
            if field in {"native", "sources", "resolved", "active", "explicit_enables"}:
                continue
            if hasattr(value, "__dataclass_fields__"):
                continue
            records.append(
                {
                    "parameter": field,
                    "value": value,
                    "source": sources.get(field, "typed config/default"),
                }
            )
    for (field, alias, families, diagnostic), value in zip(
        NATIVE_FIELDS, native.values
    ):
        if value != UNSET:
            records.append(
                {
                    "parameter": "native." + field,
                    "value": value,
                    "legacy_alias": alias,
                    "source": native.sources[field],
                    "role": "diagnostic" if diagnostic else "calculation",
                }
            )
    return records


def explain_moe_plan(
    family, plan, policy, *, raw_scale=False, admission=(), observed=None
):
    """No new selection: the caller supplies the plan from the existing selector."""
    stages = []
    if plan is not None:
        for stage in ("w13", "w2"):
            mode = getattr(plan, stage).value
            if family in ("awq", "fp8"):
                if mode == "chunked":
                    binding = (
                        "awq_moe_chunked_w2_sm70_out",
                        "w2+reduce",
                        "chunked routes",
                        "original FP16 boundaries",
                    )
                else:
                    spec = STAGE_BINDINGS[stage][mode]
                    binding = (native_binding(family, stage, mode), *spec[1:])
            else:
                key = fp4_binding_mode(
                    family, mode, raw_scale=raw_scale, qpn_mtp=plan.qpn_mtp
                )
                binding = FP4_STAGE_BINDINGS[family, stage, key]
            stages.append(
                dict(
                    stage=stage,
                    mode=mode,
                    predicted_operator=binding[0],
                    covers=binding[1],
                    layout=binding[2],
                    arithmetic=binding[3],
                    launches=2 if mode == "indexed_split_fused" else 1,
                )
            )
    return {
        "evidence": "static explanation of an existing selected plan",
        "parameters": configuration_report(policy),
        "admission_and_fallback": list(admission),
        "selected_plan": asdict(plan) if plan is not None else None,
        "stages": stages,
        "observed_execution": observed,
    }


def explain_linear(kernel, family, *, provider=None, observed=None):
    """Report retained selector decisions; this function never selects a kernel."""
    return {
        "evidence": "static explanation of existing linear selection",
        "parameters": configuration_report(getattr(kernel, "sm70_" + family)),
        "admission_and_fallback": kernel.linear_kernel_selections,
        "prepared_provider": None
        if provider is None
        else {
            "name": provider.name,
            "fused_gated": provider.fused is not None,
            "contiguous_input": provider.contiguous_input,
        },
        "observed_execution": observed,
    }
