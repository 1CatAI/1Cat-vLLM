# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explain common MoE stages for an already admitted TurboMind layer.

This is a static prediction with explicit native availability, not a native-hit
counter. Format-specific model qualification is in the B0 source ledger.
"""

import importlib.util
import itertools
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.sm70.explain import explain_moe_plan
from vllm import envs
from vllm.config.sm70_moe import Sm70MoEFormatConfig
from vllm.model_executor.layers.fused_moe.sm70.declarations import native_binding
from vllm.model_executor.layers.quantization import awq_sm70_moe, fp8_sm70_moe
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    select_single_token_plan,
    select_sm70_quantized_moe_route,
)


def load_baseline(ref, directory):
    result = {}
    for family in ("awq", "fp8"):
        source = f"vllm/model_executor/layers/quantization/{family}_sm70_moe.py"
        path = Path(directory) / (family + "_moe.py")
        path.write_text(
            subprocess.check_output(["git", "show", f"{ref}:{source}"], text=True)
        )
        spec = importlib.util.spec_from_file_location("baseline_moe_" + family, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[family] = module
    return result


def snapshot(modules=None):
    modules = modules or {"awq": awq_sm70_moe, "fp8": fp8_sm70_moe}
    rows = []
    explanations = []
    for family, tokens, batched, indexed, strict, native in itertools.product(
        ("awq", "fp8"),
        (0, 1, 2, 32, 33, 64, 65),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
    ):
        if family == "fp8" and strict:
            continue
        flags = {
            f"VLLM_SM70_{family.upper()}_MOE_BATCHED_GEMM": str(int(batched)),
            f"VLLM_SM70_{family.upper()}_MOE_LEGACY_SINGLE_TOKEN_COMPACT": "0",
            "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH": str(int(indexed)),
            "VLLM_SM70_AWQ_MOE_BATCHED_SINGLE_TOKEN_DENSE_W13": str(int(strict)),
        }
        available = (
            {
                native_binding(family, stage, mode): object()
                for stage in ("w13", "w2")
                for mode in ("indexed", "active_dense")
            }
            if native
            else {}
        )
        with (
            patch.dict(os.environ, flags),
            patch.object(torch.ops, "_C", SimpleNamespace(**available)),
        ):
            envs.disable_envs_cache()
            module = modules[family]
            i13 = module._single_token_indexed_w13_enabled()
            i2 = module._single_token_indexed_w2_enabled()
            single = tokens == 1 and (
                not batched or (family == "awq" and (strict or (i13 and i2)))
            )
            plan = None
            if not tokens:
                stages = {}
                operators = []
            elif single:
                plan = select_single_token_plan(
                    compact_w13=False,
                    indexed_w13=i13,
                    indexed_w2=i2,
                    weighted_reduce=False,
                    strict=strict,
                )
                stages = {
                    "w13": "indexed" if i13 and not strict else "active_dense",
                    "w2": "indexed" if i2 and not strict else "active_dense",
                }
                operators = [
                    native_binding(family, stage, mode)
                    for stage, mode in stages.items()
                ]
            else:
                plan = select_sm70_quantized_moe_route(
                    batched_enabled=batched,
                    num_tokens=tokens,
                    total_slots=tokens * 2,
                    strict_dense_w13=strict,
                    w13_per_expert_dispatch=family == "awq",
                    w2_per_expert_dispatch=family == "awq",
                )
                stages = {"w13": plan.w13.value, "w2": plan.w2.value}
                operators = [
                    native_binding(family, stage, mode)
                    for stage, mode in stages.items()
                ]
            rows.append(
                {
                    "config": f"{family}:M{tokens}:B{int(batched)}:I{int(indexed)}"
                    f":S{int(strict)}:N{int(native)}",
                    "parameters": flags,
                    "selected_stages": stages,
                    "predicted_operators": operators,
                    "fallback": "indexed operator absent"
                    if indexed and not native
                    else None,
                }
            )
            policy = Sm70MoEFormatConfig()
            policy.resolve(family)
            explanations.append(
                {
                    "config": rows[-1]["config"],
                    **explain_moe_plan(
                        family,
                        plan,
                        policy,
                        admission=(
                            {"contract": "prepared SM70 FP16 E4/top-k2 layer"},
                            {
                                "reason": rows[-1]["fallback"],
                                "indexed_native_available": native,
                            },
                        ),
                    ),
                }
            )
    envs.disable_envs_cache()
    from tools.sm70.path_inventory import binding_catalog

    return {
        "stage_declarations": binding_catalog(),
        "evidence": "static selector prediction; native execution unobserved",
        "contract": "prepared SM70 FP16 layer; E4/top-k2; no model-specific route",
        "cases": rows,
        "explanations": explanations,
        "edge_cases": [],
    }
