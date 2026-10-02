# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare FP8 policy removal against the actual historical config statements.

The native kernels are not invoked. Dense route flags come from the real
Fp8LinearMethod constructor; model/TP/KV policy statements come independently
from git. This category checks policy, not the QPN8 subkernel selection.
"""

import ast
import itertools
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from vllm import envs
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.quantization import fp8

POLICY_NAMES = {
    "VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK",
    "VLLM_SM70_FP8_TURBOMIND",
}
CONFIG_FILE = "vllm/config/vllm.py"


def policy_code(source):
    tree = ast.parse(source)
    config = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VllmConfig"
    )
    init = next(
        n
        for n in config.body
        if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
    )
    blocks = []
    for statement in init.body:
        if isinstance(statement, ast.If) and any(
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in POLICY_NAMES
            for node in ast.walk(statement)
        ):
            blocks.append(statement)
    return compile(ast.Module(body=blocks, type_ignores=[]), CONFIG_FILE, "exec")


def snapshot(source=None):
    code = policy_code(Path(CONFIG_FILE).read_text() if source is None else source)
    matrix = list(
        itertools.product(
            ("27b_dflash2_nvfp4", "flash_next_mtp4_nvfp4", "35b_a3b_awq"),
            ("auto", "fp8_e4m3", "fp8_e5m2"),
            (2, 4),
            ("none", "mtp", "dflash"),
            (1, 4, 8),
            (4096, 8192),
        )
    )
    edges = list(
        itertools.product(
            ("auto", "turbomind", "marlin"),
            (None, "0", "1"),
            (None, "0", "1"),
            ("0", "1"),
        )
    )
    platform = NS(is_cuda=lambda: True, has_device_capability=lambda value: value == 70)

    def probe(model, kv, tp, spec, concurrency, budget, overrides):
        config = NS(
            kernel_config=KernelConfig(),
            model_config=NS(
                quantization=model.rsplit("_", 1)[-1],
                is_moe=model.startswith("35b"),
                dtype=torch.float16,
            ),
            parallel_config=NS(tensor_parallel_size=tp),
            cache_config=NS(cache_dtype=kv),
        )
        scope = dict(
            self=config,
            envs=envs,
            os=os,
            current_platform=platform,
            _any_participating_device_is_capability=lambda *_: True,
            sm70_fp8_kv_requested=kv.startswith("fp8"),
            logger=NS(info_once=lambda *args: None),
        )
        clean = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
        clean.update(overrides)
        with patch.dict(os.environ, clean, clear=True):
            envs.disable_envs_cache()
            exec(code, scope)
            envs.disable_envs_cache()
            applicable = config.model_config.quantization == "fp8"
            quant = fp8.Fp8Config(True, "dynamic", weight_block_size=[128, 128])
            quant.use_deep_gemm = False
            with (
                patch.object(fp8, "current_platform", platform),
                patch.object(fp8, "get_current_vllm_config", lambda: config),
                patch.object(fp8, "cutlass_block_fp8_supported", lambda: False),
            ):
                method = fp8.Fp8LinearMethod(quant) if applicable else None
            return {
                "config": f"{model}/{kv}/tp{tp}/{spec}/c{concurrency}/budget{budget}",
                "applicable": applicable,
                "dense_turbomind": bool(method and method.use_sm70_fp8_turbomind),
                "dense_dequant": bool(method and method.use_sm70_dequant_fallback),
                "moe_dequant_requested": envs.VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK,
                "forced_marlin": envs.force_sm70_marlin(),
            }

    try:
        cases = [probe(*row, {}) for row in matrix]
        fp8_matrix = itertools.product(
            ("27b_fp8", "35b_a3b_fp8"),
            ("auto", "fp8_e4m3", "fp8_e5m2"),
            (2, 4),
            ("none", "mtp", "dflash"),
            (1, 4, 8),
            (4096, 8192),
        )
        edge_cases = [probe(*row, {}) for row in fp8_matrix]
        for backend, tm, moe, dequant in edges:
            overrides = {
                "VLLM_SM70_QUANT_BACKEND": backend,
                "VLLM_SM70_FP8_DEQUANT_FALLBACK": dequant,
            }
            if tm is not None:
                overrides["VLLM_SM70_FP8_TURBOMIND"] = tm
            if moe is not None:
                overrides["VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK"] = moe
            row = probe("35b_a3b_fp8", "fp8_e4m3", 2, "none", 4, 8192, overrides)
            row["config"] = f"backend={backend}/tm={tm}/moe={moe}/dequant={dequant}"
            edge_cases.append(row)
        return {"category": "fp8-policy", "cases": cases, "edge_cases": edge_cases}
    finally:
        envs.disable_envs_cache()


def baseline_snapshot(ref):
    source = subprocess.check_output(["git", "show", f"{ref}:{CONFIG_FILE}"], text=True)
    return snapshot(source)
