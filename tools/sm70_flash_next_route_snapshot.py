# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare actual Flash-Next loader/dispatch predicates with a historical revision.

CPU doubles preserve tensor layout/dtype/alignment without launching native ops.
This records configured routes, not observed CUDA execution or native tuning.
"""

import argparse
import ast
import itertools
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from vllm import envs
from vllm.config.kernel import KernelConfig
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc

BASE = "vllm/models/qwen4_exp/nvidia/"


def historical(ref, path, current):
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    tree = ast.parse(source)
    # Run the historical functions against the same tensor doubles. Imports,
    # Triton definitions and custom-op registrations are deliberately excluded.
    names = {name for name, value in vars(current).items() if callable(value)}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names and not n.decorator_list
    ]
    namespace = dict(vars(current))
    exec(compile(ast.Module(nodes, type_ignores=[]), path, "exec"), namespace)
    return NS(**namespace)


def defaults(ref):
    source = subprocess.check_output(["git", "show", f"{ref}:vllm/envs.py"], text=True)
    tree = ast.parse(source)
    dictionary = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "environment_variables"
    )
    return {
        key.value: eval(
            compile(ast.Expression(value.args[0]), "envs.py", "eval"),
            {"os": os, "envs": envs},
        )
        for key, value in zip(dictionary.keys, dictionary.values)
        if key.value
        in {
            name
            for name, getter in envs.environment_variables.items()
            if "Flash-Next qualified batch" in getter.metadata.acceleration_paths
        }
    }


def tensor(shape):
    return NS(
        shape=shape,
        ndim=2,
        dtype=torch.float16,
        device="cuda:0",
        is_cuda=True,
        stride=lambda: (shape[1], 1),
        is_contiguous=lambda: True,
        data_ptr=lambda: 16,
    )


def collect(ref=None):
    g = historical(ref, BASE + "sm70_fp16_gemv.py", gemv) if ref else gemv
    h = historical(ref, BASE + "sm70_fp16_hc.py", hc) if ref else hc
    # Historical HC imports the historical GEMV admission, not today's helper.
    if ref:
        h.enable_qwen38_sm70_fp16_fused_hc.__globals__["_batch_runtime_contract"] = (
            g._batch_runtime_contract
        )
    getters = defaults(ref) if ref else None
    rows = {}

    class Layer(torch.nn.Module):
        def __init__(self, role, shape):
            super().__init__()
            self.prefix = "model.layers.0." + role
            self.weight = torch.nn.Parameter(torch.empty(shape, device="meta"))
            self.quant_method = gemv.UnquantizedLinearMethod()

    for model, kv, tp, spec, concurrency, budget in itertools.product(
        ("27b_dflash2", "flash_next", "35b_awq"),
        ("fp16", "e4m3", "e5m2"),
        (2, 4),
        ("none", "mtp", "dflash"),
        (1, 4, 8),
        (4096, 8192),
    ):
        cfg = NS(
            kernel_config=KernelConfig(),
            model_config=NS(
                architectures=["Qwen4ExpForCausalLM"]
                if model == "flash_next"
                else ["Qwen3_5ForCausalLM"],
                dtype=torch.float16,
            ),
            speculative_config=None
            if spec == "none"
            else NS(method=spec, num_speculative_tokens=4),
            parallel_config=NS(tensor_parallel_size=tp, use_ubatching=False),
        )
        cfg.kernel_config.resolve_sm70_rmsnorm_gated(
            qualified=model == "flash_next" and spec in ("none", "mtp")
        )
        key = f"{model}/{kv}/tp{tp}/{spec}/c{concurrency}/p{budget}"
        with (
            patch.object(gemv, "LinearBase", Layer),
            patch.object(gemv.current_platform, "is_device_capability", lambda _: True),
        ):
            # Force the already-enabled base FP16 lane in both arms. This PR
            # changes only batch defaults and MTP admission.
            overrides = {
                "VLLM_SM70_QWEN38_FP16_GEMV": True,
                "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16": True,
                "VLLM_SM70_QWEN38_FUSED_HC_FP16": True,
                "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8": False,
                "VLLM_BATCH_INVARIANT": False,
            }
            if getters:
                overrides.update({name: getter() for name, getter in getters.items()})
                overrides["VLLM_SM70_QWEN38_FP16_GEMV"] = True
            with patch.multiple(envs, **overrides):
                backend = torch.backends.cuda.matmul
                previous = backend.allow_fp16_reduced_precision_reduction
                backend.allow_fp16_reduced_precision_reduction = spec == "mtp"
                try:
                    layer = Layer("linear_attn.out_proj", (2560, 1536))
                    module = torch.nn.Module()
                    module.add_module("output", layer)
                    gdn = torch.nn.Module()
                    gdn.add_module(
                        "in_proj_qkvz", Layer("linear_attn.in_proj_qkvz", (4096, 2560))
                    )
                    gdn.add_module(
                        "in_proj_ba", Layer("linear_attn.in_proj_ba", (24, 2560))
                    )
                    gdn.gqa_interleaved_layout = gdn.disable_tp_for_ba_proj = False
                    module.add_module("gdn", gdn)
                    g.enable_qwen38_sm70_fp16_gemv(module, torch.float16, cfg)
                    connection = torch.nn.Module()
                    connection.use_combine, connection.lora_rank = True, 320
                    connection.hc_count, connection.hidden_size = 4, 2560
                    for name in (
                        "input_mix_weight_down_block_inject",
                        "input_mix_weight_up",
                    ):
                        projection = torch.nn.Module()
                        projection.quant_method = gemv.UnquantizedLinearMethod()
                        connection.add_module(name, projection)
                    h.enable_qwen38_sm70_fp16_fused_hc(connection, torch.float16, cfg)
                    rows[key] = {
                        "rmsnorm_gated_permission": (
                            model == "flash_next" and spec == "none"
                            if ref
                            else cfg.kernel_config.sm70_rmsnorm_gated_exact
                        ),
                        "dense_batch_prepared": getattr(
                            layer, "_sm70_qwen38_dense_batch", False
                        ),
                        "batch_qualified": bool(g._batch_runtime_contract(cfg)),
                        "gdn_packed": getattr(
                            gdn.in_proj_qkvz, "_sm70_qwen38_prepare_gdn_batch", False
                        ),
                        "hc_packed": hasattr(
                            connection.input_mix_weight_up, "_sm70_qwen38_hc_batch_role"
                        ),
                        "hc_fp32_partials": getattr(
                            connection.input_mix_weight_up,
                            "_sm70_qwen38_hc_batch_concurrent",
                            False,
                        ),
                    }
                    if model == "flash_next":
                        rows[key]["controls"] = {
                            name: bool(getattr(envs, name))
                            for name, getter in envs.environment_variables.items()
                            if "Flash-Next qualified batch"
                            in getter.metadata.acceleration_paths
                        }
                        rows[key]["controls"]["VLLM_SM70_RMSNORM_GATED_EXACT"] = rows[
                            key
                        ]["rmsnorm_gated_permission"]
                        rows[key]["dense_by_m"] = {
                            str(m): bool(
                                getattr(layer, "_sm70_qwen38_dense_batch", False)
                                and g._can_use_dense_batch(
                                    tensor((m, 1536)),
                                    tensor((2560, 1536)),
                                    layer.prefix,
                                )
                            )
                            for m in (1, 2, 4, 5, 8, 10, 16, 8192)
                        }
                finally:
                    backend.allow_fp16_reduced_precision_reduction = previous
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-changes", type=Path)
    args = parser.parse_args()
    envs.disable_envs_cache()
    baseline, candidate = collect(args.baseline_ref), collect()
    changed = sorted(key for key in baseline if baseline[key] != candidate[key])
    args.output.write_text(
        json.dumps(
            {"baseline": baseline, "candidate": candidate, "changed": changed}, indent=2
        )
        + "\n"
    )
    print(f"{len(candidate)} configurations; {len(changed)} changed rows")
    if args.expected_changes and changed != json.loads(
        args.expected_changes.read_text()
    ):
        raise SystemExit("unexpected route changes")


if __name__ == "__main__":
    main()
