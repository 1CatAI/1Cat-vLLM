# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record real serialized FP8 prepare/apply dispatch using meta/native doubles."""

import ast
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from vllm import _sm70_ops, envs
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels import linear
from vllm.model_executor.kernels.linear.scaled_mm import sm70_fp8
from vllm.model_executor.layers.quantization import fp8
from vllm.platforms import PlatformEnum
from vllm.platforms.interface import DeviceCapability

SOURCE = "vllm/model_executor/layers/quantization/fp8.py"
ROLES = {
    "qkv_proj": (5120, 14336),
    "in_proj_qkvz": (5120, 16384),
    "o_proj": (6144, 5120),
    "gate_up_proj": (5120, 34816),
    "down_proj": (17408, 5120),
}
M_VALUES = (1, 4, 8, 16, 4096, 8192)


def load_baseline(ref, directory):
    path = Path(directory) / "baseline_fp8.py"
    path.write_text(
        subprocess.check_output(["git", "show", f"{ref}:{SOURCE}"], text=True)
    )
    spec = importlib.util.spec_from_file_location("baseline_fp8_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "clear_sm70_fp8_workspaces"):
        # After #794, the dispatcher lives in the ordinary linear kernel.
        # Load that historical implementation too, rather than accidentally
        # pairing a historical loader with today's kernel/predicates.
        kernel_path = "vllm/model_executor/kernels/linear/scaled_mm/sm70_fp8.py"
        kernel_file = Path(directory) / "baseline_sm70_fp8.py"
        kernel_file.write_text(
            subprocess.check_output(["git", "show", f"{ref}:{kernel_path}"], text=True)
        )
        kernel_name = "vllm.model_executor.kernels.linear.scaled_mm._snapshot_fp8"
        kernel_spec = importlib.util.spec_from_file_location(kernel_name, kernel_file)
        kernel_module = importlib.util.module_from_spec(kernel_spec)
        sys.modules[kernel_name] = kernel_module
        kernel_spec.loader.exec_module(kernel_module)
        module._snapshot_kernel = kernel_module
        for name, value in vars(kernel_module).items():
            if name in vars(module) and callable(value):
                setattr(module, name, value)
    config_source = subprocess.check_output(
        ["git", "show", f"{ref}:vllm/config/vllm.py"], text=True
    )
    old_defaults = next(
        (
            node.value
            for node in ast.parse(config_source).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_SM70_DFLASH2_VERIFIER_DEFAULTS"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ),
        None,
    )
    if old_defaults is None:
        config_source = subprocess.check_output(
            ["git", "show", f"{ref}:vllm/config/sm70_dflash2.py"], text=True
        )
        old_defaults = next(
            node.value
            for node in ast.parse(config_source).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "SM70_DFLASH2_VERIFIER_DEFAULTS"
                for target in node.targets
            )
        )
    module._snapshot_verifier_defaults = ast.literal_eval(old_defaults)
    return module


def probe(module, cfg, role, k, n, *, missing_qpn8=False, workspace=True, bmm=False):
    calls = []
    prepare = []
    target = sm70_fp8 if module is fp8 else getattr(module, "_snapshot_kernel", module)
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0." + role
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.output_partition_sizes = (
        [n // 2] * 2 if role.endswith("gate_up_proj") else [n]
    )
    layer.weight_block_size = [128, 128]
    layer.tp_size = cfg.parallel_config.tensor_parallel_size
    layer.orig_dtype = torch.float16
    layer.is_bmm = bmm
    layer.bmm_batch_size = 2
    layer.weight = torch.nn.Parameter(
        torch.empty((n, k), dtype=torch.float8_e4m3fn, device="meta"),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.empty(((n + 127) // 128, k // 128), device="meta"), requires_grad=False
    )
    scratch = torch.empty(1, dtype=torch.float16) if workspace else None
    platform = NS(
        _enum=PlatformEnum.CUDA,
        is_cuda=lambda: True,
        has_device_capability=lambda c: c == 70,
        get_device_capability=lambda: DeviceCapability(7, 0),
        fp8_dtype=lambda: torch.float8_e4m3fn,
    )

    def native_prepare(weight, scales, group_size=128, gated=False):
        prepare.append(
            {
                "op": "turbomind",
                "shape": list(weight.shape),
                "group": group_size,
                "gated": gated,
            }
        )
        rows, cols = weight.shape
        return (
            torch.empty((cols, rows), dtype=torch.uint8, device="meta"),
            torch.ones(1, dtype=torch.float16),
            torch.tensor([cols, rows]),
        )

    def qpn8_prepare(weight, scales):
        prepare.append({"op": "qpn8", "shape": list(weight.shape)})
        rows, cols = weight.shape
        return (
            torch.empty((cols, rows), dtype=torch.uint8, device="meta"),
            torch.empty((cols // 16, rows // 32), dtype=torch.float16, device="meta"),
        )

    def record(name):
        def call(*args):
            calls.append(
                {
                    "op": name,
                    "input": list(
                        args[
                            2 if name in {"qpn8_dispatch", "prefill_dispatch"} else 1
                        ].shape
                    ),
                    "scalar_plan": [v for v in args if type(v) in (int, bool)],
                }
            )

        return call

    native = NS(**{name: lambda: None for name in sm70_fp8._SM70_FP8_QPN8_REQUIRED_OPS})
    native.fp8_sm70_prepare = lambda: None
    for name in (
        "fp8_gemm_sm70_prefill_prescaled_out",
        "fp8_gemm_sm70_prefill_dispatch_out",
        "fp8_gemm_sm70_prescaled_m1_out",
    ):
        setattr(native, name, lambda: None)
    opaque = NS(
        sm70_fp8_qpn8_dispatch=record("qpn8_dispatch"),
        sm70_fp8_prefill_dispatch=record("prefill_dispatch"),
    )
    saved_environment = dict(os.environ)
    try:
        with ExitStack() as stack:
            contexts = (
                set_current_vllm_config(cfg),
                patch.object(module, "current_platform", platform),
                patch.object(sm70_fp8, "current_platform", platform),
                patch.object(target, "current_platform", platform),
                patch.object(linear, "current_platform", platform),
                patch.object(
                    linear,
                    "Sm70Fp8LinearLayerConfig",
                    getattr(
                        target,
                        "Sm70Fp8LinearLayerConfig",
                        sm70_fp8.Sm70Fp8LinearLayerConfig,
                    ),
                ),
                patch.object(
                    linear,
                    "TurboMindFp8LinearKernel",
                    getattr(
                        target,
                        "TurboMindFp8LinearKernel",
                        sm70_fp8.TurboMindFp8LinearKernel,
                    ),
                ),
                patch.dict(
                    linear._POSSIBLE_FP8_BLOCK_KERNELS,
                    {
                        PlatformEnum.CUDA: [
                            getattr(target, "TurboMindFp8LinearKernel", kernel)
                            if kernel is sm70_fp8.TurboMindFp8LinearKernel
                            else kernel
                            for kernel in linear._POSSIBLE_FP8_BLOCK_KERNELS[
                                PlatformEnum.CUDA
                            ]
                        ]
                    },
                ),
                patch.object(module, "get_current_vllm_config", lambda: cfg),
                patch.object(target, "get_current_vllm_config", lambda: cfg),
                patch.object(module, "cutlass_block_fp8_supported", lambda: False),
                patch.object(_sm70_ops, "fp8_sm70_prepare", native_prepare),
                patch.object(_sm70_ops, "fp8_qpn8_prepare_sm70", qpn8_prepare),
                patch.object(
                    target,
                    "_missing_sm70_fp8_qpn8_ops",
                    lambda: ["test_missing"] if missing_qpn8 else [],
                ),
                patch.object(
                    target,
                    "_get_sm70_fp8_prefill_exact_dense_workspace",
                    lambda _: scratch,
                ),
                patch.object(
                    target, "_get_sm70_fp8_qpn8_pp2_tp4_workspace", lambda _: scratch
                ),
                patch.object(_sm70_ops, "fp8_gemm_sm70_out", record("turbomind")),
                patch.object(
                    _sm70_ops, "fp8_gemm_sm70_prescaled_m1_out", record("prescaled_m1")
                ),
                patch.object(
                    _sm70_ops,
                    "fp8_gemm_sm70_prefill_prescaled_out",
                    record("prescaled_prefill"),
                ),
                patch.object(torch.ops, "_C", native),
                patch.object(torch.ops, "vllm", opaque),
            )
            for context in contexts:
                stack.enter_context(context)
            quant = module.Fp8Config(True, "dynamic", weight_block_size=[128, 128])
            quant.use_deep_gemm = False
            method = module.Fp8LinearMethod(quant)
            if not method.use_sm70_fp8_turbomind:
                return {
                    "route": "dequant"
                    if method.use_sm70_dequant_fallback
                    else "other_backend"
                }
            method.process_weights_after_loading(layer)
            for m in M_VALUES:
                x = torch.empty(
                    (m, 2, k) if bmm else (m, k), dtype=torch.float16, device="meta"
                )
                method.apply(layer, x)
                if role.endswith("gate_up_proj") and not bmm:
                    method.apply_fused_silu_and_mul(layer, x)
            return {
                "prepared": prepare,
                "dispatch": calls,
                "qpn8": bool(getattr(layer, "sm70_fp8_qpn8", False)),
                "gated": bool(getattr(layer, "sm70_fp8_gated_silu", False)),
            }
    except (RuntimeError, ValueError) as error:
        return {"error": str(error)}
    finally:
        target.clear_sm70_fp8_workspaces()
        assert saved_environment == dict(os.environ)


def snapshot(module=fp8):
    cases, edges, plans = [], [], {}
    matrix = itertools.product(
        (
            "27b_dflash2_nvfp4",
            "flash_next_mtp4_nvfp4",
            "35b_a3b_awq",
            "27b_fp8",
            "35b_a3b_fp8",
        ),
        ("auto", "fp8_e4m3", "fp8_e5m2"),
        (2, 4),
        ("none", "mtp", "dflash"),
        (1, 4, 8),
        (4096, 8192),
    )

    def run(
        row,
        overrides=None,
        *,
        role=None,
        shape=None,
        missing_qpn8=False,
        workspace=True,
        pp=1,
        bmm=False,
    ):
        model, kv, tp, spec, concurrency, budget = row
        label = f"{model}/{kv}/tp{tp}/{spec}/c{concurrency}/budget{budget}"
        if not model.endswith("_fp8"):
            return {"config": label, "applicable": False}
        clean = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
        clean.update(overrides or {})
        cfg = NS(
            kernel_config=KernelConfig(),
            compilation_config=CompilationConfig(),
            model_config=NS(dtype=torch.float16),
            parallel_config=NS(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                enable_dbo=False,
                ubatch_size=0,
            ),
            scheduler_config=NS(
                max_num_seqs=concurrency, max_num_batched_tokens=budget
            ),
            speculative_config=None if spec == "none" else NS(method=spec),
        )
        with patch.dict(os.environ, clean, clear=True):
            envs.disable_envs_cache()
            if model == "27b_fp8" and spec == "dflash":
                if module is fp8:
                    from vllm.config.sm70_dflash2 import Sm70DFlash2Config

                    policy = Sm70DFlash2Config()
                    policy.resolve(qualified=True)
                    cfg.speculative_config.sm70_dflash2 = policy
                    cfg.kernel_config.sm70_fp8.qpn8 = policy.target_fp8_qpn8
                else:
                    # Replay the actual historical configuration defaults.
                    for name, value in module._snapshot_verifier_defaults.items():
                        os.environ.setdefault(name, value)
                    envs.disable_envs_cache()
            cfg.kernel_config.sm70_fp8.resolve()
            roles = (
                {role: shape}
                if role
                else {
                    name: (
                        k // tp if name in {"down_proj", "o_proj"} else k,
                        n // tp if name != "down_proj" and name != "o_proj" else n,
                    )
                    for name, (k, n) in ROLES.items()
                }
            )
            result = {
                "config": label,
                "applicable": True,
                "layers": {
                    name: probe(
                        module,
                        cfg,
                        name,
                        k,
                        n,
                        missing_qpn8=missing_qpn8,
                        workspace=workspace,
                        bmm=bmm,
                    )
                    for name, (k, n) in roles.items()
                },
            }
            for name, plan in result["layers"].items():
                key = hashlib.sha256(
                    json.dumps(plan, sort_keys=True).encode()
                ).hexdigest()
                plans[key] = plan
                result["layers"][name] = key
            envs.disable_envs_cache()
            return result

    try:
        rows = list(matrix)
        cases = [run(row) for row in rows[:324]]
        edges = [run(row) for row in rows[324:]]
        row = ("27b_fp8", "fp8_e4m3", 4, "none", 1, 8192)
        for overrides, missing, workspace in (
            ({"VLLM_SM70_FP8_QPN8": "1"}, False, True),
            ({"VLLM_SM70_FP8_QPN8": "0"}, False, True),
            ({"VLLM_SM70_FP8_QPN8": "1"}, True, True),
            ({"VLLM_SM70_FP8_QPN8": "1"}, False, False),
            ({"VLLM_SM70_FP8_DENSE_GATED_SILU": "0"}, False, True),
            ({"VLLM_SM70_FP8_PREFILL_EXACT_DENSE": "0"}, False, True),
        ):
            record = run(row, overrides, missing_qpn8=missing, workspace=workspace)
            record["config"] = (
                f"override={overrides}/missing={missing}/workspace={workspace}"
            )
            edges.append(record)
        for role, shape, bmm in (
            ("fused_wqa_wkv", (4096, 1536), False),
            ("wq_b", (1024, 8192), False),
            ("wo_a", (4096, 2048), True),
            ("shared_experts.gate_up_proj", (4096, 1024), False),
        ):
            record = run(
                row,
                {
                    "VLLM_SM70_FP8_QPN8_PP2_TP4": "1",
                    "VLLM_SM70_FP8_QPN8_PP2_TP4_SHARED_GATE": "1",
                },
                role=role,
                shape=shape,
                pp=2,
                bmm=bmm,
            )
            record["config"] = "pp2_tp4/" + role
            edges.append(record)
        return {
            "scope": "serialized_sm70_fp8_dispatch",
            "matrix_cases": len(cases),
            "cases": cases,
            "edge_cases": edges,
            "plans": plans,
        }
    finally:
        envs.disable_envs_cache()
