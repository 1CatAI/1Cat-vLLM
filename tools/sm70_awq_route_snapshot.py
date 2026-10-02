# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent legacy AWQ loading/dispatch recordings for the route CLI."""

import importlib.util
import itertools
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch

from vllm import _sm70_ops, envs
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels import linear
from vllm.model_executor.kernels.linear.mixed_precision import sm70_awq
from vllm.model_executor.layers.quantization import awq as candidate
from vllm.platforms import PlatformEnum

SCHEME = "vllm/model_executor/layers/quantization/awq.py"
MODELS = ("27b_dflash2_nvfp4", "flash_next_mtp4_nvfp4", "35b_a3b_awq")


def load_baseline(ref, directory):
    source = subprocess.check_output(["git", "show", f"{ref}:{SCHEME}"], text=True)
    path = Path(directory) / "baseline_awq.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("baseline_awq", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_layer(
    module,
    *,
    tp=4,
    group_size=32,
    role="qkv_proj",
    native=True,
    workspace=True,
    capability=70,
):
    k, n = 2560, 3584 if role == "qkv_proj" else 2560
    calls = []
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0." + role
    layer.tp_size = tp
    layer.output_size_per_partition = n
    layer.output_partition_sizes = [n // 2, n // 2] if role == "gate_up_proj" else [n]
    for name, shape, dtype in (
        ("qweight", (k, n // 8), torch.int32),
        ("qzeros", (k // group_size, n // 8), torch.int32),
        ("scales", (k // group_size, n), torch.float16),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False
            ),
        )

    def prepare(weight, scales, zeros, group, gated):
        calls.append({"op": "awq_sm70_prepare", "group": group, "gated": gated})
        return weight, scales, torch.tensor([k, k])

    def dispatch(out, x, weight, scales, group, k_ld, q_ld, *extra):
        calls.append(
            {
                "op": "awq_gemm_sm70_out",
                "m": x.shape[0],
                "group": group,
                "k_ld": k_ld,
                "q_ld": q_ld,
                "gated": bool(extra and extra[0]),
            }
        )

    def dequant(out, weight, scales, group):
        calls.append({"op": "awq_sm70_dequantize_out", "group": group})

    def mm(x, weight):
        calls.append({"op": "torch.mm", "m": x.shape[0]})
        return x.new_empty((x.shape[0], weight.shape[-1]))

    cfg = NS(kernel_config=KernelConfig())
    cfg.kernel_config.sm70_awq.resolve()
    native_ops = (
        NS(awq_sm70_prepare=True, awq_sm70_dequantize_out=True) if native else NS()
    )
    with (
        patch.object(linear, "current_platform", NS(_enum=PlatformEnum.CUDA)),
        patch.object(
            module, "get_current_vllm_config_or_none", lambda: cfg, create=True
        ),
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ),
        patch.object(
            torch.cuda,
            "get_device_capability",
            lambda device: (capability // 10, capability % 10),
        ),
        patch.object(torch.ops, "_C", native_ops),
        patch.object(_sm70_ops, "awq_sm70_prepare", prepare),
        patch.object(_sm70_ops, "awq_gemm_sm70_out", dispatch),
        patch.object(_sm70_ops, "awq_sm70_dequantize_out", dequant),
        patch.object(torch, "mm", mm),
        patch.object(
            module,
            "_get_sm70_awq_prefill_exact_dense_workspace",
            lambda weight: weight.new_empty(k * n) if workspace else None,
            create=True,
        ),
        patch.object(
            sm70_awq,
            "_get_sm70_awq_prefill_exact_dense_workspace",
            lambda weight: weight.new_empty(k * n) if workspace else None,
        ),
    ):
        method = module.AWQLinearMethod(module.AWQConfig(4, group_size, True))
        try:
            method.process_weights_after_loading(layer)
        except RuntimeError:
            return {"route": "error", "calls": calls}
        if not getattr(layer, "_awq_sm70_prepared", False):
            return {"route": "legacy_fallback", "calls": calls}
        for m in (1, 4, 8, 4096, 8192):
            method.apply(layer, torch.empty(m, k, device="meta", dtype=torch.float16))
        method.apply_fused_silu_and_mul(
            layer, torch.empty(1, k, device="meta", dtype=torch.float16)
        )
    return {"route": "turbomind", "calls": calls}


def snapshot(module=candidate):
    cases = []
    for model, kv, tp, spec, concurrency, budget in itertools.product(
        MODELS,
        ("float16", "fp8_e4m3", "fp8_e5m2"),
        (2, 4),
        ("none", "mtp", "dflash"),
        (1, 4, 8),
        (4096, 8192),
    ):
        route = (
            snapshot_layer(module, tp=tp)
            if model == "35b_a3b_awq"
            else {"route": "not_applicable", "format": "nvfp4"}
        )
        cases.append(
            {
                "config": f"{model}/{kv}/tp{tp}/{spec}/c{concurrency}/b{budget}",
                "route": route,
            }
        )
    edges = []
    probes = [
        ("group64", {}, {"group_size": 64}),
        ("group128_prefill", {}, {"group_size": 128, "role": "down_proj"}),
        ("unqualified_role", {}, {"group_size": 128, "role": "renamed_projection"}),
        ("native_missing", {}, {"native": False}),
        (
            "workspace_missing",
            {},
            {"group_size": 128, "role": "down_proj", "workspace": False},
        ),
        ("turbo_off", {"VLLM_SM70_AWQ_TURBOMIND": "0"}, {}),
        (
            "prefill_off",
            {"VLLM_SM70_AWQ_PREFILL_EXACT_DENSE": "0"},
            {"group_size": 128, "role": "down_proj"},
        ),
        (
            "fused_tp2",
            {"VLLM_SM70_AWQ_MLP_ENGINE": "1"},
            {"tp": 2, "role": "gate_up_proj"},
        ),
        ("fused_tp4", {"VLLM_SM70_AWQ_MLP_ENGINE": "1"}, {"role": "gate_up_proj"}),
        ("non_sm70", {}, {"capability": 75}),
    ]
    for name, environment, arguments in probes:
        with patch.dict(os.environ, environment):
            envs.disable_envs_cache()
            edges.append({"config": name, "route": snapshot_layer(module, **arguments)})
        envs.disable_envs_cache()
    return {"category": "awq", "cases": cases, "edge_cases": edges}
