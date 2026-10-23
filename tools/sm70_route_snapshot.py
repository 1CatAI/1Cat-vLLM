# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch snapshots for the migrated CT NVFP4 linear category.

No weights, GPU kernels or model service are loaded. Native calls are recorded
with CPU/meta tensor doubles. --baseline-ref runs the unmodified quantization
module from git, independently of the candidate's kernel predicates.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch.nn.parameter import Parameter

from vllm import _sm70_ops, envs
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.linear.nvfp4 import sm70
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4 as candidate,
)
from vllm.model_executor.models.config import sm70_dflash2_nvfp4_qualified

SCHEME = (
    "vllm/model_executor/layers/quantization/compressed_tensors/schemes/"
    "compressed_tensors_w4a4_nvfp4.py"
)
MODELS = ("27b_dflash2_nvfp4", "flash_next_mtp4_nvfp4", "35b_a3b_awq")


def make_config(spec, tp, concurrency, kv, budget):
    cfg = NS(
        kernel_config=KernelConfig(),
        parallel_config=NS(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            enable_dbo=False,
            ubatch_size=0,
        ),
        speculative_config=None
        if spec == "none"
        else NS(
            method=spec,
            num_speculative_tokens=4 if spec == "mtp" else 7,
            draft_model_config=NS(hf_config=NS(dflash_config={"selector_top_k": 16})),
        ),
        scheduler_config=NS(max_num_seqs=concurrency, max_num_batched_tokens=budget),
        cache_config=NS(cache_dtype=kv),
    )
    cfg.kernel_config.sm70_nvfp4.resolve(qualified=sm70_dflash2_nvfp4_qualified(cfg))
    return cfg


def _layer(k, n, role):
    layer = torch.nn.Module()
    layer.prefix = f"model.layers.0.{role}"
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.logical_widths = [n // 2] * 2 if role == "gate_up_proj" else [n]
    layer.weight_packed = Parameter(
        torch.empty(n, k // 2, dtype=torch.uint8, device="meta"), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.empty(n, k // 16, dtype=torch.float8_e4m3fn, device="meta"),
        requires_grad=False,
    )
    layer.input_global_scale = Parameter(torch.tensor([4.0]), requires_grad=False)
    layer.weight_global_scale = Parameter(torch.tensor([2.0]), requires_grad=False)
    return layer


def snapshot_layer(
    module,
    cfg,
    *,
    k=5120,
    n=4120,
    role="in_proj_qkvz",
    batch=False,
    native=True,
    shared_native=True,
    compact_version=1,
    qpn4_native=True,
    qpn4_workspace=True,
):
    calls = []
    ops = set(sm70._SM70_NVFP4_QPN2_REQUIRED_OPS)
    ops.update(sm70._SM70_NVFP4_QPN2_PREFILL_REQUIRED_OPS)
    if qpn4_native:
        ops.update(sm70._SM70_NVFP4_QPN4_REQUIRED_OPS)
    if shared_native:
        ops.update(
            ("nvfp4_qpn2_prepare_scales_sm70", "nvfp4_qpn2_tm_dispatch_sm70_out")
        )
    native_ops = NS(**{name: lambda: None for name in ops}) if native else NS()
    native_ops.nvfp4_qpn2_compact_tm_gemm_sm70_out = lambda: None
    native_ops.nvfp4_qpn2_compact_scales_version_sm70 = lambda: compact_version

    def prepare(layer, *, interleave_gated_silu=False, prescale_for_batch=False):
        setattr(
            layer,
            tm.STATE_ATTR,
            tm.SM70TurboMindLinearState(
                weight=torch.empty(1, dtype=torch.int32),
                scales=torch.empty(1, dtype=torch.float16),
                group_size=16,
                k_ld=k,
                q_ld=k,
                output_size=n,
                op_kind="nvfp4",
                gated_silu=interleave_gated_silu,
                padded_output_size=(n + 31) // 32 * 32,
                prescaled_scales=prescale_for_batch,
            ),
        )

    def dispatch(name):
        def run(*args):
            # Record the opaque native entry and its scalar plan, not tensor addresses.
            # Shared weights omit the separate TurboMind weight argument.
            gated_index = 11 if name == "nvfp4_qpn2_tm_dispatch_sm70_out" else 12
            calls.append(
                {
                    "op": name,
                    "m": args[1].shape[0],
                    "split_k": args[5],
                    "nacc": args[6],
                    "gated": args[gated_index],
                    "prefill_min_m": (
                        args[gated_index + 1] if len(args) > gated_index + 1 else 0
                    ),
                }
            )
            args[0].zero_()

        return run

    def prepare_qpn4(layer, workspace, *, gated_silu):
        prepare(layer, interleave_gated_silu=gated_silu)
        getattr(layer, tm.STATE_ATTR).op_kind = "nvfp4_qpn4"

    with (
        patch.object(module, "get_current_vllm_config", lambda: cfg),
        patch.object(tm, "use_turbomind", lambda enabled: True),
        patch.object(tm, "should_prepare_turbomind", lambda tensor, enabled: True),
        patch.object(tm, "is_exact_sm70_cuda", lambda tensor, enabled: True),
        patch(
            "vllm.config.sm70_native.capture_linear_native_config",
            return_value=NS(values=()),
        ),
        patch.object(tm, "use_batched_gemm_layouts", lambda: batch),
        patch.object(tm, "prepare_nvfp4_linear", prepare),
        patch.object(
            tm,
            "get_nvfp4_qpn4_dense_workspace",
            lambda weight: torch.empty(1) if qpn4_workspace else None,
        ),
        patch.object(tm, "prepare_nvfp4_qpn4_linear", prepare_qpn4),
        patch.object(
            tm,
            "apply_prepared_linear",
            lambda layer, x, bias=None: torch.empty(x.shape[0], n),
        ),
        patch.object(torch.ops, "_C", native_ops),
        patch.object(
            sm70.TurboMindNvFp4LinearKernel,
            "is_supported",
            classmethod(lambda cls, compute_capability=None: (True, None)),
        ),
        patch.object(
            _sm70_ops,
            "nvfp4_qpn2_prepare_sm70",
            lambda w, s: (
                torch.empty_like(w),
                torch.empty(s.shape, dtype=torch.uint8, device="meta"),
            ),
        ),
        patch.object(
            _sm70_ops,
            "nvfp4_qpn2_prepare_scales_sm70",
            lambda s: torch.empty(s.shape, dtype=torch.uint8, device="meta"),
        ),
        patch.object(
            _sm70_ops,
            "nvfp4_qpn2_dispatch_sm70_out",
            dispatch("nvfp4_qpn2_dispatch_sm70_out"),
        ),
        patch.object(
            _sm70_ops,
            "nvfp4_qpn2_prefill_dispatch_sm70_out",
            dispatch("nvfp4_qpn2_prefill_dispatch_sm70_out"),
        ),
        patch.object(
            _sm70_ops,
            "nvfp4_qpn2_tm_dispatch_sm70_out",
            dispatch("nvfp4_qpn2_tm_dispatch_sm70_out"),
        ),
    ):
        layer = _layer(k, n, role)
        scheme = module.CompressedTensorsW4A4Fp4()
        scheme.process_weights_after_loading(layer)
        state = getattr(layer, tm.STATE_ATTR)
        qpn2 = bool(getattr(layer, "sm70_nvfp4_qpn2", False))
        if qpn2:
            for m in (1, 8, 32, 64, 1024):
                scheme.apply_weights(layer, torch.empty(m, k, dtype=torch.float16))
        return {
            "prepared_layout": state.op_kind,
            "qpn2": qpn2,
            "shared": bool(getattr(layer, "sm70_nvfp4_qpn2_shared_weight", False)),
            "prefill": bool(getattr(layer, "sm70_nvfp4_qpn2_prefill_enabled", False)),
            "compact_scales": state.use_scale_code,
            "batch_prescale": state.prescaled_scales,
            "gated_turbomind": state.gated_silu,
            "dispatch": calls,
        }


def snapshot(module=candidate):
    rows = []
    # Engine dimensions whose independence matters even when they do not alter
    # this particular leaf. Other categories extend the tool in their own PRs.
    for model, kv, tp, spec, concurrency, budget in itertools.product(
        MODELS,
        ("float16", "fp8_e4m3", "fp8_e5m2"),
        (2, 4),
        ("none", "mtp", "dflash"),
        (1, 4, 8),
        (4096, 8192),
    ):
        key = f"{model}/{kv}/tp{tp}/{spec}/c{concurrency}/q{budget}"
        if model != MODELS[0]:
            route = {"status": "not_ct_nvfp4_linear", "qpn2": False}
        else:
            cfg = make_config(spec, tp, concurrency, kv, budget)
            # TP2 and TP4 GDN local logical N; the kernel pads physical N to 32.
            route = snapshot_layer(module, cfg, n=8240 if tp == 2 else 4120, batch=True)
        rows.append({"config": key, "route": route})
    return {
        "schema": 1,
        "scope": "ct_nvfp4_linear",
        "basis": (
            "CPU native-entry dispatch with tensor doubles; not EngineArgs validity, "
            "numerical output or GPU kernel-hit evidence"
        ),
        "cases": rows,
        "edge_cases": snapshot_edges(module),
    }


def snapshot_edges(module=candidate):
    rows = []
    probes = [
        ("native_missing", {}, {"native": False}),
        ("shared_native_missing", {}, {"shared_native": False}),
        ("compact_revision_zero", {}, {"batch": False, "compact_version": 0}),
        ("compact_enabled", {}, {"batch": False}),
        ("batch_supersedes_compact", {}, {"batch": True}),
        ("new_role_retains_fallback", {}, {"role": "unqualified_projection"}),
        ("misaligned_k", {}, {"k": 5000}),
        ("gated_projection", {}, {"role": "gate_up_proj", "n": 8704}),
        ("qpn2_off", {"VLLM_SM70_NVFP4_QPN2": "0"}, {}),
        ("prefill_off", {"VLLM_SM70_NVFP4_QPN2_PREFILL": "0"}, {}),
        ("separate_codes", {"VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT": "0"}, {}),
        (
            "persistent_scales",
            {"VLLM_SM70_NVFP4_QPN2_SHARED_SCALES": "0"},
            {"batch": False},
        ),
        ("legacy_threshold", {"VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M": "9"}, {}),
        ("qpn4_down", {}, {"role": "down_proj", "k": 4352, "n": 5120}),
        (
            "qpn4_gate",
            {"VLLM_SM70_NVFP4_DENSE_GATED_SILU": "1"},
            {"role": "gate_up_proj", "n": 8704},
        ),
        (
            "qpn4_native_missing",
            {},
            {"role": "down_proj", "k": 4352, "n": 5120, "qpn4_native": False},
        ),
        (
            "qpn4_workspace_missing",
            {},
            {"role": "down_proj", "k": 4352, "n": 5120, "qpn4_workspace": False},
        ),
        (
            "qpn4_off",
            {"VLLM_SM70_NVFP4_QPN4": "0"},
            {"role": "down_proj", "k": 4352, "n": 5120},
        ),
    ]
    for name, environment, arguments in probes:
        with patch.dict(os.environ, environment):
            envs.disable_envs_cache()
            cfg = make_config(
                "none" if name.startswith("qpn4_") else "dflash",
                4,
                1 if name.startswith("qpn4_") else 4,
                "fp8_e4m3",
                8192,
            )
            rows.append(
                {"config": name, "route": snapshot_layer(module, cfg, **arguments)}
            )
        envs.disable_envs_cache()
    return rows


def load_baseline(ref, directory):
    source = subprocess.check_output(["git", "show", f"{ref}:{SCHEME}"], text=True)
    path = Path(directory) / "baseline_nvfp4.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("baseline_nvfp4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        choices=("nvfp4", "awq", "fp8-policy", "fp8", "dflash2", "moe"),
        default="nvfp4",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--observed-trace",
        type=Path,
        help="Attach a NativeDispatchTrace JSON as separate observed evidence",
    )
    parser.add_argument(
        "--expected-changes",
        type=Path,
        help="JSON list of exact permitted row IDs for a measured broadening PR",
    )
    parser.add_argument("--check", type=Path, help="Compare with a retained snapshot")
    parser.add_argument(
        "--baseline-ref",
        help="Compare with the independent pre-migration module at this git ref",
    )
    args = parser.parse_args()
    snapshot_fn, load_fn = snapshot, load_baseline
    if args.category == "moe":
        from tools.sm70_moe_route_snapshot import load_baseline as load_fn
        from tools.sm70_moe_route_snapshot import snapshot as snapshot_fn
    elif args.category == "awq":
        from tools.sm70_awq_route_snapshot import load_baseline as load_fn
        from tools.sm70_awq_route_snapshot import snapshot as snapshot_fn
    elif args.category == "fp8":
        from tools.sm70_fp8_route_snapshot import load_baseline as load_fn
        from tools.sm70_fp8_route_snapshot import snapshot as snapshot_fn
    elif args.category == "dflash2":
        from tools.sm70_dflash2_route_snapshot import load_baseline as load_fn
        from tools.sm70_dflash2_route_snapshot import snapshot as snapshot_fn
    # Explicit test sandbox: an interactive shell's production knobs must not
    # leak into the reproducible defaults matrix.
    clean = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    with patch.dict(os.environ, clean, clear=True):
        envs.disable_envs_cache()
        if args.category == "fp8-policy":
            from tools import sm70_fp8_policy_snapshot as policy

            result = policy.snapshot()
        else:
            result = snapshot_fn()
        reference = None
        if args.baseline_ref:
            if args.category == "fp8-policy":
                reference = policy.baseline_snapshot(args.baseline_ref)
            else:
                with tempfile.TemporaryDirectory() as directory:
                    reference = snapshot_fn(load_fn(args.baseline_ref, directory))
        elif args.check:
            reference = json.loads(args.check.read_text())
        if reference is not None:
            changes = [
                new["config"]
                for old, new in zip(reference["cases"], result["cases"], strict=True)
                if old != new
            ]
            changes += [
                "edge:" + new["config"]
                for old, new in zip(
                    reference["edge_cases"], result["edge_cases"], strict=True
                )
                if old != new
            ]
            expected = (
                json.loads(args.expected_changes.read_text())
                if args.expected_changes
                else []
            )
            if set(changes) != set(expected):
                raise SystemExit(
                    f"Unexpected route differences: actual={changes}, "
                    f"expected={expected}"
                )
            print(
                f"{len(result['cases'])} configurations: "
                f"expected changes={len(changes)}"
            )
        if args.observed_trace:
            result["observed_execution"] = json.loads(args.observed_trace.read_text())
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    envs.disable_envs_cache()


if __name__ == "__main__":
    main()
