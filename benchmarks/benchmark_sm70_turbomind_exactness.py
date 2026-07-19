# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict op-level checks for the SM70 TurboMind dense GEMM path.

This is intentionally small. It checks only the migrated dense AWQ and FP8
TurboMind ops, and fails by default unless the result is bitwise identical to
the selected reference (`torch.equal` and `max_diff == 0`).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import regex as re
import torch

import vllm._custom_ops as ops
from vllm import _sm70_ops as sm70_ops

Mode = Literal[
    "all",
    "awq",
    "fp8",
    "awq_moe",
    "fp8_moe",
    "awq_moe_graph_replay",
    "awq_moe_compile",
]
MoEActual = Literal[
    "batched",
    "batched_per_expert_dispatch",
    "batched_w13_per_expert_dispatch",
    "batched_w2_per_expert_dispatch",
    "dense_out",
    "dense_graphsafe",
    "single_token_dense",
    "single_token_indexed",
    "single_token_compact_w13",
    "legacy_single_token_compact",
]
FP16_MOE_OUTPUT_BOUND = 2.0e-3
FP16_MOE_MAX_NONZERO_ULP_DIFF = 256
FP16_MOE_SIGN_MISMATCH_ABS_BOUND = 2.0e-6
SINGLE_TOKEN_STAGE_ACTUALS = (
    "single_token_dense",
    "single_token_indexed",
    "single_token_compact_w13",
)
AWQ_SINGLE_TOKEN_ACTUALS = (*SINGLE_TOKEN_STAGE_ACTUALS, "legacy_single_token_compact")


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This check requires a CUDA device.")


def _require_sm70(device: torch.device, require_sm70: bool) -> None:
    if not require_sm70:
        return
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(
            "This check was asked to require SM70, but the active device is "
            f"sm_{capability[0]}{capability[1]}."
        )


def _require_torch_op(name: str) -> None:
    if not hasattr(torch.ops._C, name):
        raise RuntimeError(
            f"Missing torch op _C::{name}. Build vLLM with CUDA arch 7.0 and "
            "the SM70 TurboMind extension before running this check."
        )


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % 8 != 0:
        raise ValueError("The last dimension must be divisible by 8.")
    chunks = values.reshape(*values.shape[:-1], values.shape[-1] // 8, 8)
    packed = torch.zeros(chunks.shape[:-1], dtype=torch.int64, device=values.device)
    for i in range(8):
        packed |= (chunks[..., i].to(torch.int64) & 0xF) << (4 * i)
    return packed.to(torch.int32)


def _unpack_int4_cols(
    packed: torch.Tensor, order: list[int] | None = None
) -> torch.Tensor:
    vals = [((packed >> (4 * i)) & 0xF).to(torch.float16) for i in range(8)]
    if order is not None:
        vals = [vals[i] for i in order]
    return torch.stack(vals, dim=-1).reshape(packed.shape[0], packed.shape[1] * 8)


def _unpack_awq_gemm_qweight(qweight: torch.Tensor) -> torch.Tensor:
    k = qweight.shape[0]
    n = qweight.shape[1] * 8
    if k % 8 != 0:
        raise ValueError("AWQ qweight K must be divisible by 8.")
    vals = [((qweight >> (4 * i)) & 0xF).to(torch.float16) for i in range(8)]
    flat = torch.stack(vals, dim=-1).reshape(-1)
    return flat.reshape(n, k // 8, 2, 4).permute(0, 1, 3, 2).contiguous().reshape(k, n)


def _awq_reference_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    # Matches the AWQ unpacking used by awq_sm70_prepare. The upstream
    # awq_dequantize CUDA reference uses SM75-only inline PTX and is not a
    # valid SM70 quality oracle.
    zero_order = [0, 4, 1, 5, 2, 6, 3, 7]
    w = _unpack_awq_gemm_qweight(qweight)
    zeros = _unpack_int4_cols(qzeros, zero_order)
    row_groups = (
        torch.arange(qweight.shape[0], device=qweight.device) // group_size
    ).long()
    return ((w - zeros[row_groups]) * scales[row_groups]).to(torch.float16)


def _load_awq_layer(
    model_path: Path,
    layer_prefix: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    filename = weight_map[f"{layer_prefix}.qweight"]
    with safe_open(model_path / filename, framework="pt", device="cpu") as f:
        qweight = f.get_tensor(f"{layer_prefix}.qweight").to(device).contiguous()
        qzeros = f.get_tensor(f"{layer_prefix}.qzeros").to(device).contiguous()
        scales = f.get_tensor(f"{layer_prefix}.scales").to(device).contiguous()
    return qweight, scales, qzeros


def _load_tensors_by_key(
    model_path: Path,
    keys: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    keys_by_file: dict[str, list[str]] = {}
    for key in keys:
        filename = weight_map[key]
        keys_by_file.setdefault(filename, []).append(key)

    loaded: dict[str, torch.Tensor] = {}
    for filename, file_keys in keys_by_file.items():
        with safe_open(model_path / filename, framework="pt", device="cpu") as f:
            for key in file_keys:
                loaded[key] = f.get_tensor(key).to(device).contiguous()
    return loaded


def _load_awq_moe_layer(
    model_path: Path,
    layer_prefix: str,
    num_experts: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    keys: list[str] = []
    for expert_id in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"{layer_prefix}.{expert_id}.{proj}"
            keys.extend(
                [
                    f"{prefix}.qweight",
                    f"{prefix}.qzeros",
                    f"{prefix}.scales",
                ]
            )
    loaded = _load_tensors_by_key(model_path, keys, device)

    w13_qweights, w13_scales, w13_qzeros = [], [], []
    w2_qweights, w2_scales, w2_qzeros = [], [], []
    for expert_id in range(num_experts):
        gate = f"{layer_prefix}.{expert_id}.gate_proj"
        up = f"{layer_prefix}.{expert_id}.up_proj"
        down = f"{layer_prefix}.{expert_id}.down_proj"
        w13_qweights.append(
            torch.cat(
                [loaded[f"{gate}.qweight"], loaded[f"{up}.qweight"]],
                dim=-1,
            )
        )
        w13_scales.append(
            torch.cat([loaded[f"{gate}.scales"], loaded[f"{up}.scales"]], dim=-1)
        )
        w13_qzeros.append(
            torch.cat([loaded[f"{gate}.qzeros"], loaded[f"{up}.qzeros"]], dim=-1)
        )
        w2_qweights.append(loaded[f"{down}.qweight"])
        w2_scales.append(loaded[f"{down}.scales"])
        w2_qzeros.append(loaded[f"{down}.qzeros"])

    return (
        torch.stack(w13_qweights),
        torch.stack(w13_scales),
        torch.stack(w13_qzeros),
        torch.stack(w2_qweights),
        torch.stack(w2_scales),
        torch.stack(w2_qzeros),
    )


def _load_fp8_layer(
    model_path: Path,
    layer_prefix: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    filename = weight_map[f"{layer_prefix}.weight"]
    with safe_open(model_path / filename, framework="pt", device="cpu") as f:
        qweight = f.get_tensor(f"{layer_prefix}.weight").to(device).contiguous()
        scales = (
            f.get_tensor(f"{layer_prefix}.weight_scale_inv")
            .to(device)
            .to(torch.float32)
            .contiguous()
        )
    return qweight, scales


def _load_fp8_moe_layer(
    model_path: Path,
    layer_prefix: str,
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys: list[str] = []
    for expert_id in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"{layer_prefix}.{expert_id}.{proj}"
            keys.extend(
                [
                    f"{prefix}.weight",
                    f"{prefix}.weight_scale_inv",
                ]
            )
    loaded = _load_tensors_by_key(model_path, keys, device)

    w13_weights, w13_scales = [], []
    w2_weights, w2_scales = [], []
    for expert_id in range(num_experts):
        gate = f"{layer_prefix}.{expert_id}.gate_proj"
        up = f"{layer_prefix}.{expert_id}.up_proj"
        down = f"{layer_prefix}.{expert_id}.down_proj"
        w13_weights.append(
            torch.cat([loaded[f"{gate}.weight"], loaded[f"{up}.weight"]], dim=0)
        )
        w13_scales.append(
            torch.cat(
                [
                    loaded[f"{gate}.weight_scale_inv"],
                    loaded[f"{up}.weight_scale_inv"],
                ],
                dim=0,
            ).float()
        )
        w2_weights.append(loaded[f"{down}.weight"])
        w2_scales.append(loaded[f"{down}.weight_scale_inv"].float())

    return (
        torch.stack(w13_weights),
        torch.stack(w13_scales),
        torch.stack(w2_weights),
        torch.stack(w2_scales),
    )


def _make_input(m: int, k: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(m * k, device=device, dtype=torch.int32)
    values = ((values % 1024).to(torch.float32) / 512.0) - 1.0
    return values.reshape(m, k).to(torch.float16)


def _make_moe_input(
    m: int,
    k: int,
    device: torch.device,
    pattern: str,
) -> torch.Tensor:
    if pattern == "range":
        return _make_input(m, k, device)
    if pattern == "random":
        return torch.randn((m, k), device=device, dtype=torch.float16)
    if pattern == "random_scaled":
        return torch.randn((m, k), device=device, dtype=torch.float16) * 3.0
    raise ValueError(f"Unsupported MoE input pattern: {pattern}")


def _stats(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    torch.accelerator.synchronize(actual.device)
    diff = (actual - expected).abs()
    finite_diff = diff[torch.isfinite(diff)]
    max_diff = float(finite_diff.max().item()) if finite_diff.numel() else float("nan")
    mean_diff = (
        float(finite_diff.float().mean().item())
        if finite_diff.numel()
        else float("nan")
    )
    rms_diff = (
        float(torch.sqrt(torch.mean(finite_diff.float() ** 2)).item())
        if finite_diff.numel()
        else float("nan")
    )
    equal = torch.equal(actual, expected)
    result = {
        "name": name,
        "equal": equal,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rms_diff": rms_diff,
        "num_different": int((actual != expected).sum().item()),
        "actual_nan_count": int(torch.isnan(actual).sum().item()),
        "expected_nan_count": int(torch.isnan(expected).sum().item()),
        "finite_diff_count": int(finite_diff.numel()),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "shape": list(actual.shape),
    }
    if actual.dtype == torch.float16 and expected.dtype == torch.float16:
        actual_np = actual.detach().cpu().contiguous().numpy()
        expected_np = expected.detach().cpu().contiguous().numpy()
        finite_mask = np.isfinite(actual_np) & np.isfinite(expected_np)
        if finite_mask.any():
            actual_bits = actual_np.view(np.uint16).astype(np.int32)
            expected_bits = expected_np.view(np.uint16).astype(np.int32)
            sign_mismatch = ((actual_bits ^ expected_bits) & 0x8000)[finite_mask] != 0

            def _ordered_float16_bits(bits: np.ndarray) -> np.ndarray:
                sign = bits & 0x8000
                magnitude = bits & 0x7FFF
                return np.where(sign != 0, 0x8000 - magnitude, bits)

            finite_abs_diff = np.abs(
                actual_np.astype(np.float32) - expected_np.astype(np.float32)
            )[finite_mask]
            ulp_diff = np.abs(
                _ordered_float16_bits(actual_bits)
                - _ordered_float16_bits(expected_bits)
            )[finite_mask]
            nonzero_ulp_mask = ulp_diff != 0
            nonzero_ulp = ulp_diff[nonzero_ulp_mask]
            max_abs_diff_at_max_nonzero_ulp = 0.0
            if nonzero_ulp.size:
                nonzero_positions = np.flatnonzero(nonzero_ulp_mask)
                max_ulp_position = nonzero_positions[np.argmax(nonzero_ulp)]
                max_abs_diff_at_max_nonzero_ulp = float(
                    finite_abs_diff[max_ulp_position]
                )
            result.update(
                {
                    "max_ulp_diff": int(ulp_diff.max()) if ulp_diff.size else 0,
                    "mean_ulp_diff": float(ulp_diff.mean()) if ulp_diff.size else 0.0,
                    "max_nonzero_ulp_diff": int(nonzero_ulp.max())
                    if nonzero_ulp.size
                    else 0,
                    "max_abs_diff_at_max_nonzero_ulp": max_abs_diff_at_max_nonzero_ulp,
                    "mean_nonzero_ulp_diff": float(nonzero_ulp.mean())
                    if nonzero_ulp.size
                    else 0.0,
                    "sign_mismatch_count": int(sign_mismatch.sum()),
                    "max_abs_diff_on_sign_mismatch": float(
                        finite_abs_diff[sign_mismatch].max()
                    )
                    if sign_mismatch.any()
                    else 0.0,
                }
            )
    return result


def _with_moe_metadata(
    result: dict[str, Any],
    *,
    m: int,
    top_k: int,
    num_experts: int,
    actual_impl: MoEActual,
    expert_pattern: str,
    stage: str,
    layer: str | None = None,
) -> dict[str, Any]:
    result.update(
        {
            "m": m,
            "top_k": top_k,
            "total_slots": m * top_k,
            "num_experts": num_experts,
            "actual_impl": actual_impl,
            "expert_pattern": expert_pattern,
            "stage": stage,
        }
    )
    if layer is not None:
        result["layer"] = layer
    return result


def _classify_moe_results(
    *,
    mode: Mode,
    results: list[dict[str, Any]],
    actual_impl: MoEActual | None,
    max_diff_bound: float,
    max_nonzero_ulp_diff: int,
    sign_mismatch_abs_bound: float,
    model_quality_gate_passed: bool,
) -> dict[str, Any] | None:
    if mode not in ("awq_moe", "fp8_moe"):
        return None

    nonzero = [r for r in results if not r["equal"] or r["max_diff"] != 0.0]
    nan_reports = [
        r for r in results if r["actual_nan_count"] != 0 or r["expected_nan_count"] != 0
    ]
    max_diff = max((float(r["max_diff"]) for r in results), default=0.0)
    max_mean_diff = max((float(r["mean_diff"]) for r in results), default=0.0)
    above_bound = [r for r in results if float(r["max_diff"]) > max_diff_bound]
    raw_above_ulp_bound = [
        r
        for r in results
        if int(r.get("max_nonzero_ulp_diff", 0)) > max_nonzero_ulp_diff
    ]
    above_ulp_bound = [
        r
        for r in raw_above_ulp_bound
        if float(r.get("max_abs_diff_at_max_nonzero_ulp", float("inf")))
        > sign_mismatch_abs_bound
    ]
    near_zero_ulp_bound = [
        r
        for r in raw_above_ulp_bound
        if float(r.get("max_abs_diff_at_max_nonzero_ulp", float("inf")))
        <= sign_mismatch_abs_bound
    ]
    bad_sign_mismatch = [
        r
        for r in results
        if int(r.get("sign_mismatch_count", 0)) > 0
        and float(r.get("max_abs_diff_on_sign_mismatch", 0.0)) > sign_mismatch_abs_bound
    ]
    stages: dict[str, dict[str, Any]] = {}
    for result in results:
        stage = str(result.get("stage") or _stage_from_name(result["name"]))
        current = stages.setdefault(
            stage,
            {
                "max_diff": 0.0,
                "max_mean_diff": 0.0,
                "max_rms_diff": 0.0,
                "max_nonzero_ulp_diff": 0,
                "max_abs_diff_at_max_nonzero_ulp": 0.0,
                "sign_mismatch_count": 0,
                "max_abs_diff_on_sign_mismatch": 0.0,
                "num_reports": 0,
                "num_nonzero": 0,
            },
        )
        current["num_reports"] += 1
        current["max_diff"] = max(current["max_diff"], float(result["max_diff"]))
        current["max_mean_diff"] = max(
            current["max_mean_diff"], float(result["mean_diff"])
        )
        current["max_rms_diff"] = max(
            current["max_rms_diff"], float(result["rms_diff"])
        )
        current["max_nonzero_ulp_diff"] = max(
            current["max_nonzero_ulp_diff"],
            int(result.get("max_nonzero_ulp_diff", 0)),
        )
        current["max_abs_diff_at_max_nonzero_ulp"] = max(
            current["max_abs_diff_at_max_nonzero_ulp"],
            float(result.get("max_abs_diff_at_max_nonzero_ulp", 0.0)),
        )
        current["sign_mismatch_count"] += int(result.get("sign_mismatch_count", 0))
        current["max_abs_diff_on_sign_mismatch"] = max(
            current["max_abs_diff_on_sign_mismatch"],
            float(result.get("max_abs_diff_on_sign_mismatch", 0.0)),
        )
        if not result["equal"] or result["max_diff"] != 0.0:
            current["num_nonzero"] += 1

    cleared = []
    pending = []
    failed = []
    op_b_accept_candidate = (
        bool(nonzero)
        and not nan_reports
        and not above_bound
        and not above_ulp_bound
        and not bad_sign_mismatch
    )
    if not nan_reports:
        cleared.append("no NaNs in actual or expected outputs")
    else:
        failed.append(f"{len(nan_reports)} reports contain NaNs")

    if actual_impl in ("dense_out", "dense_graphsafe", "single_token_dense"):
        if nonzero:
            label = "A-bug"
            failed.append("per-expert dense bridge is expected to be Type-A exact")
            default_acceptance = "blocked until dense bridge exactness is repaired"
        else:
            label = "A-pass"
            cleared.append("per-expert dense bridge matches the oracle exactly")
            default_acceptance = "op-level exact bridge"
    elif not nonzero:
        label = "A-pass"
        cleared.append("grouped/batched path is bitwise exact for this run")
        default_acceptance = "op-level exact for this run"
    elif not op_b_accept_candidate:
        label = "B-fail"
        default_acceptance = "blocked: fp16-diff acceptance gate failed"
        if above_bound:
            failed.append(
                f"{len(above_bound)} reports exceed fp16 output bound {max_diff_bound}"
            )
        if above_ulp_bound:
            failed.append(
                f"{len(above_ulp_bound)} reports exceed nonzero ULP bound "
                f"{max_nonzero_ulp_diff}"
            )
        if bad_sign_mismatch:
            failed.append(
                f"{len(bad_sign_mismatch)} reports have sign mismatches above "
                f"abs bound {sign_mismatch_abs_bound}"
            )
    elif model_quality_gate_passed:
        label = "B-accept"
        default_acceptance = "accepted fp16 dispatch difference with model gate"
        cleared.append(
            "op-level fp16 diff gate passed and model-level quality gate was "
            "explicitly marked passed"
        )
    else:
        label = "B-pending"
        default_acceptance = "not default-accepted"
        cleared.append(
            f"all nonzero diffs are within fp16 output bound {max_diff_bound}"
        )
        cleared.append(
            "all nonzero ULP/sign-mismatch stats are within the configured "
            "op-level fp16 gate"
        )
        if near_zero_ulp_bound:
            cleared.append(
                f"{len(near_zero_ulp_bound)} raw ULP outlier reports are "
                "near-zero cases under the configured abs gate"
            )
        pending.append("grouped TurboMind schedule/reduction source is not yet proven")
        pending.append(
            "model-level greedy/logprob/perplexity quality gate has not been "
            "marked passed"
        )
        pending.append(
            "StridedPtr/offset bugs are weakened by diagnostics but not fully "
            "mechanically excluded in this summary"
        )

    return {
        "label": label,
        "path_type": f"{mode}:{actual_impl}",
        "default_acceptance": default_acceptance,
        "fp16_nominal_output_bound": max_diff_bound,
        "fp16_max_nonzero_ulp_diff_bound": max_nonzero_ulp_diff,
        "fp16_sign_mismatch_abs_bound": sign_mismatch_abs_bound,
        "fp16_large_ulp_abs_bound": sign_mismatch_abs_bound,
        "model_quality_gate_passed": model_quality_gate_passed,
        "op_b_accept_candidate": op_b_accept_candidate,
        "num_reports": len(results),
        "num_nonzero": len(nonzero),
        "num_nan_reports": len(nan_reports),
        "num_above_fp16_bound": len(above_bound),
        "num_raw_above_ulp_bound": len(raw_above_ulp_bound),
        "num_above_ulp_bound": len(above_ulp_bound),
        "num_near_zero_ulp_reports": len(near_zero_ulp_bound),
        "num_bad_sign_mismatch_reports": len(bad_sign_mismatch),
        "max_diff": max_diff,
        "max_mean_diff": max_mean_diff,
        "stages": stages,
        "cleared_evidence": cleared,
        "pending_evidence": pending,
        "failed_evidence": failed,
    }


def _stage_from_name(name: str) -> str:
    match = re.search(r"_(w13_gate_up|silu_and_mul|w2_sorted_output)$", name)
    return match.group(1) if match else "unknown"


def _check_awq(
    m: int,
    k: int,
    n: int,
    group_size: int,
    device: torch.device,
    awq_model: Path | None,
    awq_layer: str | None,
) -> dict[str, Any]:
    _require_torch_op("awq_sm70_prepare")
    _require_torch_op("awq_gemm_sm70")

    if awq_model is not None:
        if awq_layer is None:
            raise ValueError("--awq-layer is required when --awq-model is set.")
        qweight, scales, qzeros = _load_awq_layer(awq_model, awq_layer, device)
        k = qweight.shape[0]
        n = qweight.shape[1] * 8
    else:
        if k % group_size != 0:
            raise ValueError("AWQ K must be divisible by group_size.")
        if k % 8 != 0 or n % 8 != 0:
            raise ValueError("AWQ K and N must be divisible by 8.")

        unpacked_qweight = torch.randint(
            0, 16, (k, n), device=device, dtype=torch.int32
        )
        unpacked_qzeros = torch.randint(
            0, 16, (k // group_size, n), device=device, dtype=torch.int32
        )
        qweight = _pack_int4(unpacked_qweight)
        qzeros = _pack_int4(unpacked_qzeros)
        scales = (
            torch.rand((k // group_size, n), device=device, dtype=torch.float16) * 0.25
            + 0.01
        )

    if k % group_size != 0:
        raise ValueError("AWQ K must be divisible by group_size.")

    x = torch.randn((m, k), device=device, dtype=torch.float16)

    tm_weight, tm_scales, meta = sm70_ops.awq_sm70_prepare(
        qweight, scales, qzeros, group_size
    )
    actual = sm70_ops.awq_gemm_sm70(
        x,
        tm_weight,
        tm_scales,
        group_size,
        int(meta[0].item()),
        int(meta[1].item()),
    )
    ref_weight = _awq_reference_weight(qweight, scales, qzeros, group_size)
    expected = torch.matmul(x, ref_weight)
    name = f"awq_m{m}_k{k}_n{n}_g{group_size}"
    if awq_model is not None and awq_layer is not None:
        name = f"{name}_{awq_layer}"
    return _stats(name, actual, expected)


def _prepare_awq_moe_weights(
    qweights: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    tm_weights, tm_scales, metas = [], [], []
    for expert_id in range(qweights.shape[0]):
        tm_weight, tm_scale, meta = sm70_ops.awq_sm70_prepare(
            qweights[expert_id],
            scales[expert_id],
            qzeros[expert_id],
            group_size,
            interleave_gated_silu,
        )
        tm_weights.append(tm_weight)
        tm_scales.append(tm_scale)
        metas.append(meta)

    stacked_weights = torch.stack(tm_weights)
    stacked_scales = torch.stack(tm_scales)
    k_ld = int(metas[0][0].item())
    q_ld = int(metas[0][1].item())
    ptrs = sm70_ops.awq_moe_build_strided_ptrs(
        stacked_weights,
        stacked_scales,
        k_ld,
        q_ld,
        qweights.shape[0],
    )
    return stacked_weights, stacked_scales, ptrs[0], ptrs[1], k_ld, q_ld


def _prepare_fp8_moe_weights(
    weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tm_weights, tm_scales, metas = [], [], []
    for expert_id in range(weights.shape[0]):
        tm_weight, tm_scale, meta = sm70_ops.fp8_sm70_prepare(
            weights[expert_id],
            scales[expert_id],
            group_size,
        )
        tm_weights.append(tm_weight)
        tm_scales.append(tm_scale)
        metas.append(meta)

    stacked_weights = torch.stack(tm_weights)
    stacked_scales = torch.stack(tm_scales)
    stacked_metas = torch.stack(metas)
    k_ld = int(stacked_metas[0][0].item())
    q_ld = int(stacked_metas[0][1].item())
    ptrs = sm70_ops.awq_moe_build_strided_ptrs(
        stacked_weights,
        stacked_scales,
        k_ld,
        q_ld,
        weights.shape[0],
    )
    return stacked_weights, stacked_scales, stacked_metas, ptrs[0], ptrs[1]


def _expert_offsets(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(expert_ids, minlength=num_experts)
    offsets64 = torch.empty(
        num_experts + 1,
        dtype=torch.int64,
        device=expert_ids.device,
    )
    offsets64[0] = 0
    torch.cumsum(counts, dim=0, out=offsets64[1:])
    return offsets64.to(torch.int32), offsets64


def _make_logical_expert_ids(
    total_slots: int,
    num_experts: int,
    device: torch.device,
    pattern: str,
) -> torch.Tensor:
    if pattern == "round_robin":
        expert_ids = torch.arange(total_slots, device=device, dtype=torch.int64)
        return expert_ids % num_experts
    if pattern == "single":
        return torch.zeros(total_slots, device=device, dtype=torch.int64)
    if pattern.startswith("single:"):
        expert_id = int(pattern.split(":", 1)[1])
        if expert_id < 0 or expert_id >= num_experts:
            raise ValueError(
                f"single expert id {expert_id} is outside [0, {num_experts})."
            )
        return torch.full(
            (total_slots,),
            expert_id,
            device=device,
            dtype=torch.int64,
        )
    if pattern.startswith("random_unique"):
        seed = 0
        if ":" in pattern:
            seed = int(pattern.split(":", 1)[1])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        if total_slots <= num_experts:
            expert_ids = torch.randperm(num_experts, generator=gen)[:total_slots]
        else:
            repeats = (total_slots + num_experts - 1) // num_experts
            expert_ids = torch.cat(
                [torch.randperm(num_experts, generator=gen) for _ in range(repeats)]
            )[:total_slots]
        return expert_ids.to(device=device, dtype=torch.int64)
    if pattern.startswith("random"):
        seed = 0
        if ":" in pattern:
            seed = int(pattern.split(":", 1)[1])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        return torch.randint(
            0,
            num_experts,
            (total_slots,),
            generator=gen,
            dtype=torch.int64,
        ).to(device=device)
    raise ValueError(f"Unsupported MoE expert pattern: {pattern}")


def _make_topk_weights(top_k: int, device: torch.device) -> torch.Tensor:
    weights = torch.arange(1, top_k + 1, device=device, dtype=torch.float32)
    return (weights / weights.sum()).view(1, top_k).contiguous()


def _make_router_logits_for_pattern(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    logits = torch.full(
        (topk_ids.shape[0], num_experts),
        -1000.0,
        dtype=torch.float32,
        device=topk_ids.device,
    )
    ranks = torch.arange(
        topk_ids.shape[1],
        device=topk_ids.device,
        dtype=torch.float32,
    ).view(1, -1)
    logits.scatter_(1, topk_ids.to(torch.int64), 1000.0 - ranks)
    return logits.contiguous()


def _dense_awq_moe_stage(
    sorted_input: torch.Tensor,
    expert_offsets64: torch.Tensor,
    tm_weights: torch.Tensor,
    tm_scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    n: int,
) -> torch.Tensor:
    out = torch.empty(
        (sorted_input.shape[0], n),
        dtype=sorted_input.dtype,
        device=sorted_input.device,
    )
    for expert_id in range(tm_weights.shape[0]):
        start = int(expert_offsets64[expert_id].item())
        end = int(expert_offsets64[expert_id + 1].item())
        if start == end:
            continue
        out[start:end] = sm70_ops.awq_gemm_sm70(
            sorted_input[start:end],
            tm_weights[expert_id],
            tm_scales[expert_id],
            group_size,
            k_ld,
            q_ld,
        )
    return out


def _dense_awq_moe_stage_out(
    sorted_input: torch.Tensor,
    expert_offsets64: torch.Tensor,
    tm_weights: torch.Tensor,
    tm_scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    n: int,
) -> torch.Tensor:
    out = torch.empty(
        (sorted_input.shape[0], n),
        dtype=sorted_input.dtype,
        device=sorted_input.device,
    )
    for expert_id in range(tm_weights.shape[0]):
        start = int(expert_offsets64[expert_id].item())
        end = int(expert_offsets64[expert_id + 1].item())
        if start == end:
            continue
        sm70_ops.awq_gemm_sm70_out(
            out[start:end],
            sorted_input[start:end],
            tm_weights[expert_id],
            tm_scales[expert_id],
            group_size,
            k_ld,
            q_ld,
            False,
        )
    return out


def _dense_fp8_moe_stage_out(
    sorted_input: torch.Tensor,
    expert_offsets64: torch.Tensor,
    tm_weights: torch.Tensor,
    tm_scales: torch.Tensor,
    metas: torch.Tensor,
    n: int,
) -> torch.Tensor:
    out = torch.empty(
        (sorted_input.shape[0], n),
        dtype=sorted_input.dtype,
        device=sorted_input.device,
    )
    for expert_id in range(tm_weights.shape[0]):
        start = int(expert_offsets64[expert_id].item())
        end = int(expert_offsets64[expert_id + 1].item())
        if start == end:
            continue
        sm70_ops.fp8_gemm_sm70_out_meta(
            out[start:end],
            sorted_input[start:end],
            tm_weights[expert_id],
            tm_scales[expert_id],
            metas[expert_id],
        )
    return out


def _check_awq_moe(
    m: int,
    group_size: int,
    device: torch.device,
    awq_moe_model: Path,
    awq_moe_layer: str,
    num_experts: int,
    top_k: int,
    actual_impl: MoEActual,
    expert_pattern: str,
) -> list[dict[str, Any]]:
    _require_torch_op("awq_sm70_prepare")
    _require_torch_op("awq_gemm_sm70")
    _require_torch_op("awq_moe_build_strided_ptrs")
    _require_torch_op("awq_moe_gemm_sm70_out")
    if actual_impl in (
        "batched",
        "batched_per_expert_dispatch",
        "batched_w13_per_expert_dispatch",
        "batched_w2_per_expert_dispatch",
    ):
        _require_torch_op("awq_moe_gemm_sm70_per_expert_dispatch_out")
    if actual_impl in ("dense_graphsafe", "batched_w2_per_expert_dispatch"):
        _require_torch_op("awq_moe_dense_stage_sm70_out")
    if actual_impl == "active_dense_stage":
        _require_torch_op("awq_moe_dense_stage_sm70_out")
        _require_torch_op("awq_moe_active_dense_stage_sm70_out")
    if actual_impl in AWQ_SINGLE_TOKEN_ACTUALS:
        _require_torch_op("awq_moe_single_token_dense_w13_sm70_out")
        _require_torch_op("awq_moe_single_token_dense_stage_sm70_out")
        if actual_impl == "single_token_indexed":
            _require_torch_op("awq_moe_single_token_indexed_dense_w13_sm70_out")
            _require_torch_op("awq_moe_single_token_indexed_dense_stage_sm70_out")
        if actual_impl == "single_token_compact_w13":
            _require_torch_op("awq_moe_single_token_compact_dense_w13_sm70_out")
        if actual_impl == "legacy_single_token_compact":
            _require_torch_op("awq_moe_single_token_sm70_out")
            _require_torch_op("awq_moe_single_token_weighted_reduce_out")
        if m != 1:
            raise ValueError(f"{actual_impl} AWQ MoE check requires --m 1.")

    (
        w13_qweight,
        w13_scales,
        w13_qzeros,
        w2_qweight,
        w2_scales,
        w2_qzeros,
    ) = _load_awq_moe_layer(awq_moe_model, awq_moe_layer, num_experts, device)

    w13_tm_weight, w13_tm_scales, w13_ptrs_w, w13_ptrs_s, w13_k_ld, w13_q_ld = (
        _prepare_awq_moe_weights(w13_qweight, w13_scales, w13_qzeros, group_size)
    )
    w2_tm_weight, w2_tm_scales, w2_ptrs_w, w2_ptrs_s, w2_k_ld, w2_q_ld = (
        _prepare_awq_moe_weights(w2_qweight, w2_scales, w2_qzeros, group_size)
    )
    if actual_impl == "legacy_single_token_compact":
        (
            _w13_legacy_tm_weight,
            _w13_legacy_tm_scales,
            w13_legacy_ptrs_w,
            w13_legacy_ptrs_s,
            _w13_legacy_k_ld,
            _w13_legacy_q_ld,
        ) = _prepare_awq_moe_weights(
            w13_qweight,
            w13_scales,
            w13_qzeros,
            group_size,
            interleave_gated_silu=True,
        )
    else:
        _w13_legacy_tm_weight = w13_tm_weight
        w13_legacy_ptrs_w = w13_ptrs_w
        w13_legacy_ptrs_s = w13_ptrs_s

    total_slots = m * top_k
    logical_expert_ids = _make_logical_expert_ids(
        total_slots,
        num_experts,
        device,
        expert_pattern,
    )
    topk_ids = logical_expert_ids.view(m, top_k)
    sorted_expert_ids, order = torch.sort(logical_expert_ids)
    expert_offsets, expert_offsets64 = _expert_offsets(sorted_expert_ids, num_experts)
    dense_expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)

    hidden_size = int(w13_qweight.shape[1])
    if actual_impl in AWQ_SINGLE_TOKEN_ACTUALS:
        single_token_input = _make_input(1, hidden_size, device)
        sorted_input = single_token_input.expand(total_slots, hidden_size).contiguous()
        sorted_input = sorted_input[order]
    else:
        single_token_input = None
        sorted_input = _make_input(total_slots, hidden_size, device)[order]
    w13_n = int(w13_qweight.shape[2]) * 8
    intermediate_size = w13_n // 2
    hidden_out = int(w2_qweight.shape[2]) * 8

    gate_up_actual = torch.empty(
        (total_slots, w13_n), dtype=torch.float16, device=device
    )
    intermediate_actual: torch.Tensor | None = None
    sorted_output_actual: torch.Tensor | None = None
    final_output_actual: torch.Tensor | None = None
    final_output_expected: torch.Tensor | None = None
    single_token_offsets = None
    single_token_offsets64 = None
    single_token_inv_permuted_idx = None
    single_token_sorted_expert_ids = None
    if actual_impl in (
        "batched",
        "batched_per_expert_dispatch",
        "batched_w13_per_expert_dispatch",
    ):
        sm70_ops.awq_moe_gemm_sm70_per_expert_dispatch_out(
            gate_up_actual,
            sorted_input,
            expert_offsets,
            w13_ptrs_w,
            w13_ptrs_s,
            num_experts,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
        )
    elif actual_impl == "batched_w2_per_expert_dispatch" or actual_impl in (
        "dense_graphsafe",
        "active_dense_stage",
    ):
        sm70_ops.awq_moe_dense_stage_sm70_out(
            gate_up_actual,
            sorted_input,
            expert_offsets,
            dense_expert_ids,
            w13_ptrs_w,
            w13_ptrs_s,
            num_experts,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
        )
    elif actual_impl in AWQ_SINGLE_TOKEN_ACTUALS:
        single_token_offsets = torch.empty(top_k + 1, dtype=torch.int32, device=device)
        single_token_offsets64 = torch.empty(
            top_k + 1, dtype=torch.int64, device=device
        )
        single_token_inv_permuted_idx = torch.empty(
            top_k, dtype=torch.int32, device=device
        )
        single_token_sorted_expert_ids = torch.empty(
            top_k, dtype=torch.int32, device=device
        )
        if actual_impl == "single_token_indexed":
            sm70_ops.awq_moe_single_token_indexed_dense_w13_sm70_out(
                gate_up_actual,
                torch.empty_like(sorted_input),
                single_token_input,
                topk_ids.to(torch.int32).contiguous(),
                w13_ptrs_w,
                w13_ptrs_s,
                single_token_offsets,
                single_token_offsets64,
                single_token_inv_permuted_idx,
                single_token_sorted_expert_ids,
                int(w13_tm_weight.shape[1]),
                w13_n,
                group_size,
                hidden_size,
            )
        elif actual_impl == "legacy_single_token_compact":
            topk_weights = _make_topk_weights(top_k, device)
            sorted_weights = torch.empty(top_k, dtype=torch.float32, device=device)
            compact_input = torch.empty_like(sorted_input)
            intermediate_actual = torch.empty(
                (total_slots, intermediate_size),
                dtype=torch.float16,
                device=device,
            )
            sorted_output_actual = torch.empty(
                (total_slots, hidden_out),
                dtype=torch.float16,
                device=device,
            )
            final_output_actual = torch.empty(
                (1, hidden_size),
                dtype=torch.float16,
                device=device,
            )
            final_output_actual.zero_()
            ptr_row_bytes = int(w13_ptrs_w.numel() // num_experts)
            compact_ptrs_w13_w = torch.empty(
                (top_k, ptr_row_bytes), dtype=torch.uint8, device=device
            )
            compact_ptrs_w13_s = torch.empty_like(compact_ptrs_w13_w)
            compact_ptrs_w2_w = torch.empty_like(compact_ptrs_w13_w)
            compact_ptrs_w2_s = torch.empty_like(compact_ptrs_w13_w)
            sm70_ops.awq_moe_single_token_sm70_out(
                final_output_actual,
                single_token_input,
                topk_weights,
                topk_ids.to(torch.int32).contiguous(),
                w13_legacy_ptrs_w.view(num_experts, ptr_row_bytes),
                w13_legacy_ptrs_s.view(num_experts, ptr_row_bytes),
                w2_ptrs_w.view(num_experts, ptr_row_bytes),
                w2_ptrs_s.view(num_experts, ptr_row_bytes),
                compact_input,
                intermediate_actual,
                sorted_output_actual,
                sorted_weights,
                compact_ptrs_w13_w,
                compact_ptrs_w13_s,
                compact_ptrs_w2_w,
                compact_ptrs_w2_s,
                single_token_offsets,
                single_token_inv_permuted_idx,
                int(w13_tm_weight.shape[1]),
                w13_n,
                int(w2_tm_weight.shape[1]),
                hidden_out,
                group_size,
                hidden_size,
            )
        elif actual_impl == "single_token_compact_w13":
            compact_ptrs_w = torch.empty(top_k * 16, dtype=torch.uint8, device=device)
            compact_ptrs_s = torch.empty(top_k * 16, dtype=torch.uint8, device=device)
            sm70_ops.awq_moe_single_token_compact_dense_w13_sm70_out(
                gate_up_actual,
                torch.empty_like(sorted_input),
                single_token_input,
                topk_ids.to(torch.int32).contiguous(),
                w13_ptrs_w,
                w13_ptrs_s,
                compact_ptrs_w,
                compact_ptrs_s,
                single_token_offsets,
                single_token_offsets64,
                single_token_inv_permuted_idx,
                single_token_sorted_expert_ids,
                int(w13_tm_weight.shape[1]),
                w13_n,
                group_size,
                hidden_size,
            )
        else:
            sm70_ops.awq_moe_single_token_dense_w13_sm70_out(
                gate_up_actual,
                torch.empty_like(sorted_input),
                single_token_input,
                topk_ids.to(torch.int32).contiguous(),
                w13_ptrs_w,
                w13_ptrs_s,
                single_token_offsets,
                single_token_offsets64,
                single_token_inv_permuted_idx,
                single_token_sorted_expert_ids,
                int(w13_tm_weight.shape[1]),
                w13_n,
                group_size,
                hidden_size,
            )
    else:
        gate_up_actual = _dense_awq_moe_stage_out(
            sorted_input,
            expert_offsets64,
            w13_tm_weight,
            w13_tm_scales,
            group_size,
            w13_k_ld,
            w13_q_ld,
            w13_n,
        )
    gate_up_expected = _dense_awq_moe_stage(
        sorted_input,
        expert_offsets64,
        w13_tm_weight,
        w13_tm_scales,
        group_size,
        w13_k_ld,
        w13_q_ld,
        w13_n,
    )

    if intermediate_actual is None:
        intermediate_actual = torch.empty(
            (total_slots, intermediate_size), dtype=torch.float16, device=device
        )
    intermediate_expected = torch.empty_like(intermediate_actual)
    if actual_impl != "legacy_single_token_compact":
        torch.ops._C.silu_and_mul(intermediate_actual, gate_up_actual)
    torch.ops._C.silu_and_mul(intermediate_expected, gate_up_expected)

    if sorted_output_actual is None:
        sorted_output_actual = torch.empty(
            (total_slots, hidden_out), dtype=torch.float16, device=device
        )
    if actual_impl in ("batched", "batched_w13_per_expert_dispatch"):
        sm70_ops.awq_moe_gemm_sm70_out(
            sorted_output_actual,
            intermediate_actual,
            expert_offsets,
            w2_ptrs_w,
            w2_ptrs_s,
            num_experts,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
            False,
        )
    elif actual_impl in (
        "batched_per_expert_dispatch",
        "batched_w2_per_expert_dispatch",
    ):
        sm70_ops.awq_moe_gemm_sm70_per_expert_dispatch_out(
            sorted_output_actual,
            intermediate_actual,
            expert_offsets,
            w2_ptrs_w,
            w2_ptrs_s,
            num_experts,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
            False,
        )
    elif actual_impl == "dense_graphsafe":
        sm70_ops.awq_moe_dense_stage_sm70_out(
            sorted_output_actual,
            intermediate_actual,
            expert_offsets,
            dense_expert_ids,
            w2_ptrs_w,
            w2_ptrs_s,
            num_experts,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
        )
    elif actual_impl == "active_dense_stage":
        active_expert_offsets = torch.empty(
            total_slots + 1, dtype=torch.int32, device=device
        )
        active_expert_ids = torch.empty(total_slots, dtype=torch.int32, device=device)
        sm70_ops.awq_moe_active_dense_stage_sm70_out(
            sorted_output_actual,
            intermediate_actual,
            sorted_expert_ids.to(torch.int32).contiguous(),
            active_expert_offsets,
            active_expert_ids,
            w2_ptrs_w,
            w2_ptrs_s,
            total_slots,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
        )
    elif actual_impl in AWQ_SINGLE_TOKEN_ACTUALS:
        assert single_token_offsets is not None
        assert single_token_sorted_expert_ids is not None
        if actual_impl == "legacy_single_token_compact":
            pass
        elif actual_impl == "single_token_indexed":
            sm70_ops.awq_moe_single_token_indexed_dense_stage_sm70_out(
                sorted_output_actual,
                intermediate_actual,
                single_token_offsets,
                single_token_sorted_expert_ids,
                w2_ptrs_w,
                w2_ptrs_s,
                top_k,
                int(w2_tm_weight.shape[1]),
                hidden_out,
                group_size,
            )
        else:
            sm70_ops.awq_moe_single_token_dense_stage_sm70_out(
                sorted_output_actual,
                intermediate_actual,
                single_token_offsets,
                single_token_sorted_expert_ids,
                w2_ptrs_w,
                w2_ptrs_s,
                top_k,
                int(w2_tm_weight.shape[1]),
                hidden_out,
                group_size,
            )
    else:
        sorted_output_actual = _dense_awq_moe_stage_out(
            intermediate_actual,
            expert_offsets64,
            w2_tm_weight,
            w2_tm_scales,
            group_size,
            w2_k_ld,
            w2_q_ld,
            hidden_out,
        )
    sorted_output_expected = _dense_awq_moe_stage(
        intermediate_expected,
        expert_offsets64,
        w2_tm_weight,
        w2_tm_scales,
        group_size,
        w2_k_ld,
        w2_q_ld,
        hidden_out,
    )
    if actual_impl == "legacy_single_token_compact":
        assert final_output_actual is not None
        assert single_token_inv_permuted_idx is not None
        topk_weights = _make_topk_weights(top_k, device)
        final_output_expected = torch.empty_like(final_output_actual)
        sm70_ops.awq_moe_single_token_weighted_reduce_out(
            sorted_output_expected[:, :hidden_size],
            topk_weights,
            single_token_inv_permuted_idx,
            final_output_expected,
            top_k,
            hidden_size,
        )

    base = (
        f"awq_moe_{actual_impl}_tokens{m}_topk{top_k}_experts{num_experts}"
        f"_{awq_moe_layer}"
    )
    results = []
    if actual_impl != "legacy_single_token_compact":
        results.append(
            _with_moe_metadata(
                _stats(f"{base}_w13_gate_up", gate_up_actual, gate_up_expected),
                m=m,
                top_k=top_k,
                num_experts=num_experts,
                actual_impl=actual_impl,
                expert_pattern=expert_pattern,
                stage="w13_gate_up",
                layer=awq_moe_layer,
            )
        )
    results.append(
        _with_moe_metadata(
            _stats(
                f"{base}_fused_w13_silu_and_mul"
                if actual_impl == "legacy_single_token_compact"
                else f"{base}_silu_and_mul",
                intermediate_actual,
                intermediate_expected,
            ),
            m=m,
            top_k=top_k,
            num_experts=num_experts,
            actual_impl=actual_impl,
            expert_pattern=expert_pattern,
            stage="fused_w13_silu_and_mul"
            if actual_impl == "legacy_single_token_compact"
            else "silu_and_mul",
            layer=awq_moe_layer,
        )
    )
    if actual_impl != "legacy_single_token_compact":
        results.append(
            _with_moe_metadata(
                _stats(
                    f"{base}_w2_sorted_output",
                    sorted_output_actual,
                    sorted_output_expected,
                ),
                m=m,
                top_k=top_k,
                num_experts=num_experts,
                actual_impl=actual_impl,
                expert_pattern=expert_pattern,
                stage="w2_sorted_output",
                layer=awq_moe_layer,
            )
        )
    if final_output_actual is not None and final_output_expected is not None:
        results.append(
            _with_moe_metadata(
                _stats(
                    f"{base}_final_weighted_output",
                    final_output_actual,
                    final_output_expected,
                ),
                m=m,
                top_k=top_k,
                num_experts=num_experts,
                actual_impl=actual_impl,
                expert_pattern=expert_pattern,
                stage="final_weighted_output",
                layer=awq_moe_layer,
            )
        )
    return results


def _check_awq_moe_single_token_indexed_graph_replay(
    group_size: int,
    device: torch.device,
    awq_moe_model: Path,
    awq_moe_layer: str,
    num_experts: int,
    top_k: int,
    expert_patterns: list[str],
) -> list[dict[str, Any]]:
    _require_torch_op("awq_sm70_prepare")
    _require_torch_op("awq_moe_build_strided_ptrs")
    _require_torch_op("awq_moe_single_token_dense_w13_sm70_out")
    _require_torch_op("awq_moe_single_token_dense_stage_sm70_out")
    _require_torch_op("awq_moe_single_token_indexed_dense_w13_sm70_out")
    _require_torch_op("awq_moe_single_token_indexed_dense_stage_sm70_out")
    _require_torch_op("awq_moe_single_token_weighted_reduce_out")

    if not expert_patterns:
        raise ValueError("At least one expert pattern is required.")

    (
        w13_qweight,
        w13_scales,
        w13_qzeros,
        w2_qweight,
        w2_scales,
        w2_qzeros,
    ) = _load_awq_moe_layer(awq_moe_model, awq_moe_layer, num_experts, device)

    w13_tm_weight, _w13_tm_scales, w13_ptrs_w, w13_ptrs_s, _w13_k_ld, _w13_q_ld = (
        _prepare_awq_moe_weights(w13_qweight, w13_scales, w13_qzeros, group_size)
    )
    w2_tm_weight, _w2_tm_scales, w2_ptrs_w, w2_ptrs_s, _w2_k_ld, _w2_q_ld = (
        _prepare_awq_moe_weights(w2_qweight, w2_scales, w2_qzeros, group_size)
    )

    total_slots = top_k
    hidden_size = int(w13_qweight.shape[1])
    w13_n = int(w13_qweight.shape[2]) * 8
    intermediate_size = w13_n // 2
    hidden_out = int(w2_qweight.shape[2]) * 8

    x_static = _make_input(1, hidden_size, device)
    topk_ids_static = torch.empty((1, top_k), dtype=torch.int32, device=device)
    topk_weights_static = _make_topk_weights(top_k, device)
    graph_compact_input = torch.empty(
        (total_slots, hidden_size), dtype=torch.float16, device=device
    )
    graph_gate_up = torch.empty(
        (total_slots, w13_n), dtype=torch.float16, device=device
    )
    graph_intermediate = torch.empty(
        (total_slots, intermediate_size), dtype=torch.float16, device=device
    )
    graph_sorted_output = torch.empty(
        (total_slots, hidden_out), dtype=torch.float16, device=device
    )
    graph_final_output = torch.empty(
        (1, hidden_size), dtype=torch.float16, device=device
    )
    graph_offsets = torch.empty(top_k + 1, dtype=torch.int32, device=device)
    graph_offsets64 = torch.empty(top_k + 1, dtype=torch.int64, device=device)
    graph_inv_idx = torch.empty(top_k, dtype=torch.int32, device=device)
    graph_sorted_expert_ids = torch.empty(top_k, dtype=torch.int32, device=device)

    def _copy_pattern(pattern: str) -> torch.Tensor:
        logical_ids = _make_logical_expert_ids(
            total_slots,
            num_experts,
            device,
            pattern,
        )
        topk_ids = logical_ids.view(1, top_k).to(torch.int32).contiguous()
        topk_ids_static.copy_(topk_ids)
        return topk_ids

    def _run_indexed_graph_path() -> None:
        sm70_ops.awq_moe_single_token_indexed_dense_w13_sm70_out(
            graph_gate_up,
            graph_compact_input,
            x_static,
            topk_ids_static,
            w13_ptrs_w,
            w13_ptrs_s,
            graph_offsets,
            graph_offsets64,
            graph_inv_idx,
            graph_sorted_expert_ids,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
            hidden_size,
        )
        torch.ops._C.silu_and_mul(graph_intermediate, graph_gate_up)
        sm70_ops.awq_moe_single_token_indexed_dense_stage_sm70_out(
            graph_sorted_output,
            graph_intermediate,
            graph_offsets,
            graph_sorted_expert_ids,
            w2_ptrs_w,
            w2_ptrs_s,
            top_k,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
        )
        sm70_ops.awq_moe_single_token_weighted_reduce_out(
            graph_sorted_output[:, :hidden_size],
            topk_weights_static,
            graph_inv_idx,
            graph_final_output,
            top_k,
            hidden_size,
        )

    def _run_dense_reference(
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ref_compact_input = torch.empty_like(graph_compact_input)
        ref_gate_up = torch.empty_like(graph_gate_up)
        ref_intermediate = torch.empty_like(graph_intermediate)
        ref_sorted_output = torch.empty_like(graph_sorted_output)
        ref_final_output = torch.empty_like(graph_final_output)
        ref_offsets = torch.empty_like(graph_offsets)
        ref_offsets64 = torch.empty_like(graph_offsets64)
        ref_inv_idx = torch.empty_like(graph_inv_idx)
        ref_sorted_expert_ids = torch.empty_like(graph_sorted_expert_ids)
        sm70_ops.awq_moe_single_token_dense_w13_sm70_out(
            ref_gate_up,
            ref_compact_input,
            x_static,
            topk_ids,
            w13_ptrs_w,
            w13_ptrs_s,
            ref_offsets,
            ref_offsets64,
            ref_inv_idx,
            ref_sorted_expert_ids,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
            hidden_size,
        )
        torch.ops._C.silu_and_mul(ref_intermediate, ref_gate_up)
        sm70_ops.awq_moe_single_token_dense_stage_sm70_out(
            ref_sorted_output,
            ref_intermediate,
            ref_offsets,
            ref_sorted_expert_ids,
            w2_ptrs_w,
            w2_ptrs_s,
            top_k,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
        )
        sm70_ops.awq_moe_single_token_weighted_reduce_out(
            ref_sorted_output[:, :hidden_size],
            topk_weights_static,
            ref_inv_idx,
            ref_final_output,
            top_k,
            hidden_size,
        )
        return (
            ref_offsets,
            ref_inv_idx,
            ref_sorted_expert_ids,
            ref_gate_up,
            ref_intermediate,
            ref_sorted_output,
            ref_final_output,
        )

    capture_pattern = expert_patterns[0]
    _copy_pattern(capture_pattern)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _run_indexed_graph_path()
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.accelerator.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_indexed_graph_path()

    results: list[dict[str, Any]] = []
    for replay_index, pattern in enumerate(expert_patterns):
        topk_ids = _copy_pattern(pattern)
        graph.replay()
        torch.accelerator.synchronize(device)
        (
            ref_offsets,
            ref_inv_idx,
            ref_sorted_expert_ids,
            ref_gate_up,
            ref_intermediate,
            ref_sorted_output,
            ref_final_output,
        ) = _run_dense_reference(topk_ids)
        base = (
            "awq_moe_single_token_indexed_graph_replay"
            f"_topk{top_k}_experts{num_experts}_{awq_moe_layer}"
            f"_capture_{capture_pattern}_replay_{pattern}"
        )
        stage_tensors = [
            ("metadata_offsets", graph_offsets, ref_offsets),
            ("metadata_inv_permuted_idx", graph_inv_idx, ref_inv_idx),
            (
                "metadata_sorted_expert_ids",
                graph_sorted_expert_ids,
                ref_sorted_expert_ids,
            ),
            ("w13_gate_up", graph_gate_up, ref_gate_up),
            ("silu_and_mul", graph_intermediate, ref_intermediate),
            ("w2_sorted_output", graph_sorted_output, ref_sorted_output),
            ("final_weighted_output", graph_final_output, ref_final_output),
        ]
        for stage, actual, expected in stage_tensors:
            result = _with_moe_metadata(
                _stats(f"{base}_{stage}", actual, expected),
                m=1,
                top_k=top_k,
                num_experts=num_experts,
                actual_impl="single_token_indexed",
                expert_pattern=pattern,
                stage=stage,
                layer=awq_moe_layer,
            )
            result["capture_expert_pattern"] = capture_pattern
            result["replay_index"] = replay_index
            results.append(result)
    return results


def _check_awq_moe_single_token_indexed_compile(
    group_size: int,
    device: torch.device,
    awq_moe_model: Path,
    awq_moe_layer: str,
    num_experts: int,
    top_k: int,
    expert_patterns: list[str],
    copy_metadata_inside_graph: bool,
    router_inside_graph: bool,
    input_pattern: str,
) -> list[dict[str, Any]]:
    _require_torch_op("awq_sm70_prepare")
    _require_torch_op("awq_moe_build_strided_ptrs")
    _require_torch_op("awq_moe_single_token_dense_w13_sm70_out")
    _require_torch_op("awq_moe_single_token_dense_stage_sm70_out")
    _require_torch_op("awq_moe_single_token_indexed_dense_w13_sm70_out")
    _require_torch_op("awq_moe_single_token_indexed_dense_stage_sm70_out")
    _require_torch_op("awq_moe_single_token_weighted_reduce_out")

    if not expert_patterns:
        raise ValueError("At least one expert pattern is required.")

    (
        w13_qweight,
        w13_scales,
        w13_qzeros,
        w2_qweight,
        w2_scales,
        w2_qzeros,
    ) = _load_awq_moe_layer(awq_moe_model, awq_moe_layer, num_experts, device)

    w13_tm_weight, _w13_tm_scales, w13_ptrs_w, w13_ptrs_s, _w13_k_ld, _w13_q_ld = (
        _prepare_awq_moe_weights(w13_qweight, w13_scales, w13_qzeros, group_size)
    )
    w2_tm_weight, _w2_tm_scales, w2_ptrs_w, w2_ptrs_s, _w2_k_ld, _w2_q_ld = (
        _prepare_awq_moe_weights(w2_qweight, w2_scales, w2_qzeros, group_size)
    )

    total_slots = top_k
    hidden_size = int(w13_qweight.shape[1])
    w13_n = int(w13_qweight.shape[2]) * 8
    intermediate_size = w13_n // 2
    hidden_out = int(w2_qweight.shape[2]) * 8

    x_static = _make_moe_input(1, hidden_size, device, input_pattern)
    router_logits_source = torch.empty(
        (1, num_experts), dtype=torch.float32, device=device
    )
    topk_ids_source = torch.empty((1, top_k), dtype=torch.int32, device=device)
    topk_weights_source = torch.empty((1, top_k), dtype=torch.float32, device=device)
    token_expert_indices_source = torch.empty(
        (1, top_k), dtype=torch.int32, device=device
    )
    topk_ids_static = torch.empty((1, top_k), dtype=torch.int32, device=device)
    topk_weights_static = _make_topk_weights(top_k, device)
    compact_input = torch.empty(
        (total_slots, hidden_size), dtype=torch.float16, device=device
    )
    gate_up = torch.empty((total_slots, w13_n), dtype=torch.float16, device=device)
    intermediate = torch.empty(
        (total_slots, intermediate_size), dtype=torch.float16, device=device
    )
    sorted_output = torch.empty(
        (total_slots, hidden_out), dtype=torch.float16, device=device
    )
    final_output = torch.empty((1, hidden_size), dtype=torch.float16, device=device)
    offsets = torch.empty(top_k + 1, dtype=torch.int32, device=device)
    offsets64 = torch.empty(top_k + 1, dtype=torch.int64, device=device)
    inv_idx = torch.empty(top_k, dtype=torch.int32, device=device)
    sorted_expert_ids = torch.empty(top_k, dtype=torch.int32, device=device)

    def _copy_pattern(pattern: str) -> torch.Tensor:
        logical_ids = _make_logical_expert_ids(
            total_slots,
            num_experts,
            device,
            pattern,
        )
        topk_ids = logical_ids.view(1, top_k).to(torch.int32).contiguous()
        if router_inside_graph:
            router_logits_source.copy_(
                _make_router_logits_for_pattern(topk_ids, num_experts)
            )
        else:
            topk_ids_source.copy_(topk_ids)
            topk_weights_source.copy_(topk_weights_static)
            if not copy_metadata_inside_graph:
                topk_ids_static.copy_(topk_ids_source)
        return topk_ids

    def _indexed_chain() -> torch.Tensor:
        if router_inside_graph:
            ops.topk_softmax(
                topk_weights_source,
                topk_ids_source,
                token_expert_indices_source,
                router_logits_source,
                True,
            )
        if copy_metadata_inside_graph or router_inside_graph:
            topk_ids_static.copy_(topk_ids_source, non_blocking=True)
        sm70_ops.awq_moe_single_token_indexed_dense_w13_sm70_out(
            gate_up,
            compact_input,
            x_static,
            topk_ids_static,
            w13_ptrs_w,
            w13_ptrs_s,
            offsets,
            offsets64,
            inv_idx,
            sorted_expert_ids,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
            hidden_size,
        )
        torch.ops._C.silu_and_mul(intermediate, gate_up)
        sm70_ops.awq_moe_single_token_indexed_dense_stage_sm70_out(
            sorted_output,
            intermediate,
            offsets,
            sorted_expert_ids,
            w2_ptrs_w,
            w2_ptrs_s,
            top_k,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
        )
        sm70_ops.awq_moe_single_token_weighted_reduce_out(
            sorted_output[:, :hidden_size],
            (
                topk_weights_source
                if (copy_metadata_inside_graph or router_inside_graph)
                else topk_weights_static
            ),
            inv_idx,
            final_output,
            top_k,
            hidden_size,
        )
        return final_output

    def _dense_reference(topk_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if router_inside_graph:
            ref_topk_ids = torch.empty_like(topk_ids_source)
            ref_topk_weights = torch.empty_like(topk_weights_source)
            ref_token_expert_indices = torch.empty_like(token_expert_indices_source)
            ops.topk_softmax(
                ref_topk_weights,
                ref_topk_ids,
                ref_token_expert_indices,
                router_logits_source,
                True,
            )
            topk_ids = ref_topk_ids
            reduce_weights = ref_topk_weights
        else:
            reduce_weights = (
                topk_weights_source
                if copy_metadata_inside_graph
                else topk_weights_static
            )
        ref_compact_input = torch.empty_like(compact_input)
        ref_gate_up = torch.empty_like(gate_up)
        ref_intermediate = torch.empty_like(intermediate)
        ref_sorted_output = torch.empty_like(sorted_output)
        ref_final_output = torch.empty_like(final_output)
        ref_offsets = torch.empty_like(offsets)
        ref_offsets64 = torch.empty_like(offsets64)
        ref_inv_idx = torch.empty_like(inv_idx)
        ref_sorted_expert_ids = torch.empty_like(sorted_expert_ids)
        sm70_ops.awq_moe_single_token_dense_w13_sm70_out(
            ref_gate_up,
            ref_compact_input,
            x_static,
            topk_ids,
            w13_ptrs_w,
            w13_ptrs_s,
            ref_offsets,
            ref_offsets64,
            ref_inv_idx,
            ref_sorted_expert_ids,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
            hidden_size,
        )
        torch.ops._C.silu_and_mul(ref_intermediate, ref_gate_up)
        sm70_ops.awq_moe_single_token_dense_stage_sm70_out(
            ref_sorted_output,
            ref_intermediate,
            ref_offsets,
            ref_sorted_expert_ids,
            w2_ptrs_w,
            w2_ptrs_s,
            top_k,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
        )
        sm70_ops.awq_moe_single_token_weighted_reduce_out(
            ref_sorted_output[:, :hidden_size],
            reduce_weights,
            ref_inv_idx,
            ref_final_output,
            top_k,
            hidden_size,
        )
        return (
            ref_offsets,
            ref_inv_idx,
            ref_sorted_expert_ids,
            ref_gate_up,
            ref_intermediate,
            ref_sorted_output,
            ref_final_output,
        )

    def _backend(graph_module: torch.fx.GraphModule, example_inputs: list[Any]):
        from copy import deepcopy

        from torch._inductor.compile_fx import compile_fx

        from vllm.compilation.passes.inductor_pass import pass_context
        from vllm.compilation.passes.utility.fix_functionalization import (
            FixFunctionalizationPass,
        )
        from vllm.config import CompilationConfig, ModelConfig, VllmConfig
        from vllm.config.utils import Range

        vllm_config = VllmConfig(
            model_config=ModelConfig(dtype=torch.float16),
            compilation_config=CompilationConfig(custom_ops=["all"]),
        )
        inductor_config = deepcopy(
            vllm_config.compilation_config.inductor_compile_config
        )
        inductor_config["force_disable_caches"] = True
        fix_pass = FixFunctionalizationPass(vllm_config)

        def _post_pass(graph: torch.fx.Graph) -> None:
            fix_pass(graph)

        inductor_config["post_grad_custom_post_pass"] = _post_pass
        with pass_context(Range(1, 4096)):
            return compile_fx(
                graph_module,
                example_inputs,
                config_patches=inductor_config,
            )

    _copy_pattern(expert_patterns[0])
    compiled_chain = torch.compile(_indexed_chain, backend=_backend, fullgraph=True)
    compiled_chain()
    torch.accelerator.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled_chain()

    results: list[dict[str, Any]] = []
    for pattern in expert_patterns:
        topk_ids = _copy_pattern(pattern)
        graph.replay()
        torch.accelerator.synchronize(device)
        (
            ref_offsets,
            ref_inv_idx,
            ref_sorted_expert_ids,
            ref_gate_up,
            ref_intermediate,
            ref_sorted_output,
            ref_final_output,
        ) = _dense_reference(topk_ids)
        base = (
            "awq_moe_single_token_indexed_compile_graph"
            f"_copy_in_graph_{int(copy_metadata_inside_graph)}"
            f"_router_in_graph_{int(router_inside_graph)}"
            f"_topk{top_k}_experts{num_experts}_{awq_moe_layer}_{pattern}"
        )
        stage_tensors = [
            ("router_topk_ids", topk_ids_source, topk_ids),
            ("metadata_offsets", offsets, ref_offsets),
            ("metadata_inv_permuted_idx", inv_idx, ref_inv_idx),
            ("metadata_sorted_expert_ids", sorted_expert_ids, ref_sorted_expert_ids),
            ("w13_gate_up", gate_up, ref_gate_up),
            ("silu_and_mul", intermediate, ref_intermediate),
            ("w2_sorted_output", sorted_output, ref_sorted_output),
            ("final_weighted_output", final_output, ref_final_output),
        ]
        for stage, actual, expected in stage_tensors:
            result = _with_moe_metadata(
                _stats(f"{base}_{stage}", actual, expected),
                m=1,
                top_k=top_k,
                num_experts=num_experts,
                actual_impl="single_token_indexed",
                expert_pattern=pattern,
                stage=stage,
                layer=awq_moe_layer,
            )
            result["copy_metadata_inside_graph"] = copy_metadata_inside_graph
            result["router_inside_graph"] = router_inside_graph
            result["input_pattern"] = input_pattern
            results.append(result)
    return results


def _check_fp8_moe(
    m: int,
    group_size: int,
    device: torch.device,
    fp8_moe_model: Path,
    fp8_moe_layer: str,
    num_experts: int,
    top_k: int,
    actual_impl: MoEActual,
    expert_pattern: str,
) -> list[dict[str, Any]]:
    _require_torch_op("fp8_sm70_prepare")
    _require_torch_op("fp8_gemm_sm70_out_meta")
    _require_torch_op("awq_moe_build_strided_ptrs")
    _require_torch_op("fp8_moe_gemm_sm70_out")
    if actual_impl == "dense_graphsafe":
        _require_torch_op("fp8_moe_dense_stage_sm70_out")
    if actual_impl in (
        "single_token_dense",
        "single_token_indexed",
        "single_token_compact_w13",
        "legacy_single_token_compact",
    ):
        _require_torch_op("fp8_moe_single_token_dense_w13_sm70_out")
        _require_torch_op("fp8_moe_single_token_dense_stage_sm70_out")
        if actual_impl == "single_token_indexed":
            _require_torch_op("fp8_moe_single_token_indexed_dense_w13_sm70_out")
            _require_torch_op("fp8_moe_single_token_indexed_dense_stage_sm70_out")
        if actual_impl == "single_token_compact_w13":
            _require_torch_op("fp8_moe_single_token_compact_dense_w13_sm70_out")
        if actual_impl == "legacy_single_token_compact":
            _require_torch_op("fp8_moe_single_token_sm70_out")
        if m != 1:
            raise ValueError(f"{actual_impl} FP8 MoE check requires --m 1.")

    if group_size != 128:
        raise ValueError("FP8 SM70 MoE path currently supports group_size=128 only.")

    w13_weight, w13_scale, w2_weight, w2_scale = _load_fp8_moe_layer(
        fp8_moe_model,
        fp8_moe_layer,
        num_experts,
        device,
    )
    w13_tm_weight, w13_tm_scales, w13_metas, w13_ptrs_w, w13_ptrs_s = (
        _prepare_fp8_moe_weights(w13_weight, w13_scale, group_size)
    )
    w2_tm_weight, w2_tm_scales, w2_metas, w2_ptrs_w, w2_ptrs_s = (
        _prepare_fp8_moe_weights(w2_weight, w2_scale, group_size)
    )

    total_slots = m * top_k
    logical_expert_ids = _make_logical_expert_ids(
        total_slots,
        num_experts,
        device,
        expert_pattern,
    )
    topk_ids = logical_expert_ids.view(m, top_k)
    sorted_expert_ids, order = torch.sort(logical_expert_ids)
    expert_offsets, expert_offsets64 = _expert_offsets(sorted_expert_ids, num_experts)
    dense_expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)

    hidden_size = int(w13_weight.shape[2])
    if actual_impl in (
        "single_token_dense",
        "single_token_indexed",
        "single_token_compact_w13",
        "legacy_single_token_compact",
    ):
        single_token_input = _make_input(1, hidden_size, device)
        sorted_input = single_token_input.expand(total_slots, hidden_size).contiguous()
        sorted_input = sorted_input[order]
    else:
        single_token_input = None
        sorted_input = _make_input(total_slots, hidden_size, device)[order]
    w13_n = int(w13_weight.shape[1])
    intermediate_size = w13_n // 2
    hidden_out = int(w2_weight.shape[1])

    gate_up_actual = torch.empty(
        (total_slots, w13_n), dtype=torch.float16, device=device
    )
    legacy_intermediate_actual = None
    legacy_sorted_output_actual = None
    single_token_offsets = None
    single_token_offsets64 = None
    single_token_inv_permuted_idx = None
    single_token_sorted_expert_ids = None
    if actual_impl in ("batched", "batched_w2_per_expert_dispatch"):
        sm70_ops.fp8_moe_gemm_sm70_out(
            gate_up_actual,
            sorted_input,
            expert_offsets,
            w13_ptrs_w,
            w13_ptrs_s,
            num_experts,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
            False,
        )
    elif actual_impl in (
        "batched_per_expert_dispatch",
        "batched_w13_per_expert_dispatch",
    ):
        _require_torch_op("fp8_moe_gemm_sm70_per_expert_dispatch_out")
        sm70_ops.fp8_moe_gemm_sm70_per_expert_dispatch_out(
            gate_up_actual,
            sorted_input,
            expert_offsets,
            w13_ptrs_w,
            w13_ptrs_s,
            num_experts,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
            False,
        )
    elif actual_impl == "dense_graphsafe":
        sm70_ops.fp8_moe_dense_stage_sm70_out(
            gate_up_actual,
            sorted_input,
            expert_offsets,
            dense_expert_ids,
            w13_ptrs_w,
            w13_ptrs_s,
            num_experts,
            int(w13_tm_weight.shape[1]),
            w13_n,
            group_size,
        )
    elif actual_impl in (
        "single_token_dense",
        "single_token_indexed",
        "single_token_compact_w13",
        "legacy_single_token_compact",
    ):
        single_token_offsets = torch.empty(top_k + 1, dtype=torch.int32, device=device)
        single_token_offsets64 = torch.empty(
            top_k + 1, dtype=torch.int64, device=device
        )
        single_token_inv_permuted_idx = torch.empty(
            top_k, dtype=torch.int32, device=device
        )
        single_token_sorted_expert_ids = torch.empty(
            top_k, dtype=torch.int32, device=device
        )
        if actual_impl == "single_token_indexed":
            sm70_ops.fp8_moe_single_token_indexed_dense_w13_sm70_out(
                gate_up_actual,
                torch.empty_like(sorted_input),
                single_token_input,
                topk_ids.to(torch.int32).contiguous(),
                w13_ptrs_w,
                w13_ptrs_s,
                single_token_offsets,
                single_token_offsets64,
                single_token_inv_permuted_idx,
                single_token_sorted_expert_ids,
                int(w13_tm_weight.shape[1]),
                w13_n,
                group_size,
                hidden_size,
            )
        elif actual_impl == "single_token_compact_w13":
            compact_ptrs_w = torch.empty(top_k * 16, dtype=torch.uint8, device=device)
            compact_ptrs_s = torch.empty(top_k * 16, dtype=torch.uint8, device=device)
            sm70_ops.fp8_moe_single_token_compact_dense_w13_sm70_out(
                gate_up_actual,
                torch.empty_like(sorted_input),
                single_token_input,
                topk_ids.to(torch.int32).contiguous(),
                w13_ptrs_w,
                w13_ptrs_s,
                compact_ptrs_w,
                compact_ptrs_s,
                single_token_offsets,
                single_token_offsets64,
                single_token_inv_permuted_idx,
                single_token_sorted_expert_ids,
                int(w13_tm_weight.shape[1]),
                w13_n,
                group_size,
                hidden_size,
            )
        elif actual_impl == "legacy_single_token_compact":
            assert single_token_input is not None
            legacy_intermediate_actual = torch.empty(
                (total_slots, intermediate_size),
                dtype=torch.float16,
                device=device,
            )
            legacy_sorted_output_actual = torch.empty(
                (total_slots, hidden_out),
                dtype=torch.float16,
                device=device,
            )
            sorted_weights = torch.empty(top_k, dtype=torch.float32, device=device)
            topk_weights = torch.full(
                (top_k,), 1.0 / max(top_k, 1), dtype=torch.float32, device=device
            )
            row_bytes = int(w13_ptrs_w.numel() // num_experts)
            dst_w13_ptrs_w = torch.empty(
                top_k, row_bytes, dtype=torch.uint8, device=device
            )
            dst_w13_ptrs_s = torch.empty_like(dst_w13_ptrs_w)
            dst_w2_ptrs_w = torch.empty_like(dst_w13_ptrs_w)
            dst_w2_ptrs_s = torch.empty_like(dst_w13_ptrs_w)
            broadcast_input_indices = torch.empty(
                top_k, dtype=torch.int32, device=device
            )
            output_actual = torch.empty(
                (1, hidden_out), dtype=torch.float16, device=device
            )
            sm70_ops.fp8_moe_single_token_sm70_out(
                output_actual,
                single_token_input,
                topk_weights,
                topk_ids.to(torch.int32).contiguous(),
                w13_ptrs_w.view(num_experts, row_bytes),
                w13_ptrs_s.view(num_experts, row_bytes),
                w2_ptrs_w.view(num_experts, row_bytes),
                w2_ptrs_s.view(num_experts, row_bytes),
                torch.empty_like(sorted_input),
                gate_up_actual,
                legacy_intermediate_actual,
                legacy_sorted_output_actual,
                sorted_weights,
                dst_w13_ptrs_w,
                dst_w13_ptrs_s,
                dst_w2_ptrs_w,
                dst_w2_ptrs_s,
                expert_offsets,
                single_token_inv_permuted_idx,
                single_token_sorted_expert_ids,
                broadcast_input_indices,
                torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
                int(w13_tm_weight.shape[1]),
                w13_n,
                int(w2_tm_weight.shape[1]),
                hidden_out,
                group_size,
                hidden_size,
                False,
                False,
                False,
                False,
                False,
                True,
            )
        else:
            sm70_ops.fp8_moe_single_token_dense_w13_sm70_out(
                gate_up_actual,
                torch.empty_like(sorted_input),
                single_token_input,
                topk_ids.to(torch.int32).contiguous(),
                w13_ptrs_w,
                w13_ptrs_s,
                single_token_offsets,
                single_token_offsets64,
                single_token_inv_permuted_idx,
                single_token_sorted_expert_ids,
                int(w13_tm_weight.shape[1]),
                w13_n,
                group_size,
                hidden_size,
            )
    else:
        gate_up_actual = _dense_fp8_moe_stage_out(
            sorted_input,
            expert_offsets64,
            w13_tm_weight,
            w13_tm_scales,
            w13_metas,
            w13_n,
        )
    gate_up_expected = _dense_fp8_moe_stage_out(
        sorted_input,
        expert_offsets64,
        w13_tm_weight,
        w13_tm_scales,
        w13_metas,
        w13_n,
    )

    intermediate_actual = torch.empty(
        (total_slots, intermediate_size), dtype=torch.float16, device=device
    )
    intermediate_expected = torch.empty_like(intermediate_actual)
    if legacy_intermediate_actual is None:
        torch.ops._C.silu_and_mul(intermediate_actual, gate_up_actual)
    else:
        intermediate_actual.copy_(legacy_intermediate_actual)
    torch.ops._C.silu_and_mul(intermediate_expected, gate_up_expected)

    sorted_output_actual = torch.empty(
        (total_slots, hidden_out), dtype=torch.float16, device=device
    )
    if legacy_sorted_output_actual is not None:
        sorted_output_actual.copy_(legacy_sorted_output_actual)
    elif actual_impl in ("batched", "batched_w13_per_expert_dispatch"):
        sm70_ops.fp8_moe_gemm_sm70_out(
            sorted_output_actual,
            intermediate_actual,
            expert_offsets,
            w2_ptrs_w,
            w2_ptrs_s,
            num_experts,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
            False,
        )
    elif actual_impl in (
        "batched_per_expert_dispatch",
        "batched_w2_per_expert_dispatch",
    ):
        sm70_ops.fp8_moe_gemm_sm70_per_expert_dispatch_out(
            sorted_output_actual,
            intermediate_actual,
            expert_offsets,
            w2_ptrs_w,
            w2_ptrs_s,
            num_experts,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
            False,
        )
    elif actual_impl == "dense_graphsafe":
        sm70_ops.fp8_moe_dense_stage_sm70_out(
            sorted_output_actual,
            intermediate_actual,
            expert_offsets,
            dense_expert_ids,
            w2_ptrs_w,
            w2_ptrs_s,
            num_experts,
            int(w2_tm_weight.shape[1]),
            hidden_out,
            group_size,
        )
    elif actual_impl in (
        "single_token_dense",
        "single_token_indexed",
        "single_token_compact_w13",
        "legacy_single_token_compact",
    ):
        assert single_token_offsets is not None
        assert single_token_sorted_expert_ids is not None
        if actual_impl == "single_token_indexed":
            sm70_ops.fp8_moe_single_token_indexed_dense_stage_sm70_out(
                sorted_output_actual,
                intermediate_actual,
                single_token_offsets,
                single_token_sorted_expert_ids,
                w2_ptrs_w,
                w2_ptrs_s,
                top_k,
                int(w2_tm_weight.shape[1]),
                hidden_out,
                group_size,
            )
        else:
            sm70_ops.fp8_moe_single_token_dense_stage_sm70_out(
                sorted_output_actual,
                intermediate_actual,
                single_token_offsets,
                single_token_sorted_expert_ids,
                w2_ptrs_w,
                w2_ptrs_s,
                top_k,
                int(w2_tm_weight.shape[1]),
                hidden_out,
                group_size,
            )
    else:
        sorted_output_actual = _dense_fp8_moe_stage_out(
            intermediate_actual,
            expert_offsets64,
            w2_tm_weight,
            w2_tm_scales,
            w2_metas,
            hidden_out,
        )
    sorted_output_expected = _dense_fp8_moe_stage_out(
        intermediate_expected,
        expert_offsets64,
        w2_tm_weight,
        w2_tm_scales,
        w2_metas,
        hidden_out,
    )

    final_output_actual = None
    final_output_expected = None
    if actual_impl == "legacy_single_token_compact":
        assert output_actual is not None
        assert single_token_inv_permuted_idx is not None
        final_output_actual = output_actual
        final_output_expected = torch.empty_like(final_output_actual)
        sm70_ops.awq_moe_single_token_weighted_reduce_out(
            sorted_output_expected,
            topk_weights,
            single_token_inv_permuted_idx,
            final_output_expected,
            top_k,
            hidden_out,
        )

    base = (
        f"fp8_moe_{actual_impl}_tokens{m}_topk{top_k}_experts{num_experts}"
        f"_{fp8_moe_layer}"
    )
    results = [
        _with_moe_metadata(
            _stats(f"{base}_w13_gate_up", gate_up_actual, gate_up_expected),
            m=m,
            top_k=top_k,
            num_experts=num_experts,
            actual_impl=actual_impl,
            expert_pattern=expert_pattern,
            stage="w13_gate_up",
        ),
        _with_moe_metadata(
            _stats(
                f"{base}_silu_and_mul",
                intermediate_actual,
                intermediate_expected,
            ),
            m=m,
            top_k=top_k,
            num_experts=num_experts,
            actual_impl=actual_impl,
            expert_pattern=expert_pattern,
            stage="silu_and_mul",
        ),
        _with_moe_metadata(
            _stats(
                f"{base}_w2_sorted_output",
                sorted_output_actual,
                sorted_output_expected,
            ),
            m=m,
            top_k=top_k,
            num_experts=num_experts,
            actual_impl=actual_impl,
            expert_pattern=expert_pattern,
            stage="w2_sorted_output",
        ),
    ]
    if final_output_actual is not None and final_output_expected is not None:
        results.append(
            _with_moe_metadata(
                _stats(
                    f"{base}_final_weighted_output",
                    final_output_actual,
                    final_output_expected,
                ),
                m=m,
                top_k=top_k,
                num_experts=num_experts,
                actual_impl=actual_impl,
                expert_pattern=expert_pattern,
                stage="final_weighted_output",
            )
        )
    return results


def _check_fp8(
    m: int,
    k: int,
    n: int,
    group_size: int,
    device: torch.device,
    fp8_model: Path | None,
    fp8_layer: str | None,
) -> dict[str, Any]:
    _require_torch_op("fp8_sm70_prepare")
    _require_torch_op("fp8_gemm_sm70_out_meta")

    if group_size != 128:
        raise ValueError("FP8 SM70 path currently supports group_size=128 only.")

    if fp8_model is not None:
        if fp8_layer is None:
            raise ValueError("--fp8-layer is required when --fp8-model is set.")
        qweight, scales = _load_fp8_layer(fp8_model, fp8_layer, device)
        n = qweight.shape[0]
        k = qweight.shape[1]
    else:
        if k % group_size != 0 or n % group_size != 0:
            raise ValueError("FP8 K and N must be divisible by group_size.")
        weight_f16 = torch.randn((n, k), device=device, dtype=torch.float16) * 0.25
        qweight = weight_f16.to(torch.float8_e4m3fn)
        scales = (
            torch.rand(
                (n // group_size, k // group_size),
                device=device,
                dtype=torch.float32,
            )
            * 0.25
            + 0.01
        )

    if k % group_size != 0 or n % group_size != 0:
        raise ValueError("FP8 K and N must be divisible by group_size.")

    x = torch.randn((m, k), device=device, dtype=torch.float16)

    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(qweight, scales, group_size)
    actual = torch.empty((m, n), device=device, dtype=torch.float16)
    sm70_ops.fp8_gemm_sm70_out_meta(actual, x, tm_weight, tm_scales, meta)

    scale_expanded = scales.repeat_interleave(group_size, 0).repeat_interleave(
        group_size, 1
    )
    expected_weight = qweight.to(torch.float16) * scale_expanded.to(torch.float16)
    expected = torch.matmul(x, expected_weight.t())
    name = f"fp8_m{m}_k{k}_n{n}_g{group_size}"
    if fp8_model is not None and fp8_layer is not None:
        name = f"{name}_{fp8_layer}"
    return _stats(name, actual, expected)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "awq",
            "fp8",
            "awq_moe",
            "fp8_moe",
            "awq_moe_graph_replay",
            "awq_moe_compile",
        ],
        default="all",
    )
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--awq-model", type=Path)
    parser.add_argument(
        "--awq-layer",
        default="model.language_model.layers.1.linear_attn.out_proj",
    )
    parser.add_argument("--awq-moe-model", type=Path)
    parser.add_argument(
        "--awq-moe-layer",
        default="model.language_model.layers.1.mlp.experts",
    )
    parser.add_argument(
        "--awq-moe-layers",
        nargs="+",
        help="Optional list of AWQ MoE expert layer prefixes to sweep.",
    )
    parser.add_argument("--awq-moe-num-experts", type=int, default=256)
    parser.add_argument("--awq-moe-top-k", type=int, default=8)
    parser.add_argument(
        "--awq-moe-actual",
        choices=[
            "batched",
            "batched_per_expert_dispatch",
            "batched_w13_per_expert_dispatch",
            "batched_w2_per_expert_dispatch",
            "dense_out",
            "dense_graphsafe",
            "active_dense_stage",
            "single_token_dense",
            "single_token_indexed",
            "single_token_compact_w13",
            "legacy_single_token_compact",
        ],
        default="batched",
        help="Implementation to compare against the per-expert dense return op.",
    )
    parser.add_argument(
        "--moe-expert-pattern",
        default="round_robin",
        help=(
            "Diagnostic routing pattern for AWQ/FP8 MoE strict checks: "
            "round_robin, single, or single:<expert_id>."
        ),
    )
    parser.add_argument(
        "--moe-expert-patterns",
        nargs="+",
        help="Optional list of MoE expert routing patterns to sweep.",
    )
    parser.add_argument(
        "--awq-moe-compile-copy-metadata-inside-graph",
        action="store_true",
        help=(
            "For awq_moe_compile, copy topk metadata inside the compiled CUDA "
            "graph before the indexed MoE ops. This matches the full-model "
            "router->MoE metadata dependency more closely."
        ),
    )
    parser.add_argument(
        "--awq-moe-compile-router-inside-graph",
        action="store_true",
        help=(
            "For awq_moe_compile, run vLLM topk_softmax inside the compiled "
            "CUDA graph to produce topk metadata before the indexed MoE ops."
        ),
    )
    parser.add_argument(
        "--awq-moe-input-pattern",
        choices=["range", "random", "random_scaled"],
        default="range",
        help="Input tensor pattern for AWQ MoE op-level diagnostics.",
    )
    parser.add_argument("--fp8-model", type=Path)
    parser.add_argument(
        "--fp8-layer",
        default="model.language_model.layers.1.linear_attn.out_proj",
    )
    parser.add_argument("--fp8-moe-model", type=Path)
    parser.add_argument(
        "--fp8-moe-layer",
        default="model.language_model.layers.1.mlp.experts",
    )
    parser.add_argument(
        "--fp8-moe-layers",
        nargs="+",
        help="Optional list of FP8 MoE expert layer prefixes to sweep.",
    )
    parser.add_argument("--fp8-moe-num-experts", type=int, default=256)
    parser.add_argument("--fp8-moe-top-k", type=int, default=8)
    parser.add_argument(
        "--fp8-moe-actual",
        choices=[
            "batched",
            "batched_per_expert_dispatch",
            "batched_w13_per_expert_dispatch",
            "batched_w2_per_expert_dispatch",
            "dense_out",
            "dense_graphsafe",
            "single_token_dense",
            "single_token_indexed",
            "single_token_compact_w13",
            "legacy_single_token_compact",
        ],
        default="batched",
        help="Implementation to compare against the per-expert dense FP8 op.",
    )
    parser.add_argument("--allow-nonzero-diff", action="store_true")
    parser.add_argument(
        "--moe-max-diff-bound",
        type=float,
        default=FP16_MOE_OUTPUT_BOUND,
        help="Max absolute diff allowed for op-level MoE fp16 acceptance.",
    )
    parser.add_argument(
        "--moe-max-nonzero-ulp-diff",
        type=int,
        default=FP16_MOE_MAX_NONZERO_ULP_DIFF,
        help="Max nonzero fp16 ULP diff allowed for op-level MoE acceptance.",
    )
    parser.add_argument(
        "--moe-sign-mismatch-abs-bound",
        type=float,
        default=FP16_MOE_SIGN_MISMATCH_ABS_BOUND,
        help=(
            "Largest abs diff allowed for fp16 sign mismatches before the "
            "MoE op-level acceptance gate fails."
        ),
    )
    parser.add_argument(
        "--moe-model-quality-gate-passed",
        action="store_true",
        help=(
            "Mark the separate model-level greedy/logprob/perplexity quality "
            "gate as passed, allowing bounded op diffs to classify as B-accept."
        ),
    )
    parser.add_argument("--require-sm70", action="store_true", default=True)
    parser.add_argument(
        "--no-require-sm70",
        action="store_false",
        dest="require_sm70",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mode: Mode = args.mode
    awq_moe_actual: MoEActual = args.awq_moe_actual
    fp8_moe_actual: MoEActual = args.fp8_moe_actual
    device = torch.device(args.device)
    _require_cuda(device)
    _require_sm70(device, args.require_sm70)

    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    results: list[dict[str, Any]] = []
    moe_expert_patterns = args.moe_expert_patterns or [args.moe_expert_pattern]
    awq_moe_layers = args.awq_moe_layers or [args.awq_moe_layer]
    fp8_moe_layers = args.fp8_moe_layers or [args.fp8_moe_layer]
    if mode in ("awq_moe_graph_replay", "awq_moe_compile"):
        args.m = [1]
    for m in args.m:
        if mode in ("all", "awq"):
            results.append(
                _check_awq(
                    m,
                    args.k,
                    args.n,
                    args.group_size,
                    device,
                    args.awq_model,
                    args.awq_layer,
                )
            )
        if mode in ("all", "fp8"):
            results.append(
                _check_fp8(
                    m,
                    args.k,
                    args.n,
                    args.group_size,
                    device,
                    args.fp8_model,
                    args.fp8_layer,
                )
            )
        if mode == "awq_moe":
            model = args.awq_moe_model or args.awq_model
            if model is None:
                raise ValueError("--awq-moe-model or --awq-model is required.")
            for awq_moe_layer in awq_moe_layers:
                for moe_expert_pattern in moe_expert_patterns:
                    results.extend(
                        _check_awq_moe(
                            m,
                            args.group_size,
                            device,
                            model,
                            awq_moe_layer,
                            args.awq_moe_num_experts,
                            args.awq_moe_top_k,
                            awq_moe_actual,
                            moe_expert_pattern,
                        )
                    )
        if mode == "awq_moe_graph_replay":
            model = args.awq_moe_model or args.awq_model
            if model is None:
                raise ValueError("--awq-moe-model or --awq-model is required.")
            for awq_moe_layer in awq_moe_layers:
                results.extend(
                    _check_awq_moe_single_token_indexed_graph_replay(
                        args.group_size,
                        device,
                        model,
                        awq_moe_layer,
                        args.awq_moe_num_experts,
                        args.awq_moe_top_k,
                        moe_expert_patterns,
                    )
                )
        if mode == "awq_moe_compile":
            model = args.awq_moe_model or args.awq_model
            if model is None:
                raise ValueError("--awq-moe-model or --awq-model is required.")
            for awq_moe_layer in awq_moe_layers:
                results.extend(
                    _check_awq_moe_single_token_indexed_compile(
                        args.group_size,
                        device,
                        model,
                        awq_moe_layer,
                        args.awq_moe_num_experts,
                        args.awq_moe_top_k,
                        moe_expert_patterns,
                        args.awq_moe_compile_copy_metadata_inside_graph,
                        args.awq_moe_compile_router_inside_graph,
                        args.awq_moe_input_pattern,
                    )
                )
        if mode == "fp8_moe":
            model = args.fp8_moe_model or args.fp8_model
            if model is None:
                raise ValueError("--fp8-moe-model or --fp8-model is required.")
            for fp8_moe_layer in fp8_moe_layers:
                for moe_expert_pattern in moe_expert_patterns:
                    results.extend(
                        _check_fp8_moe(
                            m,
                            args.group_size,
                            device,
                            model,
                            fp8_moe_layer,
                            args.fp8_moe_num_experts,
                            args.fp8_moe_top_k,
                            fp8_moe_actual,
                            moe_expert_pattern,
                        )
                    )

    report = {
        "strict": not args.allow_nonzero_diff,
        "awq_model": str(args.awq_model) if args.awq_model else None,
        "awq_layer": args.awq_layer,
        "awq_moe_model": str(args.awq_moe_model) if args.awq_moe_model else None,
        "awq_moe_layer": args.awq_moe_layer,
        "awq_moe_layers": awq_moe_layers,
        "awq_moe_num_experts": args.awq_moe_num_experts,
        "awq_moe_top_k": args.awq_moe_top_k,
        "awq_moe_actual": awq_moe_actual,
        "awq_moe_dispatch_policy_env": os.getenv("VLLM_SM70_AWQ_MOE_DISPATCH_POLICY"),
        "fp8_model": str(args.fp8_model) if args.fp8_model else None,
        "fp8_layer": args.fp8_layer,
        "fp8_moe_model": str(args.fp8_moe_model) if args.fp8_moe_model else None,
        "fp8_moe_layer": args.fp8_moe_layer,
        "fp8_moe_layers": fp8_moe_layers,
        "fp8_moe_num_experts": args.fp8_moe_num_experts,
        "fp8_moe_top_k": args.fp8_moe_top_k,
        "fp8_moe_actual": fp8_moe_actual,
        "moe_expert_pattern": args.moe_expert_pattern,
        "moe_expert_patterns": moe_expert_patterns,
        "device": str(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "moe_classification": _classify_moe_results(
            mode=mode,
            results=results,
            actual_impl=awq_moe_actual if mode == "awq_moe" else fp8_moe_actual,
            max_diff_bound=args.moe_max_diff_bound,
            max_nonzero_ulp_diff=args.moe_max_nonzero_ulp_diff,
            sign_mismatch_abs_bound=args.moe_sign_mismatch_abs_bound,
            model_quality_gate_passed=args.moe_model_quality_gate_passed,
        ),
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")

    failed = [r for r in results if not r["equal"] or r["max_diff"] != 0.0]
    if failed and not args.allow_nonzero_diff:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
