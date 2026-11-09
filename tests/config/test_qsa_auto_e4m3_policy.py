# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatic QSA storage requires real scalar calibration payloads."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config import KernelConfig
from vllm.models.qwen4_exp.common.kv_policy import (
    calibrated_qsa_checkpoint_reason,
    resolve_qsa_auto_e4m3,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa


def _checkpoint(tmp_path, scales=None):
    if scales is None:
        scales = {
            f"model.layers.{owner}.self_attn.{kind}_scale": torch.tensor(0.125)
            for owner in (0, 2)
            for kind in ("k", "v")
        }
    save_file(scales, tmp_path / "scales.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "scales.safetensors" for name in scales}})
    )
    return SimpleNamespace(
        model=str(tmp_path),
        dtype=torch.float16,
        hf_text_config=SimpleNamespace(
            indexer_n_heads=3,
            layer_types=["full_attention", "linear_attention", "full_attention"],
        ),
    )


def _config(model):
    return SimpleNamespace(
        kernel_config=KernelConfig(),
        model_config=model,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            cache_dtype_from_checkpoint=False,
            calculate_kv_scales=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, tensor_parallel_size=1
        ),
    )


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_auto_admission_uses_calibration_and_capability(tmp_path, monkeypatch, tp):
    cfg = _config(_checkpoint(tmp_path))
    cfg.parallel_config.tensor_parallel_size = tp
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda dtype: None)
    before = dict(os.environ)
    assert resolve_qsa_auto_e4m3(cfg)
    assert cfg.cache_config.cache_dtype == "fp8_e4m3"
    assert cfg.kernel_config.qsa_auto_e4m3_active
    assert cfg.kernel_config.qsa_auto_e4m3_reason is None
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "mode",
    ["disabled", "explicit", "speculation", "runtime_scales", "pcp", "capability"],
)
def test_unqualified_route_retains_cache_dtype(tmp_path, monkeypatch, mode):
    cfg = _config(_checkpoint(tmp_path))
    monkeypatch.setattr(
        qsa,
        "qsa_e4m3_capability_reason",
        lambda dtype: "unsupported dtype" if mode == "capability" else None,
    )
    if mode == "disabled":
        cfg.kernel_config.qsa_auto_e4m3 = False
    if mode == "explicit":
        cfg.cache_config.cache_dtype = "float16"
    if mode == "speculation":
        cfg.speculative_config = object()
    if mode == "runtime_scales":
        cfg.cache_config.calculate_kv_scales = True
    if mode == "pcp":
        cfg.parallel_config.prefill_context_parallel_size = 2
    previous = cfg.cache_config.cache_dtype
    assert not resolve_qsa_auto_e4m3(cfg)
    assert cfg.cache_config.cache_dtype == previous
    assert cfg.kernel_config.qsa_auto_e4m3_reason


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_scale_payload_is_rejected(tmp_path, value):
    model = _checkpoint(
        tmp_path,
        {
            f"model.layers.{owner}.self_attn.{kind}_scale": torch.tensor(value)
            for owner in (0, 2)
            for kind in ("k", "v")
        },
    )
    assert "finite and positive" in calibrated_qsa_checkpoint_reason(model)


@pytest.mark.parametrize("dtype,shape", [(torch.float16, ()), (torch.float32, (2,))])
def test_scale_storage_contract(tmp_path, dtype, shape):
    model = _checkpoint(
        tmp_path,
        {
            f"model.layers.{owner}.self_attn.{kind}_scale": torch.full(
                shape, 0.125, dtype=dtype
            )
            for owner in (0, 2)
            for kind in ("k", "v")
        },
    )
    assert "scalar FP32" in calibrated_qsa_checkpoint_reason(model)


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "corrupt_shard", "outside", "missing_shard"]
)
def test_incomplete_or_invalid_checkpoint_is_not_auto_selected(
    tmp_path, monkeypatch, fault
):
    cfg = _config(_checkpoint(tmp_path))
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda dtype: None)
    index = tmp_path / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    name = "model.layers.0.self_attn.k_scale"
    if fault == "missing":
        del data["weight_map"][name]
    if fault == "duplicate":
        data["weight_map"]["other.layers.0.self_attn.k_scale"] = "scales.safetensors"
    if fault == "corrupt_shard":
        (tmp_path / "scales.safetensors").write_bytes(b"invalid")
    if fault == "outside":
        data["weight_map"][name] = "../scales.safetensors"
    if fault == "missing_shard":
        data["weight_map"][name] = "missing.safetensors"
    index.write_text(json.dumps(data))
    assert not resolve_qsa_auto_e4m3(cfg)
    assert cfg.cache_config.cache_dtype == "auto"
    assert cfg.kernel_config.qsa_auto_e4m3_reason


def test_unrelated_and_unavailable_checkpoints_keep_auto(tmp_path, monkeypatch):
    cfg = _config(_checkpoint(tmp_path))
    cfg.model_config.model = "remote/unavailable"
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda dtype: None)
    assert not resolve_qsa_auto_e4m3(cfg)
    cfg.model_config.hf_text_config.indexer_n_heads = 0
    assert not resolve_qsa_auto_e4m3(cfg)
    cfg.model_config = None
    assert not resolve_qsa_auto_e4m3(cfg)


def test_diagnostics_do_not_change_inactive_kernel_hash():
    policy = KernelConfig()
    original = policy.compute_hash()
    policy.qsa_auto_e4m3 = False
    policy.qsa_auto_e4m3_reason = "uncalibrated"
    assert policy.compute_hash() == original
    policy.qsa_auto_e4m3_active = True
    assert policy.compute_hash() != original
