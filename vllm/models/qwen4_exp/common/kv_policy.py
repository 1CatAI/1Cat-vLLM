# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrated QSA cache admission without loading model weights."""

import json
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open


def calibrated_qsa_checkpoint_reason(model_config) -> str | None:
    text = model_config.hf_text_config
    layer_types = getattr(text, "layer_types", None)
    if not layer_types:
        return "checkpoint does not describe QSA cache owners"
    owners = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    if not owners:
        return "checkpoint has no full-attention cache owners"
    model = getattr(model_config, "model", None)
    if not isinstance(model, str):
        return "calibrated checkpoint index is not locally available"
    root = Path(model)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return "calibrated checkpoint index is not locally available"
    try:
        weight_map = json.loads(index.read_text())["weight_map"]
        required = {}
        for owner in owners:
            for kind in ("k", "v"):
                suffix = f".layers.{owner}.self_attn.{kind}_scale"
                matches = [name for name in weight_map if name.endswith(suffix)]
                if len(matches) != 1:
                    return "checkpoint index lacks complete unambiguous QSA K/V scales"
                required[matches[0]] = weight_map[matches[0]]
        for filename in set(required.values()):
            path = root / filename
            if not path.resolve().is_relative_to(root.resolve()):
                return "QSA scale shard lies outside the checkpoint"
            with safe_open(path, framework="pt", device="cpu") as shard:
                for name, owner_file in required.items():
                    if owner_file != filename:
                        continue
                    tensor = shard.get_tensor(name)
                    if tensor.dtype != torch.float32 or tensor.numel() != 1:
                        return "QSA calibration requires scalar FP32 scales"
                    if not torch.isfinite(tensor).all() or tensor.item() <= 0:
                        return "QSA calibration scales must be finite and positive"
    except (OSError, ValueError, KeyError, TypeError, SafetensorError):
        return "QSA calibration metadata or scale shard is invalid"
    return None


def resolve_qsa_auto_e4m3(cfg) -> bool:
    policy = cfg.kernel_config
    policy.qsa_auto_e4m3_active = False
    model = cfg.model_config
    reason = None
    if not policy.qsa_auto_e4m3:
        reason = "disabled by KernelConfig"
    elif model is None or not getattr(model.hf_text_config, "indexer_n_heads", None):
        reason = "no QSA selector metadata"
    elif cfg.cache_config.cache_dtype != "auto":
        reason = "explicit KV dtype takes precedence"
    elif cfg.speculative_config is not None:
        reason = "automatic calibrated storage is qualified without speculation"
    elif cfg.cache_config.calculate_kv_scales:
        reason = "runtime scale calculation takes precedence"
    elif cfg.parallel_config.prefill_context_parallel_size != 1:
        reason = "QSA does not support prefill context parallelism"
    else:
        from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_e4m3_capability_reason

        reason = qsa_e4m3_capability_reason(model.dtype)
        if reason is None:
            reason = calibrated_qsa_checkpoint_reason(model)
    policy.qsa_auto_e4m3_reason = reason
    policy.qsa_auto_e4m3_active = reason is None
    if reason is None:
        cfg.cache_config.cache_dtype = "fp8_e4m3"
        cfg.cache_config.cache_dtype_from_checkpoint = False
    return policy.qsa_auto_e4m3_active
