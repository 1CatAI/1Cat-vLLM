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


def supports_qsa_host_kv_offload(cfg) -> bool:
    """Admit the local native connector with private per-rank host history."""
    transfer = cfg.kv_transfer_config
    if transfer is None or transfer.kv_connector != "OffloadingConnector":
        return False
    extra = transfer.kv_connector_extra_config
    parallel = cfg.parallel_config
    spec_name = extra.get("spec_name", "CPUOffloadingSpec")
    if spec_name == "TieringOffloadingSpec":
        tiers = extra.get("secondary_tiers", [])
        if not (
            parallel.nnodes == 1
            and isinstance(tiers, list)
            and all(
                isinstance(tier, dict) and tier.get("type") == "fs" for tier in tiers
            )
        ):
            return False
    elif spec_name != "CPUOffloadingSpec":
        return False
    return (
        transfer.kv_role == "kv_both"
        and transfer.kv_connector_module_path is None
        and "block_size" not in extra
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.decode_context_parallel_size == 2
        and parallel.prefill_context_parallel_size == 1
        # The internal MTP config runs inside the target's MRV2 worker even
        # when the draft architecture does not itself select MRV2.
        and (cfg.is_speculative_draft or cfg.use_v2_model_runner)
        and cfg.cache_config.enable_prefix_caching
        and not cfg.scheduler_config.disable_hybrid_kv_cache_manager
        and not cfg.kernel_config.qsa_host_kv_device_reference
        and not cfg.kernel_config.qsa_host_indexer_history
        and cfg.kernel_config.qsa_host_kv_state_blocks > 0
    )


def resolve_qsa_host_kv(cfg) -> bool:
    """Admit per-vector host storage independently of draft cache precision."""
    from vllm.platforms import current_platform

    policy = cfg.kernel_config
    model = cfg.model_config
    reason = None
    if not policy.qsa_host_kv:
        reason = "disabled by KernelConfig"
    elif model is None or not getattr(model.hf_text_config, "indexer_n_heads", None):
        reason = "no QSA selector metadata"
    elif not current_platform.is_cuda() or not current_platform.is_device_capability(
        70
    ):
        reason = "requires SM70 CUDA"
    elif model.dtype != torch.float16:
        reason = "host staging requires FP16 activations"
    elif getattr(model.hf_text_config, "head_dim", None) != 256:
        reason = "host staging requires D256"
    elif getattr(model.hf_text_config, "indexer_compress_ratio", None) != 4 or (
        getattr(model.hf_text_config, "indexer_budget", 0) % 4
    ):
        reason = "host staging requires complete four-token selector pages"
    elif model.get_num_kv_heads(cfg.parallel_config) != 1:
        reason = "host writer requires one TP-local KV head"
    elif cfg.parallel_config.decode_context_parallel_size not in (1, 2) or (
        cfg.parallel_config.prefill_context_parallel_size != 1
    ):
        reason = "host QSA storage requires DCP1/DCP2 and PCP1"
    elif (
        cfg.cache_config.block_size // cfg.parallel_config.decode_context_parallel_size
    ) % 4:
        reason = "host QSA DCP local block size must be a multiple of 4"
    elif cfg.cache_config.cache_dtype not in (
        "auto",
        "float16",
        "int8",
        "int8_per_token_head",
    ):
        reason = "host KV uses separate per-vector scales; keep draft KV FP16 or INT8"
    elif (
        (transfer := cfg.kv_transfer_config) is not None
        and transfer.kv_connector
        and not supports_qsa_host_kv_offload(cfg)
    ):
        reason = "active host storage requires the qualified native CPU connector"
    policy.qsa_host_kv_reason = reason
    policy.qsa_host_kv_active = reason is None
    return policy.qsa_host_kv_active
