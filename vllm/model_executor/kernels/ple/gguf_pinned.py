# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission for bounded, TP-sharded pinned GGUF PLE decode tables."""

from vllm.config import get_current_vllm_config_or_none


def pinned_decode_active(config=None) -> bool:
    if config is None:
        config = get_current_vllm_config_or_none()
    return bool(
        getattr(
            getattr(config, "kernel_config", None), "ple_pinned_decode_active", False
        )
    )


def pinned_table_capability(
    tables,
    *,
    enabled,
    sm70,
    fp16,
    tp_size,
    local_ranks,
    max_seqs,
    local_workers,
    available_bytes,
    reserve_bytes,
    explicit_host_bytes=None,
):
    """Resolve metadata and fair host capacity before any table allocation."""
    reason = None
    required = sum(t["bytes"] for t in tables) // max(tp_size, 1)
    if not enabled:
        reason = "disabled_by_kernel_config"
    elif not tables:
        reason = "no_packed_ple_table"
    elif not sm70 or not fp16:
        reason = "requires_sm70_fp16_output"
    elif not local_workers or tp_size != 4:
        reason = "requires_local_tp4_workers"
    elif max_seqs > 4:
        reason = "decode_capacity_has_no_calibration"
    elif len(tables) != 1 or any(t["type"] != 20 for t in tables):
        reason = "pinned_row_decoder_has_no_calibration"
    elif any(t["k"] != 160 or t["n"] % tp_size for t in tables):
        reason = "pinned_row_shape_has_no_calibration"
    elif available_bytes is None or local_ranks < 1:
        reason = "host_capacity_unavailable"
    elif required > max(0, available_bytes - reserve_bytes) // local_ranks:
        reason = "insufficient_fair_host_capacity"
    elif explicit_host_bytes is not None and required > explicit_host_bytes:
        reason = "explicit_host_budget_too_small"
    return {
        "enabled": reason is None,
        "reason": reason,
        "operator": "ple_pinned_iq4nl_lookup",
        "family": "lut4",
        "source_type": "IQ4_NL",
        "min_m": 1,
        "max_m": 20,
        "graph_safe": True,
        "packed_bytes_per_rank": required,
        "scope": "packed_ple_row_capability",
    }


def prepare_pinned_gguf_ple(config, tensors, names):
    """Resolve the same route on every TP worker before model construction."""
    import torch
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_tp_group
    from vllm.model_executor.layers.ple_offload_layer import is_offload_process
    from vllm.models.qwen4_exp.common.ple import (
        available_host_bytes,
        ple_host_budget_bytes,
        ple_host_reserve_bytes,
        total_host_bytes,
    )
    from vllm.platforms import current_platform

    policy = config.kernel_config
    policy.ple_pinned_decode_active = False
    if is_offload_process():
        return
    tables = [
        {
            "type": int(tensors[raw].tensor_type),
            "k": int(tensors[raw].shape[0]),
            "n": int(tensors[raw].shape[1]),
            "bytes": tensors[raw].n_bytes,
        }
        for raw, name in names.items()
        if name.endswith(".ple_embedding.ngram_embedding.weight")
    ]
    parallel = config.parallel_config
    total = total_host_bytes()
    status = pinned_table_capability(
        tables,
        enabled=policy.ple_pinned_decode,
        sm70=current_platform.is_device_capability(70),
        fp16=config.model_config.dtype == torch.float16,
        tp_size=parallel.tensor_parallel_size,
        local_ranks=parallel.tensor_parallel_size * parallel.data_parallel_size_local,
        max_seqs=config.scheduler_config.max_num_seqs,
        local_workers=(
            parallel.nnodes == 1
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_backend == "mp"
            and not parallel.use_ubatching
        ),
        available_bytes=available_host_bytes(),
        reserve_bytes=ple_host_reserve_bytes(total) if total else 0,
        explicit_host_bytes=ple_host_budget_bytes(),
    )
    if tables and parallel.tensor_parallel_size > 1:
        reasons = [None] * parallel.tensor_parallel_size
        dist.all_gather_object(
            reasons, status["reason"], group=get_tp_group().cpu_group
        )
        status["reason"] = next((r for r in reasons if r is not None), None)
        status["enabled"] = status["reason"] is None
    policy.ple_pinned_decode_active = status["enabled"]
    policy.ple_pinned_decoders["gguf_rows"] = status
