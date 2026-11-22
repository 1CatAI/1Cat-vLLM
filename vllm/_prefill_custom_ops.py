# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-dispatched prefill custom-op hooks for generic model_executor layers."""

from __future__ import annotations

from typing import Any

import torch

from vllm._sm70_flashnext_prefill_ops import (
    has_sm70_flashnext_prefill_ops,
    wait_hc_ag_peers_if_needed,
)
from vllm.compilation.sm70_decode_graph import is_sm70_decode_graph_compiling
from vllm.diagnostics import layer_dump_requested
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.runtime_resources import current_runtime_resources


def wait_peer_prefill_if_needed(x: torch.Tensor | None = None) -> None:
    wait_hc_ag_peers_if_needed(x)


def _extract_moe_layer_idx(layer_name: str) -> int:
    parts = layer_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return -1
    return -1


def _gdn_projection_dump_requested(layer_name: str) -> bool:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _gdn_diagnostics,
    )

    return bool(_gdn_diagnostics.projection_requested(layer_name))


def _moe_runner_dump_requested(layer_idx: int, label: str) -> bool:
    return layer_dump_requested(layer_idx, label, moe=True)


def maybe_tp_sharded_router_topk(
    runner: Any,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    from vllm.models.qwen4_exp.nvidia.sm70_hcx import (
        hc_prefill_sequence_parallel_active,
    )

    if (
        runner.gate is None
        or not getattr(layer, "_hc_prefill_peer_consumer", False)
        or runner._fse_fuse_gate
        or runner.do_naive_dispatch_combine
        or runner.moe_config.pcp_size > 1
        or runner._quant_method.is_monolithic
        or isinstance(runner.router, ZeroExpertRouter)
        or hidden_states.ndim != 2
        or hidden_states.shape[0] < 320
        or not hidden_states.is_cuda
        or not hidden_states.is_contiguous()
        or is_sm70_decode_graph_compiling()
        or torch.compiler.is_compiling()
        or torch.cuda.is_current_stream_capturing()
        or not hc_prefill_sequence_parallel_active()
        or getattr(runner.router, "capture_fn", None) is not None
        or getattr(runner.router, "_capture_fn", None) is not None
        or getattr(runner.router, "eplb_state", None) is not None
    ):
        return None
    layer_idx = _extract_moe_layer_idx(runner.layer_name)
    if _moe_runner_dump_requested(layer_idx, "moe_router_logits"):
        return None
    resources = current_runtime_resources()
    owner = resources.get("hc_prefill_peer") if resources is not None else None
    if (
        owner is None
        or not getattr(owner, "p2p_enabled", False)
        or not getattr(owner, "pointers", None)
        or hidden_states.device != owner.device
    ):
        return None
    m = int(hidden_states.shape[0])
    step = (
        int(owner._ag_pending_local_rows)
        if getattr(owner, "_ag_pull_pending", False)
        and int(hidden_states.data_ptr()) == getattr(owner, "_ag_pending_ptr", 0)
        else (m + 3) // 4
    )
    if step * 128 > int(owner.rs_bank_bytes):
        return None
    r = int(owner.rank)
    first = min(m, r * step)
    last = min(m, first + step)
    n_loc = last - first
    if n_loc <= 0:
        return None
    loc_x = hidden_states[first:last]
    loc_logits, _ = runner.gate(loc_x)
    loc_in_ids = None if input_ids is None else input_ids[first:last]
    loc_w, loc_ids = runner.router.select_experts(
        hidden_states=loc_x,
        router_logits=loc_logits,
        input_ids=loc_in_ids,
    )
    loc_w_c = loc_w.contiguous()
    loc_ids_c = loc_ids.contiguous()
    if (
        hasattr(owner, "allgather_topk_rs")
        and loc_w_c.dtype == torch.float32
        and loc_ids_c.dtype in (torch.int32, torch.int64)
    ):
        return owner.allgather_topk_rs(loc_w_c, loc_ids_c, m, step)
    k = int(loc_ids_c.shape[1])
    cols = ((2 * k + 3) // 4) * 4
    local_packed = torch.zeros(
        (step, cols), dtype=torch.int32, device=hidden_states.device
    )
    local_packed[:n_loc, :k].copy_(loc_ids_c)
    local_packed[:n_loc, k : 2 * k].copy_(loc_w_c.view(torch.int32))
    gathered = torch.empty(
        (4 * step, cols), dtype=torch.int32, device=hidden_states.device
    )
    owner.allgather_rs_into(gathered, local_packed)
    topk_ids = gathered[:m, :k].contiguous()
    if loc_ids_c.dtype != torch.int32:
        topk_ids = topk_ids.to(dtype=loc_ids_c.dtype)
    topk_weights = gathered[:m, k : 2 * k].contiguous().view(torch.float32)
    return topk_weights, topk_ids


def maybe_gdn_rmsnorm_gated(
    attn: Any,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    num_tokens: int,
    layer_name: str,
) -> torch.Tensor | None:
    if not (
        num_tokens >= 256
        and not torch.compiler.is_compiling()
        and has_sm70_flashnext_prefill_ops()
        and core_attn_out.ndim == 3
        and core_attn_out.shape == (num_tokens, attn.num_v_heads // attn.tp_size, 128)
        and core_attn_out.shape[1] == 12
        and core_attn_out.dtype == torch.float16
        and core_attn_out.is_contiguous()
        and z.shape == core_attn_out.shape
        and z.dtype == torch.float16
        and z.stride(1) == 128
        and z.stride(2) == 1
        and (z.stride(0) & 3) == 0
        and attn.norm.weight.shape == (128,)
        and attn.norm.weight.dtype in (torch.float16, torch.float32)
        and attn.norm.weight.is_contiguous()
        and attn.norm.group_size is None
        and attn.norm.norm_before_gate
        and attn.norm.activation in ("silu", "swish")
        and not _gdn_projection_dump_requested(layer_name)
    ):
        return None
    profile_start = attn._gdn_profiler.start()
    norm_2d = torch.empty(
        (num_tokens, 1536),
        dtype=torch.float16,
        device=core_attn_out.device,
    )
    torch.ops.vllm_sm70_fn_prefill.gdn_rmsnorm_gated_h12d128(
        norm_2d,
        core_attn_out,
        z,
        attn.norm.weight,
        float(attn.norm.eps),
    )
    attn._gdn_profiler.end(
        layer_name,
        "projection_norm",
        profile_start,
        tokens=num_tokens,
    )
    return norm_2d


def maybe_gdn_input_prefill(
    attn: Any,
    hidden_states: torch.Tensor,
    num_tokens: int,
    side: Any,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not (
        side is None
        and getattr(attn, "sm70_qwen38_fp16_fused_input", False)
        and isinstance(attn.in_proj_qkvz, LinearBase)
        and isinstance(attn.in_proj_ba, LinearBase)
        and getattr(attn.in_proj_qkvz, "bias", None) is None
        and getattr(attn.in_proj_ba, "bias", None) is None
        and not attn.gqa_interleaved_layout
        and not attn.disable_tp_for_ba_proj
        and attn.head_k_dim == attn.head_v_dim == 128
        and not _gdn_projection_dump_requested(layer_name)
    ):
        return None
    qkvz_w = attn.in_proj_qkvz.weight
    ba_w = attn.in_proj_ba.weight
    if not (
        has_sm70_flashnext_prefill_ops()
        and qkvz_w.is_cuda
        and qkvz_w.dtype == ba_w.dtype == torch.float16
        and qkvz_w.shape == (4096, 2560)
        and ba_w.shape == (24, 2560)
    ):
        return None
    mixed_qkv, z_flat, b, a = torch.ops.vllm.qwen38_fn_gdn_input_prefill(
        hidden_states, qkvz_w, ba_w
    )
    z = z_flat.view(num_tokens, attn.num_v_heads // attn.tp_size, attn.head_v_dim)
    if num_tokens < 256 and attn.gdn_policy.projection.mixed_qkv_contiguous:
        mixed_qkv = mixed_qkv.contiguous()
    return mixed_qkv, z, b, a


def ple_prefill_slice_range(
    hidden_states: torch.Tensor, num_tokens: int
) -> tuple[int, int]:
    copy_start = 0
    copy_rows = num_tokens
    local_rows = (num_tokens + 3) // 4
    if (
        num_tokens >= 320
        and hidden_states.ndim == 2
        and hidden_states.shape[0] == local_rows
        and local_rows < num_tokens
    ):
        try:
            from vllm.distributed import get_tp_group
            from vllm.models.qwen4_exp.nvidia.sm70_hcx import (
                hc_prefill_sequence_parallel_active,
            )

            tp_group = get_tp_group()
            if hc_prefill_sequence_parallel_active() and tp_group.world_size == 4:
                rk = tp_group.rank_in_group
                copy_start = min(num_tokens, rk * local_rows)
                copy_end = min(num_tokens, (rk + 1) * local_rows)
                copy_rows = copy_end - copy_start
        except Exception:
            copy_start = 0
            copy_rows = num_tokens
    return copy_start, copy_rows
