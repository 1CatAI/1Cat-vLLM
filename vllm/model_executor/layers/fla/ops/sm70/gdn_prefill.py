# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashQLA prefill/decode bindings with explicit layout and gate semantics."""

import functools

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops.gdn_profiling import GdnPrefillProfiler
from vllm.model_executor.layers.fla.ops.gdn_stages import normalize_qk

logger = init_logger(__name__)


def log_runtime_route_once(message: str, *args) -> None:
    if not torch.compiler.is_compiling():
        logger.info_once(message, *args)


def flashqla_sm70_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    core_attn_out: torch.Tensor | None = None,
    gate_is_exp: bool = False,
    *,
    use_original_tilelang: bool,
    profiler: GdnPrefillProfiler,
):
    if use_original_tilelang:
        from flash_qla.ops.gated_delta_rule.chunk import (
            chunk_gated_delta_rule_fwd_sm70_tilelang,
        )

        log_runtime_route_once(
            "Using original FlashQLA-SM70 TileLang GDN prefill path "
            "(indexed_state=%s, direct_output=%s).",
            state_indices is not None,
            core_attn_out is not None,
        )
    else:
        from flash_qla.ops.gated_delta_rule.chunk.sm70 import (
            chunk_gated_delta_rule_fwd_sm70_vlk_varlen,
        )

    if use_qk_l2norm_in_kernel:
        profile_start = profiler.start()
        q, k = normalize_qk(q, k)
        profiler.end(
            "flashqla",
            "qk_l2norm",
            profile_start,
            tokens=q.shape[1],
        )
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32)
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    output = None
    if core_attn_out is not None:
        candidate = core_attn_out[: q.shape[1]].unsqueeze(0)
        if (
            candidate.shape == v.shape
            and candidate.dtype == v.dtype
            and candidate.is_contiguous()
        ):
            output = candidate

    q_contiguous = q.is_contiguous()
    k_contiguous = k.is_contiguous()
    v_contiguous = v.is_contiguous()
    g_contiguous = g.is_contiguous()
    beta_contiguous = beta.is_contiguous()
    state_contiguous = initial_state.is_contiguous()
    profile_start = profiler.start()
    q_arg = q if q_contiguous else q.contiguous()
    k_arg = k if k_contiguous else k.contiguous()
    v_arg = v if v_contiguous else v.contiguous()
    g_arg = g if g_contiguous else g.contiguous()
    beta_arg = beta if beta_contiguous else beta.contiguous()
    state_arg = initial_state if state_contiguous else initial_state.contiguous()
    cu_arg = cu_seqlens if cu_seqlens.is_contiguous() else cu_seqlens.contiguous()
    profiler.end(
        "flashqla",
        "input_contiguous",
        profile_start,
        tokens=q.shape[1],
        details=(
            f"q={q_contiguous} k={k_contiguous} v={v_contiguous} "
            f"g={g_contiguous} beta={beta_contiguous} "
            f"state={state_contiguous} direct_output={output is not None} "
            f"gate_is_exp={gate_is_exp}"
        ),
    )

    profile_start = profiler.start()
    if use_original_tilelang:
        if gate_is_exp:
            logger.warning_once(
                "SM70 original TileLang FlashQLA prefill received exp(g); "
                "converting back to log(g). Configure fused_post_conv_prep to "
                "emit raw g for best performance."
            )
            g_arg = torch.log(g_arg)
        _, _, out, _, final_state = chunk_gated_delta_rule_fwd_sm70_tilelang(
            q=q_arg,
            k=k_arg,
            v=v_arg,
            g=g_arg,
            beta=beta_arg,
            cu_seqlens=cu_arg,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            initial_state=state_arg,
            scale=q.shape[-1] ** -0.5,
            output_final_state=output_final_state,
            output_h=False,
            auto_cp=False,
            state_layout_vlk=True,
            output=output,
            inplace_final_state=inplace_final_state,
        )
    else:
        out, final_state = chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q=q_arg,
            k=k_arg,
            v=v_arg,
            g=g_arg,
            beta=beta_arg,
            cu_seqlens=cu_arg,
            initial_state=state_arg,
            scale=q.shape[-1] ** -0.5,
            output_final_state=output_final_state,
            validate_cu_seqlens=False,
            output=output,
            gate_is_exp=gate_is_exp,
        )
    profiler.end(
        "flashqla",
        "kernel",
        profile_start,
        tokens=q.shape[1],
        details=(
            f"direct_output={output is not None} gate_is_exp={gate_is_exp} "
            f"original_tilelang={use_original_tilelang}"
        ),
    )
    if core_attn_out is not None and output is None:
        profile_start = profiler.start()
        out_flat = out.squeeze(0).reshape(-1)
        out_view = core_attn_out.reshape(-1)[: out_flat.numel()]
        out_view.copy_(out_flat)
        out = core_attn_out[: out.shape[1]].unsqueeze(0)
        profiler.end(
            "flashqla",
            "output_copy",
            profile_start,
            tokens=q.shape[1],
        )
    return out, final_state


def flashqla_sm70_chunk_gated_delta_rule_vllm_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import _load_ext

    if use_qk_l2norm_in_kernel:
        q, k = normalize_qk(q, k)
    ext = _load_ext()
    return ext.gdn_forward_vlk_varlen(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state.contiguous(),
        cu_seqlens.contiguous(),
        float(q.shape[-1] ** -0.5),
        True,
        False,
        False,
    )


@functools.cache
def _flashqla_sm70_decode_available() -> bool:
    try:
        from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import (  # noqa: F401
            gdn_decode_mixed_qkv_global_state_sm70,
        )
    except ImportError:
        return False
    return True
