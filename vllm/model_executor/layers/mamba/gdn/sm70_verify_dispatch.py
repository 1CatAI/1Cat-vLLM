# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact verifier admission and execution, with explicit common-layer callbacks.

The common module is supplied by the caller rather than imported here. This
keeps upstream metadata resolution and diagnostics bound to their one owner,
including callback overrides, without a cyclic operator import.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from vllm.diagnostics import diagnostic_channel
from vllm.model_executor.layers import sm70_fuse47 as _fuse47
from vllm.model_executor.layers.mamba.gdn import (
    sm70_gdn_verify_fused as _gdn_verify_fused,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import LayerNameType
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def try_projection(
    common: Any, self: Any, hidden_states: torch.Tensor, layer_name: LayerNameType
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Try upstream fused verifier projection unless dumps need eager tensors."""
    if common._sm70_gdn_projection_dump_requested(layer_name):
        return None
    from vllm.model_executor.layers.quantization.sm70_gdn_ba_verify import (
        apply_gdn_ba_verify,
    )

    return apply_gdn_ba_verify(self, hidden_states)


def mixed_decode_requested(
    common: Any,
    self: Any,
    mixed_qkv: torch.Tensor | None,
    attn_metadata: GDNAttentionMetadata,
) -> bool:
    """Shared admission for the upstream mixed-QKV decode/verifier route."""
    return bool(
        self.enable_sm70_fused_sigmoid_mixed_qkv
        and mixed_qkv is not None
        and mixed_qkv.is_cuda
        and (mixed_qkv.dtype == torch.float16)
        and (
            mixed_qkv.is_contiguous()
            or (
                attn_metadata.spec_sequence_masks is not None
                and mixed_qkv.shape[0] in (5, 20)
                and (mixed_qkv.stride(1) == 1)
                and (mixed_qkv.stride(0) >= mixed_qkv.shape[1])
            )
        )
        and (self.num_k_heads % self.tp_size == 0)
        and (self.num_v_heads % self.tp_size == 0)
    )


def mixed_verify_requested(
    common: Any,
    self: Any,
    mixed_qkv: torch.Tensor | None,
    ssm_state: torch.Tensor | None,
    attn_metadata: GDNAttentionMetadata,
    *,
    decode_requested: bool | None = None,
) -> bool:
    """Exact shared guard for the SM70 mixed-QKV speculative verifier."""
    if mixed_qkv is None or ssm_state is None:
        return False
    if decode_requested is None:
        decode_requested = mixed_decode_requested(
            common, self, mixed_qkv, attn_metadata
        )
    return bool(
        decode_requested
        and attn_metadata.spec_sequence_masks is not None
        and (attn_metadata.ddtree_parent_ids is None)
        and (attn_metadata.num_spec_decodes > 0)
        and current_platform.is_device_capability(70)
        and (self.num_k_heads // self.tp_size == 4)
        and (self.num_v_heads // self.tp_size == 12)
        and (self.head_k_dim == self.head_v_dim == 128)
        and (ssm_state.dtype == torch.float32)
        and (
            mixed_qkv.is_contiguous()
            or (
                mixed_qkv.shape[0] in (5, 20)
                and mixed_qkv.stride(1) == 1
                and (mixed_qkv.stride(0) >= mixed_qkv.shape[1])
            )
        )
        and (1 < mixed_qkv.shape[0] <= 16 or mixed_qkv.shape[0] == 20)
    )


def block_reason(
    common: Any,
    self: Any,
    layer_name: LayerNameType,
    mixed_qkv: torch.Tensor | None = None,
) -> str | None:
    """Why the exact fused verify kernels cannot serve this call (None: they can).

    The fused path covers exactly the production MTP verify route: the Qwen3.5
    layout inside the full-forward guard, reaching the standard spec core
    with a pure speculative batch, FP16 activations, no diagnostics.
    """
    if (
        self.gqa_interleaved_layout
        or self.disable_tp_for_ba_proj
        or getattr(self, "sm70_qwen38_fp16_fused_input", False)
    ):
        return "layout"
    projection = self.gdn_policy.projection
    if (
        projection.context_core
        or projection.mixed_qkv_contiguous
        or projection.z_contiguous
        or projection.input_projection
        or projection.output_projection
        or common._sm70_qwen_gdn_input_core_boundary_enabled(projection)
    ):
        return "env"
    graph_dump = diagnostic_channel("gdn_graph").policy
    if (
        diagnostic_channel("gdn_core").policy.directory
        or (graph_dump.capture and graph_dump.directory)
        or common._sm70_gdn_projection_dump_requested(layer_name)
    ):
        return "dump"
    if (
        getattr(self, "auto_sm70_qwen_gdn_003_spec_core", False)
        or getattr(self, "auto_sm70_qwen_gdn_spec_core", False)
        or (not getattr(self, "auto_sm70_qwen_gdn_full_forward", False))
    ):
        return "route"
    if (
        self.enable_sm70_dflash2_fused_gdn_verify
        or self.enable_sm70_dflash2_fused_gdn_norm
        or self.enable_sm70_dflash2_fused_gdn_split
        or self.enable_sm70_dflash2_fused_gdn_combined_split
        or self.enable_sm70_dflash2_fused_qkv_pack
        or self.enable_sm70_gdn_rmsnorm_onepass
    ):
        return "dflash"
    q_dim = self.key_dim // self.tp_size
    if (
        self.head_k_dim != _gdn_verify_fused.HEAD_DIM
        or self.head_v_dim != _gdn_verify_fused.HEAD_DIM
        or q_dim % 256 != 0
        or (self.conv1d.weight.dtype != torch.float16)
        or (self.activation not in ("silu", "swish"))
        or (self.norm.group_size is not None)
        or (not self.norm.norm_before_gate)
        or (self.norm.activation != "sigmoid")
        or (self.norm.weight.dtype != torch.float16)
    ):
        return "shape"
    try:
        forward_context = common.get_forward_context()
    except AssertionError:
        return "no_context"
    attn_metadata_raw = forward_context.attn_metadata
    if not isinstance(attn_metadata_raw, dict):
        return "no_metadata"
    attn_metadata = attn_metadata_raw.get(common._resolve_layer_name(layer_name))
    if not isinstance(attn_metadata, GDNAttentionMetadata):
        return "no_metadata"
    if not (
        attn_metadata.spec_sequence_masks is not None
        and attn_metadata.num_spec_decodes > 0
        and (attn_metadata.num_prefills == 0)
        and (attn_metadata.num_decodes == 0)
    ):
        return "not_pure_spec"
    if common._ddtree_parent_ids_require_branch(
        attn_metadata.ddtree_parent_ids,
        attn_metadata.ddtree_num_tree_tokens_cpu,
        attn_metadata.num_spec_decodes,
    ):
        return "ddtree"
    if mixed_qkv is not None:
        forward_context = common.get_forward_context()
        no_compile_layers = getattr(forward_context, "no_compile_layers", None)
        layer = (
            no_compile_layers.get(common._resolve_layer_name(layer_name))
            if no_compile_layers is not None and hasattr(no_compile_layers, "get")
            else None
        )
        kv_cache = getattr(layer, "kv_cache", None)
        ssm_state = kv_cache[1] if kv_cache is not None else None
        if mixed_verify_requested(common, self, mixed_qkv, ssm_state, attn_metadata):
            return "upstream_mixed_qkv"
    return None


def plan(
    common: Any,
    self: Any,
    layer_name: LayerNameType,
    mixed_qkv: torch.Tensor | None = None,
) -> frozenset[str]:
    """Units of sm70_gdn_verify_fused (ONECAT_GDN_FUSE) used by this call."""
    units = _gdn_verify_fused.enabled_units()
    if not units:
        return units
    reason = block_reason(common, self, layer_name, mixed_qkv)
    _gdn_verify_fused.ROUTE_COUNTS[
        "fused" if reason is None else f"fallback:{reason}"
    ] += 1
    return units if reason is None else frozenset()


def run_core(
    common: Any,
    self: Any,
    *,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    units: frozenset[str],
) -> torch.Tensor:
    """qwen_gdn_attention_core_standard_spec for a pure speculative batch with
    the exact fused kernels; same metadata resolution, same results."""
    (
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        spec_query_start_loc,
        spec_state_indices_tensor,
        spec_token_indx,
        non_spec_token_indx,
        spec_sequence_masks,
        num_accepted_tokens,
        spec_state_slot_selectors,
    ) = common._qwen_gdn_metadata_tensors(layer_name, core_attn_out.device)
    attn_metadata_by_layer = cast(
        dict[str, GDNAttentionMetadata], common.get_forward_context().attn_metadata
    )
    attn_metadata = attn_metadata_by_layer[common._resolve_layer_name(layer_name)]
    if conv_state_cache.numel() == 0 and ssm_state_cache.numel() == 0:
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None and kv_cache[0].numel() > 0:
            conv_state_cache, ssm_state_cache = (kv_cache[0], kv_cache[1])
    restore_fields: dict[str, object] = {}

    def _patch_metadata(name: str, tensor: torch.Tensor) -> None:
        if tensor.numel() > 0:
            restore_fields[name] = getattr(attn_metadata, name)
            setattr(attn_metadata, name, tensor)

    _patch_metadata("non_spec_query_start_loc", non_spec_query_start_loc)
    _patch_metadata("non_spec_state_indices_tensor", non_spec_state_indices_tensor)
    _patch_metadata("spec_query_start_loc", spec_query_start_loc)
    _patch_metadata("spec_state_indices_tensor", spec_state_indices_tensor)
    _patch_metadata("spec_token_indx", spec_token_indx)
    _patch_metadata("non_spec_token_indx", non_spec_token_indx)
    _patch_metadata("spec_sequence_masks", spec_sequence_masks)
    _patch_metadata("num_accepted_tokens", num_accepted_tokens)
    if spec_state_slot_selectors.numel() == 0:
        spec_state_slot_selectors = num_accepted_tokens
    _patch_metadata("spec_state_slot_selectors", spec_state_slot_selectors)
    try:
        self._forward_core_verify_fused(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            attn_metadata=attn_metadata,
            kv_cache=(conv_state_cache, ssm_state_cache),
            units=units,
        )
    finally:
        for name, value in restore_fields.items():
            setattr(attn_metadata, name, value)
    return core_attn_out


def forward_core_verify_fused(
    common: Any,
    self: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    attn_metadata: GDNAttentionMetadata,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    units: frozenset[str],
) -> None:
    """_forward_core for a pure speculative verify batch (no prefill, no
    non-spec decode, no DDTree/DFlash2) with sm70_gdn_verify_fused:
    u2 packs the conv output directly, u3 folds the gating into the
    recurrent kernel and writes core_attn_out. Bitwise equal to
    _forward_core on every real token row and on both state caches."""
    conv_state = (
        kv_cache[0]
        if common.is_conv_state_dim_first()
        else kv_cache[0].transpose(-1, -2)
    )
    ssm_state = kv_cache[1]
    num_actual_tokens = attn_metadata.num_actual_tokens
    num_spec_decodes = attn_metadata.num_spec_decodes
    spec_query_start_loc = attn_metadata.spec_query_start_loc
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
    assert spec_query_start_loc is not None
    assert spec_state_indices_tensor is not None
    spec_state_slot_selectors = (
        attn_metadata.spec_state_slot_selectors
        if attn_metadata.spec_state_slot_selectors is not None
        else attn_metadata.num_accepted_tokens
    )
    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]
    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )
    conv_state_indices = spec_state_indices_tensor[:, 0][:num_spec_decodes]
    q_dim = self.key_dim // self.tp_size
    v_dim = self.value_dim // self.tp_size
    if "u2" in units and mixed_qkv.dtype == conv_state.dtype:
        packed = torch.empty(
            num_actual_tokens * (2 * q_dim + v_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        _gdn_verify_fused.causal_conv1d_update_packed(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=spec_state_slot_selectors,
            query_start_loc=spec_query_start_loc,
            max_query_len=spec_state_indices_tensor.size(-1),
            out=packed,
            q_dim=q_dim,
            k_dim=q_dim,
            v_dim=v_dim,
        )
        seq_len = num_actual_tokens
        query = packed[: seq_len * q_dim].view(1, seq_len, -1, self.head_k_dim)
        key = packed[seq_len * q_dim : 2 * seq_len * q_dim].view(
            1, seq_len, -1, self.head_k_dim
        )
        value = packed[2 * seq_len * q_dim :].view(1, seq_len, -1, self.head_v_dim)
    else:
        mixed_qkv = common.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=spec_state_slot_selectors,
            query_start_loc=spec_query_start_loc,
            max_query_len=spec_state_indices_tensor.size(-1),
            validate_data=False,
        )
        query, key, value = self.rearrange_mixed_qkv(mixed_qkv)
    cu_seqlens = spec_query_start_loc[: num_spec_decodes + 1]
    if "u3" in units:
        _gdn_verify_fused.fused_recurrent_gdn_verify(
            q=query,
            k=key,
            v=value,
            b=b,
            a=a,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=ssm_state,
            out=core_attn_out[:num_actual_tokens],
            cu_seqlens=cu_seqlens,
            ssm_state_indices=spec_state_indices_tensor,
            num_accepted_tokens=spec_state_slot_selectors,
            zero_fill_rows=core_attn_out.shape[0]
            if _fuse47.unit_enabled("u3z")
            else None,
        )
    else:
        g_spec, beta_spec = common.fused_gdn_gating(
            self.A_log, a, b, self.dt_bias, beta_dtype=torch.float32
        )
        core_attn_out_spec, _ = common.fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g_spec,
            beta=beta_spec,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=spec_state_indices_tensor,
            num_accepted_tokens=spec_state_slot_selectors,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)


def norm_and_projection(self, core_attn_out, z):
    """Apply the exact fused gated norm, then the existing output projection."""
    core_attn_out = _gdn_verify_fused.gdn_rmsnorm_sigmoid_gate_exact(
        core_attn_out, z.reshape(z.shape[0], -1), self.norm.weight, self.norm.eps
    )
    core_attn_out = core_attn_out.reshape(z.shape).flatten(-2)
    proj_out, _ = self.out_proj(core_attn_out)
    return proj_out


def skip_zero_fill(units) -> bool:
    enabled = "u3" in units and _fuse47.unit_enabled("u3z")
    if enabled:
        _fuse47.note_route("u3z", "fused")
    return enabled
