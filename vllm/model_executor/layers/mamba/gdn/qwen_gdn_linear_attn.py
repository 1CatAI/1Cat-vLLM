# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import functools
import os
import time
from types import SimpleNamespace
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_mixed_qkv,
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    gdn_spec_metadata_tensors,
)
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

# Optional ROCm AITER Triton kernels for the GDN decode fast-path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = rocm_aiter_ops.are_gdn_triton_kernels_available()

if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)

_SM70_GDN_DUMP_COUNTS: dict[str, int] = {}
_SM70_GDN_PROJ_DUMP_COUNTS: dict[str, int] = {}
_SM70_GDN_PACKED_COMPARE_COUNTS: dict[str, int] = {}
_SM70_GDN_PACKED_COMPARE_REPORTS = 0
_SM70_GDN_GRAPH_BUFFERS: dict[str, torch.Tensor] = {}
_SM70_GDN_GRAPH_META: dict[str, dict[str, object]] = {}
_SM70_FLASHQLA_DECODE_ROUTE_DEBUG_COUNTS: dict[str, int] = {}
_SM70_GDN_PREFILL_PROFILE_COUNTS: dict[str, int] = {}
_SM70_GDN_PREFILL_WARMUP_KEYS: set[tuple[object, ...]] = set()


def _log_runtime_route_once(message: str, *args) -> None:
    if torch.compiler.is_compiling():
        return
    logger.info_once(message, *args)


def _sm70_gdn_prefill_profile_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_GDN_PREFILL_PROFILE")
    if raw is None or raw.strip().lower() not in ("1", "true", "yes", "on"):
        return False
    if torch.compiler.is_compiling():
        return False
    return not (
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    )


def _sm70_profile_trace_enabled() -> bool:
    return envs.VLLM_SM70_PROFILE_TRACE and not torch.compiler.is_compiling()


def _sm70_flashqla_original_prefill_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL")
    if raw is None:
        raw = os.getenv("FLASH_QLA_SM70_USE_ORIGINAL_TILELANG")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _sm70_flashqla_indexed_prefill_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_FLASHQLA_INDEXED_PREFILL")
    if raw is None:
        return False
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _sm70_flashqla_direct_output_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_FLASHQLA_DIRECT_OUTPUT")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _sm70_flashqla_decode_warmup_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_FLASHQLA_DECODE_WARMUP")
    if raw is None:
        return False
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _sm70_gdn_prefill_profile_start() -> float | None:
    if not _sm70_gdn_prefill_profile_enabled():
        return None
    torch.cuda.synchronize()
    return time.perf_counter()


def _sm70_gdn_prefill_profile_end(
    layer_name: LayerNameType,
    stage: str,
    start: float | None,
    *,
    tokens: int | None = None,
    details: str = "",
) -> None:
    if start is None:
        return
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    total_key = "__total__"
    max_logs = int(os.getenv("VLLM_SM70_GDN_PREFILL_PROFILE_MAX_LOGS", "256"))
    total = _SM70_GDN_PREFILL_PROFILE_COUNTS.get(total_key, 0)
    if total >= max_logs:
        return
    key = f"{os.getpid()}:{layer_name}:{stage}"
    per_stage_max = int(
        os.getenv("VLLM_SM70_GDN_PREFILL_PROFILE_MAX_PER_STAGE", "2")
    )
    count = _SM70_GDN_PREFILL_PROFILE_COUNTS.get(key, 0)
    if count >= per_stage_max:
        return
    _SM70_GDN_PREFILL_PROFILE_COUNTS[key] = count + 1
    _SM70_GDN_PREFILL_PROFILE_COUNTS[total_key] = total + 1
    logger.info(
        "SM70 GDN prefill profile: layer=%s stage=%s elapsed_ms=%.3f "
        "tokens=%s %s",
        layer_name,
        stage,
        elapsed_ms,
        tokens,
        details,
    )


def _sm70_log_flashqla_decode_route(
    *,
    layer_name: LayerNameType,
    stage: str,
    decision: str,
    reason: str,
    mixed_qkv: torch.Tensor,
    state_indices: torch.Tensor | None,
    num_decode_tokens: int,
) -> None:
    if not envs.VLLM_SM70_GDN_DECODE_FLASHQLA_ROUTE_DEBUG:
        return
    if torch.compiler.is_compiling():
        return
    key = f"{os.getpid()}:{stage}:{decision}:{reason}:{layer_name}"
    count = _SM70_FLASHQLA_DECODE_ROUTE_DEBUG_COUNTS.get(key, 0)
    if count >= 1:
        return
    if len(_SM70_FLASHQLA_DECODE_ROUTE_DEBUG_COUNTS) >= 96:
        return
    _SM70_FLASHQLA_DECODE_ROUTE_DEBUG_COUNTS[key] = count + 1
    state_desc = "None"
    if state_indices is not None:
        state_desc = (
            f"shape={tuple(state_indices.shape)} "
            f"dtype={state_indices.dtype} "
            f"stride={tuple(state_indices.stride())} "
            f"contiguous={state_indices.is_contiguous()}"
        )
    logger.info(
        "SM70 FlashQLA GDN decode route debug: layer=%s stage=%s "
        "decision=%s reason=%s capture=%s tokens=%s "
        "mixed_shape=%s mixed_dtype=%s mixed_stride=%s "
        "mixed_contiguous=%s state_indices=%s",
        layer_name,
        stage,
        decision,
        reason,
        torch.cuda.is_current_stream_capturing(),
        num_decode_tokens,
        tuple(mixed_qkv.shape),
        mixed_qkv.dtype,
        tuple(mixed_qkv.stride()),
        mixed_qkv.is_contiguous(),
        state_desc,
    )


def _sm70_dump_gdn_core_tensor(
    label: str,
    layer_name: LayerNameType,
    tensor: torch.Tensor,
) -> None:
    _sm70_gdn_graph_buffer_copy(label, layer_name, tensor, "core")
    if torch.cuda.is_current_stream_capturing():
        return
    dump_dir = os.getenv("VLLM_SM70_DUMP_GDN_CORE_DIR")
    if not dump_dir:
        return
    raw_layer_ids = os.getenv("VLLM_SM70_DUMP_GDN_CORE_LAYER_IDS")
    if raw_layer_ids:
        layer_idx = _sm70_gdn_layer_idx(layer_name)
        layer_ids = _sm70_parse_int_set(raw_layer_ids, set())
        if layer_idx not in layer_ids:
            return
    enable_file = os.getenv("VLLM_SM70_DUMP_GDN_CORE_ENABLE_FILE")
    if enable_file and not os.path.exists(enable_file):
        return
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return

    try:
        max_dumps = int(os.getenv("VLLM_SM70_DUMP_GDN_CORE_MAX_DUMPS", "4"))
    except ValueError:
        max_dumps = 4
    if max_dumps <= 0:
        return

    key = f"{os.getpid()}:{label}"
    count = _SM70_GDN_DUMP_COUNTS.get(key, 0)
    if count >= max_dumps:
        return
    _SM70_GDN_DUMP_COUNTS[key] = count + 1

    safe_layer = str(layer_name).replace("/", "_").replace(".", "_")
    path = os.path.join(
        dump_dir,
        f"pid{os.getpid()}_{label}_{count:03d}_{safe_layer}.pt",
    )
    os.makedirs(dump_dir, exist_ok=True)
    torch.save(
        {
            "label": label,
            "layer_name": str(layer_name),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "tensor": tensor.detach().cpu(),
        },
        path,
    )


def _sm70_gdn_layer_idx(layer_name: LayerNameType) -> int | None:
    parts = str(layer_name).split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def _sm70_parse_int_set(raw: str | None, default: set[int]) -> set[int]:
    if raw is None or not raw.strip():
        return default
    try:
        return {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError:
        return default


def _sm70_parse_int_ranges(raw_ranges: str | None) -> set[int] | None:
    if not raw_ranges:
        return None
    values: set[int] = set()
    for raw_part in raw_ranges.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return values


def _sm70_qwen_gdn_has_active_spec_decode(
    layer_name: LayerNameType,
) -> bool:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return False
    attn_metadata_raw = forward_context.attn_metadata
    if not isinstance(attn_metadata_raw, dict):
        return False
    attn_metadata = attn_metadata_raw.get(_resolve_layer_name(layer_name))
    if not isinstance(attn_metadata, GDNAttentionMetadata):
        return False
    return (
        attn_metadata.spec_sequence_masks is not None
        and attn_metadata.num_spec_decodes > 0
    )


def _sm70_qwen_gdn_full_forward_enabled(
    layer_name: LayerNameType,
    *,
    force_enabled: bool,
    disabled: bool,
    auto_enabled: bool,
) -> bool:
    if disabled:
        return False
    if force_enabled:
        return True
    if not auto_enabled:
        return False
    return _sm70_qwen_gdn_has_active_spec_decode(layer_name)


def _sm70_qwen_gdn_input_core_boundary_enabled() -> bool:
    if envs.VLLM_SM70_QWEN_GDN_DISABLE_INPUT_CORE_OP:
        return False
    return envs.VLLM_SM70_QWEN_GDN_INPUT_CORE_OP


def _sm70_qwen_gdn_spec_core_enabled(
    layer_name: LayerNameType,
    *,
    auto_enabled: bool,
) -> bool:
    return auto_enabled and _sm70_qwen_gdn_has_active_spec_decode(layer_name)


def _qwen_gdn_run_recurrent_core(
    self: "QwenGatedDeltaNetAttention",
    *,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    conv_state_cache: torch.Tensor | None = None,
    ssm_state_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the Qwen GDN recurrent core through one strategy boundary.

    Input projection and output projection stay on the common forward path.
    Only the recurrent-state commit semantics differ between non-spec and
    active speculative decode, so keep that dispatch localized here.
    """
    if envs.VLLM_SM70_QWEN_GDN_CONTEXT_CORE:
        torch.ops.vllm.qwen_gdn_attention_core_context(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            layer_name,
        )
        return core_attn_out

    if conv_state_cache is None or ssm_state_cache is None:
        conv_state_cache, ssm_state_cache = _resolve_qwen_gdn_kv_cache_args(
            layer_name,
            core_attn_out,
        )

    if _sm70_qwen_gdn_spec_core_enabled(
        layer_name,
        auto_enabled=getattr(self, "auto_sm70_qwen_gdn_003_spec_core", False),
    ):
        return torch.ops.vllm.qwen_gdn_attention_core_003_spec(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_state_cache,
            ssm_state_cache,
            layer_name,
        )

    if _sm70_qwen_gdn_spec_core_enabled(
        layer_name,
        auto_enabled=getattr(self, "auto_sm70_qwen_gdn_spec_core", False),
    ):
        (
            non_spec_query_start_loc,
            non_spec_state_indices_tensor,
            spec_query_start_loc,
            spec_state_indices_tensor,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
        ) = _qwen_gdn_metadata_tensors(
            layer_name,
            core_attn_out.device,
        )
        return torch.ops.vllm.qwen_gdn_attention_core_spec_commit(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_state_cache,
            ssm_state_cache,
            non_spec_query_start_loc,
            non_spec_state_indices_tensor,
            spec_query_start_loc,
            spec_state_indices_tensor,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
            layer_name,
        )

    if getattr(self, "auto_sm70_qwen_gdn_full_forward", False) and (
        _sm70_qwen_gdn_has_active_spec_decode(layer_name)
    ):
        (
            non_spec_query_start_loc,
            non_spec_state_indices_tensor,
            spec_query_start_loc,
            spec_state_indices_tensor,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
        ) = _qwen_gdn_metadata_tensors(
            layer_name,
            core_attn_out.device,
        )
        torch.ops.vllm.qwen_gdn_attention_core_standard_spec(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_state_cache,
            ssm_state_cache,
            non_spec_query_start_loc,
            non_spec_state_indices_tensor,
            spec_query_start_loc,
            spec_state_indices_tensor,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
            layer_name,
        )
        return core_attn_out

    (
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
    ) = _qwen_gdn_non_spec_metadata_tensors(
        layer_name,
        core_attn_out.device,
    )
    torch.ops.vllm.qwen_gdn_attention_core_standard(
        mixed_qkv,
        b,
        a,
        core_attn_out,
        conv_state_cache,
        ssm_state_cache,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        layer_name,
    )
    return core_attn_out


def _qwen_gdn_metadata_tensors(
    layer_name: LayerNameType,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return gdn_spec_metadata_tensors(None, device)
    attn_metadata_raw = forward_context.attn_metadata
    if not isinstance(attn_metadata_raw, dict):
        return gdn_spec_metadata_tensors(None, device)
    attn_metadata = attn_metadata_raw.get(_resolve_layer_name(layer_name))
    if not isinstance(attn_metadata, GDNAttentionMetadata):
        return gdn_spec_metadata_tensors(None, device)
    return gdn_spec_metadata_tensors(attn_metadata, device)


def _qwen_gdn_non_spec_metadata_tensors(
    layer_name: LayerNameType,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    empty = torch.empty(0, dtype=torch.int32, device=device)
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return empty, empty
    attn_metadata_raw = forward_context.attn_metadata
    if not isinstance(attn_metadata_raw, dict):
        return empty, empty
    attn_metadata = attn_metadata_raw.get(_resolve_layer_name(layer_name))
    if not isinstance(attn_metadata, GDNAttentionMetadata):
        return empty, empty

    def _or_empty(tensor: torch.Tensor | None) -> torch.Tensor:
        return tensor if tensor is not None else empty

    return (
        _or_empty(attn_metadata.non_spec_query_start_loc),
        _or_empty(attn_metadata.non_spec_state_indices_tensor),
    )


def _sm70_gdn_graph_buffer_copy(
    label: str,
    layer_name: LayerNameType,
    tensor: torch.Tensor,
    source: str,
) -> None:
    dump_dir = os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_DIR")
    if os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_BUFFERS") != "1" or not dump_dir:
        return
    if not tensor.is_cuda:
        return

    target_labels = {
        item.strip()
        for item in os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_LABELS", "").split(",")
        if item.strip()
    }
    if target_labels and label not in target_labels:
        return

    shape = tuple(tensor.shape)
    raw_shapes = os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_SHAPES")
    if raw_shapes:
        allowed_shapes = {
            item.strip() for item in raw_shapes.split(",") if item.strip()
        }
        shape_text = "x".join(str(dim) for dim in shape)
        if shape_text not in allowed_shapes:
            return

    layer_idx = _sm70_gdn_layer_idx(layer_name)
    raw_layer_ids = os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_LAYER_IDS")
    if raw_layer_ids:
        try:
            layer_ids = _sm70_parse_int_ranges(raw_layer_ids) or set()
        except ValueError:
            layer_ids = set()
        if layer_idx not in layer_ids:
            return

    key = f"{os.getpid()}:{source}:{layer_name}:{label}:{shape}"
    buffer = _SM70_GDN_GRAPH_BUFFERS.get(key)
    if (
        buffer is None
        or tuple(buffer.shape) != shape
        or buffer.dtype != tensor.dtype
        or buffer.device != tensor.device
    ):
        buffer = torch.empty_like(tensor)
        _SM70_GDN_GRAPH_BUFFERS[key] = buffer
        _SM70_GDN_GRAPH_META[key] = {
            "label": label,
            "layer_name": str(layer_name),
            "layer_idx": layer_idx,
            "source": source,
            "shape": shape,
            "dtype": str(tensor.dtype),
            "pid": os.getpid(),
        }
    buffer.copy_(tensor)


def _sm70_gdn_graph_buffer_copy_state_slice(
    label: str,
    layer_name: LayerNameType,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    num_tokens: int,
) -> None:
    if state_indices is None or num_tokens <= 0:
        return
    if os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_BUFFERS") != "1":
        return
    indices = state_indices[:num_tokens].to(device=state.device, dtype=torch.long)
    indices = indices.clamp(0, state.shape[0] - 1)
    _sm70_gdn_graph_buffer_copy(
        label,
        layer_name,
        state.index_select(0, indices),
        "state",
    )
    if os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_STATE_INDICES") == "1":
        _sm70_gdn_graph_buffer_copy(
            f"{label}_indices",
            layer_name,
            indices.to(dtype=torch.int32),
            "state",
        )


def _sm70_dump_gdn_spec_metadata_graph_buffers(
    layer_name: LayerNameType,
    *,
    non_spec_query_start_loc: torch.Tensor | None = None,
    non_spec_state_indices_tensor: torch.Tensor | None = None,
    spec_query_start_loc: torch.Tensor | None = None,
    spec_state_indices_tensor: torch.Tensor | None = None,
    spec_token_indx: torch.Tensor | None = None,
    non_spec_token_indx: torch.Tensor | None = None,
    spec_sequence_masks: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> None:
    if os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_METADATA") != "1":
        return
    for label, tensor in (
        ("meta_non_spec_query_start_loc", non_spec_query_start_loc),
        ("meta_non_spec_state_indices", non_spec_state_indices_tensor),
        ("meta_spec_query_start_loc", spec_query_start_loc),
        ("meta_spec_state_indices", spec_state_indices_tensor),
        ("meta_spec_token_indx", spec_token_indx),
        ("meta_non_spec_token_indx", non_spec_token_indx),
        ("meta_spec_sequence_masks", spec_sequence_masks),
        ("meta_num_accepted_tokens", num_accepted_tokens),
    ):
        if tensor is None or tensor.numel() == 0:
            continue
        _sm70_gdn_graph_buffer_copy(label, layer_name, tensor, "meta")


def dump_sm70_gdn_graph_buffers(step: int, stage: str) -> None:
    dump_dir = os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_DIR")
    if not dump_dir:
        return
    enable_file = os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_ENABLE_FILE")
    if enable_file and not os.path.exists(enable_file):
        return
    target_steps = _sm70_parse_int_ranges(
        os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_STEPS")
    )
    if target_steps is not None and step not in target_steps:
        return
    if not _SM70_GDN_GRAPH_BUFFERS:
        return

    os.makedirs(dump_dir, exist_ok=True)
    for key, buffer in _SM70_GDN_GRAPH_BUFFERS.items():
        meta = _SM70_GDN_GRAPH_META.get(key, {})
        label = str(meta.get("label", "unknown")).replace("/", "_").replace(".", "_")
        source = str(meta.get("source", "unknown")).replace("/", "_").replace(".", "_")
        layer_idx = meta.get("layer_idx")
        layer_text = "none" if layer_idx is None else f"{int(layer_idx):02d}"
        shape = "x".join(str(dim) for dim in tuple(buffer.shape))
        path = os.path.join(
            dump_dir,
            (
                f"pid{os.getpid()}_step{step:04d}_layer{layer_text}_"
                f"{source}_{label}_shape{shape}.pt"
            ),
        )
        torch.save(
            {
                **meta,
                "step": step,
                "stage": stage,
                "graph_buffer_key": key,
                "tensor": buffer.detach().cpu(),
            },
            path,
        )


def _sm70_gdn_packed_compare_request(
    layer_name: LayerNameType,
) -> tuple[str, int] | None:
    dump_dir = os.getenv("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_DIR")
    if not dump_dir:
        return None
    enable_file = os.getenv("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_ENABLE_FILE")
    if enable_file and not os.path.exists(enable_file):
        return None
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None

    layer_idx = _sm70_gdn_layer_idx(layer_name)
    if layer_idx is None:
        return None
    layer_ids = _sm70_parse_int_set(
        os.getenv("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_LAYER_IDS"),
        {0},
    )
    if layer_idx not in layer_ids:
        return None

    global _SM70_GDN_PACKED_COMPARE_REPORTS
    max_reports = envs.VLLM_SM70_COMPARE_GDN_PACKED_DECODE_MAX_REPORTS
    if max_reports <= 0 or max_reports <= _SM70_GDN_PACKED_COMPARE_REPORTS:
        return None

    key = f"{os.getpid()}:{layer_name}"
    step = _SM70_GDN_PACKED_COMPARE_COUNTS.get(key, 0) + 1
    _SM70_GDN_PACKED_COMPARE_COUNTS[key] = step

    target_steps = _sm70_parse_int_set(
        os.getenv("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_STEPS"),
        set(),
    )
    if target_steps and step not in target_steps:
        return None

    _SM70_GDN_PACKED_COMPARE_REPORTS += 1
    os.makedirs(dump_dir, exist_ok=True)
    safe_layer = str(layer_name).replace("/", "_").replace(".", "_")
    path = os.path.join(
        dump_dir,
        f"pid{os.getpid()}_step{step:04d}_layer{safe_layer}.pt",
    )
    return path, step


def _sm70_save_gdn_packed_compare_report(
    path: str,
    layer_name: LayerNameType,
    step: int,
    state_indices: torch.Tensor,
    packed_out: torch.Tensor,
    packed_state: torch.Tensor,
    mixed_out: torch.Tensor,
    mixed_state: torch.Tensor,
    ref_out: torch.Tensor,
    ref_state: torch.Tensor,
) -> None:
    def _canonical_out(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return tensor

    packed_out_2d = _canonical_out(packed_out)
    mixed_out_2d = _canonical_out(mixed_out)
    ref_out_2d = _canonical_out(ref_out)

    def _comparison(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> dict[str, float | int | tuple[int, ...]]:
        max_diff, mean_diff, num_different = _sm70_diff_stats(left, right)
        return {
            "left_shape": tuple(left.shape),
            "right_shape": tuple(right.shape),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "num_different": num_different,
        }

    torch.save(
        {
            "layer_name": str(layer_name),
            "layer_idx": _sm70_gdn_layer_idx(layer_name),
            "pid": os.getpid(),
            "step": step,
            "state_indices": state_indices.detach().cpu(),
            "out_shape": tuple(packed_out_2d.shape),
            "state_shape": tuple(packed_state[1:].shape),
            "packed_vs_mixed_out": _comparison(packed_out_2d, mixed_out_2d),
            "packed_vs_ref_out": _comparison(packed_out_2d, ref_out_2d),
            "mixed_vs_ref_out": _comparison(mixed_out_2d, ref_out_2d),
            "packed_vs_mixed_state": _comparison(packed_state[1:], mixed_state[1:]),
            "packed_vs_ref_state": _comparison(packed_state[1:], ref_state[1:]),
            "mixed_vs_ref_state": _comparison(mixed_state[1:], ref_state[1:]),
        },
        path,
    )


def _sm70_gdn_projection_dump_requested(layer_name: LayerNameType) -> bool:
    if not os.getenv("VLLM_SM70_DUMP_GDN_PROJ_DIR") and (
        os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_BUFFERS") != "1"
        or not os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_DIR")
    ):
        return False
    layer_idx = _sm70_gdn_layer_idx(layer_name)
    if layer_idx is None:
        return False
    raw_graph_layer_ids = os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_LAYER_IDS")
    if raw_graph_layer_ids and os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_BUFFERS") == "1":
        try:
            graph_layer_ids = _sm70_parse_int_ranges(raw_graph_layer_ids) or set()
        except ValueError:
            graph_layer_ids = set()
        return layer_idx in graph_layer_ids
    raw_layer_ids = os.getenv("VLLM_SM70_DUMP_GDN_PROJ_LAYER_IDS", "0,1")
    try:
        layer_ids = {
            int(item.strip()) for item in raw_layer_ids.split(",") if item.strip()
        }
    except ValueError:
        layer_ids = {0, 1}
    return layer_idx in layer_ids


def _sm70_gdn_projection_dump_impl(
    tensor: torch.Tensor,
    label: str,
    layer_name: LayerNameType,
) -> torch.Tensor:
    _sm70_gdn_graph_buffer_copy(label, layer_name, tensor, "proj")
    if torch.cuda.is_current_stream_capturing():
        return tensor
    dump_dir = os.getenv("VLLM_SM70_DUMP_GDN_PROJ_DIR")
    enable_file = os.getenv("VLLM_SM70_DUMP_GDN_PROJ_ENABLE_FILE")
    can_save = bool(dump_dir) and (
        not enable_file or os.path.exists(enable_file)
    )
    if can_save and not torch.cuda.is_current_stream_capturing():
        try:
            max_dumps = int(os.getenv("VLLM_SM70_DUMP_GDN_PROJ_MAX_DUMPS", "4"))
        except ValueError:
            max_dumps = 4
        key = f"{os.getpid()}:{layer_name}:{label}"
        count = _SM70_GDN_PROJ_DUMP_COUNTS.get(key, 0)
        if max_dumps > 0 and count < max_dumps:
            _SM70_GDN_PROJ_DUMP_COUNTS[key] = count + 1
            safe_layer = str(layer_name).replace("/", "_").replace(".", "_")
            safe_label = label.replace("/", "_").replace(".", "_")
            path = os.path.join(
                dump_dir,
                f"pid{os.getpid()}_{safe_label}_{count:03d}_{safe_layer}.pt",
            )
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(
                {
                    "label": label,
                    "layer_name": str(layer_name),
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "tensor": tensor.detach().cpu(),
                },
                path,
            )
    return tensor


def _sm70_gdn_projection_dump_fake(
    tensor: torch.Tensor,
    label: str,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return tensor


direct_register_custom_op(
    op_name="sm70_gdn_projection_dump",
    op_func=_sm70_gdn_projection_dump_impl,
    mutates_args=[],
    fake_impl=_sm70_gdn_projection_dump_fake,
)


def _sm70_dump_gdn_projection_tensor(
    label: str,
    layer_name: LayerNameType,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if not _sm70_gdn_projection_dump_requested(layer_name):
        return tensor
    tensor = torch.ops.vllm.sm70_gdn_projection_dump(tensor, label, layer_name)
    return tensor


def _sm70_diff_stats(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float, float, int]:
    diff = (left - right).abs()
    if diff.numel() == 0:
        return 0.0, 0.0, 0
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        int(torch.count_nonzero(left != right).item()),
    )


def _sm70_compile_graph_slice_dim(
    tensor: torch.Tensor,
    dim: int,
    start: int,
    size: int,
) -> torch.Tensor:
    if dim < 0:
        dim += tensor.ndim
    if start == 0 or not envs.VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH:
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(start, start + size)
        return tensor[tuple(slices)]
    indices = torch.arange(start, start + size, device=tensor.device)
    return tensor.index_select(dim, indices)


def _sm70_qwen_gdn_rmsnorm_gated_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

    return rmsnorm_fn(
        x,
        weight,
        None,
        z=z,
        eps=eps,
        group_size=None if group_size <= 0 else group_size,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )


def _sm70_qwen_gdn_rmsnorm_gated_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="sm70_qwen_gdn_rmsnorm_gated",
    op_func=_sm70_qwen_gdn_rmsnorm_gated_impl,
    fake_impl=_sm70_qwen_gdn_rmsnorm_gated_fake,
)


def _sm70_compile_graph_interleaved_indices(
    num_groups: int,
    group_width: int,
    start: int,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    group_starts = torch.arange(num_groups, device=device) * group_width + start
    offsets = torch.arange(size, device=device)
    return (group_starts[:, None] + offsets[None, :]).reshape(-1)


def _sm70_state_indices_for_tokens(
    state_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    num_tokens: int,
) -> torch.Tensor | None:
    if state_indices is None:
        return None
    if state_indices.ndim == 1:
        return state_indices[:num_tokens].reshape(-1).to(torch.long)
    if cu_seqlens is None:
        return state_indices.reshape(-1)[:num_tokens].to(torch.long)

    pieces: list[torch.Tensor] = []
    num_sequences = max(cu_seqlens.numel() - 1, 0)
    for seq_idx in range(num_sequences):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = max(end - start, 0)
        if seq_len == 0:
            continue
        pieces.append(state_indices[seq_idx, :seq_len].reshape(-1))
    if not pieces:
        return state_indices.new_empty(0, dtype=torch.long)
    return torch.cat(pieces, dim=0)[:num_tokens].to(torch.long)


def _resolve_qwen_gdn_kv_cache_args(
    layer_name: LayerNameType,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    kv_cache = getattr(layer, "kv_cache", None)
    if kv_cache is not None:
        return kv_cache[0], kv_cache[1]
    return fallback.new_empty((0,)), fallback.new_empty((0,))


def _sm70_mixed_qkv_state_diff_stats(
    mixed_state: torch.Tensor,
    ref_state: torch.Tensor,
    state_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    num_tokens: int,
) -> dict[str, float | int | bool]:
    token_indices = _sm70_state_indices_for_tokens(
        state_indices,
        cu_seqlens,
        num_tokens,
    )
    if token_indices is None:
        return {
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "num_different": 0,
            "valid_tokens": 0,
            "invalid_tokens": 0,
            "shape_mismatch": False,
        }

    token_indices = token_indices.to(device=ref_state.device, dtype=torch.long)
    token_indices = token_indices[: min(token_indices.numel(), mixed_state.shape[0])]
    valid = (token_indices > 0) & (token_indices < ref_state.shape[0])
    valid_tokens = int(torch.count_nonzero(valid).item())
    invalid_tokens = int(token_indices.numel() - valid_tokens)
    if valid_tokens == 0:
        return {
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "num_different": 0,
            "valid_tokens": valid_tokens,
            "invalid_tokens": invalid_tokens,
            "shape_mismatch": False,
        }

    mixed_valid = mixed_state[: token_indices.numel()][valid]
    ref_valid = ref_state.index_select(0, token_indices[valid])
    if mixed_valid.shape != ref_valid.shape:
        return {
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "num_different": 0,
            "valid_tokens": valid_tokens,
            "invalid_tokens": invalid_tokens,
            "shape_mismatch": True,
        }

    max_diff, mean_diff, num_different = _sm70_diff_stats(mixed_valid, ref_valid)
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "num_different": num_different,
        "valid_tokens": valid_tokens,
        "invalid_tokens": invalid_tokens,
        "shape_mismatch": False,
    }


# TODO(arpera): remove ``_is_libs_cu13_install_intact`` and its caller in
# ``_resolve_gdn_prefill_backend`` once the upstream packaging bug is
# fixed and the broken wheels are yanked / superseded on PyPI:
#   https://github.com/NVIDIA/cutlass/issues/3170
#   https://github.com/NVIDIA/cutlass/issues/3259
@functools.cache
def _is_libs_cu13_install_intact() -> bool:
    """Return True if every file installed by ``nvidia-cutlass-dsl-libs-cu13``
    matches the SHA-256 declared in its wheel ``RECORD``.

    ``nvidia-cutlass-dsl-libs-base`` and ``nvidia-cutlass-dsl-libs-cu13``
    both ship into the shared ``nvidia_cutlass_dsl/`` namespace and
    write many of the same on-disk paths (the runtime ``.so``, the MLIR
    Python bindings, cuTe-DSL Python sources, ...) with different
    content. Whichever wheel extracts last wins; with a parallel
    installer (e.g. ``uv``) the order is racy and the resulting venv
    can end up with a mix of files from both variants. The
    ``-libs-base`` variant fails MLIR legalization when JIT-compiling
    the FlashInfer Blackwell GDN prefill kernel, and any other
    cuTe-DSL-based kernel can break too if on-disk files diverge from
    what ``-libs-cu13``'s wheel expects. Tracked upstream at:

      * https://github.com/NVIDIA/cutlass/issues/3170
      * https://github.com/NVIDIA/cutlass/issues/3259

    This helper re-hashes every file the ``-libs-cu13`` wheel claims to
    own and compares against its declared SHA-256. Returns False on any
    error (uninstalled, missing RECORD, missing file, hash mismatch).
    Result is cached per-process.
    """
    import hashlib
    import importlib.metadata

    import pybase64 as base64

    try:
        dist = importlib.metadata.distribution("nvidia-cutlass-dsl-libs-cu13")
    except importlib.metadata.PackageNotFoundError:
        return False

    files = dist.files
    if not files:
        return False

    for pkg_path in files:
        file_hash = pkg_path.hash
        # Skip RECORD rows without a hash (RECORD itself, generated
        # ``.pyc`` files, ...) and any non-SHA-256 hash modes.
        if file_hash is None or not file_hash.value:
            continue
        if file_hash.mode != "sha256":
            continue
        try:
            with open(pkg_path.locate(), "rb") as f:
                digest = hashlib.sha256(f.read()).digest()
        except OSError:
            return False
        actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        if actual != file_hash.value:
            return False

    return True


def _get_gdn_head_k_dim(vllm_config: VllmConfig) -> int | None:
    for config in (
        getattr(vllm_config.model_config, "hf_text_config", None),
        getattr(vllm_config.model_config, "hf_config", None),
    ):
        if config is None:
            continue
        head_k_dim = getattr(config, "linear_key_head_dim", None)
        if head_k_dim is not None:
            return int(head_k_dim)
    return None


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl", "flashqla_sm70"]]:
    """Resolve GDN prefill backend.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) — no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``,
        and an intact ``nvidia-cutlass-dsl-libs-cu13`` install on disk
        (see :func:`_is_libs_cu13_install_intact`).

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
    """
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = _get_gdn_head_k_dim(vllm_config)
    model_dtype = getattr(vllm_config.model_config, "dtype", None)

    supports_flashinfer = False
    supports_cutedsl = False
    supports_flashqla_sm70 = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif head_k_dim == 128 and backend in ("auto", "flashqla_sm70"):
        capability = current_platform.get_device_capability()
        is_sm70_or_sm75 = (
            capability is not None
            and capability.major == 7
            and capability.minor in (0, 5)
        )
        supports_model_dtype = model_dtype == torch.float16
        try:
            from flash_qla.ops.gated_delta_rule.chunk.sm70 import (  # noqa: F401
                chunk_gated_delta_rule_fwd_sm70_vlk_varlen,
            )
        except ImportError:
            supports_flashqla_sm70 = False
        else:
            supports_flashqla_sm70 = is_sm70_or_sm75 and supports_model_dtype
        if is_sm70_or_sm75 and not supports_model_dtype:
            logger.warning_once(
                "FlashQLA-SM70 GDN prefill is V100 production-validated only "
                "for fp16 model activations; model dtype %s falls back to "
                "Triton/FLA.",
                model_dtype,
            )
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = _is_libs_cu13_install_intact()
        supports_cutedsl = True
        if not supports_flashinfer:
            logger.warning_once(
                "FlashInfer Blackwell GDN requires an intact nvidia-cutlass-dsl"
                "-libs-cu13 install, but some on-disk files do not match the "
                "SHA-256 declared in its RECORD (install-order race in "
                "nvidia-cutlass-dsl packaging -- see "
                "https://github.com/NVIDIA/cutlass/issues/3170 and "
                "https://github.com/NVIDIA/cutlass/issues/3259). Falling back "
                "to Triton/FLA. Repair with: pip install --force-reinstall "
                "--no-deps nvidia-cutlass-dsl-libs-cu13"
            )

    if backend in ("auto", "flashqla_sm70") and supports_flashqla_sm70:
        return backend, "flashqla_sm70"
    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = _get_gdn_head_k_dim(vllm_config)
    model_dtype = getattr(vllm_config.model_config, "dtype", None)
    chosen = {
        "flashinfer": "FlashInfer",
        "flashqla_sm70": "FlashQLA-SM70",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s, "
        "model_dtype=%s).",
        chosen,
        requested_backend,
        head_k_dim,
        model_dtype,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


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
):
    use_original_tilelang = _sm70_flashqla_original_prefill_enabled()
    if use_original_tilelang:
        from flash_qla.ops.gated_delta_rule.chunk import (
            chunk_gated_delta_rule_fwd_sm70_tilelang,
        )
        _log_runtime_route_once(
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
        profile_start = _sm70_gdn_prefill_profile_start()
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
        _sm70_gdn_prefill_profile_end(
            "flashqla",
            "qk_l2norm",
            profile_start,
            tokens=q.shape[1],
        )
    if cu_seqlens is None:
        cu_seqlens = torch.tensor(
            [0, q.shape[1]], device=q.device, dtype=torch.int32
        )
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
    profile_start = _sm70_gdn_prefill_profile_start()
    q_arg = q if q_contiguous else q.contiguous()
    k_arg = k if k_contiguous else k.contiguous()
    v_arg = v if v_contiguous else v.contiguous()
    g_arg = g if g_contiguous else g.contiguous()
    beta_arg = beta if beta_contiguous else beta.contiguous()
    state_arg = initial_state if state_contiguous else initial_state.contiguous()
    cu_arg = cu_seqlens if cu_seqlens.is_contiguous() else cu_seqlens.contiguous()
    _sm70_gdn_prefill_profile_end(
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

    profile_start = _sm70_gdn_prefill_profile_start()
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
    _sm70_gdn_prefill_profile_end(
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
        profile_start = _sm70_gdn_prefill_profile_start()
        out_flat = out.squeeze(0).reshape(-1)
        out_view = core_attn_out.reshape(-1)[: out_flat.numel()]
        out_view.copy_(out_flat)
        out = core_attn_out[: out.shape[1]].unsqueeze(0)
        _sm70_gdn_prefill_profile_end(
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
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
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


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend

        if (
            backend in ("flashinfer", "cutedsl", "flashqla_sm70")
            and active_backend != backend
        ):
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)

        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "flashqla_sm70":
            self._forward_method = self.forward_flashqla_sm70
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
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
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        gate_is_exp: bool = False,
    ):
        if gate_is_exp:
            g = torch.log(g)
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

    def forward_flashqla_sm70(
        self,
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
    ):
        if (
            q.dtype != torch.float16
            or k.dtype != torch.float16
            or v.dtype != torch.float16
        ):
            logger.warning_once(
                "FlashQLA-SM70 GDN prefill only runs on fp16 q/k/v tensors "
                "for SM70/V100 production use; got q=%s k=%s v=%s. Falling "
                "using the native GDN path for this call.",
                q.dtype,
                k.dtype,
                v.dtype,
            )
            return self.forward_native(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                core_attn_out=core_attn_out,
                gate_is_exp=gate_is_exp,
            )
        o, final_state = flashqla_sm70_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            inplace_final_state=inplace_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
            gate_is_exp=gate_is_exp,
        )
        return o, final_state

    def forward_native(
        self,
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
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        gate_is_exp: bool = False,
    ):
        if gate_is_exp:
            g = torch.log(g)
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )

    def forward_cutedsl(
        self,
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
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        gate_is_exp: bool = False,
    ):
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            chunk_gated_delta_rule_cutedsl,
        )

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        if gate_is_exp:
            g = torch.log(g)

        assert cu_seqlens is not None
        assert chunk_indices is not None
        assert chunk_offsets is not None

        o, final_state = chunk_gated_delta_rule_cutedsl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        if not output_final_state:
            final_state = None
        return o, final_state


@PluggableLayer.register("qwen_gated_delta_net_attention")
class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        gqa_interleaved_layout=False,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gqa_interleaved_layout = gqa_interleaved_layout
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        else:
            self._forward_method = self.forward_cuda
        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )

        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )
        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.gdn_prefill_backend = self.chunk_gated_delta_rule.gdn_prefill_backend
        self._prefill_kernels_warmed_up = False
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )
        self.enable_sm70_fused_sigmoid_mixed_qkv = (
            envs.VLLM_SM70_FUSED_SIGMOID_MIXED_QKV
        )
        self.compare_sm70_fused_sigmoid_mixed_qkv = (
            envs.VLLM_SM70_FUSED_SIGMOID_MIXED_QKV_COMPARE
        )
        self.enable_flashqla_decode = envs.VLLM_SM70_GDN_DECODE_FLASHQLA
        self.force_sm70_qwen_gdn_full_forward = (
            envs.VLLM_SM70_QWEN_GDN_FULL_FORWARD
        )
        self.disable_sm70_qwen_gdn_full_forward = (
            envs.VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD
        )
        self.auto_sm70_qwen_gdn_full_forward = (
            envs.VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH
            and vllm_config.speculative_config is not None
            and not envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
            and not envs.VLLM_SM70_QWEN_GDN_003_SPEC_CORE_OP
        )
        self.auto_sm70_qwen_gdn_spec_core = (
            envs.VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH
            and vllm_config.speculative_config is not None
            and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
        )
        self.auto_sm70_qwen_gdn_003_spec_core = (
            envs.VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH
            and vllm_config.speculative_config is not None
            and envs.VLLM_SM70_QWEN_GDN_003_SPEC_CORE_OP
        )
        self.maybe_sm70_qwen_gdn_full_forward = (
            not self.disable_sm70_qwen_gdn_full_forward
            and (
                self.force_sm70_qwen_gdn_full_forward
                or self.auto_sm70_qwen_gdn_full_forward
            )
        )
        if self.maybe_sm70_qwen_gdn_full_forward:
            logger.info_once(
                "SM70 Qwen GDN full-forward guard armed "
                "(force=%s, auto=%s, spec_core=%s).",
                self.force_sm70_qwen_gdn_full_forward,
                self.auto_sm70_qwen_gdn_full_forward,
                self.auto_sm70_qwen_gdn_spec_core,
            )
        if self.auto_sm70_qwen_gdn_003_spec_core:
            logger.info_once(
                "SM70 Qwen GDN 0.0.3-style spec recurrent-core route armed."
            )
        self.enable_sm70_legacy_prefill_prep = (
            envs.VLLM_SM70_GDN_LEGACY_PREFILL_PREP
        )
        if current_platform.is_device_capability(70) and (
            envs.VLLM_SM70_GDN_KKT_SCHEDULE
            or envs.VLLM_SM70_GDN_DELTA_H_SCHEDULE
            or envs.VLLM_SM70_GDN_CHUNK_O_SCHEDULE
            or envs.VLLM_SM70_FLA_RECURRENT_SCHEDULE
            or envs.VLLM_SM70_FUSED_SIGMOID_GATING_SCHED
        ):
            logger.info_once(
                "SM70 GDN/FLA schedule gates enabled: "
                "kkt=%s delta_h=%s chunk_o=%s recurrent=%s "
                "sigmoid_gating=%s.",
                envs.VLLM_SM70_GDN_KKT_SCHEDULE,
                envs.VLLM_SM70_GDN_DELTA_H_SCHEDULE,
                envs.VLLM_SM70_GDN_CHUNK_O_SCHEDULE,
                envs.VLLM_SM70_FLA_RECURRENT_SCHEDULE,
                envs.VLLM_SM70_FUSED_SIGMOID_GATING_SCHED,
                scope="local",
            )
        if self.enable_sm70_legacy_prefill_prep:
            logger.info_once(
                "SM70 GDN legacy prefill prep diagnostic route enabled.",
                scope="local",
            )
        if self.enable_flashqla_decode:
            logger.info_once(
                "SM70 FlashQLA GDN decode route enabled.",
                scope="local",
            )
        if envs.is_set("VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING"):
            logger.info_once(
                "VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING is upstream-split "
                "into VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE, "
                "VLLM_SM70_FUSED_SIGMOID_GATING_* and "
                "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV controls.",
                scope="local",
            )
        if envs.VLLM_SM70_GDN_EMPTY_CORE_OUT:
            logger.info_once(
                "VLLM_SM70_GDN_EMPTY_CORE_OUT is paused-unsafe; latest keeps "
                "GDN core_attn_out allocated with torch.zeros until a route-hit "
                "and model-quality gate proves empty allocation is safe.",
                scope="local",
            )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=self.maybe_disable_tp(quant_config),
        )

    def maybe_disable_tp(self, quant_config: QuantizationConfig | None) -> bool:
        """Whether to replicate ba_proj instead of TP-sharding it.

        Marlin requires output_size_per_partition >= MIN_THREAD_N=64, which
        the Qwen3.5 non-interleaved [num_v_heads]*2 layout violates at TP>=2
        (e.g. num_v_heads=64, TP=4 -> 16). Replicating the projection keeps
        each rank above the Marlin threshold; forward() then slices b/a to
        the local TP partition. Qwen3-Next's interleaved [num_v_heads*2]
        layout is unaffected and stays TP-sharded.

        See https://github.com/vllm-project/vllm/issues/35924
        """
        return (
            current_platform.is_cuda()
            and not self.gqa_interleaved_layout
            and isinstance(quant_config, (AWQMarlinConfig, AutoGPTQConfig, INCConfig))
        )

    def split_ba(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ba.chunk(2, dim=-1)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            # ba_proj is replicated for Marlin; slice b/a to local TP rank.
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        q_size = self.head_k_dim
        k_size = self.head_k_dim
        v_size = self.num_v_heads // self.num_k_heads * self.head_v_dim
        z_size = v_size
        ba_size = self.num_v_heads // self.num_k_heads

        key_start = q_size
        value_start = key_start + k_size
        z_start = value_start + v_size
        a_start = ba_size

        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size
        group_width_qkvz = q_size + k_size + v_size + z_size
        group_width_ba = 2 * ba_size

        if envs.VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH:
            q_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_qkvz, 0, q_size, mixed_qkvz.device
            )
            k_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_qkvz, key_start, k_size, mixed_qkvz.device
            )
            v_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_qkvz, value_start, v_size, mixed_qkvz.device
            )
            z_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_qkvz, z_start, z_size, mixed_qkvz.device
            )
            b_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_ba, 0, ba_size, mixed_ba.device
            )
            a_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_ba, a_start, ba_size, mixed_ba.device
            )

            query = mixed_qkvz.index_select(-1, q_idx).view(
                *base_shape_qkvz, ng, q_size
            )
            key = mixed_qkvz.index_select(-1, k_idx).view(
                *base_shape_qkvz, ng, k_size
            )
            value = mixed_qkvz.index_select(-1, v_idx).view(
                *base_shape_qkvz,
                self.num_v_heads // self.tp_size,
                self.head_v_dim,
            )
            z = mixed_qkvz.index_select(-1, z_idx).view(
                *base_shape_qkvz,
                self.num_v_heads // self.tp_size,
                self.head_v_dim,
            )
            b = mixed_ba.index_select(-1, b_idx).view(
                *base_shape_ba, self.num_v_heads // self.tp_size
            )
            a = mixed_ba.index_select(-1, a_idx).view(
                *base_shape_ba, self.num_v_heads // self.tp_size
            )
            return query, key, value, z, b, a

        new_tensor_shape_qkvz = base_shape_qkvz + (ng, group_width_qkvz)
        new_tensor_shape_ba = base_shape_ba + (ng, group_width_ba)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        query = mixed_qkvz[:, :, :q_size]
        key = mixed_qkvz[:, :, key_start : key_start + k_size]
        value = mixed_qkvz[:, :, value_start : value_start + v_size]
        z = mixed_qkvz[:, :, z_start : z_start + z_size]
        b = mixed_ba[:, :, :ba_size]
        a = mixed_ba[:, :, a_start : a_start + ba_size]

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """
        Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv = mixed_qkvz[..., :qkv_size]
            z_flat = _sm70_compile_graph_slice_dim(
                mixed_qkvz, -1, qkv_size, z_size
            )
            n = mixed_qkvz.shape[0]
            z_out = z_flat.reshape(n, -1, self.head_v_dim)
            ba_size = mixed_ba.shape[-1] // 2
            b = mixed_ba[..., :ba_size]
            a = _sm70_compile_graph_slice_dim(mixed_ba, -1, ba_size, ba_size)
            if self.disable_tp_for_ba_proj and self.tp_size > 1:
                ba_chunk = self.num_v_heads // self.tp_size
                ba_start = self.tp_rank * ba_chunk
                b = b[:, ba_start : ba_start + ba_chunk]
                a = a[:, ba_start : ba_start + ba_chunk]
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        q_size = self.head_k_dim
        k_size = self.head_k_dim
        v_size = self.num_v_heads // self.num_k_heads * self.head_v_dim
        z_size = v_size
        ba_size = self.num_v_heads // self.num_k_heads
        group_width_qkvz = q_size + k_size + v_size + z_size
        group_width_ba = 2 * ba_size

        key_start = q_size
        value_start = key_start + k_size
        z_start = value_start + v_size
        a_start = ba_size

        if envs.VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH:
            q_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_qkvz, 0, q_size, mixed_qkvz.device
            )
            k_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_qkvz, key_start, k_size, mixed_qkvz.device
            )
            v_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_qkvz, value_start, v_size, mixed_qkvz.device
            )
            z_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_qkvz, z_start, z_size, mixed_qkvz.device
            )
            b_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_ba, 0, ba_size, mixed_ba.device
            )
            a_idx = _sm70_compile_graph_interleaved_indices(
                ng, group_width_ba, a_start, ba_size, mixed_ba.device
            )

            query = mixed_qkvz.index_select(-1, q_idx).view(
                *base_shape_qkvz, ng, q_size
            )
            key = mixed_qkvz.index_select(-1, k_idx).view(
                *base_shape_qkvz, ng, k_size
            )
            value = mixed_qkvz.index_select(-1, v_idx).view(
                *base_shape_qkvz,
                self.num_v_heads // self.tp_size,
                self.head_v_dim,
            )
            z = mixed_qkvz.index_select(-1, z_idx).view(
                *base_shape_qkvz,
                self.num_v_heads // self.tp_size,
                self.head_v_dim,
            )
            b = mixed_ba.index_select(-1, b_idx).view(
                *base_shape_ba, self.num_v_heads // self.tp_size
            )
            a = mixed_ba.index_select(-1, a_idx).view(
                *base_shape_ba, self.num_v_heads // self.tp_size
            )
        else:
            mixed_qkvz = mixed_qkvz.view(*base_shape_qkvz, ng, group_width_qkvz)
            mixed_ba = mixed_ba.view(*base_shape_ba, ng, group_width_ba)
            query = mixed_qkvz[..., :q_size]
            key = mixed_qkvz[..., key_start : key_start + k_size]
            value = mixed_qkvz[..., value_start : value_start + v_size]
            z = mixed_qkvz[..., z_start : z_start + z_size]
            b = mixed_ba[..., :ba_size]
            a = mixed_ba[..., a_start : a_start + ba_size]

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def _can_use_flashqla_decode(
        self,
        mixed_qkv: torch.Tensor,
        state_indices: torch.Tensor | None,
        num_decode_tokens: int,
        *,
        layer_name: LayerNameType | None = None,
        stage: str = "decode",
    ) -> bool:
        if layer_name is None:
            layer_name = _encode_layer_name(self.prefix)
        reason = "ok"
        if not self.enable_flashqla_decode:
            reason = "env_disabled"
            _sm70_log_flashqla_decode_route(
                layer_name=layer_name,
                stage=stage,
                decision="skip",
                reason=reason,
                mixed_qkv=mixed_qkv,
                state_indices=state_indices,
                num_decode_tokens=num_decode_tokens,
            )
            return False
        if num_decode_tokens <= 0 or state_indices is None:
            reason = "no_decode_tokens_or_state_indices"
            _sm70_log_flashqla_decode_route(
                layer_name=layer_name,
                stage=stage,
                decision="skip",
                reason=reason,
                mixed_qkv=mixed_qkv,
                state_indices=state_indices,
                num_decode_tokens=num_decode_tokens,
            )
            return False
        if state_indices.dtype != torch.int32:
            reason = "state_indices_dtype"
            _sm70_log_flashqla_decode_route(
                layer_name=layer_name,
                stage=stage,
                decision="skip",
                reason=reason,
                mixed_qkv=mixed_qkv,
                state_indices=state_indices,
                num_decode_tokens=num_decode_tokens,
            )
            return False
        if (
            not mixed_qkv.is_cuda
            or mixed_qkv.dtype != torch.float16
            or not mixed_qkv.is_contiguous()
        ):
            reason = "mixed_qkv_contract"
            _sm70_log_flashqla_decode_route(
                layer_name=layer_name,
                stage=stage,
                decision="skip",
                reason=reason,
                mixed_qkv=mixed_qkv,
                state_indices=state_indices,
                num_decode_tokens=num_decode_tokens,
            )
            return False
        if self.head_k_dim != 128 or self.head_v_dim != 128:
            reason = "head_dim"
            _sm70_log_flashqla_decode_route(
                layer_name=layer_name,
                stage=stage,
                decision="skip",
                reason=reason,
                mixed_qkv=mixed_qkv,
                state_indices=state_indices,
                num_decode_tokens=num_decode_tokens,
            )
            return False
        if self.num_k_heads % self.tp_size != 0 or self.num_v_heads % self.tp_size != 0:
            reason = "head_tp_divisibility"
            _sm70_log_flashqla_decode_route(
                layer_name=layer_name,
                stage=stage,
                decision="skip",
                reason=reason,
                mixed_qkv=mixed_qkv,
                state_indices=state_indices,
                num_decode_tokens=num_decode_tokens,
            )
            return False
        capability = torch.cuda.get_device_capability(mixed_qkv.device)
        if capability[0] != 7 or capability[1] not in (0, 5):
            reason = "device_capability"
            _sm70_log_flashqla_decode_route(
                layer_name=layer_name,
                stage=stage,
                decision="skip",
                reason=reason,
                mixed_qkv=mixed_qkv,
                state_indices=state_indices,
                num_decode_tokens=num_decode_tokens,
            )
            return False
        available = _flashqla_sm70_decode_available()
        if not available:
            reason = "flashqla_decode_import"
        _sm70_log_flashqla_decode_route(
            layer_name=layer_name,
            stage=stage,
            decision="take" if available else "skip",
            reason=reason,
            mixed_qkv=mixed_qkv,
            state_indices=state_indices,
            num_decode_tokens=num_decode_tokens,
        )
        return available

    def _forward_core_decode_flashqla(
        self,
        *,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
        num_decode_tokens: int,
        cu_seqlens: torch.Tensor | None = None,
        core_attn_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cu_seqlens
        from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import (
            gdn_decode_mixed_qkv_global_state_sm70,
        )

        state_indices = state_indices[:num_decode_tokens].contiguous()
        if core_attn_out is None:
            core_attn_out = mixed_qkv.new_empty(
                (
                    num_decode_tokens,
                    self.num_v_heads // self.tp_size,
                    self.head_v_dim,
                )
            )
        out = core_attn_out[:num_decode_tokens]
        kernel_out = (
            out
            if out.is_contiguous()
            else torch.empty(out.shape, dtype=out.dtype, device=out.device)
        )
        gdn_decode_mixed_qkv_global_state_sm70(
            mixed_qkv=mixed_qkv[:num_decode_tokens],
            a=a[:num_decode_tokens].contiguous(),
            b=b[:num_decode_tokens].contiguous(),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state=ssm_state,
            state_indices=state_indices,
            output=kernel_out,
            scale=self.head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
        )
        if kernel_out is not out:
            out.copy_(kernel_out)
        return out.unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        if self.maybe_sm70_qwen_gdn_full_forward:
            layer_name = _encode_layer_name(self.prefix)
            if _sm70_qwen_gdn_full_forward_enabled(
                layer_name,
                force_enabled=self.force_sm70_qwen_gdn_full_forward,
                disabled=self.disable_sm70_qwen_gdn_full_forward,
                auto_enabled=self.auto_sm70_qwen_gdn_full_forward,
            ):
                conv_state_cache, ssm_state_cache = _resolve_qwen_gdn_kv_cache_args(
                    layer_name,
                    output,
                )
                torch.ops.vllm.qwen_gdn_full_forward(
                    hidden_states,
                    output,
                    conv_state_cache,
                    ssm_state_cache,
                    layer_name,
                )
                return
        self._forward_method(hidden_states, output)

    def _full_forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        self._forward_method(hidden_states, output)

    def _compute_output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        layer_name = _encode_layer_name(self.prefix)
        total_start = _sm70_gdn_prefill_profile_start()
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = _sm70_dump_gdn_projection_tensor(
            "proj_core_in", layer_name, core_attn_out
        )
        z = _sm70_dump_gdn_projection_tensor("proj_z", layer_name, z)
        profile_start = _sm70_gdn_prefill_profile_start()
        core_attn_out = self.norm(core_attn_out, z)
        _sm70_gdn_prefill_profile_end(
            layer_name,
            "projection_norm",
            profile_start,
            tokens=num_tokens,
        )
        core_attn_out = _sm70_dump_gdn_projection_tensor(
            "proj_norm_out", layer_name, core_attn_out
        )
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        profile_start = _sm70_gdn_prefill_profile_start()
        proj_out, _ = self.out_proj(core_attn_out)
        _sm70_gdn_prefill_profile_end(
            layer_name,
            "projection_out_proj",
            profile_start,
            tokens=num_tokens,
        )
        proj_out = _sm70_dump_gdn_projection_tensor(
            "proj_out", layer_name, proj_out
        )
        _sm70_gdn_prefill_profile_end(
            layer_name,
            "projection_total",
            total_start,
            tokens=num_tokens,
        )
        return proj_out

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        proj_out = self._compute_output_projection(core_attn_out, z, num_tokens)
        output[:num_tokens] = proj_out
        return proj_out

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path."""
        if GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            layer_name = _encode_layer_name(self.prefix)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )
            conv_state_cache, ssm_state_cache = _resolve_qwen_gdn_kv_cache_args(
                layer_name,
                core_attn_out,
            )
            (
                non_spec_query_start_loc,
                non_spec_state_indices_tensor,
            ) = _qwen_gdn_non_spec_metadata_tensors(
                layer_name,
                core_attn_out.device,
            )

            torch.ops.vllm.qwen_gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                conv_state_cache,
                ssm_state_cache,
                non_spec_query_start_loc,
                non_spec_state_indices_tensor,
                fast_kernel=True,
                layer_name=layer_name,
            )

            self._output_projection(core_attn_out, z, output, num_tokens)
        else:
            self.forward_cuda(hidden_states, output)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        layer_name = _encode_layer_name(self.prefix)
        if _sm70_qwen_gdn_input_core_boundary_enabled():
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            core_attn_out = torch.zeros_like(z)
            conv_state_cache, ssm_state_cache = _resolve_qwen_gdn_kv_cache_args(
                layer_name,
                core_attn_out,
            )
            torch.ops.vllm.qwen_gdn_input_projection_core(
                hidden_states,
                z,
                core_attn_out,
                conv_state_cache,
                ssm_state_cache,
                layer_name,
            )
            if envs.VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP:
                torch.ops.vllm.qwen_gdn_output_projection(
                    core_attn_out,
                    z,
                    output,
                    num_tokens,
                    layer_name,
                )
            else:
                self._output_projection(core_attn_out, z, output, num_tokens)
            return

        if envs.VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP:
            mixed_qkv = torch.empty(
                (num_tokens, (self.key_dim * 2 + self.value_dim) // self.tp_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            b = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            a = torch.empty_like(b)
            torch.ops.vllm.qwen_gdn_input_projection(
                hidden_states,
                mixed_qkv,
                z,
                b,
                a,
                layer_name,
            )
            core_attn_out = torch.zeros_like(z)
            conv_state_cache, ssm_state_cache = _resolve_qwen_gdn_kv_cache_args(
                layer_name,
                core_attn_out,
            )
            core_attn_out = _qwen_gdn_run_recurrent_core(
                self,
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                layer_name=layer_name,
                conv_state_cache=conv_state_cache,
                ssm_state_cache=ssm_state_cache,
            )
            if envs.VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP:
                torch.ops.vllm.qwen_gdn_output_projection(
                    core_attn_out,
                    z,
                    output,
                    num_tokens,
                    layer_name,
                )
            else:
                self._output_projection(core_attn_out, z, output, num_tokens)
            return

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        _sm70_gdn_graph_buffer_copy(
            "gdn_hidden_states",
            layer_name,
            hidden_states,
            "proj",
        )
        hidden_states = _sm70_dump_gdn_projection_tensor(
            "gdn_hidden_states",
            layer_name,
            hidden_states,
        )
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        mixed_qkvz = _sm70_dump_gdn_projection_tensor(
            "in_proj_qkvz", layer_name, mixed_qkvz
        )
        ba = _sm70_dump_gdn_projection_tensor("in_proj_ba", layer_name, ba)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv = mixed_qkvz[..., :qkv_size]
            mixed_qkv = _sm70_dump_gdn_projection_tensor(
                "split_mixed_qkv", layer_name, mixed_qkv
            )
            if envs.VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS:
                mixed_qkv = mixed_qkv.contiguous()
            z = _sm70_compile_graph_slice_dim(mixed_qkvz, -1, qkv_size, z_size)
            z = _sm70_dump_gdn_projection_tensor("split_z", layer_name, z)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            ba_size = ba.shape[-1] // 2
            b = ba[..., :ba_size]
            a = _sm70_compile_graph_slice_dim(ba, -1, ba_size, ba_size)
            if self.disable_tp_for_ba_proj and self.tp_size > 1:
                ba_chunk = self.num_v_heads // self.tp_size
                ba_start = self.tp_rank * ba_chunk
                b = b[:, ba_start : ba_start + ba_chunk]
                a = a[:, ba_start : ba_start + ba_chunk]
            b = b.contiguous()
            a = a.contiguous()

        if (
            envs.VLLM_SM70_GDN_Z_CONTIGUOUS
            and current_platform.is_device_capability(70)
        ):
            z = z.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        conv_state_cache, ssm_state_cache = _resolve_qwen_gdn_kv_cache_args(
            layer_name,
            core_attn_out,
        )
        core_attn_out = _qwen_gdn_run_recurrent_core(
            self,
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            layer_name=layer_name,
            conv_state_cache=conv_state_cache,
            ssm_state_cache=ssm_state_cache,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        if envs.VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP:
            torch.ops.vllm.qwen_gdn_output_projection(
                core_attn_out,
                z,
                output,
                num_tokens,
                layer_name,
            )
        else:
            self._output_projection(core_attn_out, z, output, num_tokens)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _warmup_sm70_causal_conv1d_real_state(self) -> bool:
        """Warm causal-conv prefill/decode variants with real KV-cache strides."""
        if not current_platform.is_device_capability(70):
            return False
        if not hasattr(self, "kv_cache"):
            return False

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self_kv_cache = self.kv_cache
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        if conv_state.size(0) == 0:
            return False

        T = FLA_CHUNK_SIZE
        device = conv_state.device
        dtype = conv_state.dtype
        dummy_conv_in = torch.randn(
            T, conv_weights.shape[0], device=device, dtype=dtype
        ).transpose(0, 1)
        query_start_loc = torch.tensor([0, T], device=device, dtype=torch.int32)
        query_start_loc_cpu = torch.tensor([0, T], dtype=torch.int32)
        nums_dict, batch_ptr, token_chunk_offset_ptr = (
            compute_causal_conv1d_metadata(query_start_loc_cpu, device=device)
        )
        conv_metadata = SimpleNamespace(
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        cache_indices = torch.zeros(1, device=device, dtype=torch.int32)
        has_initial_state = torch.ones(1, device=device, dtype=torch.bool)
        conv_state_line = conv_state[:1]
        try:
            causal_conv1d_fn(
                dummy_conv_in,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state_line,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                query_start_loc=query_start_loc,
                metadata=conv_metadata,
            )
        except Exception:
            logger.warning(
                "SM70 GDN causal-conv real-state warmup failed for layer %s. "
                "First inference may JIT _causal_conv1d_fwd_kernel.",
                self.prefix,
                exc_info=True,
            )
            return False
        finally:
            conv_state_line.zero_()

        dummy_decode_inputs = [
            torch.randn(1, conv_weights.shape[0], device=device, dtype=dtype)
        ]
        qkvz_width = conv_weights.shape[0] + self.value_dim // self.tp_size
        if qkvz_width > conv_weights.shape[0]:
            # The real no-MTP decode path slices qkv out of qkvz, so the
            # logical [1, qkv_dim] tensor has row stride qkvz_width.
            dummy_qkvz = torch.randn(1, qkvz_width, device=device, dtype=dtype)
            dummy_decode_inputs.append(dummy_qkvz[:, : conv_weights.shape[0]])
        try:
            for dummy_decode_in in dummy_decode_inputs:
                # The update Triton kernel specializes on num_cache_lines and
                # input strides, so warm both full KV cache and projection-slice
                # stride instead of only a compact [1, qkv_dim] tensor.
                causal_conv1d_update(
                    dummy_decode_in,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=cache_indices,
                    validate_data=True,
                )
        except Exception:
            logger.warning(
                "SM70 GDN causal-conv update warmup failed for layer %s. "
                "First inference may JIT _causal_conv1d_update_kernel.",
                self.prefix,
                exc_info=True,
            )
            return False
        finally:
            conv_state_line.zero_()
        return True

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` and the decode recurrent kernel with
        small dummy tensors to force compilation/autotuning while GPU memory
        is still plentiful.  The autotuner results are cached globally, so
        only the first layer incurs actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses fixed-parameter kernels, but they are still
        JIT-compiled on first use. Warm them here so the first real decode
        step does not trip the JIT monitor.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()
        warmup_key = (
            qkv_or_qkvz.device.type,
            qkv_or_qkvz.device.index,
            dtype,
            state_dtype,
            self.gdn_prefill_backend,
            _sm70_flashqla_original_prefill_enabled(),
            _sm70_flashqla_direct_output_enabled(),
            _sm70_flashqla_indexed_prefill_enabled(),
            num_k_heads,
            num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            qkv_or_qkvz.shape[-1] - v_dim,
            FLA_CHUNK_SIZE,
            is_conv_state_dim_first(),
        )
        if warmup_key in _SM70_GDN_PREFILL_WARMUP_KEYS:
            if _sm70_profile_trace_enabled():
                logger.info(
                    "SM70 profile trace: GDN prefill/decode warmup skip "
                    "layer=%s reason=shape_already_warmed",
                    self.prefix,
                )
            return
        _SM70_GDN_PREFILL_WARMUP_KEYS.add(warmup_key)

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Also run the
        # conv1d prefill kernel once; otherwise Qwen3.5/Next can still JIT
        # _causal_conv1d_fwd_kernel on the first real request after the JIT
        # monitor has been activated.
        T = FLA_CHUNK_SIZE
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        dummy_conv_in = torch.randn(
            T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
        ).transpose(0, 1)
        if is_conv_state_dim_first():
            dummy_conv_state = torch.zeros(
                1,
                conv_weights.shape[0],
                conv_weights.shape[1] - 1,
                device=device,
                dtype=dtype,
            )
        else:
            dummy_conv_state = torch.zeros(
                1,
                conv_weights.shape[1] - 1,
                conv_weights.shape[0],
                device=device,
                dtype=dtype,
            ).transpose(-1, -2)
        dummy_cache_indices = torch.zeros(1, device=device, dtype=torch.int32)
        dummy_has_initial_state = torch.ones(1, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
        try:
            causal_conv1d_fn(
                dummy_conv_in,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=dummy_conv_state,
                cache_indices=dummy_cache_indices,
                has_initial_state=dummy_has_initial_state,
                query_start_loc=cu_seqlens,
            )
        except Exception:
            logger.warning(
                "GDN causal-conv prefill warmup (T=%d) failed for layer %s. "
                "First inference may JIT _causal_conv1d_fwd_kernel.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN causal-conv prefill warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )

        # Mirror the real prefill path here: build q/k/v/g/beta via
        # fused_post_conv_prep and then run chunk_gated_delta_rule with
        # in-kernel L2 norm disabled.
        dummy_mixed_qkv = torch.randn(
            T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        del q, k, v, g, beta
        # The warmup is only meant to compile/autotune the GDN prefill
        # kernels. Keep the actual kernel inputs deterministic and bounded so
        # random profile tensors cannot trigger pathological TileLang runtime
        # waits before real inference starts.
        q = torch.full(
            (1, T, num_k_heads, self.head_k_dim),
            1.0e-3,
            device=device,
            dtype=dtype,
        )
        k = torch.full(
            (1, T, num_k_heads, self.head_k_dim),
            1.0e-3,
            device=device,
            dtype=dtype,
        )
        v = torch.full(
            (1, T, num_v_heads, self.head_v_dim),
            1.0e-3,
            device=device,
            dtype=dtype,
        )
        g = torch.zeros(
            1,
            T,
            num_v_heads,
            device=device,
            dtype=torch.float32,
        )
        beta = torch.full(
            (1, T, num_v_heads),
            1.0e-2,
            device=device,
            dtype=torch.float32,
        )
        state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        dummy_prefill_core_attn_out = None
        if (
            self.gdn_prefill_backend == "flashqla_sm70"
            and _sm70_flashqla_original_prefill_enabled()
            and _sm70_flashqla_direct_output_enabled()
        ):
            dummy_prefill_core_attn_out = torch.empty(
                T,
                num_v_heads,
                self.head_v_dim,
                device=device,
                dtype=dtype,
            )
        # CuteDSL kernels require metadata
        chunk_indices = None
        chunk_offsets = None
        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, T)

        trace_prefill_warmup = (
            _sm70_profile_trace_enabled()
            and self.gdn_prefill_backend == "flashqla_sm70"
            and _sm70_flashqla_original_prefill_enabled()
        )
        prefill_warmup_start = time.perf_counter()
        if trace_prefill_warmup:
            logger.info(
                "SM70 profile trace: GDN prefill warmup enter layer=%s "
                "T=%d q_shape=%s v_shape=%s direct_output=%s",
                self.prefix,
                T,
                tuple(q.shape),
                tuple(v.shape),
                dummy_prefill_core_attn_out is not None,
            )
        try:
            self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                core_attn_out=dummy_prefill_core_attn_out,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            if trace_prefill_warmup:
                torch.cuda.synchronize()
                logger.info(
                    "SM70 profile trace: GDN prefill warmup exit layer=%s "
                    "T=%d elapsed_ms=%.3f",
                    self.prefix,
                    T,
                    (time.perf_counter() - prefill_warmup_start) * 1000.0,
                )
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
            if (
                self.gdn_prefill_backend == "flashqla_sm70"
                and _sm70_flashqla_original_prefill_enabled()
                and _sm70_flashqla_indexed_prefill_enabled()
            ):
                indexed_state = torch.zeros(
                    1,
                    num_v_heads,
                    self.head_v_dim,
                    self.head_k_dim,
                    device=device,
                    dtype=state_dtype,
                )
                indexed_out = torch.empty(
                    T,
                    num_v_heads,
                    self.head_v_dim,
                    device=device,
                    dtype=dtype,
                )
                indexed_state_indices = torch.zeros(
                    1, device=device, dtype=torch.long
                )
                indexed_has_initial_state = torch.ones(
                    1, device=device, dtype=torch.bool
                )
                try:
                    self.chunk_gated_delta_rule(
                        q=q,
                        k=k,
                        v=v,
                        g=g,
                        beta=beta,
                        initial_state=indexed_state,
                        output_final_state=False,
                        cu_seqlens=cu_seqlens,
                        chunk_indices=chunk_indices,
                        chunk_offsets=chunk_offsets,
                        state_indices=indexed_state_indices,
                        has_initial_state=indexed_has_initial_state,
                        inplace_final_state=True,
                        use_qk_l2norm_in_kernel=False,
                        core_attn_out=(
                            indexed_out
                            if _sm70_flashqla_direct_output_enabled()
                            else None
                        ),
                    )
                except Exception:
                    logger.warning(
                        "GDN indexed/direct FlashQLA prefill warmup (T=%d) "
                        "failed for layer %s. First real prefill may JIT the "
                        "indexed original TileLang variant.",
                        T,
                        self.prefix,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "GDN indexed/direct FlashQLA prefill warmup (T=%d) "
                        "completed for layer %s",
                        T,
                        self.prefix,
                    )
                finally:
                    del (
                        indexed_state,
                        indexed_out,
                        indexed_state_indices,
                        indexed_has_initial_state,
                    )
        finally:
            del (
                dummy_mixed_qkv,
                dummy_conv_in,
                dummy_conv_state,
                dummy_cache_indices,
                dummy_has_initial_state,
                q,
                k,
                v,
                dummy_a,
                dummy_b,
                g,
                beta,
                state,
                dummy_prefill_core_attn_out,
                cu_seqlens,
                chunk_indices,
                chunk_offsets,
            )

        decode_tokens = 1
        dummy_decode_mixed_qkv = torch.zeros(
            decode_tokens,
            qkv_or_qkvz.shape[-1] - v_dim,
            device=device,
            dtype=dtype,
        )
        dummy_decode_a = torch.zeros(
            decode_tokens, num_v_heads, device=device, dtype=dtype
        )
        dummy_decode_b = torch.zeros(
            decode_tokens, num_v_heads, device=device, dtype=dtype
        )
        dummy_decode_state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        dummy_decode_out = torch.empty(
            decode_tokens,
            1,
            num_v_heads,
            self.head_v_dim,
            device=device,
            dtype=dtype,
        )
        dummy_decode_indices = torch.zeros(
            decode_tokens, device=device, dtype=torch.int32
        )
        dummy_decode_query_start_loc = torch.arange(
            decode_tokens + 1, device=device, dtype=torch.int32
        )
        trace_decode_warmup = _sm70_profile_trace_enabled()
        mixed_decode_warmup_start = time.perf_counter()
        if trace_decode_warmup:
            logger.info(
                "SM70 profile trace: GDN mixed-QKV decode warmup enter "
                "layer=%s tokens=%d mixed_shape=%s",
                self.prefix,
                decode_tokens,
                tuple(dummy_decode_mixed_qkv.shape),
            )
        try:
            fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
                A_log=self.A_log,
                a=dummy_decode_a,
                b=dummy_decode_b,
                dt_bias=self.dt_bias,
                mixed_qkv=dummy_decode_mixed_qkv,
                num_q_heads=self.num_k_heads // self.tp_size,
                num_v_heads=num_v_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                scale=self.head_k_dim**-0.5,
                initial_state=dummy_decode_state,
                out=dummy_decode_out,
                cu_seqlens=dummy_decode_query_start_loc,
                ssm_state_indices=dummy_decode_indices,
                use_qk_l2norm_in_kernel=True,
            )
        except Exception:
            logger.warning(
                "GDN mixed-QKV decode warmup failed for layer %s. "
                "First inference may JIT "
                "fused_sigmoid_gating_delta_rule_update_kernel.",
                self.prefix,
                exc_info=True,
            )
        else:
            if trace_decode_warmup:
                torch.cuda.synchronize()
                logger.info(
                    "SM70 profile trace: GDN mixed-QKV decode warmup exit "
                    "layer=%s elapsed_ms=%.3f",
                    self.prefix,
                    (time.perf_counter() - mixed_decode_warmup_start) * 1000.0,
                )
            logger.debug(
                "GDN mixed-QKV decode warmup completed for layer %s",
                self.prefix,
            )
        if self._can_use_flashqla_decode(
            dummy_decode_mixed_qkv,
            dummy_decode_indices,
            decode_tokens,
            layer_name=_encode_layer_name(self.prefix),
            stage="warmup",
        ):
            if not _sm70_flashqla_decode_warmup_enabled():
                if trace_decode_warmup:
                    logger.info(
                        "SM70 profile trace: GDN FlashQLA decode warmup skip "
                        "layer=%s reason=disabled",
                        self.prefix,
                    )
            else:
                flashqla_decode_warmup_start = time.perf_counter()
                if trace_decode_warmup:
                    logger.info(
                        "SM70 profile trace: GDN FlashQLA decode warmup enter "
                        "layer=%s tokens=%d",
                        self.prefix,
                        decode_tokens,
                    )
                try:
                    self._forward_core_decode_flashqla(
                        mixed_qkv=dummy_decode_mixed_qkv,
                        a=dummy_decode_a,
                        b=dummy_decode_b,
                        ssm_state=dummy_decode_state,
                        state_indices=dummy_decode_indices,
                        num_decode_tokens=decode_tokens,
                        cu_seqlens=dummy_decode_query_start_loc,
                        core_attn_out=dummy_decode_out.squeeze(1),
                    )
                except Exception:
                    logger.warning(
                        "GDN FlashQLA decode warmup failed for layer %s. "
                        "First inference may JIT flash_qla_sm70_gdn.",
                        self.prefix,
                        exc_info=True,
                    )
                else:
                    if trace_decode_warmup:
                        torch.cuda.synchronize()
                        logger.info(
                            "SM70 profile trace: GDN FlashQLA decode warmup "
                            "exit layer=%s elapsed_ms=%.3f",
                            self.prefix,
                            (
                                time.perf_counter()
                                - flashqla_decode_warmup_start
                            )
                            * 1000.0,
                        )
                    logger.debug(
                        "GDN FlashQLA decode warmup completed for layer %s",
                        self.prefix,
                    )
        del (
            dummy_decode_mixed_qkv,
            dummy_decode_a,
            dummy_decode_b,
            dummy_decode_state,
            dummy_decode_out,
            dummy_decode_indices,
            dummy_decode_query_start_loc,
        )

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill) interleaved-GQA layouts,
        dispatches directly to ``_forward_core_decode_fast``. Otherwise unpacks
        the packed layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # The AITER fused reshape/conv kernel expects Qwen3-Next's interleaved
        # GQA layout. Qwen3.5 uses a non-interleaved q/k/v/z layout and must use
        # the generic path below to split/rearrange inputs correctly.
        if (
            self.gqa_interleaved_layout
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_fast(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
                kv_cache=kv_cache,
            )

        core_attn_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            kv_cache=kv_cache,
        )

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        mixed_qkv_decode_requested = (
            self.enable_sm70_fused_sigmoid_mixed_qkv
            and mixed_qkv.is_cuda
            and mixed_qkv.dtype == torch.float16
            and mixed_qkv.is_contiguous()
            and self.num_k_heads % self.tp_size == 0
            and self.num_v_heads % self.tp_size == 0
        )
        if (
            self.enable_packed_recurrent_decode
            and not mixed_qkv_decode_requested
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
                kv_cache=kv_cache,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            kv_cache[0]
            if is_conv_state_dim_first()
            else kv_cache[0].transpose(-1, -2)
        )
        ssm_state = kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_non_spec_tokens = num_actual_tokens
        if (
            spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            num_non_spec_tokens = attn_metadata.num_decodes
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        layer_name = _encode_layer_name(self.prefix)

        mixed_qkv = mixed_qkv[:num_non_spec_tokens]
        b = b[:num_non_spec_tokens]
        a = a[:num_non_spec_tokens]
        _sm70_gdn_graph_buffer_copy_state_slice(
            "pre_conv_state",
            layer_name,
            conv_state,
            non_spec_state_indices_tensor,
            num_non_spec_tokens,
        )
        _sm70_gdn_graph_buffer_copy_state_slice(
            "pre_ssm_state",
            layer_name,
            ssm_state,
            non_spec_state_indices_tensor,
            num_non_spec_tokens,
        )

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        if mixed_qkv_non_spec is not None:
            _sm70_gdn_graph_buffer_copy(
                "pre_conv_input_qkv",
                layer_name,
                mixed_qkv_non_spec,
                "core",
            )

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            _sm70_gdn_graph_buffer_copy(
                "prefill_conv_out",
                layer_name,
                mixed_qkv_non_spec,
                "core",
            )
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    :num_non_spec_tokens
                ],
                validate_data=True,
            )
            _sm70_gdn_graph_buffer_copy(
                "conv_out",
                layer_name,
                mixed_qkv_non_spec,
                "core",
            )
            _sm70_gdn_graph_buffer_copy_state_slice(
                "post_conv_state",
                layer_name,
                conv_state,
                non_spec_state_indices_tensor,
                num_non_spec_tokens,
            )
        else:
            mixed_qkv_non_spec = None

        if (
            mixed_qkv_decode_requested
            and spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
            and mixed_qkv_non_spec is not None
            and mixed_qkv_non_spec.is_cuda
            and mixed_qkv_non_spec.dtype == torch.float16
            and mixed_qkv_non_spec.is_contiguous()
        ):
            if self._can_use_flashqla_decode(
                mixed_qkv_non_spec,
                non_spec_state_indices_tensor,
                num_non_spec_tokens,
                layer_name=layer_name,
                stage="standard_mixed_qkv",
            ):
                _log_runtime_route_once("SM70 FlashQLA GDN decode route enabled.")
                assert non_spec_state_indices_tensor is not None
                qla_out = self._forward_core_decode_flashqla(
                    mixed_qkv=mixed_qkv_non_spec,
                    a=a,
                    b=b,
                    ssm_state=ssm_state,
                    state_indices=non_spec_state_indices_tensor,
                    num_decode_tokens=num_non_spec_tokens,
                    cu_seqlens=non_spec_query_start_loc,
                    core_attn_out=core_attn_out,
                )
                _sm70_gdn_graph_buffer_copy(
                    "recurrent_out",
                    layer_name,
                    qla_out,
                    "core",
                )
                _sm70_gdn_graph_buffer_copy_state_slice(
                    "post_ssm_state",
                    layer_name,
                    ssm_state,
                    non_spec_state_indices_tensor,
                    num_non_spec_tokens,
                )
                return
            _log_runtime_route_once(
                "SM70 fused sigmoid GDN mixed-QKV decode route enabled."
            )
            compare_mixed_qkv = (
                self.compare_sm70_fused_sigmoid_mixed_qkv
                and not torch.cuda.is_current_stream_capturing()
            )
            core_attn_out_non_spec, mixed_final_state = (
                fused_sigmoid_gating_delta_rule_update_mixed_qkv(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    mixed_qkv=mixed_qkv_non_spec,
                    num_q_heads=self.num_k_heads // self.tp_size,
                    num_v_heads=self.num_v_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    initial_state=ssm_state,
                    inplace_final_state=not compare_mixed_qkv,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor[
                        :num_non_spec_tokens
                    ],
                    use_qk_l2norm_in_kernel=True,
                )
            )
            if compare_mixed_qkv:
                query_ref, key_ref, value_ref = self.rearrange_mixed_qkv(
                    mixed_qkv_non_spec
                )
                ref_out, ref_final_state = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_ref,
                    k=key_ref,
                    v=value_ref,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor[
                        :num_non_spec_tokens
                    ],
                    use_qk_l2norm_in_kernel=True,
                )
                out_max_diff, out_mean_diff, out_num_different = _sm70_diff_stats(
                    core_attn_out_non_spec,
                    ref_out,
                )
                state_stats = _sm70_mixed_qkv_state_diff_stats(
                    mixed_final_state,
                    ref_final_state,
                    non_spec_state_indices_tensor,
                    non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    core_attn_out_non_spec.shape[1],
                )
                state_failed = (
                    bool(state_stats["shape_mismatch"])
                    or state_stats["invalid_tokens"] != 0
                    or state_stats["num_different"] != 0
                )
                if out_num_different != 0 or state_failed:
                    logger.warning(
                        "SM70 fused sigmoid mixed-QKV compare mismatch: "
                        "layer=%s out_max_diff=%s out_mean_diff=%s "
                        "out_num_different=%s state_max_diff=%s "
                        "state_mean_diff=%s state_num_different=%s "
                        "state_valid_tokens=%s state_invalid_tokens=%s "
                        "state_shape_mismatch=%s",
                        getattr(self, "prefix", None),
                        out_max_diff,
                        out_mean_diff,
                        out_num_different,
                        state_stats["max_diff"],
                        state_stats["mean_diff"],
                        state_stats["num_different"],
                        state_stats["valid_tokens"],
                        state_stats["invalid_tokens"],
                        state_stats["shape_mismatch"],
                    )
                else:
                    _log_runtime_route_once(
                        "SM70 fused sigmoid mixed-QKV compare matched exactly: "
                        "out_max_diff=0.0 state_max_diff=0.0."
                    )
                core_attn_out_non_spec = ref_out
            core_attn_out[:num_non_spec_tokens] = core_attn_out_non_spec.squeeze(0)
            return

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        prefill_l2norm_in_kernel = False
        use_original_flashqla_prefill = _sm70_flashqla_original_prefill_enabled()
        prefill_gate_is_exp = False
        if attn_metadata.num_prefills > 0:
            profile_start = _sm70_gdn_prefill_profile_start()
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if self.enable_sm70_legacy_prefill_prep:
                query_non_spec, key_non_spec, value_non_spec = (
                    self.rearrange_mixed_qkv(mixed_qkv_non_spec)
                )
                g_non_spec, beta_non_spec = fused_gdn_gating(
                    self.A_log, a_non_spec, b_non_spec, self.dt_bias
                )
                prefill_l2norm_in_kernel = True
            else:
                prefill_gate_is_exp = (
                    self.gdn_prefill_backend == "flashqla_sm70"
                    and not use_original_flashqla_prefill
                )
                (
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                ) = fused_post_conv_prep(
                    conv_output=mixed_qkv_non_spec,
                    a=a_non_spec,
                    b=b_non_spec,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    apply_l2norm=True,
                    output_g_exp=prefill_gate_is_exp,
                )
                query_non_spec = query_non_spec.unsqueeze(0)
                key_non_spec = key_non_spec.unsqueeze(0)
                value_non_spec = value_non_spec.unsqueeze(0)
                g_non_spec = g_non_spec.unsqueeze(0)
                beta_non_spec = beta_non_spec.unsqueeze(0)
            _sm70_gdn_prefill_profile_end(
                layer_name,
                "post_conv_prep",
                profile_start,
                tokens=num_non_spec_tokens,
                details=(
                    f"backend={self.gdn_prefill_backend} "
                    f"q_contig={query_non_spec.is_contiguous()} "
                    f"k_contig={key_non_spec.is_contiguous()} "
                    f"v_contig={value_non_spec.is_contiguous()} "
                    f"g_contig={g_non_spec.is_contiguous()} "
                    f"beta_contig={beta_non_spec.is_contiguous()} "
                    f"gate_is_exp={prefill_gate_is_exp}"
                ),
            )
            _sm70_gdn_graph_buffer_copy(
                "prefill_q",
                layer_name,
                query_non_spec,
                "core",
            )
            _sm70_gdn_graph_buffer_copy(
                "prefill_k",
                layer_name,
                key_non_spec,
                "core",
            )
            _sm70_gdn_graph_buffer_copy(
                "prefill_v",
                layer_name,
                value_non_spec,
                "core",
            )
            _sm70_gdn_graph_buffer_copy(
                "prefill_g",
                layer_name,
                g_non_spec,
                "core",
            )
            _sm70_gdn_graph_buffer_copy(
                "prefill_beta",
                layer_name,
                beta_non_spec,
                "core",
            )
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                a_spec = a
                b_spec = b
            else:
                assert spec_token_indx is not None
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
            g_spec, beta_spec = fused_gdn_gating(
                self.A_log, a_spec, b_spec, self.dt_bias
            )
            core_attn_out_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            use_indexed_original_prefill = (
                self.gdn_prefill_backend == "flashqla_sm70"
                and use_original_flashqla_prefill
                and _sm70_flashqla_indexed_prefill_enabled()
                and non_spec_state_indices_tensor.ndim == 1
            )
            profile_start = _sm70_gdn_prefill_profile_start()
            if use_indexed_original_prefill:
                initial_state = ssm_state
            else:
                initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]
                initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]
            _sm70_gdn_prefill_profile_end(
                layer_name,
                "state_gather",
                profile_start,
                tokens=num_non_spec_tokens,
                details=(
                    f"state_shape={tuple(initial_state.shape)} "
                    f"indices_contig={non_spec_state_indices_tensor.is_contiguous()} "
                    f"indexed={use_indexed_original_prefill}"
                ),
            )
            if use_indexed_original_prefill:
                _sm70_gdn_graph_buffer_copy_state_slice(
                    "prefill_initial_state",
                    layer_name,
                    ssm_state,
                    non_spec_state_indices_tensor,
                    int(has_initial_state.shape[0]),
                )
            else:
                _sm70_gdn_graph_buffer_copy(
                    "prefill_initial_state",
                    layer_name,
                    initial_state,
                    "state",
                )
            prefill_core_attn_out = None
            if (
                _sm70_flashqla_direct_output_enabled()
                and spec_sequence_masks is None
                and core_attn_out.shape[0] >= num_non_spec_tokens
            ):
                prefill_core_attn_out = core_attn_out

            profile_start = _sm70_gdn_prefill_profile_start()
            chunk_kwargs = {
                "q": query_non_spec,
                "k": key_non_spec,
                "v": value_non_spec,
                "g": g_non_spec,
                "beta": beta_non_spec,
                "initial_state": initial_state,
                "output_final_state": not use_indexed_original_prefill,
                "cu_seqlens": non_spec_query_start_loc,
                "chunk_indices": attn_metadata.chunk_indices,
                "chunk_offsets": attn_metadata.chunk_offsets,
                "use_qk_l2norm_in_kernel": prefill_l2norm_in_kernel,
                "gate_is_exp": prefill_gate_is_exp,
                "core_attn_out": prefill_core_attn_out,
            }
            if self.gdn_prefill_backend == "flashqla_sm70":
                chunk_kwargs.update(
                    {
                        "state_indices": (
                            non_spec_state_indices_tensor
                            if use_indexed_original_prefill
                            else None
                        ),
                        "has_initial_state": (
                            has_initial_state
                            if use_indexed_original_prefill
                            else None
                        ),
                        "inplace_final_state": use_indexed_original_prefill,
                    }
                )
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(**chunk_kwargs)
            _sm70_gdn_prefill_profile_end(
                layer_name,
                "core_call",
                profile_start,
                tokens=num_non_spec_tokens,
                details=f"backend={self.gdn_prefill_backend}",
            )
            _sm70_gdn_graph_buffer_copy(
                "prefill_core_out",
                layer_name,
                core_attn_out_non_spec,
                "core",
            )
            if last_recurrent_state is not None:
                _sm70_gdn_graph_buffer_copy(
                    "prefill_last_recurrent_state",
                    layer_name,
                    last_recurrent_state,
                    "state",
                )
            # Init cache
            profile_start = _sm70_gdn_prefill_profile_start()
            if not use_indexed_original_prefill:
                assert last_recurrent_state is not None
                ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                    ssm_state.dtype
                )
            _sm70_gdn_prefill_profile_end(
                layer_name,
                "state_writeback",
                profile_start,
                tokens=num_non_spec_tokens,
                details=(
                    f"state_dtype={ssm_state.dtype} "
                    f"indexed={use_indexed_original_prefill}"
                ),
            )
            _sm70_gdn_graph_buffer_copy_state_slice(
                "prefill_post_ssm_state",
                layer_name,
                ssm_state,
                non_spec_state_indices_tensor,
                int(has_initial_state.shape[0]),
            )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor[
                        :num_non_spec_tokens
                    ],
                    use_qk_l2norm_in_kernel=True,
                )
            )
            _sm70_gdn_graph_buffer_copy(
                "recurrent_out",
                layer_name,
                core_attn_out_non_spec,
                "core",
            )
            _sm70_gdn_graph_buffer_copy_state_slice(
                "post_ssm_state",
                layer_name,
                ssm_state,
                non_spec_state_indices_tensor,
                num_non_spec_tokens,
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        elif core_attn_out_non_spec is not None and not (
            attn_metadata.num_prefills > 0
            and spec_sequence_masks is None
            and core_attn_out_non_spec.squeeze(0).data_ptr()
            == core_attn_out[:num_non_spec_tokens].data_ptr()
        ):
            core_attn_out[:num_non_spec_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_fast(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            kv_cache[0]
            if is_conv_state_dim_first()
            else kv_cache[0].transpose(-1, -2)
        )
        ssm_state = kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        assert attn_metadata.non_spec_query_start_loc is not None
        assert attn_metadata.non_spec_state_indices_tensor is not None
        return self._forward_core_decode_non_spec_explicit(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            non_spec_query_start_loc=attn_metadata.non_spec_query_start_loc,
            non_spec_state_indices_tensor=(
                attn_metadata.non_spec_state_indices_tensor
            ),
            num_decode_tokens=attn_metadata.num_decodes,
            kv_cache=kv_cache,
            layer_name=_encode_layer_name(self.prefix),
        )

    def _forward_core_decode_non_spec_explicit(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        non_spec_query_start_loc: torch.Tensor,
        non_spec_state_indices_tensor: torch.Tensor,
        num_decode_tokens: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer_name: LayerNameType,
    ):
        """
        Packed non-spec decode with explicit CUDA graph tensor dependencies.
        """
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            kv_cache[0]
            if is_conv_state_dim_first()
            else kv_cache[0].transpose(-1, -2)
        )
        ssm_state = kv_cache[1]

        mixed_qkv = mixed_qkv[:num_decode_tokens]
        b = b[:num_decode_tokens]
        a = a[:num_decode_tokens]
        state_indices = non_spec_state_indices_tensor[:num_decode_tokens]
        if (
            self.enable_flashqla_decode
            and mixed_qkv.is_cuda
            and mixed_qkv.dtype == torch.float16
            and not mixed_qkv.is_contiguous()
        ):
            mixed_qkv = mixed_qkv.contiguous()
        _sm70_gdn_graph_buffer_copy("state_indices", layer_name, state_indices, "state")
        _sm70_gdn_graph_buffer_copy_state_slice(
            "pre_conv_state",
            layer_name,
            conv_state,
            non_spec_state_indices_tensor,
            num_decode_tokens,
        )
        _sm70_gdn_graph_buffer_copy_state_slice(
            "pre_ssm_state",
            layer_name,
            ssm_state,
            non_spec_state_indices_tensor,
            num_decode_tokens,
        )

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices,  # type: ignore[arg-type]
            validate_data=False,
        )
        _sm70_gdn_graph_buffer_copy(
            "conv_out",
            layer_name,
            mixed_qkv_non_spec,
            "core",
        )
        _sm70_gdn_graph_buffer_copy_state_slice(
            "post_conv_state",
            layer_name,
            conv_state,
            non_spec_state_indices_tensor,
            num_decode_tokens,
        )
        compare_request = _sm70_gdn_packed_compare_request(layer_name)
        compare_tensors = None
        if compare_request is not None and non_spec_state_indices_tensor is not None:
            state_indices = non_spec_state_indices_tensor[:num_decode_tokens]
            state_indices_cpu = state_indices.detach().cpu()
            valid_positions_cpu = torch.nonzero(
                state_indices_cpu >= 0,
                as_tuple=False,
            ).flatten()
            if valid_positions_cpu.numel() > 0:
                valid_positions = valid_positions_cpu.to(
                    device=mixed_qkv_non_spec.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                selected_state_indices = state_indices.index_select(
                    0,
                    valid_positions,
                )
                selected_states = ssm_state.index_select(
                    0,
                    selected_state_indices,
                )
                local_state = selected_states.new_zeros(
                    (selected_states.shape[0] + 1, *selected_states.shape[1:])
                )
                local_state[1:].copy_(selected_states)
                local_indices = torch.arange(
                    1,
                    selected_states.shape[0] + 1,
                    device=state_indices.device,
                    dtype=state_indices.dtype,
                )
                compare_tensors = (
                    compare_request,
                    mixed_qkv_non_spec.index_select(0, valid_positions).contiguous(),
                    a.index_select(0, valid_positions).contiguous(),
                    b.index_select(0, valid_positions).contiguous(),
                    local_state,
                    local_indices,
                    selected_state_indices,
                    torch.arange(
                        0,
                        selected_states.shape[0] + 1,
                        device=state_indices.device,
                        dtype=torch.int32,
                    ),
                )
        out_buf = core_attn_out[:num_decode_tokens].unsqueeze(1)
        assert non_spec_query_start_loc is not None
        cu_seqlens = non_spec_query_start_loc[: num_decode_tokens + 1]
        if self._can_use_flashqla_decode(
            mixed_qkv_non_spec,
            state_indices,
            num_decode_tokens,
            layer_name=layer_name,
            stage="packed_explicit",
        ):
            _log_runtime_route_once("SM70 FlashQLA GDN decode route enabled.")
            qla_out = self._forward_core_decode_flashqla(
                mixed_qkv=mixed_qkv_non_spec,
                a=a,
                b=b,
                ssm_state=ssm_state,
                state_indices=state_indices,
                num_decode_tokens=num_decode_tokens,
                cu_seqlens=cu_seqlens,
                core_attn_out=core_attn_out,
            )
            _sm70_gdn_graph_buffer_copy("recurrent_out", layer_name, qla_out, "core")
            _sm70_gdn_graph_buffer_copy_state_slice(
                "post_ssm_state",
                layer_name,
                ssm_state,
                non_spec_state_indices_tensor,
                num_decode_tokens,
            )
            return
        _log_runtime_route_once("SM70 exact mixed-QKV GDN decode route enabled.")
        fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            mixed_qkv=mixed_qkv_non_spec,
            num_q_heads=self.num_k_heads // self.tp_size,
            num_v_heads=self.num_v_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,  # type: ignore[arg-type]
            use_qk_l2norm_in_kernel=True,
        )
        _sm70_gdn_graph_buffer_copy("recurrent_out", layer_name, out_buf, "core")
        _sm70_gdn_graph_buffer_copy_state_slice(
            "post_ssm_state",
            layer_name,
            ssm_state,
            non_spec_state_indices_tensor,
            num_decode_tokens,
        )
        if compare_tensors is not None:
            (
                (report_path, report_step),
                cmp_mixed_qkv,
                cmp_a,
                cmp_b,
                local_state,
                local_indices,
                selected_state_indices,
                local_cu_seqlens,
            ) = compare_tensors
            packed_state = local_state.clone()
            mixed_state = local_state.clone()
            ref_state = local_state.clone()
            packed_out = out_buf.new_empty(
                (
                    cmp_mixed_qkv.shape[0],
                    1,
                    self.num_v_heads // self.tp_size,
                    self.head_v_dim,
                )
            )
            mixed_out = torch.empty_like(packed_out)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=cmp_mixed_qkv,
                a=cmp_a,
                b=cmp_b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=self.head_k_dim**-0.5,
                initial_state=packed_state,
                out=packed_out,
                ssm_state_indices=local_indices,
                use_qk_l2norm_in_kernel=True,
            )
            fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
                A_log=self.A_log,
                a=cmp_a,
                b=cmp_b,
                dt_bias=self.dt_bias,
                mixed_qkv=cmp_mixed_qkv,
                num_q_heads=self.num_k_heads // self.tp_size,
                num_v_heads=self.num_v_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                scale=self.head_k_dim**-0.5,
                initial_state=mixed_state,
                out=mixed_out,
                cu_seqlens=local_cu_seqlens,
                ssm_state_indices=local_indices,
                use_qk_l2norm_in_kernel=True,
            )
            ref_out, ref_state = fused_sigmoid_gating_delta_rule_update_mixed_qkv(
                A_log=self.A_log,
                a=cmp_a,
                b=cmp_b,
                dt_bias=self.dt_bias,
                mixed_qkv=cmp_mixed_qkv,
                num_q_heads=self.num_k_heads // self.tp_size,
                num_v_heads=self.num_v_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                initial_state=ref_state,
                inplace_final_state=True,
                cu_seqlens=local_cu_seqlens,
                ssm_state_indices=local_indices,
                use_qk_l2norm_in_kernel=True,
            )
            _sm70_save_gdn_packed_compare_report(
                report_path,
                _encode_layer_name(self.prefix),
                report_step,
                selected_state_indices,
                packed_out,
                packed_state,
                mixed_out,
                mixed_state,
                ref_out,
                ref_state,
            )
        return


def qwen_gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    fast_kernel: bool,
    layer_name: LayerNameType,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``fast_kernel=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``fast_kernel=True`` (AITER Triton fast path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    _log_runtime_route_once("SM70 Qwen GDN standard-spec recurrent route enabled.")
    attn_metadata = None
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        candidate = attn_metadata_raw.get(layer_name)
        if isinstance(candidate, GDNAttentionMetadata):
            attn_metadata = candidate
    if conv_state_cache.numel() == 0 and ssm_state_cache.numel() == 0:
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None and kv_cache[0].numel() > 0:
            conv_state_cache, ssm_state_cache = kv_cache[0], kv_cache[1]
    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, qkv_or_qkvz)
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b_or_ba)
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a_or_z_out)
    restore_query_start_loc = False
    restore_state_indices = False
    if attn_metadata is not None and non_spec_query_start_loc.numel() > 0:
        old_non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        attn_metadata.non_spec_query_start_loc = non_spec_query_start_loc
        restore_query_start_loc = True
    else:
        old_non_spec_query_start_loc = None
    if attn_metadata is not None and non_spec_state_indices_tensor.numel() > 0:
        old_non_spec_state_indices_tensor = (
            attn_metadata.non_spec_state_indices_tensor
        )
        attn_metadata.non_spec_state_indices_tensor = (
            non_spec_state_indices_tensor
        )
        restore_state_indices = True
    else:
        old_non_spec_state_indices_tensor = None
    try:
        if fast_kernel:
            self._forward_core_rocm(
                qkvz=qkv_or_qkvz,
                ba=b_or_ba,
                z_out=a_or_z_out,
                core_attn_out=core_attn_out,
                kv_cache=(conv_state_cache, ssm_state_cache),
            )
        else:
            self._forward_core(
                mixed_qkv=qkv_or_qkvz,
                b=b_or_ba,
                a=a_or_z_out,
                core_attn_out=core_attn_out,
                kv_cache=(conv_state_cache, ssm_state_cache),
            )
    finally:
        if restore_query_start_loc and attn_metadata is not None:
            attn_metadata.non_spec_query_start_loc = old_non_spec_query_start_loc
        if restore_state_indices and attn_metadata is not None:
            attn_metadata.non_spec_state_indices_tensor = (
                old_non_spec_state_indices_tensor
            )
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out)


def qwen_gdn_attention_core_standard_spec(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_token_indx: torch.Tensor,
    non_spec_token_indx: torch.Tensor,
    spec_sequence_masks: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata = None
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        candidate = attn_metadata_raw.get(layer_name)
        if isinstance(candidate, GDNAttentionMetadata):
            attn_metadata = candidate
    if conv_state_cache.numel() == 0 and ssm_state_cache.numel() == 0:
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None and kv_cache[0].numel() > 0:
            conv_state_cache, ssm_state_cache = kv_cache[0], kv_cache[1]
    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, mixed_qkv)
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b)
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a)
    _sm70_dump_gdn_spec_metadata_graph_buffers(
        layer_name,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=spec_query_start_loc,
        spec_state_indices_tensor=spec_state_indices_tensor,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        spec_sequence_masks=spec_sequence_masks,
        num_accepted_tokens=num_accepted_tokens,
    )
    mixed_qkv_decode_requested = (
        self.enable_sm70_fused_sigmoid_mixed_qkv
        and mixed_qkv.is_cuda
        and mixed_qkv.dtype == torch.float16
        and mixed_qkv.is_contiguous()
        and self.num_k_heads % self.tp_size == 0
        and self.num_v_heads % self.tp_size == 0
    )
    if (
        attn_metadata is not None
        and self.enable_packed_recurrent_decode
        and not mixed_qkv_decode_requested
        and attn_metadata.spec_sequence_masks is None
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes > 0
        and non_spec_query_start_loc.numel() > 0
        and non_spec_state_indices_tensor.numel() > 0
    ):
        self._forward_core_decode_non_spec_explicit(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            non_spec_query_start_loc=non_spec_query_start_loc,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            num_decode_tokens=attn_metadata.num_decodes,
            kv_cache=(conv_state_cache, ssm_state_cache),
            layer_name=layer_name,
        )
        _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out)
        return

    restore_fields: dict[str, object] = {}

    def _patch_metadata(name: str, tensor: torch.Tensor) -> None:
        if attn_metadata is not None and tensor.numel() > 0:
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
    try:
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            kv_cache=(conv_state_cache, ssm_state_cache),
        )
    finally:
        if attn_metadata is not None:
            for name, value in restore_fields.items():
                setattr(attn_metadata, name, value)
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out)


def qwen_gdn_attention_core_spec_commit(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_token_indx: torch.Tensor,
    non_spec_token_indx: torch.Tensor,
    spec_sequence_masks: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Experimental active-MTP Qwen GDN core with explicit spec metadata.

    This keeps projections in the compiled graph and passes every spec
    metadata tensor as an operator argument. End-to-end long-output quality is
    still gated by the full-Qwen-GDN active-MTP boundary until this narrower
    recurrent-core boundary also proves the accepted-state commit contract.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata = None
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        candidate = attn_metadata_raw.get(layer_name)
        if isinstance(candidate, GDNAttentionMetadata):
            attn_metadata = candidate
    if attn_metadata is None or attn_metadata.num_spec_decodes <= 0:
        raise RuntimeError(
            "qwen_gdn_attention_core_spec_commit requires active GDN "
            "spec-decode metadata"
        )
    if (
        spec_query_start_loc.numel() == 0
        or spec_state_indices_tensor.numel() == 0
        or spec_token_indx.numel() == 0
        or spec_sequence_masks.numel() == 0
        or num_accepted_tokens.numel() == 0
    ):
        raise RuntimeError(
            "qwen_gdn_attention_core_spec_commit missing graph-visible "
            "spec metadata tensors"
        )
    if spec_state_indices_tensor.shape[0] < attn_metadata.num_spec_decodes:
        raise RuntimeError(
            "qwen_gdn_attention_core_spec_commit spec state rows do not "
            "cover active spec decodes"
        )
    if num_accepted_tokens.numel() < attn_metadata.num_spec_decodes:
        raise RuntimeError(
            "qwen_gdn_attention_core_spec_commit accepted-token rows do not "
            "cover active spec decodes"
        )
    if conv_state_cache.numel() == 0 and ssm_state_cache.numel() == 0:
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None and kv_cache[0].numel() > 0:
            conv_state_cache, ssm_state_cache = kv_cache[0], kv_cache[1]

    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, mixed_qkv)
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b)
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a)
    _sm70_dump_gdn_spec_metadata_graph_buffers(
        layer_name,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=spec_query_start_loc,
        spec_state_indices_tensor=spec_state_indices_tensor,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        spec_sequence_masks=spec_sequence_masks,
        num_accepted_tokens=num_accepted_tokens,
    )

    if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
        conv_state = (
            conv_state_cache
            if is_conv_state_dim_first()
            else conv_state_cache.transpose(-1, -2)
        )
        ssm_state = ssm_state_cache
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )
        mixed_qkv_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=spec_state_indices_tensor[:, 0],
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=spec_query_start_loc,
            max_query_len=spec_state_indices_tensor.size(-1),
            validate_data=False,
        )
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        g_spec, beta_spec = fused_gdn_gating(
            self.A_log,
            a,
            b,
            self.dt_bias,
        )
        core_attn_out_spec, _ = fused_recurrent_gated_delta_rule(
            q=query_spec,
            k=key_spec,
            v=value_spec,
            g=g_spec,
            beta=beta_spec,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=spec_query_start_loc,
            ssm_state_indices=spec_state_indices_tensor,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out[: mixed_qkv.shape[0]] = core_attn_out_spec.squeeze(0)
        _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out)
        return core_attn_out

    restore_fields: dict[str, object] = {}

    def _patch_metadata(name: str, value: torch.Tensor) -> None:
        restore_fields[name] = getattr(attn_metadata, name)
        setattr(attn_metadata, name, value)

    def _patch_metadata_scalar(name: str, value: int) -> None:
        restore_fields[name] = getattr(attn_metadata, name)
        setattr(attn_metadata, name, value)

    _patch_metadata("non_spec_query_start_loc", non_spec_query_start_loc)
    _patch_metadata("non_spec_state_indices_tensor", non_spec_state_indices_tensor)
    _patch_metadata("spec_query_start_loc", spec_query_start_loc)
    _patch_metadata("spec_state_indices_tensor", spec_state_indices_tensor)
    _patch_metadata("spec_token_indx", spec_token_indx)
    _patch_metadata("non_spec_token_indx", non_spec_token_indx)
    _patch_metadata("spec_sequence_masks", spec_sequence_masks)
    _patch_metadata("num_accepted_tokens", num_accepted_tokens)
    if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
        # The tensor rows are the graph-visible contract for FULL graph replay.
        # They include PAD_SLOT_ID rows with zero-length query ranges, so the
        # conv/recurrent kernels can skip padded rows without relying on the
        # capture-time Python num_spec_decodes scalar.
        _patch_metadata_scalar(
            "num_spec_decodes",
            int(spec_state_indices_tensor.shape[0]),
        )
    try:
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            kv_cache=(conv_state_cache, ssm_state_cache),
        )
    finally:
        for name, value in restore_fields.items():
            setattr(attn_metadata, name, value)
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out)
    return core_attn_out


def qwen_gdn_attention_core_standard(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata = None
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        candidate = attn_metadata_raw.get(layer_name)
        if isinstance(candidate, GDNAttentionMetadata):
            attn_metadata = candidate
    if conv_state_cache.numel() == 0 and ssm_state_cache.numel() == 0:
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None and kv_cache[0].numel() > 0:
            conv_state_cache, ssm_state_cache = kv_cache[0], kv_cache[1]
    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, mixed_qkv)
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b)
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a)
    mixed_qkv_decode_requested = (
        self.enable_sm70_fused_sigmoid_mixed_qkv
        and mixed_qkv.is_cuda
        and mixed_qkv.dtype == torch.float16
        and mixed_qkv.is_contiguous()
        and self.num_k_heads % self.tp_size == 0
        and self.num_v_heads % self.tp_size == 0
    )
    if (
        attn_metadata is not None
        and self.enable_packed_recurrent_decode
        and not mixed_qkv_decode_requested
        and attn_metadata.spec_sequence_masks is None
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes > 0
        and non_spec_query_start_loc.numel() > 0
        and non_spec_state_indices_tensor.numel() > 0
    ):
        self._forward_core_decode_non_spec_explicit(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            non_spec_query_start_loc=non_spec_query_start_loc,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            num_decode_tokens=attn_metadata.num_decodes,
            kv_cache=(conv_state_cache, ssm_state_cache),
            layer_name=layer_name,
        )
        _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out)
        return

    restore_query_start_loc = False
    restore_state_indices = False
    if attn_metadata is not None and non_spec_query_start_loc.numel() > 0:
        old_non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        attn_metadata.non_spec_query_start_loc = non_spec_query_start_loc
        restore_query_start_loc = True
    else:
        old_non_spec_query_start_loc = None
    if attn_metadata is not None and non_spec_state_indices_tensor.numel() > 0:
        old_non_spec_state_indices_tensor = (
            attn_metadata.non_spec_state_indices_tensor
        )
        attn_metadata.non_spec_state_indices_tensor = (
            non_spec_state_indices_tensor
        )
        restore_state_indices = True
    else:
        old_non_spec_state_indices_tensor = None
    try:
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            kv_cache=(conv_state_cache, ssm_state_cache),
        )
    finally:
        if restore_query_start_loc and attn_metadata is not None:
            attn_metadata.non_spec_query_start_loc = old_non_spec_query_start_loc
        if restore_state_indices and attn_metadata is not None:
            attn_metadata.non_spec_state_indices_tensor = (
                old_non_spec_state_indices_tensor
            )
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out)


def qwen_gdn_attention_core_context(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """0.0.3-style Qwen GDN core boundary for SM70 MTP diagnostics.

    The latest path passes cache tensors and non-spec metadata as explicit
    custom-op arguments. 0.0.3 resolved both through the forward context/layer
    object. Keep this narrow A/B path so recurrent-state semantics can be
    compared without changing the underlying GDN kernels.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    conv_state_cache, ssm_state_cache = _resolve_qwen_gdn_kv_cache_args(
        layer_name,
        core_attn_out,
    )
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        kv_cache=(conv_state_cache, ssm_state_cache),
    )


def qwen_gdn_attention_core_003_spec(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """0.0.3-style active-MTP Qwen GDN recurrent-core boundary.

    Metadata is resolved through the forward context like 0.0.3, while the
    state caches and core output remain explicit arguments so latest
    compile/FULL graph has graph-safe mutation dependencies. Returning
    ``core_attn_out`` gives the compiled output projection a direct data
    dependency on the opaque recurrent core instead of relying only on a
    None-returning mutation side effect.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if conv_state_cache.numel() == 0 and ssm_state_cache.numel() == 0:
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None and kv_cache[0].numel() > 0:
            conv_state_cache, ssm_state_cache = kv_cache[0], kv_cache[1]
    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, mixed_qkv)
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b)
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a)
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        kv_cache=(conv_state_cache, ssm_state_cache),
    )
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out)
    return core_attn_out


def qwen_gdn_attention_core_context_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""


def qwen_gdn_attention_core_003_spec_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Fake implementation for torch.compile."""
    return core_attn_out


def qwen_gdn_attention_core_spec_commit_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_token_indx: torch.Tensor,
    non_spec_token_indx: torch.Tensor,
    spec_sequence_masks: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Fake implementation for torch.compile."""
    return core_attn_out


def gdn_attention_core_fake(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    fast_kernel: bool,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""


def gdn_attention_core_standard_spec_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_token_indx: torch.Tensor,
    non_spec_token_indx: torch.Tensor,
    spec_sequence_masks: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""


def gdn_attention_core_standard_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""


def qwen_gdn_full_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the full Qwen GDN attention forward outside Inductor.

    The compile/FULL graph path still captures the launched kernels, but the
    projections around the recurrent GDN core keep the strict eager execution
    order instead of being rewritten by Inductor.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    # The eager body updates the same cache tensors through the layer object.
    # Keep them as explicit mutated custom-op args so Inductor/CUDA graph
    # dependency tracking can see the recurrent GDN state side effects.
    _ = conv_state_cache, ssm_state_cache
    _log_runtime_route_once("SM70 Qwen GDN full-forward route enabled.")
    self._full_forward(hidden_states, output)


def qwen_gdn_full_forward_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""


def qwen_gdn_output_projection(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    output: torch.Tensor,
    num_tokens: int,
    layer_name: LayerNameType,
) -> None:
    """Run only the Qwen GDN RMSNorm/out-projection segment outside Inductor."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._output_projection(core_attn_out, z, output, num_tokens)


def qwen_gdn_output_projection_fake(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    output: torch.Tensor,
    num_tokens: int,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""


def qwen_gdn_input_projection_core(
    hidden_states: torch.Tensor,
    z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run Qwen GDN input projection and recurrent core outside Inductor."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    hidden_states = _sm70_dump_gdn_projection_tensor(
        "gdn_hidden_states_input_core",
        layer_name,
        hidden_states,
    )
    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    ba, _ = self.in_proj_ba(hidden_states)
    mixed_qkvz = _sm70_dump_gdn_projection_tensor(
        "input_core_in_proj_qkvz",
        layer_name,
        mixed_qkvz,
    )
    ba = _sm70_dump_gdn_projection_tensor("input_core_in_proj_ba", layer_name, ba)

    if self.gqa_interleaved_layout:
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            mixed_qkvz,
            ba,
        )
        query, key, value = map(
            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)
    else:
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv = mixed_qkvz[..., :qkv_size]
        if envs.VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS:
            mixed_qkv = mixed_qkv.contiguous()
        z = _sm70_compile_graph_slice_dim(mixed_qkvz, -1, qkv_size, z_size)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba_size = ba.shape[-1] // 2
        b = ba[..., :ba_size]
        a = _sm70_compile_graph_slice_dim(ba, -1, ba_size, ba_size)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        b = b.contiguous()
        a = a.contiguous()

    if envs.VLLM_SM70_GDN_Z_CONTIGUOUS and current_platform.is_device_capability(70):
        z = z.contiguous()
    z_out.copy_(z)

    _qwen_gdn_run_recurrent_core(
        self,
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        layer_name=layer_name,
        conv_state_cache=conv_state_cache,
        ssm_state_cache=ssm_state_cache,
    )


def qwen_gdn_input_projection_core_fake(
    hidden_states: torch.Tensor,
    z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""


def qwen_gdn_input_projection(
    hidden_states: torch.Tensor,
    mixed_qkv_out: torch.Tensor,
    z_out: torch.Tensor,
    b_out: torch.Tensor,
    a_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run only the Qwen GDN input projection/splitting outside Inductor."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    hidden_states = _sm70_dump_gdn_projection_tensor(
        "gdn_hidden_states_input_projection",
        layer_name,
        hidden_states,
    )
    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    ba, _ = self.in_proj_ba(hidden_states)

    if self.gqa_interleaved_layout:
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            mixed_qkvz,
            ba,
        )
        query, key, value = map(
            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)
    else:
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv = mixed_qkvz[..., :qkv_size]
        z = _sm70_compile_graph_slice_dim(mixed_qkvz, -1, qkv_size, z_size)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba_size = ba.shape[-1] // 2
        b = ba[..., :ba_size]
        a = _sm70_compile_graph_slice_dim(ba, -1, ba_size, ba_size)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]

    if envs.VLLM_SM70_GDN_Z_CONTIGUOUS and current_platform.is_device_capability(70):
        z = z.contiguous()
    mixed_qkv_out.copy_(mixed_qkv)
    z_out.copy_(z)
    b_out.copy_(b)
    a_out.copy_(a)


def qwen_gdn_input_projection_fake(
    hidden_states: torch.Tensor,
    mixed_qkv_out: torch.Tensor,
    z_out: torch.Tensor,
    b_out: torch.Tensor,
    a_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""


direct_register_custom_op(
    op_name="qwen_gdn_full_forward",
    op_func=qwen_gdn_full_forward,
    mutates_args=["output", "conv_state_cache", "ssm_state_cache"],
    fake_impl=qwen_gdn_full_forward_fake,
)


direct_register_custom_op(
    op_name="qwen_gdn_output_projection",
    op_func=qwen_gdn_output_projection,
    mutates_args=["output"],
    fake_impl=qwen_gdn_output_projection_fake,
)


direct_register_custom_op(
    op_name="qwen_gdn_input_projection_core",
    op_func=qwen_gdn_input_projection_core,
    mutates_args=[
        "z_out",
        "core_attn_out",
        "conv_state_cache",
        "ssm_state_cache",
    ],
    fake_impl=qwen_gdn_input_projection_core_fake,
)


direct_register_custom_op(
    op_name="qwen_gdn_input_projection",
    op_func=qwen_gdn_input_projection,
    mutates_args=[
        "mixed_qkv_out",
        "z_out",
        "b_out",
        "a_out",
    ],
    fake_impl=qwen_gdn_input_projection_fake,
)


direct_register_custom_op(
    op_name="qwen_gdn_attention_core",
    op_func=qwen_gdn_attention_core,
    mutates_args=[
        "qkv_or_qkvz",
        "a_or_z_out",
        "core_attn_out",
        "conv_state_cache",
        "ssm_state_cache",
    ],
    fake_impl=gdn_attention_core_fake,
)

direct_register_custom_op(
    op_name="qwen_gdn_attention_core_standard",
    op_func=qwen_gdn_attention_core_standard,
    mutates_args=[
        "mixed_qkv",
        "core_attn_out",
        "conv_state_cache",
        "ssm_state_cache",
    ],
    fake_impl=gdn_attention_core_standard_fake,
)

direct_register_custom_op(
    op_name="qwen_gdn_attention_core_standard_spec",
    op_func=qwen_gdn_attention_core_standard_spec,
    mutates_args=[
        "mixed_qkv",
        "core_attn_out",
        "conv_state_cache",
        "ssm_state_cache",
    ],
    fake_impl=gdn_attention_core_standard_spec_fake,
)

direct_register_custom_op(
    op_name="qwen_gdn_attention_core_spec_commit",
    op_func=qwen_gdn_attention_core_spec_commit,
    mutates_args=[
        "mixed_qkv",
        "core_attn_out",
        "conv_state_cache",
        "ssm_state_cache",
    ],
    fake_impl=qwen_gdn_attention_core_spec_commit_fake,
)

direct_register_custom_op(
    op_name="qwen_gdn_attention_core_context",
    op_func=qwen_gdn_attention_core_context,
    mutates_args=["core_attn_out"],
    fake_impl=qwen_gdn_attention_core_context_fake,
)

direct_register_custom_op(
    op_name="qwen_gdn_attention_core_003_spec",
    op_func=qwen_gdn_attention_core_003_spec,
    mutates_args=[
        "mixed_qkv",
        "core_attn_out",
        "conv_state_cache",
        "ssm_state_cache",
    ],
    fake_impl=qwen_gdn_attention_core_003_spec_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
