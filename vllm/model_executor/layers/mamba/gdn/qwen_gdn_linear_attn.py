# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import os
from types import SimpleNamespace

import torch
from einops import rearrange
from torch import nn

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.config.execution_policy import graph_policy
from vllm.config.gdn import GdnProfileConfig, resolve_gdn_config
from vllm.config.gdn_projection import legacy_projection_value
from vllm.config.gdn_state import resolve_state_trace
from vllm.config.sm70_dflash2 import (
    capture_sm70_dflash2_config,
    sm70_dflash2_enabled,
)
from vllm.config.sm70_runtime import capture_runtime_trace
from vllm.diagnostics import (
    diagnostic_channel,
    legacy_channel,
    legacy_history,
)
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as _fla_chunk_rule
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_mixed_qkv,
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out,
)
from vllm.model_executor.layers.fla.ops import gdn_diagnostics as _gdn_diagnostics
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd as l2norm_fwd
from vllm.model_executor.layers.fla.ops.gdn_chunk_kernels import bind_chunk_kernels
from vllm.model_executor.layers.fla.ops.gdn_prefill import (
    GdnPrefill,
)
from vllm.model_executor.layers.fla.ops.gdn_prefill import (
    fi_chunk_gated_delta_rule as fi_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops.gdn_preparation import (
    GdnPreparation,
)
from vllm.model_executor.layers.fla.ops.gdn_preparation import (
    _sm70_pack_qwen_gdn_qkv as _sm70_pack_qwen_gdn_qkv,
)
from vllm.model_executor.layers.fla.ops.gdn_preparation import (
    _sm70_pack_qwen_gdn_qkv_kernel as _sm70_pack_qwen_gdn_qkv_kernel,
)
from vllm.model_executor.layers.fla.ops.gdn_preparation import (
    fused_gdn_gating as fused_gdn_gating,
)
from vllm.model_executor.layers.fla.ops.gdn_preparation import (
    fused_gdn_gating_kernel as fused_gdn_gating_kernel,
)
from vllm.model_executor.layers.fla.ops.gdn_profiling import (
    GdnPrefillProfiler,
    bind_gdn_profiler,
)
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    _get_gdn_head_k_dim as _get_gdn_head_k_dim,
)
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    _is_libs_cu13_install_intact as _is_libs_cu13_install_intact,
)
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    _log_gdn_backend_decision as _log_gdn_backend_decision,
)
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    _resolve_gdn_prefill_backend as _resolve_gdn_prefill_backend,
)
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    select_gdn_execution,
)
from vllm.model_executor.layers.fla.ops.gdn_stages import (
    GdnHeadContract,
    convolve_decode,
    mixed_qkv_recurrence,
)
from vllm.model_executor.layers.fla.ops.sm70 import (
    gdn_prefill as flashqla_prefill_provider,
)
from vllm.model_executor.layers.fla.ops.sm70.gdn_decode import (
    FlashQlaDecodeAdmission,
    flashqla_decode,
)
from vllm.model_executor.layers.fla.ops.sm70.gdn_verify import bind_native_verifier
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
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
    causal_conv1d_update_ddtree,
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
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    gdn_spec_metadata_tensors,
    get_registered_gdn_spec_metadata_tensors,
)
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
from vllm.v1.attention.ops.gdn_state import GdnMetadataOverride
from vllm.v1.worker.gpu.spec_decode import uses_dflash_selector_engine

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

fla_chunk_gated_delta_rule = _fla_chunk_rule  # Legacy import path.

# Retain historical imports while binding the provider once.
_flashqla_sm70_decode_available = (
    flashqla_prefill_provider._flashqla_sm70_decode_available
)
_flashqla_prefill = flashqla_prefill_provider.flashqla_sm70_chunk_gated_delta_rule
flashqla_sm70_chunk_gated_delta_rule_vllm_decode = (
    flashqla_prefill_provider.flashqla_sm70_chunk_gated_delta_rule_vllm_decode
)

logger = init_logger(__name__)

_SM70_GDN_DUMP_COUNTS = legacy_channel("gdn_core").counts
_SM70_GDN_PROJ_DUMP_COUNTS = legacy_channel("gdn_projection").counts
_SM70_GDN_PACKED_COMPARE_COUNTS = legacy_channel("gdn_compare").counts
_SM70_GDN_GRAPH_BUFFERS = legacy_channel("gdn_graph").buffers
_SM70_GDN_GRAPH_META = legacy_channel("gdn_graph").metadata
_SM70_FLASHQLA_DECODE_ROUTE_DEBUG_COUNTS = legacy_history("gdn_route")
_SM70_GDN_PREFILL_PROFILE_COUNTS: dict[str, int] = {}
_DFLASH_DDTREE_PATH_PROBE_REPORTS = 0


def _warmup_sm70_qwen_gdn_causal_conv1d(
    forward_context: dict[str, object],
) -> bool:
    """Warm one bound Qwen GDN layer using its production cache layout.

    MRV2 owns the cache on the modules registered in ``forward_context``.
    Scanning ``model.modules()`` therefore misses the bound cache and leaves
    the non-speculative causal-conv variant to JIT on the first structured
    request.
    """
    from vllm.config.sm70_runtime import capture_runtime_config
    from vllm.model_executor.warmup.sm70_runtime import warmup_bound_convolution

    return warmup_bound_convolution(
        forward_context,
        enabled=bool(
            capture_runtime_config().auxiliary_warmup
            and current_platform.is_device_capability(70)
        ),
    )


def _ddtree_parent_ids_require_branch(
    parent_ids: torch.Tensor | None,
    num_tree_tokens_cpu: torch.Tensor | None,
    num_spec_decodes: int,
) -> bool:
    if parent_ids is None or num_tree_tokens_cpu is None or num_spec_decodes <= 0:
        return False
    lengths = num_tree_tokens_cpu[:num_spec_decodes].detach().cpu().tolist()
    return any(int(num_tree_tokens) > 0 for num_tree_tokens in lengths)


def _dflash_ddtree_fused_gdn_enabled() -> bool:
    raw = os.getenv("VLLM_DFLASH_DDTREE_FUSED_GDN", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _dflash_ddtree_serial_gdn_enabled() -> bool:
    raw = os.getenv("VLLM_DFLASH_DDTREE_SERIAL_GDN", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _dflash_ddtree_linear_gdn_enabled() -> bool:
    raw = os.getenv("VLLM_DFLASH_DDTREE_LINEAR_GDN", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _dflash_ddtree_qla_gdn_enabled() -> bool:
    raw = os.getenv("VLLM_DFLASH_DDTREE_QLA_GDN", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _dflash_ddtree_conv_kernel_enabled() -> bool:
    raw = os.getenv("VLLM_DFLASH_DDTREE_CONV_KERNEL", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _dflash_ddtree_path_probe_enabled() -> bool:
    raw = os.getenv("VLLM_DFLASH_DDTREE_PATH_PROBE", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _dflash_ddtree_path_probe_layer_matches(prefix: str) -> bool:
    layer_filter = os.getenv(
        "VLLM_DFLASH_DDTREE_PATH_PROBE_LAYER",
        "language_model.model.layers.0.linear_attn",
    ).strip()
    return not layer_filter or layer_filter in prefix


def _dflash_ddtree_path_probe_max_reports() -> int:
    raw = os.getenv("VLLM_DFLASH_DDTREE_PATH_PROBE_MAX_REPORTS", "32")
    try:
        return max(0, int(raw))
    except ValueError:
        return 32


def _dflash_ddtree_path_probe_node_limit() -> int:
    raw = os.getenv("VLLM_DFLASH_DDTREE_PATH_PROBE_NODE_LIMIT", "16")
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def _sm70_spec_cache_stride_from_config() -> int:
    try:
        spec_config = get_current_vllm_config().speculative_config
    except Exception:
        return 1
    if spec_config is None:
        return 1
    num_speculative_tokens = getattr(spec_config, "num_speculative_tokens", None)
    if not num_speculative_tokens:
        return 1
    return max(1, int(num_speculative_tokens) + 1)


def _ddtree_queries_have_prefix_row(
    *,
    num_tree_tokens_cpu: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    num_spec_decodes: int,
) -> bool:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return True
    lengths = num_tree_tokens_cpu[:num_spec_decodes].detach().cpu().tolist()
    starts = spec_query_start_loc[: num_spec_decodes + 1].detach().cpu().tolist()
    for req_row, raw_num_tree_tokens in enumerate(lengths):
        num_tree_tokens = int(raw_num_tree_tokens)
        if num_tree_tokens <= 0:
            continue
        query_len = int(starts[req_row + 1]) - int(starts[req_row])
        if query_len != num_tree_tokens + 1:
            return False
    return True


def _ddtree_depth_batches(
    *,
    parent_ids: torch.Tensor,
    num_tree_tokens_cpu: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    num_spec_decodes: int,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int, int, int]]]]:
    """Return DDTree nodes grouped by depth.

    The returned prefix rows are ``(token_row, req_row)``. They correspond to
    the verifier row before the first draft token when query length is
    ``tree_nodes + 1``.

    Each tuple is ``(token_row, req_row, parent_slot, node_slot)``. Slot 0 is
    the prefix-row state when a prefix row exists, otherwise the pre-tree state.
    Non-root node slots are compact DDTree node ids.
    """

    parent_rows = parent_ids[:num_spec_decodes].detach().cpu()
    lengths = num_tree_tokens_cpu[:num_spec_decodes].detach().cpu().tolist()
    starts = spec_query_start_loc[: num_spec_decodes + 1].detach().cpu().tolist()
    prefix_rows: list[tuple[int, int]] = []
    batches_by_depth: list[list[tuple[int, int, int, int]]] = []
    for req_row, raw_num_tree_tokens in enumerate(lengths):
        num_tree_tokens = int(raw_num_tree_tokens)
        if num_tree_tokens <= 0:
            continue
        query_len = int(starts[req_row + 1]) - int(starts[req_row])
        if query_len == num_tree_tokens + 1:
            token_start = int(starts[req_row]) + 1
            prefix_rows.append((token_start - 1, req_row))
        elif query_len == num_tree_tokens:
            token_start = int(starts[req_row])
        else:
            raise RuntimeError(
                "DDTree GDN metadata mismatch: query length must equal the tree "
                "node count, with one additional row when the verifier includes "
                "an explicit root, "
                f"got {query_len} and {num_tree_tokens} for request row "
                f"{req_row}."
            )
        if parent_rows.shape[1] < num_tree_tokens + 1:
            raise RuntimeError(
                "DDTree GDN parent metadata does not cover all tree nodes: "
                f"row width {parent_rows.shape[1]}, nodes {num_tree_tokens}."
            )

        depths = [0] * (num_tree_tokens + 1)
        for node_slot in range(1, num_tree_tokens + 1):
            parent = int(parent_rows[req_row, node_slot].item())
            parent_slot = 0 if parent < 0 else parent
            if parent_slot >= node_slot:
                raise RuntimeError(
                    "DDTree GDN parent metadata must be topologically sorted: "
                    f"request row {req_row}, node slot {node_slot}, "
                    f"parent slot {parent_slot}."
                )
            depth = depths[parent_slot] + 1
            depths[node_slot] = depth
            while len(batches_by_depth) <= depth:
                batches_by_depth.append([])
            batches_by_depth[depth].append(
                (token_start + node_slot - 1, req_row, parent_slot, node_slot)
            )
    return prefix_rows, batches_by_depth


def _log_runtime_route_once(message: str, *args) -> None:
    if torch.compiler.is_compiling():
        return
    logger.info_once(message, *args)


def _sm70_gdn_prefill_profile_enabled():
    return _legacy_gdn_profiler().enabled()


def _sm70_profile_trace_enabled() -> bool:
    return capture_runtime_trace().profile_trace and not torch.compiler.is_compiling()


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


def _sm70_gdn_prefill_profile_start():
    return _legacy_gdn_profiler().start()


def _sm70_gdn_prefill_profile_end(*args, **kwargs):
    return _legacy_gdn_profiler().end(*args, **kwargs)


_sm70_log_flashqla_decode_route = _gdn_diagnostics.log_decode_route


_sm70_dump_gdn_core_tensor = _gdn_diagnostics.dump_core


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
    from vllm.config.diagnostic_dump import parse_int_filter

    selected = parse_int_filter(raw_ranges)
    return None if selected is None else set(selected)


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


def _sm70_qwen_gdn_metadata_has_active_spec(
    attn_metadata: GDNAttentionMetadata | None,
) -> bool:
    return (
        attn_metadata is not None
        and attn_metadata.spec_sequence_masks is not None
        and attn_metadata.num_spec_decodes > 0
    )


def _sm70_assert_standard_core_not_active_spec(
    layer_name: LayerNameType,
    attn_metadata: GDNAttentionMetadata | None,
    *,
    enabled: bool | None = None,
    spec_core: bool | None = None,
) -> None:
    if enabled is None:
        enabled = os.getenv("VLLM_SM70_QWEN_GDN_ASSERT_NO_ACTIVE_SPEC_STANDARD") == "1"
    if not enabled:
        return
    if spec_core is None:
        spec_core = envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
    if not spec_core:
        return
    if not _sm70_qwen_gdn_metadata_has_active_spec(attn_metadata):
        return
    assert attn_metadata is not None
    raise RuntimeError(
        "SM70 Qwen GDN active spec decode reached "
        "qwen_gdn_attention_core_standard while "
        "VLLM_SM70_QWEN_GDN_SPEC_CORE_OP=1 "
        f"(layer={layer_name}, "
        f"num_spec_decodes={attn_metadata.num_spec_decodes}, "
        f"num_spec_decode_tokens={attn_metadata.num_spec_decode_tokens})"
    )


def _sm70_qwen_gdn_block_003_spec_for_deep_native_mtp(
    vllm_config: object, policy=None
) -> bool:
    return (
        _sm70_qwen_gdn_num_speculative_tokens(vllm_config) >= 3
        and _sm70_qwen_gdn_spec_method(vllm_config) == "mtp"
        and (
            legacy_projection_value("spec_core_003")
            if policy is None
            else policy.spec_core_003
        )
        and not (
            legacy_projection_value("spec_allow_deep_mtp")
            if policy is None
            else policy.spec_allow_deep_mtp
        )
    )


def _sm70_qwen_gdn_num_speculative_tokens(vllm_config: object) -> int:
    spec_config = getattr(vllm_config, "speculative_config", None)
    return int(getattr(spec_config, "num_speculative_tokens", 0) or 0)


def _sm70_qwen_gdn_spec_method(vllm_config: object) -> str | None:
    spec_config = getattr(vllm_config, "speculative_config", None)
    return getattr(spec_config, "method", None)


def _is_dflash2_spec_config(vllm_config: object) -> bool:
    return uses_dflash_selector_engine(vllm_config)


def _sm70_current_device_is_volta() -> bool:
    """Inspect the worker's active GPU without creating a CUDA context.

    Initialized workers resolve their selected device through the platform.
    CPU/meta construction uses context-free metadata for the visibility list.
    """
    if not current_platform.is_cuda():
        return False
    return current_platform.is_device_capability((7, 0))


def _sm70_qwen_gdn_full_forward_enabled(
    layer_name: LayerNameType,
    *,
    force_enabled: bool,
    disabled: bool,
    auto_enabled: bool,
) -> bool:
    del layer_name
    if disabled:
        return False
    if force_enabled:
        return True
    # Same compile-time constraint as the split spec-core boundary: this guard
    # is evaluated while the Qwen GDN layer is being compiled, and the first
    # trace commonly comes from a profile/prefill batch without active spec
    # metadata.  If we re-check per-batch active metadata here, the quality
    # guard is captured as disabled and active verifier replays fall through to
    # the split path.  The caller only sets auto_enabled for MTP engines, so
    # no-MTP decode still keeps the lightweight standard path.
    return auto_enabled


def _sm70_qwen_gdn_input_core_boundary_enabled(policy=None) -> bool:
    if policy is not None:
        return bool(policy.input_core and not policy.disable_input_core)
    if legacy_projection_value("disable_input_core"):
        return False
    return legacy_projection_value("input_core")


def _sm70_qwen_gdn_spec_core_enabled(
    layer_name: LayerNameType,
    *,
    auto_enabled: bool,
) -> bool:
    del layer_name
    # This guard is evaluated while the Qwen GDN layer is being compiled.  A
    # per-replay Python active-spec check can be captured from an ordinary
    # decode/profile batch and then reused for the active MTP verifier graph,
    # sending verifier rows through the standard recurrent core.  Keep the
    # compile-time branch stable when the MTP engine explicitly arms this
    # boundary; the custom op itself inspects runtime metadata and falls back
    # to standard semantics for non-active batches.
    return auto_enabled


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
    if self.gdn_policy.projection.context_core:
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
            spec_state_slot_selectors,
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
            spec_state_slot_selectors,
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
            spec_state_slot_selectors,
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
            spec_state_slot_selectors,
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
    torch.Tensor,
]:
    resolved_layer_name = _resolve_layer_name(layer_name)
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return get_registered_gdn_spec_metadata_tensors(resolved_layer_name, device)
    attn_metadata_raw = forward_context.attn_metadata
    if not isinstance(attn_metadata_raw, dict):
        return get_registered_gdn_spec_metadata_tensors(resolved_layer_name, device)
    attn_metadata = attn_metadata_raw.get(resolved_layer_name)
    if not isinstance(attn_metadata, GDNAttentionMetadata):
        return get_registered_gdn_spec_metadata_tensors(resolved_layer_name, device)
    tensors = gdn_spec_metadata_tensors(attn_metadata, device)
    missing_core_spec_tensors = (
        tensors[2].numel() == 0 or tensors[3].numel() == 0 or tensors[7].numel() == 0
    )
    if missing_core_spec_tensors:
        registered_tensors = get_registered_gdn_spec_metadata_tensors(
            resolved_layer_name, device
        )
        if (
            registered_tensors[2].numel() > 0
            and registered_tensors[3].numel() > 0
            and registered_tensors[7].numel() > 0
        ):
            return registered_tensors
    return tensors


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


_sm70_gdn_graph_buffer_copy = _gdn_diagnostics.capture_tensor


_sm70_gdn_graph_buffer_copy_state_slice = _gdn_diagnostics.capture_state_slice


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
    spec_state_slot_selectors: torch.Tensor | None = None,
) -> None:
    if not diagnostic_channel("gdn_graph").policy.metadata:
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
        ("meta_spec_state_slot_selectors", spec_state_slot_selectors),
    ):
        if tensor is None or tensor.numel() == 0:
            continue
        _sm70_gdn_graph_buffer_copy(label, layer_name, tensor, "meta")


def dump_sm70_gdn_graph_buffers(step: int, stage: str) -> None:
    diagnostic_channel("gdn_graph").flush_graph(step, stage, gdn=True)


_sm70_gdn_packed_compare_request = _gdn_diagnostics.compare_request


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


_sm70_gdn_projection_dump_requested = _gdn_diagnostics.projection_requested


_sm70_gdn_projection_dump_impl = _gdn_diagnostics.dump_projection


def _sm70_gdn_projection_dump_fake(
    tensor: torch.Tensor,
    label: str,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return tensor.clone()


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
    if (
        start == 0
        or not graph_policy().compile_graph
        or not use_sm70_decode_graph_semantics()
    ):
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(start, start + size)
        return tensor[tuple(slices)]
    indices = torch.arange(start, start + size, device=tensor.device)
    return tensor.index_select(dim, indices)


def _sm70_gdn_rmsnorm_onepass_enabled(policy=None) -> bool:
    if policy is not None:
        return bool(policy.rmsnorm_onepass)
    return legacy_projection_value("rmsnorm_onepass")


_SM70_GDN_QPN8_BA_REQUIRED_OPS = (
    "fp8_qpn8_gemm_ba_split_sm70_out",
    "fp8_qpn8_dispatch_ba_split_sm70_out",
)


def _missing_sm70_gdn_qpn8_ba_ops() -> list[str]:
    return [
        name
        for name in _SM70_GDN_QPN8_BA_REQUIRED_OPS
        if not hasattr(torch.ops._C, name)
    ]


def _sm70_gdn_qpn8_ba_split_enabled(policy=None) -> bool:
    requested = (
        legacy_projection_value("qpn8_ba_split")
        if policy is None
        else policy.qpn8_ba_split
    )
    if requested and not _sm70_gdn_rmsnorm_onepass_enabled(policy):
        raise RuntimeError(
            "VLLM_SM70_GDN_QPN8_BA_SPLIT=1 requires the accepted "
            "VLLM_SM70_GDN_RMSNORM_ONEPASS=1 pair."
        )
    if requested:
        missing_ops = _missing_sm70_gdn_qpn8_ba_ops()
        if missing_ops:
            raise RuntimeError(
                "VLLM_SM70_GDN_QPN8_BA_SPLIT=1 requires the source-built "
                f"SM70 GDN extension; missing ops: {missing_ops}."
            )
    return bool(requested)


# Register at the historical checkpoint and retain the private import aliases.
from vllm.model_executor.layers.fla.ops.sm70 import (  # noqa: E402
    gdn_norm as _gdn_norm_provider,
)

_sm70_gdn_rmsnorm_gated_onepass_kernel = (
    _gdn_norm_provider._sm70_gdn_rmsnorm_gated_onepass_kernel
)
_sm70_qwen_gdn_rmsnorm_gated_fake = _gdn_norm_provider._sm70_qwen_gdn_rmsnorm_gated_fake
_sm70_qwen_gdn_rmsnorm_gated_impl = _gdn_norm_provider._sm70_qwen_gdn_rmsnorm_gated_impl


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


def flashqla_sm70_chunk_gated_delta_rule(*args, **kwargs):
    kwargs.setdefault(
        "use_original_tilelang", _sm70_flashqla_original_prefill_enabled()
    )
    kwargs.setdefault("profiler", _legacy_gdn_profiler())
    return _flashqla_prefill(*args, **kwargs)


def _legacy_gdn_profiler():
    config = GdnProfileConfig()
    config.resolve()
    profiler = GdnPrefillProfiler(config)
    profiler.counts = _SM70_GDN_PREFILL_PROFILE_COUNTS
    return profiler


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(GdnPrefill, CustomOp):
    def __init__(self) -> None:
        CustomOp.__init__(self)
        config = get_current_vllm_config()
        plan = select_gdn_execution(config)
        GdnPrefill.__init__(
            self,
            plan,
            bind_gdn_profiler(config),
            bind_chunk_kernels(config, resolve_gdn_config(config).schedule)
            if plan.prefill.backend in ("triton", "flashqla_sm70")
            else None,
            native_policy=flashqla_prefill_provider.bind_flashqla_native_policy(
                config,
                resolve_gdn_config(config),
                needed=plan.needs_native_flashqla,
            ),
        )
        # Preserve the CustomOp/nn.Module call boundary and its forward hooks.
        self._call_prefill = self.__call__
        if (
            plan.requested_backend in ("flashinfer", "cutedsl", "flashqla_sm70")
            and plan.prefill.backend != plan.requested_backend
        ):
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                plan.requested_backend,
            )
        _log_gdn_backend_decision(config, plan.requested_backend, plan.prefill.backend)


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
        self._execution_graph_policy = graph_policy(vllm_config)
        self.gdn_policy = resolve_gdn_config(vllm_config)
        self.gdn_state_trace = resolve_state_trace(vllm_config)
        self._gdn_profiler = bind_gdn_profiler(vllm_config)
        from vllm.runtime_resources import runtime_resources_for

        self._prefill_warmup_keys = runtime_resources_for(vllm_config).setdefault(
            "gdn_prefill_warmup_keys", set()
        )
        # Runtime forward/capture need not retain the initialization config.
        # Capture LoRA exclusion while the owning configuration is available.
        self.enable_sm70_gdn_ba_verify = vllm_config.lora_config is None

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gdn_heads = GdnHeadContract(
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.tp_size,
        )
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
            device=self.dt_bias.device,
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
        self._sm70_spec_cache_stride = 1
        if vllm_config.speculative_config is not None:
            self._sm70_spec_cache_stride = max(
                1,
                1 + int(vllm_config.speculative_config.num_speculative_state_tokens()),
            )
        self.verification_update = bind_native_verifier(
            self.gdn_heads,
            enabled=self.gdn_policy.native_verify,
        )
        self.enable_sm70_gdn_rmsnorm_onepass = (
            _sm70_gdn_rmsnorm_onepass_enabled(self.gdn_policy.projection)
            and current_platform.is_device_capability(70)
            and self._sm70_spec_cache_stride == 1
            and self.hidden_size in (2560, 5120)
            and self.tp_size == 4
            and self.num_v_heads == 48
            and self.head_v_dim == 128
            and not self.gqa_interleaved_layout
        )
        self.enable_sm70_gdn_qpn8_ba_split = (
            current_platform.is_device_capability(70)
            and self._sm70_spec_cache_stride == 1
            and self.hidden_size in (2560, 5120)
            and self.tp_size == 4
            and self.num_v_heads == 48
            and self.head_v_dim == 128
            and not self.gqa_interleaved_layout
            and not self.disable_tp_for_ba_proj
            and self.in_proj_ba is not None
            and _sm70_gdn_qpn8_ba_split_enabled(self.gdn_policy.projection)
        )
        self.enable_packed_recurrent_decode = self.gdn_policy.packed_recurrent_decode
        self.enable_sm70_fused_sigmoid_mixed_qkv = self.gdn_policy.mixed_qkv_decode
        verifier_policy = capture_sm70_dflash2_config(vllm_config)
        self.enable_sm70_dflash2_fused_gdn_verify = bool(
            sm70_dflash2_enabled("fused_gdn_verify", verifier_policy)
            and current_platform.is_device_capability(70)
            and _is_dflash2_spec_config(vllm_config)
        )
        self.enable_sm70_dflash2_tp2_gdn_bv2 = bool(
            self.enable_sm70_dflash2_fused_gdn_verify
            and self.tp_size == 2
            and sm70_dflash2_enabled("tp2_gdn_bv2", verifier_policy)
        )
        self.enable_sm70_dflash2_fused_gdn_norm = bool(
            sm70_dflash2_enabled("fused_gdn_norm", verifier_policy)
            and current_platform.is_device_capability(70)
            and _is_dflash2_spec_config(vllm_config)
        )
        self.enable_sm70_dflash2_fused_gdn_split = bool(
            sm70_dflash2_enabled("fused_gdn_split", verifier_policy)
            and current_platform.is_device_capability(70)
            and _is_dflash2_spec_config(vllm_config)
        )
        self.enable_sm70_dflash2_fused_gdn_combined_split = bool(
            sm70_dflash2_enabled("fused_gdn_combined_split", verifier_policy)
            and current_platform.is_device_capability(70)
            and _is_dflash2_spec_config(vllm_config)
        )
        self._sm70_dflash2_combined_split_reason = (
            None
            if self.enable_sm70_dflash2_fused_gdn_combined_split
            else "policy_hardware_or_speculation"
        )
        if self.enable_sm70_dflash2_fused_gdn_combined_split:
            from vllm.model_executor.models.qwen3_5 import (
                _can_implement_sm70_combined_gdn_split,
            )

            (
                self.enable_sm70_dflash2_fused_gdn_combined_split,
                self._sm70_dflash2_combined_split_reason,
            ) = _can_implement_sm70_combined_gdn_split(
                8,
                torch.float16,
                (2 * self.key_dim + self.value_dim) // self.tp_size,
                self.value_dim // self.tp_size,
                self.num_v_heads // self.tp_size,
            )
        self.enable_sm70_dflash2_fused_qkv_pack = bool(
            sm70_dflash2_enabled("fused_qkv_pack", verifier_policy)
            and current_platform.is_device_capability(70)
            and _is_dflash2_spec_config(vllm_config)
        )
        self.compare_sm70_fused_sigmoid_mixed_qkv = capture_runtime_trace().value(
            "gdn_mixed_compare"
        )
        self.enable_flashqla_decode = self.gdn_policy.flashqla_decode
        self.flashqla_decode_admission = FlashQlaDecodeAdmission.bind(
            self.gdn_heads, self.enable_flashqla_decode
        )
        self._flashqla_native_policy = (
            flashqla_prefill_provider.bind_flashqla_native_policy(
                vllm_config,
                self.gdn_policy,
                needed=self.flashqla_decode_admission.needs_native_policy(
                    vllm_config.model_config.dtype
                ),
            )
        )
        self.force_sm70_qwen_gdn_full_forward = bool(
            self.gdn_policy.projection.full_forward
        )
        num_speculative_tokens = _sm70_qwen_gdn_num_speculative_tokens(vllm_config)
        block_003_deep_mtp = _sm70_qwen_gdn_block_003_spec_for_deep_native_mtp(
            vllm_config, self.gdn_policy.projection
        )
        if block_003_deep_mtp:
            logger.warning_once(
                "Blocking SM70 Qwen GDN 0.0.3-style spec recurrent-core "
                "route for native MTP%d. Long official-sampling validation "
                "showed this verifier path can enter a sustained all-accept "
                "repeat loop for num_speculative_tokens >= 3. Using the "
                "full-Qwen-GDN verifier guard instead. Set "
                "VLLM_SM70_QWEN_GDN_003_SPEC_ALLOW_DEEP_MTP=1 only for "
                "diagnostic reruns of the unsafe route.",
                num_speculative_tokens,
            )
        self.disable_sm70_qwen_gdn_full_forward = bool(
            self.gdn_policy.projection.disable_full_forward and not block_003_deep_mtp
        )
        self.auto_sm70_qwen_gdn_full_forward = (
            bool(self._execution_graph_policy.compile_graph)
            and vllm_config.speculative_config is not None
            and not self.gdn_policy.state.spec_core
            and (not self.gdn_policy.projection.spec_core_003 or block_003_deep_mtp)
        )
        self.auto_sm70_qwen_gdn_spec_core = (
            bool(self._execution_graph_policy.compile_graph)
            and vllm_config.speculative_config is not None
            and self.gdn_policy.state.spec_core
        )
        self.auto_sm70_qwen_gdn_003_spec_core = (
            bool(self._execution_graph_policy.compile_graph)
            and vllm_config.speculative_config is not None
            and self.gdn_policy.projection.spec_core_003
            and not block_003_deep_mtp
        )
        # The automatic arm is Volta-only. The wrapper runs the whole layer
        # through one opaque custom op, so Inductor never sees the input and
        # output projections around the recurrent core and the gated RMSNorm
        # falls back to its native chain. On Turing that costs about fifteen
        # extra elementwise launches per GDN layer and step. The recurrent
        # core keeps its own boundary either way, and an explicit
        # VLLM_SM70_QWEN_GDN_FULL_FORWARD=1 still forces the wrapper anywhere.
        self.maybe_sm70_qwen_gdn_full_forward = (
            not self.disable_sm70_qwen_gdn_full_forward
            and (
                self.force_sm70_qwen_gdn_full_forward
                or (
                    self.auto_sm70_qwen_gdn_full_forward
                    and _sm70_current_device_is_volta()
                )
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
        self.enable_sm70_legacy_prefill_prep = self.gdn_policy.legacy_prefill_prep
        self.gdn_preparation = GdnPreparation(
            self.gdn_heads,
            legacy=bool(self.enable_sm70_legacy_prefill_prep),
            fused_pack=self.enable_sm70_dflash2_fused_qkv_pack,
            gate_is_exp=(
                not self.enable_sm70_legacy_prefill_prep
                and self.gdn_prefill_backend == "flashqla_sm70"
                and not self.gdn_policy.original_prefill
            ),
        )
        if current_platform.is_device_capability(70) and (
            self.gdn_policy.schedule.kkt_enabled
            or self.gdn_policy.schedule.delta_h_enabled
            or self.gdn_policy.schedule.chunk_o_enabled
            or self.gdn_policy.schedule.recurrent_enabled
            or self.gdn_policy.schedule.sigmoid_enabled
        ):
            logger.info_once(
                "SM70 GDN/FLA schedule gates enabled: "
                "kkt=%s delta_h=%s chunk_o=%s recurrent=%s "
                "sigmoid_gating=%s.",
                self.gdn_policy.schedule.kkt_enabled,
                self.gdn_policy.schedule.delta_h_enabled,
                self.gdn_policy.schedule.chunk_o_enabled,
                self.gdn_policy.schedule.recurrent_enabled,
                self.gdn_policy.schedule.sigmoid_enabled,
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
        if self.enable_sm70_dflash2_fused_gdn_verify:
            logger.info_once(
                "SM70 DFlash2 packed GDN target-verification route enabled.",
                scope="local",
            )
        if self.enable_sm70_dflash2_fused_gdn_norm:
            logger.info_once(
                "SM70 DFlash2 one-pass GDN output RMSNorm route enabled.",
                scope="local",
            )
        if self.enable_sm70_dflash2_fused_gdn_split:
            logger.info_once(
                "SM70 DFlash2 fused GDN z/b/a materialization route enabled.",
                scope="local",
            )
        if self.enable_sm70_dflash2_fused_qkv_pack:
            logger.info_once(
                "SM70 DFlash2 fused post-convolution Q/K/V packing enabled.",
                scope="local",
            )
        if capture_runtime_trace().gdn_legacy_fused_notice:
            logger.info_once(
                "VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING is upstream-split "
                "into VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE, "
                "VLLM_SM70_FUSED_SIGMOID_GATING_* and "
                "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV controls.",
                scope="local",
            )
        if capture_runtime_trace().value("gdn_empty_output_notice"):
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

        if bool(
            self._execution_graph_policy.compile_graph
        ) and use_sm70_decode_graph_semantics(self._execution_graph_policy):
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
            key = mixed_qkvz.index_select(-1, k_idx).view(*base_shape_qkvz, ng, k_size)
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
            z_flat = _sm70_compile_graph_slice_dim(mixed_qkvz, -1, qkv_size, z_size)
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

        if bool(
            self._execution_graph_policy.compile_graph
        ) and use_sm70_decode_graph_semantics(self._execution_graph_policy):
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
            key = mixed_qkvz.index_select(-1, k_idx).view(*base_shape_qkvz, ng, k_size)
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
        return self.gdn_preparation.unpack(mixed_qkv)

    def _probe_ddtree_path_replay_states(
        self,
        *,
        mixed_qkv_spec: torch.Tensor,
        a_spec: torch.Tensor,
        b_spec: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weights: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        ddtree_parent_ids: torch.Tensor,
        ddtree_num_tree_tokens_cpu: torch.Tensor,
        num_spec_decodes: int,
        initial_conv_state: torch.Tensor,
        initial_ssm_state: torch.Tensor,
    ) -> None:
        global _DFLASH_DDTREE_PATH_PROBE_REPORTS

        max_reports = _dflash_ddtree_path_probe_max_reports()
        if max_reports <= _DFLASH_DDTREE_PATH_PROBE_REPORTS:
            return

        parent_rows = ddtree_parent_ids[:num_spec_decodes].detach().cpu()
        lengths = ddtree_num_tree_tokens_cpu[:num_spec_decodes].detach().cpu()
        starts = spec_query_start_loc[: num_spec_decodes + 1].detach().cpu()
        node_limit = _dflash_ddtree_path_probe_node_limit()
        device = mixed_qkv_spec.device
        conv_state_width = min(
            int(conv_weights.shape[1]) - 1,
            int(conv_state.shape[-1]),
        )
        zero_idx = torch.zeros(1, dtype=torch.int32, device=device)
        one_idx = torch.ones(1, dtype=torch.int32, device=device)
        worst: tuple[float, float, int, int, list[int], int] | None = None

        for req_row in range(num_spec_decodes):
            num_tree_tokens = int(lengths[req_row].item())
            if num_tree_tokens <= 0:
                continue
            query_start = int(starts[req_row].item())
            query_len = int(starts[req_row + 1].item()) - query_start
            if query_len != num_tree_tokens + 1:
                continue
            token_start = query_start + 1
            root_state_idx = int(spec_state_indices_tensor[req_row, 0].item())
            if root_state_idx >= 0:
                local_conv_state = conv_state.new_empty(
                    (2, conv_state.shape[1], conv_state.shape[2])
                )
                local_conv_state[0].copy_(initial_conv_state[req_row])
                row_idx = torch.tensor([query_start], dtype=torch.long, device=device)
                x_row = mixed_qkv_spec.index_select(0, row_idx)
                conv_state_indices = torch.tensor(
                    [[0, 1]],
                    dtype=torch.int32,
                    device=device,
                )
                y_row = causal_conv1d_update(
                    x_row,
                    local_conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=conv_state_indices,
                    block_idx_last_scheduled_token=one_idx,
                    initial_state_idx=zero_idx,
                    validate_data=False,
                )
                query_path, key_path, value_path = self.rearrange_mixed_qkv(y_row)
                path_token_idx = torch.tensor(
                    [query_start], dtype=torch.long, device=device
                )
                g_path, beta_path = fused_gdn_gating(
                    self.A_log,
                    a_spec.index_select(0, path_token_idx),
                    b_spec.index_select(0, path_token_idx),
                    self.dt_bias,
                    beta_dtype=torch.float32,
                )
                local_ssm_state = initial_ssm_state[req_row : req_row + 1].clone()
                _, root_final_state = fused_recurrent_gated_delta_rule(
                    q=query_path,
                    k=key_path,
                    v=value_path,
                    g=g_path,
                    beta=beta_path,
                    initial_state=local_ssm_state,
                    inplace_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
                conv_diff = (
                    (
                        conv_state[root_state_idx, :, :conv_state_width].float()
                        - local_conv_state[1, :, :conv_state_width].float()
                    )
                    .abs()
                    .max()
                )
                ssm_diff = (
                    (ssm_state[root_state_idx].float() - root_final_state[-1].float())
                    .abs()
                    .max()
                )
                conv_diff_value = float(conv_diff.item())
                ssm_diff_value = float(ssm_diff.item())
                if worst is None or max(conv_diff_value, ssm_diff_value) > max(
                    worst[0], worst[1]
                ):
                    worst = (
                        conv_diff_value,
                        ssm_diff_value,
                        req_row,
                        0,
                        [],
                        root_state_idx,
                    )

            max_node = min(num_tree_tokens, node_limit)
            for node_slot in range(1, max_node + 1):
                path_slots: list[int] = []
                current_slot = node_slot
                seen_slots: set[int] = set()
                while current_slot > 0:
                    if current_slot in seen_slots or current_slot > num_tree_tokens:
                        path_slots = []
                        break
                    seen_slots.add(current_slot)
                    path_slots.append(current_slot)
                    parent_slot = int(parent_rows[req_row, current_slot].item())
                    current_slot = 0 if parent_slot < 0 else parent_slot
                if not path_slots:
                    continue
                path_slots.reverse()
                token_rows = [query_start]
                token_rows.extend(token_start + slot - 1 for slot in path_slots)

                local_conv_state = conv_state.new_empty(
                    (len(token_rows) + 1, conv_state.shape[1], conv_state.shape[2])
                )
                local_conv_state[0].copy_(initial_conv_state[req_row])
                processed_rows: list[torch.Tensor] = []
                for step_idx, token_row in enumerate(token_rows):
                    row_idx = torch.tensor([token_row], dtype=torch.long, device=device)
                    x_row = mixed_qkv_spec.index_select(0, row_idx)
                    conv_state_indices = torch.tensor(
                        [[step_idx, step_idx + 1]],
                        dtype=torch.int32,
                        device=device,
                    )
                    y_row = causal_conv1d_update(
                        x_row,
                        local_conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        self.activation,
                        conv_state_indices=conv_state_indices,
                        block_idx_last_scheduled_token=one_idx,
                        initial_state_idx=zero_idx,
                        validate_data=False,
                    )
                    processed_rows.append(y_row.squeeze(0))

                path_mixed_qkv = torch.stack(processed_rows, dim=0)
                query_path, key_path, value_path = self.rearrange_mixed_qkv(
                    path_mixed_qkv
                )
                path_token_idx = torch.tensor(
                    token_rows, dtype=torch.long, device=device
                )
                g_path, beta_path = fused_gdn_gating(
                    self.A_log,
                    a_spec.index_select(0, path_token_idx),
                    b_spec.index_select(0, path_token_idx),
                    self.dt_bias,
                    beta_dtype=torch.float32,
                )
                local_ssm_state = initial_ssm_state[req_row : req_row + 1].clone()
                _, path_final_state = fused_recurrent_gated_delta_rule(
                    q=query_path,
                    k=key_path,
                    v=value_path,
                    g=g_path,
                    beta=beta_path,
                    initial_state=local_ssm_state,
                    inplace_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
                local_ssm_final = path_final_state[-1]

                actual_state_idx = int(
                    spec_state_indices_tensor[req_row, node_slot].item()
                )
                if actual_state_idx < 0:
                    continue
                conv_diff = (
                    (
                        conv_state[actual_state_idx, :, :conv_state_width].float()
                        - local_conv_state[
                            len(token_rows), :, :conv_state_width
                        ].float()
                    )
                    .abs()
                    .max()
                )
                ssm_diff = (
                    (ssm_state[actual_state_idx].float() - local_ssm_final.float())
                    .abs()
                    .max()
                )
                conv_diff_value = float(conv_diff.item())
                ssm_diff_value = float(ssm_diff.item())
                if worst is None or max(conv_diff_value, ssm_diff_value) > max(
                    worst[0], worst[1]
                ):
                    worst = (
                        conv_diff_value,
                        ssm_diff_value,
                        req_row,
                        node_slot,
                        path_slots,
                        actual_state_idx,
                    )

        if worst is None:
            return
        if max_reports <= _DFLASH_DDTREE_PATH_PROBE_REPORTS:
            return
        _DFLASH_DDTREE_PATH_PROBE_REPORTS += 1
        conv_diff_value, ssm_diff_value, req_row, node_slot, path_slots, state_idx = (
            worst
        )
        logger.info(
            "DFlash DDTree path probe layer=%s req=%d node=%d path=%s "
            "state_idx=%d conv_max=%.6g ssm_max=%.6g",
            self.prefix,
            req_row,
            node_slot,
            tuple(path_slots),
            state_idx,
            conv_diff_value,
            ssm_diff_value,
        )

    def _forward_ddtree_gdn_pure_spec_flashqla(
        self,
        *,
        mixed_qkv_spec: torch.Tensor,
        a_spec: torch.Tensor,
        b_spec: torch.Tensor,
        ssm_state: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        spec_state_slot_selectors: torch.Tensor,
        ddtree_parent_ids: torch.Tensor,
        num_spec_decodes: int,
    ) -> torch.Tensor | None:
        if not self.enable_flashqla_decode:
            _log_runtime_route_once(
                "DDTree FlashQLA GDN requested but SM70 FlashQLA decode "
                "route is disabled."
            )
            return None
        if (
            not mixed_qkv_spec.is_cuda
            or mixed_qkv_spec.dtype != torch.float16
            or self.head_k_dim != 128
            or self.head_v_dim != 128
        ):
            return None
        if self.num_k_heads % self.tp_size != 0 or self.num_v_heads % self.tp_size != 0:
            return None
        try:
            from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import (
                gdn_decode_mixed_qkv_ddtree_state_sm70,
            )
        except ImportError:
            _log_runtime_route_once(
                "DDTree FlashQLA GDN requested but flash_qla DDTree decode "
                "extension is not importable."
            )
            return None

        state_indices = spec_state_indices_tensor[:num_spec_decodes]
        parent_ids = ddtree_parent_ids[:num_spec_decodes]
        accepted = spec_state_slot_selectors[:num_spec_decodes]
        cu_seqlens = spec_query_start_loc[: num_spec_decodes + 1]
        if state_indices.dtype != torch.int32:
            state_indices = state_indices.to(dtype=torch.int32)
        if parent_ids.dtype != torch.int32:
            parent_ids = parent_ids.to(dtype=torch.int32)
        if accepted.dtype != torch.int32:
            accepted = accepted.to(dtype=torch.int32)
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(dtype=torch.int32)

        state_indices = state_indices.contiguous()
        parent_ids = parent_ids.contiguous()
        accepted = accepted.contiguous()
        cu_seqlens = cu_seqlens.contiguous()
        output = mixed_qkv_spec.new_empty(
            (
                mixed_qkv_spec.shape[0],
                self.num_v_heads // self.tp_size,
                self.head_v_dim,
            )
        )
        gdn_decode_mixed_qkv_ddtree_state_sm70(
            mixed_qkv_spec,
            a_spec.contiguous(),
            b_spec.contiguous(),
            self.A_log,
            self.dt_bias,
            ssm_state,
            state_indices,
            parent_ids,
            accepted,
            cu_seqlens,
            output,
            scale=self.head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
        )
        _log_runtime_route_once(
            "Using FlashQLA DDTree Qwen GDN pure-spec verifier path."
        )
        return output.unsqueeze(0)

    def _forward_ddtree_gdn_pure_spec(
        self,
        *,
        mixed_qkv_spec: torch.Tensor,
        a_spec: torch.Tensor,
        b_spec: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weights: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        spec_state_slot_selectors: torch.Tensor | None,
        ddtree_parent_ids: torch.Tensor,
        ddtree_num_tree_tokens_cpu: torch.Tensor,
        num_spec_decodes: int,
    ) -> torch.Tensor:
        if (
            _dflash_ddtree_fused_gdn_enabled()
            and spec_state_slot_selectors is not None
            and _ddtree_queries_have_prefix_row(
                num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
                spec_query_start_loc=spec_query_start_loc,
                num_spec_decodes=num_spec_decodes,
            )
        ):
            layer_name = _encode_layer_name(self.prefix)
            _log_runtime_route_once(
                "Using fused DDTree Qwen GDN pure-spec verifier path."
            )
            profile_start = self._gdn_profiler.start()
            mixed_qkv_spec = causal_conv1d_update_ddtree(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor,
                parent_ids=ddtree_parent_ids,
                num_accepted_tokens=spec_state_slot_selectors,
                query_start_loc=spec_query_start_loc,
            )
            if _dflash_ddtree_qla_gdn_enabled():
                core_attn_out_spec = self._forward_ddtree_gdn_pure_spec_flashqla(
                    mixed_qkv_spec=mixed_qkv_spec,
                    a_spec=a_spec,
                    b_spec=b_spec,
                    ssm_state=ssm_state,
                    spec_query_start_loc=spec_query_start_loc,
                    spec_state_indices_tensor=spec_state_indices_tensor,
                    spec_state_slot_selectors=spec_state_slot_selectors,
                    ddtree_parent_ids=ddtree_parent_ids,
                    num_spec_decodes=num_spec_decodes,
                )
                if core_attn_out_spec is not None:
                    self._gdn_profiler.end(
                        layer_name,
                        "ddtree_pure_spec_flashqla",
                        profile_start,
                        tokens=mixed_qkv_spec.shape[0],
                        details=(
                            f"requests={num_spec_decodes} "
                            "backend=flashqla_ddtree "
                            "tree_tokens="
                            f"{int(ddtree_num_tree_tokens_cpu.sum().item())}"
                        ),
                    )
                    return core_attn_out_spec
            core_attn_out_spec, _ = fused_sigmoid_gating_delta_rule_update_mixed_qkv(
                A_log=self.A_log,
                a=a_spec,
                b=b_spec,
                dt_bias=self.dt_bias,
                mixed_qkv=mixed_qkv_spec,
                num_q_heads=self.num_k_heads // self.tp_size,
                num_v_heads=self.num_v_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=spec_state_slot_selectors,
                ddtree_parent_ids=ddtree_parent_ids,
                use_qk_l2norm_in_kernel=True,
            )
            self._gdn_profiler.end(
                layer_name,
                "ddtree_pure_spec_fused",
                profile_start,
                tokens=mixed_qkv_spec.shape[0],
                details=(
                    f"requests={num_spec_decodes} "
                    f"tree_tokens={int(ddtree_num_tree_tokens_cpu.sum().item())}"
                ),
            )
            return core_attn_out_spec

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DDTree Qwen GDN tree replay is not CUDA-graph safe yet. "
                "Run dflash_ddtree tree verification with eager/piecewise graph "
                "until the tree-aware GDN kernels are fused."
            )

        layer_name = _encode_layer_name(self.prefix)
        profile_start = self._gdn_profiler.start()
        prefix_rows, depth_batches = _ddtree_depth_batches(
            parent_ids=ddtree_parent_ids,
            num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
            spec_query_start_loc=spec_query_start_loc,
            num_spec_decodes=num_spec_decodes,
        )
        total_spec_tokens = mixed_qkv_spec.shape[0]
        core_attn_out_spec = mixed_qkv_spec.new_empty(
            (
                1,
                total_spec_tokens,
                self.num_v_heads // self.tp_size,
                self.head_v_dim,
            )
        )
        if _dflash_ddtree_linear_gdn_enabled():
            mixed_qkv_linear = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0],
                num_accepted_tokens=spec_state_slot_selectors,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
            query_linear, key_linear, value_linear = self.rearrange_mixed_qkv(
                mixed_qkv_linear
            )
            g_linear, beta_linear = fused_gdn_gating(
                self.A_log,
                a_spec,
                b_spec,
                self.dt_bias,
                beta_dtype=torch.float32,
            )
            core_attn_out_linear, _ = fused_recurrent_gated_delta_rule(
                q=query_linear,
                k=key_linear,
                v=value_linear,
                g=g_linear,
                beta=beta_linear,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=spec_state_slot_selectors,
                use_qk_l2norm_in_kernel=True,
            )
            self._gdn_profiler.end(
                layer_name,
                "ddtree_pure_spec_linear_diagnostic",
                profile_start,
                tokens=total_spec_tokens,
                details=(
                    f"requests={num_spec_decodes} "
                    f"tree_tokens={int(ddtree_num_tree_tokens_cpu.sum().item())}"
                ),
            )
            return core_attn_out_linear

        device = mixed_qkv_spec.device
        conv_already_updated = _dflash_ddtree_conv_kernel_enabled()
        if conv_already_updated:
            mixed_qkv_spec = causal_conv1d_update_ddtree(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor,
                parent_ids=ddtree_parent_ids,
                num_accepted_tokens=spec_state_slot_selectors,
                query_start_loc=spec_query_start_loc,
            )
        max_state_slot = max(0, spec_state_indices_tensor.shape[1] - 1)
        if spec_state_slot_selectors is None:
            selector_offsets = torch.zeros(
                (spec_state_indices_tensor.shape[0],),
                dtype=torch.long,
                device=device,
            )
        else:
            selector_offsets = (
                spec_state_slot_selectors[: spec_state_indices_tensor.shape[0]]
                .to(device=device, dtype=torch.long, non_blocking=True)
                .sub(1)
                .clamp_(min=0, max=max_state_slot)
            )
        req_arange = torch.arange(
            spec_state_indices_tensor.shape[0],
            dtype=torch.long,
            device=device,
        )
        selected_root_state_indices = spec_state_indices_tensor[
            req_arange, selector_offsets
        ]
        root_state_indices = spec_state_indices_tensor[:, 0]
        parent_zero_state_indices = selected_root_state_indices.clone()
        path_probe_initial_conv_state: torch.Tensor | None = None
        path_probe_initial_ssm_state: torch.Tensor | None = None
        if (
            not conv_already_updated
            and _dflash_ddtree_path_probe_enabled()
            and _dflash_ddtree_path_probe_layer_matches(self.prefix)
        ):
            root_indices = selected_root_state_indices.to(torch.long)
            path_probe_initial_conv_state = conv_state.index_select(
                0, root_indices
            ).clone()
            path_probe_initial_ssm_state = ssm_state.index_select(
                0, root_indices
            ).clone()
        if prefix_rows:
            token_rows, req_rows = zip(*prefix_rows, strict=True)
            token_idx = torch.tensor(token_rows, dtype=torch.long, device=device)
            req_idx = torch.tensor(req_rows, dtype=torch.long, device=device)
            input_state_idx = selected_root_state_indices[req_idx]
            output_state_idx = root_state_indices[req_idx]
            conv_state_indices = torch.stack(
                (input_state_idx, output_state_idx), dim=1
            ).to(torch.int32)
            zero_idx = torch.zeros(
                conv_state_indices.shape[0],
                dtype=torch.int32,
                device=device,
            )
            one_idx = torch.ones_like(zero_idx)

            mixed_qkv_prefix = mixed_qkv_spec.index_select(0, token_idx)
            if not conv_already_updated:
                mixed_qkv_prefix = causal_conv1d_update(
                    mixed_qkv_prefix,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=conv_state_indices,
                    block_idx_last_scheduled_token=one_idx,
                    initial_state_idx=zero_idx,
                    validate_data=False,
                )
            query_prefix, key_prefix, value_prefix = self.rearrange_mixed_qkv(
                mixed_qkv_prefix
            )
            g_prefix, beta_prefix = fused_gdn_gating(
                self.A_log,
                a_spec.index_select(0, token_idx),
                b_spec.index_select(0, token_idx),
                self.dt_bias,
                beta_dtype=torch.float32,
            )
            cu_seqlens = torch.arange(
                input_state_idx.shape[0] + 1,
                dtype=torch.int32,
                device=device,
            )
            core_prefix, final_state = fused_recurrent_gated_delta_rule(
                q=query_prefix,
                k=key_prefix,
                v=value_prefix,
                g=g_prefix,
                beta=beta_prefix,
                initial_state=ssm_state,
                inplace_final_state=False,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=input_state_idx.to(torch.int32),
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state.index_copy_(
                0,
                output_state_idx.to(torch.long),
                final_state.to(ssm_state.dtype),
            )
            parent_zero_state_indices.index_copy_(
                0,
                req_idx,
                output_state_idx,
            )
            core_attn_out_spec.index_copy_(1, token_idx, core_prefix)

        serial_tree_replay = _dflash_ddtree_serial_gdn_enabled()
        for depth_batch in depth_batches:
            if not depth_batch:
                continue
            replay_batches = (
                ((entry,) for entry in depth_batch)
                if serial_tree_replay
                else (tuple(depth_batch),)
            )
            for replay_batch in replay_batches:
                token_rows, req_rows, parent_slots, node_slots = zip(
                    *replay_batch, strict=True
                )
                device = mixed_qkv_spec.device
                token_idx = torch.tensor(token_rows, dtype=torch.long, device=device)
                req_idx = torch.tensor(req_rows, dtype=torch.long, device=device)
                parent_idx = torch.tensor(parent_slots, dtype=torch.long, device=device)
                node_idx = torch.tensor(node_slots, dtype=torch.long, device=device)

                parent_state_indices = spec_state_indices_tensor[req_idx, parent_idx]
                parent_state_indices = torch.where(
                    parent_idx == 0,
                    parent_zero_state_indices[req_idx],
                    parent_state_indices,
                )
                node_state_indices = spec_state_indices_tensor[req_idx, node_idx]
                conv_state_indices = torch.stack(
                    (parent_state_indices, node_state_indices), dim=1
                ).to(torch.int32)
                zero_idx = torch.zeros(
                    conv_state_indices.shape[0],
                    dtype=torch.int32,
                    device=device,
                )
                one_idx = torch.ones_like(zero_idx)

                mixed_qkv_depth = mixed_qkv_spec.index_select(0, token_idx)
                if not conv_already_updated:
                    mixed_qkv_depth = causal_conv1d_update(
                        mixed_qkv_depth,
                        conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        self.activation,
                        conv_state_indices=conv_state_indices,
                        block_idx_last_scheduled_token=one_idx,
                        initial_state_idx=zero_idx,
                        validate_data=False,
                    )
                query_depth, key_depth, value_depth = self.rearrange_mixed_qkv(
                    mixed_qkv_depth
                )
                g_depth, beta_depth = fused_gdn_gating(
                    self.A_log,
                    a_spec.index_select(0, token_idx),
                    b_spec.index_select(0, token_idx),
                    self.dt_bias,
                    beta_dtype=torch.float32,
                )
                cu_seqlens = torch.arange(
                    conv_state_indices.shape[0] + 1,
                    dtype=torch.int32,
                    device=device,
                )
                core_depth, final_state = fused_recurrent_gated_delta_rule(
                    q=query_depth,
                    k=key_depth,
                    v=value_depth,
                    g=g_depth,
                    beta=beta_depth,
                    initial_state=ssm_state,
                    inplace_final_state=False,
                    cu_seqlens=cu_seqlens,
                    ssm_state_indices=parent_state_indices.to(torch.int32),
                    use_qk_l2norm_in_kernel=True,
                )
                ssm_state.index_copy_(
                    0,
                    node_state_indices.to(torch.long),
                    final_state.to(ssm_state.dtype),
                )
                core_attn_out_spec.index_copy_(1, token_idx, core_depth)

        if (
            path_probe_initial_conv_state is not None
            and path_probe_initial_ssm_state is not None
        ):
            self._probe_ddtree_path_replay_states(
                mixed_qkv_spec=mixed_qkv_spec,
                a_spec=a_spec,
                b_spec=b_spec,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv_weights=conv_weights,
                spec_query_start_loc=spec_query_start_loc,
                spec_state_indices_tensor=spec_state_indices_tensor,
                ddtree_parent_ids=ddtree_parent_ids,
                ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
                num_spec_decodes=num_spec_decodes,
                initial_conv_state=path_probe_initial_conv_state,
                initial_ssm_state=path_probe_initial_ssm_state,
            )

        self._gdn_profiler.end(
            layer_name,
            "ddtree_pure_spec_replay",
            profile_start,
            tokens=total_spec_tokens,
            details=(
                f"prefix_rows={len(prefix_rows)} "
                f"depths={sum(1 for batch in depth_batches if batch)} "
                f"tree_tokens={int(ddtree_num_tree_tokens_cpu.sum().item())}"
            ),
        )
        return core_attn_out_spec

    def _can_use_flashqla_decode(
        self,
        mixed_qkv,
        state_indices,
        num_decode_tokens,
        *,
        layer_name=None,
        stage="decode",
    ):
        if layer_name is None:
            layer_name = _encode_layer_name(self.prefix)
        reason = self.flashqla_decode_admission.rejection(
            mixed_qkv,
            state_indices,
            num_decode_tokens,
        )
        _sm70_log_flashqla_decode_route(
            layer_name=layer_name,
            stage=stage,
            decision="take" if reason is None else "skip",
            reason=reason or "ok",
            mixed_qkv=mixed_qkv,
            state_indices=state_indices,
            num_decode_tokens=num_decode_tokens,
        )
        return reason is None

    def _forward_core_decode_flashqla(self, **kwargs):
        return flashqla_decode(
            self.gdn_heads,
            self.A_log,
            self.dt_bias,
            native_policy=self._flashqla_native_policy,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor | None,
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
                    hidden_states if output is None else output,
                )
                if output is None:
                    return torch.ops.vllm.qwen_gdn_full_forward_direct(
                        hidden_states,
                        conv_state_cache,
                        ssm_state_cache,
                        layer_name,
                    )
                torch.ops.vllm.qwen_gdn_full_forward(
                    hidden_states,
                    output,
                    conv_state_cache,
                    ssm_state_cache,
                    layer_name,
                )
                return
        return self._forward_method(hidden_states, output)

    def _full_forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor | None,
    ):
        return self._forward_method(hidden_states, output)

    def _compute_output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        num_tokens: int,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        layer_name = _encode_layer_name(self.prefix)
        total_start = self._gdn_profiler.start()
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = _sm70_dump_gdn_projection_tensor(
            "proj_core_in", layer_name, core_attn_out
        )
        z = _sm70_dump_gdn_projection_tensor("proj_z", layer_name, z)
        profile_start = self._gdn_profiler.start()
        use_sm70_onepass_norm = (
            self.enable_sm70_gdn_rmsnorm_onepass
            and num_tokens == 1
            and core_attn_out.dtype == torch.float16
            and core_attn_out.shape == (12, 128)
            and core_attn_out.is_contiguous()
            and z.dtype == core_attn_out.dtype
            and z.shape == core_attn_out.shape
            and z.is_contiguous()
            and self.norm.weight.dtype == core_attn_out.dtype
            and self.norm.weight.shape == (128,)
            and self.norm.weight.is_contiguous()
            and self.norm.group_size is None
            and self.norm.norm_before_gate
            and self.norm.activation in ("silu", "swish")
        )
        if self.enable_sm70_dflash2_fused_gdn_norm:
            # RMSNormGated is decomposed into a nine-kernel native chain when
            # it is reached from the opaque Qwen GDN op under the default
            # inductor custom-op policy. Its canonical CUDA implementation is
            # a one-pass Triton kernel with the same FP32 accumulation and
            # FP16 output contract, so call it directly on this isolated path.
            core_attn_out = self.norm.forward_cuda(core_attn_out, z)
        elif use_sm70_onepass_norm:
            _log_runtime_route_once(
                "SM70 GDN 12x128 RMSNormGated one-pass route enabled."
            )
            core_attn_out = torch.ops.vllm.sm70_qwen_gdn_rmsnorm_gated_configured(
                core_attn_out,
                z,
                self.norm.weight,
                self.norm.eps,
                -1,
                True,
                self.norm.activation,
                self.gdn_policy.projection.rmsnorm_onepass,
            )
        else:
            core_attn_out = self.norm(core_attn_out, z)
        self._gdn_profiler.end(
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
        profile_start = self._gdn_profiler.start()
        projection_override = getattr(self, "output_projection_override", None)
        if projection_override is not None and output is None:
            proj_out = projection_override(core_attn_out)
        elif output is None:
            proj_out, _ = self.out_proj(core_attn_out)
        else:
            # The caller guards the bias-free, non-reducing SM70 FP8 route.
            # Write the compiler-owned destination within the original opaque
            # operator; do not publish a hidden allocator-owned intermediate.
            proj_out = self.out_proj.scheme.apply_weights(
                self.out_proj, core_attn_out, output=output
            )
        self._gdn_profiler.end(
            layer_name,
            "projection_out_proj",
            profile_start,
            tokens=num_tokens,
        )
        proj_out = _sm70_dump_gdn_projection_tensor("proj_out", layer_name, proj_out)
        self._gdn_profiler.end(
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
        output: torch.Tensor | None,
        num_tokens: int,
    ) -> torch.Tensor:
        layer_name = _encode_layer_name(self.prefix)
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            CompressedTensorsW8A16Fp8,
        )

        direct_write = bool(
            output is not None
            and self.enable_sm70_dflash2_fused_gdn_verify
            and current_platform.is_device_capability(70)
            and self.tp_size == 4
            and num_tokens == 8
            and output.shape == (8, 5120)
            and output.dtype == torch.float16
            and output.is_contiguous()
            and self.out_proj.bias is None
            and not self.out_proj.reduce_results
            and getattr(self.out_proj, "sm70_fp8_qpn8", False)
            and isinstance(self.out_proj.scheme, CompressedTensorsW8A16Fp8)
        )
        proj_out = self._compute_output_projection(
            core_attn_out, z, num_tokens, output if direct_write else None
        )
        if output is not None:
            if not direct_write:
                output[:num_tokens] = proj_out
            _sm70_gdn_graph_buffer_copy(
                "proj_output_after_write",
                layer_name,
                output[:num_tokens],
                "proj",
            )
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
        output: torch.Tensor | None,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        layer_name = _encode_layer_name(self.prefix)
        if _sm70_qwen_gdn_input_core_boundary_enabled(self.gdn_policy.projection):
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
            z, core_attn_out = torch.ops.vllm.qwen_gdn_input_projection_core(
                hidden_states,
                z,
                core_attn_out,
                conv_state_cache,
                ssm_state_cache,
                layer_name,
            )
            if self.gdn_policy.projection.output_projection:
                torch.ops.vllm.qwen_gdn_output_projection(
                    core_attn_out,
                    z,
                    output,
                    num_tokens,
                    layer_name,
                )
            else:
                return self._output_projection(core_attn_out, z, output, num_tokens)
            return None

        if self.gdn_policy.projection.input_projection:
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
            if self.gdn_policy.projection.output_projection:
                torch.ops.vllm.qwen_gdn_output_projection(
                    core_attn_out,
                    z,
                    output,
                    num_tokens,
                    layer_name,
                )
            else:
                return self._output_projection(core_attn_out, z, output, num_tokens)
            return None

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
        use_qwen38_fused_input = bool(
            getattr(self, "sm70_qwen38_fp16_fused_input", False)
            # A LoRA wrapper owns additional projection work. Only bypass plain
            # linears; other layers still benefit from their base GEMV method.
            and isinstance(self.in_proj_qkvz, LinearBase)
            and isinstance(self.in_proj_ba, LinearBase)
            and use_sm70_decode_graph_semantics(self._execution_graph_policy)
            and not _sm70_gdn_projection_dump_requested(layer_name)
        )
        from vllm.model_executor.layers.quantization.sm70_gdn_ba_verify import (
            apply_gdn_ba_verify,
        )

        fused_verify_projection = apply_gdn_ba_verify(self, hidden_states)
        if fused_verify_projection is not None:
            mixed_qkv, z, b, a = fused_verify_projection
            z = z.reshape(z.size(0), -1, self.head_v_dim)
        elif use_qwen38_fused_input:
            assert self.in_proj_ba is not None
            mixed_qkv, z, b, a = torch.ops.vllm.qwen38_sm70_fp16_gdn_input(
                hidden_states,
                self.in_proj_qkvz.weight,
                self.in_proj_ba.weight,
                getattr(self.in_proj_qkvz, "_sm70_qwen38_gdn_packed", None),
                getattr(self.in_proj_ba, "_sm70_qwen38_gdn_packed", None),
            )
            z = z.reshape(z.size(0), -1, self.head_v_dim)
        else:
            side = getattr(self, "input_projection_override", None)
            if side is not None and 1 <= hidden_states.shape[0] <= 8:
                mixed_qkvz, ba = side(hidden_states)
            else:
                mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
                ba, _ = self.in_proj_ba(hidden_states)
            mixed_qkvz = _sm70_dump_gdn_projection_tensor(
                "in_proj_qkvz", layer_name, mixed_qkvz
            )
            ba = _sm70_dump_gdn_projection_tensor("in_proj_ba", layer_name, ba)

            from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
                _can_fuse_gdn_projection_split,
                _split_gdn_projection_tails,
            )

            if (
                not self.gqa_interleaved_layout
                and not self.disable_tp_for_ba_proj
                and mixed_qkvz.shape[0] in (5, 20)
                and (self.key_dim * 2 + self.value_dim) // self.tp_size == 2560
                and self.value_dim // self.tp_size == 1536
                and _can_fuse_gdn_projection_split(mixed_qkvz, ba)
            ):
                z = mixed_qkvz.new_empty((num_tokens, 12, 128))
                mixed_qkv, b, a = _split_gdn_projection_tails(mixed_qkvz, ba, z)
                if self.gdn_policy.projection.mixed_qkv_contiguous:
                    mixed_qkv = mixed_qkv.contiguous()
                _log_runtime_route_once(
                    "SM70 default GDN projection tail-copy route hit."
                )
            elif self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba
                )
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value),
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
                if self.gdn_policy.projection.mixed_qkv_contiguous:
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
            self.gdn_policy.projection.z_contiguous
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
        if self.gdn_policy.projection.output_projection:
            torch.ops.vllm.qwen_gdn_output_projection(
                core_attn_out,
                z,
                output,
                num_tokens,
                layer_name,
            )
        else:
            return self._output_projection(core_attn_out, z, output, num_tokens)
        return None

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

        device = conv_state.device
        dtype = conv_state.dtype
        qkv_dim = conv_weights.shape[0]
        qkvz_width = qkv_dim + self.value_dim // self.tp_size
        observed_qkv_row_stride = int(
            getattr(self, "_sm70_gdn_prefill_qkv_row_stride", qkvz_width)
        )
        qkv_row_strides = sorted(
            {
                qkv_dim,
                qkvz_width,
                max(qkv_dim, observed_qkv_row_stride),
            }
        )
        prefill_token_counts = (FLA_CHUNK_SIZE, max(1, FLA_CHUNK_SIZE - 1))
        spec_cache_stride = max(
            1,
            int(getattr(self, "num_spec", 1)),
            int(getattr(self, "_sm70_spec_cache_stride", 1)),
            _sm70_spec_cache_stride_from_config(),
        )
        cache_indices_variants = [torch.zeros(1, device=device, dtype=torch.int32)]
        if spec_cache_stride > 1:
            strided_cache_base = torch.zeros(
                spec_cache_stride, device=device, dtype=torch.int32
            )
            cache_indices_variants.append(
                torch.as_strided(
                    strided_cache_base,
                    size=(1,),
                    stride=(spec_cache_stride,),
                )
            )
        has_initial_state = torch.ones(1, device=device, dtype=torch.bool)
        conv_state_line = conv_state[:1]
        try:
            for prefill_tokens in prefill_token_counts:
                dummy_conv_inputs = []
                for qkv_row_stride in qkv_row_strides:
                    dummy_qkv = torch.randn(
                        prefill_tokens,
                        qkv_row_stride,
                        device=device,
                        dtype=dtype,
                    )
                    dummy_conv_inputs.append(dummy_qkv[:, :qkv_dim].transpose(0, 1))
                query_start_loc = torch.tensor(
                    [0, prefill_tokens], device=device, dtype=torch.int32
                )
                query_start_loc_cpu = torch.tensor(
                    [0, prefill_tokens], dtype=torch.int32
                )
                nums_dict, batch_ptr, token_chunk_offset_ptr = (
                    compute_causal_conv1d_metadata(query_start_loc_cpu, device=device)
                )
                conv_metadata = SimpleNamespace(
                    nums_dict=nums_dict,
                    batch_ptr=batch_ptr,
                    token_chunk_offset_ptr=token_chunk_offset_ptr,
                )
                for dummy_conv_in in dummy_conv_inputs:
                    for cache_indices in cache_indices_variants:
                        causal_conv1d_fn(
                            dummy_conv_in,
                            conv_weights,
                            self.conv1d.bias,
                            activation=self.activation,
                            conv_states=conv_state,
                            cache_indices=cache_indices,
                            has_initial_state=has_initial_state,
                            query_start_loc=query_start_loc,
                            metadata=conv_metadata,
                        )
                        conv_state_line.zero_()
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

        try:
            for decode_tokens in (1, 2):
                dummy_decode_inputs = [
                    torch.randn(
                        decode_tokens,
                        conv_weights.shape[0],
                        device=device,
                        dtype=dtype,
                    )
                ]
                if qkvz_width > conv_weights.shape[0]:
                    # The real no-MTP decode path slices qkv out of qkvz, so the
                    # logical [T, qkv_dim] tensor has row stride qkvz_width.
                    dummy_qkvz = torch.randn(
                        decode_tokens, qkvz_width, device=device, dtype=dtype
                    )
                    dummy_decode_inputs.append(dummy_qkvz[:, : conv_weights.shape[0]])
                decode_cache_indices = torch.zeros(
                    decode_tokens, device=device, dtype=torch.int32
                )
                for dummy_decode_in in dummy_decode_inputs:
                    # The update Triton kernel specializes on num_cache_lines and
                    # input strides, so warm both full KV cache and
                    # projection-slice stride instead of only a compact tensor.
                    causal_conv1d_update(
                        dummy_decode_in,
                        conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        self.activation,
                        conv_state_indices=decode_cache_indices,
                        validate_data=True,
                    )
                    conv_state_line.zero_()
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
        """Bind the model's weights/layout; the warmup owner runs ordered stages."""
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True
        from vllm.model_executor.warmup.gdn import GdnWarmup

        qkv_dim = qkv_or_qkvz.shape[-1] - v_dim
        row_stride = qkv_or_qkvz.stride(0) if qkv_or_qkvz.dim() >= 2 else qkv_dim
        self._sm70_gdn_prefill_qkv_row_stride = max(qkv_dim, int(row_stride))
        _, state_dtype = self.get_state_dtype()
        GdnWarmup(
            heads=self.gdn_heads,
            prefill=self.chunk_gated_delta_rule,
            decode_admission=self.flashqla_decode_admission,
            native_policy=self._flashqla_native_policy,
            schedule=self.gdn_policy.schedule,
            conv_weight=self.conv1d.weight,
            conv_bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_dtype=state_dtype,
            device=qkv_or_qkvz.device,
            dtype=qkv_or_qkvz.dtype,
            input_width=qkv_or_qkvz.shape[-1],
            qkv_dim=qkv_dim,
            qkv_row_stride=row_stride,
            spec_cache_stride=max(
                1,
                int(getattr(self, "num_spec", 1)),
                int(getattr(self, "_sm70_spec_cache_stride", 1)),
                _sm70_spec_cache_stride_from_config(),
            ),
            conv_dim_first=is_conv_state_dim_first(),
            prefix=self.prefix,
            trace=_sm70_profile_trace_enabled(),
            decode_warmup=bool(self.gdn_policy.decode_warmup),
        ).run(self._prefill_warmup_keys)

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

    def _can_use_dflash2_packed_gdn_verify(
        self,
        *,
        mixed_qkv: torch.Tensor | None,
        a: torch.Tensor,
        b: torch.Tensor,
        core_attn_out: torch.Tensor,
        ssm_state: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> bool:
        return bool(
            self.enable_sm70_dflash2_fused_gdn_verify
            and mixed_qkv is not None
            and attn_metadata.spec_sequence_masks is not None
            and attn_metadata.num_spec_decodes > 0
            # Prefill chunks and plain decodes may share the batch: the
            # verifier only touches the speculative requests, whose tokens and
            # recurrent-state slots are described by the spec_* metadata.
            and attn_metadata.ddtree_parent_ids is None
            and attn_metadata.spec_query_start_loc is not None
            and attn_metadata.spec_state_indices_tensor is not None
            and attn_metadata.spec_state_slot_selectors is not None
            and mixed_qkv.is_cuda
            and mixed_qkv.dtype == torch.float16
            and a.dtype == torch.float16
            and b.dtype == torch.float16
            # The supported verifier contract keeps recurrent state in FP32;
            # an explicit FP16 cache override is also supported.
            and ssm_state.dtype in (torch.float16, torch.float32)
            and mixed_qkv.ndim == 2
            # Qwen3.5's fused projection and in-place convolution retain the
            # wider QKVZBA row stride. The packed consumer can read it directly.
            and mixed_qkv.stride(1) == 1
            and mixed_qkv.stride(0) >= mixed_qkv.shape[1]
            and a.is_contiguous()
            and b.is_contiguous()
            and core_attn_out.is_contiguous()
        )

    def _can_use_sm70_gdn_preprocess(self, mixed_qkv, a, b, conv_state, metadata):
        # Restrict the first admission to a complete single-request verifier.
        # Mixed batches, padding with fewer actual tokens, and trees retain the
        # existing convolution/gating path.
        return bool(
            self.enable_sm70_dflash2_fused_gdn_verify
            and current_platform.is_device_capability(70)
            and self.tp_size == 4
            and self.num_k_heads // self.tp_size == 4
            and self.num_v_heads // self.tp_size == 12
            and self.head_k_dim == self.head_v_dim == 128
            and not self.gqa_interleaved_layout
            and self.conv1d.bias is None
            and self.activation in ("silu", "swish")
            and metadata.num_prefills == metadata.num_decodes == 0
            and metadata.num_spec_decodes == 1
            and metadata.num_actual_tokens == 8
            and metadata.spec_sequence_masks is not None
            and metadata.ddtree_parent_ids is None
            and metadata.spec_query_start_loc is not None
            and metadata.spec_query_start_loc.ndim == 1
            and metadata.spec_query_start_loc.numel() >= 2
            and metadata.spec_state_indices_tensor is not None
            and metadata.spec_state_indices_tensor.ndim == 2
            and metadata.spec_state_indices_tensor.shape[0] >= 1
            and metadata.spec_state_indices_tensor.shape[1] == 8
            and metadata.spec_state_slot_selectors is not None
            and metadata.spec_state_slot_selectors.ndim == 1
            and metadata.spec_state_slot_selectors.numel() >= 1
            and metadata.spec_query_start_loc.dtype == torch.int32
            and metadata.spec_state_indices_tensor.dtype == torch.int32
            and metadata.spec_state_slot_selectors.dtype == torch.int32
            and mixed_qkv.is_cuda
            and mixed_qkv.dtype == a.dtype == b.dtype == torch.float16
            and mixed_qkv.shape == (8, 2560)
            and mixed_qkv.stride(1) == 1
            and a.shape == b.shape == (8, 12)
            and a.is_contiguous()
            and b.is_contiguous()
            and conv_state.dtype == torch.float16
            and conv_state.ndim == 3
            and conv_state.shape[1] == 2560
            and conv_state.shape[2] >= 10
            and self.conv1d.weight.shape == (2560, 1, 4)
            and self.conv1d.weight.dtype == torch.float16
        )

    def _allocate_core_attn_out(self, mixed_qkv, a, b, hidden_states):
        factory = torch.zeros
        # The full-forward opaque operator executes this code outside Dynamo.
        # In traced paths, keep the existing zero allocation. Never expose an
        # uninitialized buffer to a fallback or profiling warmup.
        if not torch.compiler.is_compiling():
            metadata_raw = get_forward_context().attn_metadata
            if isinstance(metadata_raw, dict):
                metadata = metadata_raw.get(self.prefix)
                if isinstance(metadata, GDNAttentionMetadata):
                    cache = getattr(self, "kv_cache", None)
                    if cache is not None:
                        conv_state = (
                            cache[0]
                            if is_conv_state_dim_first()
                            else cache[0].transpose(-1, -2)
                        )
                        if self._can_use_sm70_gdn_preprocess(
                            mixed_qkv, a, b, conv_state, metadata
                        ):
                            factory = torch.empty
        return factory(
            (mixed_qkv.shape[0], self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def _forward_dflash2_packed_gdn_verify(
        self,
        *,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        core_attn_out: torch.Tensor,
        ssm_state: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        spec_state_slot_selectors: torch.Tensor,
        num_spec_decodes: int,
        precomputed_gating: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        num_tokens = mixed_qkv.shape[0]
        out = core_attn_out[:num_tokens].unsqueeze(1)
        # Match the ordinary speculative verifier's FP32 beta materialization.
        # The gating helper otherwise defaults to the FP16 dtype of b.
        if precomputed_gating is None:
            g, beta = fused_gdn_gating(
                self.A_log, a, b, self.dt_bias, beta_dtype=torch.float32
            )
        else:
            g, beta = precomputed_gating
        fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            mixed_qkv=mixed_qkv,
            num_q_heads=self.num_k_heads // self.tp_size,
            num_v_heads=self.num_v_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out,
            cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
            ssm_state_indices=spec_state_indices_tensor[:num_spec_decodes],
            num_accepted_tokens=spec_state_slot_selectors[:num_spec_decodes],
            use_qk_l2norm_in_kernel=True,
            # Keep gating materialization and launch geometry bitwise aligned
            # with the current verifier. This stage removes packed-QKV
            # rearrangement and the final output copy without changing the
            # target recurrence. Fully fused gating stays separately testable.
            precomputed_g=g,
            precomputed_beta=beta,
            quantize_state_each_step=False,
            match_recurrent_schedule=True,
            match_recurrent_numerics=True,
            sm70_tp2_q8_bv2=getattr(self, "enable_sm70_dflash2_tp2_gdn_bv2", False),
            schedule=self.gdn_policy.schedule,
        )
        return out.transpose(0, 1)

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
            and (
                mixed_qkv.is_contiguous()
                or (
                    attn_metadata.spec_sequence_masks is not None
                    and mixed_qkv.shape[0] in (5, 20)
                    and mixed_qkv.stride(1) == 1
                    and mixed_qkv.stride(0) >= mixed_qkv.shape[1]
                )
            )
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
            kv_cache[0] if is_conv_state_dim_first() else kv_cache[0].transpose(-1, -2)
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
        spec_state_slot_selectors = (
            attn_metadata.spec_state_slot_selectors
            if attn_metadata.spec_state_slot_selectors is not None
            else num_accepted_tokens
        )
        ddtree_parent_ids = attn_metadata.ddtree_parent_ids
        ddtree_num_tree_tokens_cpu = attn_metadata.ddtree_num_tree_tokens_cpu
        ddtree_requires_branch = _ddtree_parent_ids_require_branch(
            ddtree_parent_ids,
            ddtree_num_tree_tokens_cpu,
            attn_metadata.num_spec_decodes,
        )
        ddtree_tree_gdn_pure_spec = (
            ddtree_requires_branch
            and spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
        )
        if ddtree_requires_branch and not ddtree_tree_gdn_pure_spec:
            raise RuntimeError(
                "DDTree branched Qwen GDN state replay currently supports only "
                "pure speculative verifier batches."
            )
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

        precomputed_gating = None
        # 1.1: Process the multi-query part
        if self._can_use_sm70_gdn_preprocess(
            mixed_qkv, a, b, conv_state, attn_metadata
        ):
            assert spec_state_indices_tensor is not None
            assert spec_state_slot_selectors is not None
            assert spec_query_start_loc is not None
            from vllm.model_executor.layers.mamba.gdn.sm70_preprocess import (
                conv_gate_zero,
            )

            mixed_qkv_spec, g, beta = conv_gate_zero(
                mixed_qkv,
                conv_state,
                conv_weights,
                spec_state_indices_tensor[:1, 0],
                spec_state_slot_selectors[:1],
                spec_query_start_loc[:2],
                self.A_log,
                a,
                b,
                self.dt_bias,
                core_attn_out,
            )
            precomputed_gating = (g, beta)
            _log_runtime_route_once("SM70 DFlash2 conv/gating/zero route hit.")
        elif spec_sequence_masks is not None and not ddtree_tree_gdn_pure_spec:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = convolve_decode(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=spec_state_slot_selectors,
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
            mixed_qkv_non_spec = convolve_decode(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
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
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
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
            core_attn_out_non_spec, mixed_final_state = mixed_qkv_recurrence(
                self.gdn_heads,
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                mixed_qkv=mixed_qkv_non_spec,
                initial_state=ssm_state,
                inplace_final_state=not compare_mixed_qkv,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes + 1
                ],
                state_indices=non_spec_state_indices_tensor[:num_non_spec_tokens],
                schedule=self.gdn_policy.schedule,
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
                    schedule=self.gdn_policy.schedule,
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

        # The speculative rows are the whole batch, or a subset next to prefill
        # chunks and plain decodes. Verification must not depend on that: its
        # operands are gathered once, and the result is merged back by token
        # index like every other mixed-batch output.
        spec_rows_are_batch = (
            attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0
        )
        a_spec, b_spec = a, b
        if spec_sequence_masks is not None and not spec_rows_are_batch:
            assert spec_token_indx is not None
            a_spec = a.index_select(0, spec_token_indx)
            b_spec = b.index_select(0, spec_token_indx)
        use_dflash2_packed_gdn_verify = self._can_use_dflash2_packed_gdn_verify(
            mixed_qkv=mixed_qkv_spec,
            a=a_spec,
            b=b_spec,
            core_attn_out=core_attn_out,
            ssm_state=ssm_state,
            attn_metadata=attn_metadata,
        )
        use_sm70_mixed_qkv_verify = (
            mixed_qkv_decode_requested
            and not use_dflash2_packed_gdn_verify
            and spec_sequence_masks is not None
            and ddtree_parent_ids is None
            and attn_metadata.num_spec_decodes > 0
            and current_platform.is_device_capability(70)
            and self.num_k_heads // self.tp_size == 4
            and self.num_v_heads // self.tp_size == 12
            and self.head_k_dim == self.head_v_dim == 128
            and ssm_state.dtype == torch.float32
            and mixed_qkv_spec is not None
            and (
                mixed_qkv_spec.is_contiguous()
                or (
                    mixed_qkv_spec.shape[0] in (5, 20)
                    and mixed_qkv_spec.stride(1) == 1
                    and mixed_qkv_spec.stride(0) >= mixed_qkv_spec.shape[1]
                )
            )
            and (1 < mixed_qkv_spec.shape[0] <= 16 or mixed_qkv_spec.shape[0] == 20)
        )
        if use_dflash2_packed_gdn_verify or use_sm70_mixed_qkv_verify:
            query_spec, key_spec, value_spec = None, None, None
        else:
            query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        prefill_l2norm_in_kernel = False
        prefill_gate_is_exp = False
        if attn_metadata.num_prefills > 0:
            profile_start = self._gdn_profiler.start()
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            prefill_l2norm_in_kernel = self.gdn_preparation.legacy
            prefill_gate_is_exp = self.gdn_preparation.gate_is_exp
            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = self.gdn_preparation.prefill(
                mixed_qkv_non_spec,
                a_non_spec,
                b_non_spec,
                self.A_log,
                self.dt_bias,
            )
            self._gdn_profiler.end(
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
            if use_dflash2_packed_gdn_verify:
                assert mixed_qkv_spec is not None
                assert spec_query_start_loc is not None
                assert spec_state_indices_tensor is not None
                assert spec_state_slot_selectors is not None
                # The verifier writes its rows densely from row zero. Beside
                # other requests those are not the leading rows of the layer
                # output, so it fills a scratch buffer that is merged below.
                spec_out = (
                    core_attn_out
                    if spec_rows_are_batch
                    else core_attn_out.new_empty(
                        (mixed_qkv_spec.shape[0], *core_attn_out.shape[1:])
                    )
                )
                core_attn_out_spec = self._forward_dflash2_packed_gdn_verify(
                    mixed_qkv=mixed_qkv_spec,
                    a=a_spec,
                    b=b_spec,
                    core_attn_out=spec_out,
                    ssm_state=ssm_state,
                    spec_query_start_loc=spec_query_start_loc,
                    spec_state_indices_tensor=spec_state_indices_tensor,
                    spec_state_slot_selectors=spec_state_slot_selectors,
                    num_spec_decodes=attn_metadata.num_spec_decodes,
                    precomputed_gating=precomputed_gating,
                )
                _last_recurrent_state = ssm_state
                _log_runtime_route_once(
                    "SM70 DFlash2 packed GDN target-verification route hit."
                    if spec_rows_are_batch
                    else "SM70 DFlash2 packed GDN target-verification route hit "
                    "beside prefill/decode rows."
                )
            elif ddtree_tree_gdn_pure_spec:
                assert mixed_qkv_spec is not None
                assert spec_query_start_loc is not None
                assert spec_state_indices_tensor is not None
                assert ddtree_parent_ids is not None
                assert ddtree_num_tree_tokens_cpu is not None
                assert spec_state_slot_selectors is not None
                core_attn_out_spec = self._forward_ddtree_gdn_pure_spec(
                    mixed_qkv_spec=mixed_qkv_spec,
                    a_spec=a_spec,
                    b_spec=b_spec,
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                    conv_weights=conv_weights,
                    spec_query_start_loc=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    spec_state_indices_tensor=spec_state_indices_tensor[
                        : attn_metadata.num_spec_decodes
                    ],
                    spec_state_slot_selectors=spec_state_slot_selectors[
                        : attn_metadata.num_spec_decodes
                    ],
                    ddtree_parent_ids=ddtree_parent_ids[
                        : attn_metadata.num_spec_decodes
                    ],
                    ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu[
                        : attn_metadata.num_spec_decodes
                    ],
                    num_spec_decodes=attn_metadata.num_spec_decodes,
                )
                _last_recurrent_state = ssm_state
            elif use_sm70_mixed_qkv_verify:
                assert mixed_qkv_spec is not None
                assert spec_query_start_loc is not None
                # Reuse the decode loader with the original fused verifier's
                # gating, BV32 reduction and FP32 state. Avoid materializing
                # three contiguous splits and concatenating them again.
                direct_verify_out = bool(
                    spec_rows_are_batch
                    and num_actual_tokens == mixed_qkv_spec.shape[0]
                    and num_actual_tokens in (5, 20)
                    and core_attn_out.is_contiguous()
                    and core_attn_out.dtype == mixed_qkv_spec.dtype
                )
                verified = (
                    self.verification_update(
                        self.A_log,
                        self.dt_bias,
                        attn_metadata.num_spec_decodes,
                        mixed_qkv_spec,
                        a_spec,
                        b_spec,
                        ssm_state,
                        core_attn_out,
                        num_actual_tokens,
                        direct_verify_out,
                        spec_query_start_loc,
                        spec_state_indices_tensor,
                        spec_state_slot_selectors,
                    )
                    if self.verification_update is not None
                    else None
                )
                if verified is not None:
                    core_attn_out_spec, _last_recurrent_state = verified
                else:
                    core_attn_out_spec, _last_recurrent_state = (
                        fused_sigmoid_gating_delta_rule_update_mixed_qkv(
                            A_log=self.A_log,
                            a=a_spec,
                            b=b_spec,
                            dt_bias=self.dt_bias,
                            mixed_qkv=mixed_qkv_spec,
                            num_q_heads=self.num_k_heads // self.tp_size,
                            num_v_heads=self.num_v_heads // self.tp_size,
                            head_k_dim=self.head_k_dim,
                            head_v_dim=self.head_v_dim,
                            initial_state=ssm_state,
                            inplace_final_state=True,
                            cu_seqlens=spec_query_start_loc[
                                : attn_metadata.num_spec_decodes + 1
                            ],
                            ssm_state_indices=spec_state_indices_tensor,
                            num_accepted_tokens=spec_state_slot_selectors,
                            use_qk_l2norm_in_kernel=True,
                            out=(
                                core_attn_out[:num_actual_tokens].unsqueeze(0)
                                if direct_verify_out
                                else None
                            ),
                            schedule=self.gdn_policy.schedule,
                        )
                    )
                _log_runtime_route_once(
                    "SM70 mixed-QKV fused GDN target-verification route hit."
                )
            else:
                assert query_spec is not None
                assert key_spec is not None
                assert value_spec is not None
                # One fused launch instead of gating + recurrent update: the
                # kernel computes the sigmoid gating itself, so g/beta never
                # materialize and the surrounding elementwise work disappears.
                # Same routine the DFlash2 branch above already uses.
                core_attn_out_spec, _last_recurrent_state = (
                    fused_sigmoid_gating_delta_rule_update(
                        A_log=self.A_log,
                        a=a_spec,
                        b=b_spec,
                        dt_bias=self.dt_bias,
                        q=query_spec,
                        k=key_spec,
                        v=value_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                            : attn_metadata.num_spec_decodes
                            + 1  # type: ignore[attr-defined]
                        ],
                        ssm_state_indices=spec_state_indices_tensor,
                        num_accepted_tokens=spec_state_slot_selectors,
                        use_qk_l2norm_in_kernel=True,
                        schedule=self.gdn_policy.schedule,
                    )
                )
        else:
            core_attn_out_spec, _last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            prefill_core_attn_out = (
                core_attn_out
                if self.gdn_policy.direct_prefill_output
                and spec_sequence_masks is None
                and core_attn_out.shape[0] >= num_non_spec_tokens
                else None
            )
            core_attn_out_non_spec, _last_recurrent_state = (
                self.chunk_gated_delta_rule.execute_prefill(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                    ssm_state=ssm_state,
                    state_indices=non_spec_state_indices_tensor,
                    has_initial_state=has_initial_state,
                    cu_seqlens=non_spec_query_start_loc,
                    chunk_indices=attn_metadata.chunk_indices,
                    chunk_offsets=attn_metadata.chunk_offsets,
                    use_qk_l2norm_in_kernel=prefill_l2norm_in_kernel,
                    gate_is_exp=prefill_gate_is_exp,
                    core_attn_out=prefill_core_attn_out,
                    layer_name=layer_name,
                    num_tokens=num_non_spec_tokens,
                )
            )
        elif attn_metadata.num_decodes > 0:
            assert non_spec_state_indices_tensor is not None
            core_attn_out_non_spec, _last_recurrent_state = (
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
                    schedule=self.gdn_policy.schedule,
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
            core_attn_out_non_spec, _last_recurrent_state = None, None

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
            if (
                not use_dflash2_packed_gdn_verify
                and core_attn_out_spec.squeeze(0).data_ptr()
                != core_attn_out[:num_actual_tokens].data_ptr()
            ):
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
            kv_cache[0] if is_conv_state_dim_first() else kv_cache[0].transpose(-1, -2)
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
            non_spec_state_indices_tensor=(attn_metadata.non_spec_state_indices_tensor),
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
            kv_cache[0] if is_conv_state_dim_first() else kv_cache[0].transpose(-1, -2)
        )
        ssm_state = kv_cache[1]

        mixed_qkv = mixed_qkv[:num_decode_tokens]
        b = b[:num_decode_tokens]
        a = a[:num_decode_tokens]
        state_indices = non_spec_state_indices_tensor[:num_decode_tokens]
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
        mixed_qkv_non_spec = convolve_decode(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            state_indices=state_indices,  # type: ignore[arg-type]
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
        mixed_qkv_recurrence(
            self.gdn_heads,
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            mixed_qkv=mixed_qkv_non_spec,
            initial_state=ssm_state,
            out=out_buf,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,  # type: ignore[arg-type]
            schedule=self.gdn_policy.schedule,
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
                schedule=self.gdn_policy.schedule,
            )
            mixed_qkv_recurrence(
                self.gdn_heads,
                A_log=self.A_log,
                a=cmp_a,
                b=cmp_b,
                dt_bias=self.dt_bias,
                mixed_qkv=cmp_mixed_qkv,
                initial_state=mixed_state,
                out=mixed_out,
                cu_seqlens=local_cu_seqlens,
                state_indices=local_indices,
                schedule=self.gdn_policy.schedule,
            )
            ref_out, ref_state = mixed_qkv_recurrence(
                self.gdn_heads,
                A_log=self.A_log,
                a=cmp_a,
                b=cmp_b,
                dt_bias=self.dt_bias,
                mixed_qkv=cmp_mixed_qkv,
                initial_state=ref_state,
                inplace_final_state=True,
                cu_seqlens=local_cu_seqlens,
                state_indices=local_indices,
                schedule=self.gdn_policy.schedule,
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
    _sm70_assert_standard_core_not_active_spec(
        layer_name,
        attn_metadata,
        enabled=self.gdn_state_trace.assert_standard_boundary,
        spec_core=self.gdn_policy.state.spec_core,
    )
    if conv_state_cache.numel() == 0 and ssm_state_cache.numel() == 0:
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None and kv_cache[0].numel() > 0:
            conv_state_cache, ssm_state_cache = kv_cache[0], kv_cache[1]
    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, qkv_or_qkvz, "core_generic")
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b_or_ba, "core_generic")
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a_or_z_out, "core_generic")
    restore_query_start_loc = False
    restore_state_indices = False
    if attn_metadata is not None and non_spec_query_start_loc.numel() > 0:
        old_non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        attn_metadata.non_spec_query_start_loc = non_spec_query_start_loc
        restore_query_start_loc = True
    else:
        old_non_spec_query_start_loc = None
    if attn_metadata is not None and non_spec_state_indices_tensor.numel() > 0:
        old_non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        attn_metadata.non_spec_state_indices_tensor = non_spec_state_indices_tensor
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
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out, "core_generic")


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
    spec_state_slot_selectors: torch.Tensor,
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
    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, mixed_qkv, "core_standard_spec")
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b, "core_standard_spec")
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a, "core_standard_spec")
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
        spec_state_slot_selectors=spec_state_slot_selectors,
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
        _sm70_dump_gdn_core_tensor(
            "core_out", layer_name, core_attn_out, "core_standard_spec"
        )
        return

    with GdnMetadataOverride(attn_metadata, skip_empty=True) as view:
        view.set("non_spec_query_start_loc", non_spec_query_start_loc)
        view.set("non_spec_state_indices_tensor", non_spec_state_indices_tensor)
        view.set("spec_query_start_loc", spec_query_start_loc)
        view.set("spec_state_indices_tensor", spec_state_indices_tensor)
        view.set("spec_token_indx", spec_token_indx)
        view.set("non_spec_token_indx", non_spec_token_indx)
        view.set("spec_sequence_masks", spec_sequence_masks)
        view.set("num_accepted_tokens", num_accepted_tokens)
        if spec_state_slot_selectors.numel() == 0:
            spec_state_slot_selectors = num_accepted_tokens
        view.set("spec_state_slot_selectors", spec_state_slot_selectors)
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            kv_cache=(conv_state_cache, ssm_state_cache),
        )
    _sm70_dump_gdn_core_tensor(
        "core_out", layer_name, core_attn_out, "core_standard_spec"
    )


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
    spec_state_slot_selectors: torch.Tensor,
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
    fallback_to_standard = (
        attn_metadata is None
        or attn_metadata.num_spec_decodes <= 0
        or attn_metadata.spec_sequence_masks is None
    )
    if fallback_to_standard:
        with GdnMetadataOverride(attn_metadata) as view:
            if (
                attn_metadata is not None
                and attn_metadata.num_spec_decodes <= 0
                and attn_metadata.spec_sequence_masks is not None
            ):
                view.set("spec_sequence_masks", None)
                view.set("spec_token_indx", None)
                view.set("non_spec_token_indx", None)
                view.set("spec_query_start_loc", None)
                view.set("spec_state_indices_tensor", None)
                view.set("num_accepted_tokens", None)
            qwen_gdn_attention_core_standard(
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
    assert attn_metadata is not None
    pure_spec_decode = (
        attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0
    )
    metadata_source = "core_spec_commit"
    missing_core_spec_tensors = (
        spec_query_start_loc.numel() == 0
        or spec_state_indices_tensor.numel() == 0
        or num_accepted_tokens.numel() == 0
    )
    missing_mixed_spec_tensors = (not pure_spec_decode) and (
        spec_token_indx.numel() == 0 or spec_sequence_masks.numel() == 0
    )
    if missing_core_spec_tensors or missing_mixed_spec_tensors:
        metadata_tensors = gdn_spec_metadata_tensors(attn_metadata, mixed_qkv.device)
        metadata_missing_core_spec_tensors = (
            metadata_tensors[2].numel() == 0
            or metadata_tensors[3].numel() == 0
            or metadata_tensors[7].numel() == 0
        )
        metadata_missing_mixed_spec_tensors = (not pure_spec_decode) and (
            metadata_tensors[4].numel() == 0 or metadata_tensors[6].numel() == 0
        )
        if (
            not metadata_missing_core_spec_tensors
            and not metadata_missing_mixed_spec_tensors
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
                spec_state_slot_selectors,
            ) = metadata_tensors
            missing_core_spec_tensors = False
            missing_mixed_spec_tensors = False
            metadata_source = "core_spec_commit_metadata_fallback"
    if missing_core_spec_tensors or missing_mixed_spec_tensors:
        raise RuntimeError(
            "qwen_gdn_attention_core_spec_commit missing graph-visible "
            "spec metadata tensors "
            f"(pure_spec_decode={pure_spec_decode}, "
            f"spec_query_start_loc_shape={tuple(spec_query_start_loc.shape)}, "
            "spec_state_indices_tensor_shape="
            f"{tuple(spec_state_indices_tensor.shape)}, "
            f"spec_token_indx_shape={tuple(spec_token_indx.shape)}, "
            f"spec_sequence_masks_shape={tuple(spec_sequence_masks.shape)}, "
            f"num_accepted_tokens_shape={tuple(num_accepted_tokens.shape)}, "
            f"num_prefills={attn_metadata.num_prefills}, "
            f"num_decodes={attn_metadata.num_decodes}, "
            f"num_spec_decodes={attn_metadata.num_spec_decodes}, "
            f"num_spec_decode_tokens={attn_metadata.num_spec_decode_tokens})"
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

    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, mixed_qkv, metadata_source)
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b, metadata_source)
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a, metadata_source)
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
        spec_state_slot_selectors=spec_state_slot_selectors,
    )
    if spec_state_slot_selectors.numel() == 0:
        spec_state_slot_selectors = num_accepted_tokens
    ddtree_parent_ids = attn_metadata.ddtree_parent_ids
    ddtree_num_tree_tokens_cpu = attn_metadata.ddtree_num_tree_tokens_cpu
    ddtree_requires_branch = _ddtree_parent_ids_require_branch(
        ddtree_parent_ids,
        ddtree_num_tree_tokens_cpu,
        attn_metadata.num_spec_decodes,
    )
    if ddtree_requires_branch and not pure_spec_decode:
        raise RuntimeError(
            "DDTree branched Qwen GDN state replay currently supports only "
            "pure speculative verifier batches."
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
        if ddtree_requires_branch:
            assert ddtree_parent_ids is not None
            assert ddtree_num_tree_tokens_cpu is not None
            core_attn_out_spec = self._forward_ddtree_gdn_pure_spec(
                mixed_qkv_spec=mixed_qkv,
                a_spec=a,
                b_spec=b,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv_weights=conv_weights,
                spec_query_start_loc=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ],
                spec_state_indices_tensor=spec_state_indices_tensor[
                    : attn_metadata.num_spec_decodes
                ],
                spec_state_slot_selectors=spec_state_slot_selectors[
                    : attn_metadata.num_spec_decodes
                ],
                ddtree_parent_ids=ddtree_parent_ids[: attn_metadata.num_spec_decodes],
                ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu[
                    : attn_metadata.num_spec_decodes
                ],
                num_spec_decodes=attn_metadata.num_spec_decodes,
            )
            core_attn_out[: mixed_qkv.shape[0]] = core_attn_out_spec.squeeze(0)
            _sm70_dump_gdn_core_tensor(
                "core_out", layer_name, core_attn_out, metadata_source
            )
            return core_attn_out
        precomputed_gating = None
        if self._can_use_sm70_gdn_preprocess(
            mixed_qkv, a, b, conv_state, attn_metadata
        ):
            assert spec_state_indices_tensor is not None
            assert spec_state_slot_selectors is not None
            assert spec_query_start_loc is not None
            from vllm.model_executor.layers.mamba.gdn.sm70_preprocess import (
                conv_gate_zero,
            )

            mixed_qkv_spec, g, beta = conv_gate_zero(
                mixed_qkv,
                conv_state,
                conv_weights,
                spec_state_indices_tensor[:1, 0],
                spec_state_slot_selectors[:1],
                spec_query_start_loc[:2],
                self.A_log,
                a,
                b,
                self.dt_bias,
                core_attn_out,
            )
            precomputed_gating = (g, beta)
            _log_runtime_route_once("SM70 DFlash2 conv/gating/zero route hit.")
        else:
            mixed_qkv_spec = convolve_decode(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                state_indices=spec_state_indices_tensor[:, 0],
                num_accepted_tokens=spec_state_slot_selectors,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
        if isinstance(self, QwenGatedDeltaNetAttention) and (
            self._can_use_dflash2_packed_gdn_verify(
                mixed_qkv=mixed_qkv_spec,
                a=a,
                b=b,
                core_attn_out=core_attn_out,
                ssm_state=ssm_state,
                attn_metadata=attn_metadata,
            )
        ):
            self._forward_dflash2_packed_gdn_verify(
                mixed_qkv=mixed_qkv_spec,
                a=a,
                b=b,
                core_attn_out=core_attn_out,
                ssm_state=ssm_state,
                spec_query_start_loc=spec_query_start_loc,
                spec_state_indices_tensor=spec_state_indices_tensor,
                spec_state_slot_selectors=spec_state_slot_selectors,
                num_spec_decodes=attn_metadata.num_spec_decodes,
                precomputed_gating=precomputed_gating,
            )
            _log_runtime_route_once(
                "SM70 DFlash2 packed GDN target-verification route hit."
            )
            _sm70_dump_gdn_core_tensor(
                "core_out", layer_name, core_attn_out, metadata_source
            )
            return core_attn_out
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        g_spec, beta_spec = fused_gdn_gating(
            self.A_log,
            a,
            b,
            self.dt_bias,
            beta_dtype=torch.float32,
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
            num_accepted_tokens=spec_state_slot_selectors,
            use_qk_l2norm_in_kernel=True,
            schedule=self.gdn_policy.schedule,
        )
        core_attn_out[: mixed_qkv.shape[0]] = core_attn_out_spec.squeeze(0)
        _sm70_dump_gdn_core_tensor(
            "core_out", layer_name, core_attn_out, metadata_source
        )
        return core_attn_out

    with GdnMetadataOverride(attn_metadata) as view:
        view.set("non_spec_query_start_loc", non_spec_query_start_loc)
        view.set("non_spec_state_indices_tensor", non_spec_state_indices_tensor)
        view.set("spec_query_start_loc", spec_query_start_loc)
        view.set("spec_state_indices_tensor", spec_state_indices_tensor)
        view.set("spec_token_indx", spec_token_indx)
        view.set("non_spec_token_indx", non_spec_token_indx)
        view.set("spec_sequence_masks", spec_sequence_masks)
        view.set("num_accepted_tokens", num_accepted_tokens)
        view.set("spec_state_slot_selectors", spec_state_slot_selectors)
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            # The tensor rows are the graph-visible contract for FULL graph replay.
            # They include PAD_SLOT_ID rows with zero-length query ranges, so the
            # conv/recurrent kernels can skip padded rows without relying on the
            # capture-time Python num_spec_decodes scalar.
            view.set(
                "num_spec_decodes",
                int(spec_state_indices_tensor.shape[0]),
            )
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            kv_cache=(conv_state_cache, ssm_state_cache),
        )
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out, metadata_source)
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
    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, mixed_qkv, "core_standard")
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b, "core_standard")
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a, "core_standard")
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
        _sm70_dump_gdn_core_tensor(
            "core_out", layer_name, core_attn_out, "core_standard"
        )
        return

    with GdnMetadataOverride(attn_metadata, skip_empty=True) as view:
        view.set("non_spec_query_start_loc", non_spec_query_start_loc)
        view.set("non_spec_state_indices_tensor", non_spec_state_indices_tensor)
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            kv_cache=(conv_state_cache, ssm_state_cache),
        )
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out, "core_standard")


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
    _sm70_dump_gdn_core_tensor("input_qkv", layer_name, mixed_qkv, "core_003_spec")
    _sm70_dump_gdn_core_tensor("input_b", layer_name, b, "core_003_spec")
    _sm70_dump_gdn_core_tensor("input_a", layer_name, a, "core_003_spec")
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        kv_cache=(conv_state_cache, ssm_state_cache),
    )
    _sm70_dump_gdn_core_tensor("core_out", layer_name, core_attn_out, "core_003_spec")
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
    spec_state_slot_selectors: torch.Tensor,
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
    spec_state_slot_selectors: torch.Tensor,
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


def qwen_gdn_full_forward_direct(
    hidden_states: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Keep the full-forward order while returning its projection allocation."""
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    # Match the original opaque op's explicit recurrent-state dependencies.
    # Its eager body accesses these same caches through the layer object.
    _ = conv_state_cache, ssm_state_cache
    output = layer._full_forward(hidden_states, None)
    if output is None:
        raise RuntimeError("Direct GDN full-forward did not return a projection")
    _log_runtime_route_once("SM70 Qwen GDN direct full-forward output route hit.")
    return output


def qwen_gdn_full_forward_direct_fake(
    hidden_states: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


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


def _sm70_gdn_qpn8_ba_weight_contract(
    self: "QwenGatedDeltaNetAttention",
) -> bool:
    """Check the loaded operator layout without using model identity."""
    qkvz = self.in_proj_qkvz
    ba = self.in_proj_ba
    qkvz_weight = getattr(qkvz, "weight", None)
    qkvz_scales = getattr(qkvz, "weight_scale_inv", None)
    ba_weight = getattr(ba, "weight", None) if ba is not None else None
    input_width = (
        int(qkvz_weight.shape[0]) if isinstance(qkvz_weight, torch.Tensor) else 0
    )
    return bool(
        self.enable_sm70_gdn_qpn8_ba_split
        and ba is not None
        and getattr(qkvz, "sm70_fp8_qpn8", False)
        and int(getattr(qkvz, "sm70_fp8_prefill_exact_dense_workspace_ptr", 0)) > 0
        and isinstance(qkvz_weight, torch.Tensor)
        and qkvz_weight.dtype == torch.uint8
        and input_width in (2560, 5120)
        and qkvz_weight.shape == (input_width, 4096)
        and qkvz_weight.is_contiguous()
        and isinstance(qkvz_scales, torch.Tensor)
        and qkvz_scales.dtype == torch.float16
        and qkvz_scales.shape == (1, 4096)
        and qkvz_scales.is_contiguous()
        and isinstance(ba_weight, torch.Tensor)
        and ba_weight.dtype == torch.float16
        and ba_weight.shape == (24, input_width)
        and ba_weight.is_contiguous()
        and qkvz_weight.device == qkvz_scales.device == ba_weight.device
        and getattr(qkvz, "bias", None) is None
        and getattr(ba, "bias", None) is None
    )


def _sm70_gdn_qpn8_ba_dispatch_eligible(
    self: "QwenGatedDeltaNetAttention",
    hidden_states: torch.Tensor,
    layer_name: LayerNameType,
) -> bool:
    qkvz = self.in_proj_qkvz
    ba = self.in_proj_ba
    return bool(
        _sm70_gdn_qpn8_ba_weight_contract(self)
        and ba is not None
        and not _sm70_gdn_projection_dump_requested(layer_name)
        and hasattr(torch.ops._C, "fp8_qpn8_dispatch_ba_split_sm70_out")
        and hasattr(torch.ops._C, "fp8_qpn8_gemm_ba_split_sm70_out")
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.float16
        and hidden_states.ndim == 2
        and hidden_states.shape[1] == self.in_proj_qkvz.weight.shape[0]
        and hidden_states.is_contiguous()
        and hidden_states.device == qkvz.weight.device == ba.weight.device
    )


def _sm70_gdn_qpn8_ba_split_eligible(
    self: "QwenGatedDeltaNetAttention",
    hidden_states: torch.Tensor,
    z_out: torch.Tensor,
    layer_name: LayerNameType,
) -> bool:
    return (
        _sm70_gdn_qpn8_ba_dispatch_eligible(self, hidden_states, layer_name)
        and hidden_states.shape == (1, 5120)
        and z_out.is_cuda
        and z_out.device == hidden_states.device
        and z_out.dtype == hidden_states.dtype
        and z_out.shape == (1, 12, 128)
        and z_out.is_contiguous()
    )


def qwen_gdn_input_projection_core(
    hidden_states: torch.Tensor,
    z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Qwen GDN input projection and recurrent core outside Inductor."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    hidden_states = _sm70_dump_gdn_projection_tensor(
        "gdn_hidden_states_input_core",
        layer_name,
        hidden_states,
    )
    from vllm.model_executor.layers.quantization.sm70_gdn_ba_verify import (
        apply_gdn_ba_verify,
    )

    fused_verify_projection = None
    if hidden_states.shape == (8, 5120) and not _sm70_gdn_projection_dump_requested(
        layer_name
    ):
        fused_verify_projection = apply_gdn_ba_verify(
            self,
            hidden_states,
            outputs=(
                hidden_states.new_empty((8, 2560)),
                z_out.view(8, 1536) if z_out.shape == (8, 12, 128) else z_out,
                hidden_states.new_empty((8, 12)),
                hidden_states.new_empty((8, 12)),
            ),
        )
    if fused_verify_projection is not None:
        mixed_qkv, _, b, a = fused_verify_projection
    elif _sm70_gdn_qpn8_ba_split_eligible(
        self,
        hidden_states,
        z_out,
        layer_name,
    ):
        mixed_qkv = torch.empty(
            (1, 2560), dtype=hidden_states.dtype, device=hidden_states.device
        )
        b = torch.empty((1, 12), dtype=hidden_states.dtype, device=hidden_states.device)
        a = torch.empty_like(b)
        sm70_ops.fp8_qpn8_gemm_ba_split_sm70_out(
            mixed_qkv,
            z_out,
            b,
            a,
            hidden_states,
            self.in_proj_qkvz.weight,
            self.in_proj_qkvz.weight_scale_inv,
            self.in_proj_ba.weight,
        )
        _log_runtime_route_once(
            "SM70 GDN QPN8 N4096 plus FP16 b/a N24 split route enabled."
        )
        z = z_out
    else:
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        mixed_qkvz = _sm70_dump_gdn_projection_tensor(
            "input_core_in_proj_qkvz",
            layer_name,
            mixed_qkvz,
        )
        ba = _sm70_dump_gdn_projection_tensor("input_core_in_proj_ba", layer_name, ba)

        from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
            _can_fuse_gdn_projection_split,
            _split_gdn_projection_tails,
        )

        copied_tails = (
            not self.gqa_interleaved_layout
            and not self.disable_tp_for_ba_proj
            and mixed_qkvz.shape[0] in (5, 20)
            and (self.key_dim * 2 + self.value_dim) // self.tp_size == 2560
            and self.value_dim // self.tp_size == 1536
            and z_out.shape == (mixed_qkvz.shape[0], 12, 128)
            and z_out.dtype == mixed_qkvz.dtype
            and z_out.is_contiguous()
            and z_out.device == mixed_qkvz.device
            and _can_fuse_gdn_projection_split(mixed_qkvz, ba)
        )
        if copied_tails:
            mixed_qkv, b, a = _split_gdn_projection_tails(mixed_qkvz, ba, z_out)
            if self.gdn_policy.projection.mixed_qkv_contiguous:
                mixed_qkv = mixed_qkv.contiguous()
            _log_runtime_route_once("SM70 GDN projection tail-copy route hit.")
        elif self.gqa_interleaved_layout:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz,
                ba,
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"),
                (query, key, value),
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv = mixed_qkvz[..., :qkv_size]
            if self.gdn_policy.projection.mixed_qkv_contiguous:
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

        if not copied_tails:
            if (
                self.gdn_policy.projection.z_contiguous
                and current_platform.is_device_capability(70)
            ):
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
    return z_out, core_attn_out


def qwen_gdn_input_projection_core_fake(
    hidden_states: torch.Tensor,
    z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile."""
    return z_out, core_attn_out


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
    from vllm.model_executor.layers.quantization.sm70_gdn_ba_verify import (
        apply_gdn_ba_verify,
    )

    if hidden_states.shape == (8, 5120) and not _sm70_gdn_projection_dump_requested(
        layer_name
    ):
        fused_verify_projection = apply_gdn_ba_verify(
            self,
            hidden_states,
            outputs=(
                mixed_qkv_out,
                z_out.view(8, 1536) if z_out.shape == (8, 12, 128) else z_out,
                b_out,
                a_out,
            ),
        )
        if fused_verify_projection is not None:
            return
    if _sm70_gdn_qpn8_ba_split_eligible(
        self,
        hidden_states,
        z_out,
        layer_name,
    ):
        sm70_ops.fp8_qpn8_gemm_ba_split_sm70_out(
            mixed_qkv_out,
            z_out,
            b_out,
            a_out,
            hidden_states,
            self.in_proj_qkvz.weight,
            self.in_proj_qkvz.weight_scale_inv,
            self.in_proj_ba.weight,
        )
        _log_runtime_route_once(
            "SM70 GDN QPN8 N4096 plus FP16 b/a N24 split route enabled."
        )
        return

    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    assert self.in_proj_ba is not None
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

    if (
        self.gdn_policy.projection.z_contiguous
        and current_platform.is_device_capability(70)
    ):
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
    op_name="qwen_gdn_full_forward_direct",
    op_func=qwen_gdn_full_forward_direct,
    mutates_args=["conv_state_cache", "ssm_state_cache"],
    fake_impl=qwen_gdn_full_forward_direct_fake,
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
