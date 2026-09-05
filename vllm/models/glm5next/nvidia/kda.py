# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate."""

import os

import torch
from torch import nn

import vllm._sm70_ops as sm70_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops.kda import (
    FusedRMSNormGated,
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import (
    maybe_disable_graph_partition,
    set_weight_attrs,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

logger = init_logger(__name__)

_DEBUG_DFLASH_TARGET_FINITE = bool(
    int(os.getenv("VLLM_DFLASH_DEBUG_PROPOSAL_STAGES", "0"))
)
_DEBUG_DFLASH_TARGET_TRACE = bool(
    int(os.getenv("VLLM_DFLASH_DEBUG_TARGET_LAYER_TRACE", "0"))
)
_DFLASH_KDA_TRACE_SEEN: set[tuple[str, str]] = set()
_DFLASH_KDA_TRACE_ARMED_PREFIXES: set[str] = set()
_DFLASH_KDA_TRACE_TOKEN_INDEX: dict[str, int] = {}


def arm_dflash_target_kda_trace(prefix: str, token_index: int) -> None:
    if _DEBUG_DFLASH_TARGET_TRACE:
        _DFLASH_KDA_TRACE_ARMED_PREFIXES.add(prefix)
        _DFLASH_KDA_TRACE_TOKEN_INDEX[prefix] = token_index


def _debug_dflash_kda_finite(
    prefix: str, stage: str, tensor: torch.Tensor | None
) -> None:
    if (
        not _DEBUG_DFLASH_TARGET_FINITE
        or tensor is None
        or tensor.shape[-2 if tensor.ndim >= 2 else 0] <= 1
    ):
        return
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    finite_values = tensor[finite]
    finite_max = (
        float(finite_values.abs().max().item()) if finite_values.numel() else 0.0
    )
    if tensor.ndim >= 2:
        token_dim = 1 if tensor.ndim >= 3 else 0
        moved = (~finite).movedim(token_dim, 0).reshape(tensor.shape[token_dim], -1)
        bad_by_token = moved.sum(dim=1).tolist()
    else:
        bad_by_token = [int((~finite).sum().item())]
    logger.error(
        "DFlash target KDA nonfinite: prefix=%s stage=%s shape=%s "
        "nonfinite=%d nan=%d posinf=%d neginf=%d finite_max=%s "
        "bad_by_token=%s",
        prefix,
        stage,
        tuple(tensor.shape),
        int((~finite).sum().item()),
        int(torch.isnan(tensor).sum().item()),
        int(torch.isposinf(tensor).sum().item()),
        int(torch.isneginf(tensor).sum().item()),
        finite_max,
        bad_by_token,
    )


def _debug_dflash_kda_trace(
    prefix: str,
    stage: str,
    tensor: torch.Tensor,
    token_dim: int,
    state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
) -> None:
    if (
        not _DEBUG_DFLASH_TARGET_TRACE
        or get_tensor_model_parallel_rank() != 0
        or prefix not in _DFLASH_KDA_TRACE_ARMED_PREFIXES
        or tensor.shape[token_dim] > 8
        or (state_indices is not None and not bool((state_indices >= 0).any().item()))
    ):
        return
    key = (prefix, stage)
    if key in _DFLASH_KDA_TRACE_SEEN:
        return
    _DFLASH_KDA_TRACE_SEEN.add(key)
    token_index = min(
        _DFLASH_KDA_TRACE_TOKEN_INDEX.get(prefix, 0), tensor.shape[token_dim] - 1
    )
    row = tensor.select(token_dim, token_index).detach().float().reshape(-1)
    state_row = (
        None
        if state_indices is None
        else state_indices[0].detach().cpu().reshape(-1).tolist()
    )
    accepted = (
        None
        if num_accepted_tokens is None
        else num_accepted_tokens.detach().cpu().reshape(-1).tolist()
    )
    logger.warning(
        "DFLASH_TARGET_KDA_TRACE prefix=%s stage=%s shape=%s "
        "state_row=%s accepted=%s sum=%.9g sqsum=%.9g absmax=%.9g sample=%s",
        prefix,
        stage,
        tuple(tensor.shape),
        state_row,
        accepted,
        float(row.sum().item()),
        float((row * row).sum().item()),
        float(row.abs().max().item()),
        row[:8].cpu().tolist(),
    )


def _debug_dflash_sequence_delta(
    prefix: str,
    stage: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    token_dim: int,
) -> None:
    if (
        not _DEBUG_DFLASH_TARGET_TRACE
        or get_tensor_model_parallel_rank() != 0
        or prefix not in _DFLASH_KDA_TRACE_ARMED_PREFIXES
    ):
        return
    key = (prefix, stage)
    if key in _DFLASH_KDA_TRACE_SEEN:
        return
    _DFLASH_KDA_TRACE_SEEN.add(key)
    delta = (actual - reference).detach().float().movedim(token_dim, 0)
    delta = delta.reshape(delta.shape[0], -1)
    logger.warning(
        "DFLASH_TARGET_KDA_SEQUENCE_DELTA prefix=%s stage=%s "
        "max_by_token=%s mean_by_token=%s sqsum_by_token=%s",
        prefix,
        stage,
        delta.abs().amax(dim=1).cpu().tolist(),
        delta.abs().mean(dim=1).cpu().tolist(),
        (delta * delta).sum(dim=1).cpu().tolist(),
    )


def _sm70_exact_kda_gemv_enabled() -> bool:
    return os.getenv("VLLM_SM70_GLM53_EXACT_KDA_GEMV", "1") != "0"


def _sm70_glm53_tp8_cublaslt_enabled() -> bool:
    return (
        os.getenv("VLLM_SM70_GLM53_TP8_CUBLASLT", "0") != "0"
        and torch.version.cuda == "12.8"
    )


def _sm70_glm53_tp8_fused_fg_b_enabled() -> bool:
    return os.getenv("VLLM_SM70_GLM53_TP8_FUSED_FG_B", "0") != "0"


class _Glm5NextMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with multiple replicated output shards.

    Extends K3's ``_KimiGDNMergedColumnParallelLinear`` to support two
    replicated shards (f_a, g_a) instead of one. Pre-multiplies each
    replicated entry's output_size by tp_size so the per-rank shard
    divides back to the full size, and forces tp_rank=0 during weight
    loading for replicated shards.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_ids: tuple[int, ...],
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = set(replicated_shard_ids)
        output_sizes = output_sizes.copy()
        for sid in self.replicated_shard_ids:
            output_sizes[sid] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def _cast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Fuse the fp32 cast + sigmoid into one Inductor kernel."""
    return x.float().sigmoid()


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    # Declared int (set in __init__ from config) so mypy doesn't see the
    # getattr-derived `Any | None` at the kernel call sites.
    head_dim: int
    num_heads: int
    conv_size: int

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        q_dtype, _, _, recurrent_dtype = MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )
        return q_dtype, recurrent_dtype

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # conv_state width must include num_spec so the spec-decode conv update
        # (causal_conv1d_update with num_accepted_tokens + max_query_len) can
        # slide the window across the draft-verify tokens without reading past
        # the allocated width. Matches qwen_gdn_linear_attn.get_state_shape.
        q_shape, k_shape, v_shape, recurrent_shape = (
            MambaStateShapeCalculator.kda_state_shape(
                self.tp_size,
                self.num_heads,
                self.head_dim,
                conv_kernel_size=self.conv_size,
                num_spec=self.num_spec,
            )
        )
        if is_conv_state_dim_first():
            merged_conv_shape = (
                q_shape[0] + k_shape[0] + v_shape[0],
                q_shape[1],
            )
        else:
            merged_conv_shape = (
                q_shape[0],
                q_shape[1] + k_shape[1] + v_shape[1],
            )
        return merged_conv_shape, recurrent_shape

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # KDA projections remain BF16 because fp8 checkpoints omit their scales.
        saved_quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config

        # Linear-attention head config: read the flattened top-level fields when
        # present (new schema); fall back to the legacy linear_attn_config dict
        # otherwise (shared base is also used by KimiLinearConfig). Narrow via
        # locals so the int-typed attrs are assigned a non-None value.
        head_dim = getattr(config, "linear_head_dim", None)
        num_heads = getattr(config, "linear_num_heads", None)
        conv_size = getattr(config, "linear_conv_kernel_dim", None)
        if head_dim is None or num_heads is None or conv_size is None:
            kda_config = config.linear_attn_config  # type: ignore[attr-defined]
            assert kda_config is not None, "linear_attn_config must be set"
            head_dim = kda_config["head_dim"]
            num_heads = kda_config["num_heads"]
            conv_size = kda_config["short_conv_kernel_size"]
        assert head_dim is not None
        assert num_heads is not None
        assert conv_size is not None
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.conv_size = conv_size
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.tp_size)
        use_sm70_tp4_fused_fg_b = (
            current_platform.is_cuda()
            and current_platform.get_device_capability() == (7, 0)
            and self.tp_size == 4
            and self.head_dim == 128
            and self.local_projection_size == 2048
        )
        use_sm70_tp8_fused_fg_b = (
            current_platform.is_cuda()
            and current_platform.get_device_capability() == (7, 0)
            and self.tp_size == 8
            and self.head_dim == 128
            and self.local_projection_size == 1024
            and _sm70_glm53_tp8_fused_fg_b_enabled()
        )
        self._use_sm70_fused_fg_b_decode = (
            use_sm70_tp4_fused_fg_b or use_sm70_tp8_fused_fg_b
        )
        self._use_sm70_exact_kda_gemv = (
            self._use_sm70_fused_fg_b_decode
            and self.hidden_size == 4096
            and _sm70_exact_kda_gemv_enabled()
        )
        self._use_sm70_fp32_recurrent_output = (
            current_platform.is_cuda()
            and current_platform.get_device_capability() == (7, 0)
            and self.head_dim == 128
        )
        self._use_sm70_glm53_tp8_cublaslt = (
            current_platform.is_cuda()
            and current_platform.get_device_capability() == (7, 0)
            and self.tp_size == 8
            and self.hidden_size == 4096
            and self.local_projection_size == 1024
            and _sm70_glm53_tp8_cublaslt_enabled()
        )

        # Merge q, k, v, b, f_a, g_a projections into one GEMM (6→1 launches).
        # Order matches checkpoint's fused_qkvbfg_a_proj convention.
        # Shards 4 (f_a) and 5 (g_a) are replicated across TP ranks.
        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                projection_size,  # q (shard 0)
                projection_size,  # k (shard 1)
                projection_size,  # v (shard 2)
                self.num_heads,  # b (shard 3)
                self.head_dim,  # f_a (shard 4, replicated)
                self.head_dim,  # g_a (shard 5, replicated)
            ],
            replicated_shard_ids=(4, 5),
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvbfg_a",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)
        # Lazily-built merged q|k|v conv weight (built on first forward, after
        # weights are loaded). See _forward.
        self._merged_conv_weight: torch.Tensor | None = None

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.in_proj_qkvbfg_a._sm70_glm53_tp8_cublaslt = (
            self._use_sm70_glm53_tp8_cublaslt
        )
        self.o_proj._sm70_glm53_tp8_cublaslt = self._use_sm70_glm53_tp8_cublaslt

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Checkpoints store A_log as 1-D; the model parameter is 4-D.
        def _a_log_weight_loader(param, loaded_weight):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        self.A_log.weight_loader = _a_log_weight_loader

        # GLM-5.3-Flash uses a bounded sigmoid gate instead of the default
        # unbounded softplus gate.
        linear_lower_bound = getattr(config, "linear_lower_bound", None)
        if linear_lower_bound is not None:
            self.kda_safe_gate = True
            self.kda_lower_bound = linear_lower_bound
        else:
            legacy = getattr(config, "linear_attn_config", None) or {}
            if legacy.get("safe_gate", True):
                self.kda_safe_gate = True
                self.kda_lower_bound = legacy.get("lower_bound", -5.0)
            else:
                self.kda_safe_gate = False
                self.kda_lower_bound = -5.0
        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        self._conv_state_dim_first = is_conv_state_dim_first()

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 separate GEMMs).
        weight = self.in_proj_qkvbfg_a.weight
        use_sm70_exact_gemv = (
            self._use_sm70_exact_kda_gemv
            and 1 <= num_tokens <= 8
            and hidden_states.dtype == torch.float16
            and weight.dtype == torch.float16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and weight.shape == (6416, 4096)
        )
        if use_sm70_exact_gemv:
            if not hasattr(torch.ops._C, "sm70_glm53_fp16_gemv_out"):
                raise RuntimeError(
                    "SM70 GLM KDA decode requires the exact native FP16 GEMV op. "
                    "Rebuild vLLM from source with CUDA arch 7.0."
                )
            projected = torch.empty(
                (num_tokens, 6416),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            sm70_ops.sm70_glm53_fp16_gemv_out(projected, hidden_states, weight)
            logger.info_once("SM70 GLM KDA exact FP16 B1-B8 projection path enabled.")
        else:
            projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        _debug_dflash_kda_finite(self.prefix, "projection", projected)
        _debug_dflash_kda_trace(
            self.prefix, "projection_full", projected, 0, None, None
        )
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )

        # Beta stays raw (bf16) here: the recurrent kernel sigmoids it in fp32
        # at load (SIGMOID_BETA), and only the chunked prefill path needs the
        # pre-computed fp32 sigmoid — computed lazily in _forward. Pure decode
        # / spec-verify steps then skip the _cast_sigmoid kernel and its fp32
        # intermediate entirely.
        beta = beta_raw.unsqueeze(0)
        if self._use_sm70_fused_fg_b_decode and 1 <= num_tokens <= 8:
            if not hasattr(torch.ops._C, "sm70_glm_kda_fg_b_out"):
                raise RuntimeError(
                    "SM70 GLM KDA decode requires the native fused f_b/g_b op. "
                    "Rebuild vLLM from source with CUDA arch 7.0."
                )
            g1 = torch.empty(
                (num_tokens, self.local_projection_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            g_proj_states = torch.empty_like(g1)
            sm70_ops.sm70_glm_kda_fg_b_out(
                g1,
                g_proj_states,
                f_a,
                g_a,
                self.f_b_proj.weight,
                self.g_b_proj.weight,
            )
            logger.info_once("SM70 GLM KDA fused B1-B8 f_b/g_b path enabled.")
        else:
            g1 = self.f_b_proj(f_a)[0]
            g_proj_states = self.g_b_proj(g_a)[0]
        _debug_dflash_kda_finite(self.prefix, "gates", g1)
        _debug_dflash_kda_finite(self.prefix, "output_gate", g_proj_states)
        _debug_dflash_kda_trace(self.prefix, "gates", g1, 0, None, None)
        _debug_dflash_kda_trace(
            self.prefix, "output_gate", g_proj_states, 0, None, None
        )
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
        g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)

        is_recurrent_step = num_tokens <= self.num_spec + 1
        attn_metadata_raw = get_forward_context().attn_metadata
        if isinstance(attn_metadata_raw, dict):
            layer_metadata = attn_metadata_raw.get(self.prefix)
            if isinstance(layer_metadata, GDNAttentionMetadata):
                is_recurrent_step = layer_metadata.num_prefills == 0
        core_output_dtype = (
            torch.float32
            if self._use_sm70_fp32_recurrent_output
            and hidden_states.dtype == torch.float16
            and is_recurrent_step
            else hidden_states.dtype
        )
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=core_output_dtype,
            device=hidden_states.device,
        )
        # Call the decorated eager break directly so host-side prefill branches
        # are not captured by PIECEWISE CUDA graphs.
        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        if core_output_dtype != hidden_states.dtype:
            core_attn_out = self.o_norm(
                core_attn_out,
                g2,
                out_dtype=hidden_states.dtype,
            )
            logger.info_once(
                "SM70 GLM KDA keeps recurrent output in FP32 through RMSNorm."
            )
        else:
            core_attn_out = self.o_norm(core_attn_out, g2)
        _debug_dflash_kda_trace(self.prefix, "normalized", core_attn_out, 1, None, None)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(1), -1)
        projected_output = self.o_proj(core_attn_out)[0]
        _debug_dflash_kda_trace(
            self.prefix, "output_projection", projected_output, 0, None, None
        )
        return projected_output

    @eager_break_during_capture
    def _forward(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            #     # V1 profile run
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        # Spec-decode metadata (all None when speculative decoding is disabled).
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = spec_sequence_masks is not None and num_spec_decodes > 0
        # Safe-gate checkpoints use the bounded sigmoid variant.
        safe_gate = self.kda_safe_gate
        lower_bound = self.kda_lower_bound
        constant_caches = self.kv_cache

        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        # Layout is process-global and resolved once at init (see __init__).
        if not self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)

        trace_state_indices = (
            spec_state_indices_tensor if use_spec else non_spec_state_indices_tensor
        )
        if trace_state_indices is not None and attn_metadata_narrowed.num_prefills == 0:
            _debug_dflash_kda_trace(
                self.prefix,
                "projection",
                qkv_proj_states,
                0,
                trace_state_indices,
                num_accepted_tokens,
            )

        # One merged short-conv over q|k|v instead of three separate calls. The
        # 1D conv is independent per channel, so concatenating q/k/v along the
        # channel dim and running a single causal_conv1d is bit-identical to
        # three calls. The merged weight is q|k|v conv weights concatenated;
        # built once and cached (params are fixed after load). conv_state is
        # already stored as the merged q|k|v state, so it is used directly.
        if self._merged_conv_weight is None:

            def _w(m):
                return m.weight.view(m.weight.size(0), m.weight.size(2))

            self._merged_conv_weight = torch.cat(
                [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                dim=0,
            ).contiguous()
        conv_weights = self._merged_conv_weight
        conv_bias = self.q_conv1d.bias

        # Split projections / gating into spec (draft-verify) and non-spec token
        # groups when speculative decoding is active. Spec tokens carry
        # num_spec+1 recurrent-state columns each and are advanced with
        # num_accepted_tokens for rejection-sampling rollback; non-spec tokens
        # are one-per-request. Mirrors olmo_gdn_linear_attn.py. Projections are
        # [n, *] (token dim 0); g1/beta are [1, n, h, d] (token dim 1).
        if use_spec:
            # In a pure spec-verify step (no non-spec tokens) the metadata
            # builder sets spec_token_indx = arange(num_actual_tokens), making
            # the index_select calls below identity copies. Skip them on this
            # steady-state decode hot path. The outputs alias the inputs here;
            # the downstream conv/recurrent kernels read them without mutating
            # in place, so the aliasing is safe.
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0:
                qkv_spec = qkv_proj_states
                g1_spec = g1
                beta_spec = beta
            else:
                qkv_spec = qkv_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
            if non_spec_token_indx is not None and non_spec_token_indx.numel() > 0:
                qkv_ns = qkv_proj_states.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
            else:
                qkv_ns = g1_ns = beta_ns = None
        else:
            qkv_spec = g1_spec = beta_spec = None
            qkv_ns, g1_ns, beta_ns = qkv_proj_states, g1, beta

        # --- causal conv1d: spec (draft-verify) path ---
        debug_spec_conv_reference = None
        if use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            conv_idx = spec_state_indices_tensor[:, 0][:num_spec_decodes]
            conv_mql = spec_state_indices_tensor.size(-1)
            debug_spec_sequence = (
                _DEBUG_DFLASH_TARGET_TRACE
                and get_tensor_model_parallel_rank() == 0
                and self.prefix in _DFLASH_KDA_TRACE_ARMED_PREFIXES
                and self.prefix.endswith(".layers.0.self_attn")
                and num_spec_decodes == 1
            )
            if debug_spec_sequence:
                assert spec_query_start_loc is not None
                debug_start = int(spec_query_start_loc[0].item())
                debug_end = int(spec_query_start_loc[1].item())
                debug_accepted_offset = int(num_accepted_tokens[0].item()) - 1
                debug_conv_width = conv_weights.shape[1] - 1
                debug_conv_state = conv_state[
                    int(conv_idx[0].item()) : int(conv_idx[0].item()) + 1,
                    :,
                    debug_accepted_offset : debug_accepted_offset + debug_conv_width,
                ].clone()
                debug_conv_indices = torch.zeros(
                    (1,), dtype=conv_idx.dtype, device=conv_idx.device
                )
                debug_conv_rows = []
                for token_idx in range(debug_start, debug_end):
                    debug_conv_rows.append(
                        causal_conv1d_update(
                            qkv_spec[token_idx : token_idx + 1].clone(),
                            debug_conv_state,
                            conv_weights,
                            conv_bias,
                            activation="silu",
                            conv_state_indices=debug_conv_indices,
                        )
                    )
                debug_spec_conv_reference = torch.cat(debug_conv_rows, dim=0)
            qkv_spec = causal_conv1d_update(
                qkv_spec,
                conv_state,
                conv_weights,
                conv_bias,
                activation="silu",
                conv_state_indices=conv_idx,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=conv_mql,
            )
            _debug_dflash_kda_trace(
                self.prefix,
                "spec_conv",
                qkv_spec,
                0,
                spec_state_indices_tensor,
                num_accepted_tokens,
            )
            if debug_spec_conv_reference is not None:
                _debug_dflash_sequence_delta(
                    self.prefix,
                    "spec_conv_sequential",
                    qkv_spec[debug_start:debug_end],
                    debug_spec_conv_reference,
                    0,
                )
            _debug_dflash_kda_finite(self.prefix, "spec_conv", qkv_spec)
            q_spec, k_spec, v_spec = qkv_spec.split(self.local_projection_size, dim=-1)

        # --- causal conv1d: non-spec path (prefill or plain decode) ---
        q_ns = k_ns = v_ns = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert qkv_ns is not None
            qkv_ns = causal_conv1d_fn(
                qkv_ns.transpose(0, 1),
                conv_weights,
                conv_bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_decodes
            ]
            qkv_ns = causal_conv1d_update(
                qkv_ns,
                conv_state,
                conv_weights,
                conv_bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
            )
            _debug_dflash_kda_trace(
                self.prefix,
                "decode_conv",
                qkv_ns,
                0,
                decode_conv_indices,
                None,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)

        def _rearr(x):
            return x.reshape(1, -1, self.local_num_heads, self.head_dim)

        # --- core attention: spec (draft-verify) path ---
        core_attn_out_spec = None
        # In a pure spec-verify step (no non-spec tokens) the recurrent kernel
        # can write straight into the layer output buffer, skipping the
        # fresh allocation + copy below. Mixed steps must scatter via
        # spec_token_indx, so they keep the kernel-managed output.
        spec_out = (
            core_attn_out[0, :num_actual_tokens].unsqueeze(0)
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0
            else None
        )
        debug_spec_b1_reference = None
        debug_spec_sequence_reference = None
        if use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            assert spec_query_start_loc is not None
            if debug_spec_sequence:
                accepted_offset = num_accepted_tokens[:1].to(torch.long) - 1
                state_slot = spec_state_indices_tensor[:1].gather(
                    1, accepted_offset.view(1, 1)
                )
                debug_initial_state = recurrent_state.index_select(
                    0, state_slot.reshape(-1).to(torch.long)
                ).clone()
                _debug_dflash_kda_trace(
                    self.prefix,
                    "spec_initial_state",
                    debug_initial_state,
                    0,
                    state_slot,
                    num_accepted_tokens,
                )
                debug_spec_b1_reference = torch.empty(
                    (1, 1, self.local_num_heads, self.head_dim),
                    dtype=core_attn_out.dtype,
                    device=core_attn_out.device,
                )
                debug_state_indices = torch.zeros(
                    (1,),
                    dtype=spec_state_indices_tensor.dtype,
                    device=spec_state_indices_tensor.device,
                )
                debug_cu_seqlens = torch.tensor(
                    [0, 1],
                    dtype=spec_query_start_loc.dtype,
                    device=spec_query_start_loc.device,
                )
                debug_b1_state = debug_initial_state.clone()
                fused_recurrent_kda(
                    q=_rearr(q_spec)[:, :1],
                    k=_rearr(k_spec)[:, :1],
                    v=_rearr(v_spec)[:, :1],
                    g=g1_spec[:, :1],
                    beta=beta_spec[:, :1],
                    initial_state=debug_b1_state,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=debug_cu_seqlens,
                    ssm_state_indices=debug_state_indices,
                    out=debug_spec_b1_reference,
                    sigmoid_beta=True,
                    a_log=self.A_log,
                    g_bias=self.dt_bias,
                    compute_gate=True,
                    lower_bound=lower_bound,
                )
                debug_spec_sequence_reference = torch.empty(
                    (
                        1,
                        debug_end - debug_start,
                        self.local_num_heads,
                        self.head_dim,
                    ),
                    dtype=core_attn_out.dtype,
                    device=core_attn_out.device,
                )
                debug_sequence_state = debug_initial_state.clone()
                for token_idx in range(debug_end - debug_start):
                    fused_recurrent_kda(
                        q=_rearr(q_spec)[
                            :, debug_start + token_idx : debug_start + token_idx + 1
                        ],
                        k=_rearr(k_spec)[
                            :, debug_start + token_idx : debug_start + token_idx + 1
                        ],
                        v=_rearr(v_spec)[
                            :, debug_start + token_idx : debug_start + token_idx + 1
                        ],
                        g=g1_spec[
                            :, debug_start + token_idx : debug_start + token_idx + 1
                        ],
                        beta=beta_spec[
                            :, debug_start + token_idx : debug_start + token_idx + 1
                        ],
                        initial_state=debug_sequence_state,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=debug_cu_seqlens,
                        ssm_state_indices=debug_state_indices,
                        out=debug_spec_sequence_reference[:, token_idx : token_idx + 1],
                        sigmoid_beta=True,
                        a_log=self.A_log,
                        g_bias=self.dt_bias,
                        compute_gate=True,
                        lower_bound=lower_bound,
                    )
            # Gate computed inside the recurrent kernel (COMPUTE_GATE) from
            # raw g1 — replicates fused_kda_gate's arithmetic bit-for-bit and
            # skips its launch + fp32 [n, H, D] intermediate per layer.
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=_rearr(q_spec),
                k=_rearr(k_spec),
                v=_rearr(v_spec),
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )
            _debug_dflash_kda_trace(
                self.prefix,
                "spec_recurrent",
                core_attn_out_spec,
                1,
                spec_state_indices_tensor,
                num_accepted_tokens,
            )
            if debug_spec_b1_reference is not None:
                _debug_dflash_kda_trace(
                    self.prefix,
                    "spec_b1_reference",
                    debug_spec_b1_reference,
                    1,
                    spec_state_indices_tensor,
                    num_accepted_tokens,
                )
            if debug_spec_sequence_reference is not None:
                _debug_dflash_sequence_delta(
                    self.prefix,
                    "spec_recurrent_sequential",
                    core_attn_out_spec[:, debug_start:debug_end],
                    debug_spec_sequence_reference,
                    1,
                )
                _debug_dflash_kda_trace(
                    self.prefix,
                    "spec_b1_delta",
                    core_attn_out_spec[:, :1] - debug_spec_b1_reference,
                    1,
                    spec_state_indices_tensor,
                    num_accepted_tokens,
                )
            if _DEBUG_DFLASH_TARGET_FINITE and not bool(
                torch.isfinite(core_attn_out_spec).all().item()
            ):
                state_rows = torch.arange(
                    num_spec_decodes,
                    dtype=torch.long,
                    device=spec_state_indices_tensor.device,
                )
                accepted_offsets = (
                    num_accepted_tokens[:num_spec_decodes].to(torch.long) - 1
                )
                state_slots = spec_state_indices_tensor[
                    state_rows, accepted_offsets
                ].to(torch.long)
                selected_state = recurrent_state.index_select(0, state_slots)
                logger.error(
                    "DFlash target KDA state diagnostic: prefix=%s "
                    "accepted=%s state_slots=%s state_finite=%s "
                    "state_max=%s q_max=%s k_max=%s v_max=%s g_max=%s "
                    "beta_max=%s",
                    self.prefix,
                    num_accepted_tokens[:num_spec_decodes].tolist(),
                    state_slots.tolist(),
                    bool(torch.isfinite(selected_state).all().item()),
                    float(selected_state.abs().max().item()),
                    float(q_spec.abs().max().item()),
                    float(k_spec.abs().max().item()),
                    float(v_spec.abs().max().item()),
                    float(g1_spec.abs().max().item()),
                    float(beta_spec.abs().max().item()),
                )
            _debug_dflash_kda_finite(self.prefix, "spec_recurrent", core_attn_out_spec)

        # --- core attention: non-spec path (prefill or plain decode) ---
        core_attn_out_non_spec = None
        # Only the plain-decode recurrent kernel can write straight into the
        # layer output buffer; the chunked prefill kernel cannot, so this
        # stays None there and the merge copy below runs as before.
        ns_out = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert q_ns is not None
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            initial_state = gather_initial_states(
                recurrent_state, non_spec_state_indices_tensor, has_initial_state
            )
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                raw_g=g1_ns,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_ns is raw bf16 from forward.
                beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
            # Init cache
            scatter_states(
                recurrent_state,
                last_recurrent_state,
                non_spec_state_indices_tensor,
            )
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            # Plain decode step (no spec tokens): token order is dense, so the
            # kernel can write straight into the layer output buffer. A mixed
            # step scatters non-spec output via non_spec_token_indx instead.
            # Gate computed in-kernel (COMPUTE_GATE), beta sigmoided in-kernel.
            if not use_spec:
                ns_out = spec_out
            core_attn_out_non_spec, _ = fused_recurrent_kda(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                g=g1_ns,
                beta=beta_ns,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata_narrowed.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                out=ns_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )
            _debug_dflash_kda_trace(
                self.prefix,
                "decode_recurrent",
                core_attn_out_non_spec,
                1,
                non_spec_state_indices_tensor,
                None,
            )

        # --- merge spec / non-spec outputs back into token order ---
        if use_spec and core_attn_out_non_spec is not None:
            assert core_attn_out_spec is not None
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged.squeeze(0)
        elif use_spec:
            assert core_attn_out_spec is not None
            if spec_out is None:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            assert core_attn_out_non_spec is not None
            if ns_out is None:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                    0, :num_actual_tokens
                ]
