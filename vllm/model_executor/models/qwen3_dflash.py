# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


def _get_dflash_per_layer_sliding_window(
    config: Qwen3Config,
    layer_idx: int,
) -> int | None:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return None
    layer_type = layer_types[layer_idx]
    if layer_type == "sliding_attention":
        return getattr(config, "sliding_window", None)
    if layer_type == "full_attention":
        return None
    raise ValueError(f"Invalid DFlash layer_type {layer_type}")


def _debug_sync_context_kv_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_SYNC_CONTEXT_KV", "0") == "1"


def _skip_context_kv_precompute_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_SKIP_CONTEXT_KV_PRECOMPUTE", "0") == "1"


def _dflash_profile_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_PROFILE", "0") == "1"


def _dflash_profile_log_interval() -> int:
    return max(1, int(os.getenv("VLLM_DFLASH_PROFILE_LOG_INTERVAL", "32")))


def _debug_context_kv_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DEBUG_CONTEXT_KV", "0") == "1"


def _dump_draft_layer_hiddens_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DUMP_LAYER_HIDDENS", "0") == "1"


def _dump_draft_layer0_components_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DUMP_LAYER0_COMPONENTS", "0") == "1"


def _dump_draft_attn_components_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DUMP_ATTN_COMPONENTS", "0") == "1"


class DFlashQwen3Attention(nn.Module):
    """Attention for DFlash speculative decoding.

    Context KVs are pre-inserted into the KV cache before the forward pass.
    This layer handles only query tokens via standard attention.
    Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        self.per_layer_sliding_window = per_layer_sliding_window
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,  # DFlash has o_proj bias when using attention bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            per_layer_sliding_window=per_layer_sliding_window,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self._attn_components_dumped = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This forward op
        computes attention for the query tokens only.
        See also: precompute_and_store_context_kv"""
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        if (
            _dump_draft_attn_components_enabled()
            and not self._attn_components_dumped
            and positions.shape[-1] == 17
            and int(positions.max().item()) > 0
            and ".layers.32.self_attn" in self.layer_name
        ):
            dump_path = f"/tmp/dflash_draft_attn_components_pid{os.getpid()}.pt"
            torch.save(
                {
                    "positions": positions.detach().cpu(),
                    "q": q.detach().to(torch.float16).cpu(),
                    "k": k.detach().to(torch.float16).cpu(),
                    "v": v.detach().to(torch.float16).cpu(),
                    "attn_output": attn_output.detach().to(torch.float16).cpu(),
                    "o_proj_output": output.detach().to(torch.float16).cpu(),
                },
                dump_path,
            )
            self._attn_components_dumped = True
            logger.warning("Saved DFlash draft attn components to %s", dump_path)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_idx: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER
        per_layer_sliding_window = _get_dflash_per_layer_sliding_window(
            config, layer_idx
        )

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            per_layer_sliding_window=per_layer_sliding_window,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer0_components_dumped = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        ln_out = hidden_states

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        attn_out = hidden_states

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        post_attn_ln_out = hidden_states
        hidden_states = self.mlp(hidden_states)
        if (
            _dump_draft_layer0_components_enabled()
            and not self._layer0_components_dumped
            and positions.shape[-1] == 17
            and int(positions.max().item()) > 0
            and ".layers.32.self_attn" in self.self_attn.layer_name
        ):
            mlp_gate_up, _ = self.mlp.gate_up_proj(post_attn_ln_out)
            mlp_act_out = self.mlp.act_fn(mlp_gate_up)
            dump_path = f"/tmp/dflash_draft_layer0_components_pid{os.getpid()}.pt"
            torch.save(
                {
                    "positions": positions.detach().cpu(),
                    "input_layernorm_out": ln_out.detach().to(torch.float16).cpu(),
                    "self_attn_out": attn_out.detach().to(torch.float16).cpu(),
                    "post_attention_layernorm_out": post_attn_ln_out.detach()
                    .to(torch.float16)
                    .cpu(),
                    "mlp_gate_up": mlp_gate_up.detach().to(torch.float16).cpu(),
                    "mlp_act_out": mlp_act_out.detach().to(torch.float16).cpu(),
                    "mlp_out": hidden_states.detach().to(torch.float16).cpu(),
                    "residual": residual.detach().to(torch.float16).cpu(),
                },
                dump_path,
            )
            self._layer0_components_dumped = True
            logger.warning("Saved DFlash draft layer0 components to %s", dump_path)
        return hidden_states, residual


@support_torch_compile
class DFlashQwen3Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self._profile_context_kv_calls = 0
        self._profile_context_kv_tokens = 0
        self._profile_context_kv_elapsed_s = 0.0
        self._debug_context_kv_dumped = False
        self._draft_layer_hiddens_dumped = False

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights: list of [head_dim] tensors, one per layer.
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Precompute K/V for context states write them into each layer's KV cache.

        Input context states are projected to K/V, normed, and have RoPE applied.
        Since the context shape is different than the query shape, we can't rely on the
        regular forward pass to apply torch.compile and CUDA graphs to this section.
        As such, this function is optimized to minimize the number of torch ops present:
        we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
        large projection, and avoid cloning buffers (with .contiguous()) where possible.

        When context_slot_mapping is None (e.g. during dummy_run) only
        the computation runs, and no K/V is written to cache.
        """
        if _skip_context_kv_precompute_enabled():
            logger.warning("Skipping DFlash context KV precompute for debugging.")
            return

        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads
        start_event = None
        end_event = None
        start_time = None
        if _dflash_profile_enabled() and context_slot_mapping is not None:
            if context_states.is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

        # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            ops.rms_norm(
                all_k_normed[i],
                all_k[i],
                self._k_norm_weights[i],
                self._rms_norm_eps,
            )

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is not None:
            # --- Per-layer cache insert ---
            all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
            debug_sample_tokens = min(8, num_ctx)
            debug_ranges: list[tuple[str, int, int]] = [
                ("head", 0, debug_sample_tokens)
            ]
            if num_ctx > debug_sample_tokens:
                tail_start = max(0, num_ctx - debug_sample_tokens)
                if tail_start > 0:
                    debug_ranges.append(("tail", tail_start, num_ctx))
            debug_direct_refs: dict[
                int, list[tuple[str, int, torch.Tensor, torch.Tensor]]
            ] = {}
            if _debug_context_kv_enabled() and not self._debug_context_kv_dumped:
                debug_layers = sorted({0, L // 2, L - 1})
                debug_cos_sin_cache = self._rope_cos_sin_cache
                for layer_idx in debug_layers:
                    attn_layer = self.layers[layer_idx].self_attn
                    layer_refs: list[tuple[str, int, torch.Tensor, torch.Tensor]] = []
                    for sample_name, sample_start, sample_end in debug_ranges:
                        sample_tokens = sample_end - sample_start
                        direct_kv = F.linear(
                            normed_context_states[sample_start:sample_end],
                            attn_layer.qkv_proj.weight[attn_layer.q_size :],
                            (
                                None
                                if attn_layer.qkv_proj.bias is None
                                else attn_layer.qkv_proj.bias[attn_layer.q_size :]
                            ),
                        )
                        direct_k, direct_v = direct_kv.split([kv, kv], dim=-1)
                        direct_k = direct_k.view(sample_tokens, nkv, hd).contiguous()
                        direct_v = direct_v.view(sample_tokens, nkv, hd).contiguous()
                        direct_k_norm = torch.empty_like(direct_k)
                        ops.rms_norm(
                            direct_k_norm,
                            direct_k,
                            attn_layer.k_norm.weight.data,
                            self._rms_norm_eps,
                        )
                        direct_k_flat = direct_k_norm.view(sample_tokens, kv)
                        if debug_cos_sin_cache.dtype != direct_k_flat.dtype:
                            debug_cos_sin_cache = debug_cos_sin_cache.to(
                                dtype=direct_k_flat.dtype
                            )
                        ops.rotary_embedding(
                            context_positions[sample_start:sample_end],
                            direct_k_flat,
                            None,
                            self._rope_head_size,
                            debug_cos_sin_cache,
                            self._rope_is_neox,
                        )
                        direct_k = direct_k_flat.view(sample_tokens, nkv, hd)
                        layer_refs.append(
                            (sample_name, sample_start, direct_k, direct_v)
                        )
                        fused_k_diff = (
                            all_k_final[layer_idx, sample_start:sample_end] - direct_k
                        ).abs()
                        fused_v_diff = (
                            all_v[layer_idx, sample_start:sample_end] - direct_v
                        ).abs()
                        logger.warning(
                            "DFlash context KV fused-vs-direct layer=%d sample=%s "
                            "start=%d tokens=%d k_max=%.6f k_mean=%.6f "
                            "v_max=%.6f v_mean=%.6f",
                            layer_idx,
                            sample_name,
                            sample_start,
                            sample_tokens,
                            float(fused_k_diff.max().item()),
                            float(fused_k_diff.mean().item()),
                            float(fused_v_diff.max().item()),
                            float(fused_v_diff.mean().item()),
                        )
                    debug_direct_refs[layer_idx] = layer_refs
            for i in range(L):
                attn = self._attn_layers[i]
                layer_context_slot_mapping = context_slot_mapping
                if isinstance(context_slot_mapping, dict):
                    layer_context_slot_mapping = context_slot_mapping[attn.layer_name]
                assert layer_context_slot_mapping is not None
                kv_cache = attn.kv_cache
                if isinstance(kv_cache, list):
                    kv_cache = kv_cache[0]
                if _debug_sync_context_kv_enabled():
                    logger.warning(
                        "DFlash context KV update layer=%s kv_cache_shape=%s "
                        "slot_min=%s slot_max=%s num_ctx=%d",
                        attn.layer_name,
                        tuple(kv_cache.shape),
                        int(layer_context_slot_mapping.min().item()),
                        int(layer_context_slot_mapping.max().item()),
                        num_ctx,
                    )
                attn.impl.do_kv_cache_update(
                    attn,
                    all_k_final[i],
                    all_v[i],
                    kv_cache,
                    layer_context_slot_mapping,
                )
                if (
                    i in debug_direct_refs
                    and _debug_context_kv_enabled()
                    and not self._debug_context_kv_dumped
                ):
                    block_size = kv_cache.shape[2]
                    for (
                        sample_name,
                        sample_start,
                        debug_direct_k,
                        debug_direct_v,
                    ) in debug_direct_refs[i]:
                        sample_tokens = debug_direct_k.shape[0]
                        sample_end = sample_start + sample_tokens
                        slots = layer_context_slot_mapping[
                            sample_start:sample_end
                        ].to(torch.long)
                        block_idx = torch.div(
                            slots, block_size, rounding_mode="floor"
                        )
                        block_off = torch.remainder(slots, block_size)
                        cache_k = kv_cache[block_idx, 0, block_off]
                        cache_v = kv_cache[block_idx, 1, block_off]
                        cache_k_diff = (cache_k - debug_direct_k).abs()
                        cache_v_diff = (cache_v - debug_direct_v).abs()
                        logger.warning(
                            "DFlash context KV cache-readback layer=%d sample=%s "
                            "start=%d tokens=%d k_max=%.6f k_mean=%.6f "
                            "v_max=%.6f v_mean=%.6f",
                            i,
                            sample_name,
                            sample_start,
                            sample_tokens,
                            float(cache_k_diff.max().item()),
                            float(cache_k_diff.mean().item()),
                            float(cache_v_diff.max().item()),
                            float(cache_v_diff.mean().item()),
                        )
                    if i == max(debug_direct_refs):
                        self._debug_context_kv_dumped = True
                if _debug_sync_context_kv_enabled():
                    torch.cuda.synchronize(all_k_final.device)

        if _dflash_profile_enabled() and context_slot_mapping is not None:
            if start_event is not None and end_event is not None:
                end_event.record()
                end_event.synchronize()
                elapsed_s = start_event.elapsed_time(end_event) / 1000.0
            else:
                assert start_time is not None
                elapsed_s = time.perf_counter() - start_time
            self._profile_context_kv_calls += 1
            self._profile_context_kv_tokens += num_ctx
            self._profile_context_kv_elapsed_s += elapsed_s
            calls = self._profile_context_kv_calls
            if calls % _dflash_profile_log_interval() == 0:
                avg_tokens = self._profile_context_kv_tokens / calls
                avg_ms = self._profile_context_kv_elapsed_s * 1000.0 / calls
                per_token_us = (
                    self._profile_context_kv_elapsed_s
                    * 1_000_000.0
                    / self._profile_context_kv_tokens
                )
                logger.info(
                    "DFlash context KV profile: calls=%d avg_tokens=%.1f "
                    "avg_elapsed_ms=%.3f avg_per_token_us=%.3f",
                    calls,
                    avg_tokens,
                    avg_ms,
                    per_token_us,
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        layer_hidden_states: list[torch.Tensor] | None = None
        should_dump_layer_hiddens = (
            _dump_draft_layer_hiddens_enabled()
            and not self._draft_layer_hiddens_dumped
            and positions.shape[-1] == 17
            and int(positions.max().item()) > 0
        )
        if should_dump_layer_hiddens:
            layer_hidden_states = []
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if layer_hidden_states is not None:
                layer_hidden_states.append(
                    hidden_states.detach().to(torch.float16).cpu()
                )
        hidden_states, _ = self.norm(hidden_states, residual)
        if layer_hidden_states is not None:
            dump_path = f"/tmp/dflash_draft_layers_pid{os.getpid()}.pt"
            torch.save(
                {
                    "positions": positions.detach().cpu(),
                    "input_embeds": input_embeds.detach().to(torch.float16).cpu(),
                    "layer_hidden_states": layer_hidden_states,
                    "final_hidden_states": hidden_states.detach()
                    .to(torch.float16)
                    .cpu(),
                },
                dump_path,
            )
            self._draft_layer_hiddens_dumped = True
            logger.warning("Saved DFlash draft layer hiddens to %s", dump_path)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix="model",
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        # Keep the auxiliary hidden-state projection on the same dtype as the
        # DFlash FC weights. AWQ target paths may surface fp32 hidden states
        # even when the draft projection is initialized in fp16.
        fc_weight = getattr(self.model.fc, "weight", None)
        if fc_weight is not None and hidden_states.dtype != fc_weight.dtype:
            hidden_states = hidden_states.to(dtype=fc_weight.dtype)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
