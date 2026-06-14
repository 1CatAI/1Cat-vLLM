# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

from collections.abc import Iterable
from itertools import islice
import os
import time

import torch
from torch import nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.utils.torch_utils import direct_register_custom_op

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)
_SM70_QWEN_LAYER_DUMP_COUNTS: dict[str, int] = {}
_SM70_QWEN_LAYER_DUMP_SAVE_COUNTS: dict[str, int] = {}
_SM70_QWEN_LAYER_GRAPH_BUFFERS: dict[str, torch.Tensor] = {}
_SM70_QWEN_LAYER_GRAPH_META: dict[str, dict[str, object]] = {}


def _sm70_profile_trace_enabled() -> bool:
    return envs.VLLM_SM70_PROFILE_TRACE and not torch.compiler.is_compiling()


def _sm70_profile_trace(message: str, *args: object) -> None:
    if _sm70_profile_trace_enabled():
        if args:
            message = message % args
        logger.info("SM70 model trace: %s", message)


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


def _sm70_qwen_layer_dump_requested(layer_idx: int) -> bool:
    if not os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_DIR"):
        return False
    raw_layer_ids = os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_IDS", "0,1")
    if raw_layer_ids.strip().lower() in {"*", "all"}:
        return True
    try:
        layer_ids = _sm70_parse_int_ranges(raw_layer_ids) or {0, 1}
    except ValueError:
        layer_ids = {0, 1}
    return layer_idx in layer_ids


def _sm70_qwen_layer_dump_impl(
    tensor: torch.Tensor,
    label: str,
    layer_idx: int,
    layer_type: str,
) -> torch.Tensor:
    dump_dir = os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_DIR")
    graph_buffers = os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_GRAPH_BUFFERS") == "1"
    target_labels = {
        item.strip()
        for item in os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_LABELS", "").split(",")
        if item.strip()
    }
    if target_labels and label not in target_labels:
        return tensor
    if graph_buffers and dump_dir and tensor.is_cuda:
        shape = tuple(tensor.shape)
        key = f"{os.getpid()}:{layer_idx}:{label}:{shape}"
        buffer = _SM70_QWEN_LAYER_GRAPH_BUFFERS.get(key)
        if (
            buffer is None
            or tuple(buffer.shape) != shape
            or buffer.dtype != tensor.dtype
            or buffer.device != tensor.device
        ):
            buffer = torch.empty_like(tensor)
            _SM70_QWEN_LAYER_GRAPH_BUFFERS[key] = buffer
            _SM70_QWEN_LAYER_GRAPH_META[key] = {
                "label": label,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "shape": shape,
                "dtype": str(tensor.dtype),
                "pid": os.getpid(),
            }
        buffer.copy_(tensor)
        if torch.cuda.is_current_stream_capturing():
            return tensor

    enable_file = os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_ENABLE_FILE")
    can_save = bool(dump_dir) and (
        not enable_file or os.path.exists(enable_file)
    )
    direct_save = os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_DIRECT_SAVE", "1") != "0"
    if direct_save and can_save and not torch.cuda.is_current_stream_capturing():
        target_counts = _sm70_parse_int_ranges(
            os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_COUNTS")
        )
        try:
            max_dumps = int(os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_MAX_DUMPS", "4"))
        except ValueError:
            max_dumps = 4
        key = f"{os.getpid()}:{layer_idx}:{label}"
        count = _SM70_QWEN_LAYER_DUMP_COUNTS.get(key, 0)
        _SM70_QWEN_LAYER_DUMP_COUNTS[key] = count + 1
        if target_counts is not None and count not in target_counts:
            return tensor
        save_count = _SM70_QWEN_LAYER_DUMP_SAVE_COUNTS.get(key, 0)
        if max_dumps <= 0 or save_count < max_dumps:
            _SM70_QWEN_LAYER_DUMP_SAVE_COUNTS[key] = save_count + 1
            safe_label = label.replace("/", "_").replace(".", "_")
            safe_type = layer_type.replace("/", "_").replace(".", "_")
            path = os.path.join(
                dump_dir,
                (
                    f"pid{os.getpid()}_layer{layer_idx:02d}_{safe_type}_"
                    f"{safe_label}_{count:03d}.pt"
                ),
            )
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(
                {
                    "label": label,
                    "layer_idx": layer_idx,
                    "layer_type": layer_type,
                    "count": count,
                    "pid": os.getpid(),
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "tensor": tensor.detach().cpu(),
                },
                path,
            )
    return tensor


def dump_sm70_qwen_layer_graph_buffers(step: int, stage: str) -> None:
    dump_dir = os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_DIR")
    if not dump_dir:
        return
    enable_file = os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_ENABLE_FILE")
    if enable_file and not os.path.exists(enable_file):
        return
    target_steps = _sm70_parse_int_ranges(
        os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_GRAPH_STEPS")
        or os.getenv("VLLM_SM70_DUMP_QWEN_LAYER_COUNTS")
    )
    if target_steps is not None and step not in target_steps:
        return
    if not _SM70_QWEN_LAYER_GRAPH_BUFFERS:
        return
    os.makedirs(dump_dir, exist_ok=True)
    for key, buffer in _SM70_QWEN_LAYER_GRAPH_BUFFERS.items():
        meta = _SM70_QWEN_LAYER_GRAPH_META.get(key, {})
        label = str(meta.get("label", "unknown")).replace("/", "_").replace(".", "_")
        layer_type = (
            str(meta.get("layer_type", "unknown")).replace("/", "_").replace(".", "_")
        )
        layer_idx = int(meta.get("layer_idx", -1))
        shape = "x".join(str(dim) for dim in tuple(buffer.shape))
        path = os.path.join(
            dump_dir,
            (
                f"pid{os.getpid()}_step{step:04d}_layer{layer_idx:02d}_"
                f"{layer_type}_{label}_shape{shape}.pt"
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
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        dump_sm70_moe_runner_graph_buffers,
    )

    dump_sm70_moe_runner_graph_buffers(step, stage)


def _sm70_qwen_layer_dump_fake(
    tensor: torch.Tensor,
    label: str,
    layer_idx: int,
    layer_type: str,
) -> torch.Tensor:
    return tensor


direct_register_custom_op(
    op_name="sm70_qwen_layer_dump",
    op_func=_sm70_qwen_layer_dump_impl,
    mutates_args=[],
    fake_impl=_sm70_qwen_layer_dump_fake,
)


def _sm70_dump_qwen_layer_tensor(
    label: str,
    layer_idx: int,
    layer_type: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if not _sm70_qwen_layer_dump_requested(layer_idx):
        return tensor
    tensor = torch.ops.vllm.sm70_qwen_layer_dump(
        tensor,
        label,
        layer_idx,
        layer_type,
    )
    return tensor


KVCache = tuple[torch.Tensor, torch.Tensor]


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        self.layer_idx = extract_layer_index(prefix)

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
            or config.shared_expert_intermediate_size <= 0
        ):
            self.shared_expert = None
        else:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )
            if (
                envs.VLLM_SM70_DISABLE_QWEN3NEXT_SHARED_MOE_OVERLAP
                or not envs.VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP
            ):
                self.shared_expert._vllm_disable_shared_experts_stream = True
            if envs.VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP:
                logger.info_once(
                    "Enabling Qwen3Next FusedMoE shared_experts stream "
                    "overlap by explicit request.",
                    scope="local",
                )

        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        hidden_states = _sm70_dump_qwen_layer_tensor(
            "moe_input",
            self.layer_idx,
            "moe",
            hidden_states,
        )
        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            router_logits = _sm70_dump_qwen_layer_tensor(
                "moe_router_logits",
                self.layer_idx,
                "moe",
                router_logits,
            )
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        final_hidden_states = _sm70_dump_qwen_layer_tensor(
            "moe_output",
            self.layer_idx,
            "moe",
            final_hidden_states,
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        self.layer_idx = extract_layer_index(prefix)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = _sm70_dump_qwen_layer_tensor(
            "full_attn_qkv",
            self.layer_idx,
            "full_attention",
            qkv,
        )

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
        v = _sm70_dump_qwen_layer_tensor(
            "full_attn_v",
            self.layer_idx,
            "full_attention",
            v,
        )

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)
        q = _sm70_dump_qwen_layer_tensor(
            "full_attn_q_rot",
            self.layer_idx,
            "full_attention",
            q,
        )
        k = _sm70_dump_qwen_layer_tensor(
            "full_attn_k_rot",
            self.layer_idx,
            "full_attention",
            k,
        )

        attn_output = self.attn(q, k, v)
        attn_output = _sm70_dump_qwen_layer_tensor(
            "full_attn_core_out",
            self.layer_idx,
            "full_attention",
            attn_output,
        )

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            gate = _sm70_dump_qwen_layer_tensor(
                "full_attn_gate",
                self.layer_idx,
                "full_attention",
                gate,
            )
            attn_output = attn_output * gate
            attn_output = _sm70_dump_qwen_layer_tensor(
                "full_attn_gated_out",
                self.layer_idx,
                "full_attention",
                attn_output,
            )

        output[:], _ = self.o_proj(attn_output)
        _sm70_dump_qwen_layer_tensor(
            "full_attn_o_proj_out",
            self.layer_idx,
            "full_attention",
            output,
        )


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        hidden_states = _sm70_dump_qwen_layer_tensor(
            "layer_input_hidden",
            self.layer_idx,
            self.layer_type,
            hidden_states,
        )
        if residual is not None:
            residual = _sm70_dump_qwen_layer_tensor(
                "layer_input_residual",
                self.layer_idx,
                self.layer_type,
                residual,
            )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = _sm70_dump_qwen_layer_tensor(
            "input_norm_out",
            self.layer_idx,
            self.layer_type,
            hidden_states,
        )

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output
        hidden_states = _sm70_dump_qwen_layer_tensor(
            "attn_out",
            self.layer_idx,
            self.layer_type,
            hidden_states,
        )

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = _sm70_dump_qwen_layer_tensor(
            "post_attn_norm_out",
            self.layer_idx,
            self.layer_type,
            hidden_states,
        )
        residual = _sm70_dump_qwen_layer_tensor(
            "post_attn_residual",
            self.layer_idx,
            self.layer_type,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = _sm70_dump_qwen_layer_tensor(
            "mlp_out",
            self.layer_idx,
            self.layer_type,
            hidden_states,
        )

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        trace_enabled = _sm70_profile_trace_enabled()
        if trace_enabled:
            _sm70_profile_trace(
                "Qwen3NextModel.forward start layers=%s:%s hidden_shape=%s",
                self.start_layer,
                self.end_layer,
                tuple(hidden_states.shape),
            )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            layer_start = time.perf_counter() if trace_enabled else 0.0
            if trace_enabled:
                _sm70_profile_trace(
                    "layer %s enter type=%s hidden_shape=%s residual=%s",
                    layer_idx,
                    getattr(layer, "layer_type", None),
                    tuple(hidden_states.shape),
                    residual is not None,
                )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if trace_enabled:
                _sm70_profile_trace(
                    "layer %s exit elapsed_s=%.3f hidden_shape=%s residual=%s",
                    layer_idx,
                    time.perf_counter() - layer_start,
                    tuple(hidden_states.shape),
                    residual is not None,
                )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        num_experts = getattr(self.config, "num_experts", 0)
        if rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
            num_experts += 1
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        is_fse = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        num_routed = getattr(self.config, "num_experts", 0)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # FSE: remap shared_expert weights to the fused expert slot
            if is_fse and "mlp.shared_expert." in name:
                name = name.replace(
                    "mlp.shared_expert.",
                    f"mlp.experts.{num_routed}.",
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
