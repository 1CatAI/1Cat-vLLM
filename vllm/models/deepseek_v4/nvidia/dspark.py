# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model.

DSpark predicts a non-causal block in one three-layer forward pass, then adds
an autoregressive low-rank Markov bias while sampling the block left to right.
The checkpoint stores these weights under ``mtp.0``, ``mtp.1`` and ``mtp.2``;
they are not compatible with the ordinary DeepSeek MTP module.
"""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform

from .model import DeepseekV4DecoderLayer, make_deepseek_v4_expert_params_mapping

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DSparkMarkovHead(nn.Module):
    """Replicated low-rank token-transition head."""

    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        prefix: str,
    ) -> None:
        super().__init__()
        # Replication avoids one all-reduce and one vocab gather per DSpark step.
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = ReplicatedLinear(
            markov_rank,
            vocab_size,
            bias=False,
            return_bias=False,
            prefix=maybe_prefix(prefix, "markov_w2"),
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(markov_embed)


class DSparkConfidenceHead(nn.Module):
    """Replicated FP32 conditional-acceptance predictor."""

    def __init__(self, input_size: int, prefix: str) -> None:
        super().__init__()
        # The checkpoint tensor is BF16, but the official implementation keeps
        # this projection and its input in FP32 for calibration accuracy.
        self.proj = ReplicatedLinear(
            input_size,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            return_bias=False,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((hidden_states.float(), markov_embeds.float()), dim=-1)
        return self.proj(features).squeeze(-1)


class DSparkDeepseekV4Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_confidence_head: bool = False,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or 3

        # Both modules are replaced with target-model weights after loading.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # DSpark's target aux streams can reach O(1e4) before main_proj. The
        # SM70 W8A16 kernel writes FP16, so the otherwise valid projection can
        # overflow before the following RMSNorm. A power-of-two input scale is
        # exact in FP16 and RMSNorm removes it; 2^-6 leaves ample headroom while
        # preserving the FP32-reference normalized result.
        self.main_proj_input_scale = (
            2.0**-6 if current_platform.is_device_capability((7, 0)) else 1.0
        )

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        draft_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    draft_vllm_config,
                    prefix=maybe_prefix(
                        prefix, f"layers.{self.num_hidden_layers + stage}"
                    ),
                    topk_indices_buffer=self.topk_indices_buffer,
                    aux_stream_list=aux_stream_list,
                )
                for stage in range(self.num_dspark_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head = (
            DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )
            if use_confidence_head
            else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.main_proj(aux_hidden_states * self.main_proj_input_scale)
        return self.main_norm(projected)

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        for layer in self.layers:
            attn = layer.attn
            cache_layer = attn.mla_attn.swa_cache_layer
            slot_mapping = (
                None
                if context_slot_mappings is None
                else context_slot_mappings.get(cache_layer.prefix)
            )
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            kv = attn.kv_norm(qr_kv[..., attn.q_lora_rank :]).contiguous()
            if slot_mapping is not None:
                _insert_context_kv(attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        # Match the official DSpark reference exactly. The local target model's
        # 2-D mHC broadcast path is a separate optimization candidate and must
        # not be enabled until acceptance equivalence is established.
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        return hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    mla = attn.mla_attn
    cache_layer = mla.swa_cache_layer
    swa_cache = cache_layer.kv_cache
    dummy_q = torch.zeros(
        (kv.shape[0], attn.n_local_heads, attn.head_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    if current_platform.is_device_capability((7, 0)):
        from vllm.models.deepseek_v4.sm70.qnorm_rope_kv_fp8_insert import (
            sm70_qnorm_rope_kv_fp8_insert,
        )

        sm70_qnorm_rope_kv_fp8_insert(
            dummy_q,
            kv,
            swa_cache,
            slot_mapping,
            positions,
            attn.rotary_emb.cos_sin_cache,
            attn.eps,
            cache_layer.block_size,
        )
        return

    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        dummy_q,
        kv,
        swa_cache.view(swa_cache.shape[0], -1),
        slot_mapping,
        positions.to(torch.int64),
        attn.rotary_emb.cos_sin_cache,
        mla.padded_heads,
        attn.eps,
        cache_layer.block_size,
    )


class DSparkDeepseekV4ForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        speculative_config = vllm_config.speculative_config
        self._confidence_head_required = bool(
            speculative_config.dspark_confidence_threshold > 0.0
            or envs.VLLM_SPEC_DUMP_ALIGNMENT
        )
        self._confidence_head_loaded = False
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            use_confidence_head=self._confidence_head_required,
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            layer.attn.mla_attn.swa_cache_layer.prefix for layer in self.model.layers
        ]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed)

    def confidence_logits(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if not self._confidence_head_loaded:
            raise RuntimeError(
                "DSpark confidence logits were requested, but the draft "
                "checkpoint did not provide confidence_head.proj.weight."
            )
        confidence_head = self.model.confidence_head
        if confidence_head is None:
            raise RuntimeError("DSpark confidence head was not initialized.")
        return confidence_head(hidden_states, markov_embeds)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        first_layer = self.model.layers[0]
        if first_layer.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = FusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        local_heads = self.config.num_attention_heads // tp_size
        head_start = local_heads * tp_rank
        head_end = local_heads * (tp_rank + 1)

        for checkpoint_name, loaded_weight in weights:
            name = self._remap_dspark_name(checkpoint_name)
            if name is None:
                continue
            # Ordinary serving does not instantiate the optional diagnostic /
            # scheduler head, so ignore its checkpoint tensor without adding
            # an unused parameter to the runtime model.
            if name.startswith("model.confidence_head.") and name not in params_dict:
                continue
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    param = params_dict[mapped_name]
                    loader = typing.cast(Callable[..., bool], param.weight_loader)
                    if loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    ):
                        loaded_params.add(mapped_name)
                        break
                continue

            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader: Callable[..., object] = (
                    getattr(param, "weight_loader", None) or default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()
        if (
            get_pp_group().world_size > 1
            and "model.embed_tokens.weight" not in loaded_params
        ):
            # Under pipeline parallelism the drafter cannot share the
            # target's embedding (first stage) and runs on its own table;
            # an unloaded table means random embeddings, not an error.
            raise RuntimeError(
                "DSpark drafter: the checkpoint supplied no 'embed.weight' for "
                "the drafter's own embedding, which pipeline parallelism needs."
            )
        confidence_weight_name = "model.confidence_head.proj.weight"
        self._confidence_head_loaded = confidence_weight_name in loaded_params
        if self._confidence_head_required and not self._confidence_head_loaded:
            raise ValueError(
                "DSpark confidence collection requires the draft checkpoint "
                f"parameter {confidence_weight_name!r}."
            )
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    @staticmethod
    def _remap_dspark_name(name: str) -> str | None:
        if name == "embed.weight":
            # The target's embedding also fills the drafter's own table. With
            # PP=1 the proposer replaces it by the shared target embedding
            # afterwards; under pipeline parallelism the target embedding
            # lives on the first stage and the drafter on the last, so this
            # copy is the one the drafter runs on.
            return "model.embed_tokens.weight"
        match = re.match(r"mtp\.(\d+)\.(.*)", name)
        if match is None:
            return None
        stage = int(match.group(1))
        rest = match.group(2)
        head_prefixes = (
            "norm.",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "markov_head.",
            "confidence_head.",
        )
        if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(
            head_prefixes
        ):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"
