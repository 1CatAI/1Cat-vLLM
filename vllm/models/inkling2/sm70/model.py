# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 model for SM70 (V100): text, and images through the vision
encoder (`vision.py`, `processing.py`).

Port of upstream `vllm/models/inkling2/nvidia/model.py`. Decoder layers are
tied together by block attention residuals: each sublayer input is a learned
softmax mix over the outputs of previous blocks of
`n_layers_per_attn_res_block` layers plus the running partial block. Runs in
fp16 (mixes, routing, convs and softmax statistics in fp32); TP uses plain
all-reduces (no fused all-reduce / sequence-parallel kernels on SM70).

Looped checkpoints (`loop_config`) use `Inkling2LoopModel` (`loop.py`).
"""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from ..configs import Inkling2MMConfig, Inkling2ModelConfig
from ..quant_config import Inkling2QuantConfig
from .attention import Inkling2Attention
from .conv_state import ATTN_STREAM, MLP_STREAM, Inkling2ShortConv, InklingConvState
from .moe import Inkling2MoE, QuantFormat
from .ops import local_argmax as _local_argmax
from .ops.attention import compute_log_scaling_tau
from .ops.attn_res import attn_res
from .ops.norm import rms_norm
from .processing import (
    IMAGE_PLACEHOLDER,
    VISION,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from .vision import Inkling2VisionEncoder

logger = init_logger(__name__)


def _mm_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`x @ weight.T` with fp32 output (fp16 inputs, fp32 accumulate)."""
    try:
        return torch.mm(x, weight.t(), out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return torch.mm(x.float(), weight.float().t())


class Inkling2LogitsProcessor(LogitsProcessor):
    """fp32 logits (as upstream's fp32 head) before the softcap."""

    def _get_logits(self, hidden_states, lm_head, embedding_bias):
        logits = _mm_fp32(hidden_states, lm_head.weight)
        if embedding_bias is not None:
            logits = logits + embedding_bias
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(self, lm_head, hidden_states, embedding_bias=None):
        """int64 `[rows]`: per row the argmax token id of the logits `forward`
        would return (fp32 head, softcap, vocab trimmed), lowest id on ties,
        without the dense logits: each rank reduces its own shard to
        (value, id) and the ranks exchange 64 B per row in one all-reduce
        (`ops/local_argmax.py`). CUDA-graph safe."""
        logits = _mm_fp32(hidden_states, lm_head.weight)
        if embedding_bias is not None:
            logits = logits + embedding_bias
        # The softcap can round distinct logits to one value: apply it first, with
        # the dense path's own ops, so ties fall exactly as they do there.
        logits = self._apply_logits_transforms(logits)
        tp_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        local = logits.shape[-1]  # the all-gather concatenates the shards in rank order
        start = rank * local
        limit = max(0, min(local, self.org_vocab_size - start))
        packed = _local_argmax.pack_local_argmax(logits, limit, start, rank, tp_size)
        if tp_size > 1:
            packed = tensor_model_parallel_all_reduce(packed)
        return _local_argmax.select_top_tokens(packed)


class Inkling2RMSNorm(nn.Module):
    """RMSNorm; `num_inputs > 1` stacks the weights of per-input norms."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = None,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size), requires_grad=False
        )
        self.weight.weight_loader = self._load
        self.eps = eps
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2AttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32), pre-scaled by
    `hidden_size**-0.5` at load time."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local_ids else (3, hidden), dtype=torch.float32
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self._load_query
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.view(param.shape)).mul_(self.scale)


class Inkling2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
        num_passes: int = 1,
    ) -> None:
        """`num_passes > 1` (a loop core layer) gives each pass its own KV and
        conv-state caches."""
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_local
            else config.num_key_value_heads
        )
        self.conv_states = tuple(
            InklingConvState(
                num_kv_heads=num_kv_heads,
                head_dim=config.swa_head_dim if self.is_local else config.head_dim,
                hidden_size=config.hidden_size,
                kernel_size=config.sconv_kernel_size,
                prefix=f"{prefix}.conv_state" + (f".pass{p}" if p else ""),
                replicate_hidden=True,
            )
            for p in range(num_passes)
        )
        self.conv_state = self.conv_states[0]
        self.attn = Inkling2Attention(
            config,
            self.is_local,
            self.conv_state,
            prefix=f"{prefix}.attn",
            pass_conv_states=self.conv_states[1:],
        )
        eps = config.rms_norm_eps
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2RMSNorm(
            config.hidden_size, eps, num_inputs=self.num_attn_inputs
        )
        self.mlp_norm = Inkling2RMSNorm(config.hidden_size, eps)
        expert_format: QuantFormat = (
            "mxfp8"
            if quant_config is not None and quant_config.is_mxfp8_layer(layer_id)
            else "nvfp4"
        )
        self.mlp = Inkling2MoE(
            config, expert_format, quant_config, prefix=f"{prefix}.mlp"
        )
        self.attn_sconv, self.mlp_sconv = (
            Inkling2ShortConv(
                config.hidden_size, config.sconv_kernel_size, self.conv_state, stream
            )
            for stream in (ATTN_STREAM, MLP_STREAM)
        )

    def attention(
        self,
        positions: torch.Tensor,
        inputs: Sequence[torch.Tensor],
        tau: torch.Tensor | None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """This rank's partial attention output on the normalized `inputs`
        (one for local layers, or separate q/k/v inputs)."""
        if self.is_local and len(inputs) == 1:
            return self.attn(positions, inputs[0], out=out, pass_idx=pass_idx)
        return self.attn(
            positions,
            inputs[0],
            inputs[1],
            inputs[2],
            None if self.is_local else tau,
            out,
            pass_idx,
        )


class Inkling2Model(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            self._make_layer(config, i, quant_config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        return Inkling2DecoderLayer(config, layer_id, quant_config, prefix)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def _mix(
        self,
        stack: torch.Tensor,
        last: torch.Tensor,
        query: torch.Tensor,
        norm: Inkling2RMSNorm,
        addend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return attn_res(
            stack, last, addend, query, norm.weight, norm.eps, self.normalize_values
        )

    def _log_scaling_tau(self, positions: torch.Tensor) -> torch.Tensor | None:
        if self.config.log_scaling_n_floor is None:
            return None
        return compute_log_scaling_tau(
            positions, self.config.log_scaling_n_floor, self.config.log_scaling_alpha
        )

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Non-looped checkpoints (upstream's non-fused path)."""
        num_tokens, hidden = inputs_embeds.shape
        aux_hidden_states = []
        num_layers = len(self.layers)
        max_blocks = -(-num_layers // self.block_size)
        stack = inputs_embeds.new_empty((max_blocks + 1, num_tokens, hidden))
        stack[0].copy_(inputs_embeds)
        num_blocks = 0
        tau = self._log_scaling_tau(positions)
        for i, layer in enumerate(self.layers):
            partial = stack[num_blocks]
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(partial.clone())
            if num_blocks == 0:
                inputs = [
                    layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
                ]
            else:
                query = self.attn_res.attn_queries[i - 1]
                inputs = self._mix(stack[:num_blocks], partial, query, layer.attn_norm)
            attn_partial = layer.attention(positions, inputs, tau)
            attn_out = layer.attn_sconv(
                tensor_model_parallel_all_reduce(attn_partial), positions
            )
            mlp_query = self.attn_res.mlp_queries[i]
            first = i % self.block_size == 0
            if first:
                mlp_in = self._mix(
                    stack[: num_blocks + 1], attn_out, mlp_query, layer.mlp_norm
                )
            else:
                mlp_in = self._mix(
                    stack[:num_blocks], partial, mlp_query, layer.mlp_norm, attn_out
                )
            residual = None if first else partial
            if residual is None:
                num_blocks += 1
            layer.mlp_sconv(
                layer.mlp(mlp_in[0]),
                positions,
                addend=attn_out,
                residual=residual,
                out=stack[num_blocks],
            )
        out = self._mix(
            stack[:num_blocks], stack[num_blocks], self.attn_res.output_query, self.norm
        )[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2ForConditionalGeneration(nn.Module, SupportsMultiModal):
    """Inkling2 on SM70. Images (VLLM_INKLING2_SM70_VISION, default on) go
    through the bf16 vision encoder, replicated on every TP rank; its
    embeddings replace the image pad tokens' (after embed_norm, as upstream)."""

    # Checkpoint name substitutions, then fused per-layer weights.
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        assert quant_config is None or isinstance(quant_config, Inkling2QuantConfig)
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        if vllm_config.model_config.dtype != torch.float16:
            logger.warning(
                "Inkling2 on SM70 is validated in float16, got %s",
                vllm_config.model_config.dtype,
            )
        self.config = config
        self.text_config = text_config = config.text_config
        mm_config = vllm_config.model_config.multimodal_config
        self.visual = (
            Inkling2VisionEncoder(config.vision_config, text_config.hidden_size)
            if VISION
            and getattr(config, "vision_config", None) is not None
            and mm_config is not None
            and mm_config.get_limit_per_prompt("image")
            else None
        )
        model_cls: type[Inkling2Model] = Inkling2Model
        if text_config.loop_config is not None:
            from .loop import Inkling2LoopModel

            model_cls = Inkling2LoopModel
        self.model = model_cls(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size, text_config.hidden_size
        )
        if _local_argmax.ENABLED:
            _local_argmax._load()  # JIT build / load now, never inside CUDA-graph capture
        self.logits_processor = Inkling2LogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )

    def get_language_model(self) -> nn.Module:
        return self

    # SupportsEagle3 protocol (DSpark / EAGLE3 aux features): the layer ids are
    # layer visits (loop passes counted); the last one (the model's visit
    # count) is the final-norm output.
    supports_eagle3 = True
    has_own_lm_head = False
    has_own_embed_tokens = False

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return (5, 23, 41, 48)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Inkling2 accepts image inputs only")

    def embed_multimodal(self, **kwargs: object):
        pixels = kwargs.get("vision_pixels")
        grid = kwargs.get("vision_grid_thw")
        if pixels is None or grid is None:
            return []
        if isinstance(pixels, (list, tuple)):
            pixels = torch.cat(list(pixels))
        grid = torch.as_tensor(
            grid if not isinstance(grid, (list, tuple)) else torch.stack(list(grid))
        )
        grid = grid.reshape(-1, 3)
        assert self.visual is not None
        embeds = self.visual(pixels, grid)
        dtype = self.model.embed_tokens.weight.dtype
        if dtype == torch.float16:
            # bf16 -> fp16 is exact in fp16's range; clamp the rare outlier.
            embeds = embeds.float().clamp(-65504.0, 65504.0)
        return list(embeds.to(dtype).split(grid.prod(-1).tolist()))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        embeds = self.model.embed_input_ids(input_ids)
        if not multimodal_embeddings or is_multimodal is None:
            return embeds
        return _merge_multimodal_embeddings(
            embeds, multimodal_embeddings, is_multimodal
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    if (
        _local_argmax.ENABLED
    ):  # VLLM_INKLING2_SM70_LOCAL_ARGMAX (off: no such attribute)

        def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """int64 `[rows]` argmax token ids of `compute_logits(hidden_states)`
            (lowest id on ties), computed shard-locally: no dense logits, no
            all-gather. `hidden_states`: fp16 `[rows, hidden]` (the rows to sample)."""
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()
        visual_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("model.visual."):
                if self.visual is not None:
                    visual_weights.append((name.removeprefix("model.visual."), weight))
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            if name not in params:
                m = re.search(
                    r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name
                )
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if ".mlp.experts." in name and weight.dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                raise ValueError(
                    f"{name}: BF16 expert weights are not supported; the Inkling2 "
                    "experts are NVFP4 / MXFP8 (hf_quant_config.json)"
                )
            loader: Callable[..., None] = getattr(
                param, "weight_loader", default_weight_loader
            )
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                loaded_shards.add((name, shard_id))
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            fused = next(f for f in num_shards if f in name)
            if count == num_shards[fused]:
                loaded.add(name)
        if self.visual is not None:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual_weights)}
        derived = {n for n, p in params.items() if getattr(p, "inkling_derived", False)}
        if missing := params.keys() - loaded - derived:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        for module in self.modules():
            post = getattr(module, "inkling_post_load", None)
            if post is not None and module is not self:
                post()
        return loaded
