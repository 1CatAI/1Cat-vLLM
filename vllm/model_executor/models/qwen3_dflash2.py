# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

from .dflash_sm70 import (
    DFLASH_SM70_WIDE_OUTPUT_SCALE,
    DFlashSM70MLP,
    DFlashSM70RMSNorm,
)
from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix

logger = init_logger(__name__)


def _use_sm70_bf16_emulation(config) -> bool:
    declared_dtype = getattr(config, "dtype", None)
    if declared_dtype is None:
        declared_dtype = getattr(config, "torch_dtype", None)
    is_bf16 = declared_dtype is torch.bfloat16 or str(declared_dtype).lower() in {
        "bfloat16",
        "bf16",
        "torch.bfloat16",
    }
    return (
        is_bf16
        and current_platform.is_cuda()
        and current_platform.is_device_capability(70)
    )


@cache
def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """Return FlashInfer radix top-k only where the installed kernel can run.

    Presence is not a sufficient capability check: FlashInfer can be installed
    in the V100 environment while its radix top-k has no SM70 implementation.
    """
    if not current_platform.is_cuda():
        return None
    if not current_platform.has_device_capability(80):
        logger.info_once(
            "DFlash2 disables FlashInfer top-k below SM80; using torch.topk."
        )
        return None
    if not has_flashinfer():
        logger.info_once(
            "FlashInfer is unavailable; the DFlash2 selector uses torch.topk."
        )
        return None
    from flashinfer import top_k

    return top_k


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _flashinfer_topk()
    if impl is None or not scores.is_cuda:
        return torch.topk(scores, k, dim=-1)
    return impl(scores, k, sorted=True, deterministic=True)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.use_sm70_bf16_emulation = _use_sm70_bf16_emulation(config)
        if self.use_sm70_bf16_emulation:
            runtime_dtype = vllm_config.model_config.dtype
            if runtime_dtype != torch.float16:
                raise ValueError(
                    "BF16 DFlash2 on SM70 requires FP16 runtime transport; "
                    f"got runtime dtype {runtime_dtype}."
                )
            self.self_attn.output_input_scale = DFLASH_SM70_WIDE_OUTPUT_SCALE
            self.mlp = DFlashSM70MLP(self.mlp)
            self.input_layernorm = DFlashSM70RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                runtime_dtype,
            )
            self.post_attention_layernorm = DFlashSM70RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                runtime_dtype,
            )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.use_sm70_bf16_emulation = _use_sm70_bf16_emulation(self.config)
        if self.use_sm70_bf16_emulation:
            runtime_dtype = vllm_config.model_config.dtype
            self.hidden_norm = DFlashSM70RMSNorm(
                self.config.hidden_size,
                self.config.rms_norm_eps,
                runtime_dtype,
                residual_scale=1.0,
            )
            self.norm = DFlashSM70RMSNorm(
                self.config.hidden_size,
                self.config.rms_norm_eps,
                runtime_dtype,
            )
            logger.info_once(
                "Using range-preserving BF16 DFlash2 arithmetic on SM70 "
                "(FP32 residual, BF16 RNE, scaled FP16 projections)."
            )
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # The selector is compiled separately from the draft backbone. A unique
        # model tag prevents the two incompatible input signatures from sharing
        # a persistent compile-cache namespace.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale

    def _normalize_context_states(self, context_states: torch.Tensor) -> torch.Tensor:
        if self.use_sm70_bf16_emulation:
            return self.hidden_norm(context_states)
        return super()._normalize_context_states(context_states)


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(
            self.lm_head.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            raise ValueError(
                "DFlash2 requires an unquantized target LM head for candidate TopK; "
                f"got {type(self.lm_head.quant_method).__name__}."
            )

        selector = self.model.candidate_selector
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        values, ids = _topk(logits, selector.top_k)
        ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index

        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, selector.top_k)
            ids = ids.gather(-1, selected)

        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids, values


EntryClass = DFlash2Qwen3ForCausalLM
