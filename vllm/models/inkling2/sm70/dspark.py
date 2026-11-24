# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark draft (`mtp.safetensors`) for SM70.

Port of upstream `vllm/models/inkling2/nvidia/dspark.py`: an anchor plus mask
block runs through three small dense layers over DFlare-mixed target-feature
K/V, then a recurrent (`rnn_hidden`) Markov head corrects the base logits.
fp16 activations (the draft is trained in bf16; its numerics only change the
acceptance rate, the target verifies every token). Differences from upstream:

- the paged K/V live in the port's `(blocks, page, H_kv, 2 * D)` cache, written
  by one fused q/k-norm + RoPE kernel and read by the port's `rel_attention`
  in its DRAFT mode (`ops/dspark.py::block_attention`);
- plain TP all-reduces and a fused add + RMSNorm kernel (no FlashInfer);
- projections are cuBLAS GEMMs on fp16 copies of the bf16 weights.
"""

import os
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from ..configs import Inkling2DSparkConfig
from .attention import Inkling2AttentionMetadata, Inkling2AttentionPassCache
from .ops.attention import plan_decode_attention
from .ops.dspark import (
    add_rms_norm,
    add_softcap,
    block_attention,
    markov_gate,
    mix_rmsnorm,
    qkv_prep_,
    silu_mul,
)
from .ops.dspark_markov import markov_sample
from .ops.norm import rms_norm

# Shard-local Markov head (ops/dspark_markov.py): no vocabulary all-gather per
# draft step, only the verified steps run. `=0`: the unfused head.
FUSED_MARKOV = os.environ.get("VLLM_INKLING2_SM70_DSPARK_FUSED_MARKOV", "1") != "0"
# Draft layers: one key block per attention program in the local layers too
# (they ran 8 KV splits, 4 key blocks each), and a fused SiLU * up. `=0`: old.
FAST_LAYERS = os.environ.get("VLLM_INKLING2_SM70_DSPARK_LAYERS", "1") != "0"


def mm_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`x @ weight.T` with fp32 output (fp16 inputs, fp32 accumulate)."""
    try:
        return torch.mm(x, weight.t(), out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return torch.mm(x.float(), weight.float().t())


def _make_cos_sin(head_dim: int, theta: float, max_len: int, device) -> torch.Tensor:
    """fp32 `[cos | sin]` table `(max_len, head_dim)` of neox RoPE."""
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    freqs = torch.outer(
        torch.arange(max_len, dtype=torch.float32, device=device), inv_freq
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


class _Norm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class Inkling2DraftAttention(nn.Module):
    """Bidirectional block attention over context K/V (from target features)
    and the block itself; local layers see `sliding_window_size` positions
    before the block start."""

    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= config.num_key_value_heads
        hidden, self.head_dim = config.hidden_size, config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps
        self.block = config.draft_block_size
        self.is_local = is_local
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.kv_proj_ctx = MergedColumnParallelLinear(
            hidden,
            [config.num_key_value_heads * self.head_dim] * 2,
            bias=False,
            prefix=f"{prefix}.kv_proj_ctx",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            hidden,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = _Norm(self.head_dim, self.eps)
        self.k_norm = _Norm(self.head_dim, self.eps)
        self.k_norm_ctx = _Norm(self.head_dim, self.eps)
        # Owns the paged KV cache and the attention metadata of this layer:
        # the same spec as the target's layers (window layers keep the last
        # `sliding_window_size` positions before the block).
        self.cache = Inkling2AttentionPassCache(
            f"{prefix}.cache",
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            config.sliding_window_size if is_local else None,
        )
        # Keys a block sees: itself and, if local, the window before it.
        self.window = config.sliding_window_size + self.block if is_local else 0
        self.rel_extent = config.sliding_window_size

    def _metadata(self) -> Inkling2AttentionMetadata | None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or self.cache.kv_cache.numel() == 0:
            return None
        return cast(Inkling2AttentionMetadata, attn_metadata[self.cache.prefix])

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, cos_sin: torch.Tensor
    ) -> torch.Tensor:
        """This rank's `(T, hidden)` partial attention output."""
        qkv = F.linear(x, self.qkv_proj.weight)
        md = self._metadata()
        out = torch.zeros((x.shape[0], self.q_size), dtype=x.dtype, device=x.device)
        if md is not None:
            kv_cache = self.cache.kv_cache
            # The draft's query slots live in the speculator's own buffer
            # (the metadata's `slot_mapping` is the target batch's).
            slot_mappings = get_forward_context().slot_mapping
            slots = (
                slot_mappings[self.cache.prefix]
                if isinstance(slot_mappings, dict)
                else md.slot_mapping
            )
            qkv_prep_(
                qkv,
                positions,
                slots[: x.shape[0]],
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin,
                kv_cache,
                self.num_heads,
                self.num_kv_heads,
                self.eps,
            )
            block_attention(
                qkv[:, : self.q_size].view(-1, self.num_heads, self.head_dim),
                kv_cache[..., : self.head_dim],
                kv_cache[..., self.head_dim :],
                out,
                block_table=md.block_table[: md.num_reqs],
                cache_seqlens=md.seq_lens[: md.num_reqs],
                cu_seqlens_q=md.query_start_loc[: md.num_reqs + 1],
                block=self.block,
                softmax_scale=self.scaling,
                window=self.window,
                num_splits=self._num_splits(md.num_reqs),
            )
        return F.linear(out, self.o_proj.weight)

    def _num_splits(self, num_reqs: int) -> int:
        if not (FAST_LAYERS and self.is_local):
            return plan_decode_attention(
                self.is_local, num_reqs, self.num_kv_heads, self.rel_extent
            )
        # A block's 8 queries are one tile whose cost is its key blocks (SM70
        # `tl.dot` is scalar FMA): spread the window's blocks over the SMs as
        # the global layer does, at most one split per key block.
        splits = plan_decode_attention(
            False, num_reqs, self.num_kv_heads, self.rel_extent
        )
        return max(1, min(splits, -(-self.window // 32)))

    def store_context_kv(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if slot_mapping is None or self.cache.kv_cache.numel() == 0:
            return
        kv = F.linear(context, self.kv_proj_ctx.weight)
        qkv_prep_(
            kv,
            positions,
            slot_mapping,
            None,
            self.k_norm_ctx.weight,
            cos_sin,
            self.cache.kv_cache,
            0,
            self.num_kv_heads,
            self.eps,
        )


class Inkling2DraftMLP(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.gate_up_proj.weight)
        if FAST_LAYERS:
            return F.linear(silu_mul(gate_up), self.down_proj.weight)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.down_proj.weight)


class Inkling2DraftLayer(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _Norm(hidden, eps)
        self.post_attention_layernorm = _Norm(hidden, eps)
        self.dflare_norm = _Norm(hidden, eps)
        self.self_attn = Inkling2DraftAttention(config, is_local, f"{prefix}.self_attn")
        self.mlp = Inkling2DraftMLP(config, f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns this rank's partial MLP output and the residual."""
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x = add_rms_norm(
                tensor_model_parallel_all_reduce(x),
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
            )
        x = self.self_attn(positions, x, cos_sin)
        x = add_rms_norm(
            tensor_model_parallel_all_reduce(x),
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        return self.mlp(x), residual


class Inkling2MarkovHead(nn.Module):
    """Recurrent (`rnn_hidden`) correction of the draft's base logits.

    Its rank-`r` latent goes through `hidden_out` and the shared LM head,
    folded once into a `(vocab, r)` projection. The `[hidden, f_rnn]` columns
    of `rnn_proj` (and its bias) are applied to all draft steps at once in
    fp32; each step adds the `[state, previous]` columns."""

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(
            config.padded_vocab_size,
            rank,
            _weight=torch.empty(config.padded_vocab_size, rank),
            _freeze=True,
        )
        self.rnn_proj = ReplicatedLinear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = ReplicatedLinear(rank, hidden, bias=True)
        self.dflare_norm = _Norm(hidden, config.rms_norm_eps)
        self.fold_weight: torch.Tensor | None = None
        self.fold_bias: torch.Tensor | None = None
        self.rnn_bias: torch.Tensor | None = None
        self.w_state: torch.Tensor | None = None
        self.w_ctx: torch.Tensor | None = None

    @torch.no_grad()
    def fold(self, lm_head_weight: torch.Tensor) -> None:
        """Fold `hidden_out` into the (TP-local) LM head rows (fp32 math).
        Later calls refill the same tensors, which CUDA graphs point at."""
        projection = self.hidden_out.weight.float()
        bias = self.hidden_out.bias.float()
        num_rows = lm_head_weight.shape[0]
        if self.fold_weight is None or self.fold_bias is None:
            self.fold_weight = lm_head_weight.new_empty((num_rows, self.markov_rank))
            self.fold_bias = torch.empty(
                num_rows, dtype=torch.float32, device=lm_head_weight.device
            )
        for start in range(0, num_rows, 4096):
            rows = slice(start, start + 4096)
            head = lm_head_weight[rows].float()
            self.fold_weight[rows].copy_(head @ projection)
            self.fold_bias[rows].copy_(head @ bias)

    @torch.no_grad()
    def prepare(self) -> None:
        """Contiguous column blocks of `rnn_proj` and the fp32 bias."""
        rank = self.markov_rank
        weight = self.rnn_proj.weight
        self.w_state = weight[:, : 2 * rank].contiguous()
        self.w_ctx = weight[:, 2 * rank :].contiguous()
        if self.rnn_bias is None:
            self.rnn_bias = torch.empty(
                weight.shape[0], dtype=torch.float32, device=weight.device
            )
        self.rnn_bias.copy_(self.rnn_proj.bias)

    def context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """fp32 `rnn_proj` bias plus its `[hidden, f_rnn]` term for the
        `(req, step)`-ordered `hidden`; returns `(step, req, 3 * r)`."""
        assert self.w_ctx is not None and self.rnn_bias is not None
        num_reqs, width = f_rnn.shape
        hidden = hidden.view(num_reqs, -1, width).transpose(0, 1)
        inputs = torch.cat([hidden, f_rnn.expand_as(hidden)], dim=-1).flatten(0, 1)
        out = mm_fp32(inputs, self.w_ctx) + self.rnn_bias
        return out.view(hidden.shape[0], num_reqs, -1)

    def step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step; returns the new fp32 state and the TP-local
        fp32 bias over the vocabulary shard."""
        assert self.fold_weight is not None and self.fold_bias is not None
        assert self.w_state is not None
        inputs = torch.cat(
            [state.to(self.w_state.dtype), self.markov_w1(prev_tokens)], dim=-1
        )
        raw = (mm_fp32(inputs, self.w_state) + context).contiguous()
        state, latent = markov_gate(raw, state, self.w_state.dtype)
        return state, mm_fp32(latent, self.fold_weight) + self.fold_bias


class Inkling2DSparkDecoder(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, max_model_len: int, prefix: str
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(
                config, i in config.local_layer_ids, f"{prefix}.layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = _Norm(hidden, config.rms_norm_eps)
        self.markov_head = Inkling2MarkovHead(config)
        self.fuse_weight = nn.Parameter(
            torch.zeros(config.num_hidden_layers + 1, len(config.target_layer_ids)),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self.register_buffer(
            "fuse_alpha", torch.empty_like(self.fuse_weight), persistent=False
        )
        self.max_model_len = max_model_len
        self.cos_sin: torch.Tensor | None = None
        # Shared with the target model (see Inkling2DSparkModel.attach_target).
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None

    def mix(self, features: torch.Tensor, row: int, norm: _Norm) -> torch.Tensor:
        targets = features.view(features.shape[0], -1, self.config.hidden_size)
        return mix_rmsnorm(targets, self.fuse_alpha[row], norm.weight, norm.eps)

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normalized target embedding of each block's anchor, then masks."""
        assert self.embed_tokens is not None and self.embed_norm is not None
        block = self.config.draft_block_size
        ids = input_ids.view(-1, block)
        x = self.mask_embed.expand(ids.shape[0], block, -1).clone()
        x[:, 0] = self.embed_norm(self.embed_tokens(ids[:, 0].contiguous()))
        return x.view(-1, self.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert self.cos_sin is not None
        x = self.embed_block(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(positions, x, residual, self.cos_sin)
        assert residual is not None
        return add_rms_norm(
            tensor_model_parallel_all_reduce(x),
            residual,
            self.norm.weight,
            self.norm.eps,
        )


class Inkling2DSparkModel(nn.Module):
    """DSpark draft for Inkling2; plugged in by `Inkling2DSparkSpeculator`."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    _stacked = (
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
        (".k_proj_ctx.", ".kv_proj_ctx.", 0),
        (".v_proj_ctx.", ".kv_proj_ctx.", 1),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        assert isinstance(config, Inkling2DSparkConfig)
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        if num_spec != config.draft_block_size - 1:
            raise ValueError(
                "Inkling2 DSpark requires num_speculative_tokens = "
                f"{config.draft_block_size - 1} (the checkpoint's block size - 1)"
            )
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(
                f"Inkling2 DSpark drafts up to max_position_embeddings="
                f"{config.max_position_embeddings} positions"
            )
        self.config = config
        self.checkpoint_dir = vllm_config.speculative_config.draft_model_config.model
        self.model = Inkling2DSparkDecoder(
            config, vllm_config.model_config.max_model_len, f"{prefix}dspark"
        )
        self.lm_head: nn.Module | None = None
        self.final_logit_softcapping = config.final_logit_softcapping

    @torch.no_grad()
    def attach_target(self, target: nn.Module) -> None:
        """Share the target's normalized embeddings and LM head and derive the
        weights that depend on them. `target` is the Inkling2 text model."""
        self.model.embed_tokens = target.model.embed_tokens
        self.model.embed_norm = target.model.embed_norm
        object.__setattr__(self, "lm_head", target.lm_head)
        self.refresh_derived_weights()

    @torch.no_grad()
    def refresh_derived_weights(self) -> None:
        assert self.lm_head is not None
        head = self.model.markov_head
        head.fold(self.lm_head.weight)
        head.prepare()
        self.model.fuse_alpha.copy_(
            torch.softmax(self.model.fuse_weight.float(), dim=-1)
        )
        if self.model.cos_sin is None:
            self.model.cos_sin = _make_cos_sin(
                self.config.head_dim,
                self.config.rope_theta,
                self.model.max_model_len + 16,
                self.lm_head.weight.device,
            )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.cache.prefix for layer in self.model.layers]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Each draft layer mixes the raw target features itself.
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        assert self.model.cos_sin is not None
        for i, layer in enumerate(self.model.layers):
            context = self.model.mix(context_states, i, layer.dflare_norm)
            layer.self_attn.store_context_kv(
                context,
                context_positions,
                self.model.cos_sin,
                None if context_slot_mappings is None else context_slot_mappings[i],
            )

    def compute_f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        """Recurrent-head context from the last committed token's features."""
        return self.model.mix(features, -1, self.model.markov_head.dflare_norm)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """TP-local fp32 base logits of the shared LM head."""
        assert self.lm_head is not None
        return mm_fp32(hidden_states, self.lm_head.weight)

    def markov_sample(
        self,
        hidden_states: torch.Tensor,
        f_rnn: torch.Tensor,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        anchor_stride: int,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dense: torch.Tensor | None,
        num_spec: int,
        use_fp64: bool,
    ) -> None:
        """The shard-local recurrent head (`ops/dspark_markov.py`): samples
        `tokens[request, step]` for the `(request, step)`-ordered rows of
        `hidden_states` `(reqs * steps, hidden)`; `f_rnn` is `(reqs,
        hidden)`. `dense` receives this rank's temperature-applied logits."""
        assert self.lm_head is not None
        head = self.model.markov_head
        assert head.fold_weight is not None and head.fold_bias is not None
        assert head.w_state is not None
        shard = head.fold_weight.shape[0]
        rank = get_tensor_model_parallel_rank()
        id_base = rank * shard
        markov_sample(
            self.compute_base_logits(hidden_states),
            head.context(hidden_states, f_rnn),
            head.w_state,
            head.markov_w1.weight,
            head.fold_weight,
            head.fold_bias,
            tokens,
            anchors,
            anchor_stride,
            idx_map,
            sample_pos,
            temperature,
            seeds,
            dense,
            num_spec,
            min(max(self.config.vocab_size - id_base, 0), shard),
            id_base,
            rank,
            get_tensor_model_parallel_world_size(),
            self.final_logit_softcapping,
            use_fp64,
            tensor_model_parallel_all_reduce,
        )

    def gather_dense_logits(self, dense: torch.Tensor) -> torch.Tensor:
        """All ranks' dense shard rows side by side `(rows, world * shard)`."""
        if get_tensor_model_parallel_world_size() == 1:
            return dense
        return tensor_model_parallel_all_gather(dense)

    def markov_logits(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Full-vocab softcapped fp32 logits."""
        logits = add_softcap(base_logits, bias, self.final_logit_softcapping)
        logits = tensor_model_parallel_all_gather(logits)
        return logits[:, : self.config.vocab_size]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load `mtp.safetensors` from the checkpoint directory. The loader's
        own iterator is not used: it drops files that the target's
        `model.safetensors.index.json` does not list."""
        from safetensors import safe_open

        del weights
        with safe_open(os.path.join(self.checkpoint_dir, "mtp.safetensors"), "pt") as f:
            return self._load((name, f.get_tensor(name)) for name in f.keys())  # noqa: SIM118

    def _load(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            name, shard = f"model.{name}", None
            for old, new, shard_id in self._stacked:
                if old in name:
                    name, shard = name.replace(old, new), shard_id
                    break
            param = params[name]
            if weight.dtype != param.dtype:
                weight = weight.to(param.dtype)
            if shard is None:
                getattr(param, "weight_loader", _default_loader)(param, weight)
            else:
                param.weight_loader(param, weight, shard)
            loaded.add(name)
        return loaded


def _default_loader(param: nn.Parameter, weight: torch.Tensor) -> None:
    param.data.copy_(weight)
