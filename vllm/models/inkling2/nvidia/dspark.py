# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark draft (`mtp.safetensors`): an anchor plus mask block run
through a small dense transformer over DFlare-mixed target-feature K/V, then a
recurrent (`rnn_hidden`) Markov head. Numerics follow the reference
implementation's draft (flashinfer RMSNorms, bf16 logits)."""

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.flash_v100.backend import FlashAttnV100Backend

from ..configs import Inkling2DSparkConfig
from .ops import is_sm70, model_act_dtype
from .ops.attention import block_decode_attention, plan_decode_attention
from .ops.dspark import (
    add_softcap,
    markov_gate,
    mix_rmsnorm,
    qk_norm_rope_,
    rope_,
    silu_and_mul,
)
from .ops.linear import linear
from .ops.norm import rms_norm


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=model_act_dtype()),
            requires_grad=False,
        )
        self.eps = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        if is_sm70():
            if residual is not None:
                residual.add_(x)
                target = residual
            else:
                target = x
            return rms_norm(target, self.weight, self.eps, out_dtype=model_act_dtype())
        from flashinfer.norm import fused_add_rmsnorm, rmsnorm

        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
        # In place: residual += x; x = norm(residual), from the unrounded sum.
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x


def all_reduce_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, norm: _RMSNorm
) -> torch.Tensor:
    """In place: `residual += all_reduce(x); x = norm(residual)`.

    Batches that the TP group's FlashInfer MNNVL all-reduce takes run its
    fused all-reduce + residual add + RMSNorm on the same workspace (the
    same rank-order fp32 sum, rounded once to bf16); the norm then reads the
    bf16 residual. Others all-reduce, then add and normalize.
    """
    fi_ar = getattr(get_tp_group().device_communicator, "fi_ar_comm", None)
    if fi_ar is None or not fi_ar.should_use_fi_ar(x):
        return norm(tensor_model_parallel_all_reduce(x), residual)
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        flashinfer_comm,
        flashinfer_trtllm_fused_allreduce_norm,
    )
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_workspace,
    )

    workspace = get_fi_ar_workspace(
        fi_ar.world_size,
        fi_ar.rank,
        fi_ar.max_num_tokens,
        x.shape[1],
        x.dtype,
        fi_ar.group,
    )
    assert flashinfer_comm is not None and workspace is not None
    assert workspace.backend == "mnnvl"
    assert x.dtype == residual.dtype == norm.weight.dtype == model_act_dtype()
    assert x.shape == residual.shape and residual.is_contiguous()
    flashinfer_trtllm_fused_allreduce_norm(
        allreduce_in=x,
        residual=residual,
        rms_gamma=norm.weight,
        rms_eps=norm.eps,
        world_size=fi_ar.world_size,
        launch_with_pdl=True,
        fp32_acc=True,
        max_token_num=fi_ar.max_num_tokens,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
    )
    return x


class Inkling2DraftSelfAttention(nn.Module):
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
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm_ctx = _RMSNorm(self.head_dim, self.eps)
        # Owns the paged KV cache only; the same spec as the target's layers
        # (the scheduler keeps the last sliding_window_size positions before
        # the block, which is what the draft reads).
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=config.sliding_window_size if is_local else None,
            prefix=f"{prefix}.attn",
            attn_backend=FlashAttnV100Backend if is_sm70() else FlashAttentionBackend,
        )
        # Keys a block sees: itself and, if local, the window before it.
        self.window = config.sliding_window_size + self.block if is_local else 0
        vllm_config = get_current_vllm_config()
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.register_buffer(
            "cu_seqlens_q",
            torch.arange(max_num_seqs + 1, dtype=torch.int32) * self.block,
            persistent=False,
        )

    def _metadata(self) -> FlashAttentionMetadata | None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or self.attn.kv_cache.numel() == 0:
            return None
        return cast(FlashAttentionMetadata, attn_metadata[self.attn.layer_name])

    def _write_kv(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        impl = cast(FlashAttentionImpl, self.attn.impl)
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        qkv = linear(x, self.qkv_proj.weight)
        qk_norm_rope_(
            qkv,
            positions,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            self.num_heads,
            self.num_kv_heads,
            self.eps,
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        out = q.new_zeros((q.shape[0], self.q_size))
        md = self._metadata()
        if md is not None:
            self._write_kv(
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                md.slot_mapping[: q.shape[0]],
            )
            kv_cache = self.attn.kv_cache
            if kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
                key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]
            elif kv_cache.ndim == 5 and kv_cache.shape[0] == 2:
                key_cache, value_cache = kv_cache[0], kv_cache[1]
            else:
                key_cache, value_cache = kv_cache.transpose(1, 2).split(
                    self.head_dim, dim=-1
                )
            num_reqs = min(q.shape[0] // self.block, md.seq_lens.shape[0])
            num_tokens = num_reqs * self.block
            # The plan's 0 (FA4, for very large target batches) is one split
            # here.
            num_splits = max(
                1,
                plan_decode_attention(
                    self.is_local,
                    num_reqs,
                    self.block,
                    self.num_heads,
                    self.num_kv_heads,
                    self.window,
                ),
            )
            block_decode_attention(
                q[:num_tokens].view(num_tokens, self.num_heads, self.head_dim),
                canonicalize_singleton_dim_strides(key_cache),
                canonicalize_singleton_dim_strides(value_cache),
                out[:num_tokens],
                block_table=md.block_table[:num_reqs],
                cache_seqlens=md.seq_lens[:num_reqs],
                cu_seqlens_q=self.cu_seqlens_q[: num_reqs + 1],
                block=self.block,
                window=self.window,
                num_splits=num_splits,
            )
        return linear(out, self.o_proj.weight)

    def store_context_kv(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        k, v = linear(context, self.kv_proj_ctx.weight).split(
            [self.kv_size, self.kv_size], -1
        )
        k = self.k_norm_ctx(k.reshape(-1, self.head_dim))
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        rope_(k, positions, cos_sin_cache)
        if slot_mapping is not None and self.attn.kv_cache.numel() > 0:
            self._write_kv(
                k, v.view(-1, self.num_kv_heads, self.head_dim), slot_mapping
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
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = silu_and_mul(linear(x, self.gate_up_proj.weight))
        return linear(x, self.down_proj.weight)


class Inkling2DraftLayer(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _RMSNorm(hidden, eps)
        self.post_attention_layernorm = _RMSNorm(hidden, eps)
        self.dflare_norm = _RMSNorm(hidden, eps)
        self.self_attn = Inkling2DraftSelfAttention(
            config, is_local, f"{prefix}.self_attn"
        )
        self.mlp = Inkling2DraftMLP(config, f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns this rank's partial MLP output and the residual."""
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x = all_reduce_add_rmsnorm(x, residual, self.input_layernorm)
        x = self.self_attn(positions, x, cos_sin_cache)
        x = all_reduce_add_rmsnorm(x, residual, self.post_attention_layernorm)
        return self.mlp(x), residual


class Inkling2MarkovHead(nn.Module):
    """Recurrent (`rnn_hidden`) correction of the draft's base logits.

    Its rank-`r` latent goes through `hidden_out` and the shared LM head,
    folded once into a `(vocab, r)` projection. The `[hidden, f_rnn]`
    columns of `rnn_proj` (and its bias) are applied to all draft steps at
    once in fp32; each step adds the `[state, previous]` columns and rounds.
    """

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        act_dtype = model_act_dtype()
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(
            config.padded_vocab_size,
            rank,
            _weight=torch.empty(config.padded_vocab_size, rank, dtype=act_dtype),
            _freeze=True,
        )
        self.rnn_proj = ReplicatedLinear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = ReplicatedLinear(rank, hidden, bias=True)
        self.dflare_norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.fold_weight: torch.Tensor | None = None
        self.fold_bias: torch.Tensor | None = None
        self.rnn_bias: torch.Tensor | None = None

    @torch.no_grad()
    def fold(self, lm_head_weight: torch.Tensor) -> None:
        """Fold `hidden_out` into the (TP-local) LM head rows in fp64. Later
        calls refill the same tensors, which CUDA graphs keep pointing at."""
        projection = self.hidden_out.weight.double()
        bias = self.hidden_out.bias.double()
        num_rows = lm_head_weight.shape[0]
        if self.fold_weight is None or self.fold_bias is None:
            self.fold_weight = lm_head_weight.new_empty((num_rows, self.markov_rank))
            self.fold_bias = lm_head_weight.new_empty((num_rows,))
        for start in range(0, num_rows, 4096):
            rows = slice(start, start + 4096)
            head = lm_head_weight[rows].double()
            self.fold_weight[rows].copy_(head @ projection)
            self.fold_bias[rows].copy_(head @ bias)

    def context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """fp32 `rnn_proj` bias plus its `[hidden, f_rnn]` term for the
        `(req, step)`-ordered `hidden`; returns `(step, req, 3 * r)`."""
        assert self.rnn_bias is not None
        num_reqs, width = f_rnn.shape
        hidden = hidden.view(num_reqs, -1, width).transpose(0, 1)
        inputs = torch.cat([hidden, f_rnn.expand_as(hidden)], dim=-1)
        weight = self.rnn_proj.weight[:, 2 * self.markov_rank :]
        out = torch.addmm(
            self.rnn_bias, inputs.flatten(0, 1), weight.T, out_dtype=torch.float32
        )
        return out.view(hidden.shape[0], num_reqs, -1)

    def step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step; returns the new state and the TP-local bias."""
        assert self.fold_weight is not None and self.fold_bias is not None
        inputs = torch.cat([state, self.markov_w1(prev_tokens)], dim=-1)
        weight = self.rnn_proj.weight[:, : 2 * self.markov_rank]
        raw = torch.addmm(context, inputs, weight.T, out_dtype=torch.float32)
        state, latent = markov_gate(raw, state)
        return state, torch.addmm(self.fold_bias, latent, self.fold_weight.T)


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
        self.norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.markov_head = Inkling2MarkovHead(config)
        act_dtype = model_act_dtype()
        self.fuse_weight = nn.Parameter(
            torch.zeros(
                config.num_hidden_layers + 1,
                len(config.target_layer_ids),
                dtype=act_dtype,
            ),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(
            torch.zeros(hidden, dtype=act_dtype), requires_grad=False
        )
        self.register_buffer("fuse_alpha", torch.empty_like(self.fuse_weight))
        # fp32 [cos | sin] table, read directly by the draft's RoPE kernels.
        self.cos_sin_cache = get_rope(
            config.head_dim,
            max_position=max_model_len,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            dtype=torch.float32,
        ).cos_sin_cache
        # Shared with the target model (see Inkling2DSparkModel.attach_target).
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None

    def mix(self, features: torch.Tensor, row: int, norm: _RMSNorm) -> torch.Tensor:
        targets = features.view(features.shape[0], -1, self.config.hidden_size)
        return mix_rmsnorm(targets, self.fuse_alpha[row], norm.weight, norm.eps)

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normalized target embedding of each block's anchor, then masks."""
        assert self.embed_tokens is not None and self.embed_norm is not None
        block = self.config.draft_block_size
        n = input_ids.numel()
        if n % block != 0:
            padded = ((n + block - 1) // block) * block
            pad_ids = input_ids.new_zeros((padded,))
            pad_ids[:n].copy_(input_ids.reshape(-1))
            ids = pad_ids.view(-1, block)
        else:
            ids = input_ids.view(-1, block)
        x = self.mask_embed.expand(ids.shape[0], block, -1).clone()
        x[:, 0] = self.embed_norm(self.embed_tokens(ids[:, 0].contiguous()))
        return x.view(-1, self.config.hidden_size)[:n]

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_block(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(positions, x, residual, self.cos_sin_cache)
        assert residual is not None
        return all_reduce_add_rmsnorm(x, residual, self.norm)


class Inkling2DSparkModel(nn.Module):
    """DSpark draft for Inkling2; plugged into 1Cat `DSparkProposer`."""

    sidecar_weight_file = "mtp.safetensors"
    has_own_embed_tokens = False
    has_own_lm_head = False
    expects_raw_aux_hidden_states = True
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
                "Inkling2 DSpark drafts up to max_position_embeddings="
                f"{config.max_position_embeddings} positions; set --max-model-len "
                "to at most that"
            )
        self.config = config
        self.model = Inkling2DSparkDecoder(
            config, vllm_config.model_config.max_model_len, f"{prefix}model"
        )
        self.lm_head: nn.Module | None = None
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        self.register_buffer(
            "_f_rnn",
            torch.zeros(max_reqs, config.hidden_size, dtype=model_act_dtype()),
            persistent=False,
        )

    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        """Share the target's embeddings, embed_norm and LM head, then
        derive folded Markov weights."""
        lang = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        self.model.embed_tokens = lang.model.embed_tokens
        self.model.embed_norm = lang.model.embed_norm
        self.lm_head = lang.lm_head
        assert self.lm_head is not None and self.lm_head.weight.dtype in (
            torch.bfloat16,
            torch.float16,
        )
        head = self.model.markov_head
        head.rnn_bias = torch.empty_like(head.rnn_proj.bias, dtype=torch.float32)
        self.refresh_derived_weights()

    @torch.no_grad()
    def refresh_derived_weights(self) -> None:
        """Re-derive, in place, the weights computed from the target's LM head
        and the draft's own weights; called again after the target's weights
        are replaced (reload, RL weight transfer)."""
        assert self.lm_head is not None
        head = self.model.markov_head
        assert head.rnn_bias is not None
        head.fold(self.lm_head.weight)
        head.rnn_bias.copy_(head.rnn_proj.bias)
        self.model.fuse_alpha.copy_(
            torch.softmax(self.model.fuse_weight.float(), dim=-1).to(model_act_dtype())
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Each draft layer mixes the raw target features itself.
        return hidden_states

    def prepare_context_anchor(self, anchor_rows: torch.Tensor) -> None:
        n = anchor_rows.shape[0]
        if n:
            self._f_rnn[:n].copy_(self.compute_f_rnn(anchor_rows))

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: dict[str, torch.Tensor]
        | list[torch.Tensor | None]
        | None = None,
    ) -> None:
        if isinstance(context_slot_mappings, dict):
            slot_list = [
                context_slot_mappings.get(layer.self_attn.attn.layer_name)
                for layer in self.model.layers
            ]
        else:
            slot_list = context_slot_mappings
        for i, layer in enumerate(self.model.layers):
            context = self.model.mix(context_states, i, layer.dflare_norm)
            layer.self_attn.store_context_kv(
                context,
                context_positions,
                self.model.cos_sin_cache,
                None if slot_list is None else slot_list[i],
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
        """TP-local activation-dtype base logits."""
        assert self.lm_head is not None
        return linear(hidden_states, self.lm_head.weight)

    def markov_step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the recurrent head; return the state and TP-local bias."""
        return self.model.markov_head.step(state, prev_tokens, context)

    def markov_logits(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Full-vocab softcapped logits."""
        logits = add_softcap(base_logits, bias, self.config.final_logit_softcapping)
        logits = tensor_model_parallel_all_gather(logits)
        return logits[:, : self.config.vocab_size]

    def markov_argmax(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """`markov_logits(...).argmax(-1)`, masking NaN to -inf."""
        logits = self.markov_logits(base_logits, bias)
        clean = torch.where(logits.isnan(), float("-inf"), logits)
        return clean.argmax(dim=-1)

    def sample_block_tokens(
        self,
        hidden_states: torch.Tensor,
        anchors: torch.Tensor,
        batch_size: int,
        n_spec: int,
        sample_from_logits_fn,
        sampling_metadata,
        active_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run Inkling2's recurrent Markov sampling across `active_steps` steps."""
        steps = n_spec if active_steps is None else max(1, min(n_spec, active_steps))
        base_logits = self.compute_base_logits(hidden_states).view(
            batch_size, n_spec, -1
        )
        context = self.model.markov_head.context(
            hidden_states, self._f_rnn[:batch_size]
        )
        vocab = self.model.markov_head.markov_w1.num_embeddings
        prev = anchors.to(torch.long).clamp(0, vocab - 1)
        state = hidden_states.new_zeros(batch_size, self.config.markov_rank)
        draft_tokens: list[torch.Tensor] = []
        draft_probs: list[torch.Tensor] | None = None
        for i in range(steps):
            state, bias = self.markov_step(state, prev, context[i])
            if (
                sampling_metadata is None
                or sampling_metadata.all_greedy
                or sample_from_logits_fn is None
            ):
                prev = self.markov_argmax(base_logits[:, i], bias)
                step_probs = None
            else:
                logits = self.markov_logits(base_logits[:, i], bias)
                prev, step_probs = sample_from_logits_fn(logits, sampling_metadata)
            draft_tokens.append(prev)
            if step_probs is not None:
                if draft_probs is None:
                    draft_probs = []
                draft_probs.append(step_probs)
        if steps < n_spec:
            pad_tok = torch.zeros_like(prev)
            for _ in range(n_spec - steps):
                draft_tokens.append(pad_tok)
                if draft_probs is not None:
                    draft_probs.append(torch.zeros_like(draft_probs[-1]))
        flat_tokens = torch.stack(draft_tokens, dim=1).reshape(-1)
        if draft_probs is None:
            return flat_tokens, None
        flat_probs = torch.stack(draft_probs, dim=1).flatten(0, 1).contiguous()
        return flat_tokens, flat_probs

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            name, shard = f"model.{name}", None
            for old, new, shard_id in self._stacked:
                if old in name:
                    name, shard = name.replace(old, new), shard_id
                    break
            param = params[name]
            if weight.dtype == torch.bfloat16 and param.dtype == torch.float16:
                weight = weight.to(torch.float16)
            elif weight.dtype != param.dtype:
                raise ValueError(f"{name}: expected {param.dtype}, got {weight.dtype}")
            if shard is None:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            else:
                param.weight_loader(param, weight, shard)
            loaded.add(name)
        return loaded
