# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent V4.1 DSpark5, sharing the target's GGUF embedding and head."""

from collections.abc import Iterable
from copy import copy

import regex as re
import torch
from torch import nn

import vllm.envs as envs
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.deepseek_v41 import collapse_hc
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import Mxfp4SM70MoEMethod
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.shared_weights import bind_shared
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from .attention_ops import (
    attend_draft_latents,
    attend_draft_latents_batch,
    insert_context_slots,
)
from .contract import validate_runtime_geometry
from .fp8_linear import V41Fp8LinearMethod
from .markov_fused import MarkovGreedy, markov_bias
from .markov_shard import MarkovShardGreedy
from .model import V41Attention, V41Block, V41MoE, V41Norm, _SharedAttention
from .quant_rope import apply_rope, attention_qdq, rope_table
from .query_graph import V41ContextGraphs, V41GreedyDraftGraph, V41QueryGraph

_BATCH_CONTEXT = envs.VLLM_V41_BATCH == "1"
# VLLM_V41_DRAFT_FUSED: draft layers use the target's fused HC transitions.
_DRAFT_FUSED = envs.VLLM_V41_DRAFT_FUSED == "1"

_MARKOV_FUSED = __import__("os").environ.get("VLLM_V41_MARKOV_FUSED", "1") == "1"
# Native-weights route: the drafter's MXFP4 experts on the skinny kernels.
_NATIVE_EXPERTS = envs.VLLM_V41_NATIVE == "1"
# Greedy B=1 drafts from vocabulary shards (markov_shard.py); see
# DSparkDeepseekV41ForCausalLM.prepare_markov_shard.
_MARKOV_SHARD = envs.VLLM_V41_MARKOV_SHARD == "1"
# Draft-only E4M3 copy of the head shard for the greedy draft (default off).
_DRAFT_HEAD_FP8 = envs.VLLM_V41_DRAFT_HEAD_FP8 == "1"
# main_proj (15360 -> 5120 FP8, 79 MB) split by output rows over TP and
# all-gathered, instead of read whole on every rank (default off).
_DRAFT_PROJ_TP = envs.VLLM_V41_DRAFT_PROJ_TP == "1"

logger = init_logger(__name__)


class V41DraftQuantConfig(Fp8Config):
    """Native FP8 dense and MXFP4 experts; no change to model identity."""

    is_scale_e8m0 = True

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            if _NATIVE_EXPERTS:
                from .native import V41NativeMoEMethod

                # The target's skinny kernels (one read per distinct expert).
                return V41NativeMoEMethod(layer.moe_config)
            return Mxfp4SM70MoEMethod(layer.moe_config)
        if isinstance(layer, LinearBase):
            return V41Fp8LinearMethod()
        return super().get_quant_method(layer, prefix)

    def is_mxfp4_quant(self, prefix, layer):
        return isinstance(layer, FusedMoE)


class V41DraftAttention(V41Attention):
    ring_capable = False  # draft context KV stays in its paged window cache

    def forward(self, x, positions, table):
        count = x.shape[0]
        # The draft attends only its sliding window: RoPE at the true positions
        # (the table rotates on the fly past 16K), as the context keys use.
        rotary_positions = positions
        qr = self.q_norm(self.wq_a(x))
        q = apply_rope(
            self.wq_b(qr).view(count, self.heads, 512), rotary_positions, table
        )
        kv = attention_qdq(
            apply_rope(self.kv_norm(self.wkv(x)), rotary_positions, table)
        )
        metadata = self.window_cache.metadata()
        if metadata is None:
            output = torch.zeros_like(q)
        else:
            valid = metadata.slot_mapping[:count] >= 0
        if metadata is not None and metadata.layout is not None:
            # VLLM_V41_BATCH: every request drafts the same number of queries.
            layout = metadata.layout
            output = attend_draft_latents_batch(
                q,
                kv,
                positions,
                valid,
                self.window_cache.kv_cache,
                metadata.block_table,
                self.attn_sink,
                metadata.block_size,
                layout,
                count // max(1, layout.num_reqs),
            )
        elif metadata is not None:
            output = attend_draft_latents(
                q,
                kv,
                positions,
                valid,
                self.window_cache.kv_cache,
                metadata.block_table,
                self.attn_sink,
                metadata.block_size,
            )
        return self.project_output(
            apply_rope(output, rotary_positions, table, inverse=True)
        )


class V41DraftMoE(V41MoE):
    def forward(self, x):
        routed, shared = self.experts.forward_partials(
            hidden_states=x, router_logits=self._draft_gate(x)
        )
        output = routed.float()
        if shared is not None:
            output = output + shared.float()
        from .native import tp_all_reduce

        return tp_all_reduce(output)

    def _draft_gate(self, x):
        """With native experts and VLLM_V41_H16: gate + top-k + slot plan in
        one launch (v41sk.gate_route), stashed for V41MoE._route."""
        if _NATIVE_EXPERTS and x.is_cuda:
            from .native import attach_h16_gate, gate_route

            if not getattr(self.gate, "v41_h16_checked", False):
                # First (eager, pre-capture) forward: exact FP16 gate copy.
                self.gate.v41_h16_checked = True
                attach_h16_gate(self.gate)
            fused = gate_route(self.gate, x, self.experts.top_k, self.scale)
            if fused is not None:
                self._fused_route = fused
                return fused[0]
        return self.gate(x)


class V41DraftBlock(V41Block):
    attention_cls = V41DraftAttention
    moe_cls = V41DraftMoE


class V41DraftModel(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        self.hidden_size = c.hidden_size
        # The proposer replaces these modules before profiling or generation.
        # Identity placeholders allocate no second embedding/head table.
        self.embed_tokens = nn.Identity()
        if _DRAFT_PROJ_TP:
            self.main_proj = ColumnParallelLinear(
                c.hidden_size * 3,
                c.hidden_size,
                bias=False,
                gather_output=False,
                return_bias=False,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".main_proj",
            )
        else:
            self.main_proj = ReplicatedLinear(
                c.hidden_size * 3,
                c.hidden_size,
                bias=False,
                return_bias=False,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".main_proj",
            )
        self.main_norm, self.norm = V41Norm(c.hidden_size), V41Norm(c.hidden_size)
        shared = _SharedAttention()
        self.layers = nn.ModuleList(
            [
                V41DraftBlock(vllm_config, i, shared, prefix + f".layers.{i}")
                for i in range(3)
            ]
        )
        self.markov_embed = nn.Embedding(c.vocab_size, 256, dtype=torch.float16)
        self.markov_bias = ReplicatedLinear(
            256,
            c.vocab_size,
            bias=False,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=prefix + ".markov_bias",
        )
        self.confidence = ReplicatedLinear(
            c.hidden_size + 256,
            1,
            bias=False,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=prefix + ".confidence",
        )
        self.register_buffer("rope", rope_table(), persistent=False)

    def combine_hidden_states(self, auxiliary):
        # Exact power-of-two scaling protects the FP16 projection output;
        # the following RMS normalization removes this common factor.
        if auxiliary.dtype != torch.float16:  # target stores pre-scaled FP16
            auxiliary = (auxiliary * (2.0**-6)).to(torch.float16)
        projected = self.main_proj(auxiliary)
        if _DRAFT_PROJ_TP:
            from .native import tp_all_gather

            projected = tp_all_gather(projected.contiguous())
        return self.main_norm(projected)

    def precompute_and_store_context_kv(self, states, positions, slots=None):
        for layer in self.layers:
            attention = layer.attn
            kv = attention_qdq(
                apply_rope(
                    attention.kv_norm(attention.wkv(states)), positions, self.rope
                )
            )
            if slots is not None:
                insert_context_slots(
                    attention.window_cache.kv_cache,
                    kv,
                    slots[attention.window_cache.prefix],
                )

    def forward(self, input_ids, positions, inputs_embeds=None):
        embeddings = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        hidden = embeddings.float()[:, None].expand(-1, 4, -1).contiguous()
        pre = torch.zeros(hidden.shape[0], 4, device=hidden.device, dtype=torch.float32)
        pre[:, 0] = 1
        if _DRAFT_FUSED and hidden.is_cuda:
            return self._forward_fused(hidden, positions, pre)
        for layer in self.layers:
            hidden, pre = layer(hidden, positions, pre, self.rope)
        # Confidence consumes the collapsed, unnormalized hidden state.
        return collapse_hc(hidden, pre)

    def _forward_fused(self, hidden, positions, pre):
        """VLLM_V41_DRAFT_FUSED: the target's fused HC transitions (in-place
        expand + next mixes + collapse per sublayer boundary) instead of the
        reference mixes / expand / collapse kernels per sublayer."""
        from vllm.model_executor.kernels.mhc.deepseek_v41 import expand_hc, hc_mixes
        from vllm.model_executor.kernels.mhc.deepseek_v41_fused import (
            hc_join,
            hc_transition,
        )

        layers = list(self.layers)
        state = None
        for i, layer in enumerate(layers):
            if state is None:
                hc_join()
                attn_mix = hc_mixes(
                    hidden, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
                )
                inputs = collapse_hc(hidden, pre, layer.attn_norm.weight)
            else:
                inputs, attn_mix = state
            y = layer.attn(inputs, positions, self.rope)
            inputs, ffn_mix = hc_transition(
                hidden,
                y,
                attn_mix[1],
                attn_mix[2],
                attn_mix[0],
                layer.ffn_norm.weight,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
            )
            y = layer.ffn(inputs)
            pre = ffn_mix[0]
            following = layers[i + 1] if i + 1 < len(layers) else None
            if following is not None:
                state = hc_transition(
                    hidden,
                    y,
                    ffn_mix[1],
                    ffn_mix[2],
                    pre,
                    following.attn_norm.weight,
                    following.hc_attn_fn,
                    following.hc_attn_scale,
                    following.hc_attn_base,
                )
            else:
                hc_join()
                hidden = expand_hc(y, hidden, ffn_mix[1], ffn_mix[2], inplace=True)
        # Confidence consumes the collapsed, unnormalized hidden state.
        return collapse_hc(hidden, pre)


def draft_weight_name(name):
    """Map only the three mtp stages; VL routing bias is explicitly unused."""
    match = re.fullmatch(r"mtp\.([0-2])\.(.+)", name)
    if match is None:
        return None
    stage, rest = int(match[1]), match[2]
    if rest == "ffn.gate.bias_vl":
        return None
    if rest.startswith(("main_proj.", "main_norm.")):
        if stage != 0:
            raise ValueError("V4.1 main projection must belong to draft stage zero")
        return "model." + rest
    for source, target in (
        ("norm.", "norm."),
        ("markov_head.embed.", "markov_embed."),
        ("markov_head.head.", "markov_bias."),
        ("confidence_head.proj.", "confidence."),
    ):
        if rest.startswith(source):
            if stage != 2:
                raise ValueError("V4.1 DSpark heads must belong to the final stage")
            return "model." + rest.replace(source, target, 1)
    for source, target in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
        rest = rest.replace(f".shared_experts.{source}.", f".shared_experts.{target}.")
        rest = re.sub(rf"(\.experts\.\d+\.){source}\.", rf"\g<1>{target}.", rest)
    return f"model.layers.{stage}.{rest}"


class DSparkDeepseekV41ForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        draft_config = vllm_config.speculative_config.draft_model_config
        c = self.config = draft_config.hf_text_config
        if (
            c.model_type != "deepseek_v41_text"
            or (
                c.hidden_size,
                c.num_hidden_layers,
                c.n_routed_experts,
                c.num_experts_per_tok,
            )
            != (5120, 3, 128, 3)
            or tuple(c.dspark_target_layer_ids) != (37, 38, 39)
            or c.dspark_block_size != 5
            or c.dspark_markov_rank != 256
            or get_tensor_model_parallel_world_size() != 4
            or draft_config.dtype != torch.float16
            or vllm_config.speculative_config.num_speculative_tokens != 5
            or any(c.compress_ratios)
            or c.engram_layer_ids
        ):
            raise ValueError("V4.1 DSpark requires native three-layer FP16 TP4 N=5")
        validate_runtime_geometry(c, draft=True)
        configuration = copy(vllm_config)
        configuration.model_config = draft_config
        configuration.quant_config = V41DraftQuantConfig.from_config(
            c.quantization_config
        )
        with set_current_vllm_config(configuration):
            # The proposer loads with an empty prefix by default. Keep the
            # three draft cache/MoE registrations distinct from target 0..39;
            # checkpoint parameter paths still belong to this model object.
            runtime_prefix = maybe_prefix(maybe_prefix(prefix, "dspark"), "model")
            self.model = V41DraftModel(configuration, runtime_prefix)
        self.lm_head = nn.Identity()
        self.logits_processor = LogitsProcessor(c.vocab_size)
        self._markov = None
        self._markov_shard = None
        self.query_graph = (
            None
            if vllm_config.speculative_config.enforce_eager
            else V41QueryGraph(self.model)
        )
        self.greedy_graph = (
            None
            if vllm_config.speculative_config.enforce_eager
            else V41GreedyDraftGraph(self._greedy_draft)
        )
        self.context_graphs = (
            None
            if vllm_config.speculative_config.enforce_eager
            else V41ContextGraphs(self.model.precompute_and_store_context_kv)
        )

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def bind_pipeline_embedding(self, target_language_model):
        """Borrow the target's embedding when the proposer's sharing did not.

        The draft has no embedding table (``embed_tokens`` is an Identity
        placeholder). The proposer's legacy sharing contract skips the
        embedding whenever pipeline parallelism is on, but V4.1 builds
        ``embed_tokens`` on every stage, including the last one, where the
        drafter runs. Returns True if it bound the target's module here.
        """
        if not isinstance(self.model.embed_tokens, nn.Identity):
            return False  # already shared (one stage)
        embed = getattr(target_language_model.model, "embed_tokens", None)
        if embed is None or isinstance(embed, PPMissingLayer):
            raise ValueError(
                "V4.1 DSpark needs the target embedding on the drafter's stage"
            )
        bind_shared(self.model, "embed_tokens", embed)
        return True

    def combine_hidden_states(self, auxiliary):
        return self.model.combine_hidden_states(auxiliary)

    def get_draft_kv_cache_layer_names(self):
        return [layer.attn.window_cache.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self, states, positions, context_slot_mappings=None
    ):
        rows = states.shape[0]
        if (
            self.context_graphs is not None
            and 1 <= rows <= 6
            and context_slot_mappings is not None
        ):
            self.context_graphs(states, positions, context_slot_mappings)
            return
        if (
            self.context_graphs is not None
            and _BATCH_CONTEXT
            and 6 < rows <= 192
            and context_slot_mappings is not None
        ):
            # Several requests (VLLM_V41_BATCH): pad to 12/24/48/96/192 rows
            # so a few captured graphs cover every batch; padding rows have
            # slot -1 (nothing stored).
            padded = 12
            while padded < rows:
                padded *= 2
            pad = padded - rows
            states = torch.cat([states, states.new_zeros(pad, states.shape[1])])
            positions = torch.cat([positions, positions.new_zeros(pad)])
            context_slot_mappings = {
                name: torch.cat([value, value.new_full((pad,), -1)])
                for name, value in context_slot_mappings.items()
            }
            self.context_graphs(states, positions, context_slot_mappings)
            return
        self.model.precompute_and_store_context_kv(
            states, positions, context_slot_mappings
        )

    def forward(self, input_ids, positions, inputs_embeds=None):
        # DFlash's allocation/profile dummy forward has no attention
        # metadata. Real query forwards always supply the three caches.
        if (
            self.query_graph is not None
            and is_forward_context_available()
            and get_forward_context().attn_metadata
        ):
            return self.query_graph(input_ids, positions, inputs_embeds)
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(
            self.lm_head, self.model.norm(hidden_states).to(torch.float16)
        )

    def local_logits(self, hidden_states):
        """This rank's vocabulary shard of ``compute_logits`` (no all-gather)."""
        x = self.model.norm(hidden_states).to(torch.float16)
        if _DRAFT_HEAD_FP8 and getattr(self, "_fp8_ready", False) and x.shape[0] <= 8:
            head = getattr(self, "_fp8_head", None)
            if head is None and self.lm_head.weight.is_cuda:
                from .fp8_linear import q8_channel_prepare

                # A draft-only E4M3 copy of this rank's head shard (half the
                # bytes); the target still verifies with the FP16 head.
                head = self._fp8_head = q8_channel_prepare(self.lm_head.weight.data)
            if head:
                from .fp8_linear import q8_channel_linear

                return q8_channel_linear(x, head, self.lm_head.weight.shape[0])
        return self.lm_head.quant_method.apply(self.lm_head, x, bias=None)

    def _greedy_draft(self, hidden_states, anchor):
        if self._markov_shard is not None:
            if anchor.numel() > 1:  # VLLM_V41_BATCH: one anchor per request
                return self._markov_shard.call_batch(
                    self.local_logits(hidden_states), anchor
                )
            return self._markov_shard(self.local_logits(hidden_states), anchor)
        logits = self.compute_logits(hidden_states)
        if self._markov is not None:
            return self._markov(logits, anchor)
        previous = anchor.to(torch.long)
        tokens = []
        for step in range(5):
            embedding = self.markov_embed(previous)
            previous = (logits[step : step + 1] + self.markov_bias(embedding)).argmax(
                -1
            )
            tokens.append(previous)
        return torch.stack(tokens, dim=1).reshape(-1)

    def greedy_draft_tokens(self, hidden_states, anchor):
        if self.greedy_graph is None:
            return None
        if anchor.numel() > 1 and self._markov_shard is None:
            return None  # batched drafts need the sharded Markov head
        return self.greedy_graph(hidden_states, anchor)

    def markov_embed(self, token_ids):
        # Under async scheduling a non-sampling step can carry an unfilled
        # anchor (read from the persistent query buffer at a request boundary);
        # its drafts are discarded, but the lookup must stay in range. The fused
        # greedy path clamps the same way; nn.Embedding (index_select) asserts.
        vocab = self.model.markov_embed.num_embeddings
        return self.model.markov_embed(token_ids.clamp(0, vocab - 1))

    def markov_bias(self, embedding):
        if self._markov is not None:
            return markov_bias(self._markov, embedding)
        return self.model.markov_bias(embedding.float())

    def confidence_logits(self, hidden, embedding):
        return self.model.confidence(
            torch.cat((hidden.float(), embedding.float()), -1)
        ).squeeze(-1)

    def map_draft_to_target(self, ids):
        return ids

    def skip_checkpoint_weight(self, name):
        return draft_weight_name(name) is None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        parameters = dict(self.named_parameters())
        expert_mapping = FusedMoE.make_expert_params_mapping(
            self, "gate_proj", "down_proj", "up_proj", 128
        )
        loaded = set()
        for source, value in weights:
            name = draft_weight_name(source)
            if name is None:
                continue
            expert = ".experts." in name
            if name.endswith(".scale"):
                name = name.removesuffix(".scale") + (
                    ".weight_scale" if expert else ".weight_scale_inv"
                )
                if expert:
                    value = value.view(torch.uint8)
            if expert:
                for target, checkpoint, expert_id, shard in expert_mapping:
                    if checkpoint not in name:
                        continue
                    mapped = name.replace(checkpoint, target)
                    parameter = parameters[mapped]
                    parameter.weight_loader(
                        parameter,
                        value,
                        mapped,
                        shard_id=shard,
                        expert_id=expert_id,
                    )
                    loaded.add(mapped)
                    break
                else:
                    raise ValueError(f"Unmapped V4.1 draft expert tensor: {source}")
            else:
                parameter = parameters[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, value)
                loaded.add(name)
        missing = set(parameters) - loaded
        if missing:
            raise ValueError(f"Missing V4.1 draft parameters: {sorted(missing)}")
        self._prepare_markov()
        return loaded

    def _prepare_markov(self):
        """Keep the Markov head in its checkpoint BF16 and fuse the greedy steps."""
        if not _MARKOV_FUSED:
            return
        head = self.model.markov_bias
        weight = head.weight.data
        if not weight.is_cuda:
            return
        self._markov = MarkovGreedy(
            weight.to(torch.bfloat16), self.model.markov_embed.weight.data
        )
        head.weight = nn.Parameter(
            torch.empty(0, dtype=torch.float32, device=weight.device), False
        )

    def prepare_markov_shard(self):
        """VLLM_V41_MARKOV_SHARD=1 (default 0): greedy drafts from vocab shards.

        The greedy B=1 draft graph then never builds full-vocabulary logits:
        each TP rank scores its own vocabulary rows (``markov_shard``) and one
        small all-reduce per step exchanges the ranks' best rows. Other draft
        paths are unchanged. The proposer calls this once after sharing the
        target's head. It checks the layout, then compares the shard path with
        the full fused path (local logits bit for bit; tokens from random hidden
        states and anchors and from forced ties, -inf and NaN; the winning
        values). Any problem on any rank makes every rank raise. Returns True
        when the shard path is on.
        """
        if not _MARKOV_SHARD:
            return False
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        shard = None
        try:
            problems = self._markov_shard_layout(tp, rank)
            if not problems:
                from .native import tp_all_reduce

                shard = MarkovShardGreedy(self._markov, rank, tp, tp_all_reduce)
                shard.warm()  # a rank-local launch failure becomes a problem
        except Exception as error:  # noqa: BLE001 - it must reach the all-reduce
            problems = [f"{type(error).__name__}: {error}"]
        self._markov_shard_agree(problems, "cannot run")
        assert shard is not None
        failures, cases = self._markov_shard_check(shard)
        self._markov_shard_agree(failures, "self-check failed")
        self._markov_shard = shard
        # Only after the bitwise self-check: the FP8 draft head is lossy.
        self._fp8_ready = _DRAFT_HEAD_FP8
        logger.info(
            "VLLM_V41_MARKOV_SHARD=1: greedy drafts score vocabulary rows %d-%d "
            "on TP rank %d of %d; self-check passed (%d cases)",
            shard.start,
            shard.start + shard.n - 1,
            rank,
            tp,
            cases,
        )
        return True

    def _markov_shard_layout(self, tp, rank):
        """Problems that keep the shard path from equalling the full path."""
        if self._markov is None:
            return ["it needs the fused Markov path (VLLM_V41_MARKOV_FUSED=1, CUDA)"]
        problems = []
        vocab = self._markov.vocab
        n, start = vocab // tp, rank * (vocab // tp)
        head, lp = self.lm_head, self.logits_processor
        indices = getattr(head, "shard_indices", None)
        if (
            vocab % tp
            or indices is None
            or head.org_vocab_size != vocab
            or head.num_embeddings_padded != vocab
            or head.num_embeddings_per_partition != n
            or indices.org_vocab_start_index != start
            or indices.org_vocab_end_index != start + n
            or indices.num_org_vocab_padding
        ):
            problems.append(f"the head is not {tp} unpadded vocab shards in order")
        if (
            lp.logits_as_input
            or lp.soft_cap is not None
            or lp.scale != 1.0
            or not lp.use_all_gather
            or lp.org_vocab_size != vocab
        ):
            problems.append("the logits processor is not a plain all-gather")
        return problems

    def _markov_shard_agree(self, problems, what):
        """Raise on every TP rank when any rank reports a problem."""
        device = self.model.norm.weight.device
        flag = torch.tensor([float(len(problems))], device=device)
        total = int(tensor_model_parallel_all_reduce(flag).item())
        if total:
            raise RuntimeError(
                f"VLLM_V41_MARKOV_SHARD=1 {what}: {total} problem(s) across the "
                f"TP group; this rank: {problems[:6] or 'none'}"
            )

    def _markov_shard_check(self, shard):
        """This rank's failures of the shard path against the full fused path.

        Every rank runs the same cases in the same order (one CPU seed), so
        the collectives pair up. Returns (failures, number of cases).
        """
        markov = self._markov
        assert markov is not None  # the shard path needs the fused Markov head
        vocab, n, start = shard.vocab, shard.n, shard.start
        device, steps, tp = markov.weight.device, shard.steps, shard.tp
        gen = torch.Generator(device="cpu").manual_seed(41)
        failures, cases = [], 0

        def bits(t):  # bitwise: signed zeros and NaN payloads included
            return t.contiguous().view(torch.uint8)

        def tokens(name, full, local, anchor, top=False):
            nonlocal cases
            cases += 1
            anchor = torch.tensor([anchor], dtype=torch.int64, device=device)
            want = markov(full, anchor)
            want_top = markov.best_val.max()  # the last step's winning value
            got = shard(local, anchor)
            if not torch.equal(want, got):
                failures.append(f"{name}: tokens {got.tolist()} != {want.tolist()}")
            if top:  # every rank's last-step slot, then the largest maximum
                slots = tensor_model_parallel_all_reduce(shard.slots[-1].clone())
                if not bool(slots[:, 0].max() == want_top):
                    failures.append(f"{name}: winning value")

        hidden = int(self.config.hidden_size)
        for case in range(3):
            h = (torch.randn((steps, hidden), generator=gen) * 4.0).to(torch.float16)
            h = h.to(device)
            full = self.compute_logits(h)
            local = self.local_logits(h)
            if not torch.equal(bits(full[:, start : start + n]), bits(local)):
                failures.append(f"hidden {case}: local logits")
            picks = torch.randint(0, vocab, (2,), generator=gen).tolist()
            for anchor in (*picks, 0, vocab - 1, -1, vocab + 3):
                top = anchor in picks
                tokens(f"hidden {case} anchor {anchor}", full, local, anchor, top)

        def rows():
            return (torch.randn((steps, vocab), generator=gen) * 8.0).half()

        inf, nan = float("inf"), float("nan")
        fixtures = {}
        r = fixtures["inf tie across ranks 0|1"] = rows()
        r[:, n - 1] = r[:, n] = inf
        r = fixtures["inf in rank 1 and the last rank"] = rows()
        r[:, n + n // 2] = r[:, vocab - 2] = inf
        r = fixtures["inf twice in the last rank"] = rows()
        r[:, (tp - 1) * n + 10] = r[:, (tp - 1) * n + 20] = inf
        r = fixtures["inf tie at each boundary"] = rows()
        for step in range(steps):
            edge = (step % (tp - 1) + 1) * n
            r[step, edge - 1] = r[step, edge] = inf
        fixtures["all -inf"] = torch.full((steps, vocab), -inf).half()
        r = fixtures["NaN every 7th row"] = rows()
        r[:, ::7] = nan
        r = fixtures["rank 1 all NaN"] = rows()
        r[:, n : 2 * n] = nan
        r = fixtures["NaN, then -inf in the last rank"] = rows()
        r[:, : (tp - 1) * n] = nan
        r[:, (tp - 1) * n :] = -inf
        r = fixtures["rank 0 NaN, the rest -inf"] = rows()
        r[:, :n] = nan
        r[:, n:] = -inf
        r = fixtures["NaN against +inf"] = rows()
        r[:, 5] = nan
        r[:, vocab - 3] = inf
        for name, full in fixtures.items():
            anchor = int(torch.randint(0, vocab, (1,), generator=gen))
            full = full.to(device)
            tokens(name, full, full[:, start : start + n], anchor)
        return failures, cases
