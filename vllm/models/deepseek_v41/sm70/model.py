# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 on SM70: decoder layer, model, causal LM, weight loading, PP payload (PORT_DESIGN §3.2, §3.5,
§3.7; owner L-CORE).

Attention, MoE and Engram come from the other lanes through the §3 signatures. They are imported when the model
is built (not at module import), so the registry can inspect this class before those lanes merge and tests can
substitute plain-torch stand-ins with the same signatures.

Numerics (§4.1): HC stream BF16 with FP32 math (common/hc.py), sublayer inputs FP16, sublayer outputs FP32,
FP16 head with FP32 logits, FP16 GEMMs with FP32 accumulation (reduced-precision reductions disabled).
"""

from __future__ import annotations

import os
import typing
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import regex as re
import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import PPMissingLayer, extract_layer_index, make_layers, maybe_prefix
from vllm.sequence import IntermediateTensors

from .. import knobs
from ..common import contracts as C
from ..common import dump as ds41_dump
from ..common.contracts import LayerTopology, SharedAttnBuffers, StagePlan, allocate_shared_attn_buffers
from ..common.hc import hc_collapse, hc_expand, hc_mixes, hc_post, hc_pre, rmsnorm_to_act
from ..common.topology import N_BACKBONE_LAYERS, layer_topology, make_stage_plan

if TYPE_CHECKING:
    from ..common.engram_host import EngramHostService
    from .moe import ExpertSpillPlan

# Model knobs (PORT_DESIGN §5.5: "--ds41-*" flags are L-CORE knobs)
ENGRAM_DIR_KNOB = "VLLM_DS41_CORE_ENGRAM_DIR"
SPILL_KNOB = "VLLM_DS41_CORE_SPILL_EXPERTS_PER_LAYER"
LAYER_SUBSET_KNOB = "VLLM_DS41_CORE_LAYER_SUBSET"   # e.g. "0,1,2,3": run only these backbone layers (I1 / smoke)
LAYER_SUBSET_ALLOW_KNOB = "VLLM_DS41_CORE_ALLOW_LAYER_SUBSET"   # must be 1 too: a subset is a debug-only partial model
# L-ATTN's cross-stage KV-mirror check (ds41/attn kv_mirror.py MIRROR_CHECK_ENV / PP_KEY_KV20_CRC; contracts.py is
# frozen, so the extra payload key lives here and is cross-checked against kv_mirror when the knob is on)
MIRROR_CHECK_KNOB = "VLLM_DS41_ATTN_MIRROR_CHECK"
KV20_CRC_KEY = "kv20_crc"

logger = init_logger(__name__)
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_SKIP_PREFIXES = ("vision.", "aligner.", "image_", "mtp.")


# ---------------------------------------------------------------- configuration checks
def check_contract_constants(config: Any) -> None:
    """The frozen contract constants (§3.1) must equal the checkpoint's config."""
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    pairs = (
        ("hidden_size", C.HIDDEN), ("hc_mult", C.HC), ("num_attention_heads", C.N_HEADS), ("head_dim", C.HEAD_DIM),
        ("qk_rope_head_dim", C.ROPE_DIM), ("q_lora_rank", C.Q_LORA), ("o_groups", C.O_GROUPS),
        ("o_lora_rank", C.O_LORA), ("sliding_window", C.WINDOW), ("index_n_heads", C.IDX_HEADS),
        ("index_head_dim", C.IDX_DIM), ("index_topk", C.IDX_TOPK), ("candidate_source_layer_id", C.CAND_SOURCE),
        ("candidate_block_size", C.CAND_BLOCK), ("candidate_topk_blocks", C.CAND_TOPK_BLOCKS),
        ("engram_head_dim", C.ENGRAM_HEAD_DIM), ("n_routed_experts", C.N_EXPERTS),
        ("num_experts_per_tok", C.TOP_K), ("moe_intermediate_size", C.MOE_INTER),
        ("routed_scaling_factor", C.ROUTED_SCALE), ("swiglu_limit", C.SWIGLU_LIMIT), ("rms_norm_eps", C.NORM_EPS),
        ("hc_eps", C.HC_EPS), ("hc_sinkhorn_iters", C.HC_SINKHORN_ITERS),
        ("compress_rope_theta", C.COMPRESS_ROPE_THETA), ("num_hidden_layers", N_BACKBONE_LAYERS),
    )
    bad = [f"{name}={getattr(config, name, None)!r} (contract {want!r})" for name, want in pairs
           if getattr(config, name, None) != want]
    if rope_theta != C.ROPE_THETA:
        bad.append(f"rope_theta={rope_theta!r} (contract {C.ROPE_THETA!r})")
    engram_subtables = (getattr(config, "engram_max_ngram_size", 0) - 1) * getattr(config, "engram_n_heads", 0)
    if engram_subtables != C.ENGRAM_SUBTABLES:
        bad.append(f"engram subtables {engram_subtables} (contract {C.ENGRAM_SUBTABLES})")
    if bad:
        raise ValueError("DeepSeek-V4.1 config does not match the frozen contracts: " + "; ".join(bad))


def pipeline_partition(num_layers: int, pp_size: int) -> list[int]:
    """Per-stage layer counts exactly as vLLM's make_layers will split them (VLLM_PP_LAYER_PARTITION aware)."""
    counts = []
    for rank in range(pp_size):
        start, end = get_pp_indices(num_layers, rank, pp_size)
        counts.append(end - start)
    return counts


def engram_io_threads(pp_size: int) -> int:
    """Engram pread threads per rank (D11 / AM-12): 2 under PP2, 1 under PP3+ (CPU budget: 12 cores for 8-12
    workers), the service default 4 without PP. VLLM_DS41_ENGRAM_IO_THREADS still overrides inside the service."""
    return 4 if pp_size == 1 else (2 if pp_size == 2 else 1)


def engram_row_dir(model_path: str) -> str:
    """Directory holding the Engram shards (the checkpoint shards that carry ``*.engram.embed.*``).

    VLLM_DS41_CORE_ENGRAM_DIR wins when set (e.g. shards placed on a faster disk); otherwise the model directory
    itself, which is where a plain Hugging Face snapshot keeps them. A model given as a hub id is not a directory
    and needs the knob: fail loud rather than guess."""
    explicit = knobs.env_str(ENGRAM_DIR_KNOB, "")
    if explicit:
        return explicit
    if not os.path.isdir(model_path):
        raise ValueError(
            f"DeepSeek-V4.1 Engram rows are read from the checkpoint shards on disk, but the model {model_path!r} is "
            f"not a local directory; download the checkpoint or set {ENGRAM_DIR_KNOB} to the directory of its shards")
    return model_path


def parse_layer_subset(stage: StagePlan, config: Any) -> frozenset[int] | None:
    """Optional layer subset (knob) restricted to this stage; must be closed under the source relation."""
    raw = knobs.env_str(LAYER_SUBSET_KNOB, "")
    if raw == "":
        return None
    if not knobs.env_bool(LAYER_SUBSET_ALLOW_KNOB, False):
        # MC-CORE F6: a subset inherited from the shell would silently serve a partial model
        raise ValueError(f"{LAYER_SUBSET_KNOB}={raw!r} is set but {LAYER_SUBSET_ALLOW_KNOB} is not 1: a layer subset runs a "
                         "partial (debug) model; set both to use it, or unset the subset")
    try:
        subset = frozenset(int(tok) for tok in raw.split(",") if tok.strip() != "")
    except ValueError as exc:
        raise ValueError(f"{LAYER_SUBSET_KNOB}={raw!r} must be comma-separated layer ids") from exc
    if not subset or any(not 0 <= layer < N_BACKBONE_LAYERS for layer in subset):
        raise ValueError(f"{LAYER_SUBSET_KNOB}={raw!r}: ids must be backbone layers 0..{N_BACKBONE_LAYERS - 1}")
    for layer in sorted(subset):
        topo = layer_topology(config, layer)
        for src in (topo.kv_source, topo.index_source):
            if src is not None and src not in subset:
                raise ValueError(f"{LAYER_SUBSET_KNOB}: layer {layer} needs source layer {src}, which is not in "
                                 f"the subset {sorted(subset)}")
    logger.warning("DeepSeek-V4.1 DEBUG layer subset active: running only backbone layers %s (partial model)",
                   sorted(subset))
    return frozenset(layer for layer in subset if stage.first_layer <= layer <= stage.last_layer)


# ---------------------------------------------------------------- modules
class DeepseekV41Norm(nn.Module):
    """Holds an RMSNorm weight in FP32 (§3.7); the math lives in common.hc.rmsnorm_to_act."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)


def _fp32_param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology, stage: StagePlan,
                 shared: SharedAttnBuffers, engram_service: EngramHostService | None,
                 spill: ExpertSpillPlan | None) -> None:
        super().__init__()
        from ..attention import DeepseekV41Attention
        from .moe import DeepseekV41MoE

        self.layer_id = topo.layer_id
        self.topo = topo
        self.engram: nn.Module | None = None
        if topo.has_engram:
            if engram_service is None:
                raise ValueError(f"{prefix}: Engram layer {topo.layer_id} built without an EngramHostService")
            from ..common.engram import DeepseekV41Engram

            self.engram = DeepseekV41Engram(vllm_config, f"{prefix}.engram", topo.layer_id, engram_service)
        self.attn = DeepseekV41Attention(vllm_config, f"{prefix}.attn", topo, stage, shared, aux_streams=None)
        self.ffn = DeepseekV41MoE(vllm_config, f"{prefix}.ffn", topo.layer_id, n_routed_experts=C.N_EXPERTS,
                                  top_k=C.TOP_K, spill=spill)
        self.attn_norm = DeepseekV41Norm(C.HIDDEN)
        self.ffn_norm = DeepseekV41Norm(C.HIDDEN)
        mix, hc_dim = (2 + C.HC) * C.HC, C.HC * C.HIDDEN
        self.hc_attn_fn = _fp32_param(mix, hc_dim)
        self.hc_ffn_fn = _fp32_param(mix, hc_dim)
        self.hc_attn_base = _fp32_param(mix)
        self.hc_ffn_base = _fp32_param(mix)
        self.hc_attn_scale = _fp32_param(3)
        self.hc_ffn_scale = _fp32_param(3)

    def forward(self, stream: torch.Tensor, pre_in: torch.Tensor, positions: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # stream [T,4,5120] bf16, pre_in [T,4] f32, positions [T] int64  ->  (stream' bf16, ffn_pre f32)
        dumping = ds41_dump.ENABLED
        layer = self.layer_id
        if dumping:
            ds41_dump.dump(layer, "stream_in", stream)
            ds41_dump.dump(layer, "pre_in", pre_in)
        if self.engram is not None:
            stream = self.engram(stream, positions)                    # before the block's mixes (ref:m.py:1262)
            if dumping:
                ds41_dump.dump(layer, "engram.out", stream)
        attn_pre, attn_post, attn_comb = hc_mixes(stream, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        collapsed = hc_pre(stream, pre_in)                                          # uses the PREVIOUS pre
        x = rmsnorm_to_act(collapsed, self.attn_norm.weight)
        attn_out = self.attn(positions, x)
        if dumping:
            for name, t in (("hc.attn_pre", attn_pre), ("hc.attn_post", attn_post), ("hc.attn_comb", attn_comb),
                            ("hc.attn_x", collapsed), ("attn.x", x), ("attn.out", attn_out)):
                ds41_dump.dump(layer, name, t)
        stream = hc_post(attn_out, stream, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = hc_mixes(stream, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        collapsed = hc_pre(stream, attn_pre)                                        # uses this block's attn pre
        x = rmsnorm_to_act(collapsed, self.ffn_norm.weight)
        if dumping:
            for name, t in (("stream_attn", stream), ("hc.ffn_pre", ffn_pre), ("hc.ffn_post", ffn_post),
                            ("hc.ffn_comb", ffn_comb), ("hc.ffn_x", collapsed), ("moe.x", x)):
                ds41_dump.dump(layer, name, t)
        moe_out = self.ffn(x)
        stream = hc_post(moe_out, stream, ffn_post, ffn_comb)
        if dumping:
            for name, t in (("moe.out", moe_out), ("stream_out", stream), ("pre_out", ffn_pre)):
                ds41_dump.dump(layer, name, t)
        return stream, ffn_pre


class DeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        check_contract_constants(config)
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError("DeepSeek-V4.1 on SM70 requires --dtype half (no BF16 tensor cores on Volta); got "
                             f"{vllm_config.model_config.dtype}")
        if vllm_config.parallel_config.use_ubatching:
            # gpu_ubatch_wrapper builds per-ubatch forward contexts without ForwardContext.is_dummy_run, which
            # Engram keys on (PORT_DESIGN §9 AM-2), and the Engram step binding assumes one forward per step.
            raise NotImplementedError("DeepSeek-V4.1 does not support DBO / micro-batching (--enable-dbo, "
                                      "ubatch_size > 1)")
        if ds41_dump.ENABLED and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError(f"{ds41_dump.DIR_KNOB} (golden dump) needs eager execution: launch with --enforce-eager")
        # §4.1 MUST: FP16 GEMMs accumulate in FP32 without reduced-precision split-K reductions
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        self.config = config
        self.vllm_config = vllm_config
        pp = get_pp_group()
        self.pp_is_first, self.pp_is_last = pp.is_first_rank, pp.is_last_rank
        partition = pipeline_partition(config.num_hidden_layers, pp.world_size)
        self.stage = make_stage_plan(config, pp.rank_in_group, pp.world_size, partition)
        self.layer_subset = parse_layer_subset(self.stage, config)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        device = torch.get_default_device()
        self.shared = allocate_shared_attn_buffers(max_tokens, self.stage, device)

        self._engram_service: EngramHostService | None = None
        engram_layers = tuple(layer for layer in self.stage.engram_layers if self._in_subset(layer))
        if engram_layers:
            from ..common.engram_host import EngramHostService

            self._engram_service = EngramHostService(
                config, engram_layers, get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size(),
                engram_row_dir(vllm_config.model_config.model), vllm_config.model_config.tokenizer,
                max_tokens, device, io_threads=engram_io_threads(pp.world_size))
        self.spill: ExpertSpillPlan | None = None
        n_spill = knobs.env_int(SPILL_KNOB, 0, minimum=0, maximum=C.N_EXPERTS - 1)
        if n_spill > 0:
            from .moe import make_spill_plan

            self.spill = make_spill_plan(n_spill, C.N_EXPERTS)

        if self.pp_is_first:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                       prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        def build(prefix: str) -> nn.Module:
            layer_id = extract_layer_index(prefix)
            if not self._in_subset(layer_id):
                return PPMissingLayer()
            return DeepseekV41DecoderLayer(vllm_config, prefix, layer_topology(config, layer_id), self.stage,
                                           self.shared, self._engram_service, self.spill)

        layers_prefix = f"{prefix}.layers"
        self.start_layer, self.end_layer, self.layers = make_layers(config.num_hidden_layers, build,
                                                                    prefix=layers_prefix)
        if (self.start_layer, self.end_layer) != (self.stage.first_layer, self.stage.last_layer + 1):
            raise RuntimeError(f"make_layers range [{self.start_layer},{self.end_layer}) != stage plan "
                               f"[{self.stage.first_layer},{self.stage.last_layer}]")
        # with VLLM_DS41_ATTN_MIRROR_CHECK=1 the kv-source-20 boundary also carries kv20_crc [T] int32
        self.kv20_crc_check = knobs.env_bool(MIRROR_CHECK_KNOB, False)
        self._kv20_crc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        if self.kv20_crc_check and C.CAND_SOURCE in self.stage.exports_kv_sources + self.stage.mirrored_kv_sources:
            from .. import kv_mirror

            if (kv_mirror.PP_KEY_KV20_CRC, kv_mirror.MIRROR_CHECK_ENV) != (KV20_CRC_KEY, MIRROR_CHECK_KNOB):
                raise RuntimeError(f"kv_mirror's CRC key/knob {kv_mirror.PP_KEY_KV20_CRC!r}/"
                                   f"{kv_mirror.MIRROR_CHECK_ENV!r} != the model's {KV20_CRC_KEY!r}/"
                                   f"{MIRROR_CHECK_KNOB!r}")
            self._kv20_crc = kv_mirror.kv20_crc
        self.mirror: nn.Module | None = None
        if C.CAND_SOURCE in self.stage.mirrored_kv_sources:
            from ..kv_mirror import DeepseekV41KVSourceMirror

            self.mirror = DeepseekV41KVSourceMirror(vllm_config, C.CAND_SOURCE, self.shared)
        self.norm = DeepseekV41Norm(C.HIDDEN) if self.pp_is_last else PPMissingLayer()

    def _in_subset(self, layer_id: int) -> bool:
        return self.layer_subset is None or layer_id in self.layer_subset

    @property
    def engram_service(self) -> EngramHostService | None:
        return self._engram_service

    def owns_layer(self, layer_id: int) -> bool:
        return self.stage.first_layer <= layer_id <= self.stage.last_layer and self._in_subset(layer_id)

    # ---- PP payload (§3.5) ----
    def _schema(self, num_tokens: int, with_kv20: bool) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        schema = {
            C.PP_KEY_HIDDEN: ((num_tokens, C.HC, C.HIDDEN), C.STREAM_DTYPE),
            C.PP_KEY_PRE_MIX: ((num_tokens, C.HC), C.MIX_DTYPE),
        }
        if with_kv20:
            schema[C.PP_KEY_KV20_CKV] = ((num_tokens, C.CKV_RECORD_DIM), C.KV_RECORD_DTYPE)
            schema[C.PP_KEY_KV20_IK] = ((num_tokens, C.IK_RECORD_DIM), C.KV_RECORD_DTYPE)
            schema[C.PP_KEY_CAND20] = ((num_tokens, C.CAND_TOPK_BLOCKS), torch.int32)
            if self.kv20_crc_check:
                schema[KV20_CRC_KEY] = ((num_tokens,), torch.int32)
        return schema

    def pp_static_schema(self, num_tokens: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Keys/shapes/dtypes of the boundary INTO this stage (receive side), in send order."""
        if self.pp_is_first:
            return {}
        return self._schema(num_tokens, C.CAND_SOURCE in self.stage.mirrored_kv_sources)

    def pp_send_schema(self, num_tokens: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Keys/shapes/dtypes this stage sends to the next one (== the next stage's pp_static_schema)."""
        if self.pp_is_last:
            return {}
        return self._schema(num_tokens, C.CAND_SOURCE in self.stage.exports_kv_sources)

    def make_empty_intermediate_tensors(self, batch_size: int, dtype: torch.dtype,
                                        device: torch.device) -> IntermediateTensors:
        # keys for the boundary INTO this stage; ``dtype`` is ignored (the stream is BF16, the mixes FP32)
        schema = self._schema(batch_size, C.CAND_SOURCE in self.stage.mirrored_kv_sources)
        return IntermediateTensors({key: torch.zeros(shape, dtype=dt, device=device)
                                    for key, (shape, dt) in schema.items()})

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor | IntermediateTensors:
        if ds41_dump.ENABLED:
            ds41_dump.begin_step(positions, positions.shape[0])
        if self.pp_is_first:
            embedded = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            stream, pre = hc_expand(embedded)
        else:
            if intermediate_tensors is None:
                raise RuntimeError("non-first DeepSeek-V4.1 stage called without intermediate tensors")
            stream = intermediate_tensors[C.PP_KEY_HIDDEN]
            pre = intermediate_tensors[C.PP_KEY_PRE_MIX]
            if self.mirror is not None:
                self.mirror.ingest(positions, intermediate_tensors[C.PP_KEY_KV20_CKV],
                                   intermediate_tensors[C.PP_KEY_KV20_IK], intermediate_tensors[C.PP_KEY_CAND20],
                                   crc=intermediate_tensors[KV20_CRC_KEY] if self.kv20_crc_check else None)
        for layer in self.layers[self.start_layer:self.end_layer]:
            if isinstance(layer, PPMissingLayer):   # layer outside the configured subset
                continue
            stream, pre = layer(stream, pre, positions)
        if not self.pp_is_last:
            num_tokens = stream.shape[0]
            out = {C.PP_KEY_HIDDEN: stream, C.PP_KEY_PRE_MIX: pre}
            if C.CAND_SOURCE in self.stage.exports_kv_sources:
                assert self.shared.export_ckv is not None and self.shared.export_ik is not None
                out[C.PP_KEY_KV20_CKV] = self.shared.export_ckv[:num_tokens]
                out[C.PP_KEY_KV20_IK] = self.shared.export_ik[:num_tokens]
                out[C.PP_KEY_CAND20] = self.shared.candidate_blocks[:num_tokens]
                if self.kv20_crc_check:
                    assert self._kv20_crc is not None
                    out[KV20_CRC_KEY] = self._kv20_crc(out[C.PP_KEY_KV20_CKV], out[C.PP_KEY_KV20_IK])
            if ds41_dump.ENABLED:
                ds41_dump.end_forward()
            return IntermediateTensors(out)
        if not ds41_dump.ENABLED:
            return hc_collapse(stream, pre, self.norm.weight)
        collapsed = hc_pre(stream, pre)
        hidden = rmsnorm_to_act(collapsed, self.norm.weight)
        for name, t in (("final.stream_in", stream), ("final.hc", collapsed), ("final.h", hidden)):
            ds41_dump.dump(None, name, t)
        ds41_dump.end_forward()
        return hidden


# ---------------------------------------------------------------- checkpoint names (§3.7)
_STACKED = (
    # (vLLM param fragment, checkpoint fragment, shard id)
    ("attn.fused_wqa_wkv.", "attn.wq_a.", 0),
    ("attn.fused_wqa_wkv.", "attn.wkv.", 1),
    ("compressor.fused_wkv_wgate.", "compressor.wkv.", 0),
    ("compressor.fused_wkv_wgate.", "compressor.wgate.", 1),
    ("shared_experts.gate_up_proj.", "shared_experts.w1.", 0),
    ("shared_experts.gate_up_proj.", "shared_experts.w3.", 1),
)
_STACKED_SHARDS: dict[str, dict[int, str]] = {}   # fused param fragment -> {shard id: checkpoint fragment}
for _param_frag, _ckpt_frag, _shard in _STACKED:
    _STACKED_SHARDS.setdefault(_param_frag, {})[_shard] = _ckpt_frag
_RENAMES = (("shared_experts.w2.", "shared_experts.down_proj."),)
_EXPERT_SCALE_RE = re.compile(r"(\.experts\.\d+\.w[123])\.scale$")
_EXPERT_KEY_RE = re.compile(r"experts\.\d+\.w[123]\.")


def map_checkpoint_name(name: str) -> str:
    """HF/inference name -> vLLM parameter name (before stacking)."""
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if not _LAYER_RE.match(name):
        raise KeyError(f"unexpected DeepSeek-V4.1 checkpoint tensor {name!r}")
    mapped = "model." + name
    if _EXPERT_SCALE_RE.search(mapped):
        mapped = _EXPERT_SCALE_RE.sub(r"\1.weight_scale", mapped)
    elif mapped.endswith(".scale"):
        mapped = mapped[: -len(".scale")] + ".weight_scale_inv"
    if mapped.endswith(".ffn.gate.bias"):
        mapped = mapped[: -len(".bias")] + ".e_score_correction_bias"
    for old, new in _RENAMES:
        mapped = mapped.replace(old, new)
    return mapped


class DeepseekV41ForCausalLM(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = DeepseekV41Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if self.model.pp_is_last:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
            if self.lm_head.num_embeddings_padded != config.vocab_size:
                # compute_logits all-gathers the per-rank shards and trims once: valid only without padding
                raise ValueError(f"LM head pads the vocabulary {config.vocab_size} -> "
                                 f"{self.lm_head.num_embeddings_padded}; compute_logits does not support padding")
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors  # type: ignore[method-assign]
        self.pp_static_schema = self.model.pp_static_schema
        self.pp_send_schema = self.model.pp_send_schema
        self._delegates: dict[str, nn.Module] = {}

    @property
    def engram_service(self) -> EngramHostService | None:
        return self.model.engram_service

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[T,5120] fp16 -> [T,129280] f32: FP16 operands, FP32 accumulate AND FP32 output (§4.1)."""
        weight = self.lm_head.weight
        if hidden_states.is_cuda:
            local = torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
        else:   # CPU (unit tests): same math -- FP16 values are exact in FP32
            local = torch.mm(hidden_states.float(), weight.float().t())
        if get_tensor_model_parallel_world_size() > 1:
            local = tensor_model_parallel_all_gather(local, dim=-1)
        logits = local[:, : self.config.vocab_size]
        if ds41_dump.ENABLED:
            ds41_dump.dump(None, "logits", logits)
            ds41_dump.end_logits()
        return logits

    # ---- loading ----
    def skip_checkpoint_weight(self, name: str) -> bool:
        if name.startswith(_SKIP_PREFIXES) or ".engram.embed." in name or name.endswith(".bias_vl"):
            return True
        match = _LAYER_RE.match(name)
        if match is not None:
            return not self.model.owns_layer(int(match.group(1)))
        if name == "embed.weight":
            return not self.model.pp_is_first
        if name in ("head.weight", "norm.weight"):
            return not self.model.pp_is_last
        return False

    def _delegate_for(self, param_name: str) -> str | None:
        """Prefix of the deepest lane module that loads its own tensors (``load_weights``), e.g. Engram."""
        parts = param_name.split(".")
        for cut in range(len(parts) - 1, 1, -1):
            prefix = ".".join(parts[:cut])
            module = self._delegates.get(prefix)
            if module is not None:
                return prefix
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        self._delegates = {name: module for name, module in self.named_modules()
                           if name.startswith("model.layers.") and name.endswith(".engram")
                           and callable(getattr(module, "load_weights", None))}
        expert_mapping: dict[str, tuple[str, int, str]] = {}
        # N1: a fused / expert-stacked parameter is complete only when EVERY shard reached it -- one shard marks the
        # parameter "loaded" but leaves the other rows torch.empty garbage. parts[param] = the shard keys it got,
        # expected[param] = the full key set (stacked: shard ids; routed experts: (expert_id, shard_id) pairs).
        parts: dict[str, set[object]] = {}
        expected: dict[str, frozenset[object]] = {}
        expert_keys: dict[str, set[object]] = {}
        if any(isinstance(m, DeepseekV41DecoderLayer) for m in self.model.layers):
            from .moe import DeepseekV41MoE

            for param_name, weight_name, expert_id, shard_id in DeepseekV41MoE.expert_params_mapping(C.N_EXPERTS):
                if not _EXPERT_KEY_RE.fullmatch(weight_name):
                    raise ValueError(f"unexpected expert mapping entry {weight_name!r} (expected experts.<e>.w<k>.)")
                expert_mapping[weight_name] = (param_name, expert_id, shard_id)
                expert_keys.setdefault(param_name, set()).add((expert_id, shard_id))
        tp_rank, tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        loaded: set[str] = set()
        delegated: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for ckpt_name, tensor in weights:
            if self.skip_checkpoint_weight(ckpt_name):
                continue
            name = map_checkpoint_name(ckpt_name)
            delegate = self._delegate_for(name)
            if delegate is not None:
                delegated.setdefault(delegate, []).append((ckpt_name.split(".engram.", 1)[1], tensor))
                continue
            if ".ffn.experts." in name:
                mapped, param_frag, key = self._load_expert(params, name, tensor, expert_mapping)
                loaded.add(mapped)
                parts.setdefault(mapped, set()).add(key)
                expected[mapped] = frozenset(expert_keys[param_frag])
                continue
            for param_frag, ckpt_frag, shard_id in _STACKED:
                if ckpt_frag in name:
                    stacked = name.replace(ckpt_frag, param_frag)
                    if stacked in params:   # layer 20's compressor has wkv only (no fused wkv/wgate)
                        param = params[stacked]
                        param.weight_loader(param, tensor, shard_id)
                        loaded.add(stacked)
                        parts.setdefault(stacked, set()).add(shard_id)
                        expected[stacked] = frozenset(_STACKED_SHARDS[param_frag])
                        break
            else:
                if name not in params:
                    raise KeyError(f"checkpoint tensor {ckpt_name!r} -> {name!r} has no parameter in this stage")
                param = params[name]
                loader = getattr(param, "weight_loader", None)
                if loader is None and name.endswith(".attn.attn_sink"):
                    heads = C.N_HEADS // tp_size
                    tensor = tensor.narrow(0, heads * tp_rank, heads)
                (loader or default_weight_loader)(param, tensor)
                loaded.add(name)
        starved = sorted(set(self._delegates) - set(delegated))
        if starved:
            raise RuntimeError(f"no checkpoint tensors reached {starved} (Engram q/k/wkv live in shards 47/48: are "
                               "they present and verified in the model directory?)")
        for prefix, items in delegated.items():
            # PORT_DESIGN §9 AM-1: the module returns the module-relative checkpoint names it consumed
            module = self._delegates[prefix]
            consumed = set(module.load_weights(items))
            given = {rel for rel, _ in items}
            # the delegate may also report the parameters it completed (L-ENGRAM returns 'wkv_r', 'qk' for vLLM's
            # every-parameter-loaded check); those are not checkpoint names. Anything else unknown still raises.
            consumed -= {name for name, _ in module.named_parameters()} - given
            if consumed != given:
                raise RuntimeError(f"{prefix}.load_weights consumed {sorted(consumed)} of {sorted(given)}; "
                                   f"unconsumed {sorted(given - consumed)}, unknown {sorted(consumed - given)}")
            loaded.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
        self._check_complete(loaded, {name: expected[name] - got for name, got in parts.items()})
        return loaded

    @staticmethod
    def _describe_missing_parts(name: str, missing: set[object]) -> str:
        keys = sorted(missing, key=repr)
        if all(isinstance(k, tuple) for k in keys):     # routed experts: (expert_id, shard_id)
            shown = ", ".join(f"expert {e} {shard}" for e, shard in keys[:8])
        else:
            frag = next(f for f in _STACKED_SHARDS if f in name)
            shown = ", ".join(_STACKED_SHARDS[frag][k].rstrip(".") for k in keys[:8])
        return f"{name} lacks {shown}" + (f", ... ({len(keys)} total)" if len(keys) > 8 else "")

    def _check_complete(self, loaded: set[str], missing_parts: dict[str, set[object]] | None = None) -> None:
        """Every parameter that needs checkpoint data must have been filled. vLLM's own completeness check is off
        whenever a quantization method is set (always, for this model), so a missing shard would otherwise leave
        torch.empty garbage / initial values and the server would come up (rule 9). Exempt: zero-size parameters and
        names a module lists in ``ds41_no_checkpoint_params`` (module-relative; parameters it derives itself)."""
        derived: set[str] = set()
        for mod_name, module in self.named_modules():
            for rel in getattr(module, "ds41_no_checkpoint_params", ()):
                derived.add(f"{mod_name}.{rel}" if mod_name else rel)
        missing = sorted(name for name, param in self.named_parameters()
                         if name not in loaded and name not in derived and param.numel() > 0)
        partial = sorted((name, keys) for name, keys in (missing_parts or {}).items() if keys)
        problems: list[str] = []
        if missing:
            shown = ", ".join(missing[:20]) + (f", ... ({len(missing)} total)" if len(missing) > 20 else "")
            problems.append(f"did not provide {len(missing)} parameter(s) of this stage: {shown}")
        if partial:
            shown = "; ".join(self._describe_missing_parts(name, keys) for name, keys in partial[:20])
            more = f"; ... ({len(partial)} total)" if len(partial) > 20 else ""
            problems.append(f"filled only part of {len(partial)} fused/expert parameter(s): {shown}{more}")
        if problems:
            raise RuntimeError(f"DeepSeek-V4.1 checkpoint {' and '.join(problems)}. Check that all 48 shards are "
                               "present and verified in the model directory.")

    @staticmethod
    def _load_expert(params: dict[str, nn.Parameter], name: str, tensor: torch.Tensor,
                     expert_mapping: dict[str, tuple[str, int, str]]) -> tuple[str, str, tuple[int, str]]:
        """Load one routed-expert tensor; returns (parameter name, mapping param fragment, (expert_id, shard_id))."""
        if name.endswith(".weight_scale") and tensor.dtype == torch.float8_e8m0fnu:
            tensor = tensor.view(torch.uint8)   # keep raw E8M0 bytes (copy_ would convert values)
        match = _EXPERT_KEY_RE.search(name)
        entry = expert_mapping.get(match.group(0)) if match is not None else None
        if entry is None:
            raise KeyError(f"routed-expert tensor {name!r} has no entry in DeepseekV41MoE.expert_params_mapping")
        param_name, expert_id, shard_id = entry
        mapped = name.replace(match.group(0), param_name)
        if mapped not in params:
            raise KeyError(f"routed-expert tensor {name!r} -> {mapped!r} has no parameter in this stage")
        param = params[mapped]
        loader = typing.cast(Callable[..., bool], param.weight_loader)
        if not loader(param, tensor, mapped, shard_id=shard_id, expert_id=expert_id, return_success=True):
            raise RuntimeError(f"expert weight_loader refused {name!r} -> {mapped!r} (expert {expert_id})")
        return mapped, param_name, (expert_id, shard_id)
