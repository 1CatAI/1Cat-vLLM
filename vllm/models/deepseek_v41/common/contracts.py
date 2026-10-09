# vllm/models/deepseek_v41/common/contracts.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ---- model constants: asserted against hf_config in DeepseekV41Model.__init__ ----
HIDDEN = 5120
HC = 4
N_HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = 448
Q_LORA = 1280
O_GROUPS = 8
O_LORA = 1024
WINDOW = 128
IDX_HEADS = 32
IDX_DIM = 128
IDX_TOPK = 512
CAND_SOURCE = 20
CAND_BLOCK = 8
CAND_TOPK_BLOCKS = 2048
KV_SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
ENGRAM_LAYERS = (1, 14)
ENGRAM_SUBTABLES = 24          # (max_ngram 4 - 1) orders x 8 heads, order-major: s = (n - 2) * 8 + head
ENGRAM_ROW_BYTES = 264         # 256 x E4M3 + 8 x UE8M0 (one per 32 channels)
ENGRAM_HEAD_DIM = 256
N_EXPERTS = 384
TOP_K = 6
MOE_INTER = 2304
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0
NORM_EPS = 1e-20
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20
COMPRESS_ROPE_THETA = 160000.0
ROPE_THETA = 10000.0

# ---- dtypes (numerics contract, PORT_DESIGN §4) ----
STREAM_DTYPE = torch.bfloat16        # HC residual stream [T, HC, HIDDEN]; software BF16 on SM70
ACT_DTYPE = torch.float16            # sublayer inputs and GEMM operands
SUBLAYER_OUT_DTYPE = torch.float32   # attention (after wo_b) and MoE outputs, TP-all-reduced
MIX_DTYPE = torch.float32            # pre / post / comb

# ---- KV records: FP16 rows holding QAT values exactly (PORT_DESIGN A3) ----
KV_RECORD_DTYPE = torch.float16
SWA_RECORD_DIM = 512                 # window K=V: E4M3 x UE8M0/32 QAT over all 512 dims      -> 1024 B/token/layer
CKV_RECORD_DIM = 512                 # compressed latent: E2M1 x E4M3/16 QAT, RoPE applied   -> 1024 B/latent
IK_RECORD_DIM = 128                  # index key: E2M1 x UE8M0/32 QAT, RoPE applied          ->  256 B/latent
KV_MODEL_VERSION = "deepseek_v41"
SWA_CACHE_DTYPE_STR = "v41_fp16_qat_swa"
CKV_CACHE_DTYPE_STR = "v41_fp16_qat_ckv"
IK_CACHE_DTYPE_STR = "v41_fp16_qat_ik"
CAND_ALL = -2                        # cand row[0] sentinel: <= CAND_TOPK_BLOCKS blocks reachable, no masking


@dataclass(frozen=True)
class LayerTopology:
    layer_id: int
    compress_ratio: int              # 0 = SWA only, 1 = full-length compressed, 2 = 2:1 pooled
    mode: str                        # "swa" | "full" | "reindex" | "reuse"
    kv_source: int | None            # max(s in KV_SOURCES if s <= layer_id) when compress_ratio > 0
    index_source: int | None         # max(s in INDEX_SOURCES if s <= layer_id) when compress_ratio > 0
    owns_compressor: bool            # layer_id in KV_SOURCES
    owns_indexer: bool               # layer_id in INDEX_SOURCES
    is_candidate_source: bool        # layer_id == CAND_SOURCE
    uses_candidates: bool            # owns_indexer and CAND_SOURCE < layer_id
    has_engram: bool                 # layer_id in ENGRAM_LAYERS
    rope_theta: float                # COMPRESS_ROPE_THETA if compress_ratio > 0 else ROPE_THETA
    yarn: bool                       # compress_ratio > 0


@dataclass(frozen=True)
class StagePlan:
    pp_rank: int
    pp_size: int
    first_layer: int                 # inclusive
    last_layer: int                  # inclusive (backbone layers only; DSpark rides on the last stage)
    mirrored_kv_sources: tuple[int, ...]   # kv sources on an earlier stage replicated here (v1: () or (20,))
    exports_kv_sources: tuple[int, ...]    # kv sources on this stage that a later stage mirrors (v1: () or (20,))
    engram_layers: tuple[int, ...]


# legal first layers of a stage (v1): SWA-only layer 1, index sources; any source referenced from
# before the cut must be 20 (mirrored). PP2 -> cut 20; PP3 -> cuts (14, 28) is the only split with <= 17 layers/stage.
LEGAL_STAGE_STARTS = (1, 2, 8, 14, 20, 24, 28, 32, 36)


@dataclass
class SharedAttnBuffers:
    """Allocated once per stage by DeepseekV41Model, handed to every attention layer and the mirror."""
    topk_indices: torch.Tensor           # [max_num_batched_tokens, IDX_TOPK] int32: logical compressed positions
                                         #   (ascending, -1 pad) published by the latest index source, read by Reuse layers
    candidate_blocks: torch.Tensor       # [max_num_batched_tokens, CAND_TOPK_BLOCKS] int32: block ids (ascending, -1 pad);
                                         #   row[0] == CAND_ALL => no mask. Written by layer 20 or the mirror.
    export_ckv: torch.Tensor | None      # [max_num_batched_tokens, CKV_RECORD_DIM] fp16 (stage exporting source 20 only)
    export_ik: torch.Tensor | None       # [max_num_batched_tokens, IK_RECORD_DIM] fp16 (stage exporting source 20 only)


def allocate_shared_attn_buffers(max_num_batched_tokens: int, stage: StagePlan,
                                 device: torch.device) -> SharedAttnBuffers:
    exports = CAND_SOURCE in stage.exports_kv_sources
    return SharedAttnBuffers(
        topk_indices=torch.full((max_num_batched_tokens, IDX_TOPK), -1, dtype=torch.int32, device=device),
        candidate_blocks=torch.full((max_num_batched_tokens, CAND_TOPK_BLOCKS), -1, dtype=torch.int32, device=device),
        export_ckv=(torch.zeros((max_num_batched_tokens, CKV_RECORD_DIM), dtype=KV_RECORD_DTYPE, device=device)
                    if exports else None),
        export_ik=(torch.zeros((max_num_batched_tokens, IK_RECORD_DIM), dtype=KV_RECORD_DTYPE, device=device)
                   if exports else None),
    )


# ---- PP payload keys (IntermediateTensors) ----
PP_KEY_HIDDEN = "hidden_states"      # [T, HC, HIDDEN] STREAM_DTYPE        every boundary
PP_KEY_PRE_MIX = "pre_mix"           # [T, HC] float32                     every boundary
PP_KEY_KV20_CKV = "kv20_ckv"         # [T, CKV_RECORD_DIM] float16         boundary into a stage mirroring 20
PP_KEY_KV20_IK = "kv20_ik"           # [T, IK_RECORD_DIM] float16          idem
PP_KEY_CAND20 = "cand20"             # [T, CAND_TOPK_BLOCKS] int32         idem


# ---- Engram step plan (worker/runner -> EngramHostService) ----
@dataclass(frozen=True)
class EngramReqStep:
    req_id: str
    start_pos: int                       # absolute position of the first scheduled token (= num_computed_tokens)
    num_tokens: int                      # scheduled tokens of this request this step (incl. spec tokens)
    token_ids: np.ndarray | None         # [num_tokens] int32 raw token ids; None => unknown until bind_batch (async PP)
    prompt_token_ids: np.ndarray | None  # full prompt; set when the request is new or resumed this step


@dataclass(frozen=True)
class EngramStepPlan:
    step_id: int
    reqs: tuple[EngramReqStep, ...]      # scheduler order
    finished_req_ids: frozenset[str]


@dataclass(frozen=True)
class EngramBatchLayout:
    step_id: int
    req_order: tuple[str, ...]           # runner's flattened order (decodes first)
    query_start_loc: np.ndarray          # [num_reqs + 1] int32, CPU
    num_tokens: int                      # T
    num_tokens_padded: int               # T_pad
    sampled_fill: tuple[torch.Tensor, torch.cuda.Event] | None   # async PP, or pp_size == 1: pinned int32 [num_reqs] ids for
                                                                 #   the EngramReqStep entries with token_ids None (valid after event)
