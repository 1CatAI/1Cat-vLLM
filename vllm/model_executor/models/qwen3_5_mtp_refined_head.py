# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary draft proposals followed by original-weight candidate scoring.

The target head uses its existing quantization method. Each draft scores the
complete vocabulary in NVFP4, selects candidates in vocabulary-stable order,
and recomputes their scores from the original effective FP16 weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

import vllm.envs as envs
from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import NativeBindings, call_native, register_policy_op
from vllm.config.sm70_native import capture_linear_native_config
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5_mtp_sm70_head import _hot_rows
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)
_CANDIDATES = 32
_MAX_ROWS = 8


@triton.jit
def _topk_tiles(
    Logits,
    Scratch,
    Vocab: tl.constexpr,
    Tiles: tl.constexpr,
    Keep: tl.constexpr,
    Block: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    offsets = tile * Block + tl.arange(0, Block)
    values = tl.load(
        Logits + row * Vocab + offsets, offsets < Vocab, other=-float("inf")
    )
    # An FP16 score and a shard-local ID fit in one signed 32-bit key.
    # Signed zeros share a rank, and NaNs precede finite scores.
    bits = values.to(tl.uint16, bitcast=True).to(tl.uint32)
    ordered = tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
    ordered = tl.where(values == 0, 0x8000, ordered)
    ordered = tl.where(values != values, 0xFFFF, ordered)
    keys = ((ordered.to(tl.int32) - 0x8000) << 16) | (65535 - offsets).to(tl.int32)
    keys = tl.where(offsets < Vocab, keys, -2147483648)
    selected = tl.topk(keys, Keep)
    lane = tl.arange(0, Keep)
    tl.store(Scratch + row * Tiles * Keep + tile * Keep + lane, selected)


@triton.jit
def _topk_merge(
    Scratch, Ids, Count: tl.constexpr, Keep: tl.constexpr, Block: tl.constexpr
):
    row = tl.program_id(0)
    offsets = tl.arange(0, Block)
    keys = tl.load(Scratch + row * Count + offsets, offsets < Count, -2147483648)
    keys = tl.topk(keys, Keep)
    ids = 65535 - (keys & 65535)
    lane = tl.arange(0, Keep)
    tl.store(Ids + row * Keep + lane, ids.to(tl.int32))


@triton.jit
def _rerank(
    X, Weight, Ids, Scores, Width: tl.constexpr, Keep: tl.constexpr, Block: tl.constexpr
):
    row, candidate = tl.program_id(0), tl.program_id(1)
    index = tl.load(Ids + row * Keep + candidate)
    col = tl.arange(0, Block)
    x = tl.load(X + row * Width + col, col < Width, 0).to(tl.float32)
    weight = tl.load(Weight + index * Width + col, col < Width, 0).to(tl.float32)
    score = tl.sum(x * weight, 0).to(tl.float16)
    tl.store(Scores + row * Keep + candidate, score)


@triton.jit
def _finish(
    Ids, Scores, Pairs, Keep: tl.constexpr, Start: tl.constexpr, Block: tl.constexpr
):
    row = tl.program_id(0)
    lane = tl.arange(0, Block)
    score = tl.load(Scores + row * Keep + lane, lane < Keep, other=-float("inf")).to(
        tl.float32
    )
    ids = tl.load(Ids + row * Keep + lane, lane < Keep, other=2147483647)
    nan = score != score
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    best = tl.max(tl.where(nan, -float("inf"), score), 0)
    winner = tl.where(has_nan, nan, score == best)
    best_id = tl.min(tl.where(winner, ids, 2147483647), 0)
    best = tl.where(has_nan, float("nan"), best)
    tl.store(Pairs + row * 2, best)
    tl.store(Pairs + row * 2 + 1, (best_id + Start).to(tl.float32))


def _select_candidates(logits):
    keep = _CANDIDATES
    rows, vocab = logits.shape
    assert vocab <= 65536
    tiles = triton.cdiv(vocab, 1024)
    scratch = torch.empty(rows, tiles * keep, device=logits.device, dtype=torch.int32)
    ids = torch.empty(rows, keep, device=logits.device, dtype=torch.int32)
    _topk_tiles[(rows, tiles)](logits, scratch, vocab, tiles, keep, 1024, num_warps=4)
    _topk_merge[(rows,)](
        scratch,
        ids,
        tiles * keep,
        keep,
        triton.next_power_of_2(tiles * keep),
        num_warps=8,
    )
    return ids


def _rerank_candidates(x, weight, ids, start, reference_mma=False):
    rows, keep = ids.shape
    scores = torch.empty(rows, keep, device=x.device, dtype=torch.float16)
    pairs = torch.empty(rows, 2, device=x.device, dtype=torch.float32)
    if reference_mma:
        torch.ops._C.sm70_fp16_candidate_scores_out(scores, x, weight, ids)
    else:
        _rerank[(rows, keep)](
            x, weight, ids, scores, x.shape[1], keep, 8192, num_warps=8
        )
    _finish[(rows,)](
        ids, scores, pairs, keep, start, triton.next_power_of_2(keep), num_warps=1
    )
    return pairs


def _refined_head(
    x,
    codes,
    scales,
    global_scale,
    weight,
    start,
    native_policy=None,
    reference_mma=False,
):
    x = x.contiguous()
    logits = torch.empty(
        x.shape[0], weight.shape[0], device=x.device, dtype=torch.float16
    )
    call_native(
        sm70_ops.nvfp4_qpn2_gemm_sm70_out,
        native_policy,
        logits,
        x,
        codes,
        scales,
        global_scale,
        16,
        2,
    )
    return _rerank_candidates(
        x, weight, _select_candidates(logits), start, reference_mma
    )


def _refined_head_fake(
    x,
    codes,
    scales,
    global_scale,
    weight,
    start,
    native_policy=None,
    reference_mma=False,
):
    return x.new_empty((x.shape[0], 2), dtype=torch.float32)


register_policy_op(
    "sm70_mtp_refined_head",
    "(Tensor x, Tensor codes, Tensor scales, float global_scale, Tensor weight, "
    "int start, str[]? native_policy=None, bool reference_mma=False) -> Tensor",
    _refined_head,
    _refined_head_fake,
)


class MTPRefinedHead(nn.Module):
    """Score original weights for candidates from a complete-vocabulary proposal."""

    @torch.inference_mode()
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head
        self.native_ops = NativeBindings(capture_linear_native_config("nvfp4").values)
        self.shard_indices = head.shard_indices
        self.start = self.shard_indices.org_vocab_start_index
        rows = self.shard_indices.org_vocab_end_index - self.start
        assert rows == 62080 and head.embedding_dim == 5120
        ids = torch.arange(rows, device=head.weight.device, dtype=torch.int64)
        self.reference_mma = head.weight.dtype == torch.float16 and tuple(
            head.weight.shape
        ) == (rows, 5120)
        # A dense head already owns the exact effective weights. Share its
        # storage and retain the reference Tensor Core accumulation order.
        dense = head.weight.detach() if self.reference_mma else _hot_rows(head, ids)
        self.register_buffer("dense", dense, persistent=False)
        global_scale = float(dense.abs().max().float().clamp_min(1e-12) / 2688)
        self.global_scale = global_scale
        packed = torch.empty(rows, 2560, device=dense.device, dtype=torch.uint8)
        raw_scales = torch.empty(
            rows, 320, device=dense.device, dtype=torch.float8_e4m3fn
        )
        thresholds = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dense.device
        )
        for begin in range(0, rows, 512):
            values = dense[begin : begin + 512].float().reshape(-1, 320, 16)
            scale = (values.abs().amax(-1) / 6 / global_scale).to(torch.float8_e4m3fn)
            effective = (scale.float() * global_scale).half().float().clamp_min(1e-12)
            code = torch.bucketize(
                (values.abs() / effective.unsqueeze(-1)).contiguous(), thresholds
            ).to(torch.uint8)
            code |= (values < 0).to(torch.uint8) * 8
            flat = code.flatten(1)
            packed[begin : begin + 512] = flat[:, ::2] | (flat[:, 1::2] << 4)
            raw_scales[begin : begin + 512] = scale
        codes, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw_scales)
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        logger.info_once(
            "SM70 MTP refined head enabled with full-vocabulary NVFP4 proposals "
            "and %d original-weight candidates per rank.",
            _CANDIDATES,
        )

    @property
    def weight(self):
        return self.head.weight

    @property
    def quant_method(self):
        return self

    def apply(self, layer, x, bias=None):
        return self.head.quant_method.apply(self.head, x, bias)

    def maybe_get_sm70_lm_head_top1(self, hidden_states, bias=None):
        return None

    def maybe_get_sm70_lm_head_top1_pair(self, hidden_states, bias=None):
        if (
            bias is not None
            or hidden_states.ndim != 2
            or hidden_states.dtype != torch.float16
            or not 1 <= hidden_states.shape[0] <= _MAX_ROWS
        ):
            return None
        return self.native_ops.invoke(
            torch.ops.vllm.sm70_mtp_refined_head,
            hidden_states,
            self.codes,
            self.scales,
            self.global_scale,
            self.dense,
            self.start,
            self.native_ops.arguments,
            self.reference_mma,
        )


def prepare_mtp_refined_head(
    head: nn.Module, vllm_config: VllmConfig
) -> MTPRefinedHead | None:
    """Prepare the measured TP4, H5120, 248320-token greedy MTP contract."""
    weight = getattr(head, "weight", None)
    shard = getattr(head, "shard_indices", None)
    speculative = vllm_config.speculative_config
    if (
        not isinstance(weight, torch.Tensor)
        or not weight.is_cuda
        or shard is None
        or not current_platform.is_device_capability(70)
        or envs.VLLM_BATCH_INVARIANT
        or vllm_config.model_config is None
        or vllm_config.model_config.dtype != torch.float16
        or vllm_config.lora_config is not None
        or speculative is None
        or speculative.method != "mtp"
        or speculative.draft_sample_method == "probabilistic"
        or getattr(head, "embedding_dim", 0) != 5120
        or getattr(shard, "num_added_elements", 0)
        or getattr(shard, "num_org_vocab_padding", 0)
        or shard.org_vocab_end_index - shard.org_vocab_start_index != 62080
    ):
        return None
    if get_tp_group().world_size != 4:
        return None
    if weight.dtype == torch.float16 and not hasattr(
        torch.ops._C, "sm70_fp16_candidate_scores_out"
    ):
        logger.warning_once(
            "SM70 refined FP16 draft heads require the reference-order "
            "candidate scoring operator. Using the original head."
        )
        return None
    if any(
        not hasattr(torch.ops._C, name)
        for name in ("nvfp4_qpn2_prepare_sm70", "nvfp4_qpn2_gemm_sm70_out")
    ):
        logger.warning_once("SM70 MTP refined head requires the NVFP4 QPN2 operators.")
        return None
    return MTPRefinedHead(head)
