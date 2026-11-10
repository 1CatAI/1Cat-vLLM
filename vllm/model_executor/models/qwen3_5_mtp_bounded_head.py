# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded vocabulary search for the original FP16 MTP target head.

For original weights W and QPN8 effective FP16 weights Q, Cauchy-Schwarz
bounds the score difference by ||x|| ||W-Q||. The stored coefficient also
covers arbitrary FP32 accumulation order and FP16 input subnormal flushing
for both projections. FP64 preparation rounds these positive coefficients
upward. Runtime bounds include input-norm and coarse-score rounding.

Candidates are selected by their upper bounds and scored from original weights
with FP32 Tensor Cores. The bounds include FP32 reduction-order differences.
A winner must strictly exceed every discarded upper
bound across TP ranks. Other rows execute the complete original-weight
projection before the normal global argmax. The device mask is graph-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

import vllm.envs as envs
from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import NativeBindings, call_native, register_policy_op
from vllm.config.sm70_native import capture_linear_native_config
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen3_5_mtp_refined_head import (
    _finish,
    _score_id_keys,
    _select_candidates,
)
from vllm.model_executor.models.qwen3_5_mtp_sm70_head import _mtp_head_shard_rows
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)
KEEP = 32


@triton.jit
def _input_norms(X, Norms, Width: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 8192)
    x = tl.load(X + row * Width + col, col < Width, 0).to(tl.float32)
    tiny = tl.where(tl.abs(x) < 0.00006103515625, x, 0.0)
    # 1 + 2^-10 covers even a serial 5120-term positive FP32 sum,
    # followed by sqrt and rounding. Nonfinite inputs force the dense path.
    norm = tl.sqrt(tl.sum(x * x, 0)) * 1.0009765625
    subnorm = tl.sqrt(tl.sum(tiny * tiny, 0)) * 1.0009765625
    bad = tl.sum(((x != x) | (tl.abs(x) == float("inf"))).to(tl.int32), 0) > 0
    tl.store(Norms + row * 2, tl.where(bad, float("inf"), norm))
    tl.store(Norms + row * 2 + 1, tl.where(bad, float("inf"), subnorm))


@triton.jit
def _upper_bounds(Coarse, Coeff, Norms, Upper, Vocab: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * 512 + tl.arange(0, 512)
    coarse = tl.load(Coarse + row * Vocab + col, col < Vocab, 0).to(tl.float32)
    coefficient = tl.load(Coeff + col * 2, col < Vocab, 0)
    norm_sum = tl.load(Coeff + col * 2 + 1, col < Vocab, 0)
    xnorm, subnorm = tl.load(Norms + row * 2), tl.load(Norms + row * 2 + 1)
    error = (
        xnorm * coefficient
        + subnorm * norm_sum
        + tl.abs(coarse) * 0.0009765625
        + 0.000000059604644775390625
    )
    error = error * 1.0000038146972656
    upper = (coarse + error).to(tl.float16)
    bits = upper.to(tl.uint16, bitcast=True)
    next_bits = tl.where(upper < 0, bits - 1, bits + 1)
    next_bits = tl.where(upper == 0, 1, next_bits)
    next_bits = tl.where(upper == float("inf"), 0x7C00, next_bits)
    next_upper = next_bits.to(tl.uint16).to(tl.float16, bitcast=True)
    valid = (
        (coarse == coarse)
        & (tl.abs(coarse) != float("inf"))
        & (error == error)
        & (error != float("inf"))
    )
    tl.store(
        Upper + row * Vocab + col,
        tl.where(valid, next_upper, float("inf")),
        col < Vocab,
    )


@triton.jit
def _candidate_packet(
    Pairs, Upper, Ids, Packet, Vocab: tl.constexpr, Keep: tl.constexpr
):
    row = tl.program_id(0)
    last = tl.load(Ids + row * Keep + Keep - 1)
    cutoff = tl.load(Upper + row * Vocab + last).to(tl.float32)
    tl.store(Packet + row * 3, tl.load(Pairs + row * 2))
    tl.store(Packet + row * 3 + 1, tl.load(Pairs + row * 2 + 1))
    tl.store(Packet + row * 3 + 2, cutoff)


@triton.jit
def _certify(Gathered, Flags, GlobalPairs, Ranks: tl.constexpr):
    row = tl.program_id(0)
    ranks = tl.arange(0, Ranks)
    value = tl.load(Gathered + row * Ranks * 3 + ranks * 3)
    ids = tl.load(Gathered + row * Ranks * 3 + ranks * 3 + 1).to(tl.int32)
    cutoff = tl.load(Gathered + row * Ranks * 3 + ranks * 3 + 2)
    best = tl.max(value, 0)
    best_id = tl.min(tl.where(value == best, ids, 2147483647), 0)
    valid = (
        tl.sum(((value != value) | (tl.abs(value) == float("inf"))).to(tl.int32), 0)
        == 0
    )
    flag = valid & (best > tl.max(cutoff, 0))
    tl.store(Flags + row, flag.to(tl.int32))
    tl.store(GlobalPairs + row * 2, best)
    tl.store(GlobalPairs + row * 2 + 1, best_id.to(tl.float32))


@triton.jit
def _masked_max_tiles(
    Dense,
    Flags,
    Scratch,
    Vocab: tl.constexpr,
    Tiles: tl.constexpr,
    WideIds: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    if tl.load(Flags + row) == 0:
        col = tile * 1024 + tl.arange(0, 1024)
        value = tl.load(Dense + row * Vocab + col, col < Vocab, -float("inf"))
        keys = _score_id_keys(value, col, WideIds)
        sentinel: tl.constexpr = -9223372036854775808 if WideIds else -2147483648
        keys = tl.where(col < Vocab, keys, sentinel)
        tl.store(Scratch + row * Tiles + tile, tl.max(keys, 0))


@triton.jit
def _masked_max_finish(
    Dense,
    Flags,
    Scratch,
    GlobalPairs,
    Output,
    Vocab: tl.constexpr,
    Tiles: tl.constexpr,
    Start: tl.constexpr,
    Block: tl.constexpr,
    WideIds: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(Flags + row) != 0:
        value = tl.load(GlobalPairs + row * 2)
        token = tl.load(GlobalPairs + row * 2 + 1)
    else:
        offsets = tl.arange(0, Block)
        sentinel: tl.constexpr = -9223372036854775808 if WideIds else -2147483648
        key = tl.max(
            tl.load(Scratch + row * Tiles + offsets, offsets < Tiles, sentinel), 0
        )
        id_mask: tl.constexpr = 0xFFFFFFFF if WideIds else 65535
        local_id = id_mask - (key & id_mask)
        value = tl.load(Dense + row * Vocab + local_id).to(tl.float32)
        token = (local_id + Start).to(tl.float32)
    tl.store(Output + row * 2, value)
    tl.store(Output + row * 2 + 1, token)


def _prepare_proposal(weight, native_ops):
    """Compute the exact QPN8 effective weights and upward-rounded coefficients."""
    vocab, width = weight.shape
    quant = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty(vocab, 1, device=weight.device, dtype=torch.float32)
    coefficients = torch.empty(vocab, 2, device=weight.device, dtype=torch.float32)
    epsilon = 2**-23
    gamma = (2 * width * epsilon) / (1 - 2 * width * epsilon)
    for begin in range(0, vocab, 512):
        original = weight[begin : begin + 512]
        scale = original.float().abs().amax(-1, keepdim=True) / 448
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = (original.float() / scale).to(torch.float8_e4m3fn)
        represented = ((codes.float() / 256) * (scale * 256).half().float()).half()
        quant[begin : begin + 512], scales[begin : begin + 512] = codes, scale
        residual = (original.double() - represented.double()).norm(dim=-1)
        norm_sum = original.double().norm(dim=-1) + represented.double().norm(dim=-1)
        subnorm_sum = torch.where(original.abs() < 2**-14, original, 0).double().norm(
            dim=-1
        ) + torch.where(represented.abs() < 2**-14, represented, 0).double().norm(
            dim=-1
        )
        exact = torch.stack((residual + gamma * norm_sum + subnorm_sum, norm_sum), -1)
        rounded = torch.nextafter(
            exact.float(), torch.full_like(exact.float(), float("inf"))
        )
        coefficients[begin : begin + 512] = rounded
    codes, scales = native_ops.fp8_qpn8_prepare_sm70(quant, scales)
    return codes, scales, coefficients


def _proposal(x, codes, scales, coefficients, native_policy=()):
    rows, vocab = x.size(0), codes.numel() // x.size(1)
    coarse = torch.empty(rows, vocab, device=x.device, dtype=torch.float16)
    call_native(
        sm70_ops.fp8_qpn8_gemm_sm70_out,
        native_policy,
        coarse,
        x,
        codes,
        scales,
        16,
        2,
        False,
        False,
    )
    norms = torch.empty(rows, 2, device=x.device, dtype=torch.float32)
    upper = torch.empty_like(coarse)
    _input_norms[(rows,)](x, norms, x.size(1), num_warps=8)
    _upper_bounds[(rows, triton.cdiv(vocab, 512))](
        coarse, coefficients, norms, upper, vocab, num_warps=4
    )
    ids = _select_candidates(upper)
    return coarse, upper, ids


def _bounded_pair(x, weight, codes, scales, coefficients, start, native_policy=()):
    x = x.contiguous()
    rows, vocab = x.size(0), weight.size(0)
    coarse, upper, ids = _proposal(x, codes, scales, coefficients, native_policy)
    scores = torch.empty(rows, KEEP, device=x.device, dtype=torch.float16)
    pairs = torch.empty(rows, 2, device=x.device, dtype=torch.float32)
    torch.ops._C.sm70_fp16_candidate_scores_out(scores, x, weight, ids)
    _finish[(rows,)](ids, scores, pairs, KEEP, start, KEEP, num_warps=1)
    packet = torch.empty(rows, 3, device=x.device, dtype=torch.float32)
    _candidate_packet[(rows,)](pairs, upper, ids, packet, vocab, KEEP, num_warps=1)
    gathered = tensor_model_parallel_all_gather(packet, dim=-1)
    flags = torch.empty(rows, device=x.device, dtype=torch.int32)
    global_pairs = torch.empty_like(pairs)
    _certify[(rows,)](gathered, flags, global_pairs, gathered.size(1) // 3, num_warps=1)
    dense = torch.empty_like(coarse)
    torch.ops._C.sm70_fp16_masked_head_out(dense, x, weight, flags)
    tiles = triton.cdiv(vocab, 1024)
    wide_ids = vocab > 65536
    scratch = torch.empty(
        rows, tiles, device=x.device, dtype=torch.int64 if wide_ids else torch.int32
    )
    output = torch.empty_like(pairs)
    _masked_max_tiles[(rows, tiles)](
        dense, flags, scratch, vocab, tiles, wide_ids, num_warps=4
    )
    _masked_max_finish[(rows,)](
        dense,
        flags,
        scratch,
        global_pairs,
        output,
        vocab,
        tiles,
        start,
        triton.next_power_of_2(tiles),
        wide_ids,
        num_warps=2,
    )
    return output, flags, upper


def _bounded_head(x, weight, codes, scales, coefficients, start, native_policy=None):
    return _bounded_pair(x, weight, codes, scales, coefficients, start, native_policy)[
        0
    ]


def _bounded_head_fake(
    x, weight, codes, scales, coefficients, start, native_policy=None
):
    return x.new_empty((x.shape[0], 2), dtype=torch.float32)


register_policy_op(
    "sm70_mtp_bounded_target_head",
    "(Tensor x, Tensor weight, Tensor codes, Tensor scales, Tensor coefficients, "
    "int start, str[]? native_policy=None) -> Tensor",
    _bounded_head,
    _bounded_head_fake,
)


class MTPBoundedTargetHead(nn.Module):
    """Preserve the dense head while owning disposable search weights."""

    @torch.inference_mode()
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head
        self.shard_indices = head.shard_indices
        self.start = self.shard_indices.org_vocab_start_index
        self.native_ops = NativeBindings(capture_linear_native_config("fp8").values)
        codes, scales, coefficients = _prepare_proposal(
            head.weight.detach(), self.native_ops
        )
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer("coefficients", coefficients, persistent=False)

    @property
    def weight(self):
        return self.head.weight

    @property
    def quant_method(self):
        return self

    def apply(self, layer, x, bias=None):
        return self.head.quant_method.apply(self.head, x, bias)

    def maybe_get_sm70_lm_head_top1(self, hidden_states, bias=None):
        return self.head.maybe_get_sm70_lm_head_top1(hidden_states, bias)

    def maybe_get_sm70_lm_head_top1_pair(self, hidden_states, bias=None):
        if (
            bias is not None
            or hidden_states.ndim != 2
            or hidden_states.shape[1] != 5120
            or not 1 <= hidden_states.shape[0] <= 8
            or hidden_states.dtype != torch.float16
            or hidden_states.device != self.weight.device
        ):
            return None
        return self.native_ops.invoke(
            torch.ops.vllm.sm70_mtp_bounded_target_head,
            hidden_states,
            self.weight,
            self.codes,
            self.scales,
            self.coefficients,
            self.start,
            self.native_ops.arguments,
        )


def prepare_mtp_bounded_target_head(
    head: nn.Module, config: VllmConfig
) -> MTPBoundedTargetHead | None:
    weight = getattr(head, "weight", None)
    shard = getattr(head, "shard_indices", None)
    speculative = config.speculative_config
    if (
        not isinstance(weight, torch.Tensor)
        or not weight.is_cuda
        or weight.dtype != torch.float16
        or weight.ndim != 2
        or weight.shape[1] != 5120
        or not weight.is_contiguous()
        or shard is None
        or not isinstance(
            getattr(head, "quant_method", None), UnquantizedEmbeddingMethod
        )
        or not current_platform.is_device_capability(70)
        or envs.VLLM_BATCH_INVARIANT
        or config.lora_config is not None
        or speculative is None
        or speculative.method != "mtp"
        or speculative.draft_sample_method == "probabilistic"
        or _mtp_head_shard_rows(head) != weight.shape[0]
    ):
        return None
    # Old extensions only accept the original shard width. Keep their dense
    # fallback instead of admitting a wider shard to an incompatible binary.
    revision = getattr(torch.ops._C, "sm70_fp16_head_vocab_revision", None)
    if weight.shape[0] != 62080 and (revision is None or revision() < 1):
        return None
    state = getattr(head, "_sm70_lm_head_state", None)
    if any(
        getattr(state, field, False)
        for field in ("_sm70_f16_prepared", "_sm70_dflash2_fp32_logits")
    ):
        return None
    required = (
        "sm70_fp16_candidate_scores_out",
        "sm70_fp16_masked_head_out",
        "fp8_qpn8_prepare_sm70",
        "fp8_qpn8_gemm_sm70_out",
    )
    if any(not hasattr(torch.ops._C, name) for name in required):
        logger.warning_once(
            "SM70 bounded target head operators are unavailable. Using the dense head."
        )
        return None
    if not bool(torch.isfinite(weight).all()):
        logger.warning_once(
            "SM70 bounded target head requires finite weights. Using the dense head."
        )
        return None
    prepared = MTPBoundedTargetHead(head)
    logger.info_once(
        "SM70 MTP bounded target head prepared with original FP16 scoring "
        "and device-side full-vocabulary fallback."
    )
    return prepared
