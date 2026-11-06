# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise-exact small-kernel fusions for the SM70 Qwen3.8 MTP verify path.

Each unit replaces an eager multi-kernel sequence and reproduces the replaced
computation bit for bit on every row that production defines:

* h1: shared-expert gate, ``F.sigmoid(gate) * out`` in Qwen2MoeMLP (2 ATen
  kernels -> 1). FP32 math, IEEE division, FP16 rounding after the sigmoid and
  after the product, as ATen does.
* p1: PLE n-gram ids of ``Qwen4ExpNGramEmbedding.compute_ngram_ids`` for GPU
  batches (about 55 int64 kernels -> 1), written straight into the op output.
* p2: PLE dilated short convolution of the speculative-verify path
  (``_short_conv_dilated_spec_batched``, about 50 kernels -> 1): state
  rollback, 4-tap FMA chain as ATen's depthwise kernel, FP16 rounding, SiLU,
  in-place conv-state update. Graph-padding rows get +0.
* u3z: the zero fill of the GDN verify output is folded into the u3 recurrent
  kernel (sm70_gdn_verify_fused.py); only the selection lives here.
* s1: greedy MTP verification from TP-local (max, index) pairs instead of
  all-gathered logits; argmax semantics of torch.argmax (first maximum,
  first NaN), which is production's on every NaN-free row.
* h2: TurboMind MoE permute (output zero fill, top-k id copy, radix sort,
  expert offsets, row expansion, int32 offset copy -> 1 kernel).

``ONECAT_FUSE47`` selects the units: a comma list of h1,p1,p2,u3z,s1,h2 or
"all". Unset or empty keeps every production path.
"""

import collections
import os

import torch

from vllm.config.sm70_runtime import capture_spec_decode_trace
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

try:
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra import libdevice

logger = init_logger(__name__)

UNITS = ("h1", "p1", "p2", "u3z", "s1", "h2")


def parse_units(raw: str | None) -> frozenset[str]:
    if raw is None:
        return frozenset()
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if "all" in items:
        return frozenset(UNITS)
    unknown = items - set(UNITS) - {"none", "0", "off"}
    if unknown:
        raise ValueError(f"ONECAT_FUSE47: unknown units {sorted(unknown)}")
    return frozenset(items & set(UNITS))


_ENV_UNITS = parse_units(os.getenv("ONECAT_FUSE47"))
_UNITS_OVERRIDE: frozenset[str] | None = None
# Route decisions taken in Python (eager calls and CUDA graph captures).
ROUTE_COUNTS: collections.Counter = collections.Counter()


def enabled_units() -> frozenset[str]:
    return _ENV_UNITS if _UNITS_OVERRIDE is None else _UNITS_OVERRIDE


def unit_enabled(unit: str) -> bool:
    return unit in enabled_units()


def set_units_override(units: frozenset[str] | None) -> None:
    """Tests and diagnostics: replace the environment selection (None restores)."""
    global _UNITS_OVERRIDE
    _UNITS_OVERRIDE = units


def note_route(unit: str, route: str) -> None:
    ROUTE_COUNTS[(unit, route)] += 1
    logger.info_once("ONECAT_FUSE47 %s route: %s", unit, route)


def _is_sm70(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (7, 0)


# --------------------------------------------------------------------------
# h1: shared-expert gate sigmoid * out
# --------------------------------------------------------------------------
@triton.jit
def _shared_gate_sigmoid_mul_kernel(
    gate_ptr,  # fp16 [M, 1]
    x_ptr,  # fp16 [M, N]
    out_ptr,  # fp16 [M, N]
    stride_gate,
    stride_x,
    stride_out,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    g = tl.load(gate_ptr + row * stride_gate + cols * 0).to(tl.float32)
    one = tl.full([BLOCK], 1.0, tl.float32)
    # ATen sigmoid (Half): opmath 1 / (1 + exp(-x)), IEEE division, then FP16.
    sig = libdevice.div_rn(one, one + libdevice.exp(-g))
    sig = sig.to(tl.float16).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # ATen mul (Half): exact FP32 product of the two FP16 operands, then FP16.
    tl.store(out_ptr + row * stride_out + cols, (sig * x).to(tl.float16), mask=mask)


def shared_gate_sigmoid_mul(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Returns F.sigmoid(gate) * x for gate [M, 1] and x [M, N] (FP16)."""
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = 1024
    _shared_gate_sigmoid_mul_kernel[(M, triton.cdiv(N, BLOCK))](
        gate,
        x,
        out,
        gate.stride(0),
        x.stride(0),
        out.stride(0),
        N,
        BLOCK=BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def h1_supported(gate: torch.Tensor, x: torch.Tensor) -> bool:
    return bool(
        gate.dtype == torch.float16
        and x.dtype == torch.float16
        and gate.ndim == 2
        and x.ndim == 2
        and gate.shape[0] == x.shape[0]
        and gate.shape[1] == 1
        and x.shape[0] > 0
        and x.stride(1) == 1
        and _is_sm70(x)
    )


# --------------------------------------------------------------------------
# p1: PLE n-gram ids
# --------------------------------------------------------------------------
@triton.jit
def _ple_ngram_ids_kernel(
    ids_ptr,  # int32/int64 [num_tokens]
    qsl_ptr,  # int32/int64 [num_reqs + 1]
    ctx_ptr,  # int [>= num_reqs, 2]
    stride_ctx,
    mult_ptr,  # int64 [3]
    sizes_ptr,  # int64 [16]
    offsets_ptr,  # int64 [16]
    out_ptr,  # int64 [num_tokens, 16]
    num_tokens,
    num_reqs,
    EOS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_tokens
    r_off = tl.arange(0, BLOCK_R)
    qsl = tl.load(qsl_ptr + r_off, mask=r_off <= num_reqs, other=0).to(tl.int64)
    # searchsorted(query_start_loc, t, right=True) - 1, clamp(max=num_reqs - 1)
    le = (qsl[None, :] <= t.to(tl.int64)[:, None]) & (r_off <= num_reqs)[None, :]
    req = tl.sum(le.to(tl.int32), axis=1) - 1
    req = tl.maximum(tl.minimum(req, num_reqs - 1), 0)
    start = tl.load(qsl_ptr + req, mask=tmask, other=0).to(tl.int64)
    col = t.to(tl.int64) - start
    col = tl.minimum(tl.maximum(col, 0), num_tokens - 1)

    cur = tl.load(ids_ptr + t, mask=tmask, other=0).to(tl.int64)
    ctx0 = tl.load(ctx_ptr + req * stride_ctx, mask=tmask, other=0).to(tl.int64)
    ctx1 = tl.load(ctx_ptr + req * stride_ctx + 1, mask=tmask, other=0).to(tl.int64)
    tm1 = tl.maximum(t - 1, 0)
    tm2 = tl.maximum(t - 2, 0)
    ids_m1 = tl.load(ids_ptr + tm1, mask=tmask & (col >= 1), other=0).to(tl.int64)
    ids_m2 = tl.load(ids_ptr + tm2, mask=tmask & (col >= 2), other=0).to(tl.int64)
    # Context of token t in compute_ngram_ids: [ngram_context[req], packed row].
    prev = tl.where(col >= 1, ids_m1, ctx1)
    older = tl.where(col >= 2, ids_m2, tl.where(col == 1, ctx1, ctx0))
    # _shift_apply: the 3-gram source is EOS when the segment restarts at prev.
    older = tl.where(prev == EOS, EOS, older)

    m0 = tl.load(mult_ptr).to(tl.int64)
    m1 = tl.load(mult_ptr + 1).to(tl.int64)
    m2 = tl.load(mult_ptr + 2).to(tl.int64)
    mixed2 = (cur * m0) ^ (prev * m1)
    mixed3 = mixed2 ^ (older * m2)

    head = tl.arange(0, 16)
    size = tl.load(sizes_ptr + head).to(tl.int64)
    offset = tl.load(offsets_ptr + head).to(tl.int64)
    mixed = tl.where(head[None, :] < 8, mixed2[:, None], mixed3[:, None])
    rem = mixed % size[None, :]
    # torch.remainder is non-negative for positive divisors.
    rem = tl.where(rem < 0, rem + size[None, :], rem)
    tl.store(
        out_ptr + t.to(tl.int64)[:, None] * 16 + head[None, :],
        rem + offset[None, :],
        mask=tmask[:, None],
    )


P1_MAX_TOKENS = 4096
P1_MAX_REQS = 256


def p1_supported(
    emb: torch.nn.Module,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    out: torch.Tensor | None,
) -> bool:
    num_tokens = input_ids.numel()
    num_reqs = query_start_loc.numel() - 1
    return bool(
        out is not None
        and 0 < num_tokens <= P1_MAX_TOKENS
        and 0 < num_reqs <= P1_MAX_REQS
        and input_ids.is_cuda
        and _is_sm70(input_ids)
        and input_ids.dtype in (torch.int32, torch.int64)
        and query_start_loc.dtype in (torch.int32, torch.int64)
        and query_start_loc.is_cuda
        and query_start_loc.is_contiguous()
        and ngram_context.is_cuda
        and ngram_context.ndim == 2
        and ngram_context.shape[0] >= num_reqs
        and ngram_context.shape[1] == 2
        and ngram_context.stride(1) == 1
        and ngram_context.dtype in (torch.int32, torch.int64)
        and emb.ngram_size == 3
        and emb.heads_per_ngram == 8
        and emb.ngram_heads == 16
        and out.dtype == torch.int64
        and out.is_contiguous()
        and tuple(out.shape) == (num_tokens, 16)
    )


def ple_ngram_ids(
    emb: torch.nn.Module,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    input_ids = input_ids.reshape(-1)
    num_tokens = input_ids.numel()
    num_reqs = query_start_loc.numel() - 1
    BLOCK_T = 64
    BLOCK_R = triton.next_power_of_2(num_reqs + 1)
    _ple_ngram_ids_kernel[(triton.cdiv(num_tokens, BLOCK_T),)](
        input_ids,
        query_start_loc,
        ngram_context,
        ngram_context.stride(0),
        emb.layer_multipliers,
        emb.ngram_heads_vocab_sizes,
        emb.ngram_heads_offsets,
        out,
        num_tokens,
        num_reqs,
        EOS=int(emb.eos_token_id),
        BLOCK_T=BLOCK_T,
        BLOCK_R=BLOCK_R,
        num_warps=4,
    )
    return out


# --------------------------------------------------------------------------
# p2: PLE dilated short convolution, speculative-verify batches
# --------------------------------------------------------------------------
@triton.jit
def _ple_short_conv_spec_kernel(
    x_ptr,  # fp16 [num_x_rows, C]
    stride_xt,
    state_ptr,  # fp16 view [slots, C, L]
    stride_s0,
    stride_sc,
    stride_sl,
    w_ptr,  # fp16 [C, 4]
    stride_wc,
    stride_wk,
    state_idx_ptr,  # int [num_reqs]
    qsl_ptr,  # int [num_reqs + 1]
    nacc_ptr,  # int [num_reqs]
    out_ptr,  # fp16 [>= num_x_rows, C]
    stride_output_token,
    num_reqs,
    num_x_rows,
    C,
    NULL_ID: tl.constexpr,
    MAX_LEN: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DIL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < C
    if r == num_reqs:
        # Graph-padding token rows (routed to the dummy request): +0.
        start = tl.load(qsl_ptr + num_reqs).to(tl.int64)
        zero = tl.zeros([BLOCK_C], dtype=tl.float16)
        for p in range(start, num_x_rows):
            tl.store(out_ptr + p * stride_output_token + c, zero, mask=cmask)
        return

    idx = tl.load(state_idx_ptr + r).to(tl.int64)
    valid = idx != NULL_ID
    safe = tl.where(valid, idx, 0)
    q0 = tl.load(qsl_ptr + r).to(tl.int64)
    q1 = tl.load(qsl_ptr + r + 1).to(tl.int64)
    qlen = q1 - q0
    nacc = tl.load(nacc_ptr + r).to(tl.int64)
    off = tl.where(valid, tl.minimum(tl.maximum(nacc - 1, 0), MAX_LEN - 1), 0)

    base = state_ptr + safe * stride_s0 + c * stride_sc
    w0 = tl.load(w_ptr + c * stride_wc, mask=cmask, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + c * stride_wc + stride_wk, mask=cmask, other=0.0).to(
        tl.float32
    )
    w2 = tl.load(w_ptr + c * stride_wc + 2 * stride_wk, mask=cmask, other=0.0).to(
        tl.float32
    )
    w3 = tl.load(w_ptr + c * stride_wc + 3 * stride_wk, mask=cmask, other=0.0).to(
        tl.float32
    )
    one = tl.full([BLOCK_C], 1.0, tl.float32)

    # history[k] (k < STATE_LEN): rolled-back cached state, zero if invalid;
    # history[STATE_LEN + j]: token j of this request, zero past qlen.
    for j in tl.static_range(MAX_LEN):
        # ATen conv_depthwise2d (Half, acc float, no bias): value = 0, then
        # value += w * in over the taps in order, contracted to FMA by nvcc.
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for m in tl.static_range(4):
            k = j + m * DIL
            if k < STATE_LEN:
                v = tl.load(base + (off + k) * stride_sl, mask=cmask & valid, other=0.0)
            else:
                v = tl.load(
                    x_ptr + (q0 + (k - STATE_LEN)) * stride_xt + c,
                    mask=cmask & ((k - STATE_LEN) < qlen),
                    other=0.0,
                )
            if m == 0:
                wm = w0
            elif m == 1:
                wm = w1
            elif m == 2:
                wm = w2
            else:
                wm = w3
            acc = tl.fma(wm, v.to(tl.float32), acc)
        conv = acc.to(tl.float16).to(tl.float32)
        # ATen silu (Half): x / (1 + exp(-x)) in FP32, then FP16.
        y = libdevice.div_rn(conv, one + libdevice.exp(-conv)).to(tl.float16)
        # output * valid_tokens (1.0 for real tokens), FP16 multiply.
        y = (y.to(tl.float32) * one).to(tl.float16)
        tl.store(
            out_ptr + (q0 + j) * stride_output_token + c, y, mask=cmask & (j < qlen)
        )

    # Extended state: position p <- history[p + 1] for p < STATE_LEN + qlen - 1.
    # Every read is at a position above every position already written.
    for p in tl.static_range(STATE_LEN + MAX_LEN - 1):
        k = p + 1
        if k < STATE_LEN:
            v = tl.load(base + (off + k) * stride_sl, mask=cmask & valid, other=0.0)
        else:
            v = tl.load(
                x_ptr + (q0 + (k - STATE_LEN)) * stride_xt + c,
                mask=cmask & ((k - STATE_LEN) < qlen),
                other=0.0,
            )
        tl.store(
            base + p * stride_sl,
            v,
            mask=cmask & valid & (p < STATE_LEN + qlen - 1),
        )


def p2_supported(
    layer: torch.nn.Module,
    x_spec: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_query_len: int,
) -> bool:
    return bool(
        x_spec.dtype == torch.float16
        and conv_state.dtype == torch.float16
        and conv_weights.dtype == torch.float16
        and x_spec.ndim == 2
        and x_spec.stride(1) == 1
        and conv_state.ndim == 3
        and conv_weights.ndim == 2
        and conv_weights.shape[1] == 4
        and int(layer.conv_kernel_size) == 4
        and int(layer.short_conv_dilation) == 3
        and int(layer.conv_state_len) == 9
        and 1 <= int(spec_query_len) <= 8
        and conv_state.size(-1) >= 9 + int(spec_query_len) - 1
        and conv_state.size(1) == x_spec.size(1)
        and _is_sm70(x_spec)
    )


def ple_short_conv_spec(
    x_spec: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor,
    null_block_id: int,
) -> torch.Tensor:
    """In-place conv_state update; writes rows [0, x_spec.size(0)) of out."""
    num_reqs = spec_state_indices_tensor.numel()
    C = x_spec.size(1)
    BLOCK_C = 256
    grid = (num_reqs + 1, triton.cdiv(C, BLOCK_C))
    _ple_short_conv_spec_kernel[grid](
        x_spec,
        x_spec.stride(0),
        conv_state,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weights,
        conv_weights.stride(0),
        conv_weights.stride(1),
        spec_state_indices_tensor,
        spec_query_start_loc,
        num_accepted_tokens,
        out,
        out.stride(0),
        num_reqs,
        x_spec.size(0),
        C,
        NULL_ID=int(null_block_id),
        MAX_LEN=int(spec_query_len),
        STATE_LEN=9,
        DIL=3,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out[: x_spec.size(0)]


# --------------------------------------------------------------------------
# s1: greedy MTP verification from TP-local top-1 pairs
# --------------------------------------------------------------------------
def _model_lm_head(model: torch.nn.Module):
    inner = getattr(model, "language_model", model)
    lp = getattr(inner, "logits_processor", None)
    lm_head = getattr(inner, "lm_head", None)
    if lp is None or lm_head is None or not hasattr(lm_head, "shard_indices"):
        return None
    return lp, lm_head


def tp_local_top1(model: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Global argmax of compute_logits(hidden_states) rows (torch.argmax
    semantics) from TP-local maxima: the torch path of
    LogitsProcessor.get_top_tokens without the SM70 top-1 GEMV and custom
    all-reduce shortcuts, so the local logits come from the same lm_head GEMM
    as compute_logits."""
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )

    lp, lm_head = _model_lm_head(model)
    logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=None)
    if lp.soft_cap is not None:
        logits = torch.tanh(logits / lp.soft_cap) * lp.soft_cap
    if lp.scale != 1.0:
        logits = logits * lp.scale
    num_pad = lm_head.shard_indices.num_org_vocab_padding
    if num_pad > 0:
        logits[..., -num_pad:] = -float("inf")
    local_max, local_idx = logits.max(dim=-1)
    global_idx = local_idx + lm_head.shard_indices.org_vocab_start_index
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return global_idx
    local_pair = torch.stack([local_max.float(), global_idx.float()], dim=-1)
    gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
    gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
    max_rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    top = gathered[:, :, 1].gather(dim=-1, index=max_rank)
    return top.squeeze(-1).to(torch.int64)


def s1_block_reason(runner, input_batch, grammar_output) -> str | None:
    """Why S1 must not run for this batch (None: S1 is exact here)."""
    import numpy as np

    if input_batch.num_draft_tokens <= 0:
        return "no_drafts"
    if grammar_output is not None:
        return "grammar"
    if getattr(runner, "lora_config", None) is not None:
        return "lora"
    rs = getattr(runner, "rejection_sampler", None)
    sampler = getattr(runner, "sampler", None)
    if rs is None or sampler is None or runner.speculator is None:
        return "no_sampler"
    if getattr(rs, "synthetic_conditional_rates", None) is not None:
        return "synthetic"
    if getattr(runner.speculator, "_debug_proposal_stages", False):
        return "debug"
    if capture_spec_decode_trace().resolve().target_logits:
        return "debug_logits"
    if not sampler.can_use_sm70_greedy_token_fastpath(input_batch):
        return "sampling_params"
    idx = input_batch.idx_mapping_np
    states = sampler.sampling_states
    for name, neutral in (("top_k", None), ("top_p", 1.0), ("min_p", 0.0)):
        buf = getattr(states, name, None)
        if buf is None:
            continue
        vals = buf.np[idx]
        if neutral is None:
            if np.any(vals != states.vocab_size):
                return name
        elif np.any(vals != neutral):
            return name
    if _model_lm_head(runner.model) is None:
        return "no_lm_head"
    if not _is_sm70(input_batch.input_ids):
        return "not_sm70"
    return None


# --------------------------------------------------------------------------
# h2: TurboMind MoE permute in one kernel
# --------------------------------------------------------------------------
@triton.jit
def _moe_permute_fused_kernel(
    x_ptr,  # fp16 [n_tokens, hidden]
    stride_x,
    topk_ids_ptr,  # int32/int64 [n_slots] (row-major [n_tokens, top_k])
    tei_ptr,  # int32 [n_slots] token_expert_indices
    output_ptr,  # fp16 [n_tokens, hidden] (zero filled)
    stride_output,
    permuted_input_ptr,  # fp16 [n_slots, hidden]
    stride_pi,
    offsets64_ptr,  # int64 [n_experts + 1]
    offsets32_ptr,  # int32 [n_experts + 1]
    inv_ptr,  # int32 [n_slots]
    permuted_idx_ptr,  # int32 [n_slots]
    experts_id_ptr,  # int32 [n_slots]
    sorted_row_ptr,  # int32 [n_slots]
    topk_buf_ptr,  # int32 [n_slots]
    n_slots,
    n_tokens,
    top_k,
    hidden,
    n_experts,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    smask = s < n_slots
    keys = tl.load(topk_ids_ptr + s, mask=smask, other=0).to(tl.int32)
    if pid < n_slots:
        key = tl.load(topk_ids_ptr + pid).to(tl.int32)
        # Stable ascending sort by expert id (cub LSD radix sort): the rank is
        # the number of smaller keys plus equal keys earlier in the input.
        before = (keys < key) | ((keys == key) & (s < pid))
        rank = tl.sum((before & smask).to(tl.int32), axis=0)
        val = tl.load(tei_ptr + pid)
        tl.store(sorted_row_ptr + rank, val)
        tl.store(experts_id_ptr + rank, key)
        tl.store(permuted_idx_ptr + rank, val)
        tl.store(inv_ptr + val, rank)
        tl.store(topk_buf_ptr + pid, key)
        src = (val // top_k).to(tl.int64)
        for h0 in range(0, hidden, BLOCK_H):
            h = h0 + tl.arange(0, BLOCK_H)
            hm = h < hidden
            row = tl.load(x_ptr + src * stride_x + h, mask=hm)
            tl.store(
                permuted_input_ptr + rank.to(tl.int64) * stride_pi + h, row, mask=hm
            )
    if pid <= n_experts:
        cnt = tl.sum(((keys < pid) & smask).to(tl.int32), axis=0)
        tl.store(offsets64_ptr + pid, cnt.to(tl.int64))
        tl.store(offsets32_ptr + pid, cnt)
    if pid < n_tokens:
        for h0 in range(0, hidden, BLOCK_H):
            h = h0 + tl.arange(0, BLOCK_H)
            tl.store(
                output_ptr + pid * stride_output + h,
                tl.zeros([BLOCK_H], dtype=tl.float16),
                mask=h < hidden,
            )


H2_MAX_SLOTS = 1024


def h2_supported(layer, x: torch.Tensor, topk_ids: torch.Tensor, buffers: dict) -> bool:
    n_tokens, hidden = x.shape
    n_slots = topk_ids.numel()
    try:
        ok = bool(
            layer.expert_map is None
            and int(layer.local_num_experts) == int(layer.global_num_experts)
            and 0 < n_slots <= H2_MAX_SLOTS
            and x.dtype == torch.float16
            and x.stride(1) == 1
            and topk_ids.is_contiguous()
            and topk_ids.dtype in (torch.int32, torch.int64)
            and buffers["permuted_input"].shape[0] >= n_slots
            and buffers["permuted_input"].stride(1) == 1
            and buffers["output"].stride(1) == 1
            and buffers["token_expert_indices"].is_contiguous()
            and buffers["token_expert_indices"].dtype == torch.int32
            and buffers["expert_offsets64"].numel() == int(layer.local_num_experts) + 1
            and buffers["expert_offsets"].numel() == int(layer.local_num_experts) + 1
            and buffers["permuted_idx"].numel() == n_slots
            and _is_sm70(x)
        )
    except (AttributeError, KeyError):
        return False
    return ok


def moe_permute_fused(
    layer, x: torch.Tensor, topk_ids: torch.Tensor, buffers: dict
) -> None:
    n_tokens, hidden = x.shape
    n_slots = topk_ids.numel()
    n_experts = int(layer.local_num_experts)
    top_k = topk_ids.shape[1]
    BLOCK_S = max(16, triton.next_power_of_2(n_slots))
    grid = (max(n_slots, n_experts + 1, n_tokens),)
    _moe_permute_fused_kernel[grid](
        x,
        x.stride(0),
        topk_ids,
        buffers["token_expert_indices"],
        buffers["output"],
        buffers["output"].stride(0),
        buffers["permuted_input"],
        buffers["permuted_input"].stride(0),
        buffers["expert_offsets64"],
        buffers["expert_offsets"],
        buffers["inv_permuted_idx"],
        buffers["permuted_idx"],
        buffers["permuted_experts_id"],
        buffers["sorted_row_idx"],
        buffers["topk_ids"],
        n_slots,
        n_tokens,
        top_k,
        hidden,
        n_experts,
        BLOCK_S=BLOCK_S,
        BLOCK_H=512,
        num_warps=4,
    )
