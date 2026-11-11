# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 target from the original safetensors checkpoint (VLLM_V41_NATIVE=1).

The official DeepSeek-V4.1-Flash checkpoint keeps dense projections as FP8
E4M3 with 32x32 UE8M0 block scales, routed experts as MXFP4 (E2M1 codes with
one E8M0 scale per 32) and the remaining tensors in BF16/FP32. This route keeps
every resident byte in that format:

- FP8 dense layers use the drafter's native ``V41Fp8LinearMethod``;
- BF16 norms, gates, compressors and the indexer's small projections load
  into the FP32/FP16 parameters of the GGUF route (checked to be exact);
- routed experts run on the skinny QPN MXFP4 kernels (``torch.ops.v41sk``,
  benchmarks/deepseek_v41/v41k/v41sk.cu), which read each distinct expert of
  a step once and keep the expert bytes in place.

The module tree is the GGUF route's; only the parameter source differs.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import regex as re
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    qpn_prepack,
    rebase_e8m0_for_fp16,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import Mxfp4SM70MoEMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.triton_utils import tl, triton

from .fp8_linear import V41Fp8LinearMethod

logger = init_logger(__name__)

NATIVE = envs.VLLM_V41_NATIVE == "1"
# Timing experiments only (default off): VLLM_V41_ABLATE=moe,dense skips that
# compute and returns zeros of the right shape. Outputs become meaningless.
ABLATE = set(filter(None, envs.VLLM_V41_ABLATE.split(",")))
# Router top-k and slot plan in one CUDA launch (v41sk.route_plan; same
# values as _route_i32 + _plan_kernel). Default off.
_ROUTE_CUDA = envs.VLLM_V41_ROUTE_CUDA == "1"
_ROUTE_MATCH = envs.VLLM_V41_ROUTE_MATCH == "1"
# Decode w13 on the double-buffered moe_dec kernel (v41sk). Default off.
_MOE_DEC = envs.VLLM_V41_MOE_DEC == "1"
# Decode gate/up projection with clipped SwiGLU in the same CUDA launch.
_W13_SWIGLU = envs.VLLM_V41_W13_SWIGLU == "1"
# FP16-weight HMMA GEMV (v41sk.h16_gemv) for the vocabulary heads, and the
# router gate fused with top-k + slot plan (v41sk.gate_route). Default off.
_H16 = envs.VLLM_V41_H16 == "1"
# Decode w2 + route sum in one launch (v41sk.moe_w2_route). Default off.
_W2_ROUTE = envs.VLLM_V41_W2_ROUTE == "1"
# Decode w13 as a persistent (expert, tile) stream: "ctas,warps,xch" (v41sk.
# moe_w13_units). Default off.
_W13_UNITS = tuple(int(v) for v in envs.VLLM_V41_W13_UNITS.split(",") if v) or None


def h16_prepack(weight: torch.Tensor) -> torch.Tensor:
    """[N, K] FP16 -> [N/32][K/16][32 lanes][16] in the QPN lane order."""
    n, k = weight.shape
    lane = torch.arange(32, device=weight.device)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    view = weight.reshape(n // 32, 32, k // 16, 16)[:, col]
    return view.permute(0, 2, 1, 3).contiguous().view(-1)


class H16Head:
    """Quant-method wrapper: M <= 8 FP16 head rows on h16_gemv."""

    def __init__(self, base, weight):
        self.base = base
        self.n = weight.shape[0]
        self.packed = h16_prepack(weight)
        self.tickets = torch.zeros(
            self.n // 32 + 1, dtype=torch.int32, device=weight.device
        )

    def apply(self, layer, x, bias=None):
        small = 1 <= x.shape[0] <= 8 and x.dtype == torch.float16
        if bias is None and x.is_cuda and small:
            out = torch.empty(x.shape[0], self.n, dtype=torch.float16, device=x.device)
            torch.ops.v41sk.h16_gemv(
                out, x.contiguous(), self.packed, self.tickets, 8, 1
            )
            return out
        return self.base.apply(layer, x, bias)

    def __getattr__(self, name):
        return getattr(self.base, name)


def attach_h16_head(head) -> bool:
    w = getattr(head, "weight", None)
    if (
        not _H16
        or w is None
        or not w.is_cuda
        or w.dtype != torch.float16
        or w.dim() != 2
        or w.shape[0] % 32
        or (w.shape[1] // 16) % 8
        or isinstance(head.quant_method, H16Head)
    ):
        return False
    load_v41sk()
    head.quant_method = H16Head(head.quant_method, w.data)
    return True


_BATCH_GATE = envs.VLLM_V41_BATCH == "1"


def attach_h16_gate(gate) -> bool:
    """FP16 copy of a BF16 gate for gate_route, only when the cast is exact."""
    w = gate.weight.data
    if (
        not _H16
        or not w.is_cuda
        or w.dtype not in (torch.bfloat16, torch.float32)
        or w.shape[0] not in (128, 384)
    ):
        return False
    # FP16 subnormals cannot hold the smallest BF16 coefficients: store
    # w * 2^s (largest s keeping |w| < 2^15) and scale the FP32 sums back by
    # 2^-s, both exact, so the logits equal the unscaled ones.
    peak = w.float().abs().max().item()
    shift = 0
    while peak * 2.0 ** (shift + 1) < 32768.0 and shift < 24:
        shift += 1
    w16 = (w.float() * 2.0**shift).half()
    if not torch.equal(w16.float() * 2.0**-shift, w.float()):
        return False
    load_v41sk()
    gate.v41_h16 = (
        h16_prepack(w16),
        torch.zeros(w.shape[0] // 32 + 1, dtype=torch.int32, device=w.device),
        2.0**-shift,
    )
    if _BATCH_GATE:
        # VLLM_V41_BATCH: the same exact FP16 * 2^s weight for M > 8 (cuBLAS).
        gate.v41_h16_plain = w16.t().contiguous()
    return True


def h16_exact(w: torch.Tensor, keep_plain: bool = False):
    """(packed FP16 * 2^s, tickets, 2^-s) of an FP32/BF16 [N, K] weight when the
    scaled FP16 copy is exact, else None (h16_gemv with oscale = 2^-s).
    keep_plain appends the unpacked [K, N] FP16 * 2^s copy (cuBLAS, M > 8)."""
    if (
        not _H16
        or not w.is_cuda
        or w.dtype not in (torch.bfloat16, torch.float32)
        or w.dim() != 2
        or w.shape[0] % 32
        or (w.shape[1] // 16) % 8
    ):
        return None
    peak = w.float().abs().max().item()
    if peak == 0:
        return None
    shift = 0
    while peak * 2.0 ** (shift + 1) < 32768.0 and shift < 24:
        shift += 1
    w16 = (w.float() * 2.0**shift).half()
    if not torch.equal(w16.float() * 2.0**-shift, w.float()):
        return None
    load_v41sk()
    tickets = torch.zeros(w.shape[0] // 32 + 1, dtype=torch.int32, device=w.device)
    if keep_plain:
        return h16_prepack(w16), tickets, 2.0**-shift, w16.t().contiguous()
    return h16_prepack(w16), tickets, 2.0**-shift


def gate_route(gate, x, top_k, scale):
    """(logits, probs, ids, plan) of the gate + routing in one launch, or None."""
    h16 = getattr(gate, "v41_h16", None)
    tokens = x.shape[0]
    if (
        h16 is None
        or x.dtype != torch.float16
        or not 1 <= tokens <= 8
        or tokens * top_k >= 64
        or getattr(gate, "temperature", 1.0) != 1.0
    ):
        return None
    packed, tickets, oscale = h16
    n, device = gate.weight.shape[0], x.device
    slots = tokens * top_k
    logits = torch.empty(tokens, n, dtype=torch.float32, device=device)
    probs = torch.empty(tokens, top_k, dtype=torch.float32, device=device)
    ids = torch.empty(tokens, top_k, dtype=torch.int32, device=device)
    perm = torch.empty(slots, dtype=torch.int32, device=device)
    experts = torch.empty(slots, dtype=torch.int32, device=device)
    offsets = torch.empty(slots + 1, dtype=torch.int32, device=device)
    # 8 warps x 5 K-parts measured best for K = 5120 (60 CTAs for 384 experts).
    parts = 5 if (x.shape[1] // 16) % 40 == 0 else 1
    torch.ops.v41sk.gate_route(
        logits,
        x.contiguous(),
        packed,
        tickets,
        gate.bias,
        probs,
        ids,
        perm,
        experts,
        offsets,
        parts,
        scale,
        oscale,
        _ROUTE_MATCH,
    )
    return logits, probs, ids, (perm, experts, offsets)


# TP4 one-shot push all-reduce (v41sk.ll_allreduce) for small FP16/FP32
# payloads: no barriers, the receive slots carry a sentinel. Default off.
_LL_AR = envs.VLLM_V41_LL_AR == "1"
_LL_CAP = 1024 * 1024
_LL_STATE: list = []


def _ll_state():
    if not _LL_STATE:
        import torch.distributed as dist

        from vllm import _custom_ops as ops
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        if group.world_size != 4:
            _LL_STATE.append(None)
            return None
        load_v41sk()
        size = 2 * 4 * _LL_CAP
        pointer, handle = ops.allocate_shared_buffer_and_handle(size)
        torch.ops.v41sk.ll_arm(pointer, size)
        torch.accelerator.synchronize()
        handles = [None] * 4
        dist.all_gather_object(handles, handle, group=group.cpu_group)
        rank = group.rank_in_group
        ptrs = [
            pointer if i == rank else ops.open_mem_handle(h)
            for i, h in enumerate(handles)
        ]
        dist.barrier(group=group.cpu_group)
        counter = torch.zeros(16, dtype=torch.int32, device="cuda")
        _LL_STATE.append((ptrs, rank, counter))
        logger.info(
            "V4.1 native: TP4 one-shot push all-reduce (%d KiB slots)", _LL_CAP // 1024
        )
    return _LL_STATE[0]


def tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_reduce

    if (
        _LL_AR
        and x.is_cuda
        and x.dtype in (torch.float16, torch.float32)
        and x.numel() * x.element_size() <= _LL_CAP
        and (x.numel() * x.element_size()) % 16 == 0
    ):
        state = _ll_state()
        if state is not None:
            ptrs, rank, counter = state
            x = x.contiguous()
            out = torch.empty_like(x)
            torch.ops.v41sk.ll_allreduce(out, x, *ptrs, rank, _LL_CAP, counter)
            return out
    return tensor_model_parallel_all_reduce(x)


def tp_all_gather(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Last-dim TP4 all-gather on the one-shot push slots, else the stock path."""
    from vllm.distributed import tensor_model_parallel_all_gather

    if (
        _LL_AR
        and x.is_cuda
        and x.dim() == 2
        and dim in (-1, 1)
        and x.dtype in (torch.float16, torch.float32)
        and x.numel() * x.element_size() <= _LL_CAP
        and (x.shape[1] * x.element_size()) % 16 == 0
    ):
        state = _ll_state()
        if state is not None:
            ptrs, rank, counter = state
            x = x.contiguous()
            out = torch.empty(
                x.shape[0], 4 * x.shape[1], dtype=x.dtype, device=x.device
            )
            torch.ops.v41sk.ll_allgather(out, x, *ptrs, rank, _LL_CAP, counter)
            return out
    return tensor_model_parallel_all_gather(x, dim)


# Checkpoint dense projections stored as FP8 E4M3 with 32x32 UE8M0 scales.
_FP8_SUFFIXES = (
    ".attn.wq_a",
    ".attn.wq_b",
    ".attn.wkv",
    ".attn.wo_a",
    ".attn.wo_b",
    ".indexer.wq_b",
    ".shared_experts.gate_proj",
    ".shared_experts.up_proj",
    ".shared_experts.down_proj",
    ".engram.wkv",
)


class V41TargetQuantConfig(Fp8Config):
    """Native FP8 dense, BF16-origin unquantized layers and MXFP4 experts."""

    is_scale_e8m0 = True
    v41_native = True

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            return V41NativeMoEMethod(layer.moe_config)
        if isinstance(layer, LinearBase):
            if prefix.endswith(_FP8_SUFFIXES):
                return V41Fp8LinearMethod()
            return UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        return super().get_quant_method(layer, prefix)

    def is_mxfp4_quant(self, prefix, layer):
        return isinstance(layer, FusedMoE)


# ---------------------------------------------------------------- experts

_V41SK: dict[str, bool] = {}


def v41sk_library() -> str:
    return envs.VLLM_V41SK_LIB or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        *("..",) * 3,
        "benchmarks",
        "deepseek_v41",
        "v41k",
        "build",
        "v41sk.so",
    )


def load_v41sk() -> None:
    if not _V41SK.get("loaded"):
        path = v41sk_library()
        if not os.path.exists(path):
            raise RuntimeError(
                f"V4.1 native experts need the v41sk extension at {path} "
                "(python benchmarks/deepseek_v41/v41k/build.py)"
            )
        torch.ops.load_library(path)
        _V41SK["loaded"] = True


def _splitk(k: int) -> int:
    """Measured-best specialization dividing K/16 (W13 K=5120: 16; W2 576: 6).

    On V100 at M=6 with ~20 distinct experts: W13 16/nacc1 639 GB/s; W2 split
    4/6 637 GB/s, 9 616, 12 529, 18 436 GB/s.
    """
    for split in (16, 6, 4, 8, 9, 10, 12, 18):
        if k % 64 == 0 and (k // 16) % split == 0:
            return split
    raise ValueError(f"no skinny split-K specialization for K={k}")


@triton.jit
def _plan_kernel(ids, perm, experts, offsets, S, SB: tl.constexpr):
    """One CTA: stable sort of S <= SB slot experts and their groups."""
    i = tl.arange(0, SB)
    j = tl.arange(0, SB)
    live_i, live_j = i < S, j < S
    a = tl.load(ids + i, live_i, 2**30)
    b = tl.load(ids + j, live_j, 2**30)
    before = (b[None, :] < a[:, None]) | (
        (b[None, :] == a[:, None]) & (j[None, :] < i[:, None])
    )
    rank = tl.sum((before & live_j[None, :]).to(tl.int32), 1)
    tl.store(perm + rank, i, live_i)
    # first slot of each expert: no earlier slot with the same expert
    first = (
        tl.sum(((b[None, :] == a[:, None]) & (j[None, :] < i[:, None])).to(tl.int32), 1)
        == 0
    )
    distinct_below = tl.sum(
        (
            (b[None, :] < a[:, None])
            & live_j[None, :]
            & (
                tl.sum((b[:, None] == b[None, :]) & (j[None, :] < j[:, None]), 1)[
                    None, :
                ]
                == 0
            )
        ).to(tl.int32),
        1,
    )
    groups = tl.sum((first & live_i).to(tl.int32), 0)
    tl.store(experts + distinct_below, a, live_i & first)
    tl.store(offsets + distinct_below, rank, live_i & first)
    tl.store(experts + j, 0, (j >= groups) & (j < S))
    tl.store(offsets + j, S, (j >= groups) & (j <= S))


@triton.jit
def _swiglu_prob_kernel(gate_up, probs, out, INTER: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = columns < INTER
    gate = tl.load(gate_up + slot * 2 * INTER + columns, live, 0).to(tl.float32)
    up = tl.load(gate_up + slot * 2 * INTER + INTER + columns, live, 0).to(tl.float32)
    gate = tl.minimum(gate, 10.0)
    up = tl.minimum(tl.maximum(up, -10.0), 10.0)
    value = gate / (1.0 + tl.exp(-gate)) * up * tl.load(probs + slot)
    tl.store(out + slot * INTER + columns, value.to(tl.float16), live)


@triton.jit
def _route_sum_kernel(
    slots,
    out,
    shared,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    SHARED: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), tl.float32)
    for k in range(TOPK):
        total += tl.load(slots + (row * TOPK + k) * H + columns).to(tl.float32)
    if SHARED:
        # the shared expert's FP16 output, added last as `routed + shared.float()`
        total += tl.load(shared + row * H + columns).to(tl.float32)
    tl.store(out + row * H + columns, total)


@triton.jit
def _route_i32(
    logits,
    bias,
    probs,
    ids,
    E: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """route_topk's kernel (same selection and FP32 values), int32 ids."""
    t = tl.program_id(0)
    e = tl.arange(0, BLOCK)
    valid = e < E
    g = tl.load(logits + t * E + e, valid, 0.0)
    soft = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(tl.minimum(g, 20.0))))
    score = tl.sqrt(soft)
    key = tl.where(valid, score + tl.load(bias + e, valid, 0.0), float("-inf"))
    k = tl.arange(0, 8)
    chosen = tl.zeros((8,), tl.int32)
    values = tl.zeros((8,), tl.float32)
    for j in tl.static_range(K):
        best = tl.max(key, 0)
        index = tl.min(tl.where(key == best, e, BLOCK), 0)
        value = tl.sum(tl.where(e == index, score, 0.0), 0)
        chosen = tl.where(k == j, index, chosen)
        values = tl.where(k == j, value, values)
        key = tl.where(e == index, float("-inf"), key)
    total = tl.sum(tl.where(k < K, values, 0.0), 0)
    tl.store(probs + t * K + k, values / (total + 1e-20) * SCALE, k < K)
    tl.store(ids + t * K + k, chosen, k < K)


def route_and_plan(logits, bias, top_k, scale):
    """(probs, int32 ids, (perm, experts, offsets)) or None when ineligible."""
    tokens, n = logits.shape
    if (
        _ROUTE_CUDA
        and logits.is_cuda
        and logits.dtype == torch.float32
        and 64 <= tokens * top_k <= 640
        and top_k <= 8
        and n in (128, 256, 384)
    ):
        # Many tokens (VLLM_V41_BATCH): top-k and the expert-sorted slot plan
        # in one CUDA launch (v41sk.route_plan_wide) instead of Triton route +
        # plan kernels.
        load_v41sk()
        slots = tokens * top_k
        device = logits.device
        probs = torch.empty(tokens, top_k, dtype=torch.float32, device=device)
        ids = torch.empty(tokens, top_k, dtype=torch.int32, device=device)
        perm = torch.empty(slots, dtype=torch.int32, device=device)
        experts = torch.empty(slots, dtype=torch.int32, device=device)
        offsets = torch.empty(slots + 1, dtype=torch.int32, device=device)
        torch.ops.v41sk.route_plan_wide(
            logits.contiguous(), bias, probs, ids, perm, experts, offsets, scale
        )
        return probs, ids, (perm, experts, offsets)
    if (
        not logits.is_cuda
        or logits.dtype != torch.float32
        or tokens * top_k >= 64
        or tokens == 0
        or top_k > 8
    ):
        return None
    device = logits.device
    slots = tokens * top_k
    probs = torch.empty(tokens, top_k, dtype=torch.float32, device=device)
    ids = torch.empty(tokens, top_k, dtype=torch.int32, device=device)
    perm = torch.empty(slots, dtype=torch.int32, device=device)
    experts = torch.empty(slots, dtype=torch.int32, device=device)
    offsets = torch.empty(slots + 1, dtype=torch.int32, device=device)
    if _ROUTE_CUDA and tokens <= 8 and n in (128, 256, 384):
        load_v41sk()
        torch.ops.v41sk.route_plan(
            logits.contiguous(),
            bias,
            probs,
            ids,
            perm,
            experts,
            offsets,
            scale,
            _ROUTE_MATCH,
        )
        return probs, ids, (perm, experts, offsets)
    _route_i32[(tokens,)](
        logits.contiguous(),
        bias,
        probs,
        ids,
        E=n,
        K=top_k,
        SCALE=scale,
        BLOCK=triton.next_power_of_2(n),
        num_warps=4,
    )
    _plan_kernel[(1,)](ids, perm, experts, offsets, slots, SB=64)
    return probs, ids, (perm, experts, offsets)


@triton.jit
def _plan_batch_kernel(
    ids, perm, experts, offsets, S, SB: tl.constexpr, JB: tl.constexpr
):
    """_route_plan_torch in one program for S <= SB slots: the stable rank of
    every slot by expert, the first slot of each expert, the expert groups."""
    s = tl.arange(0, SB)
    live = s < S
    mine = tl.load(ids + s, live, 1 << 30)
    rank = tl.zeros((SB,), tl.int32)
    first = live
    for j0 in range(0, SB, JB):
        j = j0 + tl.arange(0, JB)
        other = tl.load(ids + j, j < S, 1 << 30)
        before = (j[None, :] < s[:, None]) & (j[None, :] < S)
        below = (other[None, :] < mine[:, None]) & (j[None, :] < S)
        same = (other[None, :] == mine[:, None]) & before
        rank += tl.sum((below | same).to(tl.int32), 1)
        first &= tl.sum(same.to(tl.int32), 1) == 0
    tl.store(perm + rank, s, live)
    # Distinct experts below each slot's expert = its group index.
    group = tl.zeros((SB,), tl.int32)
    for j0 in range(0, SB, JB):
        j = j0 + tl.arange(0, JB)
        other = tl.load(ids + j, j < S, 1 << 30)
        jfirst = tl.load(ids + j, j < S, 1 << 30)
        # first flag of slot j: recompute (no j' < j with the same expert).
        jrank_first = tl.full((JB,), 1, tl.int1)
        for k0 in range(0, SB, JB):
            k = k0 + tl.arange(0, JB)
            prior = tl.load(ids + k, k < S, 1 << 30)
            dup = (prior[None, :] == jfirst[:, None]) & (k[None, :] < j[:, None])
            jrank_first &= tl.sum(dup.to(tl.int32), 1) == 0
        counted = jrank_first & (j < S)
        group += tl.sum(
            (counted[None, :] & (other[None, :] < mine[:, None])).to(tl.int32), 1
        )
    groups = tl.sum(first.to(tl.int32), 0)
    tl.store(experts + group, mine, live & first)
    tl.store(offsets + group, rank, live & first)
    tail = tl.arange(0, SB)
    tl.store(experts + tail, 0, (tail >= groups) & (tail < S))
    tl.store(offsets + tail, S, (tail >= groups) & (tail <= S))
    tl.store(offsets + S, S, groups <= S)


def _route_plan(topk_ids: torch.Tensor):
    """Slots sorted by expert; one group per distinct expert (graph safe)."""
    slots = topk_ids.numel()
    if 64 < slots <= 512:
        device = topk_ids.device
        perm = torch.empty(slots, dtype=torch.int32, device=device)
        experts = torch.empty(slots, dtype=torch.int32, device=device)
        offsets = torch.empty(slots + 1, dtype=torch.int32, device=device)
        _plan_batch_kernel[(1,)](
            topk_ids.reshape(-1).to(torch.int32),
            perm,
            experts,
            offsets,
            slots,
            SB=triton.next_power_of_2(slots),
            JB=32,
            num_warps=8,
        )
        return perm, experts, offsets
    if slots <= 64:
        device = topk_ids.device
        perm = torch.empty(slots, dtype=torch.int32, device=device)
        experts = torch.empty(slots, dtype=torch.int32, device=device)
        offsets = torch.empty(slots + 1, dtype=torch.int32, device=device)
        _plan_kernel[(1,)](
            topk_ids.reshape(-1).to(torch.int32), perm, experts, offsets, slots, SB=64
        )
        return perm, experts, offsets
    return _route_plan_torch(topk_ids)


def _route_plan_torch(topk_ids: torch.Tensor):
    flat = topk_ids.reshape(-1).to(torch.int64)
    slots = flat.numel()
    device = flat.device
    perm = torch.argsort(flat, stable=True)
    ordered = flat[perm]
    first = torch.ones(slots, dtype=torch.bool, device=device)
    first[1:] = ordered[1:] != ordered[:-1]
    group = torch.cumsum(first, 0) - 1
    offsets = torch.full((slots + 1,), slots, dtype=torch.int64, device=device)
    offsets.scatter_reduce_(
        0, group, torch.arange(slots, dtype=torch.int64, device=device), reduce="amin"
    )
    experts = torch.zeros(slots, dtype=torch.int64, device=device)
    experts.scatter_(0, group, ordered)
    return perm.to(torch.int32), experts.to(torch.int32), offsets.to(torch.int32)


class V41NativeMoEMethod(Mxfp4SM70MoEMethod):
    """384 x top-6 MXFP4 target experts on the skinny grouped QPN kernels.

    Weight parameters and loading are the MXFP4 layout of ``Mxfp4MoEMethod``.
    After loading, each expert's codes and scales are permuted in place into
    mma.m8n8k4 fragment order and every expert's E8M0 scales are rebased into
    the FP16 decode window (exact; the shift returns as a per-expert global
    scale). ``apply`` returns the unweighted-sum FP32 routed output, with the
    probability applied to the clamped SwiGLU before the FP16 down input, the
    rounding points of the GGUF route and of the drafter.
    """

    def __init__(self, moe: FusedMoEConfig):
        FusedMoEMethodBase.__init__(self, moe)
        from vllm.config.sm70_moe import capture_mxfp4_moe_config

        self.sm70_moe_policy = capture_mxfp4_moe_config()
        self.weight_dtype = "mxfp4"
        # Target 384 x top-6 and the DSpark drafter's 128 x top-3.
        if (moe.num_experts, moe.experts_per_token) not in ((384, 6), (128, 3)) or (
            moe.hidden_dim,
            moe.intermediate_size_per_partition,
        ) != (5120, 2304 // max(moe.tp_size, 1)):
            raise NotImplementedError("V4.1 native experts: 384 x 6 or 128 x 3")
        if moe.moe_parallel_config.use_all2all_kernels:
            raise NotImplementedError("V4.1 native experts do not run DP+EP all-to-all")

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        return FusedMoEMethodBase.maybe_roundup_sizes(
            self,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )

    def process_weights_after_loading(self, layer) -> None:
        load_v41sk()
        w13, w2 = layer.w13_weight.data, layer.w2_weight.data
        s13, s2 = layer.w13_weight_scale.data, layer.w2_weight_scale.data
        # Rebase per expert (exact powers of two), then permute in place.
        layer.v41_g13 = rebase_e8m0_for_fp16(s13).to(w13.device).contiguous()
        layer.v41_g2 = rebase_e8m0_for_fp16(s2).to(w13.device).contiguous()
        for expert in range(w13.shape[0]):
            for codes, scales in ((w13, s13), (w2, s2)):
                packed_codes, packed_scales = qpn_prepack(
                    codes[expert], scales[expert], 32
                )
                codes[expert].view(-1).copy_(packed_codes)
                scales[expert].view(-1).copy_(packed_scales)
        layer.v41_split13 = _splitk(w13.shape[2] * 2)
        layer.v41_split2 = _splitk(w2.shape[2] * 2)
        logger.info_once(
            "V4.1 native experts: %d MXFP4 experts in QPN fragment order, "
            "skinny split-K %d/%d (v41sk)",
            w13.shape[0],
            layer.v41_split13,
            layer.v41_split2,
        )

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        tokens, top_k = topk_ids.shape
        if tokens == 0 or "moe" in ABLATE:
            return x.new_zeros((tokens, x.shape[1]), dtype=torch.float32)
        x = x.contiguous()
        plan = getattr(layer, "v41_plan", None)
        if plan is not None and plan[0] is topk_ids:
            perm, experts, offsets = plan[1]
        else:
            perm, experts, offsets = _route_plan(topk_ids)
        slots = tokens * top_k
        w13, w2 = layer.w13_weight, layer.w2_weight
        inter = w2.shape[2] * 2
        activated = torch.empty((slots, inter), dtype=x.dtype, device=x.device)
        probabilities = topk_weights.reshape(-1).float().contiguous()
        if _W13_SWIGLU and 1 <= tokens <= 8 and x.shape[1] == 5120 and inter % 32 == 0:
            torch.ops.v41sk.moe_w13_swiglu(
                x,
                w13,
                layer.w13_weight_scale,
                layer.v41_g13,
                probabilities,
                perm,
                experts,
                offsets,
                top_k,
                activated,
                8,
            )
        else:
            gate_up = torch.empty((slots, w13.shape[1]), dtype=x.dtype, device=x.device)
            if _W13_UNITS and tokens <= 6 and (x.shape[1] // 16) % _W13_UNITS[1] == 0:
                # Decode: persistent (expert, tile) stream, x staged once.
                torch.ops.v41sk.moe_w13_units(
                    x,
                    w13,
                    layer.w13_weight_scale,
                    layer.v41_g13,
                    perm,
                    experts,
                    offsets,
                    top_k,
                    gate_up,
                    *_W13_UNITS,
                )
            elif _MOE_DEC and tokens <= 8 and (x.shape[1] // 16) % 4 == 0:
                # Decode: double-buffered activation chunks (moe_dec_kernel),
                # 4 warps x 8-group chunks measured best for K = 5120.
                torch.ops.v41sk.skinny_moe_dec_sm70(
                    x,
                    w13,
                    layer.w13_weight_scale,
                    layer.v41_g13,
                    perm,
                    experts,
                    offsets,
                    top_k,
                    gate_up,
                    False,
                    4,
                    8,
                )
            else:
                torch.ops.v41sk.skinny_moe_qpn_sm70(
                    x,
                    w13,
                    layer.w13_weight_scale,
                    layer.v41_g13,
                    perm,
                    experts,
                    offsets,
                    top_k,
                    gate_up,
                    False,
                    tokens,
                    layer.v41_split13,
                    1,
                    1,
                )
            _swiglu_prob_kernel[(slots, triton.cdiv(inter, 256))](
                gate_up,
                probabilities,
                activated,
                INTER=inter,
                BLOCK=256,
                num_warps=4,
            )
        hidden = w2.shape[1]
        if _W2_ROUTE and tokens <= 8 and (inter // 16) % 6 == 0:
            # w2 + route sum (+ the side-stream shared expert) in one launch.
            routed = torch.empty((tokens, hidden), dtype=torch.float32, device=x.device)
            join = getattr(layer, "v41_shared_join", None)
            shared = None
            if join is not None:
                shared, event = join
                layer.v41_shared_join = None
                torch.cuda.current_stream().wait_event(event)
            torch.ops.v41sk.moe_w2_route(
                activated,
                w2,
                layer.w2_weight_scale,
                layer.v41_g2,
                perm,
                experts,
                offsets,
                top_k,
                shared,
                routed,
                6,
            )
            if join is not None:
                layer.v41_shared_folded = True
            return routed
        down = torch.empty((slots, w2.shape[1]), dtype=x.dtype, device=x.device)
        torch.ops.v41sk.skinny_moe_qpn_sm70(
            activated,
            w2,
            layer.w2_weight_scale,
            layer.v41_g2,
            perm,
            experts,
            offsets,
            top_k,
            down,
            True,
            tokens,
            layer.v41_split2,
            1,
            1,
        )
        # Routed partials can exceed FP16 range; sum the routes in FP32.
        hidden = w2.shape[1]
        routed = torch.empty((tokens, hidden), dtype=torch.float32, device=x.device)
        # A side-stream shared expert (V41MoE._overlapped_partials) joins here
        # and is folded into the route sum instead of a separate add.
        join = getattr(layer, "v41_shared_join", None)
        if join is not None:
            shared, event = join
            layer.v41_shared_join = None
            torch.cuda.current_stream().wait_event(event)
        _route_sum_kernel[(tokens, triton.cdiv(hidden, 1024))](
            down,
            routed,
            shared if join is not None else routed,
            TOPK=top_k,
            H=hidden,
            BLOCK=1024,
            SHARED=join is not None,
            num_warps=4,
        )
        if join is not None:
            layer.v41_shared_folded = True
        return routed


# ---------------------------------------------------------------- loading

_SKIP = re.compile(
    r"(mtp\..*|vision.*|aligner\..*|image_.*|layers\.\d+\.engram\.embed\..*"
    r"|layers\.\d+\.ffn\.gate\.bias_vl)"
)


def native_weight_name(name: str) -> str | None:
    """Checkpoint name -> target parameter name (None: not a target tensor)."""
    if name.startswith("model."):
        return name  # sidecar tensors already carry parameter names
    if _SKIP.fullmatch(name):
        return None
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if not name.startswith("layers."):
        raise ValueError(f"Unknown V4.1 checkpoint tensor: {name}")
    expert = ".experts." in name and ".shared_experts." not in name
    for source, target in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
        name = name.replace(f".shared_experts.{source}.", f".shared_experts.{target}.")
        name = re.sub(rf"(\.experts\.\d+\.){source}\.", rf"\g<1>{target}.", name)
    if name.endswith(".scale"):
        name = name.removesuffix(".scale") + (
            ".weight_scale" if expert else ".weight_scale_inv"
        )
    return "model." + name


def _layer_of(name: str) -> int | None:
    match = re.match(r"model\.layers\.(\d+)\.", name)
    return int(match[1]) if match else None


_EXPERT = re.compile(r"(model\.layers\.\d+\.ffn\.experts\.)(\d+)\.(\w+_proj)\.(.+)")
# BF16 checkpoint tensors held as FP16 parameters (as on the GGUF route):
# values below FP16's normal range round; they must stay finite.
_ROUNDED = (
    "model.embed_tokens.weight",
    "lm_head.weight",
    ".compressor.wkv.weight",
    ".indexer.weights_proj.weight",
    ".indexer.wk.weight",
)


def _converted(value, parameter, name):
    if not value.is_floating_point() or parameter.dtype == value.dtype:
        return value
    converted = value.to(parameter.dtype)
    if name.endswith(_ROUNDED) and parameter.dtype == torch.float16:
        if not torch.isfinite(converted).all():
            raise ValueError(f"V4.1 native: {name} overflows {parameter.dtype}")
        changed = int((converted.to(value.dtype) != value).sum())
        if changed:
            logger.info("V4.1 native: %s rounds %d values to FP16", name, changed)
        return converted
    if not torch.equal(converted.to(value.dtype), value):
        raise ValueError(f"V4.1 native: {name} is not exact in {parameter.dtype}")
    return converted


def load_native_weights(model, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Load this stage's target tensors; refuse a partial stage."""
    language = model.model
    start, end = language.start_layer, language.end_layer
    parameters = dict(model.named_parameters())
    shards = {
        "gate_proj": ("w13_", "w1"),
        "up_proj": ("w13_", "w3"),
        "down_proj": ("w2_", "w2"),
    }
    loaded: set[str] = set()
    for source, value in weights:
        name = native_weight_name(source)
        if name is None:
            continue
        layer = _layer_of(name)
        if layer is not None and not start <= layer < end:
            continue
        expert = _EXPERT.fullmatch(name)
        if expert is not None:
            prefix, expert_id, projection, suffix = expert.groups()
            target, shard = shards[projection]
            mapped = prefix + target + suffix
            parameter = parameters[mapped]
            parameter.weight_loader(
                parameter,
                value.view(torch.uint8),
                mapped,
                shard_id=shard,
                expert_id=int(expert_id),
            )
            loaded.add(mapped)
            continue
        if name not in parameters:
            if name.startswith(("lm_head.", "model.norm.")):
                continue  # owned by the last stage
            raise ValueError(f"V4.1 native: no parameter for {source} ({name})")
        parameter = parameters[name]
        value = _converted(value, parameter, name)
        loader = getattr(parameter, "weight_loader", default_weight_loader)
        loader(parameter, value)
        loaded.add(name)
    missing = sorted(
        name
        for name in parameters
        if name not in loaded
        and ((layer := _layer_of(name)) is None or start <= layer < end)
    )
    if missing:
        raise ValueError(f"V4.1 native: missing parameters {missing[:12]}")
    return loaded


# ---- last-stage sample bookkeeping in one launch (VLLM_V41_FUSED_CLIP) ----
# The runner's _clip_stateful_sample_limits (output budget, stop tokens) and
# _count_contiguous_spec_tokens are ~25 eager torch launches per round on the
# last stage. One program does both for the single request.
FUSED_CLIP = envs.VLLM_V41_FUSED_CLIP == "1"
_STOP_IDS: dict = {}


@triton.jit
def _clip_count_kernel(
    ids,
    count,
    prefill,
    position,
    length,
    stops,
    prompt_tokens,
    max_tokens,
    L,
    NS,
    LB: tl.constexpr,
    SB: tl.constexpr,
):
    j = tl.arange(0, LB)
    live = j < L
    v = tl.load(ids + j, live, -1).to(tl.int64)
    pre = tl.load(prefill).to(tl.int1)
    pos = tl.load(position)
    before = tl.where(pre, pos + tl.load(length), pos + 1)
    before = tl.maximum(before - prompt_tokens, 0)
    budget = tl.maximum(max_tokens - before, 0)
    v = tl.where(j >= budget, -1, v)
    k = tl.arange(0, SB)
    sv = tl.load(stops + k, k < NS, -2)
    is_stop = tl.sum((v[:, None] == sv[None, :]).to(tl.int32), 1) > 0
    s32 = is_stop.to(tl.int32)
    # after the first stop: inclusive prefix count minus self > 0
    v = tl.where(tl.cumsum(s32, 0) - s32 > 0, -1, v)
    tl.store(ids + j, v.to(ids.dtype.element_ty), live)
    first = tl.min(tl.where(live & (v != -1), L, tl.where(live, j, L)), 0)
    tl.store(count, first.to(tl.int32))


def fused_clip_count(model, ids, max_tokens, prompt_tokens, stop_ids):
    """In-place clip of row 0 of ids [1, L]; returns the contiguous count [1]."""
    prefill, position, length = model.v41_pending()
    key = (ids.device, tuple(stop_ids))
    stops = _STOP_IDS.get(key)
    if stops is None:
        values = list(stop_ids) or [-2]
        stops = _STOP_IDS[key] = torch.tensor(
            values, dtype=torch.int64, device=ids.device
        )
    count = torch.empty(1, dtype=torch.int32, device=ids.device)
    L = ids.shape[1]
    _clip_count_kernel[(1,)](
        ids,
        count,
        prefill,
        position,
        length,
        stops,
        prompt_tokens,
        max_tokens,
        L,
        len(stop_ids),
        LB=max(16, triton.next_power_of_2(L)),
        SB=max(16, triton.next_power_of_2(max(1, len(stop_ids)))),
    )
    return count
