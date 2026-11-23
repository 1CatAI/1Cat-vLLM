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
_LL_CHUNK = os.environ.get("VLLM_V41_LL_CHUNK", "1") == "1"
_LL_CAP = 1024 * 1024
_LL_SYNC_BYTES = 80 * 16
_LL_STATES: dict = {}
# Expert-parallel dispatch/combine over the DP group: up to ~200 decode rows,
# on 64 blocks (16 move ~30 GB/s at 96 rows).
_LL_DP_CAP = 4 * 1024 * 1024
_LL_DP_BLOCKS = 64


def _ll_group_state(group, cap, blocks=16, tag=""):
    """(peer slot pointers, rank, counter) of a 4-rank group, or None (one
    slot set per group and tag)."""
    key = group.unique_name + tag
    if key not in _LL_STATES:
        import torch.distributed as dist

        from vllm import _custom_ops as ops

        if group.world_size != 4:
            _LL_STATES[key] = None
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        load_v41sk()
        size = 2 * 4 * cap
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
        # One counter per block: the kernels launch one block per entry.
        counter = torch.zeros(blocks, dtype=torch.int32, device="cuda")
        _LL_STATES[key] = (ptrs, rank, counter)
        logger.info("V4.1 native: one-shot push slots on %s (%d KiB)", key, cap // 1024)
    return _LL_STATES[key]


def _ll_pf_state():
    from vllm.distributed import get_tp_group

    return _ll_group_state(get_tp_group(), _LL_CAP, blocks=64, tag="pf")


def _ll_state():
    from vllm.distributed import get_tp_group

    if _LL_CHUNK and not torch.cuda.is_current_stream_capturing():
        _ll_pf_state()
    return _ll_group_state(get_tp_group(), _LL_CAP)


def tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_reduce

    nbytes = x.numel() * x.element_size()
    if (
        _LL_AR
        and x.is_cuda
        and x.dtype in (torch.float16, torch.float32)
        and nbytes > 0
        and nbytes % 16 == 0
        and nbytes <= _LL_CAP - _LL_SYNC_BYTES
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

    nbytes = x.numel() * x.element_size()
    if (
        _LL_AR
        and x.is_cuda
        and x.dim() == 2
        and dim in (-1, 1)
        and x.dtype in (torch.float16, torch.float32)
        and nbytes > 0
        and (x.shape[1] * x.element_size()) % 16 == 0
        and (_LL_CHUNK or nbytes <= _LL_CAP - _LL_SYNC_BYTES)
    ):
        state = (
            _ll_state()
            if nbytes <= _LL_CAP - _LL_SYNC_BYTES
            else _ll_pf_state()
        )
        if state is not None:
            ptrs, rank, counter = state
            x = x.contiguous()
            out = torch.empty(
                x.shape[0], 4 * x.shape[1], dtype=x.dtype, device=x.device
            )
            torch.ops.v41sk.ll_allgather(out, x, *ptrs, rank, _LL_CAP, counter)
            return out
    return tensor_model_parallel_all_gather(x, dim)


def _dp_slots(rows, row_bytes):
    """The DP group's push slots when every rank sends `rows` equal rows."""
    from vllm.distributed import get_dp_group
    from vllm.forward_context import get_forward_context

    if (
        not _LL_AR
        or rows * row_bytes > _LL_DP_CAP - _LL_SYNC_BYTES
        or row_bytes % 16
    ):
        return None
    metadata = get_forward_context().dp_metadata
    if metadata is None or any(
        size != rows for size in metadata.num_tokens_across_dp_cpu.tolist()
    ):
        return None
    return _ll_group_state(get_dp_group(), _LL_DP_CAP, _LL_DP_BLOCKS)


def _dp_var_slots(rows, row_bytes):
    """The DP group's push slots + per-rank row counts for large/uneven steps."""
    from vllm.distributed import get_dp_group
    from vllm.forward_context import get_forward_context

    if not _LL_AR or not _LL_CHUNK or row_bytes % 16:
        return None
    metadata = get_forward_context().dp_metadata
    if metadata is None:
        return None
    sizes = metadata.num_tokens_across_dp_cpu.tolist()
    if len(sizes) != 4 or sum(sizes) * row_bytes > _LL_DP_CAP - _LL_SYNC_BYTES:
        return None
    state = _ll_group_state(get_dp_group(), _LL_DP_CAP, _LL_DP_BLOCKS)
    if state is None or sizes[state[1]] != rows:
        return None
    return state, sizes


def _stock_dp(method, *tensors):
    """The EP group's all-gatherv dispatch / reduce-scatterv combine."""
    from vllm.distributed import get_ep_group
    from vllm.forward_context import get_forward_context

    with get_forward_context().dp_metadata.sp_local_sizes(1):
        return getattr(get_ep_group(), method)(*tensors)


@triton.jit
def _pack_routes_kernel(
    x,
    probs,
    ids,
    real,
    packed,
    dead_id,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    KB: tl.constexpr,
    TILES: tl.constexpr = 1,
):
    """One dispatch row: hidden (FP16), route weights (FP32) and ids (int32,
    `dead_id` past the step's real rows) packed into WIDTH FP16 lanes."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    lanes = tile * BLOCK + tl.arange(0, BLOCK)
    for start in range(0, H, BLOCK * TILES):
        live = start + lanes < H
        value = tl.load(x + row * H + start + lanes, live)
        tl.store(packed + row * WIDTH + start + lanes, value, live)
    if tile != 0:
        return
    k = tl.arange(0, KB)
    kl = k < TOPK
    weight = tl.load(probs + row * TOPK + k, kl).to(tl.float32)
    route = tl.load(ids + row * TOPK + k, kl).to(tl.int32)
    route = tl.where(row < tl.load(real), route, dead_id)
    meta = packed + row * WIDTH + H
    tl.store(meta.to(tl.pointer_type(tl.float32)) + k, weight, kl)
    tl.store((meta + 2 * TOPK).to(tl.pointer_type(tl.int32)) + k, route, kl)


@triton.jit
def _unpack_routes_kernel(
    packed,
    x,
    probs,
    ids,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    KB: tl.constexpr,
    copies=None,
    counts=None,
    TILES: tl.constexpr = 1,
    LOGICAL: tl.constexpr = 0,
    REPLICAS: tl.constexpr = 0,
    DEAD: tl.constexpr = 0,
):
    """Gathered packed rows -> contiguous hidden, FP32 weights, int32 ids."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    lanes = tile * BLOCK + tl.arange(0, BLOCK)
    for start in range(0, H, BLOCK * TILES):
        live = start + lanes < H
        value = tl.load(packed + row * WIDTH + start + lanes, live)
        tl.store(x + row * H + start + lanes, value, live)
    if tile != 0:
        return
    k = tl.arange(0, KB)
    kl = k < TOPK
    meta = packed + row * WIDTH + H
    weight = tl.load(meta.to(tl.pointer_type(tl.float32)) + k, kl)
    route = tl.load((meta + 2 * TOPK).to(tl.pointer_type(tl.int32)) + k, kl)
    if REPLICAS:
        live = kl & (route >= 0) & (route < LOGICAL)
        n = tl.load(counts + route, live, 1)
        route = tl.load(copies + route * REPLICAS + row % n, live, DEAD)
    tl.store(probs + row * TOPK + k, weight, kl)
    tl.store(ids + row * TOPK + k, route, kl)


def solo_dispatch_mapped(x, probs, ids, real, tables, dead, owner):
    """Publish the owner's six rows and mapped top-k to its three DP peers."""
    from vllm.distributed import get_dp_group

    load_v41sk()
    op = getattr(torch.ops.v41solo, "dispatch", None)
    if op is None:
        return None
    state = _ll_group_state(get_dp_group(), 65536, blocks=6, tag="solo_routes")
    if state is None:
        raise RuntimeError("solo route exchange needs its prewarmed four-rank slots")
    peers, me, sequence = state
    out = x.new_empty((6, 5120))
    weights = probs.new_empty((6, 6), dtype=torch.float32)
    physical = ids.new_empty((6, 6), dtype=torch.int32)
    op(x.contiguous(), probs.float().contiguous(), ids.int().contiguous(),
       real, *tables, dead, out, weights, physical, *peers, me, owner, sequence, 1)
    return out, weights, physical


def dp_dispatch(x, probs, ids, real=None, dead_id=None, solo_sizes=None, replica_tables=None):
    """Expert-parallel dispatch: all-gather (hidden, route weights, route ids)
    over the DP group in one packed one-shot push, else the stock all-gatherv.
    Rows at or past real[0] route to dead_id (an id no rank owns; it must not be
    -1, the all-ones slot sentinel). Returns contiguous gathered tensors."""
    rows, hidden = x.shape
    top_k = ids.shape[1]
    width = hidden + 4 * top_k  # FP16 lanes: hidden, FP32 weights, int32 ids
    state = None
    var = None
    if solo_sizes is not None:
        from vllm.distributed import get_dp_group

        state = _ll_group_state(get_dp_group(), _LL_DP_CAP, _LL_DP_BLOCKS)
        assert state is not None and rows == solo_sizes[state[1]]
        var = state, solo_sizes
        state = None
    elif x.dtype == torch.float16 and x.is_cuda and x.is_contiguous():
        if rows:
            state = _dp_slots(rows, width * 2)
        if state is None:
            var = _dp_var_slots(rows, width * 2)
    if state is None and var is None:
        probs, ids = probs.float(), ids.int()
        if real is not None:
            mask_padded_routes(ids, real, dead_id)
        xs, ps, gathered = _stock_dp("dispatch", x, probs, ids)
        if replica_tables is not None:
            gathered = map_replica_routes(gathered, *replica_tables, dead_id)
        return xs, ps, gathered
    mapped_dead = dead_id
    tiled = (
        envs.VLLM_V41_EP4_TP4 == "1" and replica_tables is not None
        and 0 < rows <= 96 and hidden == 5120
    )
    block = 512 if tiled and rows <= 12 else 1024
    tiles = triton.cdiv(hidden, block) if tiled else 1
    meta = dict(H=hidden, TOPK=top_k, WIDTH=width, BLOCK=block, KB=8, TILES=tiles)
    if real is None:
        real = torch.full((1,), rows, dtype=torch.int32, device=x.device)
        dead_id = 0
    packed = torch.empty(rows, width, dtype=torch.float16, device=x.device)
    if rows > 0:
        _pack_routes_kernel[(rows, tiles)](
            x, probs.contiguous(), ids.contiguous(), real, packed, dead_id, **meta
        )
    if state is not None:
        ptrs, rank, counter = state
        total_rows = 4 * rows
        out = torch.empty(total_rows, width, dtype=torch.float16, device=x.device)
        torch.ops.v41sk.ll_allgather(
            out.view(1, -1), packed.view(1, -1), *ptrs, rank, _LL_DP_CAP, counter
        )
    else:
        (ptrs, rank, counter), sizes = var
        total_rows = sum(sizes)
        out = torch.empty(total_rows, width, dtype=torch.float16, device=x.device)
        torch.ops.v41sk.ll_dp_allgather_var(
            out, packed, *ptrs, rank, *sizes, _LL_DP_CAP, counter
        )
    xs = torch.empty(total_rows, hidden, dtype=torch.float16, device=x.device)
    ps = torch.empty(total_rows, top_k, dtype=torch.float32, device=x.device)
    gathered = torch.empty(total_rows, top_k, dtype=torch.int32, device=x.device)
    if total_rows > 0:
        copies, counts = replica_tables if replica_tables is not None else (ids, ids)
        _unpack_routes_kernel[(total_rows, tiles)](
            out, xs, ps, gathered, copies=copies, counts=counts,
            LOGICAL=copies.shape[0] if replica_tables is not None else 0,
            REPLICAS=copies.shape[1] if replica_tables is not None else 0,
            DEAD=mapped_dead if replica_tables is not None else 0, **meta,
        )
    return xs, ps, gathered


def dp_combine(routed):
    """Expert-parallel combine: reduce-scatter the gathered rows' FP32
    partials back to their DP ranks (one-shot push), else reduce_scatterv."""
    rows = routed.shape[0] // 4
    state = None
    if routed.is_cuda and routed.dtype == torch.float32 and rows * 4 == routed.shape[0]:
        state = _dp_slots(rows, routed.shape[1] * 4)
    if state is None:
        return _stock_dp("combine", routed)
    ptrs, rank, counter = state
    out = torch.empty(rows, routed.shape[1], dtype=routed.dtype, device=routed.device)
    torch.ops.v41sk.ll_reduce_scatter(
        out, routed.contiguous(), *ptrs, rank, _LL_DP_CAP, counter
    )
    return out


# VLLM_V41_EP_RS_COMBINE=1 (default off): batches the one-push combine does not
# take (prefill chunks, uneven DP sizes) fold the shared expert into this
# rank's own rows and finish with one EP16 reduce-scatter + all-gather
# instead of the DP reduce-scatterv (grouped ncclReduce) + TP all-reduce.
_EP_RS_COMBINE = envs.VLLM_V41_EP_RS_COMBINE == "1"


def ep_combine_rs(routed, shared):
    """This DP rank's finished MoE rows: routed FP32 partials of every
    gathered row (DP-rank order) summed over the 16 EP ranks, plus the TP
    shards of the shared expert (each TP rank adds its shard to its own rows
    before the sum). None when it does not apply."""
    from vllm.distributed import get_dp_group, get_ep_group
    from vllm.forward_context import get_forward_context

    if (
        not _EP_RS_COMBINE
        or not routed.is_cuda
        or routed.dtype != torch.float32
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    context = get_forward_context()
    with context.dp_metadata.sp_local_sizes(1):
        sizes = context.dp_metadata.get_chunk_sizes_across_dp_rank()
    if sizes is None or sum(sizes) != routed.shape[0]:
        return None
    me = get_dp_group().rank_in_group
    first, rows = sum(sizes[:me]), sizes[me]
    if tuple(shared.shape) != (rows, routed.shape[1]):
        return None
    group = get_ep_group()
    padded = -(-routed.shape[0] // group.world_size) * group.world_size
    if padded != routed.shape[0]:
        routed = torch.cat(
            [routed, routed.new_zeros(padded - routed.shape[0], routed.shape[1])]
        )
    routed[first : first + rows] += shared
    part = group.reduce_scatter(routed.contiguous(), dim=0)
    return group.all_gather(part, dim=0)[first : first + rows]


# EP16 combine in one push (v41sk.ll_ep_combine16).
# Per source rank one receive slot (2 parities x 16 sources x cap: 64 MiB per
# GPU), decode batches of up to 102 rows per DP rank. Served at 24 and 48 rows
# it beat the DP reduce-scatter + TP all-reduce pair (c16 69.5 vs 88.3 ms, c32
# 109.8 vs 119.2 ms per round) although the pair is faster in isolation: one
# wait for the slowest rank instead of two.
_LL_EP16_CAP = 4 * 1024 * 1024
_LL_SOLO_CAP = 128 * 1024
_LL_EP16: dict = {}


def _ll_ep16_state(solo=False):
    """(peer pointers [16] by DP group * 4 + TP rank, my index, counter) or None."""
    cap = _LL_SOLO_CAP if solo else _LL_EP16_CAP
    if solo not in _LL_EP16:
        import torch.distributed as dist

        from vllm import _custom_ops as ops
        from vllm.distributed import get_dp_group, get_ep_group, get_tp_group

        group = get_ep_group()
        tp, dp = get_tp_group(), get_dp_group()
        if group.world_size != 16 or tp.world_size != 4 or dp.world_size != 4:
            _LL_EP16[solo] = None
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        load_v41sk()
        size = 2 * 16 * cap
        pointer, handle = ops.allocate_shared_buffer_and_handle(size)
        torch.ops.v41sk.ll_arm(pointer, size)
        torch.accelerator.synchronize()
        me = dp.rank_in_group * 4 + tp.rank_in_group
        entries = [None] * 16
        dist.all_gather_object(entries, (me, handle), group=group.cpu_group)
        by_index = dict(entries)
        if sorted(by_index) != list(range(16)):
            raise RuntimeError("V4.1 EP16 combine: ranks are not DP x TP = 4 x 4")
        peers = torch.tensor(
            [
                pointer if i == me else ops.open_mem_handle(by_index[i])
                for i in range(16)
            ],
            dtype=torch.int64,
        )
        dist.barrier(group=group.cpu_group)
        counter = torch.zeros(16 if solo else _LL_DP_BLOCKS, dtype=torch.int32, device="cuda")
        _LL_EP16[solo] = (peers, me, counter)
        logger.info(
            "V4.1 native: EP16 one-push combine (%d KiB slots)", cap // 1024
        )
    return _LL_EP16[solo]


def dp_combine_shared(routed, shared, ids=None, num_experts=0, solo_sizes=None):
    """The finished MoE output of this rank's rows: the routed partials of all
    16 ranks plus the FP16 TP shards of the shared expert, summed in one push
    (reduce-scatter over DP, shared add and TP all-reduce at once); None when
    the batch does not fit (the caller takes the two-step path). ids: the
    gathered routes (physical ids of num_experts) for the sparse push."""
    if (
        not _LL_AR
        or not routed.is_cuda
        or routed.dtype != torch.float32
        or shared.dtype != torch.float16
    ):
        return None
    rows = routed.shape[0] // 4
    if (
        solo_sizes is None
        and rows * 4 == routed.shape[0]
        and rows * routed.shape[1] * 4 <= _LL_EP16_CAP
        and tuple(shared.shape) == (rows, routed.shape[1])
        and _dp_slots(rows, routed.shape[1] * 4) is not None
    ):
        state = _ll_ep16_state()
        if state is None:
            return None
        peers, me, counter = state
        out = torch.empty(
            rows, routed.shape[1], dtype=torch.float32, device=routed.device
        )
        # TP-sliced: each TP rank receives a quarter of the columns and shares its
        # summed slice with the group (same values, a quarter of the receive).
        sliced = (
            _COMBINE_TP
            and routed.shape[1] % 16 == 0
            and rows * routed.shape[1] <= _LL_EP16_CAP // 2
        )
        if sliced and _COMBINE_SPARSE and ids is not None and num_experts % 16 == 0:
            torch.ops.v41sk.ll_ep_combine16_tp_sparse(
                out,
                routed.contiguous(),
                shared.contiguous(),
                ids.int().contiguous(),
                peers,
                me,
                _LL_EP16_CAP,
                counter,
                num_experts // 16,
            )
            return out
        ops = torch.ops.v41sk
        op = ops.ll_ep_combine16_tp if sliced else ops.ll_ep_combine16
        op(
            out,
            routed.contiguous(),
            shared.contiguous(),
            peers,
            me,
            _LL_EP16_CAP,
            counter,
        )
        return out

    if not (_LL_CHUNK and _COMBINE_TP and routed.shape[1] % 16 == 0):
        return None
    from vllm.forward_context import get_forward_context

    if solo_sizes is None:
        metadata = get_forward_context().dp_metadata
        if metadata is None:
            return None
        sizes = metadata.num_tokens_across_dp_cpu.tolist()
    else:
        sizes = solo_sizes
    if len(sizes) != 4 or sum(sizes) != routed.shape[0]:
        return None
    state = _ll_ep16_state(solo=solo_sizes is not None)
    if state is None:
        return None
    peers, me, counter = state
    my_rows = sizes[me >> 2]
    if tuple(shared.shape) != (my_rows, routed.shape[1]):
        return None
    out = torch.empty(
        my_rows, routed.shape[1], dtype=torch.float32, device=routed.device
    )
    sparse_ids = (
        ids.int().contiguous()
        if (_COMBINE_SPARSE and ids is not None and num_experts % 16 == 0)
        else None
    )
    per_rank = num_experts // 16 if sparse_ids is not None else 1
    torch.ops.v41sk.ll_ep_combine16_tp_var(
        out,
        routed.contiguous(),
        shared.contiguous(),
        sparse_ids,
        peers,
        me,
        *sizes,
        _LL_SOLO_CAP if solo_sizes is not None else _LL_EP16_CAP,
        counter,
        per_rank,
    )
    return out


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
# Expert codes in TurboMind's B/Pack1 order (v41sk_tm.so reads them), the
# layout TurboMind's grouped MXFP4 GEMM consumes: one copy for both.
TM_CODES = envs.VLLM_V41_TM_CODES == "1"
# EP16 combine on ll_ep_combine16_tp (TP-sliced receive + slice exchange).
_COMBINE_TP = envs.VLLM_V41_COMBINE_TP == "1"
_COMBINE_SPARSE = envs.VLLM_V41_COMBINE_SPARSE == "1"


def v41sk_library() -> str:
    return envs.VLLM_V41SK_LIB or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        *("..",) * 3,
        "benchmarks",
        "deepseek_v41",
        "v41k",
        "build",
        "v41sk_tm.so" if TM_CODES else "v41sk.so",
    )


def qpn_to_tm_codes(codes: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """QPN expert codes ([N/32][K/16][lane][2 words]) -> TurboMind SM70 E2M1
    B/Pack1 words ([N/32][K/8][column]): lane l's words 2g, 2g + 1 move to
    column ((l >> 2) & 3) * 8 + (l & 3) + (4 if l & 16) of rows 2g, 2g + 1.
    A pure permutation of 32-bit words (tm_qpn_layout_check.py)."""
    words = codes.view(torch.int32).view(n // 32, k // 16, 32, 2)
    lane = torch.arange(32, device=codes.device)
    column = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    out = torch.empty(n // 32, k // 16, 2, 32, dtype=torch.int32, device=codes.device)
    out.index_copy_(3, column, words.transpose(2, 3))
    return out.view(-1).view(torch.uint8)


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
def _mask_rows_kernel(
    ids, real, rows, dead_id, TOPK: tl.constexpr, RB: tl.constexpr, KB: tl.constexpr
):
    row = tl.program_id(0) * RB + tl.arange(0, RB)
    col = tl.arange(0, KB)
    dead = ((row >= tl.load(real)) & (row < rows))[:, None] & (col < TOPK)[None, :]
    tl.store(ids + row[:, None] * TOPK + col[None, :], dead_id, dead)


@triton.jit
def _replica_routes_kernel(
    ids,
    copies,
    counts,
    total,
    logical,
    dead,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = slot < total
    expert = tl.load(ids + slot, inside, dead)
    live = inside & (expert >= 0) & (expert < logical)
    n = tl.load(counts + expert, live, 1)
    # A replicated expert's routes cycle over its copies by row.
    pick = (slot // TOPK) % n
    mapped = tl.load(copies + expert * WIDTH + pick, live, dead)
    tl.store(ids + slot, tl.where(live, mapped, dead), inside)


def map_replica_routes(ids, copies, counts, dead):
    """Gathered logical routes [rows, top-k] int32 -> physical slots in place
    (VLLM_V41_EXPERT_REPLICAS; copies [logical, width] physical slots, counts
    [logical] copies per expert); ids outside [0, logical) become ``dead``."""
    total = ids.numel()
    _replica_routes_kernel[(triton.cdiv(total, 1024),)](
        ids,
        copies,
        counts,
        total,
        copies.shape[0],
        dead,
        TOPK=ids.shape[1],
        WIDTH=copies.shape[1],
        BLOCK=1024,
        num_warps=4,
    )
    return ids


def mask_padded_routes(
    ids: torch.Tensor, real: torch.Tensor, dead: int
) -> torch.Tensor:
    """Set the routes of rows >= real[0] to `dead` (an id no rank owns) in
    place, for graph padding and dummy batches: expert parallelism then reads
    no weights for them."""
    rows, top_k = ids.shape
    if rows:
        _mask_rows_kernel[(triton.cdiv(rows, 64),)](
            ids, real, rows, dead, TOPK=top_k, RB=64, KB=triton.next_power_of_2(top_k)
        )
    return ids


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
def _ep_plan_kernel(
    ids,
    expert_map,
    local_ids,
    perm,
    experts,
    offsets,
    scratch,
    S,
    E,
    LOCAL: tl.constexpr,
    SB: tl.constexpr,
    LB: tl.constexpr,
):
    """Expert-parallel slot plan in one program: global ids -> this rank's
    expert ids (LOCAL for routes it does not own, and for ids outside [0, E)),
    then a counting sort over the LOCAL experts. Only owned routes enter perm;
    one group per hit expert; offsets past the groups hold the owned count, so
    the trailing groups are empty and read no weights. A slot's place inside
    its expert's group comes from an atomic counter: the order of rows within
    a group varies, each row's result does not."""
    bins = tl.arange(0, LB)
    tl.store(scratch + bins, 0)
    tl.debug_barrier()
    s = tl.arange(0, SB)
    live = s < S
    g = tl.load(ids + s, live, -1)
    ok = live & (g >= 0) & (g < E)
    owned = tl.load(expert_map + tl.where(ok, g, 0), ok, -1)
    local = tl.where(ok & (owned >= 0), owned, LOCAL)
    tl.store(local_ids + s, local, live)
    mine = live & (local < LOCAL)
    within = tl.atomic_add(scratch + tl.where(mine, local, 0), 1, mask=mine)
    tl.debug_barrier()
    counts = tl.load(scratch + bins, bins < LOCAL, 0, cache_modifier=".cg")
    starts = tl.cumsum(counts, 0) - counts
    has = counts > 0
    group = tl.cumsum(has.to(tl.int32), 0) - 1
    tl.store(experts + group, bins, has)
    tl.store(offsets + group, starts, has)
    groups = tl.sum(has.to(tl.int32), 0)
    total = tl.sum(counts, 0)
    tl.store(scratch + LB + bins, starts)
    tl.debug_barrier()
    start = tl.load(
        scratch + LB + tl.where(mine, local, 0), mine, 0, cache_modifier=".cg"
    )
    tl.store(perm + start + within, s, mine)
    tl.store(offsets + s, total, (s >= groups) & (s <= S))
    tl.store(experts + s, 0, (s >= groups) & live)


def _ep_plan(topk_ids, expert_map, local_experts):
    """(local ids [S] with DEAD = local_experts, perm, experts, offsets)."""
    flat = topk_ids.reshape(-1)
    slots = flat.numel()
    device = flat.device
    local = torch.empty(slots, dtype=torch.int32, device=device)
    # perm past the owned count is never read (the groups end there).
    perm = torch.empty(slots, dtype=torch.int32, device=device)
    experts = torch.empty(slots, dtype=torch.int32, device=device)
    offsets = torch.empty(slots + 1, dtype=torch.int32, device=device)
    if slots + 1 <= _EP_PLAN_MAX_SLOTS:
        bins = triton.next_power_of_2(local_experts + 1)
        scratch = torch.empty(2 * bins, dtype=torch.int32, device=device)
        _ep_plan_kernel[(1,)](
            flat.to(torch.int32),
            expert_map,
            local,
            perm,
            experts,
            offsets,
            scratch,
            slots,
            expert_map.shape[0],
            LOCAL=local_experts,
            SB=triton.next_power_of_2(slots + 1),
            LB=bins,
            num_warps=8,
        )
        return local, perm, experts, offsets
    # Longer prefill chunks: the generic plan, then the non-owned group (sorted
    # last) is emptied by ending it where it starts.
    total = expert_map.shape[0]
    flat = flat.long()
    dead = (flat < 0) | (flat >= total)
    owned = torch.where(dead, -1, expert_map[flat.clamp(0, total - 1)])
    local = torch.where(owned < 0, local_experts, owned).to(torch.int32)
    perm, experts, offsets = _route_plan(local)
    last = experts == local_experts
    offsets = torch.minimum(offsets, (owned >= 0).sum(dtype=torch.int32))
    return local, perm, experts.masked_fill(last, 0), offsets


# Gathered rows per piece of an eager expert-parallel MoE (prefill chunks).
# At EP4 x TP4 the W2 route buffer alone takes 120 MiB for 2048 rows.
# Bound eager workspace during mixed prefill while preserving the opt-in
# grouped prefill implementation, which accumulates W2 into the output.
_EP_CHUNK_ROWS = (
    1024
    if envs.VLLM_V41_EP4_TP4 == "1" and envs.VLLM_V41_MOE_PF_GEMM != "1"
    else 2048
)
# VLLM_V41_MOE_PF_GEMM=1 (default off): prefill chunks run this rank's experts
# as two grouped WMMA GEMMs (v41sk.moe_pf_gemm) over the owned slots only,
# sorted by expert: each expert's weights are read once per 64 of its rows.
_MOE_PF_GEMM = envs.VLLM_V41_MOE_PF_GEMM == "1"
_MOE_PF_COMPACT = os.environ.get("VLLM_V41_MOE_PF_COMPACT", "1") == "1"
_PF_DIRECT = (envs.VLLM_V41_EP4_TP4 == "1"
              and os.environ.get("VLLM_V41_MOE_PF_DIRECT") == "1")
_PF_W13_CTAS = 480 if _PF_DIRECT else 240
_PF_W2_CTAS = 480 if _PF_DIRECT else 240
_MOE_PF_BM = int(os.environ.get("VLLM_V41_MOE_PF_BM", "32"))
# Owned rows up to which an expert stays on the skinny kernel (one pass).
_PF_SMALL = 32


@triton.jit
def _swiglu_owned_kernel(
    gate_up, probs, ids, out, DEAD: tl.constexpr, INTER: tl.constexpr, BLOCK: tl.constexpr
):
    """_swiglu_prob_kernel over slot-major rows, skipping slots this rank does
    not own (their rows are never read)."""
    slot = tl.program_id(0)
    if tl.load(ids + slot) != DEAD:
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = columns < INTER
        gate = tl.load(gate_up + slot * 2 * INTER + columns, live, 0).to(tl.float32)
        up = tl.load(gate_up + slot * 2 * INTER + INTER + columns, live, 0).to(
            tl.float32
        )
        gate = tl.minimum(gate, 10.0)
        up = tl.minimum(tl.maximum(up, -10.0), 10.0)
        value = gate / (1.0 + tl.exp(-gate)) * up * tl.load(probs + slot)
        tl.store(out + slot * INTER + columns, value.to(tl.float16), live)


def _local_experts_pf(layer, x, topk_weights, topk_ids, expert_map, out=None):
    """_apply_local_experts for prefill chunks: one expert-parallel plan;
    groups with at most _PF_SMALL rows run on the skinny kernel (one pass),
    the hot ones on v41sk.moe_pf_gemm (their rows spread over CTAs), each
    kernel skipping the other's groups. The same decoded weights, SwiGLU
    clip, route weight before the FP16 down input and per-token FP32 route
    sum; only the GEMMs' summation order differs. No host sync."""
    tokens, top_k = topk_ids.shape
    slots = tokens * top_k
    experts = layer.w13_weight.shape[0]
    ids, perm, groups, offsets = _ep_plan(topk_ids, expert_map, experts)
    hidden, inter = x.shape[1], layer.w2_weight.shape[2] * 2
    if _MOE_PF_COMPACT and slots + 1 <= _EP_PLAN_MAX_SLOTS:
        n_groups = min(experts + 1, slots)
        empty_i32 = perm[:0]
        gate_up = torch.empty((slots, 2 * inter), dtype=x.dtype, device=x.device)
        torch.ops.v41sk.moe_pf_gemm(
            gate_up,
            x,
            perm,
            top_k,
            empty_i32,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.v41_g13,
            offsets,
            groups,
            n_groups,
            max(4, _PF_W13_CTAS // (2 * inter // 128)),
            0,
            _MOE_PF_BM,
        )
        activated = torch.empty((slots, inter), dtype=x.dtype, device=x.device)
        torch.ops.v41sk.moe_pf_swiglu(
            activated,
            gate_up,
            topk_weights.reshape(-1).float().contiguous(),
            perm,
            offsets,
            n_groups,
        )
        del gate_up
        if out is None:
            routed = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
        else:
            routed = out.zero_()
        torch.ops.v41sk.moe_pf_gemm_accum(
            routed,
            activated,
            perm,
            top_k,
            layer.w2_weight,
            layer.w2_weight_scale,
            layer.v41_g2,
            offsets,
            groups,
            n_groups,
            max(8, _PF_W2_CTAS // (hidden // 128)),
            _MOE_PF_BM,
        )
        return routed
    gate_up = torch.empty((slots, 2 * inter), dtype=x.dtype, device=x.device)
    torch.ops.v41sk.skinny_moe_qpn_sm70(
        x,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.v41_g13,
        perm,
        groups,
        offsets,
        top_k,
        gate_up,
        False,
        tokens,
        layer.v41_split13,
        1,
        1,
        _PF_SMALL,
    )
    # GEMM CTAs stride over the row tiles: ~2 per SM per launch.
    torch.ops.v41sk.moe_pf_gemm(
        gate_up,
        x,
        perm,
        top_k,
        perm,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.v41_g13,
        offsets,
        groups,
        min(experts + 1, slots),
        max(2, 320 // (2 * inter // 128)),
        _PF_SMALL,
    )
    activated = torch.empty((slots, inter), dtype=x.dtype, device=x.device)
    _swiglu_owned_kernel[(slots, triton.cdiv(inter, 256))](
        gate_up,
        topk_weights.reshape(-1).float().contiguous(),
        ids,
        activated,
        DEAD=experts,
        INTER=inter,
        BLOCK=256,
        num_warps=4,
    )
    del gate_up
    down = torch.empty((slots, hidden), dtype=x.dtype, device=x.device)
    torch.ops.v41sk.skinny_moe_qpn_sm70(
        activated,
        layer.w2_weight,
        layer.w2_weight_scale,
        layer.v41_g2,
        perm,
        groups,
        offsets,
        top_k,
        down,
        True,
        tokens,
        9 if (inter // 16) % 9 == 0 else layer.v41_split2,
        1,
        1,
        _PF_SMALL,
    )
    torch.ops.v41sk.moe_pf_gemm(
        down,
        activated,
        perm,
        1,
        perm,
        layer.w2_weight,
        layer.w2_weight_scale,
        layer.v41_g2,
        offsets,
        groups,
        min(experts + 1, slots),
        max(2, 320 // (hidden // 128)),
        _PF_SMALL,
    )
    routed = torch.empty((tokens, hidden), dtype=torch.float32, device=x.device)
    _local_route_sum_kernel[(tokens, triton.cdiv(hidden, 1024))](
        down,
        ids,
        routed,
        DEAD=experts,
        TOPK=top_k,
        H=hidden,
        BLOCK=1024,
        num_warps=4,
    )
    return routed




# Gathered rows (all DP ranks) up to which the expert-parallel experts take the
# 8-row decode kernels: they win up to ~12 rows per expert (12 experts x 6
# rows: 185 vs 258 us for w13), i.e. every decode batch up to c64; prefill
# chunks take the row-block split-K kernel (24+ rows per expert).
_EP_FEW_ROWS = 1024
# Gathered rows from which decode batches take the 16-row-block kernels (four
# requests per DP rank and up: hot experts pass 8 rows).
_EP_PF_ROWS = 96
# Row blocks per weight read of the batch kernel (8 rows each; V41SK_RB_MAX
# bounds it): a hot expert's rows beyond RB * 8 re-read its weights.
_EP_PF_RB = 2

# One-program EP plans up to this many slots (+1): decode batches of every size
# and prefill chunks of up to ~1.3K tokens per DP rank.
_EP_PLAN_MAX_SLOTS = 32768


@triton.jit
def _local_route_sum_kernel(
    slots,
    ids,
    out,
    DEAD: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """_route_sum_kernel for expert parallel: routes to experts not on this
    rank (id DEAD) were not computed and are skipped (zero contribution)."""
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), tl.float32)
    for k in range(TOPK):
        live = tl.load(ids + row * TOPK + k) != DEAD
        total += tl.load(
            slots + (row * TOPK + k) * H + columns, mask=live, other=0.0
        ).to(tl.float32)
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
        # Target 384 x top-6 (+ 16 replica slots, VLLM_V41_EXPERT_REPLICAS)
        # and the DSpark drafter's 128 x top-3.
        if (moe.num_experts, moe.experts_per_token) not in (
            (384, 6),
            (400, 6),
            (128, 3),
        ) or (
            moe.hidden_dim,
            moe.intermediate_size_per_partition,
        ) != (5120, 2304 // max(moe.tp_size, 1)):
            raise NotImplementedError("V4.1 native experts: 384 x 6 or 128 x 3")
        parallel = moe.moe_parallel_config
        if parallel.use_all2all_kernels and not parallel.use_ag_rs_all2all_kernels:
            # DP+EP runs on the runner's all-gather / reduce-scatter dispatch
            # (each rank computes its own experts for every gathered token).
            raise NotImplementedError(
                "V4.1 native experts: DP+EP needs all2all_backend="
                "allgather_reducescatter"
            )

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
                if TM_CODES:
                    n, k = codes.shape[1], codes.shape[2] * 2
                    packed_codes = qpn_to_tm_codes(packed_codes, n, k)
                codes[expert].view(-1).copy_(packed_codes)
                scales[expert].view(-1).copy_(packed_scales)
        layer.v41_split13 = _splitk(w13.shape[2] * 2)
        layer.v41_split2 = _splitk(w2.shape[2] * 2)
        logger.info_once(
            "V4.1 native experts: %d MXFP4 experts in %s fragment order, "
            "skinny split-K %d/%d (v41sk)",
            w13.shape[0],
            "TurboMind B/Pack1" if TM_CODES else "QPN",
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
        expert_map = getattr(layer, "expert_map", None)
        if expert_map is not None:
            if tokens > _EP_CHUNK_ROWS and not torch.cuda.is_current_stream_capturing():
                if (
                    _PF_DIRECT and _MOE_PF_GEMM and _MOE_PF_COMPACT
                    and _EP_CHUNK_ROWS * top_k + 1 <= _EP_PLAN_MAX_SLOTS
                ):
                    # Accumulate W2 directly into each piece of the final result.
                    # A short tail takes the same small-row kernel as a single piece.
                    output = x.new_empty((tokens, x.shape[1]), dtype=torch.float32)
                    for start in range(0, tokens, _EP_CHUNK_ROWS):
                        end = min(start + _EP_CHUNK_ROWS, tokens)
                        part = slice(start, end)
                        if end - start > _EP_FEW_ROWS:
                            _local_experts_pf(layer, x[part], topk_weights[part],
                                              topk_ids[part], expert_map, out=output[part])
                        else:
                            output[part].copy_(self._apply_local_experts(
                                layer, x[part], topk_weights[part], topk_ids[part], expert_map
                            ))
                    return output
                # Prefill: the per-slot buffers (gate/up, activations, down)
                # span every gathered route, ~1.4 GB at 4 x 2048 rows though a
                # rank owns 1/16 of them. Bounded pieces of rows (one DP
                # rank's chunk each); a piece of idle ranks' padding rows plans
                # no groups and reads no weights.
                return torch.cat(
                    [
                        self._apply_local_experts(
                            layer,
                            x[a : a + _EP_CHUNK_ROWS],
                            topk_weights[a : a + _EP_CHUNK_ROWS],
                            topk_ids[a : a + _EP_CHUNK_ROWS],
                            expert_map,
                        )
                        for a in range(0, tokens, _EP_CHUNK_ROWS)
                    ]
                )
            return self._apply_local_experts(
                layer, x, topk_weights, topk_ids, expert_map
            )
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

    def _apply_local_experts(self, layer, x, topk_weights, topk_ids, expert_map):
        """Expert parallel: this rank's experts only, for every (gathered)
        token. Routes to other ranks' experts (and padded rows) contribute
        zero; the runner's reduce-scatter and the model's TP all-reduce sum
        the ranks. The shared expert is not folded here: it belongs to this
        rank's own tokens, before the dispatch."""
        tokens, top_k = topk_ids.shape
        if (
            _MOE_PF_GEMM
            and tokens > _EP_FEW_ROWS
            and not torch.cuda.is_current_stream_capturing()
        ):
            return _local_experts_pf(layer, x, topk_weights, topk_ids, expert_map)
        local_experts = layer.w13_weight.shape[0]
        # Padded rows carry an id outside [0, E) (mask_padded_routes); routes to
        # other ranks' experts take id local_experts and are not computed.
        ids, perm, experts, offsets = _ep_plan(topk_ids, expert_map, local_experts)
        slots = tokens * top_k
        w13, w2 = layer.w13_weight, layer.w2_weight
        inter = w2.shape[2] * 2
        activated = torch.empty((slots, inter), dtype=x.dtype, device=x.device)
        probabilities = topk_weights.reshape(-1).float().contiguous()
        # Decode batches: the fused 8-row kernel; prefill chunks (many rows per
        # expert) take the split-K kernel, which reuses one weight read for up
        # to V41SK_RB_MAX x 8 rows (24 rows/expert, 10 experts: 376 vs 458 us).
        few_rows = tokens <= _EP_FEW_ROWS
        # Larger decode batches: hot experts see more than 8 rows; the row-block
        # kernel reads their weights once per 16 rows (12 experts x 12 rows:
        # w13 241 vs 362 us, w2 131 vs 195 us; at 6 rows it loses 234 vs 191).
        row_blocks = few_rows and tokens >= _EP_PF_ROWS and inter % 32 == 0
        # Batch shape here is the gathered row count across all four DP groups.
        # The 576-column EP4 shards hit fewer rows per expert than full experts.
        from .norm import short_rows

        ep4_shards = (
            envs.VLLM_V41_EP4_TP4 == "1" and TM_CODES and _W13_SWIGLU
            and top_k == 6 and local_experts == 100
            and inter == 576 and x.shape[1] == 5120
        )
        tuned_batch = ep4_shards and short_rows(tokens)
        w13_blocks = row_blocks and not (tuned_batch and tokens in (96, 144))
        w2_narrow = tuned_batch and tokens in (24, 96, 144, 192)
        if w13_blocks:
            gate_up = torch.empty((slots, w13.shape[1]), dtype=x.dtype, device=x.device)
            torch.ops.v41sk.skinny_moe_pf_sm70(
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
                4,
                _EP_PF_RB,
            )
            _swiglu_prob_kernel[(slots, triton.cdiv(inter, 256))](
                gate_up,
                probabilities,
                activated,
                INTER=inter,
                BLOCK=256,
                num_warps=4,
            )
            del gate_up
        elif ep4_shards and (tokens == 6 or (tuned_batch and tokens == 24)):
            # Four warps, eight K groups per chunk for the smallest batches.
            gate_up = x.new_empty((slots, 2 * inter))
            torch.ops.v41sk.skinny_moe_dec_sm70(
                x, w13, layer.w13_weight_scale, layer.v41_g13,
                perm, experts, offsets, top_k, gate_up, False, 4, 8,
            )
            _swiglu_prob_kernel[(slots, triton.cdiv(inter, 256))](
                gate_up, probabilities, activated, INTER=inter, BLOCK=256,
                num_warps=4,
            )
            del gate_up
        elif few_rows and _W13_SWIGLU and x.shape[1] == 5120 and inter % 32 == 0:
            # Gate/up projections, clipped SwiGLU and route weight in one
            # launch (row blocks of 8 per expert; only owned routes are planned).
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
                2 if inter == 576 and tokens <= 24 else 8,
            )
        else:
            gate_up = torch.empty((slots, w13.shape[1]), dtype=x.dtype, device=x.device)
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
            del gate_up
        down = torch.empty((slots, w2.shape[1]), dtype=x.dtype, device=x.device)
        if row_blocks and not w2_narrow and (inter // 16) % 4 == 0:
            torch.ops.v41sk.skinny_moe_pf_sm70(
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
                4,
                4,
                _EP_PF_RB,
            )
        elif few_rows and inter % 32 == 0 and (inter // 16) % 4 == 0:
            # Full-width experts (K 2304): the double-buffered decode kernel,
            # 4 warps x 4-group chunks (12-49 us at 1-6 hit experts on V100,
            # 5-7 us under the split-K skinny kernel).
            torch.ops.v41sk.skinny_moe_dec_sm70(
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
                4,
                2 if w2_narrow else 4,
            )
        else:
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
                9 if (inter // 16) % 9 == 0 else layer.v41_split2,
                1,
                1,
            )
        hidden = w2.shape[1]
        routed = torch.empty((tokens, hidden), dtype=torch.float32, device=x.device)
        _local_route_sum_kernel[(tokens, triton.cdiv(hidden, 1024))](
            down,
            ids,
            routed,
            DEAD=local_experts,
            TOPK=top_k,
            H=hidden,
            BLOCK=1024,
            num_warps=4,
        )
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
            # Physical slots of this expert (VLLM_V41_EXPERT_REPLICAS: primary
            # and copy); a slot off this rank is skipped by the loader.
            ffn = language.layers[layer].ffn
            targets = getattr(ffn, "_physical_targets", None)
            for physical in (
                targets[int(expert_id)] if targets is not None else [int(expert_id)]
            ):
                parameter.weight_loader(
                    parameter,
                    value.view(torch.uint8),
                    mapped,
                    shard_id=shard,
                    expert_id=physical,
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


# Per-request integers stay unspecialized: a new max_tokens or prompt length
# must not compile a variant while serving.
@triton.jit(do_not_specialize=["prompt_tokens", "max_tokens", "L", "NS"])
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
