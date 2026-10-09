# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch TP4 HC chain for Qwen3.8 Flash-Next verification on SM70.

The block output reaching an HC ``combine_and_mix`` is left as a TP partial
(the producing projection skips its all-reduce). For M <= 8 one kernel does the
all-reduce, combine, grouped RMSNorm, HC down/SiLU and HC up/gate-mix
(``sm70_hcx_out``); larger batches all-reduce explicitly and run the existing
chain.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

HD, KD, LORA, INJ = 2560, 10240, 320, 4
HCX_MAX_M = 8
_OUTPUT_PROJECTIONS: dict[str, tuple] = {}
_MOE_RUNNERS: dict[str, Any] = {}
AR_BYTES = 4 * 80 * 256 * 8
LORA_BYTES = 8 * 336 * 8
HB_BYTES = 8 * HD * 4
TOP1_ROWS = 32
TOP1_BYTES = 2 * 2 * TOP1_ROWS * 16
_LANE_R = [(L & 3) + (4 if L & 16 else 0) for L in range(32)]
_LANE_Q = [(L >> 2) & 3 for L in range(32)]


def register_output_projection(name, layer, defer):
    _OUTPUT_PROJECTIONS[name] = (layer, defer)


def register_moe_runner(name, runner):
    _MOE_RUNNERS[name] = runner


def _output_projection(x: torch.Tensor, name: str) -> torch.Tensor:
    # Resolve actual M inside the opaque op, rather than freezing a decode
    # choice while compiling a range that also includes C4 and prefill.
    layer, defer = _OUTPUT_PROJECTIONS[name]
    if 0 < x.shape[0] <= HCX_MAX_M and defer:
        return torch.nn.functional.pad(x, (0, layer.output_size - x.shape[1]))
    projected, _ = layer(x)
    if x.shape[0] > HCX_MAX_M:
        from vllm.distributed import tensor_model_parallel_all_reduce

        projected = tensor_model_parallel_all_reduce(projected)
    return projected


def _output_projection_fake(x: torch.Tensor, name: str) -> torch.Tensor:
    layer, _ = _OUTPUT_PROJECTIONS[name]
    return x.new_empty((x.shape[0], layer.output_size))


def _moe_output(
    shared: torch.Tensor | None, fused: torch.Tensor, name: str, trunc_size: int
) -> torch.Tensor:
    rows = fused.shape[0]
    if 0 < rows <= HCX_MAX_M:
        # An owned payload makes both contributions explicit to the compiler
        # and graph allocator. Hidden references to producer temporaries are
        # unsafe: the memory planner cannot see those additional consumers.
        first = fused[..., :trunc_size]
        second = torch.zeros_like(first) if shared is None else shared[..., :trunc_size]
        return torch.cat((first, second), dim=0)
    runner = _MOE_RUNNERS[name]
    reduced = runner._maybe_sm70_moe_sum2_allreduce(shared, fused, trunc_size)
    if reduced is None:
        summed = fused if shared is None else shared + fused
        reduced = runner._maybe_reduce_final_output(summed, trunc_size)
    # Large batches retain the original reduction. The unused second plane
    # needs no initialization; the consumer reads only the first plane.
    payload = fused.new_empty((2 * rows, trunc_size))
    payload[:rows].copy_(reduced)
    return payload


def _moe_output_fake(shared, fused, name, trunc_size):
    return fused.new_empty((2 * fused.shape[0], trunc_size))


direct_register_custom_op(
    "qwen38_sm70_hcx_output_projection",
    _output_projection,
    fake_impl=_output_projection_fake,
)
direct_register_custom_op(
    "qwen38_sm70_hcx_moe_output",
    _moe_output,
    fake_impl=_moe_output_fake,
)


def pack_down(w: torch.Tensor, rank: int) -> torch.Tensor:
    """[336, 10240] FP16 -> [80 cta, 6 warp, 4 kstep, 2, 32 lane, 8]."""
    assert w.shape == (336, KD) and w.dtype == torch.float16
    dev = w.device
    lane_r = torch.tensor(_LANE_R, device=dev)
    lane_q = torch.tensor(_LANE_Q, device=dev)
    n = torch.arange(96, device=dev)
    g = torch.where(n < 80, rank * 80 + n, 320 + n - 80)
    ok = (n < 80) | ((rank == 3) & (n < 84))
    cols = torch.where(
        ok[:, None], w[g.clamp(max=335)], torch.zeros((), dtype=w.dtype, device=dev)
    )
    i = torch.arange(80, device=dev).view(80, 1, 1, 1, 1, 1)
    wp = torch.arange(6, device=dev).view(1, 6, 1, 1, 1, 1)
    s = torch.arange(4, device=dev).view(1, 1, 4, 1, 1, 1)
    hl = torch.arange(2, device=dev).view(1, 1, 1, 2, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 1, 1, 32, 1)
    j = torch.arange(8, device=dev).view(1, 1, 1, 1, 1, 8)
    tile, kh = wp >> 1, wp & 1
    nn = tile * 32 + (lane_q[lane] * 8 + lane_r[lane])
    kl = (kh * 4 + s) * 16 + hl * 8 + j
    k = (kl // 32) * HD + 32 * i + kl % 32
    return cols[nn, k].contiguous()


def pack_up(w: torch.Tensor, rank: int) -> torch.Tensor:
    """[10240, 320] FP16 -> [80 cta, 5 warp, 4 kstep, 2, 32 lane, 8]."""
    assert w.shape == (KD, LORA) and w.dtype == torch.float16
    dev = w.device
    lane_r = torch.tensor(_LANE_R, device=dev)
    lane_q = torch.tensor(_LANE_Q, device=dev)
    i = torch.arange(80, device=dev).view(80, 1, 1, 1, 1, 1)
    uw = torch.arange(5, device=dev).view(1, 5, 1, 1, 1, 1)
    s = torch.arange(4, device=dev).view(1, 1, 4, 1, 1, 1)
    hl = torch.arange(2, device=dev).view(1, 1, 1, 2, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 1, 1, 32, 1)
    j = torch.arange(8, device=dev).view(1, 1, 1, 1, 1, 8)
    row = lane_q[lane] * HD + 640 * rank + 8 * i + lane_r[lane]
    k = (uw * 4 + s) * 16 + hl * 8 + j
    return w[row, k].contiguous()


_OPROJ_TYPES = {8: 4, 12: 0, 13: 1, 14: 2}  # Q8_0, Q4_K, Q5_K, Q6_K -> dmv13


def pack_output_projection(layer) -> tuple | None:
    """dmv13 planes of a row-parallel GGUF o-proj shard ([2560, K] raw rows)."""
    import numpy as np

    from vllm.model_executor.layers.quantization import gguf_dmv13_dense as dense

    qweight = getattr(layer, "qweight", None)
    qtype = getattr(getattr(layer, "qweight_type", None), "weight_type", None)
    if qweight is None or qtype not in _OPROJ_TYPES:
        return None
    raw = qweight.detach()
    if raw.dtype != torch.uint8 or raw.ndim != 2 or raw.shape[0] != HD:
        return None
    fmt, q, s, m, gs = dense.decode(raw.cpu().numpy(), qtype)
    if q.shape[1] % 128 or fmt != _OPROJ_TYPES[qtype]:
        return None
    layout = getattr(getattr(layer, "quant_method", None), "layout", None)
    if layout is not None:
        # The GEMV consumes input_to_gguf(x); fold that reorder into the columns
        # so the fused kernel can read the vLLM-ordered activation directly.
        order = (
            layout.input_to_gguf(torch.arange(q.shape[1], dtype=torch.float32)[None])[0]
            .round()
            .long()
            .numpy()
        )
        cols = np.argsort(order)
        if (cols.reshape(-1, gs) % gs != np.arange(gs)).any():
            return None
        q = q[:, cols]
        groups = cols.reshape(-1, gs)[:, 0] // gs
        s = s[:, groups]
        if m is not None:
            m = m[:, groups]
    codes, high, scale = dense.pack(fmt, q, s, m, gs)
    dev = raw.device
    return (
        q.shape[1],
        torch.from_numpy(np.ascontiguousarray(codes)).to(dev),
        torch.from_numpy(np.ascontiguousarray(high)).to(dev),
        torch.from_numpy(np.ascontiguousarray(scale)).to(dev),
        fmt,
    )


class Sm70HcxRuntime:
    """Peer buffers and scratch shared by every HCX call of one process."""

    def __init__(self, group, device: torch.device):
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.distributed.device_communicators.sm70_ring import find_peer_order

        self.reason: str | None = None
        self.top1_enabled = False
        self.diagnostic = False
        self.local_schedule = False
        self.snapshots: dict[str, dict[str, torch.Tensor]] = {}
        self.group = group
        self.rank = dist.get_rank(group)
        if dist.get_world_size(group) != 4:
            self.reason = "requires_tp4"
            return
        if not hasattr(torch.ops._C, "sm70_hcx_out"):
            self.reason = "operator_missing"
            return
        import pynvml

        local = device.index
        uuids = [""] * 4
        dist.all_gather_object(
            uuids, current_platform.get_device_uuid(local), group=group
        )
        visible = {
            current_platform.get_device_uuid(i): i
            for i in range(torch.accelerator.device_count())
        }
        pynvml.nvmlInit()
        try:
            handles = [pynvml.nvmlDeviceGetHandleByUUID(u) for u in uuids]
            direct = [
                [
                    i == j
                    or pynvml.nvmlDeviceGetP2PStatus(
                        handles[i], handles[j], pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    == pynvml.NVML_P2P_STATUS_OK
                    for j in range(4)
                ]
                for i in range(4)
            ]
        finally:
            pynvml.nvmlShutdown()
        self.full = all(direct[i][j] for i in range(4) for j in range(4))
        order = (0, 1, 2, 3) if self.full else find_peer_order(direct)
        if order is None:
            self.reason = "requires_two_direct_nvlink_peers_per_rank"
            return
        self.logical_rank = order.index(self.rank)
        if self.full:
            peers = {0, 1, 2, 3}
        else:
            peers = {
                order[self.logical_rank ^ 1],
                order[self.logical_rank ^ 2],
                self.rank,
            }
        ok = all(
            torch.ops._C.sm70_ring_native_peer_atomics(local, visible[uuids[p]])
            for p in peers
            if p != self.rank
        )
        admitted = [False] * 4
        dist.all_gather_object(admitted, ok, group=group)
        if not all(admitted):
            self.reason = "direct_cuda_peer_access_unavailable"
            return
        total = AR_BYTES + LORA_BYTES + HB_BYTES + TOP1_BYTES
        with torch.accelerator.device_index(local):
            pointers = CustomAllreduce.create_shared_buffer(
                total, group=group, peer_ranks=peers
            )
            base = [pointers[r] for r in order]
            self.ar = [p if p else 0 for p in base]
            self.lora = [p + AR_BYTES if p else 0 for p in base]
            self.hb = [p + AR_BYTES + LORA_BYTES if p else 0 for p in base]
            self.top1_buffers = [
                p + AR_BYTES + LORA_BYTES + HB_BYTES if p else 0 for p in base
            ]
            self.top1_seq = torch.zeros(1, device=device, dtype=torch.int32)
            self.xn = torch.zeros(HCX_MAX_M, KD, device=device, dtype=torch.float16)
            self.sq = torch.zeros(80 * 8 * 4, device=device, dtype=torch.float32)
            self.dpart = torch.zeros(
                80 * 8 * 96 * 4, device=device, dtype=torch.float32
            )
            # Legacy arrival/reset counters and separate local-schedule
            # cumulative counter/base permit alternating graph policies.
            self.bar = torch.zeros(4, device=device, dtype=torch.int32)
            self.seq = torch.zeros(1, device=device, dtype=torch.int32)
            torch.accelerator.synchronize()
        dist.barrier(group=group)
        logger.info_once(
            "SM70 HCX enabled (logical rank order %s, full mesh=%s).",
            tuple(order),
            self.full,
        )

    @property
    def enabled(self) -> bool:
        return self.reason is None

    def top1(self, pairs: torch.Tensor) -> torch.Tensor | None:
        """Global argmax ids from TP-local FP32 (value, id) pairs, or None."""
        if not self.enabled or not 1 <= pairs.shape[0] <= TOP1_ROWS:
            return None
        out = torch.empty(pairs.shape[0], dtype=torch.int64, device=pairs.device)
        torch.ops._C.sm70_top1x_out(
            out,
            pairs.float().contiguous(),
            self.top1_buffers,
            self.top1_seq,
            self.logical_rank,
        )
        return out

    def run(
        self,
        partial,
        hidden,
        injection,
        norm_weight,
        eps,
        packed_down,
        packed_up,
        oproj=None,
        secondary=None,
        snapshot_name=None,
    ):
        m = partial.shape[0]
        ox = ocodes = ohigh = oscale = None
        ofmt = -1
        if oproj is not None:
            k, ocodes, ohigh, oscale, ofmt = oproj
            # hcxo computes the producer GEMV itself and never reads p0.
            ox = partial[:, :k]
        hidden_out = torch.empty_like(hidden)
        block = partial.new_empty((m, HD))
        injection_out = partial.new_empty((m, INJ))
        snapshot = None
        if self.diagnostic and m == 5 and snapshot_name is not None:
            if oproj is not None:
                raise RuntimeError("HCX diagnosis requires separate output projections")
            inputs = {"partial": partial, "hidden": hidden, "injection": injection}
            if secondary is not None:
                inputs["secondary"] = secondary
            snapshot = self.snapshots.setdefault(snapshot_name, {})
            for key, value in inputs.items():
                if key not in snapshot:
                    snapshot[key] = torch.empty_like(value)
                snapshot[key].copy_(value)
            if "epoch" not in snapshot:
                snapshot["epoch"] = torch.empty_like(self.seq)
            snapshot["epoch"].copy_(self.seq)
        torch.ops._C.sm70_hcx_out(
            partial if ox is not None else partial.contiguous(),
            secondary,
            hidden.contiguous(),
            injection.contiguous(),
            norm_weight,
            eps,
            packed_down,
            packed_up,
            hidden_out,
            block,
            injection_out,
            self.xn,
            self.sq,
            self.dpart,
            self.bar,
            self.seq,
            self.ar,
            self.lora,
            self.hb,
            self.logical_rank,
            None,
            int(self.full),
            ox,
            ocodes,
            ohigh,
            oscale,
            ofmt,
            None,
            None,
            1e-6,
            None,
            self.local_schedule,
        )
        if snapshot is not None:
            for key, value in (
                ("hidden_out", hidden_out),
                ("block_out", block),
                ("injection_out", injection_out),
            ):
                if key not in snapshot:
                    snapshot[key] = torch.empty_like(value)
                snapshot[key].copy_(value)
        return hidden_out, block, injection_out


_RUNTIME: Sm70HcxRuntime | None = None


def current_hcx_runtime() -> Sm70HcxRuntime | None:
    return _RUNTIME


def get_hcx_runtime(device: torch.device) -> Sm70HcxRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        from vllm.distributed import get_tp_group

        _RUNTIME = Sm70HcxRuntime(get_tp_group().cpu_group, device)
    return _RUNTIME


def partial_moe_output(name, shared, routed, hidden_dim):
    return torch.ops.vllm.qwen38_sm70_hcx_moe_output(shared, routed, name, hidden_dim)


def partial_projection(name, hidden):
    return torch.ops.vllm.qwen38_sm70_hcx_output_projection(hidden, name)


def maybe_top1_exchange(local_pair):
    runtime = current_hcx_runtime()
    if runtime is None or not getattr(runtime, "top1_enabled", False):
        return None
    tokens = runtime.top1(local_pair)
    if tokens is not None:
        logger.info_once("SM70 two-hop top1 exchange enabled.")
    return tokens
