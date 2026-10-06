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

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

HD, KD, LORA, INJ = 2560, 10240, 320, 4
HCX_MAX_M = 8
AR_BYTES = 4 * 80 * 256 * 8
LORA_BYTES = 8 * 336 * 8
HB_BYTES = 8 * HD * 4
TOP1_ROWS = 32
TOP1_BYTES = 2 * 2 * TOP1_ROWS * 16
_LANE_R = [(L & 3) + (4 if L & 16 else 0) for L in range(32)]
_LANE_Q = [(L >> 2) & 3 for L in range(32)]


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
            self.bar = torch.zeros(2, device=device, dtype=torch.int32)
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
        torch.ops._C.sm70_hcx_out(
            partial if ox is not None else partial.contiguous(),
            None,
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
        )
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
