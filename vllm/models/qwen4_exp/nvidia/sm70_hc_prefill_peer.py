# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded TP4 peer reduction fused with row-owned HC residual/RMSNorm.

Existing CUDA IPC allocation and NCCL ordering are reused. Full-mesh SM70
prefill rows read the four physical TP peers and accumulate in FP32. MoE
local sums use scaled FP16 only if every element's reconstruction error is
at most 2**-14; other rows retain FP32 sums. The four-rank transport error
before the original FP16 boundary is therefore at most 2**-12, apart from
FP32 addition association. Attention partials retain their original FP16
storage. The normal sharded HC stages handle all unqualified inputs.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.runtime_resources import current_runtime_resources, runtime_resources_for
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_MAX_ROWS = 8192
_PARTIAL_ABS_LIMIT = 2**-14


@triton.jit
def _prepare_row(
    a,
    b,
    ptr,
    HALF_BYTES: tl.constexpr,
    FLAG_BYTES: tl.constexpr,
    LIMIT: tl.constexpr,
    SUM2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 4)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    offset = row * 2560 + col
    source = tl.load(a + offset, col < 2560, 0).to(tl.float32)
    if SUM2:
        source += tl.load(b + offset, col < 2560, 0).to(tl.float32)
        scaled = (source * 0.125).to(tl.float16)
    else:
        scaled = source.to(tl.float16)
    error = tl.abs(source - scaled.to(tl.float32) * (8 if SUM2 else 1))
    bad = (error > LIMIT) | (source != source) | (tl.abs(source) == float("inf"))
    unsafe = tl.sum(tl.sum(bad.to(tl.int32), 1), 0) > 0
    base = ptr.to(tl.pointer_type(tl.uint8))
    flags = (base + FLAG_BYTES).to(tl.pointer_type(tl.int32))
    tl.store(flags + row, unsafe.to(tl.int32))
    if unsafe:
        fallback = (base + HALF_BYTES).to(tl.pointer_type(tl.float32))
        tl.store(fallback + offset, source, col < 2560)
    else:
        tl.store(ptr.to(tl.pointer_type(tl.float16)) + offset, scaled, col < 2560)


@triton.jit
def _read_partial(
    ptr,
    global_row,
    col,
    HALF_BYTES: tl.constexpr,
    FLAG_BYTES: tl.constexpr,
    SUM2: tl.constexpr,
):
    base = ptr.to(tl.pointer_type(tl.uint8))
    unsafe = tl.load((base + FLAG_BYTES).to(tl.pointer_type(tl.int32)) + global_row)
    offset = global_row * 2560 + col
    if unsafe:
        value = tl.load(
            (base + HALF_BYTES).to(tl.pointer_type(tl.float32)) + offset, col < 2560, 0
        )
    else:
        value = tl.load(ptr.to(tl.pointer_type(tl.float16)) + offset, col < 2560, 0).to(
            tl.float32
        ) * (8 if SUM2 else 1)
    return value


@triton.jit(do_not_specialize=["LOCAL"])
def _peer_combine_norm(
    p0,
    p1,
    p2,
    p3,
    hidden,
    inj,
    weight,
    updated,
    normalized,
    RANK: tl.constexpr,
    LOCAL,
    HALF_BYTES: tl.constexpr,
    FLAG_BYTES: tl.constexpr,
    SUM2: tl.constexpr,
    NORMALIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 4)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    global_row = RANK * LOCAL + row
    partial = _read_partial(p0, global_row, col, HALF_BYTES, FLAG_BYTES, SUM2)
    partial += _read_partial(p1, global_row, col, HALF_BYTES, FLAG_BYTES, SUM2)
    partial += _read_partial(p2, global_row, col, HALF_BYTES, FLAG_BYTES, SUM2)
    partial += _read_partial(p3, global_row, col, HALF_BYTES, FLAG_BYTES, SUM2)
    block = partial.to(tl.float16).to(tl.float32)
    for stream in tl.static_range(4):
        position = row * 10240 + stream * 2560 + col
        residual = tl.load(hidden + position, col < 2560, 0).to(tl.float32)
        factor = 2 * tl.sigmoid(tl.load(inj + row * 4 + stream).to(tl.float32) / 4)
        value = (residual + factor * block).to(tl.float16)
        tl.store(updated + position, value, col < 2560)
        if NORMALIZE:
            fp32 = value.to(tl.float32)
            variance = tl.sum(tl.sum(fp32 * fp32, 1), 0) / 2560
            norm = fp32 * tl.rsqrt(variance + EPS)
            w = tl.load(weight + stream * 2560 + col, col < 2560, 0).to(tl.float32)
            norm += norm * w
            tl.store(normalized + position, norm, col < 2560)


class HcPrefillPeerRuntime:
    """One engine's reusable IPC banks; supports lockless NVLink P2P or NCCL."""

    def __init__(self, group, device: torch.device, max_rows: int):
        import ctypes

        from vllm._sm70_flashnext_prefill_ops import has_sm70_flashnext_prefill_ops
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.group = group
        self.device = device
        self.rank = group.rank_in_group
        self.max_rows = max_rows
        self.nccl = group.device_communicator.pynccl_comm

        # Check whether direct lockless 4-GPU NVLink P2P kernels are available.
        local_dev = device.index or 0
        p2p_local_ok = bool(
            has_sm70_flashnext_prefill_ops()
            and hasattr(torch.ops._C, "sm70_ring_native_peer_atomics")
            and torch.accelerator.device_count() >= 4
        )
        if p2p_local_ok:
            from vllm.platforms import current_platform

            uuids = [""] * 4
            dist.all_gather_object(
                uuids,
                current_platform.get_device_uuid(local_dev),
                group=group.cpu_group,
            )
            visible = {
                current_platform.get_device_uuid(i): i
                for i in range(torch.accelerator.device_count())
            }
            p2p_local_ok = all(
                u in visible
                and (
                    visible[u] == local_dev
                    or torch.ops._C.sm70_ring_native_peer_atomics(local_dev, visible[u])
                )
                for u in uuids
            )
        p2p_flags = [False] * 4
        dist.all_gather_object(p2p_flags, p2p_local_ok, group=group.cpu_group)
        self.p2p_enabled = all(p2p_flags)

        if self.p2p_enabled:
            padded_max_rows = ((max_rows + 3) // 4) * 4
            self.rs_bank_bytes = padded_max_rows * 2560 * 2
            self.w_bank_bytes = (88 * 10240 + 2560 * 320) * 2
            self.num_static_w_slots = 100
            self.ag_bank_bytes = padded_max_rows * 2560 * 2
            self.sync_bytes = 256
            self.total_bytes = (
                2 * self.rs_bank_bytes
                + (self.num_static_w_slots + 4) * self.w_bank_bytes
                + 2 * self.ag_bank_bytes
                + self.sync_bytes
            )
            self.pointers = CustomAllreduce.create_shared_buffer(
                self.total_bytes, group.cpu_group, peer_ranks={0, 1, 2, 3}
            )
            ctypes.CDLL("libcudart.so").cudaMemset(
                ctypes.c_void_p(int(self.pointers[self.rank])), 0, self.total_bytes
            )
            sw_off = 2 * self.rs_bank_bytes
            w_off = sw_off + self.num_static_w_slots * self.w_bank_bytes
            ag_off = w_off + 4 * self.w_bank_bytes
            self.rs_ptrs = [int(p) for p in self.pointers]
            self.sw_ptrs = [int(p) + sw_off for p in self.pointers]
            self.w_ptrs = [int(p) + w_off for p in self.pointers]
            self.ag_ptrs = [int(p) + ag_off for p in self.pointers]
            self.rs_sync = [int(p) + self.total_bytes - 256 for p in self.pointers]
            self.w_sync = [int(p) + self.total_bytes - 192 for p in self.pointers]
            self.ag_sync = [int(p) + self.total_bytes - 128 for p in self.pointers]
            self.rs_ep = torch.zeros(2, dtype=torch.int32, device=device)
            self._rs_ep_host = 0
            self._rs_prestaged_valid = False
            self._rs_prestaged_first_ptr = 0
            self._rs_prestaged_second_ptr = 0
            self._rs_prestaged_scaled8 = False
            self._rs_prestaged_signaled = False
            self._rs_prestaged_slot_ptr = 0
            self.w_ep = torch.zeros(2, dtype=torch.int32, device=device)
            self.ag_ep = torch.zeros(2, dtype=torch.int32, device=device)
            self._ag_ep_host = 0
            self._static_slot_by_key: dict[tuple[int, int], int] = {}
            self.full_down_bufs = [
                torch.empty((336, 10240), dtype=torch.float16, device=device),
                torch.empty((336, 10240), dtype=torch.float16, device=device),
            ]
            self.full_up_bufs = [
                torch.empty((10240, 320), dtype=torch.float16, device=device),
                torch.empty((10240, 320), dtype=torch.float16, device=device),
            ]
            self.full_down_buf = self.full_down_bufs[0]
            self.full_up_buf = self.full_up_bufs[0]
            self._w_buf_idx = 0
            self._w_prefetch_stream = torch.cuda.Stream(device=device)
            self._ag_pull_stream = torch.cuda.Stream(device=device, priority=-1)
            self._ag_mix_event = torch.cuda.Event()
            self._ag_pull_done_event = torch.cuda.Event()
            self._ag_pull_pending = False
            self._ag_pending_ptr = 0
            self._ag_pending_local_rows = 0
            self._w_ready_event = torch.cuda.Event()
            self._ag_done_event = torch.cuda.Event()
            self._w_free_events = [torch.cuda.Event(), torch.cuda.Event()]
            self._prefetched_spec: tuple[int, int, int, int, int] | None = None
            self._prefetched_buf_idx: int = -1
            self._next_spec_after: dict[
                tuple[int, int, int, int, int], tuple[int, int, int, int, int]
            ] = {}
            self._prev_spec: tuple[int, int, int, int, int] | None = None
            torch.accelerator.synchronize(device)
            dist.barrier(group=group.cpu_group)
        else:
            self.half_bytes = max_rows * 2560 * 2
            self.flag_bytes = max_rows * 2560 * 6
            self.total_bytes = self.flag_bytes + max_rows * 4
            self.pointers = CustomAllreduce.create_shared_buffer(
                self.total_bytes, group.cpu_group
            )
            self.barrier_in = torch.zeros(1, dtype=torch.int32, device=device)
            self.barrier_out = torch.empty_like(self.barrier_in)

    def can_use(self, hidden, first, second, injection, weight) -> bool:
        min_rows = 1024 if getattr(self, "p2p_enabled", False) else 4096
        valid_weights = (
            (2560, 10240) if getattr(self, "p2p_enabled", False) else (10240,)
        )
        return bool(
            self.pointers
            and min_rows <= first.shape[0] <= self.max_rows
            and first.shape == (4 * hidden.shape[0], 2560)
            and hidden.shape[1] == 10240
            and injection.shape == (hidden.shape[0], 4)
            and weight.numel() in valid_weights
            and (
                not getattr(self, "p2p_enabled", False)
                or not torch.cuda.is_current_stream_capturing()
            )
            and all(
                t.dtype == torch.float16
                and t.device == self.device
                and t.is_contiguous()
                for t in (hidden, first, injection, weight)
            )
            and (
                second is None
                or (
                    second.shape == first.shape
                    and second.dtype == first.dtype
                    and second.device == first.device
                    and second.is_contiguous()
                )
            )
        )

    def get_next_rs_view(self, padded_rows: int) -> torch.Tensor | None:
        if not (
            getattr(self, "p2p_enabled", False)
            and self.pointers
            and not torch.cuda.is_current_stream_capturing()
            and 1024 <= padded_rows <= ((self.max_rows + 3) // 4) * 4
            and padded_rows * 2560 * 2 <= self.rs_bank_bytes
        ):
            return None
        slot = self._rs_ep_host & 1
        ptr = self.rs_ptrs[self.rank] + slot * self.rs_bank_bytes
        return torch.ops.vllm_sm70_fn_prefill.wrap_ipc_half_2d(
            ptr, padded_rows, 2560, self.device.index or 0
        )

    def mark_rs_prestaged(
        self,
        first_ptr: int,
        second_ptr: int,
        scaled8: bool,
        slot_ptr: int,
        already_signaled: bool = False,
    ) -> None:
        self._rs_prestaged_valid = True
        self._rs_prestaged_first_ptr = first_ptr
        self._rs_prestaged_second_ptr = second_ptr
        self._rs_prestaged_scaled8 = scaled8
        self._rs_prestaged_signaled = already_signaled
        self._rs_prestaged_slot_ptr = slot_ptr

    def consume_rs_prestaged(
        self, first: torch.Tensor, second: torch.Tensor | None
    ) -> tuple[bool, bool, bool]:
        if not getattr(self, "_rs_prestaged_valid", False):
            return False, False, False
        valid_slot_ptr = (
            self.rs_ptrs[self.rank] + (self._rs_ep_host & 1) * self.rs_bank_bytes
        )
        ok = (
            self._rs_prestaged_slot_ptr == valid_slot_ptr
            and int(first.data_ptr()) == self._rs_prestaged_first_ptr
            and (
                (self._rs_prestaged_second_ptr == 0 and second is None)
                or (
                    second is not None
                    and int(second.data_ptr()) == self._rs_prestaged_second_ptr
                )
            )
        )
        scaled8 = self._rs_prestaged_scaled8
        signaled = self._rs_prestaged_signaled
        self._rs_prestaged_valid = False
        self._rs_prestaged_signaled = False
        return ok, scaled8, signaled

    def can_use_prestaged(self, hidden, first, injection, weight) -> bool:
        return bool(
            getattr(self, "p2p_enabled", False)
            and self.pointers
            and not torch.cuda.is_current_stream_capturing()
            and 1024 <= 4 * hidden.shape[0] <= ((self.max_rows + 3) // 4) * 4
            and first.ndim == 2
            and first.shape[1] == 2560
            and hidden.shape[1] == 10240
            and injection.shape == (hidden.shape[0], 4)
            and weight.numel() in (2560, 10240)
            and all(
                t.dtype == torch.float16
                and t.device == self.device
                and t.is_contiguous()
                for t in (hidden, first, injection, weight)
            )
        )

    def wait_ag_peers_ready(self, stream: torch.cuda.Stream | None = None) -> None:
        if getattr(self, "_ag_pull_pending", False):
            target = (
                stream if stream is not None else torch.cuda.current_stream(self.device)
            )
            target.wait_event(self._ag_pull_done_event)
            self._ag_pull_pending = False
            self._ag_pending_ptr = 0
            self._ag_pending_local_rows = 0

    def combine_norm(
        self,
        hidden,
        first,
        second,
        injection,
        weight,
        eps,
        normalize,
        *,
        skip_stage: bool = False,
        pre_scaled8: bool = False,
        already_signaled: bool = False,
    ):
        if getattr(self, "p2p_enabled", False):
            self.wait_ag_peers_ready()
            self._rs_prestaged_valid = False
            self._rs_prestaged_signaled = False
            updated = torch.empty_like(hidden)
            normalized = (
                torch.empty_like(hidden) if normalize else hidden.new_empty((0,))
            )
            torch.ops.vllm_sm70_fn_prefill.hc_p2p_reduce_combine_norm(
                updated,
                normalized,
                hidden,
                first,
                second,
                injection,
                weight,
                float(eps),
                bool(normalize),
                *self.rs_ptrs,
                *self.rs_sync,
                self.rs_ep,
                self.rs_bank_bytes,
                self.rank,
                bool(skip_stage),
                bool(pre_scaled8),
                bool(already_signaled),
            )
            self._rs_ep_host += 1
            return updated, normalized
        sum2 = second is not None
        rows = first.shape[0]
        updated = torch.empty_like(hidden)
        normalized = torch.empty_like(hidden) if normalize else hidden.new_empty((0,))
        _prepare_row[(rows,)](
            first,
            first if second is None else second,
            self.pointers[self.rank],
            self.half_bytes,
            self.flag_bytes,
            _PARTIAL_ABS_LIMIT,
            sum2,
            1024,
            num_warps=8,
        )
        self.nccl.all_reduce(self.barrier_in, self.barrier_out)
        _peer_combine_norm[(hidden.shape[0],)](
            *self.pointers,
            hidden,
            injection,
            weight,
            updated,
            normalized,
            self.rank,
            hidden.shape[0],
            self.half_bytes,
            self.flag_bytes,
            sum2,
            normalize,
            eps,
            1024,
            num_warps=8,
        )
        # The next module may reuse this bank only after all peer reads finish.
        self.nccl.all_reduce(self.barrier_in, self.barrier_out)
        return updated, normalized

    def can_project(
        self, x: torch.Tensor, down: torch.Tensor, up: torch.Tensor
    ) -> bool:
        return bool(
            getattr(self, "p2p_enabled", False)
            and self.pointers
            and not torch.cuda.is_current_stream_capturing()
            and x.ndim == 2
            and 256 <= x.shape[0] <= (self.max_rows + 3) // 4
            and x.shape[1] == 10240
            and x.dtype == torch.float16
            and x.device == self.device
            and x.is_contiguous()
            and down.dtype == torch.float16
            and up.dtype == torch.float16
            and down.device == self.device
            and up.device == self.device
            and down.is_contiguous()
            and up.is_contiguous()
            and (
                (down.shape == (3, 640, 2, 32, 8) and up.shape == (80, 20, 2, 4, 8, 8))
                or (down.shape == (88, 10240) and up.shape == (2560, 320))
            )
        )

    def project_sequence_rows(
        self,
        x: torch.Tensor,
        down: torch.Tensor,
        up: torch.Tensor,
        order: tuple[int, ...],
        *,
        defer_pull: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.wait_ag_peers_ready()
        local_rows = x.shape[0]
        out_rows = 4 * local_rows
        already_unpacked = down.ndim == 2
        o0, o1, o2, o3 = (
            int(order[0]),
            int(order[1]),
            int(order[2]),
            int(order[3]),
        )
        main_stream = torch.cuda.current_stream(self.device)
        w_key = (int(down.data_ptr()), int(up.data_ptr()))
        static_slot = self._static_slot_by_key.get(w_key)
        if (
            static_slot is None
            and len(self._static_slot_by_key) < self.num_static_w_slots
        ):
            if self._prefetched_spec is not None:
                main_stream.wait_event(self._w_ready_event)
                self._prefetched_spec = None
                self._prefetched_buf_idx = -1
            static_slot = len(self._static_slot_by_key)
            self._static_slot_by_key[w_key] = static_slot
            torch.ops.vllm_sm70_fn_prefill.hc_p2p_init_static_weight_slot(
                down,
                up,
                already_unpacked,
                static_slot,
                *self.sw_ptrs,
                *self.w_sync,
                self.w_ep,
                self.rank,
            )

        cur_spec: tuple[int, int, int, int, int] | None = None
        if static_slot is not None:
            cur_spec = (static_slot, o0, o1, o2, o3)
            if self._prev_spec is not None and self._prev_spec != cur_spec:
                self._next_spec_after[self._prev_spec] = cur_spec
            self._prev_spec = cur_spec
            if self._prefetched_spec == cur_spec and self._prefetched_buf_idx >= 0:
                buf_idx = self._prefetched_buf_idx
                main_stream.wait_event(self._w_ready_event)
            else:
                if self._prefetched_spec is not None:
                    main_stream.wait_event(self._w_ready_event)
                buf_idx = self._w_buf_idx
                torch.ops.vllm_sm70_fn_prefill.hc_p2p_gather_static_weights(
                    self.full_down_bufs[buf_idx],
                    self.full_up_bufs[buf_idx],
                    static_slot,
                    *self.sw_ptrs,
                    self.rank,
                    o0,
                    o1,
                    o2,
                    o3,
                )
            self._prefetched_spec = None
            self._prefetched_buf_idx = -1
        else:
            if self._prefetched_spec is not None:
                main_stream.wait_event(self._w_ready_event)
                self._prefetched_spec = None
                self._prefetched_buf_idx = -1
            self._prev_spec = None
            buf_idx = 0
            torch.ops.vllm_sm70_fn_prefill.hc_p2p_unpack_gather_weights(
                self.full_down_bufs[0],
                self.full_up_bufs[0],
                down,
                up,
                already_unpacked,
                *self.w_ptrs,
                *self.w_sync,
                self.w_ep,
                self.rank,
                o0,
                o1,
                o2,
                o3,
                False,
            )

        cur_down = self.full_down_bufs[buf_idx]
        cur_up = self.full_up_bufs[buf_idx]
        lora = torch.empty((local_rows, 320), dtype=torch.float16, device=self.device)
        injection = torch.empty(
            (local_rows, 4), dtype=torch.float16, device=self.device
        )
        if local_rows >= 256 and os.getenv("FN_HC_FUSED_DOWN", "0") != "0":
            torch.ops.vllm_sm70_fn_prefill.hc_down_proj_split_silu(
                lora, injection, x, cur_down
            )
        else:
            projected = torch.empty(
                (local_rows, 336), dtype=torch.float16, device=self.device
            )
            torch.mm(x, cur_down.T, out=projected)
            torch.ops.vllm_sm70_fn_prefill.hc_split_silu(lora, injection, projected)
            del projected

        slot = self._ag_ep_host & 1
        ptr = self.ag_ptrs[self.rank] + slot * self.ag_bank_bytes
        use_fused_up = local_rows >= 256 and os.getenv("FN_HC_FUSED_UP", "0") != "0"
        do_async_pull = (
            defer_pull
            and local_rows >= 256
            and os.getenv("FN_HC_ASYNC_PULL", "1") != "0"
        )
        if use_fused_up and do_async_pull:
            torch.ops.vllm_sm70_fn_prefill.hc_p2p_up_gate_mix_local_only(
                x,
                lora,
                cur_up,
                *self.ag_ptrs,
                *self.ag_sync,
                self.ag_ep,
                self.ag_bank_bytes,
                self.rank,
            )
            del lora
            self._w_free_events[buf_idx].record(main_stream)
            self._ag_mix_event.record(main_stream)
            pull_ctas = int(os.getenv("FN_HC_PULL_CTAS", "60"))
            with torch.cuda.stream(self._ag_pull_stream):
                self._ag_pull_stream.wait_event(self._ag_mix_event)
                torch.ops.vllm_sm70_fn_prefill.hc_p2p_gather_peers_async(
                    out_rows,
                    local_rows,
                    *self.ag_ptrs,
                    self.ag_sync[self.rank],
                    self.ag_ep,
                    self.ag_bank_bytes,
                    self.rank,
                    pull_ctas,
                )
                self._ag_pull_done_event.record(self._ag_pull_stream)
            self._ag_pull_pending = True
            self._ag_pending_ptr = ptr
            self._ag_pending_local_rows = local_rows
        elif use_fused_up:
            torch.ops.vllm_sm70_fn_prefill.hc_p2p_up_gate_mix_peers_pull(
                x,
                lora,
                cur_up,
                out_rows,
                *self.ag_ptrs,
                *self.ag_sync,
                self.ag_ep,
                self.ag_bank_bytes,
                self.rank,
            )
            del lora
            self._w_free_events[buf_idx].record(main_stream)
        elif do_async_pull:
            gate = torch.empty(
                (local_rows, 10240), dtype=torch.float16, device=self.device
            )
            torch.mm(lora, cur_up.T, out=gate)
            del lora
            self._w_free_events[buf_idx].record(main_stream)
            torch.ops.vllm_sm70_fn_prefill.hc_p2p_gate_mix_local_only(
                x,
                gate,
                *self.ag_ptrs,
                *self.ag_sync,
                self.ag_ep,
                self.ag_bank_bytes,
                self.rank,
            )
            del gate
            self._ag_mix_event.record(main_stream)
            pull_ctas = int(os.getenv("FN_HC_PULL_CTAS", "60"))
            with torch.cuda.stream(self._ag_pull_stream):
                self._ag_pull_stream.wait_event(self._ag_mix_event)
                torch.ops.vllm_sm70_fn_prefill.hc_p2p_gather_peers_async(
                    out_rows,
                    local_rows,
                    *self.ag_ptrs,
                    self.ag_sync[self.rank],
                    self.ag_ep,
                    self.ag_bank_bytes,
                    self.rank,
                    pull_ctas,
                )
                self._ag_pull_done_event.record(self._ag_pull_stream)
            self._ag_pull_pending = True
            self._ag_pending_ptr = ptr
            self._ag_pending_local_rows = local_rows
        else:
            gate = torch.empty(
                (local_rows, 10240), dtype=torch.float16, device=self.device
            )
            torch.mm(lora, cur_up.T, out=gate)
            del lora
            self._w_free_events[buf_idx].record(main_stream)
            torch.ops.vllm_sm70_fn_prefill.hc_p2p_gate_mix_peers_pull(
                x,
                gate,
                out_rows,
                *self.ag_ptrs,
                *self.ag_sync,
                self.ag_ep,
                self.ag_bank_bytes,
                self.rank,
            )
            del gate
        self._ag_ep_host += 1

        if cur_spec is not None and os.getenv("FN_HC_ASYNC_W", "1") != "0":
            next_spec = self._next_spec_after.get(cur_spec)
            next_buf_idx = buf_idx ^ 1
            self._w_buf_idx = next_buf_idx
            if next_spec is not None:
                after_ag_ev = (
                    self._ag_pull_done_event if do_async_pull else self._ag_done_event
                )
                if not do_async_pull:
                    self._ag_done_event.record(main_stream)
                ns, no0, no1, no2, no3 = next_spec
                with torch.cuda.stream(self._w_prefetch_stream):
                    self._w_prefetch_stream.wait_event(after_ag_ev)
                    self._w_prefetch_stream.wait_event(
                        self._w_free_events[next_buf_idx]
                    )
                    torch.ops.vllm_sm70_fn_prefill.hc_p2p_gather_static_weights(
                        self.full_down_bufs[next_buf_idx],
                        self.full_up_bufs[next_buf_idx],
                        ns,
                        *self.sw_ptrs,
                        self.rank,
                        no0,
                        no1,
                        no2,
                        no3,
                    )
                    self._w_ready_event.record(self._w_prefetch_stream)
                self._prefetched_spec = next_spec
                self._prefetched_buf_idx = next_buf_idx

        block = torch.ops.vllm_sm70_fn_prefill.wrap_ipc_half_2d(
            ptr, out_rows, 2560, self.device.index or 0
        )
        return block, injection

    def can_allgather_rs(self, src: torch.Tensor) -> bool:
        return bool(
            getattr(self, "p2p_enabled", False)
            and self.pointers
            and not torch.cuda.is_current_stream_capturing()
            and src.ndim == 2
            and src.shape[0] > 0
            and src.device == self.device
            and src.is_contiguous()
            and (src.shape[1] * src.element_size()) % 16 == 0
            and src.shape[0] * src.shape[1] * src.element_size() <= self.rs_bank_bytes
        )

    def allgather_rs_into(self, out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        self._rs_prestaged_valid = False
        torch.ops.vllm_sm70_fn_prefill.hc_p2p_allgather_rows(
            out,
            src,
            *self.rs_ptrs,
            *self.rs_sync,
            self.rs_ep,
            self.rs_bank_bytes,
            self.rank,
        )
        self._rs_ep_host += 1
        return out

    def allgather_topk_rs(
        self,
        loc_w: torch.Tensor,
        loc_ids: torch.Tensor,
        m: int,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._rs_prestaged_valid = False
        k = int(loc_ids.shape[1])
        out_w = torch.empty((m, k), dtype=torch.float32, device=self.device)
        out_ids = torch.empty((m, k), dtype=loc_ids.dtype, device=self.device)
        torch.ops.vllm_sm70_fn_prefill.hc_p2p_allgather_topk(
            out_w,
            out_ids,
            loc_w,
            loc_ids,
            step,
            *self.rs_ptrs,
            *self.rs_sync,
            self.rs_ep,
            self.rs_bank_bytes,
            self.rank,
        )
        self._rs_ep_host += 1
        return out_w, out_ids

    def can_allgather_ag(self, src: torch.Tensor) -> bool:
        return bool(
            getattr(self, "p2p_enabled", False)
            and self.pointers
            and not torch.cuda.is_current_stream_capturing()
            and src.ndim == 2
            and src.shape[0] > 0
            and src.device == self.device
            and src.is_contiguous()
            and (src.shape[1] * src.element_size()) % 16 == 0
            and src.shape[0] * src.shape[1] * src.element_size() <= self.ag_bank_bytes
        )

    def allgather_ag_into(self, out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        self.wait_ag_peers_ready()
        torch.ops.vllm_sm70_fn_prefill.hc_p2p_allgather_rows(
            out,
            src,
            *self.ag_ptrs,
            *self.ag_sync,
            self.ag_ep,
            self.ag_bank_bytes,
            self.rank,
        )
        self._ag_ep_host += 1
        return out

    def close(self) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        if not self.pointers:
            return
        torch.accelerator.synchronize(self.device)
        pointers, self.pointers = self.pointers, []
        for rank, pointer in enumerate(pointers):
            if rank != self.rank:
                torch.ops._C.sm70_ring_close_mem_handle(pointer)
        CustomAllreduce.free_shared_buffer(pointers, rank=self.rank)


def prepare_hc_prefill_peer_runtime(vllm_config, device) -> None:
    """Bind after model qualification; keep missing transport on the old stages."""
    policy = vllm_config.kernel_config.layer_execution
    if not (policy.hc_prefill_sequence_parallel and policy.hc_prefill_peer_norm):
        return
    resources = runtime_resources_for(vllm_config)
    if "hc_prefill_peer" in resources:
        return
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    communicator = group.device_communicator
    ca = getattr(communicator, "ca_comm", None)
    nccl = getattr(communicator, "pynccl_comm", None)
    max_rows = min(
        16384, max(_MAX_ROWS, vllm_config.scheduler_config.max_num_batched_tokens)
    )
    sched_rows = vllm_config.scheduler_config.max_num_batched_tokens
    reason = (
        "requires_tp4"
        if group.world_size != 4
        else "requires_full_nvlink_mesh"
        if ca is None or not getattr(ca, "fully_connected", False)
        else "nccl_unavailable"
        if nccl is None or nccl.disabled
        else "ipc_close_operator_missing"
        if not hasattr(torch.ops._C, "sm70_ring_close_mem_handle")
        else "query_capacity_below_1024"
        if sched_rows < 1024
        else None
    )
    reasons = [None] * group.world_size
    dist.all_gather_object(reasons, reason, group=group.cpu_group)
    rejected = next((value for value in reasons if value is not None), None)
    status = {
        "enabled": rejected is None,
        "reason": rejected,
        "max_query_rows": max_rows,
    }
    vllm_config.kernel_config.collective_kernel_selections["hc_prefill_peer"] = status
    if rejected is not None:
        logger.info("HC prefill peer reduction retains original stages: %s", rejected)
        return
    resources["hc_prefill_peer"] = HcPrefillPeerRuntime(group, device, max_rows)
    logger.info("HC prefill TP4 peer reduction enabled: %s", status)


def _combine_norm(
    hidden: torch.Tensor,
    payload: torch.Tensor,
    injection: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    moe_payload: bool,
    partial: bool,
    normalize: bool,
    second: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed import get_tp_group

    from .ops.hc import hc_combine, hc_combine_norm

    group = get_tp_group()
    rows = payload.shape[0] // (2 if moe_payload else 1)
    first = payload[:rows]
    if moe_payload:
        if second is not None:
            raise ValueError("HC payload cannot be both packed and separate")
        second = payload[rows:]
    if second is not None and second.shape != first.shape:
        raise ValueError("HC partial contributions must have matching shapes")
    padding = 4 * hidden.shape[0] - rows
    if not 0 <= padding <= 3:
        raise ValueError("HC sequence payload must match the padded row ownership")
    resources = current_runtime_resources()
    owner = resources.get("hc_prefill_peer") if resources is not None else None
    if partial and owner is not None and owner.group is group:
        prestaged_ok, pre_scaled8, already_signaled = owner.consume_rs_prestaged(
            first, second
        )
        if prestaged_ok and owner.can_use_prestaged(hidden, first, injection, weight):
            return owner.combine_norm(
                hidden,
                first,
                None,
                injection,
                weight,
                eps,
                normalize,
                skip_stage=True,
                pre_scaled8=pre_scaled8,
                already_signaled=already_signaled,
            )
    if padding:
        first = torch.nn.functional.pad(first, (0, 0, 0, padding))
        if second is not None:
            second = torch.nn.functional.pad(second, (0, 0, 0, padding))
    if (
        partial
        and owner is not None
        and owner.group is group
        and owner.can_use(hidden, first, second, injection, weight)
    ):
        return owner.combine_norm(
            hidden, first, second, injection, weight, eps, normalize
        )
    if partial:
        # Attention can supply one partial, while MoE supplies two. Both
        # use the qualified FP32 transport before the original FP16 combine.
        source = first.float() if second is None else first.float() + second.float()
        reduced = group.reduce_scatter(source, dim=0).to(first.dtype)
    else:
        reduced = first.reshape(4, -1, first.shape[-1])[group.rank_in_group]
    if normalize:
        return hc_combine_norm(hidden, reduced, injection, weight, eps, 4)
    return hc_combine(hidden, reduced, injection, 4), hidden.new_empty((0,))


def _combine_norm_fake(
    hidden,
    payload,
    injection,
    weight,
    eps,
    moe_payload,
    partial,
    normalize,
    second=None,
):
    return torch.empty_like(hidden), (
        torch.empty_like(hidden) if normalize else hidden.new_empty((0,))
    )


direct_register_custom_op(
    "qwen38_hc_sequence_combine_norm",
    _combine_norm,
    fake_impl=_combine_norm_fake,
)
hc_sequence_combine_norm = torch.ops.vllm.qwen38_hc_sequence_combine_norm


def _shard_rows(value: torch.Tensor) -> torch.Tensor:
    """Pad only the HC carrier, then return an owned physical-rank row span."""
    from vllm.distributed import get_tp_group

    rows = (value.shape[0] + 3) // 4
    padding = 4 * rows - value.shape[0]
    if padding:
        value = torch.nn.functional.pad(value, (0, 0, 0, padding))
    return value.reshape(4, rows, value.shape[-1])[get_tp_group().rank_in_group].clone()


def _shard_rows_fake(value: torch.Tensor) -> torch.Tensor:
    return value.new_empty(((value.shape[0] + 3) // 4, value.shape[-1]))


direct_register_custom_op(
    "qwen38_hc_sequence_shard_rows",
    _shard_rows,
    fake_impl=_shard_rows_fake,
)
hc_sequence_shard_rows = torch.ops.vllm.qwen38_hc_sequence_shard_rows


def _gather_rows_into(output: torch.Tensor, local: torch.Tensor, tokens: int) -> None:
    """Commit HC rows into their existing owner without a full temporary.

    Only complete, contiguous TP4 receive spans use the existing PyNCCL
    transport directly. A partial final rank preserves the owner's unused
    capacity (at most three rows); insufficient padded capacity or unavailable
    transport keeps the original gather/copy fallback.
    """
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if (
        group.world_size != 4
        or output.ndim != 2
        or local.ndim != 2
        or output.shape[1] != local.shape[1]
        or output.dtype != local.dtype
        or output.device != local.device
        or not 0 <= tokens <= output.shape[0]
        or local.shape[0] != (tokens + 3) // 4
    ):
        raise ValueError("HC row commit requires matching TP4 rows and capacity")
    if tokens == 0:
        return
    resources = current_runtime_resources()
    owner = resources.get("hc_prefill_peer") if resources is not None else None
    if (
        owner is not None
        and owner.group is group
        and output.is_contiguous()
        and owner.can_allgather_ag(local)
    ):
        owner.allgather_ag_into(output[:tokens], local)
        return
    padded = 4 * local.shape[0]
    communicator = getattr(group.device_communicator, "pynccl_comm", None)
    if (
        communicator is None
        or communicator.disabled
        or output.shape[0] < padded
        or not output.is_contiguous()
        or not local.is_contiguous()
    ):
        output[:tokens].copy_(group.all_gather(local, dim=0)[:tokens])
        return
    tail = output[tokens:padded].clone() if tokens < padded else None
    communicator.all_gather(output[:padded], local)
    if tail is not None:
        output[tokens:padded].copy_(tail)


def _gather_rows_into_fake(
    output: torch.Tensor, local: torch.Tensor, tokens: int
) -> None:
    return


direct_register_custom_op(
    "qwen38_hc_sequence_gather_into",
    _gather_rows_into,
    mutates_args=["output"],
    fake_impl=_gather_rows_into_fake,
)
hc_sequence_gather_into = torch.ops.vllm.qwen38_hc_sequence_gather_into


def _allgather_sequence_rows(local: torch.Tensor, tokens: int) -> torch.Tensor:
    from vllm.distributed import get_tp_group

    if tokens == 0:
        return local.new_empty((0, local.shape[1]))
    group = get_tp_group()
    resources = current_runtime_resources()
    owner = resources.get("hc_prefill_peer") if resources is not None else None
    if owner is not None and owner.group is group and owner.can_allgather_rs(local):
        out = torch.empty(
            (tokens, local.shape[1]), dtype=local.dtype, device=local.device
        )
        return owner.allgather_rs_into(out, local)
    return group.all_gather(local, dim=0)[:tokens]


def _allgather_sequence_rows_fake(local: torch.Tensor, tokens: int) -> torch.Tensor:
    return local.new_empty((tokens, local.shape[1]))


direct_register_custom_op(
    "qwen38_hc_sequence_allgather_rows",
    _allgather_sequence_rows,
    fake_impl=_allgather_sequence_rows_fake,
)
hc_sequence_allgather_rows = torch.ops.vllm.qwen38_hc_sequence_allgather_rows
