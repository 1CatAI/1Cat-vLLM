# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 NVLink P2P fused Q-exchange and LSE-combine+gate for Host-KV DCP=2."""

import ctypes
import importlib

import torch
import torch.distributed as dist

from vllm._sm70_flashnext_prefill_ops import has_sm70_flashnext_prefill_ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

DCP2_MAX_M = 64
DCP2_BULK_M = 4096
DCP2_H_LOCAL = 6
DCP2_DIM = 256
# Double-buffered uint2 (8B) ping-pong exchange slots per rank (M <= 64).
_Q_SLOT_WORDS = DCP2_MAX_M * DCP2_H_LOCAL * (DCP2_DIM // 2)
_O_SLOT_WORDS = DCP2_MAX_M * DCP2_H_LOCAL * (DCP2_DIM + 1)
_Q_BYTES = 2 * _Q_SLOT_WORDS * 8
_O_BYTES = 2 * _O_SLOT_WORDS * 8
# Bulk 16B P2P slots + 256B sync footer for prefill (arbitrary M > 64).
_Q_BULK_BYTES = 2 * DCP2_BULK_M * DCP2_H_LOCAL * DCP2_DIM * 2 + 256
_O_BULK_BYTES = 2 * DCP2_BULK_M * DCP2_H_LOCAL * (DCP2_DIM * 2 + 4) + 256


class Sm70Dcp2Runtime:
    """Direct 2-GPU NVLink P2P buffers and scratch for DCP=2 QSA decode."""

    def __init__(self, device: torch.device):
        from vllm.distributed import get_dcp_group
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.reason: str | None = None
        if not current_platform.is_device_capability(70):
            self.reason = "requires_SM70"
            return
        dcp = get_dcp_group()
        if dcp.world_size != 2:
            self.reason = "requires_dcp2"
            return
        try:
            importlib.import_module("vllm._sm70_qsa_device_C")
        except ImportError:
            self.reason = "native_extension_unavailable"
            return
        if not (
            hasattr(torch.ops.vllm_sm70_qsa_device, "dcp2_q_prep")
            and hasattr(torch.ops.vllm_sm70_qsa_device, "dcp2_combine_gate")
            and hasattr(torch.ops._C, "sm70_ring_native_peer_atomics")
        ):
            self.reason = "operator_missing"
            return

        group = dcp.cpu_group
        self.dcp_rank = int(dcp.rank_in_group)
        peer_rank = 1 - self.dcp_rank
        local_dev = device.index

        import pynvml

        uuids = ["", ""]
        dist.all_gather_object(
            uuids, current_platform.get_device_uuid(local_dev), group=group
        )
        visible = {
            current_platform.get_device_uuid(i): i
            for i in range(torch.accelerator.device_count())
        }
        if uuids[peer_rank] not in visible:
            self.reason = "peer_not_local"
            return
        peer_dev = visible[uuids[peer_rank]]
        pynvml.nvmlInit()
        try:
            h_local = pynvml.nvmlDeviceGetHandleByUUID(uuids[self.dcp_rank])
            h_peer = pynvml.nvmlDeviceGetHandleByUUID(uuids[peer_rank])
            nvlink_ok = (
                pynvml.nvmlDeviceGetP2PStatus(
                    h_local, h_peer, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                )
                == pynvml.NVML_P2P_STATUS_OK
            )
        finally:
            pynvml.nvmlShutdown()

        ok = bool(
            nvlink_ok
            and torch.ops._C.sm70_ring_native_peer_atomics(local_dev, peer_dev)
        )
        admitted = [False, False]
        dist.all_gather_object(admitted, ok, group=group)
        if not all(admitted):
            self.reason = "direct_nvlink_p2p_unavailable"
            return

        self.bulk_enabled = has_sm70_flashnext_prefill_ops()
        total = (
            _Q_BYTES
            + _O_BYTES
            + ((_Q_BULK_BYTES + _O_BULK_BYTES) if self.bulk_enabled else 0)
        )
        with torch.accelerator.device_index(local_dev):
            pointers = CustomAllreduce.create_shared_buffer(
                total, group=group, peer_ranks={0, 1}
            )
            ctypes.CDLL("libcudart.so").cudaMemset(
                ctypes.c_void_p(int(pointers[self.dcp_rank])), 0, total
            )
            self._pointers = pointers
            self.my_q_ptr = int(pointers[self.dcp_rank])
            self.peer_q_ptr = int(pointers[peer_rank])
            self.my_o_ptr = self.my_q_ptr + _Q_BYTES
            self.peer_o_ptr = self.peer_q_ptr + _Q_BYTES
            if self.bulk_enabled:
                self.my_q_bulk_ptr = self.my_o_ptr + _O_BYTES
                self.peer_q_bulk_ptr = self.peer_o_ptr + _O_BYTES
                self.my_o_bulk_ptr = self.my_q_bulk_ptr + _Q_BULK_BYTES
                self.peer_o_bulk_ptr = self.peer_q_bulk_ptr + _Q_BULK_BYTES
                self.q_bulk_seq = torch.zeros(2, dtype=torch.int32, device=device)
                self.o_bulk_seq = torch.zeros(2, dtype=torch.int32, device=device)
            self._bulk_gathered_q: torch.Tensor | None = None
            self._bulk_local_positions: torch.Tensor | None = None
            self._bulk_local_lengths: torch.Tensor | None = None
            self.q_seq = torch.zeros(2, dtype=torch.int32, device=device)
            self.o_seq = torch.zeros(2, dtype=torch.int32, device=device)
            self.gathered_q = torch.empty(
                (DCP2_MAX_M, 2 * DCP2_H_LOCAL, DCP2_DIM),
                dtype=torch.float16,
                device=device,
            )
            self.local_positions = torch.empty(
                (DCP2_MAX_M,), dtype=torch.int64, device=device
            )
            self.local_lengths = torch.empty((128,), dtype=torch.int32, device=device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)
        logger.info_once(
            "SM70 NVLink P2P fused DCP=2 Q-exchange & LSE-combine+gate enabled "
            "(dcp_rank=%d, local_dev=%d, peer_dev=%d, bulk_prefill=%s).",
            self.dcp_rank,
            local_dev,
            peer_dev,
            self.bulk_enabled,
        )

    @property
    def enabled(self) -> bool:
        return self.reason is None

    def preallocate_bulk(self, max_m: int, max_reqs: int = 128) -> None:
        if not self.enabled or not self.bulk_enabled or max_m <= DCP2_MAX_M:
            return
        device = self.gathered_q.device
        cap = max(int(max_m), DCP2_BULK_M)
        if self._bulk_gathered_q is None or self._bulk_gathered_q.shape[0] < cap:
            self._bulk_gathered_q = torch.empty(
                (cap, 2 * DCP2_H_LOCAL, DCP2_DIM),
                dtype=torch.float16,
                device=device,
            )
            self._bulk_local_positions = torch.empty(
                (cap,), dtype=torch.int64, device=device
            )
        if (
            self._bulk_local_lengths is None
            or self._bulk_local_lengths.shape[0] < max_reqs
        ):
            self._bulk_local_lengths = torch.empty(
                (max(int(max_reqs), 128),), dtype=torch.int32, device=device
            )

    def q_prep(
        self,
        query: torch.Tensor,
        positions: torch.Tensor,
        lengths: torch.Tensor,
        interleave: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = query.shape[0]
        b = lengths.shape[0]
        gq = self.gathered_q[:m]
        lp = self.local_positions[:m]
        ll = self.local_lengths[:b]
        torch.ops.vllm_sm70_qsa_device.dcp2_q_prep(
            query,
            gq,
            positions.contiguous(),
            lengths.contiguous(),
            lp,
            ll,
            self.my_q_ptr,
            self.peer_q_ptr,
            self.q_seq,
            self.dcp_rank,
            int(interleave),
        )
        return gq, lp, ll

    def q_prep_any_m(
        self,
        query: torch.Tensor,
        positions: torch.Tensor,
        lengths: torch.Tensor,
        interleave: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = query.shape[0]
        b = lengths.shape[0]
        device = query.device
        if self._bulk_gathered_q is None or self._bulk_gathered_q.shape[0] < m:
            cap = max(m, DCP2_BULK_M)
            self._bulk_gathered_q = torch.empty(
                (cap, 2 * DCP2_H_LOCAL, DCP2_DIM),
                dtype=torch.float16,
                device=device,
            )
            self._bulk_local_positions = torch.empty(
                (cap,), dtype=torch.int64, device=device
            )
        if self._bulk_local_lengths is None or self._bulk_local_lengths.shape[0] < b:
            self._bulk_local_lengths = torch.empty(
                (max(b, 128),), dtype=torch.int32, device=device
            )
        assert self._bulk_local_positions is not None
        gq = self._bulk_gathered_q[:m]
        lp = self._bulk_local_positions[:m]
        ll = self._bulk_local_lengths[:b]
        torch.ops.vllm_sm70_fn_prefill.dcp2_q_prep_any_m(
            query,
            gq,
            positions.contiguous(),
            lengths.contiguous(),
            lp,
            ll,
            self.my_q_bulk_ptr,
            self.peer_q_bulk_ptr,
            self.q_bulk_seq,
            self.dcp_rank,
            int(interleave),
        )
        return gq, lp, ll

    def combine_gate(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        gate: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        torch.ops.vllm_sm70_qsa_device.dcp2_combine_gate(
            partial_output,
            partial_lse,
            gate,
            output,
            self.my_o_ptr,
            self.peer_o_ptr,
            self.o_seq,
            self.dcp_rank,
        )

    def combine_gate_any_m(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        gate: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        torch.ops.vllm_sm70_fn_prefill.dcp2_combine_gate_any_m(
            partial_output,
            partial_lse,
            gate,
            output,
            self.my_o_bulk_ptr,
            self.peer_o_bulk_ptr,
            self.o_bulk_seq,
            self.dcp_rank,
        )


_DCP2_RUNTIME: Sm70Dcp2Runtime | None = None


def get_sm70_dcp2_runtime(device: torch.device) -> Sm70Dcp2Runtime:
    global _DCP2_RUNTIME
    if _DCP2_RUNTIME is None:
        _DCP2_RUNTIME = Sm70Dcp2Runtime(device)
    return _DCP2_RUNTIME
