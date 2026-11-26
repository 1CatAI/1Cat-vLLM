# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-rank FP16 all-reduce through pinned host memory (csrc/host_reduce.cu)
for GPU pairs that NCCL cannot connect peer to peer."""

import os
import uuid
from typing import Any

import torch
import torch.distributed as dist

from vllm.config import get_current_vllm_config_or_none
from vllm.config.kernel import HostReduceConfig
from vllm.platforms import current_platform

_OPS = (
    "host_reduce_open",
    "host_reduce_close",
    "host_reduce_out",
    "host_reduce_copy_engine_out",
)


def _nccl_peer_access(group, device: torch.device) -> bool:
    """Whether NCCL may connect the two ranks peer to peer; NCCL_P2P_DISABLE
    turns it off regardless of the hardware."""
    if os.environ.get("NCCL_P2P_DISABLE", "0") not in ("", "0"):
        return False
    local = device.index
    if local is None:
        local = torch.accelerator.current_device_index()
    uuids = [""] * 2
    dist.all_gather_object(uuids, current_platform.get_device_uuid(local), group=group)
    visible = {
        current_platform.get_device_uuid(i): i
        for i in range(torch.accelerator.device_count())
    }
    peer = visible.get(uuids[1 - dist.get_rank(group)])
    # A peer outside this worker's visible devices cannot be reached directly.
    return peer is not None and torch.cuda.can_device_access_peer(local, peer)


class HostReduceCommunicator:
    def __init__(self, group, device: torch.device, name: str):
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.handle: int | None = None
        self.status: dict[str, Any] = {
            "enabled": False,
            "reason": None,
            "scope": "collective_capability",
        }
        cfg = get_current_vllm_config_or_none()
        self.policy = cfg.kernel_config.host_reduce if cfg else HostReduceConfig()
        reason = None
        if not self.policy.enabled:
            reason = "disabled_by_configuration"
        elif dist.get_world_size(group) != 2:
            reason = "requires_two_ranks"
        elif not current_platform.is_cuda():
            reason = "requires_cuda"
        elif missing := [op for op in _OPS if not hasattr(torch.ops._C, op)]:
            reason = "operator_missing:" + ",".join(missing)
        if dist.get_world_size(group) == 2:
            reasons: list[str | None] = [None] * 2
            dist.all_gather_object(reasons, reason, group=group)
            reason = next((r for r in reasons if r is not None), None)
            if reason is None:
                access = [False] * 2
                dist.all_gather_object(
                    access, _nccl_peer_access(group, device), group=group
                )
                if any(access):
                    reason = "peer_access_available"
        if reason is None:
            self._open()
        self.status.update(
            enabled=reason is None,
            reason=reason,
            max_bytes=self.policy.max_bytes,
            copy_engine_max_bytes=self.policy.copy_engine_max_bytes,
        )
        if cfg:
            cfg.kernel_config.collective_kernel_selections[name] = self.status

    def _open(self) -> None:
        # Both ranks map one shared file; it is unlinked once both hold it, so
        # nothing outlives the processes.
        paths = [f"/dev/shm/vllm_host_reduce_{uuid.uuid4().hex}"]
        dist.broadcast_object_list(
            paths, src=dist.get_global_rank(self.group, 0), group=self.group
        )
        with torch.accelerator.device_index(self.device.index):
            self.handle = torch.ops._C.host_reduce_open(
                paths[0],
                self.rank,
                self.policy.max_bytes,
                self.policy.copy_engine_max_bytes,
            )
        dist.barrier(group=self.group)
        if self.rank == 0:
            os.unlink(paths[0])
        self.status.update(dtype="float16", transport="pinned_host_memory")

    def rejection_reason(self, tensor: torch.Tensor) -> str | None:
        if not self.status["enabled"]:
            return self.status["reason"]
        if tensor.dtype != torch.float16:
            return "fp16_input_required"
        if tensor.device != self.device or not tensor.is_contiguous():
            return "requires_contiguous_local_cuda_input"
        nbytes = tensor.numel() * tensor.element_size()
        largest = max(self.policy.max_bytes, self.policy.copy_engine_max_bytes)
        if not 0 < nbytes <= largest or nbytes % 16:
            return "payload_outside_channel"
        # The copy-engine mode orders its pieces from the host, so a graph
        # cannot replay it; captured payloads of that size keep the next
        # collective in the chain.
        if nbytes > self.policy.max_bytes and torch.cuda.is_current_stream_capturing():
            return "copy_engine_mode_not_capturable"
        if tensor.data_ptr() % 16:
            return "requires_16_byte_alignment"
        return None

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor | None:
        if self.rejection_reason(tensor) is not None:
            return None
        output = torch.empty_like(tensor)
        if tensor.numel() * tensor.element_size() <= self.policy.max_bytes:
            torch.ops._C.host_reduce_out(output, tensor, self.handle)
        else:
            torch.ops._C.host_reduce_copy_engine_out(output, tensor, self.handle)
        return output

    def close(self) -> None:
        if self.handle is not None:
            torch.ops._C.host_reduce_close(self.handle)
            self.handle = None
