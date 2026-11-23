# SPDX-License-Identifier: Apache-2.0
"""Intra-node exchange of the four DP scheduler contributions on DGX-2H."""

import os
import socket
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


class _HostAgreement:
    def __init__(self, slots: torch.Tensor, rank: int, timeout: timedelta):
        self.slots = slots
        self.rank = rank
        self.timeout_s = timeout.total_seconds()
        self.epoch = 0

    def exchange(self, original: int, padded: int, ubatch: bool, mode: int):
        self.epoch += 1
        return torch.ops.v41solo.host_agreement(
            self.slots, self.rank, self.epoch, original, padded, ubatch, mode,
            self.timeout_s,
        )


def _map_slots(group):
    # Each group gets a fresh, private, zero-filled mapping. Only opt in if
    # all four workers opened it. The inode outlives unlink through the maps.
    rank = group.rank_in_group
    path = None
    description = [None]
    if rank == 0:
        try:
            with tempfile.NamedTemporaryFile(dir="/dev/shm", delete=False) as f:
                path = f.name
                f.truncate(512)
                st = os.fstat(f.fileno())
                description[0] = (path, socket.gethostname(), st.st_dev, st.st_ino)
        except OSError:
            pass
    try:
        dist.broadcast_object_list(description, src=group.ranks[0], group=group.cpu_group)
        if description[0] is None:
            return None
        name, hostname, dev, ino = description[0]
        slots = None
        try:
            st = os.stat(name)
            if hostname == socket.gethostname() and (st.st_dev, st.st_ino) == (dev, ino):
                slots = torch.from_file(name, shared=True, size=64, dtype=torch.int64)
        except (OSError, RuntimeError):
            pass
        ready = torch.tensor(int(slots is not None), dtype=torch.int32)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group.cpu_group)
        if not ready.item():
            return None
        logger.info_once("V4.1 using shared memory for four local DP scheduler ranks")
        return _HostAgreement(slots, rank, group.cpu_group._get_backend(torch.device("cpu")).options._timeout)
    finally:
        if rank == 0 and path is not None:
            os.unlink(path)


def host_agreement(group):
    if not hasattr(group, "_v41_host_agreement"):
        from .native import load_v41sk

        load_v41sk()
        group._v41_host_agreement = (
            _map_slots(group)
            if group.world_size == 4 and hasattr(torch.ops.v41solo, "host_agreement")
            else None
        )
    return group._v41_host_agreement
