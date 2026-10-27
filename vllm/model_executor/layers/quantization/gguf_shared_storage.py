# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader and lifetime owner for one local TP4 original GGUF cold pool."""

import atexit
import json
import os
import secrets
import socket
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.shared_host import SharedHostArena

from .gguf_shared_copy import swap_original_experts
from .gguf_shared_layout import SharedExpertBudget
from .gguf_shared_runtime import (
    SHARDS,
    SharedExpertLayerSpec,
    SharedExpertOwner,
    SharedExpertRank,
)

logger = init_logger(__name__)


def _join_shared_group(specs, group, owner_factory=None):
    """Transfer descriptors over a private, authenticated local Unix socket.

    Only the socket path and nonce pass through the CPU process group. Every
    consumer checks the size-sealed memfd identities and metadata fingerprint
    before registering its own CUDA aliases. No device pointers cross ranks.
    """
    rank = group.rank_in_group
    owner = None
    data = control = None
    try:
        if rank == 0:
            owner = (
                SharedExpertOwner(tuple(specs))
                if owner_factory is None
                else owner_factory()
            )
            with tempfile.TemporaryDirectory(prefix="vllm-gex-") as directory:
                path = directory + "/fd"
                nonce = secrets.token_hex(32)
                with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
                    server.settimeout(60)
                    server.bind(path)
                    server.listen(3)
                    group.broadcast_object((path, nonce, socket.gethostname()), src=0)
                    seen = set()
                    for _ in range(3):
                        connection, _ = server.accept()
                        with connection:
                            connection.settimeout(60)
                            _, uid, _ = struct.unpack(
                                "3i",
                                connection.getsockopt(
                                    socket.SOL_SOCKET, socket.SO_PEERCRED, 12
                                ),
                            )
                            peer, token = json.loads(connection.recv(4096))
                            if (
                                uid != os.getuid()
                                or token != nonce
                                or peer not in (1, 2, 3)
                                or peer in seen
                            ):
                                raise RuntimeError("Invalid shared expert FD consumer")
                            seen.add(peer)
                            owner.send(connection)
            return owner, owner.data, owner.control
        path, nonce, hostname = group.broadcast_object(None, src=0)
        if hostname != socket.gethostname():
            raise ValueError("shared_original requires a single-node TP4 group")
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(60)
            connection.connect(path)
            connection.sendall(json.dumps((rank, nonce)).encode())
            data = SharedHostArena.receive(connection)
            control = SharedHostArena.receive(connection)
        return None, data, control
    except BaseException:
        if owner is not None:
            owner.close()
        else:
            if control is not None:
                control.close()
            if data is not None:
                data.close()
        raise


class SharedOriginalExpertCache:
    """One canonical cold allocation, raw GPU TP carriers and one workspace.

    Startup reads each original projection only on rank 0. Cold experts are
    written directly to their final allocation. Hot experts pass through the
    group exchange reserve in batches of eight, with a CPU barrier after each
    rank's GPU copy completes. This never allocates a rank-local host bank.
    """

    shared_original = True

    def __init__(self, layers, config, device, group=None):
        if group is None:
            from vllm.distributed import get_tp_group

            group = get_tp_group()
        self.group = group
        self.rank = group.rank_in_group
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.accelerator.current_device_index())
        if (
            group.world_size != 4
            or self.device.type != "cuda"
            or self.device.index != self.rank
            or (dist.is_initialized() and dist.get_world_size() != 4)
        ):
            raise ValueError("shared_original currently requires local GPU 0-3, TP4")
        config.validate_shared_expert_storage()
        self.layers = [index for index, *_ in layers]
        if len(set(self.layers)) != len(layers):
            raise ValueError("Shared expert layer ids are not unique")
        self.index = {index: i for i, index in enumerate(self.layers)}
        self.prefill_tokens = int(
            envs.VLLM_V41_PREFILL_GROUP or config.expert_cache_prefill_tokens
        )
        self.banks = {}
        specs = []
        for _, layer, method, kinds in layers:
            if (
                not method.is_v41
                or layer.tp_size != 4
                or layer.tp_rank != self.rank
                or set(kinds) != set(SHARDS)
            ):
                raise ValueError("Shared original experts require complete V4.1 banks")
            experts = method.num_experts
            cold = round(config.expert_cache_offload_fraction * experts)
            specs.append(
                SharedExpertLayerSpec(
                    method.hidden_size,
                    method.intermediate_size * 4,
                    experts,
                    experts - cold,
                    tuple(kinds[s] for s in SHARDS),
                )
            )
        if not specs or len({s.experts for s in specs}) != 1:
            raise ValueError("Shared expert layers require a common expert count")
        self.specs = tuple(specs)
        self.experts = specs[0].experts
        self.budget = SharedExpertBudget(
            self.experts,
            config.expert_cache_offload_fraction,
            tuple(s.expert_bytes for s in specs),
            config.expert_cache_spare_slots,
        )
        # Each projection bank has one upstream-reader tail, independently
        # of the number of slots. Keep a single aligned allocation for all.
        allocations, total = [], 0
        for i, spec in enumerate(specs):
            for shard, geometry in zip(SHARDS, spec.layouts(self.rank)):
                total = (total + 255) // 256 * 256
                size = (
                    self.budget.gpu_slots
                    * geometry.device_rows
                    * geometry.device_row_bytes
                )
                allocations.append((i, shard, geometry, total, size))
                total += size + geometry.guard_bytes
        self.owner = self.data = self.control = self.runtime = self.arena = None
        self.tm = None
        self.loaded = set()
        try:
            if self.rank == 0:
                with open("/proc/meminfo") as memory:
                    available = next(
                        int(line.split()[1]) * 1024
                        for line in memory
                        if line.startswith("MemAvailable:")
                    )
                if available < self.budget.arena_bytes + 8 * 1024**3:
                    raise MemoryError(
                        "Shared expert pool would breach 8 GiB host reserve"
                    )
            self.arena = torch.empty(total, dtype=torch.uint8, device=device)
            for i, shard, geometry, offset, size in allocations:
                self.banks[i, shard] = self.arena[offset : offset + size].view(
                    self.budget.gpu_slots,
                    geometry.device_rows,
                    geometry.device_row_bytes,
                )
                self.arena[offset + size : offset + size + geometry.guard_bytes].zero_()
            self.owner, self.data, self.control = _join_shared_group(specs, group)
            self.runtime = SharedExpertRank(
                self.data, self.control, self.rank, self.specs, self.banks
            )
            self.exchange_offset = self.budget.cold_bytes
            self._load_slots = torch.arange(8, dtype=torch.int32, device=device)
            self._load_offsets = torch.empty(8, dtype=torch.int64, device=device)
            self._load_saves = torch.full_like(self._load_offsets, -1)
            self._load_count = torch.empty(1, dtype=torch.int32, device=device)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)
        logger.info(
            "Shared original experts: %.3f GB cold once per TP group, %d cold "
            "experts/layer, %d GPU slots/rank, %.3f GB raw GPU arena/rank",
            self.budget.cold_bytes / 1e9,
            self.budget.cold_experts,
            self.budget.gpu_slots,
            total / 1e9,
        )

    def _barrier(self):
        dist.barrier(group=self.group.cpu_group)

    def bank(self, layer, shard):
        return self.banks[self.index[layer], shard]

    def _read_projection(self, tensor, jobs):
        """Read (expert, arena-offset) jobs into their final shared bytes."""
        from .gguf_expert_storage import _source

        source = _source(tensor)
        size = int(np.prod(tensor.data.shape[1:]))
        data = self.data
        if data is None:
            raise RuntimeError("Shared expert storage is closed")

        def read(job):
            expert, offset = job
            view = memoryview(data.buffer)[offset : offset + size]
            try:
                if source is None:
                    view[:] = memoryview(tensor.data[expert]).cast("B")
                    return
                fd, start = source
                start += expert * size
                while view:
                    got = os.preadv(fd, [view], start)
                    if got <= 0:
                        raise OSError("Short read of a shared GGUF expert")
                    remainder = view[got:]
                    view.release()
                    view = remainder
                    start += got
            finally:
                view.release()

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(read, jobs))

    def store_original_bank(self, layer, shard, tensor):
        if self.runtime is None:
            raise RuntimeError("Shared expert storage is closed")
        i = self.index[layer]
        spec = self.specs[i]
        projection = SHARDS.index(shard)
        geometry = spec.layouts(self.rank)[projection]
        if (
            (i, shard) in self.loaded
            or int(tensor.tensor_type) != spec.kinds[projection]
            or tuple(tensor.data.shape)
            != (spec.experts, geometry.source_rows, geometry.source_row_bytes)
        ):
            raise ValueError("Shared GGUF source projection metadata mismatch")
        at = sum(spec.projection_bytes[:projection])
        cold_start = sum(
            (s.experts - s.resident) * s.expert_bytes for s in self.specs[:i]
        )
        if self.rank == 0:
            self._read_projection(
                tensor,
                [
                    (e, cold_start + (e - spec.resident) * spec.expert_bytes + at)
                    for e in range(spec.resident, spec.experts)
                ],
            )
        for first in range(0, spec.resident, 8):
            count = min(8, spec.resident - first)
            offsets = [self.exchange_offset + j * spec.expert_bytes for j in range(8)]
            if self.rank == 0:
                self._read_projection(
                    tensor, [(first + j, offsets[j] + at) for j in range(count)]
                )
            self._barrier()
            self._load_slots.copy_(
                torch.arange(first, first + 8, dtype=torch.int32, device=self.device)
            )
            self._load_offsets.copy_(
                torch.tensor(offsets, dtype=torch.int64, device=self.device)
            )
            self._load_count.fill_(count)
            swap_original_experts(
                self.banks[i, shard],
                self.runtime.mapped,
                self._load_slots,
                self._load_offsets,
                self._load_saves,
                self._load_count,
                geometry,
                projection_offset=at,
            )
            torch.accelerator.synchronize(self.device)
            self._barrier()  # All readers release the exchange batch together.
        self.loaded.add((i, shard))
        if len(self.loaded) == 3 * len(self.specs):
            from .gguf_shared_prefill import SharedOriginalPrefill

            self.runtime.warmup()
            self.tm = SharedOriginalPrefill(self)
            self._barrier()
            del self._load_slots, self._load_offsets, self._load_saves, self._load_count

    def resolve(self, layer, ids, weights=None):
        if self.runtime is None:
            raise RuntimeError("Shared expert storage is closed")
        if self.tm is not None:
            self.tm.wait_decode()
        return self.runtime.resolve(self.index[layer], ids, weights)

    def prefill_moe(self, layer, x, ids, weights):
        if self.tm is None:
            raise RuntimeError("Shared original prefill workspace is not initialized")
        return self.tm.forward(self.index[layer], x, ids, weights)

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        self.tm = None
        if self.owner is not None:
            self.owner.close()
            self.owner = None
            self.data = self.control = None
        else:
            if self.control is not None:
                self.control.close()
                self.control = None
            if self.data is not None:
                self.data.close()
                self.data = None
