# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""On-demand module staging backed by immutable pinned CPU storage."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from vllm.logger import init_logger


def is_dtensor(t):
    return isinstance(t, DTensor)


def set_tensor_storage(target, value):
    if is_dtensor(target):
        target._local_tensor = value
    else:
        target.data = value


logger = init_logger(__name__)


class BoundedAllocatorCache:
    """Retain reusable allocator blocks without monopolizing device memory.

    Component offload normally calls ``empty_cache`` after every stage. That
    makes the next stage return to the device allocator even though PyTorch's
    cached blocks are immediately reusable. This policy keeps the cache while
    both of these bounds hold:

    * cached-but-unallocated memory is at most 25% of device capacity; and
    * at least 5% of device capacity is physically free.

    Missing memory telemetry is handled conservatively by releasing the cache.
    Failure paths can force release; normal executor shutdown keeps its own
    unconditional device-cache cleanup because this policy is not global.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_cached_fraction: float = 0.25,
        min_free_fraction: float = 0.05,
    ) -> None:
        if not 0.0 <= max_cached_fraction <= 1.0:
            raise ValueError(
                f"max_cached_fraction must be in [0, 1], got {max_cached_fraction}"
            )
        if not 0.0 <= min_free_fraction <= 1.0:
            raise ValueError(
                f"min_free_fraction must be in [0, 1], got {min_free_fraction}"
            )
        self.device = device
        self.max_cached_fraction = max_cached_fraction
        self.min_free_fraction = min_free_fraction

    def _should_release(self) -> bool:
        reserved = int(torch.accelerator.memory_reserved(self.device))
        allocated = int(torch.accelerator.memory_allocated(self.device))
        free, total = torch.accelerator.get_memory_info(self.device)
        cached = max(0, reserved - allocated)
        return cached > int(total * self.max_cached_fraction) or free < int(
            total * self.min_free_fraction
        )

    def release_if_needed(self, *, force: bool = False) -> bool:
        """Release cached blocks when a bound is crossed or release is forced."""
        if not force:
            try:
                if not self._should_release():
                    return False
            except Exception as exc:
                # Preserve the pre-retention behavior on platforms that do not
                # expose allocator telemetry through torch.accelerator.
                logger.debug(
                    "Allocator cache telemetry unavailable; releasing cache: %s", exc
                )
        torch.accelerator.empty_cache()
        return True


@dataclass(frozen=True)
class _TensorBinding:
    target: torch.Tensor
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


@dataclass
class _StorageGroup:
    master: torch.Tensor
    bindings: list[_TensorBinding]


class PinnedModuleStager:
    """Stage immutable module groups without copying device weights to CPU.

    ``nn.Module.to("cpu")`` performs a device-to-host copy for every parameter
    after each forward. In inference the weights are immutable, so retain one
    pinned CPU master instead. ``load`` materializes device storage from that
    master; ``offload`` only rebinds Parameters and buffers to the master.

    A module iterable is treated as one staging group. It uses one copy stream
    and one reusable completion event. Tensors sharing storage keep their
    shapes, strides, offsets, dtypes, and aliases across every transition.
    """

    def __init__(
        self,
        module: nn.Module | Iterable[nn.Module],
        device: torch.device,
        *,
        pin_memory: bool = True,
        copy_stream: Any | None = None,
        cache_retention: BoundedAllocatorCache | None = None,
    ) -> None:
        modules = (module,) if isinstance(module, nn.Module) else tuple(module)
        if not modules or not all(isinstance(item, nn.Module) for item in modules):
            raise ValueError("PinnedModuleStager requires at least one nn.Module")

        self.device = device
        self.copy_stream = (
            copy_stream if copy_stream is not None else torch.cuda.Stream()
        )
        self._ready_event = torch.cuda.Event()
        self.cache_retention = cache_retention
        self.loaded = False
        self._groups = self._snapshot_groups(modules, pin_memory=pin_memory)
        self._device_storages: list[torch.Tensor] = []
        self._restore_masters()

    @staticmethod
    def _local_tensor(target: torch.Tensor) -> torch.Tensor:
        return target.to_local() if is_dtensor(target) else target

    @classmethod
    def _snapshot_groups(
        cls,
        modules: tuple[nn.Module, ...],
        *,
        pin_memory: bool,
    ) -> list[_StorageGroup]:
        targets: list[torch.Tensor] = []
        seen_targets: set[int] = set()
        for target in chain.from_iterable(
            chain(item.parameters(), item.buffers()) for item in modules
        ):
            if id(target) not in seen_targets:
                seen_targets.add(id(target))
                targets.append(target)

        grouped: dict[tuple[Any, ...], tuple[torch.Tensor, list[_TensorBinding]]] = {}
        for target in targets:
            local = cls._local_tensor(target)
            if local.is_meta:
                raise ValueError("PinnedModuleStager cannot snapshot a meta tensor")
            storage = local.untyped_storage()
            if storage.nbytes() == 0:
                storage_key: tuple[Any, ...] = ("empty", id(target))
            else:
                storage_key = (
                    local.device.type,
                    local.device.index,
                    storage.data_ptr(),
                    storage.nbytes(),
                )
            binding = _TensorBinding(
                target=target,
                dtype=local.dtype,
                shape=tuple(local.shape),
                stride=tuple(local.stride()),
                storage_offset=local.storage_offset(),
            )
            if storage_key not in grouped:
                grouped[storage_key] = (local, [])
            grouped[storage_key][1].append(binding)

        groups: list[_StorageGroup] = []
        while grouped:
            _, (source, bindings) = grouped.popitem()
            storage = source.untyped_storage()
            storage_view = torch.empty(0, dtype=torch.uint8, device=source.device).set_(
                storage,
                0,
                (storage.nbytes(),),
                (1,),
            )
            master = (
                storage_view.detach()
                if storage_view.device.type == "cpu"
                else storage_view.to("cpu")
            )
            if pin_memory and not master.is_pinned():
                master = master.pin_memory()
            groups.append(_StorageGroup(master=master, bindings=bindings))
            # Release each pageable allocation as soon as its pinned master is
            # ready. Keeping both full copies until the entire encoder is
            # pinned can transiently double TP4 host memory.
            for binding in bindings:
                set_tensor_storage(binding.target, cls._view(master, binding))
        return groups

    @staticmethod
    def _view(backing: torch.Tensor, binding: _TensorBinding) -> torch.Tensor:
        element_size = binding.dtype.itemsize
        backing_offset = backing.storage_offset() * backing.element_size()
        if backing_offset % element_size:
            raise ValueError("shared weight storage must preserve dtype alignment")
        return torch.empty(0, dtype=binding.dtype, device=backing.device).set_(
            backing.untyped_storage(),
            backing_offset // element_size + binding.storage_offset,
            binding.shape,
            binding.stride,
        )

    def share_cpu_storage(self, directory: str | Path) -> None:
        """Share a checked immutable replica across this engine's TP workers.

        The engine owns the temporary directory and removes it after workers
        stop. Private mappings prevent accidental CPU writes from affecting a
        different rank. Only identical complete storage groups may be shared.
        """
        import torch.distributed as dist

        if self.loaded or any(group.master.is_pinned() for group in self._groups):
            raise ValueError("shared host storage requires unloaded pageable masters")
        rank = dist.get_rank() if dist.is_initialized() else 0
        directory = Path(directory)
        if rank == 0:
            self._write_shared_groups(directory)
        if dist.is_initialized():
            dist.barrier()
        self._read_shared_groups(directory)

    @staticmethod
    def _group_description(group: _StorageGroup) -> dict:
        return {
            "bytes": group.master.numel(),
            "bindings": [
                {
                    "dtype": str(binding.dtype),
                    "shape": list(binding.shape),
                    "stride": list(binding.stride),
                    "storage_offset": binding.storage_offset,
                }
                for binding in group.bindings
            ],
        }

    @staticmethod
    def _storage_digest(master: torch.Tensor) -> str:
        return hashlib.sha256(memoryview(master.numpy())).hexdigest()

    def _write_shared_groups(self, directory: Path) -> None:
        directory.mkdir(mode=0o700)
        records, total = [], 0
        for group in self._groups:
            total = (total + 255) // 256 * 256
            records.append({**self._group_description(group), "offset": total})
            total += group.master.numel()
        data_path = directory / "weights.bin"
        fd = os.open(data_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            # Reserve tmpfs space before mapping so a later write cannot SIGBUS
            # because unrelated processes consume the remaining free space.
            if total:
                os.posix_fallocate(fd, 0, total)
        finally:
            os.close(fd)
        shared = torch.from_file(
            str(data_path), shared=True, size=total, dtype=torch.uint8
        )
        for group, record in zip(self._groups, records):
            target = shared.narrow(0, record["offset"], record["bytes"])
            target.copy_(group.master)
            record["sha256"] = self._storage_digest(target)
        (directory / "metadata.json").write_text(
            json.dumps({"version": 1, "bytes": total, "groups": records})
        )

    def _read_shared_groups(self, directory: Path) -> None:
        metadata = json.loads((directory / "metadata.json").read_text())
        records = metadata["groups"]
        if metadata["version"] != 1 or len(records) != len(self._groups):
            raise ValueError("shared component storage inventory differs across ranks")
        # COW mappings share physical pages while keeping the file immutable.
        shared = torch.from_file(
            str(directory / "weights.bin"),
            shared=False,
            size=metadata["bytes"],
            dtype=torch.uint8,
        )
        for group, record in zip(self._groups, records):
            if any(
                record[key] != value
                for key, value in self._group_description(group).items()
            ):
                raise ValueError("shared component tensor layouts differ across ranks")
            if self._storage_digest(group.master) != record["sha256"]:
                raise ValueError("shared component weights differ across ranks")
            target = shared.narrow(0, record["offset"], record["bytes"])
            if self._storage_digest(target) != record["sha256"]:
                raise ValueError("shared component snapshot failed its checksum")
            group.master = target
            for binding in group.bindings:
                set_tensor_storage(binding.target, self._view(target, binding))

    def _bind(self, storages: list[torch.Tensor]) -> None:
        for storage, group in zip(storages, self._groups):
            for binding in group.bindings:
                set_tensor_storage(binding.target, self._view(storage, binding))

    def _restore_masters(self) -> None:
        self._bind([group.master for group in self._groups])

    def set_cache_retention(
        self, cache_retention: BoundedAllocatorCache | None
    ) -> None:
        self.cache_retention = cache_retention

    def _release_cache(self, *, force: bool = False) -> None:
        if self.cache_retention is None:
            torch.accelerator.empty_cache()
        else:
            self.cache_retention.release_if_needed(force=force)

    def _load_once(self) -> None:
        device_storages = [
            torch.empty_like(group.master, device=self.device) for group in self._groups
        ]
        compute_stream = torch.cuda.current_stream()
        self.copy_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self.copy_stream):
            for device_storage, group in zip(device_storages, self._groups):
                device_storage.copy_(
                    group.master, non_blocking=group.master.is_pinned()
                )
            self._ready_event.record(self.copy_stream)

        self._bind(device_storages)
        compute_stream.wait_event(self._ready_event)
        self._device_storages = device_storages
        self.loaded = True

    def _cleanup_failed_load(self) -> None:
        try:
            torch.accelerator.synchronize()
        except Exception:
            logger.debug(
                "Device synchronization failed while cleaning up module staging",
                exc_info=True,
            )
        self._restore_masters()
        self._device_storages.clear()
        self.loaded = False
        self._release_cache(force=True)

    def load(self) -> None:
        if self.loaded:
            return
        try:
            self._load_once()
        except torch.OutOfMemoryError:
            # A retained cache is normally reusable by this process, but an
            # explicit flush gives external memory pressure one bounded retry.
            self._cleanup_failed_load()
            try:
                self._load_once()
            except BaseException:
                self._cleanup_failed_load()
                raise
        except BaseException:
            self._cleanup_failed_load()
            raise

    def offload(self) -> None:
        if not self.loaded:
            return

        # The module has completed on the compute stream. Synchronize once at
        # the stage boundary, then discard device storage without any D2H copy.
        try:
            torch.accelerator.synchronize()
            self._restore_masters()
        except BaseException:
            self._device_storages.clear()
            self.loaded = False
            self._release_cache(force=True)
            raise
        self._device_storages.clear()
        self.loaded = False
        self._release_cache()


__all__ = ["BoundedAllocatorCache", "PinnedModuleStager"]
