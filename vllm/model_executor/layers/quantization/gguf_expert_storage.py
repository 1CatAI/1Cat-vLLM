# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contiguous allocation of original expert banks, preserving TP block guards."""

import math

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_native import (
    NATIVE_TYPES,
    packed_tp_span,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size

logger = init_logger(__name__)


def original_expert_bank(
    shape, weight_type, shard, hidden, intermediate, tp_size, tp_rank
):
    """Return the local byte shape and the upstream final-row safety tail."""
    k, rows, experts = map(int, shape)
    block, size = quant_size(weight_type)
    if shard in ("w1", "w3") and k == hidden and rows == intermediate * tp_size:
        physical_k = k
        local_rows = intermediate
        packed_k = k // block * size
    elif shard == "w2" and k == intermediate * tp_size and rows == hidden:
        first, last, _, physical_k = packed_tp_span(k, weight_type, tp_size, tp_rank)
        local_rows = hidden
        packed_k = last - first
    else:
        raise ValueError(f"Original GGUF expert shape {(k, rows, experts)} is invalid")
    if k % block or (-physical_k) % 512 % block:
        raise ValueError("Original GGUF expert bank is not quantization-block aligned")
    tail = ((-physical_k) % 512) // block * size
    return (experts, local_rows, packed_k), tail


def prepare_original_expert_arena(model, name_map, tensors, device):
    """Reserve immutable expert banks before replaceable dense loading storage."""
    shards = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    banks = []
    total = 0
    for raw, name in name_map.items():
        if ".experts." not in name or not name.endswith(".weight"):
            continue
        prefix, projection, _ = name.rsplit(".", 2)
        if projection not in shards:
            continue
        layer = model.get_submodule(prefix)
        method = layer.quant_method
        tensor = tensors[raw]
        value = int(tensor.tensor_type)
        if not isinstance(method, GGUFNativeMoEMethod) or value not in NATIVE_TYPES:
            continue
        shape, tail = original_expert_bank(
            tensor.shape,
            value,
            shards[projection],
            method.hidden_size,
            method.intermediate_size,
            layer.tp_size,
            layer.tp_rank,
        )
        if shape[0] != method.num_experts:
            raise ValueError("GGUF expert count does not match the model")
        # Logical input columns of the local slice within the stored rows.
        if shards[projection] == "w2":
            _, _, k_offset, _ = packed_tp_span(
                int(tensor.shape[0]), value, layer.tp_size, layer.tp_rank
            )
            source = (value, method.intermediate_size, k_offset)
        else:
            source = (value, method.hidden_size, 0)
        total = (total + 255) // 256 * 256
        elements = math.prod(shape)
        banks.append(
            (layer, method, shards[projection], shape, total, elements, tail, source)
        )
        total += elements + tail
    if not banks:
        return {}
    # Raw tensor name -> (layer, method, shard) for whole-bank loading.
    targets = {}
    for raw, name in name_map.items():
        if ".experts." not in name or not name.endswith(".weight"):
            continue
        prefix, projection, _ = name.rsplit(".", 2)
        if projection in shards:
            layer = model.get_submodule(prefix)
            if any(layer is b[0] and shards[projection] == b[2] for b in banks):
                targets[raw] = (layer, layer.quant_method, shards[projection])
    cache_config = _expert_cache_config()
    if cache_config is not None:
        _prepare_expert_cache(banks, cache_config, device)
        return targets
    arena = torch.empty(total, dtype=torch.uint8, device=device)
    for layer, method, shard, shape, offset, elements, tail, _ in banks:
        arena[offset + elements : offset + elements + tail].zero_()
        layer.register_buffer(
            "gguf_" + shard,
            arena[offset : offset + elements].view(shape),
            persistent=False,
        )
        method.guarded_shards.add(shard)
    return targets


_READ_THREADS = 16
_READ_PIECE = 32 << 20
_POOL = None
_FILES: dict[str, int] = {}


def _source(tensor):
    """(file descriptor, absolute byte offset) of a memory-mapped tensor."""
    import numpy as np

    base = tensor.data
    while base is not None and not isinstance(base, np.memmap):
        base = base.base
    if base is None or base.filename is None:
        return None
    path = str(base.filename)
    fd = _FILES.get(path)
    if fd is None:
        import os

        fd = _FILES[path] = os.open(path, os.O_RDONLY)
    # Views of a memmap keep its `offset` attribute; the reader records the
    # tensor's absolute file offset.
    offset = getattr(tensor, "data_offset", None)
    if offset is None:
        return None
    return fd, int(offset)


def _read(fd, ranges, out):
    """Read ``(file offset, length)`` ranges back to back into ``out``.

    Page-faulting a strided slice of the memory map reads ~0.1-0.3 GB/s;
    concurrent large preads keep the NVMe queue full (~1 GB/s and more).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(_READ_THREADS)
    flat = out.reshape(-1)
    jobs = []
    position = 0
    for offset, length in ranges:
        for start in range(0, length, _READ_PIECE):
            n = min(_READ_PIECE, length - start)
            jobs.append((offset + start, position + start, n))
        position += length

    def one(job):
        offset, position, n = job
        view = memoryview(flat[position : position + n])
        while n:
            got = os.preadv(fd, [view], offset)
            if got <= 0:
                raise OSError(f"Short read of GGUF experts at {offset}")
            view, offset, n = view[got:], offset + got, n - got

    list(_POOL.map(one, jobs))


def _local_slice(tensor, shard, rows, packed_span):
    """This rank's [experts, rows, bytes] slice in pinned memory, read
    without page-faulting the memory map."""
    import numpy as np

    data = tensor.data
    experts, all_rows, width = data.shape
    if shard == "w2":
        shape = (experts, all_rows, packed_span[1] - packed_span[0])
    else:
        shape = (experts, rows[1] - rows[0], width)
    local = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
    out = local.numpy()
    source = _source(tensor)
    if source is None:
        if shard == "w2":
            out[...] = data[:, :, packed_span[0] : packed_span[1]]
        else:
            out[...] = data[:, rows[0] : rows[1]]
        return local
    fd, offset = source
    stride = all_rows * width
    if shard == "w2":
        # K slices interleave with every row: read the whole bank.
        whole = np.empty((experts, all_rows, width), dtype=np.uint8)
        _read(fd, [(offset, experts * stride)], whole)
        out[...] = whole[:, :, packed_span[0] : packed_span[1]]
        return local
    length = (rows[1] - rows[0]) * width
    _read(
        fd,
        [(offset + e * stride + rows[0] * width, length) for e in range(experts)],
        out,
    )
    return local


class _Prefetcher:
    """Reads expert banks ahead, in the order the loader asks for them.

    At most ``depth`` banks (~0.1-0.5 GB each) wait in pinned memory, so
    reading the next bank overlaps storing the current one.
    """

    def __init__(self, jobs, depth=4):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self.jobs = list(jobs)  # [(raw, tensor, shard, rows, span)]
        self.order = {raw: i for i, (raw, *_) in enumerate(self.jobs)}
        self.pool = ThreadPoolExecutor(1)
        self.slots = threading.Semaphore(depth)
        self.futures = {}
        self.next = 0
        self._fill()

    def _fill(self):
        while self.next < len(self.jobs) and self.slots.acquire(blocking=False):
            raw, tensor, shard, rows, span = self.jobs[self.next]
            self.futures[raw] = self.pool.submit(
                _local_slice, tensor, shard, rows, span
            )
            self.next += 1

    def take(self, raw):
        future = self.futures.pop(raw, None)
        if future is None:
            return None
        local = future.result()
        self.slots.release()
        self._fill()
        return local


def _slice_spec(layer, method, shard, tensor):
    """(rows, packed byte span) of this rank's slice of one expert bank."""
    if shard == "w2":
        value = int(tensor.tensor_type)
        block, size = quant_size(value)
        full_k = tensor.data.shape[2] // size * block
        first, last, _, _ = packed_tp_span(full_k, value, layer.tp_size, layer.tp_rank)
        return None, (first, last)
    rows = method.intermediate_size
    return (layer.tp_rank * rows, (layer.tp_rank + 1) * rows), None


_PREFETCH = None


def start_prefetch(targets, tensors) -> None:
    """Begin reading the expert banks in loader order (see load_expert_bank)."""
    global _PREFETCH
    jobs = []
    for raw, (layer, method, shard) in targets.items():
        tensor = tensors[raw]
        rows, span = _slice_spec(layer, method, shard, tensor)
        jobs.append((raw, tensor, shard, rows, span))
    _PREFETCH = _Prefetcher(jobs) if jobs else None


def load_expert_bank(layer, method, shard, tensor, raw=None) -> None:
    """Load all experts of one stacked GGUF projection in one pass.

    Replaces per-expert payloads (tens of thousands of small page-faulting
    reads and synchronous copies per rank): this rank's TP slice is read once
    from the memory-mapped file, resident experts go to the GPU in one copy
    and cold experts to the pinned host arena.
    """
    value = int(tensor.tensor_type)
    previous = method.weight_types.setdefault(shard, value)
    if previous != value:
        raise ValueError(f"GGUF {shard} has inconsistent expert types")
    import time

    started = time.perf_counter()
    tp_size, tp_rank = layer.tp_size, layer.tp_rank
    rows, span = _slice_spec(layer, method, shard, tensor)
    if shard == "w2":
        block, size = quant_size(value)
        full_k = tensor.data.shape[2] // size * block
        _, _, left, physical_k = packed_tp_span(full_k, value, tp_size, tp_rank)
        method.input_padding[shard] = (left, physical_k)
    local = _PREFETCH.take(raw) if _PREFETCH is not None and raw else None
    if local is None:
        local = _local_slice(tensor, shard, rows, span)
    read = time.perf_counter()
    cache = getattr(method, "expert_cache", None)
    if cache is not None:
        cache.store_bank(method.cache_layer, shard, local)
    else:
        bank = getattr(layer, "gguf_" + shard)
        if tuple(bank.shape) != tuple(local.shape):
            raise ValueError(
                f"GGUF {shard} bank {tuple(bank.shape)} != {tuple(local.shape)}"
            )
        bank.copy_(local)
    method.loaded_experts[shard] = set(range(method.num_experts))
    global _LOADED_BYTES, _LOADED_BANKS, _READ_S, _STORE_S
    _LOADED_BANKS += 1
    _LOADED_BYTES += local.numel()
    _READ_S += read - started
    _STORE_S += time.perf_counter() - read
    if _LOADED_BANKS % 48 == 0:
        logger.info(
            "Loaded %d GGUF expert banks (%.2f GiB) by whole-bank "
            "reads: %.1f s reading, %.1f s storing",
            _LOADED_BANKS,
            _LOADED_BYTES / 2**30,
            _READ_S,
            _STORE_S,
        )


_LOADED_BANKS = 0
_LOADED_BYTES = 0
_READ_S = 0.0
_STORE_S = 0.0


def _expert_cache_config():
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    gguf = config.kernel_config.sm70_gguf if config is not None else None
    if gguf is None or not 0 < gguf.expert_cache_fraction < 1:
        return None
    return gguf


def _prepare_expert_cache(banks, config, device):
    """Give every original expert bank a slot bank of the routed-expert cache."""
    from vllm.model_executor.layers.quantization import gguf_expert_cache

    entries = []
    for layer, method, shard, shape, _, _, tail, source in banks:
        name = str(getattr(layer, "layer_name", None) or getattr(layer, "prefix", ""))
        index = gguf_expert_cache.layer_index(name)
        if index < 0:
            raise ValueError(f"Expert cache cannot index layer {name}")
        entries.append((index, shard, shape, tail, source))
    cache = gguf_expert_cache.create_cache(
        entries, banks[0][1].num_experts, config, device
    )
    for (layer, method, shard, *_), (index, *_) in zip(banks, entries):
        layer.register_buffer(
            "gguf_" + shard, cache.bank(index, shard), persistent=False
        )
        method.guarded_shards.add(shard)
        method.expert_cache = cache
        method.cache_layer = index
