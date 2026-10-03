# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read safetensors checkpoints with O_DIRECT, past the page cache.

Automatic loading uses this path when a checkpoint exceeds the available
host/cgroup RAM budget and aligned storage reads are supported. Otherwise
it retains mapped loading. Direct reads use private buffers instead of
filling the page cache with checkpoint weights.

Under pipeline parallelism a memory-mapped shard only reads the pages a stage
touches. A reader that fills buffers itself has to decide up front, so
tensors of decoder layers outside this stage's range are not read at all.

Only decoder-layer tensors, the bulk of a checkpoint, are read this way. The
rest (embeddings, heads, vision towers, MTP layers) and tensors the model asks
to keep mapped are memory-mapped, so a stage reads only what it uses, and their
pages are released from the page cache once the loader has consumed them.
"""

import ctypes
import json
import mmap
import os
import struct
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
import torch.distributed as dist
from safetensors import safe_open

# O_DIRECT needs offsets, lengths and buffers aligned to the logical block
# size; 4 KiB covers the devices in use.
_ALIGN = 4096
# Tensors closer than this are read in one request with the gap between them.
_MAX_GAP = 1 << 20
# Upper bound for one coalesced read; a larger single tensor gets its own read.
_MAX_RUN = 256 << 20

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# `<prefix>layers.<N>.` where <prefix> is a decoder stack whose N is the global
# layer index. MTP heads (`mtp.layers.N`) and vision towers number their own
# layers and are always read.
_DECODER_LAYER = re.compile(
    r"^(?:|model\.|model\.language_model\.|language_model\.model\.)layers\.(\d+)\."
)


MADV_DONTNEED = 4


def _madvise(address: int, length: int, advice: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.madvise(ctypes.c_void_p(address), ctypes.c_size_t(length), advice):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def is_decoder_layer_weight(name: str) -> bool:
    """Whether a tensor belongs to a decoder layer of the main stack."""
    return _DECODER_LAYER.match(name) is not None


def decoder_layer_filter(start: int, end: int) -> Callable[[str], bool]:
    """Return a predicate that is True for tensors of decoder layers outside
    [start, end), i.e. the tensors another pipeline stage owns."""

    def outside(name: str) -> bool:
        match = _DECODER_LAYER.match(name)
        return match is not None and not start <= int(match.group(1)) < end

    return outside


def _read_header(fd: int) -> tuple[dict, int]:
    head = os.pread(fd, 8, 0)
    (header_len,) = struct.unpack("<Q", head)
    header = json.loads(os.pread(fd, header_len, 8))
    return header, 8 + header_len


@dataclass(frozen=True)
class RunSharing:
    """Ranks that read the same tensors of a shard (the tensor-parallel ranks
    of one pipeline stage): the leader reads each run from disk and
    broadcasts it over the CPU group, so the group reads the checkpoint once."""

    group: dist.ProcessGroup
    leader: int  # global rank
    is_leader: bool


def _run_buffer(start: int, end: int) -> tuple[mmap.mmap, int]:
    """An aligned buffer for [start, end) and the file offset it starts at."""
    begin = start - start % _ALIGN
    stop = -(-end // _ALIGN) * _ALIGN
    # Private, not the default shared mapping: shared anonymous memory is
    # shmem, which only swap can evict and which /proc/self/maps shows as
    # the file /dev/zero.
    buffer = mmap.mmap(-1, stop - begin, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    return buffer, begin


def _read_run(fd: int, start: int, end: int) -> tuple[mmap.mmap, int]:
    """Read [start, end) with O_DIRECT; return the buffer and the aligned
    file offset it starts at."""
    buffer, begin = _run_buffer(start, end)
    view = memoryview(buffer)
    done = 0
    while begin + done < end:
        count = os.preadv(fd, [view[done:]], begin + done)
        if count == 0:
            raise EOFError(
                f"safetensors shard ended at offset {begin + done}, "
                f"expected data up to {end}"
            )
        done += count
    view.release()
    return buffer, begin


def released_mapped_weights(
    path: str, keep: Callable[[str], bool]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield the memory-mapped tensors of one shard for which `keep` is True,
    and drop each one's pages from the page cache once the consumer is done
    with it (a loader copies a tensor before it asks for the next one).

    Only whole pages inside the tensor are released, so a neighbour's pages
    stay. A consumer that keeps an unmodified tensor still reads correct data
    (the pages fault back in from disk); in-place writes to a kept tensor
    are lost, since safetensors maps the shard privately.
    """
    page = os.sysconf("SC_PAGE_SIZE")
    with safe_open(path, framework="pt") as shard, open(path, "rb", buffering=0) as raw:
        header, data_start = _read_header(raw.fileno())
        for name in shard.keys():  # noqa: SIM118
            if not keep(name):
                continue
            tensor = shard.get_tensor(name)
            yield name, tensor
            begin, end = header[name]["data_offsets"]
            if end == begin:
                continue  # posix_fadvise length 0 would mean "to end of file"
            # A page still mapped here is one the page cache keeps; unmap
            # first, then drop the file range.
            address = tensor.data_ptr()
            first = -(-address // page) * page
            last = (address + end - begin) // page * page
            if last > first:
                _madvise(first, last - first, MADV_DONTNEED)
            os.posix_fadvise(
                raw.fileno(), data_start + begin, end - begin, os.POSIX_FADV_DONTNEED
            )


def _coalesce(
    tensors: list[tuple[int, int, str, dict]],
) -> list[tuple[int, int, int, int]]:
    """(first, last, start, end): tensors[first:last] are read as one run."""
    runs = []
    index = 0
    while index < len(tensors):
        run_start, run_end = tensors[index][0], tensors[index][1]
        last = index + 1
        while (
            last < len(tensors)
            and tensors[last][0] - run_end < _MAX_GAP
            and tensors[last][1] - run_start <= _MAX_RUN
        ):
            run_end = max(run_end, tensors[last][1])
            last += 1
        runs.append((index, last, run_start, run_end))
        index = last
    return runs


def _check_same_runs(
    path: str,
    runs: list[tuple[int, int, int, int]],
    sharing: RunSharing,
    leader_ok: bool,
) -> None:
    """Every rank of the group must expect the leader's runs, and the leader
    must have opened the shard; otherwise a broadcast would pair different
    tensors or never come. All ranks agree on the outcome, so they fail
    together instead of waiting for a lost peer."""
    leader_runs: list[object] = [runs]
    dist.broadcast_object_list(leader_runs, src=sharing.leader, group=sharing.group)
    same = torch.tensor([int(leader_runs[0] == runs and leader_ok)], device="cpu")
    dist.all_reduce(same, op=dist.ReduceOp.MIN, group=sharing.group)
    if not same.item():
        raise RuntimeError(
            f"Direct I/O run sharing: the ranks of the group select different "
            f"tensors of {path}, or its leader could not open it"
        )


def _group_ready(ok: bool, sharing: RunSharing) -> bool:
    """Whether every rank of the group has its part of the run: the leader
    the data it read, the others a buffer to receive it into."""
    status = torch.tensor([int(ok)], device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=sharing.group)
    return bool(status.item())


def cgroup_available_bytes(
    root: Path = Path("/sys/fs/cgroup"),
    membership: Path = Path("/proc/self/cgroup"),
) -> int | None:
    """Respect cgroup-v2 hard and pressure limits, including parent groups."""
    try:
        entry = next(
            line[3:]
            for line in membership.read_text().splitlines()
            if line.startswith("0::")
        )
    except (OSError, StopIteration):
        return None
    root = root.resolve()
    node = (root / entry.lstrip("/")).resolve()
    if not node.is_relative_to(root):
        return None
    remaining = []
    while True:
        try:
            used = int((node / "memory.current").read_text())
            for name in ("memory.max", "memory.high"):
                try:
                    limit = int((node / name).read_text())
                    remaining.append(max(0, limit - used))
                except (OSError, ValueError):
                    pass
        except (OSError, ValueError):
            pass
        if node == root:
            break
        node = node.parent
    return min(remaining) if remaining else None


def direct_io_capability(path: str) -> tuple[bool, str | None]:
    """Probe an aligned read before auto selection yields any tensors."""
    if not hasattr(os, "O_DIRECT"):
        return False, "O_DIRECT is unavailable on this platform"
    fd = -1
    try:
        with open(path, "rb", buffering=0) as header_file:
            header, _ = _read_header(header_file.fileno())
        unsupported = {
            info["dtype"]
            for name, info in header.items()
            if name != "__metadata__" and info["dtype"] not in _DTYPES
        }
        if unsupported:
            return False, f"unsupported direct tensor formats: {sorted(unsupported)}"
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        size = os.fstat(fd).st_size
        if size:
            buffer, _ = _read_run(fd, 0, min(_ALIGN, size))
            buffer.close()
    except (OSError, EOFError) as error:
        return False, f"aligned O_DIRECT read is unavailable: {error}"
    finally:
        if fd >= 0:
            os.close(fd)
    return True, None


def direct_io_weights(
    path: str,
    keep: Callable[[str], bool],
    sharing: RunSharing | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield the tensors of one safetensors shard for which `keep` is True,
    read with O_DIRECT in coalesced runs in file order. With *sharing* only
    the group's leader reads; the others receive each run from it."""
    with open(path, "rb", buffering=0) as header_file:
        header, data_start = _read_header(header_file.fileno())

    tensors = []
    for name, info in header.items():
        if name == "__metadata__" or not keep(name):
            continue
        if info["dtype"] not in _DTYPES:
            raise ValueError(
                f"Direct I/O loading does not know safetensors dtype "
                f"{info['dtype']!r} of {name!r} in {path}"
            )
        begin, end = info["data_offsets"]
        tensors.append((data_start + begin, data_start + end, name, info))
    tensors.sort(key=lambda entry: entry[0])
    runs = _coalesce(tensors)
    reads = sharing is None or sharing.is_leader
    fd = -1
    open_error: OSError | None = None
    if reads:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError as error:
            if sharing is None:
                raise
            open_error = error

    try:
        if sharing is not None:
            try:
                _check_same_runs(path, runs, sharing, leader_ok=open_error is None)
            except RuntimeError as error:
                raise error from open_error
        for index, last, run_start, run_end in runs:
            if run_start == run_end:
                # Only empty tensors: nothing to read, and a zero-size
                # buffer cannot be mapped.
                for _, _, name, info in tensors[index:last]:
                    yield name, torch.empty(info["shape"], dtype=_DTYPES[info["dtype"]])
                continue
            run_error: Exception | None = None
            try:
                if reads:
                    buffer, buffer_start = _read_run(fd, run_start, run_end)
                else:
                    buffer, buffer_start = _run_buffer(run_start, run_end)
            except (OSError, EOFError) as error:
                if sharing is None:
                    raise
                run_error = error
            if sharing is not None:
                if not _group_ready(run_error is None, sharing):
                    raise RuntimeError(
                        f"Direct I/O run sharing: a rank of the group could not "
                        f"read or buffer {path}"
                    ) from run_error
                dist.broadcast(
                    torch.frombuffer(buffer, dtype=torch.uint8),
                    src=sharing.leader,
                    group=sharing.group,
                )
            for begin, end, name, info in tensors[index:last]:
                dtype = _DTYPES[info["dtype"]]
                shape = info["shape"]
                if end == begin:
                    yield name, torch.empty(shape, dtype=dtype)
                    continue
                raw = torch.frombuffer(
                    buffer,
                    dtype=torch.uint8,
                    count=end - begin,
                    offset=begin - buffer_start,
                )
                yield name, raw.view(dtype).reshape(shape)
    finally:
        if fd >= 0:
            os.close(fd)
