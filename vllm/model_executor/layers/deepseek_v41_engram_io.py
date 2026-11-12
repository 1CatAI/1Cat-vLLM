# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One node-level Engram reader and bounded shared Engram row buffers.

Rows come from the GGUF sidecar (IQ4_NL, 144 bytes) or, for the original
checkpoint (VLLM_V41_NATIVE=1), from a directory of repacked tables: one
``engram-LL.rows`` file of 264-byte rows (256 E4M3 codes + 8 E8M0 scales) and
an ``engram-LL.json`` description per Engram layer.
"""

import atexit
import hashlib
import json
import os
import socket
import struct
from dataclasses import asdict, dataclass

import gguf
import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton
from vllm.utils.shared_host import SharedHostArena

from .quantization.gguf_expert_cache import _native
from .quantization.gguf_shared_protocol import _fail_if, wait_shared_sequence


@dataclass(frozen=True)
class EngramTableSpec:
    path: str
    offset: int
    rows: int
    row_bytes: int = 144
    # Native rows: decoded values carry 2^row_bias (exact; the projection's
    # scales carry 2^-row_bias).
    row_bias: int = 0


NATIVE_ROW_BYTES = 264


def native_row_bias(meta):
    """Power-of-two bias of the FP16 row decode (VLLM_V41_ENGRAM_ROW_BIAS)."""
    value = envs.VLLM_V41_ENGRAM_ROW_BIAS
    if value is not None:
        return int(value)
    # No bias when FP16 already holds every row value exactly: the largest
    # (448 * 2^max) stays below 2^15 and the smallest subnormal code
    # (2^-9 * 2^min) above 2^-24. The projection's scales then stay as in the
    # checkpoint (and inside the v41q8 decode window).
    low, high = int(meta["scale_min"]) - 127, int(meta["scale_max"]) - 127
    if high + 9 <= 15 and low - 9 >= -24:
        return 0
    return int(meta["row_bias"])


def engram_table_specs(path):
    if os.path.isdir(path):
        result: list[EngramTableSpec | None] = []
        for layer in (1, 14):
            description = os.path.join(path, f"engram-{layer:02d}.json")
            if not os.path.exists(os.path.join(path, f"engram-{layer:02d}.rows")):
                result.append(None)  # a stage without this layer need not keep it
                continue
            with open(description) as file:
                meta = json.load(file)
            if meta["row_bytes"] != NATIVE_ROW_BYTES or meta["nan_codes"]:
                raise ValueError(f"V4.1 Engram rows of layer {layer} are invalid")
            result.append(
                EngramTableSpec(
                    os.path.abspath(os.path.join(path, f"engram-{layer:02d}.rows")),
                    0,
                    int(meta["rows"]),
                    NATIVE_ROW_BYTES,
                    native_row_bias(meta),
                )
            )
        return tuple(result)
    reader = gguf.GGUFReader(path, mode="r")
    expected = ("blk.1.engram_embd.weight", "blk.14.engram_embd.weight")
    tensors = {t.name: t for t in reader.tensors}
    if set(tensors) != set(expected):
        raise ValueError("V4.1 Engram sidecar must contain exactly layers 1 and 14")
    result = []
    for name in expected:
        t = tensors[name]
        if int(t.tensor_type) != 20 or len(t.shape) != 2 or int(t.shape[0]) != 256:
            raise ValueError("V4.1 shared Engram requires IQ4_NL rows of width 256")
        result.append(
            EngramTableSpec(os.path.abspath(path), int(t.data_offset), int(t.shape[1]))
        )
    return tuple(result)


def _row_bytes(specs):
    sizes = {s.row_bytes for s in specs}
    if len(sizes) != 1 or sizes - {144, NATIVE_ROW_BYTES}:
        raise ValueError("V4.1 Engram tables must share one supported row size")
    return sizes.pop()


def _fingerprint(specs, maximum):
    value = dict(
        tables=[asdict(s) for s in specs], maximum=maximum, row_bytes=_row_bytes(specs)
    )
    return int.from_bytes(
        hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()[:8], "big"
    ) & ((1 << 63) - 1)


class SharedEngramOwner:
    def __init__(self, specs, max_tokens=2048, timeout_ns=60_000_000_000):
        if len(specs) != 2 or not 1 <= max_tokens <= 16384:
            raise ValueError("V4.1 Engram requires two bounded table channels")
        maximum = max_tokens * 24
        row_bytes = _row_bytes(specs)
        native = _native()
        words, channel_words, _ = native.shared_engram_control_layout(maximum)
        self.data = self.control = self.reader = None
        try:
            self.data = SharedHostArena.create(
                2 * maximum * row_bytes, "vllm-engram-rows"
            )
            self.control = SharedHostArena.create(words * 8, "vllm-engram-control")
            files = []
            try:
                for spec in specs:
                    files.append(os.open(spec.path, os.O_RDONLY | os.O_CLOEXEC))
                self.reader = native.SharedEngramReader(
                    self.data.address,
                    self.data.size,
                    self.control.address,
                    maximum,
                    row_bytes,
                    [
                        native.SharedEngramTable(fd, s.offset, s.rows)
                        for fd, s in zip(files, specs)
                    ],
                    timeout_ns,
                )
            finally:
                for fd in files:
                    os.close(fd)
            for channel in range(2):
                struct.pack_into(
                    "=q",
                    self.control.buffer,
                    (channel * channel_words + 6) * 8,
                    _fingerprint(specs, maximum),
                )
        except BaseException:
            self.close()
            raise

    def send(self, connection: socket.socket):
        if self.data is None or self.control is None:
            raise RuntimeError("Shared Engram owner is closed")
        self.data.send(connection)
        self.control.send(connection)

    def close(self):
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.control is not None:
            self.control.close()
            self.control = None
        if self.data is not None:
            self.data.close()
            self.data = None


@triton.jit
def _can_publish(
    control,
    counters,
    CHANNEL: tl.constexpr,
    CHANNEL_WORDS: tl.constexpr,
    RANK_WORDS: tl.constexpr,
    RANK: tl.constexpr,
):
    channel = control + CHANNEL * CHANNEL_WORDS
    request = channel + 8 + RANK * RANK_WORDS
    _fail_if(
        (tl.load(request, volatile=True) >= tl.load(counters + CHANNEL))
        | tl.load(channel + 1, volatile=True)
    )


@triton.jit(do_not_specialize=["COUNT"])
def _copy_request(
    ids,
    control,
    COUNT,
    CHANNEL: tl.constexpr,
    CHANNEL_WORDS: tl.constexpr,
    RANK_WORDS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(ids + (i // 24) * 48 + CHANNEL * 24 + i % 24, i < COUNT, 0)
    request = control + CHANNEL * CHANNEL_WORDS + 8 + RANK * RANK_WORDS
    tl.store(request + 8 + i, value, i < COUNT)


@triton.jit(do_not_specialize=["COUNT"])
def _publish(
    control,
    counters,
    COUNT,
    CHANNEL: tl.constexpr,
    CHANNEL_WORDS: tl.constexpr,
    RANK_WORDS: tl.constexpr,
    RANK: tl.constexpr,
):
    request = control + CHANNEL * CHANNEL_WORDS + 8 + RANK * RANK_WORDS
    tl.store(request + 2, COUNT)
    tl.atomic_xchg(request, tl.load(counters + CHANNEL), sem="release", scope="sys")


@triton.jit(do_not_specialize=["COUNT"])
def _wait(
    control,
    counters,
    statistics,
    COUNT,
    CHANNEL: tl.constexpr,
    CHANNEL_WORDS: tl.constexpr,
):
    channel = control + CHANNEL * CHANNEL_WORDS
    sequence = tl.load(counters + CHANNEL)
    wait_shared_sequence(channel, channel + 1, sequence)
    # The CPU rewrites this field on every request. Use a volatile load so
    # the new readiness tag cannot be paired with a cached previous count.
    _fail_if(tl.load(channel + 5, volatile=True) != COUNT)
    lanes = tl.arange(0, 4)
    values = tl.load(channel + 1 + lanes, volatile=True)
    values = tl.where(lanes == 0, 1, values)
    target = statistics + CHANNEL * 4 + lanes
    tl.store(target, tl.load(target) + values)


@triton.jit
def _consume(
    control,
    counters,
    CHANNEL: tl.constexpr,
    CHANNEL_WORDS: tl.constexpr,
    RANK_WORDS: tl.constexpr,
    RANK: tl.constexpr,
):
    request = control + CHANNEL * CHANNEL_WORDS + 8 + RANK * RANK_WORDS
    sequence = tl.load(counters + CHANNEL)
    tl.atomic_xchg(request + 1, sequence, sem="release", scope="sys")
    tl.store(counters + CHANNEL, sequence + 1)


@triton.jit
def _decode_native_rows(raw, out, BIAS: tl.constexpr):
    """One 264-byte row: 256 E4M3 codes then 8 E8M0 scales (one per 32)."""
    row = tl.program_id(0)
    lanes = tl.arange(0, 256)
    codes = tl.load(raw + row * 264 + lanes).to(tl.uint32)
    exponent = tl.load(raw + row * 264 + 256 + lanes // 32).to(tl.int32) + BIAS
    # E4M3 bits -> FP32 exactly (subnormals included); NaN codes are refused
    # when the tables are repacked.
    sign = (codes & 0x80) << 24
    magnitude = codes & 0x7F
    normal = (((magnitude + (120 << 3)) << 20) | sign).to(tl.float32, bitcast=True)
    subnormal = tl.where(sign != 0, -1.0, 1.0) * magnitude.to(tl.float32) * (2.0**-9)
    value = tl.where(magnitude >= 8, normal, subnormal)
    scale = (exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    tl.store(out + row * 256 + lanes, (value * scale).to(tl.float16))


def decode_native_rows(raw, bias):
    """[n, 264] native rows -> [n, 256] FP16 values times 2^bias."""
    out = torch.empty(raw.shape[0], 256, device=raw.device, dtype=torch.float16)
    _decode_native_rows[(raw.shape[0],)](raw, out, BIAS=int(bias), num_warps=4)
    return out


class SharedEngramRank:
    """Per-process CUDA registration; each GPU dequantizes the shared raw rows."""

    def __init__(self, data, control, rank, specs, max_tokens=2048, channels=(0, 1)):
        if rank not in range(4) or torch.accelerator.current_device_index() != rank:
            raise ValueError("Select the TP rank's CUDA device before mapping Engram")
        self.data, self.control, self.rank = data, control, rank
        self.maximum = max_tokens * 24
        self.row_bytes = _row_bytes(specs)
        self.row_bias = tuple(s.row_bias for s in specs)
        # A pipeline stage publishes and consumes only its own Engram layers.
        self.channels = tuple(channels)
        if not self.channels or set(self.channels) - {0, 1}:
            raise ValueError("V4.1 Engram channels must be a subset of (0, 1)")
        self.native = _native()
        words, self.channel_words, self.rank_words = (
            self.native.shared_engram_control_layout(self.maximum)
        )
        if data.size < 2 * self.maximum * self.row_bytes or control.size < words * 8:
            raise ValueError("Engram mapping is smaller than its declared capacity")
        self._registered = []
        self.host = self.control_host = self.mapped = self.control_view = None
        self.ready = False
        try:
            self.host = torch.frombuffer(data.buffer, dtype=torch.uint8)
            self.control_host = torch.frombuffer(control.buffer, dtype=torch.int64)
            for channel in range(2):
                if int(
                    self.control_host[channel * self.channel_words + 6]
                ) != _fingerprint(specs, self.maximum):
                    raise ValueError(
                        "Engram ranks disagree about the sidecar or capacity"
                    )
            with control.registration_guard():
                for name, tensor, arena in (
                    ("rows", self.host, data),
                    ("control", self.control_host, control),
                ):
                    pointer = tensor.data_ptr()
                    size = tensor.numel() * tensor.element_size()
                    try:
                        self.native.host_register(pointer, size)
                    except RuntimeError as error:
                        raise RuntimeError(
                            f"Engram {name} registration failed on TP{rank}: "
                            f"pointer={pointer:#x}, bytes={size}, "
                            f"identity={vars(arena.identity)}"
                        ) from error
                    self._registered.append(pointer)
            self.mapped = self.native.mapped_view(self.host, rank)
            self.control_view = self.native.mapped_view(self.control_host, rank)
            self.counters = torch.ones(2, device=rank, dtype=torch.int64)
            # Requests, unique file rows, file bytes, and CPU reader request ns.
            # Counts are replicated observations, not four separate file reads.
            self.statistics = torch.zeros(2, 4, device=rank, dtype=torch.int64)
        except BaseException:
            self.close()
            raise

    def _meta(self, channel):
        return dict(
            CHANNEL=channel,
            CHANNEL_WORDS=self.channel_words,
            RANK_WORDS=self.rank_words,
            RANK=self.rank,
        )

    def warmup(self):
        from .quantization.gguf import dequantize_gguf_rows

        dummy = torch.zeros(1, 2, 24, device=self.rank, dtype=torch.int64)
        for channel in self.channels:
            meta = self._meta(channel)
            _can_publish.warmup(self.control_view, self.counters, **meta, grid=(1,))
            _copy_request.warmup(
                dummy, self.control_view, COUNT=24, BLOCK=256, **meta, grid=(1,)
            )
            _publish.warmup(
                self.control_view, self.counters, COUNT=24, **meta, grid=(1,)
            )
            _wait.warmup(
                self.control_view,
                self.counters,
                self.statistics,
                COUNT=24,
                CHANNEL=channel,
                CHANNEL_WORDS=self.channel_words,
                grid=(1,),
            )
            _consume.warmup(self.control_view, self.counters, **meta, grid=(1,))
        if self.row_bytes == NATIVE_ROW_BYTES:
            decode_native_rows(
                torch.zeros(1, NATIVE_ROW_BYTES, device=self.rank, dtype=torch.uint8),
                0,
            )
        else:
            dequantize_gguf_rows(
                torch.zeros(1, 144, device=self.rank, dtype=torch.uint8),
                20,
                256,
                torch.float16,
                True,
            )
        torch.accelerator.synchronize(self.rank)
        self.ready = True

    def prefetch(self, hashes):
        if (
            not self.ready
            or hashes.ndim != 3
            or hashes.shape[1:] != (2, 24)
            or hashes.dtype != torch.int64
            or hashes.device != torch.device("cuda", self.rank)
            or not hashes.is_contiguous()
            or not 0 < hashes.shape[0] * 24 <= self.maximum
        ):
            raise ValueError("Engram prefetch requires warmed, contiguous GPU hashes")
        count = hashes.shape[0] * 24
        for channel in self.channels:
            meta = self._meta(channel)
            _can_publish[(1,)](self.control_view, self.counters, **meta)
            _copy_request[(triton.cdiv(count, 256),)](
                hashes, self.control_view, COUNT=count, BLOCK=256, **meta
            )
            _publish[(1,)](self.control_view, self.counters, COUNT=count, **meta)

    def lookup(self, channel, tokens):
        from .quantization.gguf import dequantize_gguf_rows

        if (
            not self.ready
            or self.mapped is None
            or channel not in self.channels
            or not 0 < tokens * 24 <= self.maximum
        ):
            raise ValueError("Invalid Engram table or row count")
        count = tokens * 24
        _wait[(1,)](
            self.control_view,
            self.counters,
            self.statistics,
            COUNT=count,
            CHANNEL=channel,
            CHANNEL_WORDS=self.channel_words,
        )
        size = self.row_bytes
        first = channel * self.maximum * size
        raw = self.mapped[first : first + count * size].view(count, size)
        if size == NATIVE_ROW_BYTES:
            result = decode_native_rows(raw, self.row_bias[channel])
        else:
            result = dequantize_gguf_rows(raw, 20, 256, torch.float16, True)
        _consume[(1,)](self.control_view, self.counters, **self._meta(channel))
        return result.view(tokens, 6144)

    def close(self):
        try:
            if self._registered:
                torch.accelerator.synchronize(self.rank)
        finally:
            self.mapped = self.control_view = None
            for pointer in reversed(self._registered):
                self.native.host_unregister(pointer)
            self._registered.clear()
            self.host = self.control_host = None
            self.ready = False


class SharedEngramIO:
    def __init__(self, path, max_tokens=2048, group=None, channels=(0, 1)):
        from .quantization.gguf_shared_storage import _join_shared_group

        if group is None:
            from vllm.distributed import get_tp_group

            group = get_tp_group()
        if group.world_size != 4:
            raise ValueError("V4.1 shared Engram currently requires local TP4")
        specs = group.broadcast_object(
            engram_table_specs(path) if group.rank_in_group == 0 else None, src=0
        )
        if any(specs[c] is None for c in channels):
            raise ValueError(f"V4.1 Engram tables for channels {channels} are missing")
        if set(channels) != {0, 1}:
            # The reader keeps two channels; one this stage never publishes
            # reuses the active table's spec, so its file need not exist here.
            active = specs[channels[0]]
            specs = tuple(specs[c] if c in channels else active for c in range(2))
        self.owner, self.data, self.control = _join_shared_group(
            None, group, owner_factory=lambda: SharedEngramOwner(specs, max_tokens)
        )
        self.runtime = None
        try:
            self.specs = specs
            self.runtime = SharedEngramRank(
                self.data,
                self.control,
                group.rank_in_group,
                specs,
                max_tokens,
                channels,
            )
            self.runtime.warmup()
            torch.distributed.barrier(group=group.cpu_group)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        if self.owner is not None:
            self.owner.close()
            self.owner = self.data = self.control = None
        else:
            if self.control is not None:
                self.control.close()
                self.control = None
            if self.data is not None:
                self.data.close()
                self.data = None
