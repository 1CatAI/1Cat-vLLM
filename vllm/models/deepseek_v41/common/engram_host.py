# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngramHostService: host-side Engram row store for one TP rank of one PP stage (lane L-ENGRAM, PORT_DESIGN §3.6).

Storage (DECISIONS D4/D5 -- no repacked row file): the 256-byte FP8 rows are read IN PLACE from the official
safetensors shards (``layers.{L}.engram.embed.weight``; tensor offsets from the safetensors header; 1 row in 16
straddles a 4 KiB page because the tensors start at byte 664 / 672) through the page cache, which is the warm tier.
The rank's UE8M0 scales (``embed.scale``, 8 B per row, only its own sub-tables: ~0.77 GB per layer at TP4) are loaded
once into page-locked RAM and appended to every row on the CPU, so each staged row is the 264-byte
``[256 x E4M3 | 8 x UE8M0]`` record the module decodes.

Per step (hooks called by L-CORE, §3.6):
  begin_step(plan)   worker thread, before the PP receive: hash the scheduled tokens (CPU, numpy), dedup the
                     (layer, row) keys, submit ONE high-priority gather to the C++ reader (csrc/engram_rows.cpp)
                     into pinned slot k; new/resumed requests also queue a low-priority page-cache warm-up of the
                     rest of their prompt (admission prefetch). Never waits for I/O.
  bind_batch(layout) runner thread, after _prepare_inputs: resolves async-PP sampled tokens, writes the runner-order
                     index map (padding -> the all-zero row 0) and, when the gather has already finished, enqueues the
                     H2D copy on the service's copy stream. Never waits for I/O.
  wait_rows(layer)   inside the module's eager break: finishes the gather and the H2D if still pending, makes the
                     current stream wait for it, expands unique rows -> the static dense buffer [T_pad, n_sub, 264].
  dummy_rows(...)    AM-2: profiling / CUDA-graph-capture forwards (ForwardContext.is_dummy_run) get zero rows.
Two pinned/device slots alternate so step k+1's gather and copy overlap step k's compute; the copy into a device
slot waits for the event recorded after that slot's previous expansion.
"""

from __future__ import annotations

import functools
import json
import mmap
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import (
    ENGRAM_HEAD_DIM,
    ENGRAM_ROW_BYTES,
    EngramBatchLayout,
    EngramStepPlan,
)
from vllm.models.deepseek_v41.common.engram import (
    ROW_SCALE_BYTES,
    ROW_VALUE_BYTES,
    EngramHasher,
    EngramLayout,
    load_compressed_token_map,
    row_bias_for_exponents,
)

logger = init_logger(__name__)

O_DIRECT_KNOB = "VLLM_DS41_ENGRAM_O_DIRECT"                # bypass the page cache (cold-path measurements)
PREFETCH_KNOB = "VLLM_DS41_ENGRAM_PREFETCH"                # admission prefetch of whole prompts
REQUIRE_VERIFIED_KNOB = "VLLM_DS41_ENGRAM_REQUIRE_VERIFIED"  # shard needs its <file>.sha256-ok sidecar
BUILD_DIR_KNOB = "VLLM_DS41_ENGRAM_BUILD_DIR"              # torch extension build directory
IO_THREADS_KNOB = "VLLM_DS41_ENGRAM_IO_THREADS"            # overrides the ctor's io_threads (CPU budget per rank)
N_SLOTS = 2
_LATENCY_WINDOW = 4096


# ------------------------------------------------------------------------------------------------ extension
@functools.lru_cache(maxsize=1)
def load_engram_rows_extension() -> Any:
    """Build/load csrc/engram_rows.cpp as a runtime torch extension (no CMakeLists entry at P2)."""
    src = Path(__file__).resolve().parents[4] / "csrc" / "engram_rows.cpp"
    if not src.is_file():
        raise FileNotFoundError(f"Engram row reader source not found at {src} (needs a source checkout of 1Cat-vLLM)")
    default_dir = Path.home() / ".cache" / "vllm" / "ds41_engram_rows" / torch.__version__.replace("+", "_")
    build_dir = Path(knobs.env_str(BUILD_DIR_KNOB, str(default_dir)))
    build_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ninja") is None:
        venv_ninja = Path(sys.executable).parent / "ninja"
        if not venv_ninja.is_file():
            raise RuntimeError("building the Engram row reader needs `ninja` on PATH (or next to the interpreter)")
        os.environ["PATH"] = f"{venv_ninja.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    from torch.utils.cpp_extension import load

    return load(name="ds41_engram_rows", sources=[str(src)], build_directory=str(build_dir),
                extra_cflags=["-O3", "-std=c++17"], verbose=False)


# ------------------------------------------------------------------------------------------------ shards
@dataclass(frozen=True)
class EngramTableLocation:
    layer_id: int
    weight_path: str      # realpath of the shard holding embed.weight
    weight_offset: int    # absolute byte offset of row 0
    scale_path: str
    scale_offset: int
    num_rows: int


def read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """(header dict, absolute offset of the data section)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (short header length)")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 << 20:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        header = json.loads(f.read(n))
    return header, 8 + n


def locate_engram_tables(row_dir: str, layout: EngramLayout, layers: tuple[int, ...]) -> list[EngramTableLocation]:
    """Find ``layers.{L}.engram.embed.{weight,scale}`` under ``row_dir`` (a checkpoint dir with an index json, or
    a directory of shards) and return absolute byte offsets, validating dtype and shape against the layout."""
    index = os.path.join(row_dir, "model.safetensors.index.json")
    wanted = {f"layers.{lid}.engram.embed.{part}" for lid in layers for part in ("weight", "scale")}
    files: dict[str, str] = {}
    if os.path.isfile(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        for name in wanted:
            if name not in weight_map:
                raise KeyError(f"{index} has no entry for {name}")
            files[name] = os.path.join(row_dir, weight_map[name])
    else:
        for p in sorted(Path(row_dir).glob("*.safetensors")):
            header, _ = read_safetensors_header(str(p))
            for name in wanted & header.keys():
                files[name] = str(p)
        missing = wanted - files.keys()
        if missing:
            raise FileNotFoundError(f"Engram tables {sorted(missing)} not found in any *.safetensors under {row_dir}")
    out = []
    for lid in layers:
        rows = layout.num_embeddings[layout.layer_index(lid)]
        locs = {}
        for part, dtype, width in (("weight", "F8_E4M3", ROW_VALUE_BYTES), ("scale", "F8_E8M0", ROW_SCALE_BYTES)):
            name = f"layers.{lid}.engram.embed.{part}"
            path = os.path.realpath(files[name])
            header, data_start = read_safetensors_header(path)
            meta = header[name]
            if meta["dtype"] != dtype or list(meta["shape"]) != [rows, width]:
                raise ValueError(f"{path}: {name} is {meta['dtype']} {meta['shape']}, expected {dtype} {[rows, width]}")
            a, b = meta["data_offsets"]
            if b - a != rows * width:
                raise ValueError(f"{path}: {name} spans {b - a} bytes, expected {rows * width}")
            if data_start + b > os.path.getsize(path):
                raise ValueError(f"{path}: {name} ends at {data_start + b}, beyond the file ({os.path.getsize(path)})")
            locs[part] = (path, data_start + a)
        out.append(EngramTableLocation(lid, locs["weight"][0], locs["weight"][1], locs["scale"][0],
                                       locs["scale"][1], rows))
    return out


def _check_verified(path: str) -> None:
    sidecar = path + ".sha256-ok"
    if not os.path.isfile(sidecar):
        raise RuntimeError(f"Engram shard {path} has no {os.path.basename(sidecar)} sidecar: refusing to serve rows "
                           f"from an unverified shard (DECISIONS D5; set {REQUIRE_VERIFIED_KNOB}=0 only for tests)")


# ------------------------------------------------------------------------------------------------ host memory
class _PinnedHostBuffer:
    """Exact-size anonymous mapping, THP-advised, page-locked with cudaHostRegister (RLIMIT_MEMLOCK does not cap it;
    torch's pinned allocator would round 0.77 GB up to 1 GiB)."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self._map = mmap.mmap(-1, max(nbytes, 1), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        if hasattr(mmap, "MADV_HUGEPAGE"):
            self._map.madvise(mmap.MADV_HUGEPAGE)
        self.array = np.frombuffer(self._map, dtype=np.uint8, count=nbytes)
        self.tensor = torch.from_numpy(self.array)
        self._registered = False

    def register(self) -> None:
        rc = torch.cuda.cudart().cudaHostRegister(self.tensor.data_ptr(), self.nbytes, 0)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister of {self.nbytes} B Engram scale table failed: {rc}")
        self._registered = True

    def release(self) -> None:
        if self._registered:
            rc = torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr())
            self._registered = False
            if int(rc) != 0:
                raise RuntimeError(f"cudaHostUnregister of the Engram scale table failed: {rc}")


def _pread_into(fd: int, dst: np.ndarray, offset: int, path: str) -> None:
    view = memoryview(dst)
    done = 0
    chunk = 64 << 20
    while done < dst.nbytes:
        n = os.preadv(fd, [view[done:done + chunk]], offset + done)
        if n <= 0:
            raise OSError(f"short read of Engram scales from {path} at {offset + done}: {n}")
        done += n


# ------------------------------------------------------------------------------------------------ request state
class _History:
    """Compressed token ids of one request by absolute position.

    Positions < prompt_len (the prompt, or every committed token of a resumed request) are never dropped; a write
    beyond them ends the known history at its last position, so tokens of rejected speculative drafts can never be
    mistaken for known ones (spec-decode rollback)."""

    def __init__(self, ids: np.ndarray) -> None:
        self.buf = np.empty(max(16, int(ids.shape[0] * 1.25) + 16), dtype=np.int64)
        self.buf[: ids.shape[0]] = ids
        self.length = int(ids.shape[0])
        self.prompt_len = self.length

    def write(self, start: int, ids: np.ndarray) -> None:
        if start > self.length:
            raise ValueError(f"history gap: write at {start}, only {self.length} positions known")
        end = start + ids.shape[0]
        if end > self.buf.shape[0]:
            grown = np.empty(max(end + 16, self.buf.shape[0] * 2), dtype=np.int64)
            grown[: self.length] = self.buf[: self.length]
            self.buf = grown
        self.buf[start:end] = ids
        self.length = max(end, self.prompt_len)

    def view(self) -> np.ndarray:
        return self.buf[: self.length]


@dataclass
class _ReqSlot:
    off: int          # first token of this request in the step's scheduler-order token space
    n: int
    start_pos: int
    resolved: bool


@dataclass
class _Step:
    step_id: int
    slot: int
    reqs: dict[str, _ReqSlot]
    total_tokens: int
    inv: np.ndarray                     # [total_tokens, L, S] int32 unique-row index (+1; 0 = zero row)
    n_unique: int                       # unique rows written into the slot (row 0 excluded)
    tickets: list[int] = field(default_factory=list)
    t_begin: float = 0.0
    layout: EngramBatchLayout | None = None
    h2d_issued: bool = False
    expanded: bool = False
    t_pad: int = 0


class EngramHostService:
    """See the module docstring. One instance per TP rank of a stage that owns Engram layers."""

    def __init__(self, hf_config, layers: tuple[int, ...], tp_rank: int, tp_size: int, row_dir: str,
                 tokenizer_path: str, max_num_batched_tokens: int, device: torch.device,
                 io_threads: int = 4) -> None:
        try:
            self._init(hf_config, layers, tp_rank, tp_size, row_dir, tokenizer_path, max_num_batched_tokens,
                       device, io_threads)
        except BaseException:
            # a failing ctor must not leave scale tables cudaHostRegister'ed (or reader threads running): the
            # mapping is freed by GC while still registered and a later registration of a recycled address fails
            # (cudaErrorHostMemoryAlreadyRegistered). Re-raise unchanged.
            self.shutdown()
            raise

    def _init(self, hf_config, layers: tuple[int, ...], tp_rank: int, tp_size: int, row_dir: str,
              tokenizer_path: str, max_num_batched_tokens: int, device: torch.device, io_threads: int) -> None:
        t0 = time.perf_counter()
        self.layout = EngramLayout.from_hf_config(hf_config)
        if layout_bad := [lid for lid in layers if lid not in self.layout.layer_ids]:
            raise ValueError(f"layers {layout_bad} are not Engram layers {self.layout.layer_ids}")
        if not layers:
            raise ValueError("EngramHostService needs at least one Engram layer")
        if self.layout.head_dim != ENGRAM_HEAD_DIM:
            raise ValueError(f"Engram head_dim {self.layout.head_dim} != contract {ENGRAM_HEAD_DIM}")
        self.layers = tuple(layers)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.subtables = self.layout.subtables_for_rank(tp_rank, tp_size)
        self.n_sub = len(self.subtables)
        self.device = torch.device(device)
        self.max_tokens = int(max_num_batched_tokens)
        self.o_direct = knobs.env_bool(O_DIRECT_KNOB, False)
        self.prefetch_enabled = knobs.env_bool(PREFETCH_KNOB, True) and not self.o_direct
        token_map = load_compressed_token_map(tokenizer_path, self.layout.compressed_vocab_size)
        self.hasher = EngramHasher(self.layout, token_map, self.layers, self.subtables)

        self.tables = locate_engram_tables(row_dir, self.layout, self.layers)
        if knobs.env_bool(REQUIRE_VERIFIED_KNOB, True):
            for loc in self.tables:
                _check_verified(loc.weight_path)
                _check_verified(loc.scale_path)

        # -- per layer: weight file index, absolute row offsets, rank-local scale tables (pinned, exact size)
        L, S = len(self.layers), self.n_sub
        lis = [self.layout.layer_index(lid) for lid in self.layers]
        self._sub_offset = np.array([[self.layout.offsets[li][s] for s in self.subtables] for li in lis], np.int64)
        sub_rows = np.array([[self.layout.primes[li][s] for s in self.subtables] for li in lis], np.int64)
        self._sub_rows = sub_rows
        self._scale_base = np.concatenate([np.zeros((L, 1), np.int64), np.cumsum(sub_rows, axis=1)[:, :-1]], axis=1)
        self._weight_offset = np.array([loc.weight_offset for loc in self.tables], np.int64)
        self._scales: list[_PinnedHostBuffer] = []
        self._row_bias: dict[int, int] = {}
        self.scale_exponent_range: dict[int, tuple[int, int]] = {}
        for l_idx, loc in enumerate(self.tables):
            buf = _PinnedHostBuffer(int(sub_rows[l_idx].sum()) * ROW_SCALE_BYTES)
            fd = os.open(loc.scale_path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                for j, s in enumerate(self.subtables):
                    off = loc.scale_offset + self._sub_offset[l_idx, j] * ROW_SCALE_BYTES
                    n = int(sub_rows[l_idx, j]) * ROW_SCALE_BYTES
                    dst_a = int(self._scale_base[l_idx, j]) * ROW_SCALE_BYTES
                    _pread_into(fd, buf.array[dst_a:dst_a + n], int(off), loc.scale_path)
                    os.posix_fadvise(fd, int(off), n, os.POSIX_FADV_DONTNEED)  # the pinned copy is the only one kept
            finally:
                os.close(fd)
            e_min, e_max = int(buf.array.min()), int(buf.array.max())
            if e_max == 255:
                raise ValueError(f"layer {loc.layer_id}: Engram scales hold UE8M0 NaN (0xFF)")
            self.scale_exponent_range[loc.layer_id] = (e_min - 127, e_max - 127)
            self._row_bias[loc.layer_id] = row_bias_for_exponents(e_min, e_max)
            buf.register()
            self._scales.append(buf)

        io_threads = knobs.env_int(IO_THREADS_KNOB, int(io_threads), minimum=1, maximum=64)
        ext = load_engram_rows_extension()
        paths = [loc.weight_path for loc in self.tables]
        uniq_paths = sorted(set(paths))
        self._file_of_layer = np.array([uniq_paths.index(p) for p in paths], np.int32)
        self._reader = ext.RowReader(uniq_paths, int(io_threads), self.o_direct)
        self._table_ids = [self._reader.add_table(b.tensor, ROW_SCALE_BYTES) for b in self._scales]
        self._table_ids_arr = np.array(self._table_ids, np.int32)
        self.io_threads = int(io_threads)

        # -- staging: 2 slots of unique rows (row 0 = all zeros = padding) + runner-order index maps
        self.max_unique = self.max_tokens * L * S + 1
        # explicit CPU: vLLM constructs the model (and this service) under a `with torch.device("cuda")` context
        self._host_rows = torch.zeros((N_SLOTS, self.max_unique, ENGRAM_ROW_BYTES), dtype=torch.uint8,
                                      device="cpu").pin_memory()
        self._host_idx = torch.zeros((N_SLOTS, L * self.max_tokens * S), dtype=torch.int32, device="cpu").pin_memory()
        self._dev_rows = torch.zeros((N_SLOTS, self.max_unique, ENGRAM_ROW_BYTES), dtype=torch.uint8,
                                     device=self.device)
        self._dev_idx = torch.zeros((N_SLOTS, L * self.max_tokens * S), dtype=torch.int32, device=self.device)
        self._dense = torch.zeros((L, self.max_tokens, S, ENGRAM_ROW_BYTES), dtype=torch.uint8, device=self.device)
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._h2d_done = [torch.cuda.Event() for _ in range(N_SLOTS)]
        self._consumed = [torch.cuda.Event() for _ in range(N_SLOTS)]
        self._h2d_recorded = [False] * N_SLOTS
        self._consumed_recorded = [False] * N_SLOTS
        self._next_slot = 0

        self._hist: dict[str, _History] = {}
        self._step: _Step | None = None
        self._stats = dict(steps=0, dummy_steps=0, lookups=0, unique_rows=0, hits=0, misses=0, pages=0, prefetch_rows=0,
                           wait_blocked_s=0.0, h2d_bytes=0)
        self._lat_ms: list[float] = []
        self.init_seconds = time.perf_counter() - t0
        logger.info("Engram host service: layers %s, TP rank %d/%d sub-tables %s, scales %.2f GiB page-locked, "
                    "row bias %s, scale exponents %s, %d I/O threads (%s), staging %.1f MiB/slot, init %.1f s",
                    self.layers, tp_rank, tp_size, self.subtables, self.pinned_scale_bytes / 2**30, self._row_bias,
                    self.scale_exponent_range, io_threads, "O_DIRECT" if self.o_direct else "buffered",
                    self.max_unique * ENGRAM_ROW_BYTES / 2**20, self.init_seconds)

    # ---------------------------------------------------------------- properties used by the module / tests
    @property
    def pinned_scale_bytes(self) -> int:
        return sum(b.nbytes for b in self._scales)

    @property
    def pinned_staging_bytes(self) -> int:
        return self._host_rows.numel() + self._host_idx.numel() * 4

    def row_bias(self, layer_id: int) -> int:
        return self._row_bias[layer_id]

    def rows_buffer(self, layer_id: int, num_tokens_padded: int) -> torch.Tensor:
        """Static device view [T_pad, n_sub, 264] that wait_rows(layer_id) fills (stable across graph replays)."""
        if num_tokens_padded > self.max_tokens:
            raise ValueError(f"Engram: {num_tokens_padded} tokens exceed max_num_batched_tokens {self.max_tokens}")
        return self._dense[self.layers.index(layer_id), :num_tokens_padded]

    # ---------------------------------------------------------------- key -> I/O addresses
    def _addresses(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """rows [n, L, S] global row ids -> (inverse [n, L, S] into the unique keys, then per unique key: file index,
        absolute byte offset of the 256-B row, scale table id, scale entry index). A global row id lies in exactly one
        sub-table, so any occurrence of a key gives its sub-table column."""
        n, L, S = rows.shape
        l_flat = np.broadcast_to(np.arange(L, dtype=np.int64)[None, :, None], (n, L, S)).reshape(-1)
        j_flat = np.broadcast_to(np.arange(S, dtype=np.int64)[None, None, :], (n, L, S)).reshape(-1)
        r_flat = rows.reshape(-1)
        _, first, inv = np.unique((l_flat << 40) | r_flat, return_index=True, return_inverse=True)
        l_u, j_u, r_u = l_flat[first], j_flat[first], r_flat[first]
        local = r_u - self._sub_offset[l_u, j_u]
        if local.size and (int(local.min()) < 0 or bool((local >= self._sub_rows[l_u, j_u]).any())):
            raise AssertionError("Engram row id outside its sub-table (hash/layout mismatch)")
        file_idx = self._file_of_layer[l_u]
        offsets = self._weight_offset[l_u] + r_u * ROW_VALUE_BYTES
        tab = self._table_ids_arr[l_u]
        sidx = self._scale_base[l_u, j_u] + local
        return inv.reshape(n, L, S).astype(np.int32), file_idx, offsets, tab, sidx

    def _submit_rows(self, rows: np.ndarray, slot: int, first_unique: int) -> tuple[np.ndarray, int, int]:
        """Gather the unique rows of ``rows`` [n, L, S] into slot rows [first_unique, ...). Returns (inverse+offset,
        number of unique rows, ticket)."""
        inv, file_idx, offsets, tab, sidx = self._addresses(rows)
        n_u = offsets.shape[0]
        if first_unique + n_u > self.max_unique:
            raise RuntimeError(f"Engram staging overflow: {first_unique + n_u} rows > {self.max_unique}")
        dst = self._host_rows[slot, first_unique:first_unique + n_u].view(-1)
        ticket = self._reader.submit(torch.from_numpy(file_idx), torch.from_numpy(offsets),
                                     torch.from_numpy(tab), torch.from_numpy(sidx), dst, ENGRAM_ROW_BYTES,
                                     ROW_VALUE_BYTES)
        self._stats["lookups"] += int(rows.size)
        self._stats["unique_rows"] += n_u
        return inv + first_unique, n_u, ticket

    def _prefetch_rows(self, rows: np.ndarray) -> None:
        n, L, S = rows.shape
        keys = np.unique(((np.arange(L, dtype=np.int64)[None, :, None] << 40) | rows).reshape(-1))
        l_idx = keys >> 40
        offsets = self._weight_offset[l_idx] + (keys & ((1 << 40) - 1)) * ROW_VALUE_BYTES
        self._reader.prefetch(torch.from_numpy(self._file_of_layer[l_idx].astype(np.int32)),
                              torch.from_numpy(offsets), ROW_VALUE_BYTES)
        self._stats["prefetch_rows"] += int(keys.shape[0])

    # ---------------------------------------------------------------- hooks
    def begin_step(self, plan: EngramStepPlan) -> None:
        """Worker thread, BEFORE the PP receive. Hash every request whose token ids are known and submit this step's
        gather; admission prefetch for new/resumed requests. Never blocks on I/O (a CPU wait happens only if the H2D
        that last read this pinned slot -- two steps ago -- has not finished)."""
        self._retire_step()
        for rid in plan.finished_req_ids:
            self._hist.pop(rid, None)
        slot = self._next_slot
        self._next_slot ^= 1
        if self._h2d_recorded[slot]:
            self._h2d_done[slot].synchronize()
        reqs: dict[str, _ReqSlot] = {}
        known_rows: list[np.ndarray] = []
        known_meta: list[tuple[int, int]] = []   # (token offset, n) of each known block
        prefetch: list[np.ndarray] = []
        off = 0
        for r in plan.reqs:
            if r.req_id in reqs:
                raise ValueError(f"Engram step {plan.step_id}: request {r.req_id} scheduled twice")
            if r.num_tokens <= 0:
                raise ValueError(f"Engram step {plan.step_id}: request {r.req_id} has {r.num_tokens} tokens")
            hist = self._hist.get(r.req_id)
            if r.prompt_token_ids is not None:
                hist = _History(self.hasher.compress(np.asarray(r.prompt_token_ids)))
                self._hist[r.req_id] = hist
            elif hist is None:
                raise KeyError(f"Engram: request {r.req_id} has no token history; its first (or resumed) step must "
                               "carry prompt_token_ids")
            start, n = int(r.start_pos), int(r.num_tokens)
            if r.token_ids is not None:
                ids = np.asarray(r.token_ids)
                if ids.shape != (n,):
                    raise ValueError(f"Engram: request {r.req_id} token_ids {ids.shape} != num_tokens {n}")
                hist.write(start, self.hasher.compress(ids))
                resolved = True
            elif start + n <= hist.prompt_len:
                resolved = True                     # continuation of the known prompt (chunked prefill)
            elif n == 1 and hist.prompt_len <= start <= hist.length:
                resolved = False                    # sampled token arrives in bind_batch (no PP); positions past
                                                    # start (rejected drafts) are dropped when it is written
            else:
                raise ValueError(f"Engram: request {r.req_id} positions [{start}, {start + n}) without token ids: "
                                 f"only prompt positions (< {hist.prompt_len}) or one sampled token at <= "
                                 f"{hist.length} can be resolved")
            if resolved:
                known_rows.append(self.hasher.hash_positions(hist.view(), np.arange(start, start + n)))
                known_meta.append((off, n))
                if r.prompt_token_ids is not None and self.prefetch_enabled and start + n < hist.length:
                    prefetch.append(self.hasher.hash_positions(hist.view(), np.arange(start + n, hist.length)))
            reqs[r.req_id] = _ReqSlot(off=off, n=n, start_pos=start, resolved=resolved)
            off += n
        L, S = len(self.layers), self.n_sub
        inv = np.zeros((off, L, S), np.int32)
        step = _Step(step_id=plan.step_id, slot=slot, reqs=reqs, total_tokens=off, inv=inv, n_unique=0,
                     t_begin=time.perf_counter())
        if known_rows:
            rows = np.concatenate(known_rows, axis=0)
            kinv, n_u, ticket = self._submit_rows(rows, slot, 1)
            step.tickets.append(ticket)
            self._step = step          # published at once: a later exception still leaves the gather owned/drained
            pos = 0
            for t_off, n in known_meta:
                inv[t_off:t_off + n] = kinv[pos:pos + n]
                pos += n
            step.n_unique = n_u
        self._step = step
        self._stats["steps"] += 1
        for rows in prefetch:
            self._prefetch_rows(rows)

    def bind_batch(self, layout: EngramBatchLayout) -> None:
        """Runner thread, after _prepare_inputs and before the forward of a REAL scheduled step. Dummy forwards
        (profiling, CUDA-graph capture) are not bound: the module asks for ``dummy_rows`` when the forward context
        says ``is_dummy_run`` (AM-2)."""
        t_pad = int(layout.num_tokens_padded)
        if t_pad > self.max_tokens:
            raise ValueError(f"Engram: T_pad {t_pad} > max_num_batched_tokens {self.max_tokens}")
        if layout.step_id < 0 or layout.num_tokens <= 0:
            raise ValueError(f"Engram: bind_batch got step {layout.step_id} with {layout.num_tokens} tokens; dummy "
                             "forwards are marked by ForwardContext.is_dummy_run and are never bound (AM-2)")
        step = self._step
        if step is None or step.step_id != layout.step_id:
            raise RuntimeError(f"Engram: bind_batch for step {layout.step_id} but begin_step saw "
                               f"{None if step is None else step.step_id}")
        if step.layout is not None:
            raise RuntimeError(f"Engram: step {layout.step_id} bound twice")
        qsl = np.asarray(layout.query_start_loc)
        if qsl.shape[0] != len(layout.req_order) + 1 or int(qsl[-1]) != layout.num_tokens:
            raise ValueError(f"Engram: query_start_loc {qsl.tolist()} inconsistent with {len(layout.req_order)} "
                             f"requests / {layout.num_tokens} tokens")
        if set(layout.req_order) != set(step.reqs) or layout.num_tokens != step.total_tokens:
            raise ValueError(f"Engram: runner batch {sorted(layout.req_order)} / {layout.num_tokens} tokens differs "
                             f"from the plan {sorted(step.reqs)} / {step.total_tokens}")
        unresolved = [(i, rid) for i, rid in enumerate(layout.req_order) if not step.reqs[rid].resolved]
        if unresolved:
            self._resolve_sampled(step, layout, unresolved)
        L, S = len(self.layers), self.n_sub
        perm = np.empty(layout.num_tokens, np.int64)
        for i, rid in enumerate(layout.req_order):
            a, b = int(qsl[i]), int(qsl[i + 1])
            rs = step.reqs[rid]
            if b - a != rs.n:
                raise ValueError(f"Engram: request {rid} has {b - a} tokens in the runner batch, {rs.n} planned")
            perm[a:b] = np.arange(rs.off, rs.off + rs.n)
        idx = np.zeros((L, t_pad, S), np.int32)
        idx[:, : layout.num_tokens] = step.inv[perm].transpose(1, 0, 2)
        self._host_idx[step.slot, : L * t_pad * S].numpy()[:] = idx.reshape(-1)
        step.layout = layout
        step.t_pad = t_pad
        if all(self._reader.done(t) for t in step.tickets):
            self._issue_h2d(step)

    def _resolve_sampled(self, step: _Step, layout: EngramBatchLayout, unresolved: list[tuple[int, str]]) -> None:
        if layout.sampled_fill is None:
            raise RuntimeError(f"Engram step {step.step_id}: requests {[r for _, r in unresolved]} have unknown "
                               "tokens but the runner passed no sampled_fill")
        ids_t, event = layout.sampled_fill
        event.synchronize()
        ids = ids_t.reshape(-1).numpy()
        if ids.shape[0] != len(layout.req_order):
            raise ValueError(f"Engram step {step.step_id}: sampled_fill holds {ids.shape[0]} ids, expected one per "
                             f"request in runner order ({len(layout.req_order)})")
        rows, blocks = [], []
        for i, rid in unresolved:
            rs = step.reqs[rid]
            hist = self._hist[rid]
            hist.write(rs.start_pos, self.hasher.compress(np.array([int(ids[i])])))
            rows.append(self.hasher.hash_positions(hist.view(), np.array([rs.start_pos])))
            blocks.append(rs.off)
            rs.resolved = True
        kinv, n_u, ticket = self._submit_rows(np.concatenate(rows), step.slot, 1 + step.n_unique)
        for k, t_off in enumerate(blocks):
            step.inv[t_off] = kinv[k]
        step.n_unique += n_u
        step.tickets.append(ticket)

    def _issue_h2d(self, step: _Step) -> None:
        for t in step.tickets:
            t0 = time.perf_counter()
            n, hits, misses, pages, ns = self._reader.wait(t)
            self._stats["wait_blocked_s"] += time.perf_counter() - t0
            self._stats["hits"] += hits
            self._stats["misses"] += misses
            self._stats["pages"] += pages
            self._lat_ms.append(ns / 1e6)
        step.tickets.clear()
        if len(self._lat_ms) > _LATENCY_WINDOW:
            del self._lat_ms[: len(self._lat_ms) - _LATENCY_WINDOW]
        slot = step.slot
        n_rows = step.n_unique + 1
        n_idx = len(self.layers) * step.t_pad * self.n_sub
        with torch.cuda.stream(self._copy_stream):
            if self._consumed_recorded[slot]:
                self._copy_stream.wait_event(self._consumed[slot])
            self._dev_rows[slot, :n_rows].copy_(self._host_rows[slot, :n_rows], non_blocking=True)
            self._dev_idx[slot, :n_idx].copy_(self._host_idx[slot, :n_idx], non_blocking=True)
            self._h2d_done[slot].record(self._copy_stream)
        self._h2d_recorded[slot] = True
        self._stats["h2d_bytes"] += n_rows * ENGRAM_ROW_BYTES + n_idx * 4
        step.h2d_issued = True

    def wait_rows(self, layer_id: int) -> torch.Tensor:
        """Inside the Engram eager break: block the CPU until this step's rows are gathered and their H2D is enqueued,
        make the current stream wait for it, expand to the static device view [T_pad, n_sub, 264] u8 of layer_id."""
        li = self.layers.index(layer_id)
        step = self._step
        if step is None or step.layout is None:
            raise RuntimeError("Engram wait_rows: a REAL step reached Engram without bound rows -- the worker must call "
                               "begin_step(plan) and the runner bind_batch(layout) before every non-dummy forward "
                               "(dummy forwards must set ForwardContext.is_dummy_run, AM-2)")
        if not step.h2d_issued:
            self._issue_h2d(step)
        if not step.expanded:
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(self._h2d_done[step.slot])
            span = step.t_pad * self.n_sub
            for l_idx in range(len(self.layers)):
                out = self._dense[l_idx, : step.t_pad].view(span, ENGRAM_ROW_BYTES)
                torch.index_select(self._dev_rows[step.slot], 0,
                                   self._dev_idx[step.slot, l_idx * span:(l_idx + 1) * span], out=out)
            self._consumed[step.slot].record(cur)
            self._consumed_recorded[step.slot] = True
            step.expanded = True
        return self._dense[li, : step.t_pad]

    def dummy_rows(self, layer_id: int, num_tokens_padded: int) -> torch.Tensor:
        """AM-2 dummy forward (profiling / CUDA-graph capture): all-zero rows in the static buffer -> the Engram
        contribution is exactly zero; no host I/O, no step state touched. Counted once per forward."""
        out = self.rows_buffer(layer_id, num_tokens_padded)
        out.zero_()
        if layer_id == self.layers[0]:
            self._stats["dummy_steps"] += 1
        return out

    def end_step(self, step_id: int) -> None:
        step = self._step
        if step is not None and step.step_id == step_id:
            self._retire_step()

    def _retire_step(self) -> None:
        step = self._step
        if step is None:
            return
        if step.tickets:          # gathered but never consumed (aborted step): drain so the slot can be reused
            for t in step.tickets:
                self._reader.wait(t)
            step.tickets.clear()
        self._step = None

    def stats(self) -> dict[str, float]:
        lat = np.array(self._lat_ms) if self._lat_ms else np.zeros(1)
        s = {k: float(v) for k, v in self._stats.items()}
        s.update(gather_p50_ms=float(np.percentile(lat, 50)), gather_p99_ms=float(np.percentile(lat, 99)),
                 gather_max_ms=float(lat.max()), warmup_backlog_rows=float(self._reader.pending_prefetch_rows()),
                 pinned_scale_gib=self.pinned_scale_bytes / 2**30, pinned_staging_mib=self.pinned_staging_bytes / 2**20)
        return s

    def drop_page_cache(self) -> None:
        """POSIX_FADV_DONTNEED on the row shards (cold-path measurements; no privileges needed)."""
        for i in range(len(set(loc.weight_path for loc in self.tables))):
            self._reader.drop_cache(i)

    def shutdown(self) -> None:
        """Joins the reader's threads BEFORE the staging/scale memory they write into or read from is released."""
        if getattr(self, "_reader", None) is not None:
            self._reader.shutdown()
            self._reader = None
        for b in getattr(self, "_scales", []):
            b.release()
        self._scales = []
