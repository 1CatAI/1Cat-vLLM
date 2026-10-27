# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 packed long-context KV (G4b), like Flash-Next's host KV.

Compressed latents (FP4 + E4M3 scales, 288 B per row) live once in a shared
host memfd mapped into all four TP ranks (rank 0 writes) plus a per-GPU hot
ring of the most recent rows. Index keys (FP4, 68 B per row) stay on the GPU:
every query scans all of them. Readers:
  decode  - after each top-k, the selected rows are copied into a small GPU
            buffer once per index layer; attention layers read that buffer.
  prefill - rows older than the hot ring are staged once per source layer and
            chunk into a GPU staging buffer.
"""

import json
import secrets
import socket
import tempfile

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.models.deepseek_v41.kv_pack import C_ROW, I_ROW, pack_c_group
from vllm.triton_utils import tl, triton
from vllm.utils.shared_host import SharedHostArena

logger = init_logger(__name__)

HOT_POSITIONS = int(envs.VLLM_V41_KV_HOT_POSITIONS)
DECODE_ROWS = 8  # at most this many tokens use the selected-row buffer


def _share_arena(size, group):
    """Rank 0 creates one sealed memfd; the other ranks receive its descriptor."""
    rank = group.rank_in_group
    if rank == 0:
        arena = SharedHostArena.create(size, "vllm-v41-kv")
        with tempfile.TemporaryDirectory(prefix="vllm-v41kv-") as directory:
            path = directory + "/fd"
            nonce = secrets.token_hex(32)
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
                server.settimeout(60)
                server.bind(path)
                server.listen(3)
                group.broadcast_object((path, nonce, socket.gethostname()), src=0)
                seen = set()
                for _ in range(group.world_size - 1):
                    connection, _ = server.accept()
                    with connection:
                        connection.settimeout(60)
                        peer, token = json.loads(connection.recv(4096))
                        if (
                            token != nonce
                            or peer in seen
                            or not 0 < peer < group.world_size
                        ):
                            raise RuntimeError("Invalid V4.1 KV arena consumer")
                        seen.add(peer)
                        arena.send(connection)
        return arena
    path, nonce, hostname = group.broadcast_object(None, src=0)
    if hostname != socket.gethostname():
        raise ValueError("V4.1 host KV requires a single-node TP group")
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(60)
        connection.connect(path)
        connection.sendall(json.dumps((rank, nonce)).encode())
        return SharedHostArena.receive(connection)


@triton.jit(do_not_specialize=["HOT_ROWS"])
def _insert_c(
    values,
    positions,
    valid_tokens,
    hot,
    host,
    HOT_ROWS,
    RATIO: tl.constexpr,
    WRITE_HOST: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    position = tl.load(positions + token).to(tl.int64)
    valid = (
        tl.load(valid_tokens + token) & ((position + 1) % RATIO == 0) & (position >= 0)
    )
    if valid:
        c = position // RATIO
        x = values + token.to(tl.int64) * 512
        pack_c_group(x, hot + (c % HOT_ROWS) * 288, group)
        if WRITE_HOST:
            pack_c_group(x, host + c * 288, group)


@triton.jit(do_not_specialize=["T", "HOT_ROWS"])
def _fill_selected(
    topk,
    positions,
    hot,
    host,
    out,
    T,
    HOT_ROWS,
    RATIO: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t, part = tl.program_id(0), tl.program_id(1)
    j = part * BLOCK + tl.arange(0, BLOCK)
    c = tl.load(topk + t * TOPK + j).to(tl.int64)
    newest = (tl.load(positions + T - 1).to(tl.int64) + 1) // RATIO - 1
    valid = c >= 0
    in_hot = valid & (c > newest - HOT_ROWS)
    w = tl.arange(0, 128)
    words = (w < 72)[None, :]
    hot_words = (hot + (c % HOT_ROWS) * 288).to(tl.pointer_type(tl.int32))
    host_words = (host + c * 288).to(tl.pointer_type(tl.int32))
    a = tl.load(hot_words[:, None] + w[None, :], in_hot[:, None] & words, 0)
    b = tl.load(host_words[:, None] + w[None, :], (valid & ~in_hot)[:, None] & words, 0)
    dst = (out + (t * TOPK + j).to(tl.int64) * 288).to(tl.pointer_type(tl.int32))
    tl.store(dst[:, None] + w[None, :], tl.where(in_hot[:, None], a, b), words)


class V41PackedKV:
    """One instance per model: packed rows for the KV source layers."""

    def __init__(self, sources, max_len, device, group):
        self.sources = dict(sources)  # layer -> ratio
        self.max_len = max_len
        self.device = device
        self.rank = group.rank_in_group
        self.rows = {s: max_len // r + 1 for s, r in self.sources.items()}
        self.offsets, total = {}, 0
        for s in sorted(self.sources):
            self.offsets[s] = total
            total += self.rows[s] * C_ROW
        self.arena = _share_arena(total, group)
        from vllm.model_executor.layers.quantization.gguf_expert_cache import _native

        self.native = _native()
        self.host_cpu = torch.frombuffer(self.arena.buffer, dtype=torch.uint8)
        self.native.host_register(self.host_cpu.data_ptr(), self.host_cpu.numel())
        self.mapped = self.native.mapped_view(self.host_cpu, device.index)
        self.host = {
            s: self.mapped[
                self.offsets[s] : self.offsets[s] + self.rows[s] * C_ROW
            ].view(self.rows[s], C_ROW)
            for s in self.sources
        }
        self.hot_rows = {s: max(1, HOT_POSITIONS // r) for s, r in self.sources.items()}
        self.hot = {
            s: torch.zeros(self.hot_rows[s], C_ROW, dtype=torch.uint8, device=device)
            for s in self.sources
        }
        self.index = {
            s: torch.zeros(self.rows[s], I_ROW, dtype=torch.uint8, device=device)
            for s in self.sources
        }
        largest = max(self.rows[s] - self.hot_rows[s] for s in self.sources)
        self.staging = (
            torch.empty(largest, C_ROW, dtype=torch.uint8, device=device)
            if largest > 0
            else None
        )
        self.staged = None  # source currently held by the staging buffer
        self.selected = torch.zeros(
            DECODE_ROWS, 512, C_ROW, dtype=torch.uint8, device=device
        )
        logger.info(
            "V4.1 packed KV: max_len %d, host %.0f MiB, "
            "GPU index %.0f MiB, hot %.0f MiB, "
            "staging %.0f MiB",
            max_len,
            total / 2**20,
            sum(t.numel() for t in self.index.values()) / 2**20,
            sum(t.numel() for t in self.hot.values()) / 2**20,
            (self.staging.numel() if self.staging is not None else 0) / 2**20,
        )

    # -- writers --------------------------------------------------------
    def insert_compressed(self, source, values, positions, valid_tokens):
        """values: pre-QDQ FP16 [T, 512] (RoPE applied)."""
        if values.shape[0]:
            _insert_c[(values.shape[0], 32)](
                values.contiguous(),
                positions,
                valid_tokens,
                self.hot[source],
                self.host[source],
                self.hot_rows[source],
                RATIO=self.sources[source],
                WRITE_HOST=self.rank == 0,
            )

    # -- readers --------------------------------------------------------
    def fill_selected(self, source, topk, positions):
        """Decode: copy each token's selected rows (hot or host) into `selected`."""
        tokens = topk.shape[0]
        assert tokens <= DECODE_ROWS and topk.shape[1] == 512
        _fill_selected[(tokens, 512 // 64)](
            topk,
            positions,
            self.hot[source],
            self.host[source],
            self.selected,
            tokens,
            self.hot_rows[source],
            RATIO=self.sources[source],
            TOPK=512,
            BLOCK=64,
        )

    def stage(self, source, positions):
        """Prefill: rows older than the hot ring, copied once per source and chunk."""
        ratio = self.sources[source]
        newest = (int(positions[-1].item()) + 1) // ratio - 1
        old = newest - self.hot_rows[source] + 1
        if old > 0:
            self.staging[:old].copy_(self.host[source][:old], non_blocking=True)  # type: ignore[index]
        self.staged = source
        return old
