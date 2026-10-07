# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 Engram: layout, token map, CPU hash, TP sub-table sharding and the ``DeepseekV41Engram`` module.

Owner: lane L-ENGRAM (PORT_DESIGN §2.2, §3.6, §3.7, §4.1). Reference: official ``inference/engram.py`` and
``inference/model.py:296-365, 1250-1263`` at rev 2cba9e42 (``ref:e.py`` / ``ref:m.py``).

* The hash is a bit-exact numpy port of ``ref:e.py:159-184`` and runs on the CPU from token ids alone, so the rows
  of a step can be gathered before the forward starts (``common/engram_host.py``).
* TP rank r of a stage owns the sub-tables ``{r, r + tp, ...}`` (order-major ``s = (n - 2) * 8 + head``): it gathers
  only their rows and holds only their ``wkv`` input columns; the partial ``wkv`` products are summed by an FP32
  all-reduce (row-parallel GEMM). With tp = 4 every rank owns two heads of each n-gram order.
* Numerics (§4.1): rows decode exactly to FP16; ``wkv`` is stored as FP16 x 2^10 (its block exponents reach 2^-18,
  below FP16's normal range) with FP32 accumulation, the 2^-10 (and any row bias) applied in FP32 after the
  all-reduce; the gate runs in FP32 exactly as ``ref:m.py:350-365`` (``copysign``, ``(h * (q*k)) * key``).
"""

from __future__ import annotations

import functools
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import (
    ENGRAM_HEAD_DIM,
    ENGRAM_ROW_BYTES,
    ENGRAM_SUBTABLES,
    HC,
    HIDDEN,
    NORM_EPS,
    STREAM_DTYPE,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

DEAD_ID = -1                     # compressed-history sentinel: token takes no part in an n-gram (ref:e.py:138)
WKV_BIAS_LOG2 = 10               # wkv stored as FP16 x 2^10 (PORT_DESIGN §4.1)
GATE_CLAMP = 1e-6                # ref:m.py:340
ROW_VALUE_BYTES = ENGRAM_HEAD_DIM            # 256 x E4M3
ROW_SCALE_BYTES = ENGRAM_ROW_BYTES - ROW_VALUE_BYTES   # 8 x UE8M0, one per 32 channels
SCALE_BLOCK = ENGRAM_HEAD_DIM // ROW_SCALE_BYTES       # 32
WKV_OUT = HC * HIDDEN + HIDDEN   # 25600 = 4 keys + 1 value

IMPL_KNOB = "VLLM_DS41_ENGRAM_IMPL"                  # "sm70" (Triton) | "torch" (oracle path, §7.1)
PREFILL_CHUNK_KNOB = "VLLM_DS41_ENGRAM_PREFILL_CHUNK"  # tokens per wkv sub-chunk (bounds the FP32 partial)

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)  # deterministic Miller-Rabin for n < 3.3e24


# ------------------------------------------------------------------------------------------------ layout
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_unseen_prime(start: int, seen: set[int]) -> int:
    """ref:e.py:9-14 -- the smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def compute_hash_multipliers(layer_ids: tuple[int, ...], max_ngram_size: int,
                             compressed_vocab_size: int) -> np.ndarray:
    """ref:e.py:64-83 -> int64 [n_layers, max_ngram_size]; odd, and ``id * mult`` cannot overflow int64."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        values = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows).astype(np.int64)


def _cfg(hf_config, *names: str):
    """Field of the flattened DeepseekV41Config, or of the raw HF ``text_config`` nesting."""
    text_config = getattr(hf_config, "text_config", None)
    if isinstance(text_config, dict):
        text_config = type("_TextConfig", (), text_config)
    for source in (hf_config, text_config):
        if source is None:
            continue
        for name in names:
            if hasattr(source, name):
                return getattr(source, name)
    raise AttributeError(f"hf_config (and its text_config) has none of {names}")


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the Engram tables (ref:e.py:86-126) plus the hash constants (ref:e.py:140-157).

    Sub-table ``s`` (order-major: ``s = (n - 2) * n_heads + head``) of layer index ``li`` owns the rows
    ``[offsets[li][s], offsets[li][s] + primes[li][s])`` of that layer's table.
    """

    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[int, ...], ...]       # [layer][s]
    offsets: tuple[tuple[int, ...], ...]      # [layer][s]
    multipliers: tuple[tuple[int, ...], ...]  # [layer][lookback]
    compressed_vocab_size: int
    pad_token_id: int

    @property
    def n_subtables(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_hf_config(cls, hf_config) -> EngramLayout:
        layer_ids = tuple(int(x) for x in _cfg(hf_config, "engram_layer_ids"))
        if not layer_ids:
            raise ValueError("hf_config.engram_layer_ids is empty: the model has no Engram layers")
        max_ngram = int(_cfg(hf_config, "engram_max_ngram_size"))
        n_heads = int(_cfg(hf_config, "engram_n_heads"))
        head_dim = int(_cfg(hf_config, "engram_head_dim"))
        vocab = int(_cfg(hf_config, "engram_vocab_size"))
        num_embeddings = tuple(int(x) for x in _cfg(hf_config, "engram_num_embeddings"))
        cvocab = int(_cfg(hf_config, "engram_compressed_vocab_size"))
        pad = int(_cfg(hf_config, "engram_pad_token_id", "engram_pad_id"))
        if len(num_embeddings) != len(layer_ids):
            raise ValueError(f"engram_num_embeddings {num_embeddings} does not match layer ids {layer_ids}")
        primes: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for _ in layer_ids:
            flat: list[int] = []
            for _order in range(max_ngram - 1):
                current = vocab - 1
                for _head in range(n_heads):
                    current = _next_unseen_prime(current, seen)
                    seen.add(current)
                    flat.append(current)
            primes.append(tuple(flat))
        offsets = tuple(tuple(int(x) for x in np.cumsum([0, *p[:-1]])) for p in primes)
        for li, (p, n) in enumerate(zip(primes, num_embeddings)):
            if sum(p) != n:
                raise ValueError(f"Engram layer {layer_ids[li]}: sum of primes {sum(p)} != "
                                 f"engram_num_embeddings {n} (layout/config mismatch)")
        mult = compute_hash_multipliers(layer_ids, max_ngram, cvocab)
        return cls(layer_ids=layer_ids, max_ngram_size=max_ngram, n_heads=n_heads, head_dim=head_dim,
                   num_embeddings=num_embeddings, primes=tuple(primes), offsets=offsets,
                   multipliers=tuple(tuple(int(v) for v in row) for row in mult),
                   compressed_vocab_size=cvocab, pad_token_id=pad)

    def layer_index(self, layer_id: int) -> int:
        if layer_id not in self.layer_ids:
            raise ValueError(f"layer {layer_id} is not an Engram layer {self.layer_ids}")
        return self.layer_ids.index(layer_id)

    def subtables_for_rank(self, tp_rank: int, tp_size: int) -> tuple[int, ...]:
        """Rank r gathers sub-tables {r, r + tp, r + 2 tp, ...} (PORT_DESIGN A5)."""
        if tp_size < 1 or self.n_subtables % tp_size != 0:
            raise ValueError(f"{self.n_subtables} Engram sub-tables do not split over tp_size={tp_size}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank {tp_rank} outside [0, {tp_size})")
        return tuple(range(tp_rank, self.n_subtables, tp_size))


# ------------------------------------------------------------------------------------------------ token map
def build_compressed_token_map(tokenizer_file: str) -> tuple[np.ndarray, int]:
    """ref:e.py:17-61 on the raw Rust tokenizer (``tokenizer.json``): token id -> compressed id (int64)."""
    from tokenizers import Regex, Tokenizer, normalizers

    backend = Tokenizer.from_file(tokenizer_file)
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    n = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(n, dtype=np.int64)
    for token_id in range(n):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def resolve_tokenizer_file(tokenizer_path: str) -> str:
    path = os.path.join(tokenizer_path, "tokenizer.json") if os.path.isdir(tokenizer_path) else tokenizer_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Engram hash needs the model's tokenizer.json; not found at {path}")
    return os.path.realpath(path)


@functools.lru_cache(maxsize=4)
def _cached_token_map(path: str, mtime_ns: int, size: int) -> tuple[np.ndarray, int]:
    lookup, n = build_compressed_token_map(path)
    lookup.setflags(write=False)
    return lookup, n


def load_compressed_token_map(tokenizer_path: str, expected_size: int) -> np.ndarray:
    """Compressed token map (cached per process), asserting the config's compressed vocab size (ref:e.py:146)."""
    path = resolve_tokenizer_file(tokenizer_path)
    st = os.stat(path)
    lookup, n = _cached_token_map(path, st.st_mtime_ns, st.st_size)
    if n != expected_size:
        raise ValueError(f"tokenizer {path} compresses to {n} ids, config engram_compressed_vocab_size is "
                         f"{expected_size}: every Engram hash multiplier would differ (ref:e.py:143-146)")
    return lookup


# ------------------------------------------------------------------------------------------------ hash
class EngramHasher:
    """Bit-exact numpy port of ``NgramHashState.forward`` (ref:e.py:159-184) for a subset of layers/sub-tables.

    Works on a per-request history of compressed ids indexed by absolute position (``DEAD_ID`` for masked
    tokens). Look-back is padded with the compressed pad id once any of ``t, t-1, ..., t-s`` is before the start
    or dead (the reference's cumulative ``blocked``).
    """

    def __init__(self, layout: EngramLayout, token_map: np.ndarray, layer_ids: tuple[int, ...],
                 subtables: tuple[int, ...]) -> None:
        if token_map.ndim != 1 or token_map.dtype != np.int64:
            raise ValueError("token_map must be a 1-D int64 array")
        if int(token_map.max()) + 1 != layout.compressed_vocab_size:
            raise ValueError(f"token map has {int(token_map.max()) + 1} compressed ids, config says "
                             f"{layout.compressed_vocab_size} (ref:e.py:143-146)")
        if not layer_ids or not subtables:
            raise ValueError("EngramHasher needs at least one layer and one sub-table")
        self.layout = layout
        self.token_map = token_map
        self.pad = int(token_map[layout.pad_token_id])
        self.layer_ids = tuple(layer_ids)
        self.subtables = tuple(subtables)
        lis = [layout.layer_index(lid) for lid in self.layer_ids]
        self._mult = np.array([layout.multipliers[li] for li in lis], dtype=np.int64)            # [L, M]
        self._primes = np.array([[layout.primes[li][s] for s in self.subtables] for li in lis],
                                dtype=np.int64)                                                    # [L, S]
        self._offsets = np.array([[layout.offsets[li][s] for s in self.subtables] for li in lis],
                                 dtype=np.int64)                                                   # [L, S]
        self._order_idx = np.array([s // layout.n_heads for s in self.subtables], dtype=np.int64)  # n - 2

    def compress(self, token_ids: np.ndarray, token_mask: np.ndarray | None = None) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.token_map.shape[0]):
            raise ValueError(f"token id outside [0, {self.token_map.shape[0]}): min {int(ids.min())} "
                             f"max {int(ids.max())}")
        out = self.token_map[ids]
        if token_mask is not None:
            out = np.where(np.asarray(token_mask, dtype=bool), out, DEAD_ID)
        return out

    def hash_positions(self, history: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """history: int64 compressed ids by absolute position; positions: int64 [n], each < len(history).
        Returns global row ids int64 [n, n_layers, n_subtables] (index into the layer's whole table)."""
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size and (int(pos.min()) < 0 or int(pos.max()) >= history.shape[0]):
            raise ValueError(f"positions [{int(pos.min())}, {int(pos.max())}] outside history of length "
                             f"{history.shape[0]}")
        m = self.layout.max_ngram_size
        n = pos.shape[0]
        toks = np.empty((n, m), dtype=np.int64)
        blocked = np.zeros(n, dtype=bool)
        for shift in range(m):
            src_pos = pos - shift
            src = history[np.maximum(src_pos, 0)]
            blocked = blocked | (src_pos < 0) | (src == DEAD_ID)
            toks[:, shift] = np.where(blocked, self.pad, src)
        prods = toks[:, None, :] * self._mult[None, :, :]       # [n, L, M]; < int64 max by construction
        rolls = np.empty((n, len(self.layer_ids), m - 1), dtype=np.int64)
        rolling = prods[:, :, 0]
        for i in range(1, m):
            rolling = np.bitwise_xor(rolling, prods[:, :, i])
            rolls[:, :, i - 1] = rolling
        sel = rolls[:, :, self._order_idx]                      # [n, L, S]
        return np.mod(sel, self._primes[None]) + self._offsets[None]


# ------------------------------------------------------------------------------------------------ numerics
@functools.lru_cache(maxsize=None)
def _e4m3_lut_cpu() -> torch.Tensor:
    """float32 value of every E4M3 (fn) byte; 0x7F/0xFF -> NaN. Software conversion on the CPU (exact)."""
    return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)


def e4m3_lut(device: torch.device) -> torch.Tensor:
    return _e4m3_lut_cpu().to(device)


def pow2_f32(exponent: torch.Tensor) -> torch.Tensor:
    """2^exponent as float32 built from the bit pattern (exact); exponent must lie in [-126, 127]."""
    e = exponent.to(torch.int32)
    return ((e + 127).clamp(0, 254) << 23).view(torch.float32)


def row_bias_for_exponents(e_min: int, e_max: int) -> int:
    """Power-of-two bias b so that every row value e4m3 * 2^(e - 127 + b) is an exact FP16 number.

    E4M3 values have 4 significant bits, lowest bit >= 2^-9 and magnitude <= 448 = 1.75 * 2^8, so with
    k = e - 127 + b the value is exact in FP16 iff 2^(-9 + k) >= 2^-24 (k >= -15) and 448 * 2^k <= 65504 (k <= 7).
    """
    lo = -15 - (e_min - 127)
    hi = 7 - (e_max - 127)
    if lo > hi:
        raise ValueError(f"Engram row scale exponents [{e_min - 127}, {e_max - 127}] span more than FP16 can hold "
                         "exactly; rows would need FP32 decode")
    return 0 if lo <= 0 <= hi else (lo if lo > 0 else hi)


def decode_rows_torch(rows: torch.Tensor, row_bias: int) -> torch.Tensor:
    """rows [n, S, 264] uint8 (256 E4M3 + 8 UE8M0) -> [n, S * 256] fp16 = value * 2^(scale - 127 + row_bias)."""
    n, s, w = rows.shape
    if w != ENGRAM_ROW_BYTES:
        raise ValueError(f"Engram rows must be {ENGRAM_ROW_BYTES} bytes wide, got {w}")
    vals = e4m3_lut(rows.device)[rows[..., :ROW_VALUE_BYTES].long()]                    # [n, S, 256] f32
    scale = pow2_f32(rows[..., ROW_VALUE_BYTES:].to(torch.int32) - 127 + row_bias)      # [n, S, 8]
    out = vals.view(n, s, ROW_SCALE_BYTES, SCALE_BLOCK) * scale.unsqueeze(-1)
    return out.reshape(n, s * ROW_VALUE_BYTES).to(torch.float16)


def post_wkv_gate_torch_(stream: torch.Tensor, kv: torch.Tensor, qk: torch.Tensor, alpha: float) -> None:
    """In place: stream [n, HC, HIDDEN] bf16 += gate * value, exactly the expression of ref:m.py:350-365.

    kv [n, 25600] float32 is the all-reduced, still-biased wkv output; alpha (a power of two) removes the bias."""
    n = stream.shape[0]
    kvf = kv * alpha
    key = kvf[:, : HC * HIDDEN].view(n, HC, HIDDEN)
    value = kvf[:, HC * HIDDEN:]
    h = stream.float()
    rstd = torch.rsqrt(h.square().mean(-1) + NORM_EPS) * torch.rsqrt(key.square().mean(-1) + NORM_EPS)
    dot = (h * qk * key).sum(-1) * rstd * HIDDEN ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(GATE_CLAMP).sqrt(), dot))
    stream.copy_((h + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(STREAM_DTYPE))


# ------------------------------------------------------------------------------------------------ module
def _forward_is_dummy() -> bool:
    """AM-2: memory-profiling / CUDA-graph-capture forwards are marked by the runner (ForwardContext.is_dummy_run)."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    return is_forward_context_available() and bool(get_forward_context().is_dummy_run)


def _wait_rows_eager(service: EngramHostService, layer_id: int, out: torch.Tensor) -> None:
    """The Engram eager break (PORT_DESIGN §1 CUDA-graph row): CPU wait for this step's rows + stream wait, written
    into the service's static device buffer ``out`` (the same address on every replay).

    Dummy forwards (AM-2: profiling, CUDA-graph capture) get all-zero rows -- an exact zero contribution, no host I/O --
    while the module still launches its kernels, so a captured graph contains them and every replay re-runs this
    function with the real step's rows. A real step without bound rows raises inside ``service.wait_rows``.

    It must never be captured itself: in FULL cudagraph mode ``eager_break_during_capture`` runs the function inline,
    and a captured wait would replay the capture-time (zero) rows on every step -- silently disabling Engram."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"Engram layer {layer_id}: the row wait is being captured into a CUDA graph. It must run as "
                           "an eager break (breakable cudagraph); FULL cudagraph mode cannot serve Engram rows.")
    if _forward_is_dummy():
        rows = service.dummy_rows(layer_id, out.shape[0])
    else:
        rows = service.wait_rows(layer_id)
    if rows.data_ptr() != out.data_ptr() or rows.shape != out.shape:
        raise RuntimeError(f"Engram layer {layer_id}: service returned rows {tuple(rows.shape)} at "
                           f"{rows.data_ptr():#x}, the captured buffer is {tuple(out.shape)} at {out.data_ptr():#x}")


@functools.lru_cache(maxsize=1)
def _eager_wait():
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    return eager_break_during_capture(_wait_rows_eager)


class DeepseekV41Engram(nn.Module):
    """Engram before blocks 1 and 14 (ref:m.py:328-365, 1262-1263), row-parallel over the stage's TP ranks.

    Parameters: ``wkv_r`` [25600, n_sub * 256] fp16 = dequant(wkv columns of this rank's sub-tables) * 2^10;
    ``qk`` [4, 5120] f32 = q_weight * k_weight (exact). The tables are not parameters: ``EngramHostService`` reads
    the rows in place from the checkpoint shards.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str, layer_id: int,
                 service: EngramHostService) -> None:
        super().__init__()
        if layer_id not in service.layers:
            raise ValueError(f"Engram layer {layer_id} is not served by this stage's service {service.layers}")
        if service.layout.n_subtables != ENGRAM_SUBTABLES or service.layout.head_dim != ENGRAM_HEAD_DIM:
            raise ValueError(f"Engram layout {service.layout.n_subtables} x {service.layout.head_dim} differs from "
                             f"the contract {ENGRAM_SUBTABLES} x {ENGRAM_HEAD_DIM}")
        self.prefix = prefix
        self.layer_id = layer_id
        self.service = service
        self.subtables = service.subtables
        self.tp_size = service.tp_size
        self.impl = knobs.env_str(IMPL_KNOB, "sm70", choices=("sm70", "torch"))
        self.prefill_chunk = knobs.env_int(PREFILL_CHUNK_KNOB, 1024, minimum=1, maximum=1 << 20)
        n_cols = len(self.subtables) * ENGRAM_HEAD_DIM
        self.wkv_r = nn.Parameter(torch.empty(WKV_OUT, n_cols, dtype=torch.float16), requires_grad=False)
        self.qk = nn.Parameter(torch.empty(HC, HIDDEN, dtype=torch.float32), requires_grad=False)
        self._pending: dict[str, torch.Tensor] = {}
        self._consumed: set[str] = set()

    # ---- loading (PORT_DESIGN §3.7 + AM-1) ----
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """AM-1: every non-embed ``layers.{L}.engram.*`` checkpoint tensor, module-relative names ('wkv.weight',
        'wkv.scale', 'q_weight', 'k_weight'). Returns the consumed names plus the parameters they completed
        ('wkv_r', 'qk'), so both L-CORE's consumed-name check and vLLM's every-parameter-loaded check see them.
        Raises if a source tensor is missing: a half-loaded Engram must never run."""
        done: set[str] = set()
        for name, tensor in weights:
            param = self.load_checkpoint_tensor(name, tensor)
            done.add(name)
            if param is not None:
                done.add(param)
        if not self.weights_loaded():
            missing = sorted({"wkv.weight", "wkv.scale", "q_weight", "k_weight"} - self._consumed)
            raise ValueError(f"{self.prefix}: Engram checkpoint tensors {missing} were not provided")
        return done

    def load_checkpoint_tensor(self, name: str, tensor: torch.Tensor) -> str | None:
        """Consume one checkpoint tensor (``name`` = suffix after ``layers.{L}.engram.``).

        Returns the parameter name ("wkv_r" / "qk") once both of its source tensors have arrived, else None.
        ``embed.*`` must never reach here (``skip_checkpoint_weight``): the tables stay on disk."""
        if name.startswith("embed."):
            raise ValueError(f"{self.prefix}: checkpoint tensor engram.{name} must be skipped by "
                             "skip_checkpoint_weight -- EngramHostService reads the table rows in place")
        if name not in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
            raise KeyError(f"{self.prefix}: unexpected Engram checkpoint tensor engram.{name}")
        if name in self._pending or name in self._consumed:
            raise RuntimeError(f"{self.prefix}: checkpoint tensor engram.{name} loaded twice")
        self._pending[name] = tensor
        if "wkv.weight" in self._pending and "wkv.scale" in self._pending:
            self._load_wkv(self._pending.pop("wkv.weight"), self._pending.pop("wkv.scale"))
            self._consumed.update(("wkv.weight", "wkv.scale"))
            return "wkv_r"
        if "q_weight" in self._pending and "k_weight" in self._pending:
            q, k = self._pending.pop("q_weight"), self._pending.pop("k_weight")
            if q.shape != (HC, HIDDEN) or k.shape != (HC, HIDDEN):
                raise ValueError(f"{self.prefix}: q/k weights {tuple(q.shape)}/{tuple(k.shape)} != {(HC, HIDDEN)}")
            # bf16 x bf16 -> fp32 is exact (<= 16 significant bits); the reference only ever uses the product
            self.qk.data.copy_(q.to(self.qk.device, torch.float32) * k.to(self.qk.device, torch.float32))
            self._consumed.update(("q_weight", "k_weight"))
            return "qk"
        return None

    def weights_loaded(self) -> bool:
        return self._consumed == {"wkv.weight", "wkv.scale", "q_weight", "k_weight"}

    def _load_wkv(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        full_in = ENGRAM_SUBTABLES * ENGRAM_HEAD_DIM
        if tuple(weight.shape) != (WKV_OUT, full_in) or weight.element_size() != 1:
            raise ValueError(f"{self.prefix}: wkv.weight {tuple(weight.shape)} {weight.dtype}, "
                             f"expected FP8 {(WKV_OUT, full_in)}")
        if tuple(scale.shape) != (WKV_OUT // 32, full_in // 32) or scale.element_size() != 1:
            raise ValueError(f"{self.prefix}: wkv.scale {tuple(scale.shape)} {scale.dtype}, "
                             f"expected UE8M0 {(WKV_OUT // 32, full_in // 32)}")
        dev = self.wkv_r.device
        cols = torch.cat([torch.arange(s * ENGRAM_HEAD_DIM, (s + 1) * ENGRAM_HEAD_DIM) for s in self.subtables])
        scols = torch.cat([torch.arange(s * ENGRAM_HEAD_DIM // 32, (s + 1) * ENGRAM_HEAD_DIM // 32)
                           for s in self.subtables])
        w = weight.view(torch.uint8).to(dev)[:, cols.to(dev)]                        # [25600, n_cols] u8
        e = scale.view(torch.uint8).to(dev)[:, scols.to(dev)].to(torch.int32)        # [800, n_cols / 32]
        if bool((e == 255).any()):
            raise ValueError(f"{self.prefix}: wkv.scale holds UE8M0 NaN (0xFF)")
        k = e - 127 + WKV_BIAS_LOG2
        if int(k.min()) < -126 or int(k.max()) > 127:
            raise ValueError(f"{self.prefix}: wkv scale exponents out of FP32 range")
        vals = e4m3_lut(dev)[w.long()]
        full = vals * pow2_f32(k).repeat_interleave(32, 0).repeat_interleave(32, 1)
        w16 = full.to(torch.float16)
        bad = ~(w16.float() == full)
        if bool(bad.any()):
            idx = bad.nonzero()[0].tolist()
            raise ValueError(f"{self.prefix}: wkv x 2^{WKV_BIAS_LOG2} is not exact in FP16 at {idx}: {full[tuple(idx)]} "
                             f"-> {w16[tuple(idx)]} (scale exponents [{int(e.min()) - 127}, {int(e.max()) - 127}])")
        self.wkv_r.data.copy_(w16)

    # ---- forward ----
    def forward(self, stream: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """stream [T_pad, 4, 5120] bf16, updated IN PLACE and returned (ref:m.py:350-365). ``positions`` is unused:
        the rows were hashed on the CPU from token ids (EngramHostService)."""
        t_pad = stream.shape[0]
        if stream.dtype != STREAM_DTYPE or tuple(stream.shape[1:]) != (HC, HIDDEN):
            raise ValueError(f"{self.prefix}: stream {tuple(stream.shape)} {stream.dtype}, expected [T, {HC}, "
                             f"{HIDDEN}] {STREAM_DTYPE}")
        rows = self.service.rows_buffer(self.layer_id, t_pad)
        _eager_wait()(self.service, self.layer_id, rows)
        row_bias = self.service.row_bias(self.layer_id)
        # each rank removes ITS OWN power-of-two biases before the all-reduce (exact), so ranks whose sub-tables
        # needed different row biases still sum consistently
        alpha = 2.0 ** -(WKV_BIAS_LOG2 + row_bias)
        if self.impl == "sm70":
            from vllm.models.deepseek_v41.sm70 import engram_kernels as ek
        for a in range(0, t_pad, self.prefill_chunk):
            b = min(t_pad, a + self.prefill_chunk)
            if self.impl == "sm70":
                x16 = ek.decode_rows(rows[a:b], row_bias)
            else:
                x16 = decode_rows_torch(rows[a:b], row_bias)
            part = torch.mm(x16, self.wkv_r.t(), out_dtype=torch.float32)          # FP16 operands, FP32 out
            part.mul_(alpha)
            if self.tp_size > 1:
                from vllm.distributed import tensor_model_parallel_all_reduce

                part = tensor_model_parallel_all_reduce(part)
            if self.impl == "sm70":
                ek.post_wkv_gate_(stream[a:b], part, self.qk, 1.0)
            else:
                post_wkv_gate_torch_(stream[a:b], part, self.qk, 1.0)
        return stream
