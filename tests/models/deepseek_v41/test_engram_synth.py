# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic Engram tables with the REAL layout at small scale (lane L-ENGRAM test helper + its self-checks).

Same tensor names, dtypes and widths as official shards 47/48 (``layers.{L}.engram.embed.{weight,scale}``, F8_E4M3
[N, 256] / F8_E8M0 [N, 8]); the header is padded so ``embed.weight`` starts at byte 664 (layer 1) / 672 (layer 14)
exactly like the official files, which makes 1 row in 16 straddle a 4 KiB page. The bucket layout comes from the
same prime search with a small ``engram_vocab_size``.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REAL_DATA_START = {1: 664, 14: 672}
REF_DIR = Path(os.environ.get("DS41_REF_DIR", "/mnt/nvme2/scratch/ds41/model-ref"))


def synthetic_hf_config(vocab: int = 1000) -> SimpleNamespace:
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    probe = SimpleNamespace(engram_layer_ids=[1, 14], engram_max_ngram_size=4, engram_n_heads=8, engram_head_dim=256,
                            engram_vocab_size=vocab, engram_num_embeddings=[0, 0],
                            engram_compressed_vocab_size=99092, engram_pad_token_id=2)
    # derive num_embeddings from the prime search itself (from_hf_config checks the sums)
    sums = []
    from vllm.models.deepseek_v41.common import engram as E

    seen: set[int] = set()
    for _ in range(2):
        flat = []
        for _o in range(3):
            cur = vocab - 1
            for _h in range(8):
                cur = E._next_unseen_prime(cur, seen)
                seen.add(cur)
                flat.append(cur)
        sums.append(sum(flat))
    probe.engram_num_embeddings = sums
    EngramLayout.from_hf_config(probe)
    return probe


def random_e4m3_bytes(rng: np.random.Generator, n: int) -> np.ndarray:
    b = rng.integers(0, 256, size=n, dtype=np.uint8)
    nan = (b & 0x7F) == 0x7F
    b[nan] ^= 0x01            # 0x7F -> 0x7E: never a NaN in a synthetic table
    return b


@dataclass
class SyntheticEngram:
    hf_config: SimpleNamespace
    row_dir: Path
    weights: dict[int, np.ndarray]   # layer -> [N, 256] uint8
    scales: dict[int, np.ndarray]    # layer -> [N, 8] uint8


def write_synthetic_shard(path: Path, layer_id: int, weight: np.ndarray, scale: np.ndarray,
                          data_start: int) -> None:
    n = weight.shape[0]
    hdr = {
        f"layers.{layer_id}.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [n, 256],
                                                    "data_offsets": [0, weight.nbytes]},
        f"layers.{layer_id}.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [n, 8],
                                                   "data_offsets": [weight.nbytes, weight.nbytes + scale.nbytes]},
    }
    raw = json.dumps(hdr, separators=(",", ":")).encode()
    pad = data_start - 8 - len(raw)
    assert pad >= 0, f"header {len(raw)} B does not fit before {data_start}"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw) + pad))
        f.write(raw + b" " * pad)
        f.write(weight.tobytes())
        f.write(scale.tobytes())
        f.flush()
        os.fsync(f.fileno())   # this file only (never a global sync)


def build_synthetic_engram(tmp: Path, seed: int = 0, vocab: int = 1000,
                           exp_range: tuple[int, int] = (-10, -6)) -> SyntheticEngram:
    cfg = synthetic_hf_config(vocab)
    rng = np.random.default_rng(seed)
    weights, scales = {}, {}
    tmp.mkdir(parents=True, exist_ok=True)
    for i, lid in enumerate(cfg.engram_layer_ids):
        n = cfg.engram_num_embeddings[i]
        w = random_e4m3_bytes(rng, n * 256).reshape(n, 256)
        s = rng.integers(127 + exp_range[0], 127 + exp_range[1] + 1, size=(n, 8), dtype=np.uint8)
        write_synthetic_shard(tmp / f"model-0004{7 + i}-of-00048.safetensors", lid, w, s, REAL_DATA_START[lid])
        weights[lid], scales[lid] = w, s
    return SyntheticEngram(cfg, tmp, weights, scales)


def tokenizer_path() -> str:
    p = REF_DIR / "tokenizer.json"
    if not p.is_file():
        pytest.skip(f"official tokenizer.json not found under {REF_DIR}")
    return str(p)


# ------------------------------------------------------------------------------ self-checks of the helper
def test_synthetic_layout_matches_real_geometry(tmp_path: Path) -> None:
    from safetensors import safe_open

    from vllm.models.deepseek_v41.common.engram_host import locate_engram_tables
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    syn = build_synthetic_engram(tmp_path / "syn")
    layout = EngramLayout.from_hf_config(syn.hf_config)
    locs = locate_engram_tables(str(syn.row_dir), layout, (1, 14))
    assert [loc.weight_offset for loc in locs] == [664, 672]
    for loc in locs:
        with safe_open(loc.weight_path, framework="np") as f:
            assert f"layers.{loc.layer_id}.engram.embed.weight" in f.keys()
        assert loc.scale_offset == loc.weight_offset + loc.num_rows * 256
