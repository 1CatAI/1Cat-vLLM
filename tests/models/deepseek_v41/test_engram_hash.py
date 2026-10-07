# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram layout, token map and CPU hash vs the OFFICIAL inference/engram.py (lane L-ENGRAM; PORT_DESIGN §7.3).

The official module (rev 2cba9e42) is loaded by path from ``$DS41_REF_DIR/inference/engram.py`` and run on the CPU
with the official tokenizer; our numpy hash must match it bit for bit over >= 100K real tokens, with prefill chunks
of random length, single-token decode steps, dead (image-like) spans, every TP4 rank subset and a one-layer subset.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest

from .test_engram_synth import REF_DIR

MIN_TOKENS = int(os.environ.get("DS41_ENGRAM_HASH_TOKENS", "100000"))


def _official():
    path = REF_DIR / "inference" / "engram.py"
    if not path.is_file():
        pytest.skip(f"official engram.py not found at {path}")
    spec = importlib.util.spec_from_file_location("ds41_official_engram", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref_config() -> types.SimpleNamespace:
    p = REF_DIR / "config.json"
    if not p.is_file():
        pytest.skip(f"official config.json not found under {REF_DIR}")
    cfg = json.load(open(p))
    return types.SimpleNamespace(**cfg)  # nested text_config, exactly as shipped


@pytest.fixture(scope="module")
def hf_tokenizer():
    p = REF_DIR / "tokenizer.json"
    if not p.is_file():
        pytest.skip(f"official tokenizer.json not found under {REF_DIR}")
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=str(p))


def _official_args(tc: dict, max_seq_len: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        engram_layer_ids=tc["engram_layer_ids"], engram_max_ngram_size=tc["engram_max_ngram_size"],
        engram_n_heads=tc["engram_n_heads"], engram_vocab_size=tc["engram_vocab_size"],
        engram_num_embeddings=tc["engram_num_embeddings"], engram_head_dim=tc["engram_head_dim"],
        engram_compressed_vocab_size=tc["engram_compressed_vocab_size"], engram_pad_id=tc["engram_pad_token_id"],
        max_batch_size=1, max_seq_len=max_seq_len)


def test_layout_matches_official_and_gguf(ref_config) -> None:
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    off = _official()
    tc = ref_config.text_config
    layout = EngramLayout.from_hf_config(ref_config)
    olayout = off.EngramLayout.from_args(_official_args(tc, 1))
    assert layout.primes == tuple(tuple(p for o in lay for p in o) for lay in olayout.primes)
    assert layout.num_embeddings == (384006168, 384016682)
    assert layout.n_subtables == 24 and layout.pad_token_id == 2 and layout.compressed_vocab_size == 99092
    omult = off.compute_hash_multipliers(olayout.layer_ids, olayout.max_ngram_size, 99092).numpy()
    assert (np.array(layout.multipliers) == omult).all()
    # independent third source: the constants a GGUF converter wrote (kernelpool metadata, L-MODEL)
    gguf = Path("/mnt/nvme2/scratch/ds41/l-model/gguf/kernelpool.json")
    if gguf.is_file():
        kv = json.load(open(gguf))["kv"]
        assert list(kv["deepseek41.engram.primes"]) == [p for lay in layout.primes for p in lay]
        assert list(kv["deepseek41.engram.multipliers"]) == [m for lay in layout.multipliers for m in lay]


def test_token_map_matches_official(hf_tokenizer) -> None:
    from vllm.models.deepseek_v41.common.engram import build_compressed_token_map, load_compressed_token_map

    off = _official()
    ours, n = build_compressed_token_map(str(REF_DIR / "tokenizer.json"))
    theirs, m = off.build_compressed_token_map(hf_tokenizer)
    assert n == m == 99092 and ours.shape[0] == len(hf_tokenizer) == 129280
    assert (np.array(theirs, np.int64) == ours).all()
    assert (load_compressed_token_map(str(REF_DIR), 99092) == ours).all()
    with pytest.raises(ValueError, match="99091"):
        load_compressed_token_map(str(REF_DIR), 99091)


def _corpus(tok, min_tokens: int) -> list[np.ndarray]:
    files = sorted(glob.glob("/mnt/hdd/v100-research/docs/**/*.md", recursive=True))
    files += ["/mnt/nvme2/scratch/ds41/l-model/techreport.txt"]
    files += sorted(glob.glob("/usr/lib/python3*/json/*.py"))
    seqs, total = [], 0
    for f in files:
        if not os.path.isfile(f):
            continue
        ids = [0] + tok.encode(open(f, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)
        if len(ids) < 16:
            continue
        seqs.append(np.array(ids, np.int64))
        total += len(ids)
        if total >= min_tokens:
            break
    if total < min_tokens:
        pytest.skip(f"only {total} corpus tokens available (< {min_tokens})")
    return seqs


def test_hash_bit_exact_vs_official(ref_config, hf_tokenizer) -> None:
    import torch

    from vllm.models.deepseek_v41.common.engram import EngramHasher, EngramLayout, build_compressed_token_map

    off = _official()
    tc = ref_config.text_config
    layout = EngramLayout.from_hf_config(ref_config)
    tmap, _ = build_compressed_token_map(str(REF_DIR / "tokenizer.json"))
    olayout = off.EngramLayout.from_args(_official_args(tc, 1))
    rng = np.random.default_rng(1234)
    checked = dead = 0
    for si, ids in enumerate(_corpus(hf_tokenizer, MIN_TOKENS)):
        L = ids.shape[0]
        use_mask = si % 3 == 2
        mask = np.ones(L, dtype=bool)
        if use_mask:
            for _ in range(max(1, L // 2000)):
                a = int(rng.integers(0, L - 1))
                mask[a:a + int(rng.integers(1, 40))] = False
        state = off.NgramHashState(_official_args(tc, L), olayout, hf_tokenizer)
        full = EngramHasher(layout, tmap, layout.layer_ids, tuple(range(24)))
        ranks = [EngramHasher(layout, tmap, layout.layer_ids, layout.subtables_for_rank(r, 4)) for r in range(4)]
        l14 = EngramHasher(layout, tmap, (14,), layout.subtables_for_rank(1, 4))
        hist = np.empty(0, np.int64)
        a = 0
        while a < L:
            n = 1 if a > L - 40 else int(rng.integers(1, 4097))   # prefill chunks, then decode steps
            b = min(L, a + n)
            tm = torch.from_numpy(mask[a:b])[None] if use_mask else None
            ref = state(torch.from_numpy(ids[a:b])[None], a, tm)[0].numpy()     # [n, 2, 24]
            hist = np.concatenate([hist, full.compress(ids[a:b], mask[a:b] if use_mask else None)])
            pos = np.arange(a, b)
            np.testing.assert_array_equal(full.hash_positions(hist, pos), ref, err_msg=f"seq {si} [{a},{b})")
            for r, hr in enumerate(ranks):
                np.testing.assert_array_equal(hr.hash_positions(hist, pos), ref[:, :, list(hr.subtables)])
            np.testing.assert_array_equal(l14.hash_positions(hist, pos), ref[:, 1:2, list(l14.subtables)])
            checked += b - a
            dead += (b - a) if use_mask else 0
            a = b
    assert checked >= MIN_TOKENS and dead > 0


def test_hasher_rejects_bad_input(ref_config) -> None:
    from vllm.models.deepseek_v41.common.engram import EngramHasher, EngramLayout, build_compressed_token_map

    layout = EngramLayout.from_hf_config(ref_config)
    tmap, _ = build_compressed_token_map(str(REF_DIR / "tokenizer.json"))
    h = EngramHasher(layout, tmap, (1,), (0,))
    with pytest.raises(ValueError):
        h.compress(np.array([129280]))
    with pytest.raises(ValueError):
        h.hash_positions(np.zeros(4, np.int64), np.array([4]))
    with pytest.raises(ValueError):
        EngramHasher(layout, tmap[:-5].copy() * 0, (1,), (0,))
