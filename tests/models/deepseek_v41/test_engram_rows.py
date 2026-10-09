# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rows served in place from the official shards 47/48 == rows read directly through safetensors (lane L-ENGRAM).

PORT_DESIGN §7.3 (row store), adapted to DECISIONS D4/D5 (no repacked file): for every TP4 rank and both Engram
layers, random row ids of the rank's sub-tables plus the ids of real text are gathered by the C++ reader through
EngramHostService and compared byte for byte with ``safe_open(...).get_slice`` reads of ``embed.weight`` and
``embed.scale``; the rank's page-locked scale table is compared whole. Runs only on sha256-verified shards.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from .test_engram_synth import tokenizer_path

pytestmark = [pytest.mark.sm70, pytest.mark.weights]

ENGRAM_DIR = Path(os.environ.get("DS41_ENGRAM_DIR", "/home/mermiges/ds41-engram"))
SHARDS = {1: "model-00047-of-00048.safetensors", 14: "model-00048-of-00048.safetensors"}
N_RANDOM = int(os.environ.get("DS41_ENGRAM_ROWS_RANDOM", "200000"))


def _verified(name: str) -> Path:
    p = ENGRAM_DIR / name
    if not (p.is_file() and Path(str(p) + ".sha256-ok").is_file()):
        pytest.skip(f"{p} not downloaded + sha256-verified yet")
    out = subprocess.run(["bash", "/mnt/nvme2/models/_dl-logs/verify_shard.sh", name], capture_output=True,
                         text=True, timeout=900)
    if out.returncode != 0 or not out.stdout.strip().startswith("OK"):
        pytest.fail(f"verify_shard.sh {name}: {out.stdout} {out.stderr}")
    return p


@pytest.fixture(scope="module")
def hf_config():
    import json
    import types

    return types.SimpleNamespace(**json.load(open(Path(tokenizer_path()).parent / "config.json")))


@pytest.mark.parametrize("o_direct", ["0", "1"])
@pytest.mark.parametrize("layer", [1, 14])
def test_rows_in_place_equal_safetensors(hf_config, monkeypatch: pytest.MonkeyPatch, layer: int,
                                         o_direct: str) -> None:
    from safetensors import safe_open

    from vllm.models.deepseek_v41.common.engram_host import EngramHostService, locate_engram_tables

    shard = _verified(SHARDS[layer])
    monkeypatch.setenv("VLLM_DS41_ENGRAM_O_DIRECT", o_direct)
    rng = np.random.default_rng(layer * 10 + int(o_direct))
    n_rand = N_RANDOM if o_direct == "0" else N_RANDOM // 10
    with safe_open(str(shard), framework="pt") as f:
        w = f.get_slice(f"layers.{layer}.engram.embed.weight")
        sc = f.get_slice(f"layers.{layer}.engram.embed.scale")
        per = n_rand // 6
        for rank in range(4):
            # staging sized for the direct gather below: (per + 2) tokens x 6 sub-tables
            svc = EngramHostService(hf_config, (layer,), rank, 4, str(ENGRAM_DIR), tokenizer_path(), per + 2,
                                    torch.device("cuda"), io_threads=4)
            try:
                loc = locate_engram_tables(str(ENGRAM_DIR), svc.layout, (layer,))[0]
                assert loc.weight_offset == {1: 664, 14: 672}[layer]
                # random ids of each owned sub-table (incl. both ends of every bucket range)
                li = svc.layout.layer_index(layer)
                rows = np.empty((per + 2, 1, svc.n_sub), np.int64)
                for j, s in enumerate(svc.subtables):
                    lo, n = svc.layout.offsets[li][s], svc.layout.primes[li][s]
                    rows[:per, 0, j] = lo + rng.integers(0, n, size=per)
                    rows[per:, 0, j] = (lo, lo + n - 1)
                inv, n_u, ticket = svc._submit_rows(rows, 0, 1)
                svc._reader.wait(ticket)
                staged = svc._host_rows[0, 1:1 + n_u].numpy()
                got = staged[inv.reshape(-1) - 1].reshape(rows.shape + (264,))
                check = rng.choice(rows.size, size=min(rows.size, 20000), replace=False)
                flat_ids = rows.reshape(-1)[check]
                flat_got = got.reshape(-1, 264)[check]
                for t, rid in enumerate(flat_ids):
                    rid = int(rid)
                    assert np.array_equal(flat_got[t, :256], w[rid:rid + 1].view(torch.uint8).numpy()[0]), rid
                    assert np.array_equal(flat_got[t, 256:], sc[rid:rid + 1].view(torch.uint8).numpy()[0]), rid
                # the page-locked scale table == the rank's sub-table ranges of embed.scale
                for j, s in enumerate(svc.subtables):
                    lo, n = svc.layout.offsets[li][s], svc.layout.primes[li][s]
                    base = int(svc._scale_base[0, j]) * 8
                    a = svc._scales[0].array[base:base + n * 8]
                    probe = rng.integers(0, n, size=64)
                    for p in probe:
                        assert np.array_equal(a[p * 8:p * 8 + 8], sc[lo + p:lo + p + 1].view(torch.uint8).numpy()[0])
                e_lo, e_hi = svc.scale_exponent_range[layer]
                assert -15 <= e_lo and e_hi <= 7 and svc.row_bias(layer) == 0, svc.scale_exponent_range
                print(f"L{layer} rank {rank} o_direct={o_direct}: {rows.size} ids, {n_u} unique, "
                      f"scale exponents [{e_lo}, {e_hi}], pinned {svc.pinned_scale_bytes / 2**30:.3f} GiB")
            finally:
                svc.shutdown()
