# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of VLLM_V41_INDEX_FUSE (vllm/models/deepseek_v41/index_fuse.py).

The CUDA scorer itself is checked on the GPU by
benchmarks/deepseek_v41/v41pidx/check.py. Here the scorer is today's per-tile
torch arithmetic, so these tests pin the knob parsing and the block / tile
bookkeeping of ``select``: the same all-reduce tiles as today and the same
top-k per row.
"""

import pytest
import torch

from vllm.models.deepseek_v41 import index_fuse


def test_knob_off_loads_nothing():
    seen = []

    def getenv(name):
        seen.append(name)
        return {"VLLM_V41_INDEX_FUSE": "0"}.get(name)

    assert index_fuse.from_env(getenv) is None
    assert seen == ["VLLM_V41_INDEX_FUSE"]
    assert index_fuse.from_env(lambda name: None) is None


def test_registered_defaults_are_off():
    from vllm import envs

    assert envs.VLLM_V41_INDEX_FUSE == "0"
    assert envs.VLLM_V41_INDEX_FUSE_LIB is None
    assert envs.VLLM_V41_INDEX_FUSE_MB == "64"
    assert index_fuse.from_env() is None


@pytest.mark.parametrize("value", ["1", "ilp3", "on", "SEQ"])
def test_bad_order_refuses(value):
    with pytest.raises(ValueError, match="VLLM_V41_INDEX_FUSE"):
        index_fuse.fuse_order(value)


def test_orders():
    assert [index_fuse.fuse_order(v) for v in ("seq", "ilp2", "ilp4", "tree")] == [
        0,
        1,
        2,
        3,
    ]
    assert index_fuse.fuse_order(" 0 ") is None
    assert index_fuse.fuse_order("") is None


def test_budget():
    assert index_fuse.fuse_budget(None) == 64 << 20
    assert index_fuse.fuse_budget("3") == 3 << 20
    for bad in ("0", "-1", "1.5", "x"):
        with pytest.raises(ValueError, match="VLLM_V41_INDEX_FUSE_MB"):
            index_fuse.fuse_budget(bad)


def test_on_without_library_refuses():
    env = {"VLLM_V41_INDEX_FUSE": "ilp4"}
    with pytest.raises(RuntimeError, match="VLLM_V41_INDEX_FUSE_LIB is not set"):
        index_fuse.from_env(env.get)
    env["VLLM_V41_INDEX_FUSE_LIB"] = "/nonexistent/v41pidx.so"
    with pytest.raises(RuntimeError, match="does not exist"):
        index_fuse.from_env(env.get)


def test_block_rows_whole_tiles():
    # 2 bytes per score: a 32-row tile over 1000 keys is 64000 bytes.
    assert index_fuse.block_rows(32, 1000, 64000 * 5 + 1) == 160
    assert index_fuse.block_rows(32, 1000, 1) == 32
    assert index_fuse.today_tile(64 << 20, 8, 1000) == 1024
    assert index_fuse.today_tile(1 << 20, 8, 100000) == 32


def _today_partial(q, w, keys):
    """select_compressed_positions' per-rank head sums for one tile."""
    rows, heads = q.shape[:2]
    s = torch.mm(q.reshape(-1, 128).float(), keys.T.float()).half()
    s = s.view(rows, heads, -1).relu_().mul_(w[:, :, None])
    return s.float().sum(1).half()


def _reference_topk(scores, positions, valid, ratio, k):
    width = scores.shape[1]
    out = torch.full((scores.shape[0], k), -1, dtype=torch.int32)
    for r in range(scores.shape[0]):
        n = int(positions[r] + 1) // ratio if bool(valid[r]) else 0
        n = min(n, width)
        top = scores[r, :n].float().topk(min(k, n)).indices.sort().values
        out[r, : top.numel()] = top.int()
    return out


@pytest.mark.parametrize("budget_tiles", [1, 3])
def test_select_matches_today_on_cpu(budget_tiles):
    torch.manual_seed(0)
    tokens, keys_n, live, ratio, top_k, tile = 70, 96, 80, 4, 16, 32
    q = (torch.randn(tokens, 8, 128) * 0.1).half()
    w = (torch.rand(tokens, 8) * 0.5).half()
    keys = (torch.randn(keys_n, 128) * 0.1).half()
    positions = torch.randint(64, live * ratio, (tokens,))
    valid = torch.ones(tokens, dtype=torch.bool)
    valid[5] = False
    tiles = []

    def all_reduce(x):
        tiles.append(tuple(x.shape))
        return x.clone()

    def scorer(out, qb, wb, kb, n):
        out.copy_(_today_partial(qb, wb, kb))
        out[:, n:] = 0

    def topk(out, scores, pos, val, rat, k):
        out.copy_(_reference_topk(scores, pos, val, rat, k))

    got = index_fuse.select(
        q,
        w,
        keys,
        live,
        positions,
        valid,
        ratio,
        all_reduce,
        top_k=top_k,
        tile=tile,
        budget=tile * keys_n * 2 * budget_tiles,
        scorer=scorer,
        index_topk=topk,
    )
    # Today: one all-reduce per 32-row tile, whatever the block size.
    assert tiles == [(32, keys_n), (32, keys_n), (6, keys_n)]
    want = torch.cat(
        [
            _reference_topk(
                _today_partial(q[a : a + tile], w[a : a + tile], keys),
                positions[a : a + tile],
                valid[a : a + tile],
                ratio,
                top_k,
            )
            for a in range(0, tokens, tile)
        ]
    )
    assert torch.equal(got, want)
    assert bool((got[5] == -1).all())


def test_select_shape_checks():
    q = torch.zeros(4, 8, 128, dtype=torch.half)
    w = torch.zeros(4, 8, dtype=torch.half)
    keys = torch.zeros(8, 128, dtype=torch.half)
    kw = dict(top_k=16, tile=32, budget=1 << 20, scorer=None, index_topk=None)
    with pytest.raises(ValueError, match="K >= top_k"):
        index_fuse.select(q, w, keys, 8, None, None, 4, None, **kw)
