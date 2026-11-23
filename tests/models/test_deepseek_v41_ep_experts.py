# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 native experts under expert parallelism.

Every rank computes only its own experts for all (gathered) tokens; the
reduce-scatter and the TP all-reduce then sum the ranks. The sum of the
per-rank outputs must match one rank holding every expert (up to the FP32
order of the route sums), for any routing, including tokens routed to no
local expert and ranks with no routed slot at all.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.models.deepseek_v41.native")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from vllm.models.deepseek_v41 import native  # noqa: E402

H, INTER, TOPK = 5120, 2304, 6
RANKS, PER_RANK = 4, 4
EXPERTS = RANKS * PER_RANK


def _raw_weights(gen, experts):
    def codes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)

    def scales(*shape):  # E8M0 around 2^-5 .. 2^-2
        return torch.randint(122, 126, shape, dtype=torch.uint8, generator=gen)

    return (
        codes(experts, 2 * INTER, H // 2),
        scales(experts, 2 * INTER, H // 32),
        codes(experts, H, INTER // 2),
        scales(experts, H, INTER // 32),
    )


def _layer(raw, first, count, expert_map=None):
    w13, s13, w2, s2 = (t[first : first + count].clone().cuda() for t in raw)
    layer = SimpleNamespace(
        w13_weight=w13,
        w13_weight_scale=s13,
        w2_weight=w2,
        w2_weight_scale=s2,
        expert_map=expert_map,
    )
    method = object.__new__(native.V41NativeMoEMethod)
    method.process_weights_after_loading(layer)
    return method, layer


def _routes(gen, tokens, only=None):
    pool = torch.arange(EXPERTS) if only is None else torch.tensor(only)
    ids = torch.stack(
        [pool[torch.randperm(len(pool), generator=gen)[:TOPK]] for _ in range(tokens)]
    )
    weights = torch.rand(tokens, TOPK, generator=gen) + 0.1
    weights = weights / weights.sum(-1, keepdim=True) * 1.5
    return ids.int().cuda(), weights.float().cuda()


@pytest.mark.parametrize("tokens", [1, 6, 40, 200, 800])
@pytest.mark.parametrize("pool", ["all", "two_ranks"])
def test_rank_sum_matches_all_experts(tokens, pool):
    gen = torch.Generator().manual_seed(1000 + tokens)
    raw = _raw_weights(gen, EXPERTS)
    full_method, full_layer = _layer(raw, 0, EXPERTS)
    only = None if pool == "all" else list(range(2 * PER_RANK))  # ranks 2, 3 idle
    if only is not None and len(only) < TOPK:
        only = None
    ids, weights = _routes(gen, tokens, only)
    x = (torch.randn(tokens, H, generator=gen) * 0.5).half().cuda()
    expected = full_method.apply(full_layer, x, weights, ids)
    total = torch.zeros_like(expected)
    for rank in range(RANKS):
        expert_map = torch.full((EXPERTS,), -1, dtype=torch.int32)
        expert_map[rank * PER_RANK : (rank + 1) * PER_RANK] = torch.arange(
            PER_RANK, dtype=torch.int32
        )
        method, layer = _layer(raw, rank * PER_RANK, PER_RANK, expert_map.cuda())
        out = method.apply(layer, x, weights, ids)
        assert out.dtype == torch.float32 and out.shape == expected.shape
        assert torch.isfinite(out).all()
        total += out
    torch.accelerator.synchronize()
    scale = expected.abs().max().clamp_min(1e-6)
    # The expert-parallel path runs other kernels than the single-rank
    # reference (fused gate/up + SwiGLU, the decode w2 kernel): same math,
    # other FP32 accumulation orders.
    assert ((total - expected).abs().max() / scale) < 1e-3


def test_rank_with_no_local_route_is_zero():
    gen = torch.Generator().manual_seed(7)
    raw = _raw_weights(gen, EXPERTS)
    ids, weights = _routes(gen, 6, only=list(range(PER_RANK, 3 * PER_RANK)))
    x = torch.randn(6, H, generator=gen).half().cuda()
    expert_map = torch.full((EXPERTS,), -1, dtype=torch.int32)
    expert_map[:PER_RANK] = torch.arange(PER_RANK, dtype=torch.int32)
    method, layer = _layer(raw, 0, PER_RANK, expert_map.cuda())
    out = method.apply(layer, x, weights, ids)
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("rows,real", [(6, 1), (6, 0), (40, 37), (200, 200)])
def test_padded_rows_route_nowhere(rows, real):
    gen = torch.Generator().manual_seed(rows + real)
    raw = _raw_weights(gen, EXPERTS)
    ids, weights = _routes(gen, rows)
    x = (torch.randn(rows, H, generator=gen) * 0.5).half().cuda()
    real_rows = torch.tensor([real], dtype=torch.int32, device="cuda")
    masked = native.mask_padded_routes(ids.clone(), real_rows, EXPERTS)
    assert (masked[real:] == EXPERTS).all()
    assert torch.equal(masked[:real], ids[:real])
    full_method, full_layer = _layer(raw, 0, EXPERTS)
    expected = full_method.apply(full_layer, x, weights, ids)
    total = torch.zeros_like(expected)
    for rank in range(RANKS):
        expert_map = torch.full((EXPERTS,), -1, dtype=torch.int32)
        expert_map[rank * PER_RANK : (rank + 1) * PER_RANK] = torch.arange(
            PER_RANK, dtype=torch.int32
        )
        method, layer = _layer(raw, rank * PER_RANK, PER_RANK, expert_map.cuda())
        total += method.apply(layer, x, weights, masked)
    torch.accelerator.synchronize()
    assert torch.count_nonzero(total[real:]) == 0
    if real:
        scale = expected[:real].abs().max().clamp_min(1e-6)
        assert ((total[:real] - expected[:real]).abs().max() / scale) < 1e-3


@pytest.mark.parametrize("local", [4, 24, 96])
@pytest.mark.parametrize("slots", [6, 144, 4608, 24480])
def test_ep_plan_groups_owned_routes(local, slots):
    """Each group lists exactly the slots routed to its (owned) expert, groups
    in expert order; routes to other ranks' experts and padded ids form no
    group; offsets past the groups end at the owned count."""
    gen = torch.Generator().manual_seed(local * 7 + slots)
    total = 384
    expert_map = torch.full((total,), -1, dtype=torch.int32)
    owned_ids = torch.randperm(total, generator=gen)[:local]
    expert_map[owned_ids] = torch.arange(local, dtype=torch.int32)
    ids = torch.randint(0, total + 1, (slots,), generator=gen, dtype=torch.int32)
    ids[::7] = total  # padded rows
    local_ids, perm, experts, offsets = native._ep_plan(
        ids.cuda().view(-1, 1), expert_map.cuda(), local
    )
    local_ids, perm = local_ids.cpu(), perm.cpu()
    experts, offsets = experts.cpu(), offsets.cpu()
    expected = torch.where(ids < total, expert_map[ids.clamp(max=total - 1)].long(), -1)
    expected = torch.where(expected < 0, local, expected)
    assert torch.equal(local_ids.long(), expected)
    hit = sorted(set(expected[expected < local].tolist()))
    assert experts[: len(hit)].tolist() == hit
    owned = int((expected < local).sum())
    assert (offsets[len(hit) :] == owned).all()
    for g, expert in enumerate(hit):
        rows = perm[offsets[g] : offsets[g + 1]].tolist()
        assert sorted(rows) == (expected == expert).nonzero().flatten().tolist()


def test_ep4_tp4_reconstructs_full_experts():
    gen = torch.Generator().manual_seed(41)
    raw = _raw_weights(gen, EXPERTS)
    ids, probs = _routes(gen, 6)
    x = (torch.randn(6, H, generator=gen) * 0.5).half().cuda()
    method, layer = _layer(raw, 0, EXPERTS)
    expected = method.apply(layer, x, probs, ids)
    total = torch.zeros_like(expected, dtype=torch.float32)
    width = INTER // 4
    for tp in range(4):
        begin, end = tp * width, (tp + 1) * width
        shards = [
            torch.cat([t[:, begin:end], t[:, INTER + begin : INTER + end]], dim=1)
            for t in raw[:2]
        ]
        shards.extend(t[:, :, begin // pack : end // pack] for t, pack in zip(raw[2:], (2, 32)))
        for ep in range(4):
            first = ep * PER_RANK
            mapping = torch.full((EXPERTS,), -1, dtype=torch.int32)
            mapping[first:first + PER_RANK] = torch.arange(PER_RANK)
            shard_method, shard = _layer(shards, first, PER_RANK, mapping.cuda())
            total += shard_method.apply(shard, x, probs, ids)
    assert torch.isfinite(total).all()
    # Each rank rounds its W2 partial to FP16 before the FP32 combine.
    assert (total - expected).abs().max() / expected.abs().max() < 1e-3


@pytest.mark.parametrize("bound", [128, 512])
def test_solo_short_indices_match_index_topk(bound):
    from vllm.models.deepseek_v41.index_topk import index_topk
    from vllm.models.deepseek_v41.short_indices import all_compressed_positions

    positions = torch.tensor([0, 1, 31, 32, bound - 1, bound - 1], device="cuda")
    valid = torch.tensor([True] * 5 + [False], device="cuda")
    scores = torch.randn((6, 512), device="cuda", dtype=torch.float16)
    for ratio, compact in all_compressed_positions(positions, valid, bound).items():
        expected = torch.empty((6, 512), device="cuda", dtype=torch.int32)
        index_topk(expected, scores, positions, valid, ratio)
        width = compact.size(1)
        assert torch.equal(compact, expected[:, :width])
        assert torch.all(expected[:, width:] == -1)


@pytest.mark.parametrize(("rows", "block"), [(6, 512), (24, 1024), (96, 1024)])
def test_dp_unpack_maps_replicas_by_gathered_row(rows, block):
    hidden, top_k, logical, dead = 5120, 6, 7, 40
    width = hidden + 4 * top_k
    tiles = hidden // block
    meta = dict(H=hidden, TOPK=top_k, WIDTH=width, BLOCK=block, TILES=tiles, KB=8)
    # Five copies make the correct selection depend on the sender row offset.
    copies = torch.arange(logical * 5, device="cuda", dtype=torch.int32).reshape(-1, 5)
    counts = torch.tensor([1, 2, 5, 1, 2, 3, 1], device="cuda", dtype=torch.int32)
    x = torch.randn(4 * rows, hidden, device="cuda", dtype=torch.float16)
    weights = torch.rand(4 * rows, top_k, device="cuda")
    logical_ids = torch.arange(4 * rows * top_k, device="cuda", dtype=torch.int32)
    logical_ids = (logical_ids.reshape(-1, top_k) % (logical + 2)) - 1
    packed = torch.empty((4 * rows, width), device="cuda", dtype=torch.float16)
    masked = logical_ids.clone()
    for rank, live in enumerate((rows, 0, rows - 2, rows - 1)):
        start = rank * rows
        part = slice(start, start + rows)
        real = torch.tensor([live], dtype=torch.int32, device="cuda")
        native._pack_routes_kernel[(rows, tiles)](
            x[part], weights[part], logical_ids[part], real, packed[part], dead, **meta
        )
        masked[start + live : start + rows] = dead
    got_x, got_w, got_ids = torch.empty_like(x), torch.empty_like(weights), torch.empty_like(masked)
    native._unpack_routes_kernel[(4 * rows, tiles)](
        packed, got_x, got_w, got_ids, copies=copies, counts=counts,
        REPLICAS=5, LOGICAL=logical, DEAD=dead, **meta,
    )
    expected = native.map_replica_routes(masked, copies, counts, dead)
    assert torch.equal(got_x, x)
    assert torch.equal(got_w, weights)
    assert torch.equal(got_ids, expected)
