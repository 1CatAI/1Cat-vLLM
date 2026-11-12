# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_BATCH kernels against the single-request kernels, request by request.

Each check builds R requests with their own block-table rows and state slots,
runs the batched op once and the single-request op once per request (its table
row sliced to [1, blocks], its own single state), and compares. Run with the
server stopped: CUDA_VISIBLE_DEVICES=0 python batch_kernel_check.py
"""

import random
import sys

import torch

from vllm.model_executor.kernels.deepseek_v41_compression import (
    V41C2State,
    V41C2StateBatch,
)
from vllm.model_executor.layers.deepseek_v41_engram import (
    V41EngramHashBatch,
    V41EngramHashState,
)
from vllm.models.deepseek_v41 import attention_ops as ops
from vllm.models.deepseek_v41.cache import V41BatchLayout

DEV = torch.device("cuda")
FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def layout_of(lengths, prefill=None):
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    rows = [r for r, n in enumerate(lengths) for _ in range(n)]
    bases = [starts[r] for r, n in enumerate(lengths) for _ in range(n)]
    qsl = torch.tensor(starts, dtype=torch.int32)
    flags = torch.tensor(prefill or [False] * len(lengths), dtype=torch.bool)
    return V41BatchLayout(
        len(lengths),
        starts[-1],
        torch.tensor(rows, dtype=torch.int32, device=DEV),
        torch.tensor(bases, dtype=torch.int32, device=DEV),
        flags.to(DEV),
        qsl.to(DEV),
        qsl,
    )


def engram_check(reqs=5, steps=6, seed=0):
    rnd = random.Random(seed)
    vocab = 4096
    token_map = torch.randint(0, 1000, (vocab,), dtype=torch.int32, device=DEV)
    primes = torch.randint(1000, 100000, (2, 24), dtype=torch.int64, device=DEV)
    offsets = torch.randint(0, 1000, (2, 24), dtype=torch.int64, device=DEV)
    multipliers = torch.randint(1, 1 << 20, (2, 4), dtype=torch.int64, device=DEV)
    make = lambda: V41EngramHashState(  # noqa: E731
        token_map, primes, offsets, multipliers, max_tokens=2048
    )
    singles = [make() for _ in range(reqs)]
    batch = V41EngramHashBatch(make(), 16)
    slots = torch.tensor(
        [3 * r + 1 for r in range(reqs)], dtype=torch.int32, device=DEV
    )
    position = [0] * reqs
    worst = 0
    for step in range(steps):
        lengths = [rnd.randint(7, 40) if step == 0 else 6 for _ in range(reqs)]
        ids = [torch.randint(0, vocab, (n,), device=DEV) for n in lengths]
        pos = [torch.arange(p, p + n, device=DEV) for p, n in zip(position, lengths)]
        layout = layout_of(lengths, [step == 0] * reqs)
        flat_ids, flat_pos = torch.cat(ids), torch.cat(pos)
        mask = torch.ones_like(flat_ids, dtype=torch.bool)
        got = batch(flat_ids, flat_pos, mask, layout, slots)
        want = torch.cat([s(i, p, None) for s, i, p in zip(singles, ids, pos)])
        worst = max(worst, int((got != want).any(-1).any(-1).sum()))
        counts = [n if step == 0 else rnd.randint(1, 6) for n in lengths]
        selector = torch.tensor(counts, dtype=torch.int64, device=DEV)
        batch.commit(selector, slots)
        for s, c in zip(singles, counts):
            s.commit(torch.tensor([c], device=DEV))
        position = [p + c for p, c in zip(position, counts)]
    check("engram hashes", worst == 0, f"mismatched tokens {worst}")


def c2_check(reqs=5, steps=6, seed=1):
    rnd = random.Random(seed)
    norm = torch.rand(512, device=DEV) + 0.5
    singles = [V41C2State(norm) for _ in range(reqs)]
    batch = V41C2StateBatch(V41C2State(norm), 16)
    slots = torch.tensor(
        [2 * r + 3 for r in range(reqs)], dtype=torch.int32, device=DEV
    )
    position = [0] * reqs
    worst = 0.0
    for step in range(steps):
        lengths = [rnd.randint(7, 40) if step == 0 else 6 for _ in range(reqs)]
        kv = [torch.randn(n, 512, device=DEV) for n in lengths]
        sc = [torch.randn(n, 512, device=DEV) for n in lengths]
        pos = [torch.arange(p, p + n, device=DEV) for p, n in zip(position, lengths)]
        layout = layout_of(lengths, [step == 0] * reqs)
        mask = torch.ones(sum(lengths), dtype=torch.bool, device=DEV)
        got, done = batch(
            torch.cat(kv), torch.cat(sc), torch.cat(pos), mask, layout, slots
        )
        want = [s(k, c, p) for s, k, c, p in zip(singles, kv, sc, pos)]
        want_v = torch.cat([w[0] for w in want])
        want_d = torch.cat([w[1] for w in want])
        live = want_d[:, None]
        worst = max(worst, float(((got - want_v).abs() * live).max()))
        check(f"c2 step {step} complete flags", torch.equal(done, want_d))
        counts = [n if step == 0 else rnd.randint(1, 6) for n in lengths]
        batch.commit(torch.tensor(counts, dtype=torch.int64, device=DEV), slots)
        for s, c in zip(singles, counts):
            s.commit(torch.tensor([c], device=DEV))
        position = [p + c for p, c in zip(position, counts)]
    check("c2 pooled rows", worst == 0.0, f"max abs diff {worst}")


def caches(reqs, blocks, block_size, dim, ratio=1, pages=None):
    pages = pages or reqs * blocks + 3
    cache = torch.randn(pages, block_size // ratio, dim, device=DEV).half()
    perm = torch.randperm(pages - 3, device=DEV)[: reqs * blocks] + 3
    table = perm.view(reqs, blocks).to(torch.int32).contiguous()
    return cache, table


def insert_check(reqs=4, seed=2):
    torch.manual_seed(seed)
    block = 64
    cache, table = caches(reqs, 8, block, 512)
    reference = cache.clone()
    lengths = [5, 11, 6, 6][:reqs]
    starts = [100, 3, 250, 17][:reqs]
    pos = torch.cat(
        [torch.arange(s, s + n, device=DEV) for s, n in zip(starts, lengths)]
    )
    values = torch.randn(pos.numel(), 512, device=DEV).half()
    valid = torch.ones(pos.numel(), dtype=torch.bool, device=DEV)
    layout = layout_of(lengths)
    ops.insert_latents_batch(cache, values, pos, valid, table, layout, block_size=block)
    first = 0
    for r, n in enumerate(lengths):
        ops.insert_latents(
            reference,
            values[first : first + n],
            pos[first : first + n],
            valid[first : first + n],
            table[r : r + 1],
            block_size=block,
        )
        first += n
    check("insert latents", torch.equal(cache, reference))


def attention_check(reqs=4, seed=3, lengths=(6, 6, 6, 6), name="paged attention"):
    torch.manual_seed(seed)
    block, ratio = 64, 4
    window, cwin = caches(reqs, 12, block, 512)
    comp, ctab = caches(reqs, 12, block, 512, ratio=ratio)
    lengths = list(lengths)[:reqs]
    starts = [300, 140, 470, 133][:reqs]
    pos = torch.cat(
        [torch.arange(s, s + n, device=DEV) for s, n in zip(starts, lengths)]
    )
    tokens = pos.numel()
    q = (torch.randn(tokens, 16, 512, device=DEV) * 0.3).half()
    sink = torch.randn(16, device=DEV)
    valid = torch.ones(tokens, dtype=torch.bool, device=DEV)
    topk = torch.randint(-1, 64, (tokens, 512), device=DEV, dtype=torch.int32)
    layout = layout_of(lengths)
    got = ops.attend_paged_latents_batch(
        q,
        pos,
        valid,
        window,
        cwin,
        sink,
        layout,
        window_block=block,
        compressed_cache=comp,
        compressed_table=ctab,
        compressed_block=block,
        ratio=ratio,
        topk=topk,
    )
    first, worst = 0, 0.0
    for r, n in enumerate(lengths):
        sl = slice(first, first + n)
        want = ops.attend_paged_latents(
            q[sl].contiguous(),
            pos[sl],
            valid[sl],
            window,
            cwin[r : r + 1],
            sink,
            window_block=block,
            compressed_cache=comp,
            compressed_table=ctab[r : r + 1],
            compressed_block=block,
            ratio=ratio,
            topk=topk[sl].contiguous(),
        )
        worst = max(worst, float((got[sl].float() - want.float()).abs().max()))
        first += n
    check(name, worst < 2e-3, f"max abs diff {worst:.2e}")


def attention_prefill_check():
    """attend_paged_latents_batch on prefill-sized rows (attn_paged_tok when
    VLLM_V41_ATTN_RING_PREFILL=1) against the per-request reference."""
    attention_check(3, 13, (40, 6, 100), "paged attention prefill rows")


def index_check(reqs=4, seed=4):
    torch.manual_seed(seed)
    block, ratio = 64, 4
    keys, table = caches(reqs, 64, block, 128, ratio=ratio)
    lengths = [6, 6, 9, 6][:reqs]
    starts = [3000, 700, 1500, 64][:reqs]
    pos = torch.cat(
        [torch.arange(s, s + n, device=DEV) for s, n in zip(starts, lengths)]
    )
    tokens = pos.numel()
    q = (torch.randn(tokens, 16, 128, device=DEV) * 0.5).half()
    w = (torch.rand(tokens, 16, device=DEV) * 0.1).half()
    valid = torch.ones(tokens, dtype=torch.bool, device=DEV)
    layout = layout_of(lengths)
    identity = lambda x: x  # noqa: E731
    got = ops.select_compressed_positions_batch(
        q,
        w,
        keys,
        table,
        pos,
        valid,
        layout,
        ratio,
        identity,
        block_size=block,
        max_context=4096,
    )
    first, overlap, total = 0, 0, 0
    for r, n in enumerate(lengths):
        sl = slice(first, first + n)
        gathered = ops.gather_index_keys(
            keys,
            table[r : r + 1],
            pos[sl],
            valid[sl],
            block_size=block,
            ratio=ratio,
            max_context=4096,
        )
        want = ops.select_compressed_positions(
            q[sl], w[sl], gathered, pos[sl], valid[sl], ratio, identity
        )
        for t in range(n):
            a = set(got[first + t].tolist()) - {-1}
            b = set(want[t].tolist()) - {-1}
            overlap += len(a & b)
            total += max(len(b), 1)
        first += n
    check("index top-k", overlap / total > 0.995, f"overlap {overlap / total:.4f}")


def index_scores_check(seed=6):
    """v41sk.index_scores against the Triton _bindex_scores, H=8 (TP4), and
    their CUDA-graph times at a short and a long context."""
    torch.manual_seed(seed)
    block, ratio, heads, columns = 64, 4, 8, 4096
    for reqs, start in ((1, 200), (4, 3000), (16, 12000), (64, 1500)):
        keys, table = caches(reqs, 256, block, 128, ratio=ratio)
        lengths = [6] * reqs
        starts = [start - 37 * r % 500 for r in range(reqs)]
        pos = torch.cat(
            [torch.arange(a, a + n, device=DEV) for a, n in zip(starts, lengths)]
        )
        tokens = pos.numel()
        q = (torch.randn(tokens, heads, 128, device=DEV) * 0.5).half()
        w = (torch.rand(tokens, heads, device=DEV) * 0.1).half()
        layout = layout_of(lengths)
        want = torch.zeros(tokens, columns, dtype=torch.float16, device=DEV)
        got = torch.zeros_like(want)
        triton_run = lambda: ops._bindex_scores_launch(  # noqa: E731
            want, q, w, keys, table, pos, layout, columns, ratio, heads, block
        )
        cuda_run = lambda: torch.ops.v41sk.index_scores(  # noqa: E731
            got, q, w, keys, table, pos, layout.query_start_loc, reqs, ratio, 32
        )
        triton_run()
        cuda_run()
        torch.accelerator.synchronize()
        diff = (got.float() - want.float()).abs().max().item()
        scale = want.float().abs().max().item()
        times = []
        for run in (triton_run, cuda_run):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
                    run()
            graph.replay()
            torch.accelerator.synchronize()
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            for _ in range(10):
                graph.replay()
            end.record()
            torch.accelerator.synchronize()
            times.append(begin.elapsed_time(end) / 200 * 1000)
        check(
            f"index scores cuda reqs={reqs} ctx~{start}",
            diff <= 2e-3 * max(scale, 1.0),
            f"max diff {diff:.2e} (scale {scale:.2f}) triton {times[0]:.1f} us cuda {times[1]:.1f} us",
        )


def index_shard_check(seed=10):
    """VLLM_V41_INDEX_SHARD (_sharded_index_topk, TP4 emulated on one GPU)
    against a full-head reference top-512, short and long contexts."""
    torch.manual_seed(seed)
    world, heads, block, top_k = 4, 8, 64, 512
    for ratio, reqs, starts in ((1, 2, (300, 41000)), (2, 3, (900, 70000, 5000))):
        columns = 131072 // ratio
        keys, table = caches(reqs, columns * ratio // block, block, 128, ratio=ratio)
        lengths = [6] * reqs
        pos = torch.cat([torch.arange(a, a + 6, device=DEV) for a in starts])
        tokens = pos.numel()
        valid = torch.ones(tokens, dtype=torch.bool, device=DEV)
        valid[3] = False
        q_full = (torch.randn(tokens, heads * world, 128, device=DEV) * 0.5).half()
        w_full = (torch.rand(tokens, heads * world, device=DEV) * 0.1).half()
        layout = layout_of(lengths)
        rows, store, seen = {}, {}, {}

        # Emulated TP gather: each pass serves gathers whose inputs every rank
        # produced in an earlier pass (the queries settle in pass 1, the
        # candidates computed from them in pass 2); pass 3 is exact.
        def gather_for(rank, phase):
            calls = iter(range(1 << 20))

            def gather(x, dim):
                i = next(calls)
                seen[(rank, i)] = x.clone()
                if all((r, i) in store for r in range(world)):
                    return torch.cat([store[(r, i)] for r in range(world)], dim)
                return torch.cat([x] * world, dim)

            return gather

        for phase in (1, 2, 3):
            store.update(seen)
            for rank in range(world):
                rows[rank] = ops._sharded_index_topk(
                    q_full[:, rank * heads : (rank + 1) * heads],
                    w_full[:, rank * heads : (rank + 1) * heads],
                    keys,
                    table,
                    pos,
                    valid,
                    layout,
                    ratio,
                    columns,
                    top_k,
                    gather_for(rank, phase),
                    rank,
                    world,
                )
        got = rows[0]
        same = all(torch.equal(rows[r], got) for r in range(world))
        # Reference: every key of each request, all heads, the official rounding.
        worst, missing = 0.0, 0
        for t in range(tokens):
            r = t // 6
            n = (int(pos[t]) + 1) // ratio if bool(valid[t]) else 0
            if n == 0:
                ok = bool((got[t] == -1).all())
                missing += 0 if ok else top_k
                continue
            ids = torch.arange(n, device=DEV)
            page = table[r, ids // (block // ratio)].long()
            k = keys[page, ids % (block // ratio)]
            dots = (q_full[t].float() @ k.float().T).half().float().relu()
            score = (
                (dots.half().float() * w_full[t, :, None].float()).half().float().sum(0)
            )
            want = set(score.topk(min(top_k, n)).indices.tolist())
            have = set(int(x) for x in got[t] if x >= 0)
            missing += len(want - have)
            if want - have:
                cut = score.topk(min(top_k, n)).values[-1].item()
                worst = max(
                    worst, max(cut - score[sorted(have - want)].min().item(), 0)
                )
        check(
            f"index shard ratio={ratio} reqs={reqs}",
            same and missing <= 4 * tokens and worst <= 2e-2,
            f"ranks agree {same}, differing picks {missing}, worst score gap {worst:.4f}",
        )
        # Prefill: one request's 40-row chunk (groups of 6, row tiles).
        r0, n0 = 1, 40
        p0 = torch.arange(starts[1], starts[1] + n0, device=DEV)
        qp = (torch.randn(n0, heads * world, 128, device=DEV) * 0.5).half()
        wp = (torch.rand(n0, heads * world, device=DEV) * 0.1).half()
        vp = torch.ones(n0, dtype=torch.bool, device=DEV)
        out = {}
        store.clear()
        seen.clear()
        for phase in (1, 2, 3):
            store.update(seen)
            for rank in range(world):
                res = torch.full((n0, top_k), -1, dtype=torch.int32, device=DEV)
                ops._sharded_index_prefill(
                    res,
                    qp[:, rank * heads : (rank + 1) * heads],
                    wp[:, rank * heads : (rank + 1) * heads],
                    keys,
                    table[r0 : r0 + 1],
                    p0,
                    vp,
                    [0, n0],
                    [int(p0[-1])],
                    ratio,
                    top_k,
                    gather_for(rank, phase),
                    rank,
                    world,
                )
                out[rank] = res
        bad = 0
        for t in range(n0):
            n = (int(p0[t]) + 1) // ratio
            ids = torch.arange(n, device=DEV)
            page = table[r0, ids // (block // ratio)].long()
            k = keys[page, ids % (block // ratio)]
            dots = (qp[t].float() @ k.float().T).half().float().relu()
            score = (dots.half().float() * wp[t, :, None].float()).half().float().sum(0)
            want = set(score.topk(min(top_k, n)).indices.tolist())
            bad += len(want - set(int(x) for x in out[0][t] if x >= 0))
        check(
            f"index shard prefill ratio={ratio}",
            all(torch.equal(out[r], out[0]) for r in range(world)) and bad <= 2 * n0,
            f"differing picks {bad} over {n0} rows",
        )


def attention_fused_check(seed=7):
    """v41sk.attn_paged_dec (+ fused inverse RoPE, QDQ, window insert) against
    insert_latents_batch + attend_paged_latents_batch + apply_rope + QDQ."""
    from vllm.models.deepseek_v41.quant_rope import (
        apply_rope,
        attention_qdq,
        rope_table,
    )

    torch.manual_seed(seed)
    block, cblock, ratio = 64, 64, 4
    for reqs in (1, 4, 16):
        wcache, wtable = caches(reqs, 16, block, 512)
        ccache, ctable = caches(reqs, 16, cblock, 512, ratio=ratio)
        lengths = [6] * (reqs - 1) + [1]
        starts = [(300 + 97 * r) % 900 + (2 if r == 1 else 0) for r in range(reqs)]
        starts[0] = 2  # window before position 0 is masked
        pos = torch.cat(
            [torch.arange(a, a + n, device=DEV) for a, n in zip(starts, lengths)]
        )
        tokens = pos.numel()
        q = (torch.randn(tokens, 16, 512, device=DEV) * 0.3).half()
        values = (torch.randn(tokens, 512, device=DEV) * 0.5).half()
        valid = torch.ones(tokens, dtype=torch.bool, device=DEV)
        sink = torch.randn(16, device=DEV)
        limit = ((pos + 1) // ratio).clamp_min(1)
        topk = (torch.rand(tokens, 512, device=DEV) * limit[:, None]).long()
        topk[:, 256:] = -1
        topk = topk.to(torch.int32).contiguous()
        rope = rope_table(device=DEV)
        layout = layout_of(lengths)
        reference = wcache.clone()
        ops.insert_latents_batch(
            reference, values, pos, valid, wtable, layout, block_size=block
        )
        out = ops.attend_paged_latents_batch(
            q,
            pos,
            valid,
            reference,
            wtable,
            sink,
            layout,
            window_block=block,
            compressed_cache=ccache,
            compressed_table=ctable,
            compressed_block=cblock,
            ratio=ratio,
            topk=topk,
        )
        want = attention_qdq(
            apply_rope(out, pos, rope, inverse=True)
            .reshape(tokens, 2, 4096)
            .contiguous()
        ).view(tokens, 16, 512)
        got = torch.empty_like(q)
        run = lambda: torch.ops.v41sk.attn_paged_dec(  # noqa: E731
            got,
            q,
            values,
            wcache,
            wtable,
            ccache,
            ctable,
            layout.rows,
            layout.bases,
            pos,
            valid,
            topk,
            sink,
            512**-0.5,
            block,
            cblock,
            ratio,
            128,
            512,
            rope,
        )
        raw = torch.empty_like(q)  # attention alone (no post), cache already filled
        torch.ops.v41sk.attn_paged_dec(
            raw,
            q,
            values,
            reference,
            wtable,
            ccache,
            ctable,
            layout.rows,
            layout.bases,
            pos,
            valid,
            topk,
            sink,
            512**-0.5,
            block,
            cblock,
            ratio,
            128,
            512,
            None,
        )
        attn_diff = (raw.float() - out.float()).abs().max().item()
        # the QDQ of the fused output vs the reference QDQ of the same values
        requant = attention_qdq(
            apply_rope(raw, pos, rope, inverse=True)
            .reshape(tokens, 2, 4096)
            .contiguous()
        ).view(tokens, 16, 512)
        run()
        torch.accelerator.synchronize()
        inserted = torch.equal(wcache, reference)
        post_equal = (got != requant).float().mean().item()
        diff = (got.float() - want.float()).abs()
        close = (diff <= 0.02 * want.float().abs() + 2e-3).float().mean().item()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                run()
        graph.replay()
        torch.accelerator.synchronize()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        for _ in range(10):
            graph.replay()
        end.record()
        torch.accelerator.synchronize()
        us = begin.elapsed_time(end) / 200 * 1000
        check(
            f"attn_paged_dec reqs={reqs}",
            inserted and attn_diff < 2e-3 and post_equal < 1e-3,
            f"insert {'equal' if inserted else 'DIFFERS'} attention diff {attn_diff:.2e} "
            f"post mismatch {post_equal:.5f} close {close:.5f} "
            f"max diff {diff.max().item():.3e} {us:.1f} us",
        )


def markov_batch_check(seed=8):
    """MarkovShardGreedy.call_batch against per-request calls (tp=1)."""
    from vllm.models.deepseek_v41.markov_fused import MarkovGreedy
    from vllm.models.deepseek_v41.markov_shard import MarkovShardGreedy

    torch.manual_seed(seed)
    vocab, rank = 32768, 64
    weight = (torch.randn(vocab, rank, device=DEV) * 0.05).bfloat16()
    embed = (torch.randn(vocab, rank, device=DEV) * 0.5).half()
    shard = MarkovShardGreedy(MarkovGreedy(weight, embed), 0, 1, lambda x: x.clone())
    bad = 0
    for reqs in (2, 3, 8, 16):
        logits = torch.randn(5 * reqs, vocab, device=DEV).half()
        anchors = torch.randint(0, vocab, (reqs,), device=DEV)
        got = shard.call_batch(logits, anchors)
        want = torch.cat(
            [shard(logits[5 * r : 5 * r + 5], anchors[r : r + 1]) for r in range(reqs)]
        )
        bad += int((got != want).sum())
    check("markov drafts batched", bad == 0, f"mismatched tokens {bad}")


def route_wide_check(seed=9):
    """v41sk.route_plan_wide against Triton route_topk + _route_plan."""
    from vllm.models.deepseek_v41 import native
    from vllm.models.deepseek_v41.model import route_topk

    torch.manual_seed(seed)
    bias = (torch.randn(384, device=DEV) * 0.1).float()
    bad = 0
    for tokens in (11, 12, 24, 48, 96):
        logits = torch.randn(tokens, 384, device=DEV) * 2
        got = native.route_and_plan(logits, bias, 6, 2.5)
        probs, ids = route_topk(logits, bias, 6, 2.5)
        plan = native._route_plan(ids)
        bad += int(not torch.equal(got[0], probs)) + int(
            not torch.equal(got[1], ids.int())
        )
        bad += sum(int(not torch.equal(a, b)) for a, b in zip(got[2], plan))
    check("route plan wide", bad == 0, f"mismatched outputs {bad}")


def draft_check(reqs=3, seed=5):
    torch.manual_seed(seed)
    block, queries = 64, 5
    cache, table = caches(reqs, 8, block, 512)
    starts = [200, 90, 333][:reqs]
    pos = torch.cat([torch.arange(s, s + queries, device=DEV) for s in starts])
    n = pos.numel()
    q = (torch.randn(n, 16, 512, device=DEV) * 0.3).half()
    kv = torch.randn(n, 512, device=DEV).half()
    sink = torch.randn(16, device=DEV)
    valid = torch.ones(n, dtype=torch.bool, device=DEV)
    layout = layout_of([queries] * reqs)
    got = ops.attend_draft_latents_batch(
        q, kv, pos, valid, cache, table, sink, block, layout, queries
    )
    worst = 0.0
    for r in range(reqs):
        sl = slice(r * queries, (r + 1) * queries)
        want = ops.attend_draft_latents(
            q[sl].contiguous(),
            kv[sl].contiguous(),
            pos[sl],
            valid[sl],
            cache,
            table[r : r + 1],
            sink,
            block,
        )
        worst = max(worst, float((got[sl].float() - want.float()).abs().max()))
    check("draft attention", worst < 2e-3, f"max abs diff {worst:.2e}")


def plan_check(seed=6):
    from vllm.models.deepseek_v41 import native

    torch.manual_seed(seed)
    bad = 0
    for tokens in (11, 12, 24, 36, 48, 64, 85):
        for _ in range(4):
            ids = torch.stack(
                [torch.randperm(384, device=DEV)[:6] for _ in range(tokens)]
            )
            got = native._route_plan(ids)
            want = native._route_plan_torch(ids)
            bad += sum(not torch.equal(a, b) for a, b in zip(got, want))
    check("route plan (65..512 slots)", bad == 0, f"mismatched outputs {bad}")


if __name__ == "__main__":
    from vllm.models.deepseek_v41.native import load_v41sk

    load_v41sk()
    only = sys.argv[1:]
    for fn in (
        plan_check,
        engram_check,
        c2_check,
        insert_check,
        attention_check,
        attention_prefill_check,
        index_check,
        index_scores_check,
        index_shard_check,
        attention_fused_check,
        markov_batch_check,
        route_wide_check,
        draft_check,
    ):
        if only and fn.__name__ not in only:
            continue
        try:
            fn()
        except Exception as e:  # report every check
            check(fn.__name__, False, repr(e)[:300])
    print("FAILED:", FAILED if FAILED else "none")
    sys.exit(1 if FAILED else 0)
