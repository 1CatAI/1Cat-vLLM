# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe requests to the native TP-group expert coordinator.

The control-page layout is exported by the native module and verified before
creating these views. Only the CPU owner publishes plans/commits. Each rank
publishes its own routed ids and completion. Failure traps the requesting CUDA
context, so divergent/expired plans cannot produce apparently valid logits.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _sys_load_acquire(pointer):
    # A result-unused atomic_add(..., 0, acquire) can disappear from the
    # generated PTX. Put acquire on the readiness load itself; its result
    # controls the polling loop and cannot be discarded as a no-op.
    return tl.inline_asm_elementwise(
        "ld.acquire.sys.global.u64 $0, [$1];",
        constraints="=l,l",
        args=[pointer],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _fail_if(failed):
    if failed != 0:
        tl.inline_asm_elementwise(
            "trap; mov.u32 $0, $1;",
            constraints="=r,r",
            args=[failed.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _device_time_ns():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def wait_shared_sequence(
    ready, failure, sequence, TIMEOUT_NS: tl.constexpr = 60_000_000_000
):
    # A dead owner cannot publish the CPU timeout flag. Bound device polling
    # independently so abrupt process exit cannot leave a permanent GPU spin.
    begin = _device_time_ns()
    tag = _sys_load_acquire(ready)
    failed = tl.load(failure, volatile=True)
    while (tag != sequence) & (failed == 0):
        _fail_if(_device_time_ns() - begin >= TIMEOUT_NS)
        tag = _sys_load_acquire(ready)
        failed = tl.load(failure, volatile=True)
    _fail_if(failed | (tag != sequence))


@triton.jit(do_not_specialize=["layer", "count"])
def publish_shared_routes(
    ids,
    counter,
    control,
    plan_slots,
    plan_loads,
    plan_saves,
    plan_count,
    fast,
    layer,
    count,
    RANK: tl.constexpr,
    RANK_START: tl.constexpr,
    RANK_WORDS: tl.constexpr,
    OP_START: tl.constexpr,
    OP_WORDS: tl.constexpr,
    MAX_OPS: tl.constexpr,
    BLOCK: tl.constexpr,
    plan_evicts=None,
    EVICT: tl.constexpr = False,
    skip_mask=None,
    SKIP: tl.constexpr = False,
    DEFER: tl.constexpr = False,
    plan_incoming=None,
    plan_victims=None,
    plan_trunc=None,
    STASH: tl.constexpr = False,
):
    # resolve_resident_routes already produced every slot on this rank (and,
    # identically, on every rank of the group): no CPU round trip.
    if tl.load(fast) != 0:
        return
    seq = tl.load(counter)
    request = control + RANK_START + RANK * RANK_WORDS
    lanes = tl.arange(0, BLOCK)
    expert = tl.load(ids + lanes, mask=lanes < count, other=0).to(tl.int64)
    if SKIP:
        # Dropped routes travel as the planner's skip sentinel (INT32_MAX).
        dropped = tl.load(skip_mask + lanes, mask=lanes < count, other=0)
        if DEFER:
            # bit 30: a drop candidate the coordinator may adopt or queue
            expert = tl.where(dropped != 0, expert | (1 << 30), expert)
        else:
            expert = tl.where(dropped != 0, 2147483647, expert)
    tl.store(request + 4 + lanes, expert, mask=lanes < count)
    tl.store(request + 2, layer.to(tl.int64))
    tl.store(request + 3, count.to(tl.int64))
    tl.debug_barrier()
    tl.atomic_xchg(request, seq, sem="release", scope="sys")
    wait_shared_sequence(control, control + 2, seq)
    packed = tl.load(control + 3, volatile=True)
    # Low 16 bits: swap copies; high bits: background evictions after them.
    copies = packed & 0xFFFF
    evicts = (packed >> 16) & 0xFFFF
    _fail_if((packed < 0) | (copies + evicts > MAX_OPS))
    valid = lanes < copies + evicts
    if EVICT:
        tl.store(plan_evicts, evicts.to(tl.int32))
    else:
        _fail_if(evicts != 0)
    op = control + OP_START + lanes * OP_WORDS
    tl.store(
        plan_slots + lanes,
        tl.load(op, mask=valid, other=0, volatile=True).to(tl.int32),
        mask=lanes < MAX_OPS,
    )
    tl.store(
        plan_loads + lanes,
        tl.load(op + 3, mask=valid, other=-1, volatile=True),
        mask=lanes < MAX_OPS,
    )
    tl.store(
        plan_saves + lanes,
        tl.load(op + 4, mask=valid, other=-1, volatile=True),
        mask=lanes < MAX_OPS,
    )
    tl.store(plan_count, copies.to(tl.int32))
    if STASH:
        # stash_plan folded in: the plan words are read in this same pass.
        tl.store(
            plan_incoming + lanes,
            tl.load(op + 1, mask=valid, other=0, volatile=True),
            mask=lanes < MAX_OPS,
        )
        tl.store(
            plan_victims + lanes,
            tl.load(op + 2, mask=valid, other=0, volatile=True),
            mask=lanes < MAX_OPS,
        )
        tl.store(plan_trunc, tl.load(control + 6, volatile=True).to(tl.int32))


@triton.jit(do_not_specialize=["layer", "count"])
def commit_shared_routes(
    ids,
    table,
    cold_offsets,
    counter,
    control,
    out,
    statistics,
    fast,
    layer,
    count,
    RANK: tl.constexpr,
    RANK_START: tl.constexpr,
    RANK_WORDS: tl.constexpr,
    OP_START: tl.constexpr,
    OP_WORDS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(fast) != 0:
        return
    seq = tl.load(counter)
    request = control + RANK_START + RANK * RANK_WORDS
    tl.atomic_xchg(request + 1, seq, sem="release", scope="sys")
    wait_shared_sequence(control + 1, control + 2, seq)
    lanes = tl.arange(0, BLOCK)
    copies = tl.load(control + 3, volatile=True)
    at = statistics + layer.to(tl.int64) * 2
    tl.store(at, tl.load(at) + 1)
    tl.store(at + 1, tl.load(at + 1) + copies)
    op = control + OP_START + lanes * OP_WORDS
    valid = lanes < copies
    slot = tl.load(op, mask=valid, other=0, volatile=True).to(tl.int32)
    incoming = tl.load(op + 1, mask=valid, other=0, volatile=True)
    victim = tl.load(op + 2, mask=valid, other=0, volatile=True)
    recycled = tl.load(op + 3, mask=valid, other=-1, volatile=True)
    tl.store(table + layer.to(tl.int64) * EXPERTS + victim, -1, mask=valid)
    tl.store(table + layer.to(tl.int64) * EXPERTS + incoming, slot, mask=valid)
    # The coordinator compacts the evicted original into the incoming
    # expert's former cold allocation before publishing the commit sequence.
    tl.store(cold_offsets + layer.to(tl.int64) * EXPERTS + victim, recycled, mask=valid)
    tl.store(cold_offsets + layer.to(tl.int64) * EXPERTS + incoming, -1, mask=valid)
    tl.debug_barrier()
    expert = tl.load(ids + lanes, mask=lanes < count, other=0).to(tl.int64)
    resolved = tl.load(
        table + layer.to(tl.int64) * EXPERTS + expert, mask=lanes < count, other=0
    )
    _fail_if(tl.sum(((lanes < count) & (resolved < 0)).to(tl.int32), 0))
    tl.store(out + lanes, resolved, mask=lanes < count)
    tl.store(counter, seq + 1)


@triton.jit
def stash_plan(
    control,
    fast,
    plan_incoming,
    plan_victims,
    plan_trunc,
    OP_START: tl.constexpr,
    OP_WORDS: tl.constexpr,
    MAX_OPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy the published plan's host words to device right after publish.

    Later reads of the mapped control page would queue behind this rank's
    posted PCIe writebacks (reads may not pass posted writes), so the local
    commit must use device copies only."""
    if tl.load(fast) != 0:
        return
    lanes = tl.arange(0, BLOCK)
    packed = tl.load(control + 3, volatile=True)
    valid = lanes < (packed & 0xFFFF) + ((packed >> 16) & 0xFFFF)
    op = control + OP_START + lanes * OP_WORDS
    tl.store(
        plan_incoming + lanes,
        tl.load(op + 1, mask=valid, other=0, volatile=True),
        mask=lanes < MAX_OPS,
    )
    tl.store(
        plan_victims + lanes,
        tl.load(op + 2, mask=valid, other=0, volatile=True),
        mask=lanes < MAX_OPS,
    )
    tl.store(plan_trunc, tl.load(control + 6, volatile=True).to(tl.int32))


@triton.jit(do_not_specialize=["layer", "count"])
def commit_local_routes(
    ids,
    table,
    cold_offsets,
    counter,
    plan_slots,
    plan_loads,
    plan_count,
    plan_incoming,
    plan_victims,
    plan_trunc,
    out,
    statistics,
    fast,
    layer,
    count,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    trunc=None,
    TRUNC: tl.constexpr = False,
    plan_saves=None,
    control=None,
    ack_word=0,
    FREE: tl.constexpr = False,
    plan_evicts=None,
    EVICT: tl.constexpr = False,
    skip_mask=None,
    SKIP: tl.constexpr = False,
):
    """Local commit (device memory reads only). In-place mode: the side-stream
    writeback publishes completion. Free mode: the victim lives at its save
    allocation and this kernel publishes completion after the swap kernel."""
    if tl.load(fast) != 0:
        return
    seq = tl.load(counter)
    lanes = tl.arange(0, BLOCK)
    copies = tl.load(plan_count)
    evicts = 0
    if EVICT:
        evicts = tl.load(plan_evicts)
    at = statistics + layer.to(tl.int64) * 2
    tl.store(at, tl.load(at) + 1)
    # Misses = real transfers (rescued experts never left their slots).
    moved = tl.load(plan_loads + lanes, mask=lanes < copies, other=-1)
    # -2: loaded by the copy-engine lanes before the plan was published
    tl.store(
        at + 1, tl.load(at + 1) + tl.sum(((moved >= 0) | (moved == -2)).to(tl.int64), 0)
    )
    valid = lanes < copies + evicts
    slot = tl.load(plan_slots + lanes, mask=valid, other=0)
    incoming = tl.load(plan_incoming + lanes, mask=valid, other=-1)
    victim = tl.load(plan_victims + lanes, mask=valid, other=-1)
    if FREE:
        recycled = tl.load(plan_saves + lanes, mask=valid, other=-1)
    else:
        recycled = tl.load(plan_loads + lanes, mask=valid, other=-1)
    base = layer.to(tl.int64) * EXPERTS
    # Free-slot loads and rescues have no victim; evictions (after the copies)
    # have no incoming. An evicted expert's save lands asynchronously.
    vmask = valid & (victim >= 0)
    imask = valid & (lanes < copies) & (incoming >= 0)
    tl.store(table + base + victim, -1, mask=vmask)
    tl.store(table + base + incoming, slot, mask=imask)
    # In place: the victim now lives in the incoming expert's former allocation.
    tl.store(cold_offsets + base + victim, recycled, mask=vmask)
    tl.store(cold_offsets + base + incoming, -1, mask=imask)
    tl.debug_barrier()
    expert = tl.load(ids + lanes, mask=lanes < count, other=0).to(tl.int64)
    resolved = tl.load(table + base + expert, mask=lanes < count, other=0)
    limit = tl.full((), 1 << 20, tl.int32)
    if TRUNC:
        limit = tl.minimum(tl.load(trunc), tl.load(plan_trunc))
        tl.store(trunc, limit)
    required = (lanes < count) & (lanes // 6 < limit)
    if SKIP:
        required = required & (
            tl.load(skip_mask + lanes, mask=lanes < count, other=0) == 0
        )
    _fail_if(tl.sum((required & (resolved < 0)).to(tl.int32), 0))
    tl.store(out + lanes, resolved, mask=lanes < count)
    if FREE:  # noqa: SIM102
        if copies + evicts > 0:
            tl.atomic_xchg(control + ack_word, seq, sem="release", scope="sys")
    tl.store(counter, seq + 1)


@triton.jit(do_not_specialize=["layer", "count"])
def resolve_resident_routes(
    ids,
    table,
    out,
    fast,
    plan_count,
    statistics,
    hit_counter,
    control,
    layer,
    count,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG: tl.constexpr,
    HIT_START: tl.constexpr,
    HIT_HEADER: tl.constexpr,
    HIT_ENTRIES: tl.constexpr,
    HIT_WORDS: tl.constexpr,
    RELAXED: tl.constexpr = True,
    trunc=None,
    TRUNC: tl.constexpr = False,
    weights=None,
    hist=None,
    HIST: tl.constexpr = False,
    skip_mask=None,
    skip_below=0.0,
    skip_late=0.0,
    late_from=2,
    SKIP: tl.constexpr = False,
    min_rank=0,
    DEFER: tl.constexpr = False,
):
    """All-resident fast path. Every rank holds the same table (commits apply
    identical plans) and the coordinator verifies identical routes, so all
    ranks take the same branch. Rank 0 logs the routes for the LRU clock."""
    lanes = tl.arange(0, BLOCK)
    valid = lanes < count
    expert = tl.load(ids + lanes, mask=valid, other=0).to(tl.int64)
    slot = tl.load(table + layer.to(tl.int64) * EXPERTS + expert, mask=valid, other=0)
    needed = valid
    if TRUNC:
        needed = valid & (lanes // 6 < tl.load(trunc))
    if SKIP:
        # Lossy (VLLM_V41_SKIP_BELOW): a cold route whose gate weight is below
        # the threshold is dropped (contributes zero) instead of fetched.
        gate = tl.load(weights + lanes, mask=valid, other=1.0)
        # Rank of the route within its token by gate weight (0 = top-1); a
        # token's top min_rank routes are always fetched (VLLM_V41_SKIP_MIN_RANK).
        base = lanes - lanes % 6
        rank = tl.zeros((BLOCK,), tl.int32)
        for k in tl.static_range(6):
            other = tl.load(
                weights + base + k, mask=valid & (base + k < count), other=-1.0
            )
            rank += ((other > gate) | ((other == gate) & (k < lanes % 6))).to(tl.int32)
        limit = tl.where(lanes // 6 >= late_from, skip_late, skip_below)
        dropped = valid & (slot < 0) & (gate < limit) & (rank >= min_rank)
        if TRUNC:
            dropped = dropped & (lanes // 6 < tl.load(trunc))
        tl.store(skip_mask + lanes, dropped.to(tl.int32), mask=lanes < BLOCK)
        needed = needed & (dropped == 0)
    misses = tl.sum((needed & (slot < 0)).to(tl.int32), 0)
    hit = misses == 0
    if SKIP:  # noqa: SIM102
        if DEFER:
            # Drops still visit the coordinator (no transfer): it adopts an
            # expert a background load already placed, and queues the rest.
            hit = hit & (tl.sum(dropped.to(tl.int32), 0) == 0)
    if HIST:
        # Debug (VLLM_V41_MISS_HIST): gate-weight buckets of needed routes
        # (row 1) and of those not resident (row 0).
        w = tl.load(weights + lanes, mask=valid, other=0.0)
        b = (
            (w >= 0.02).to(tl.int32)
            + (w >= 0.05).to(tl.int32)
            + (w >= 0.1).to(tl.int32)
            + (w >= 0.15).to(tl.int32)
            + (w >= 0.2).to(tl.int32)
            + (w >= 0.3).to(tl.int32)
            + (w >= 0.45).to(tl.int32)
            + (w >= 0.7).to(tl.int32)
        )
        tl.atomic_add(hist + b, 1, mask=needed & (slot < 0))
        tl.atomic_add(hist + 16 + b, 1, mask=needed)
    tl.store(out + lanes, slot.to(out.dtype.element_ty), mask=valid & hit)
    tl.store(fast, hit.to(tl.int32))
    if hit:
        tl.store(plan_count, 0)
        at = statistics + layer.to(tl.int64) * 2
        tl.store(at, tl.load(at) + 1)
        if LOG:
            n = tl.load(hit_counter)
            entry = control + HIT_START + HIT_HEADER + (n % HIT_ENTRIES) * HIT_WORDS
            tl.store(entry + 1, layer.to(tl.int64))
            tl.store(entry + 2, count.to(tl.int64))
            tl.store(entry + 3 + lanes, expert, mask=valid)
            tl.debug_barrier()
            # Relaxed: drain_hits() re-checks each entry tag and skips a torn
            # entry (costing one LRU touch), so no PCIe system fence here.
            if RELAXED:
                tl.store(entry, n + 1)
                tl.store(control + HIT_START, n + 1)
            else:
                tl.atomic_xchg(entry, n + 1, sem="release", scope="sys")
                tl.atomic_xchg(control + HIT_START, n + 1, sem="release", scope="sys")
            tl.store(hit_counter, n + 1)


@triton.jit
def publish_round(control, round_counter, word):
    """Write this forward's number into the rank request (planner truncation)."""
    tl.store(control + word, tl.load(round_counter))


@triton.jit(do_not_specialize=["layer", "tokens"])
def publish_hints(
    logits,
    bias,
    control,
    hint_counter,
    trunc,
    layer,
    tokens,
    BASE: tl.constexpr,
    ENTRIES: tl.constexpr,
    WORDS: tl.constexpr,
    EQ: tl.constexpr,
    FIRST: tl.constexpr,
    TOK: tl.constexpr,
    CAND: tl.constexpr,
    BLOCK: tl.constexpr,
    TRUNC: tl.constexpr,
):
    """Prefetch hints: this rank's expert quarter [FIRST, FIRST + EQ) of the
    next two layers' routers applied to this layer's MoE input. Per lookahead
    and verify token, the top CAND biased selection scores are posted (relaxed;
    the coordinator checks every rank's entry tag) as expert << 32 | bits."""
    n = tl.load(hint_counter)
    entry = control + BASE + 8 + (n % ENTRIES) * WORDS
    e = tl.arange(0, BLOCK)
    valid = e < EQ
    c = tl.arange(0, CAND)
    count = tl.minimum(tokens, TOK)
    for d in tl.static_range(2):
        b = tl.load(bias + d * EQ + e, mask=valid, other=0.0)
        for t in range(0, count):
            g = tl.load(logits + t * (2 * EQ) + d * EQ + e, mask=valid, other=0.0).to(
                tl.float32
            )
            soft = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(tl.minimum(g, 20.0))))
            key = tl.where(valid, tl.sqrt(soft) + b, float("-inf"))
            words = tl.zeros((CAND,), tl.int64)
            for j in tl.static_range(CAND):
                best = tl.max(key, 0)
                index = tl.min(tl.where(key == best, e, BLOCK), 0)
                bits = best.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
                words = tl.where(
                    c == j, ((index + FIRST).to(tl.int64) << 32) | bits, words
                )
                key = tl.where(e == index, float("-inf"), key)
            tl.store(entry + 4 + (d * TOK + t) * CAND + c, words)
    limit = tl.full((), 1 << 20, tl.int64)
    if TRUNC:
        limit = tl.load(trunc).to(tl.int64)
    tl.store(entry + 1, layer.to(tl.int64))
    tl.store(entry + 2, count.to(tl.int64))
    tl.store(entry + 3, limit)
    tl.debug_barrier()
    tl.store(entry, n + 1)
    tl.store(control + BASE, n + 1)
    tl.store(hint_counter, n + 1)


@triton.jit(do_not_specialize=["count"])
def substitute_dropped(
    out,
    ids,
    weights,
    skip_mask,
    fast,
    table_row,
    logits,
    bias,
    trunc,
    count,
    EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SCALE: tl.constexpr,
    SUBST: tl.constexpr,
    TRUNC: tl.constexpr,
):
    """After a deferred plan: routes still dropped are replaced by the best
    resident expert the router did not pick (SUBST, cache-aware routing), and
    every touched token's probabilities are recomputed with the router's own
    normalization over its final set (sqrt-softplus scores / sum * SCALE)."""
    if tl.load(fast) != 0:
        return
    e = tl.arange(0, BLOCK_E)
    ev = e < EXPERTS
    resident = tl.load(table_row + e, mask=ev, other=-1) >= 0
    b = tl.load(bias + e, mask=ev, other=0.0)
    j = tl.arange(0, 8)
    jv = j < 6
    limit = 1 << 20
    if TRUNC:
        limit = tl.load(trunc)
    for t in range(0, count // 6):
        lane = t * 6 + j
        dropped = (
            jv
            & (tl.load(skip_mask + lane, mask=jv, other=0) != 0)
            & (tl.load(out + lane, mask=jv, other=0) < 0)
        )
        if (t < limit) & (tl.sum(dropped.to(tl.int32), 0) > 0):
            g = tl.load(logits + t * EXPERTS + e, mask=ev, other=0.0)
            soft = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(tl.minimum(g, 20.0))))
            score = tl.sqrt(soft)
            key = tl.where(ev & resident, score + b, float("-inf"))
            cur = tl.load(ids + lane, mask=jv, other=-1).to(tl.int32)
            for k in tl.static_range(6):
                ck = tl.sum(tl.where(j == k, cur, 0), 0)
                key = tl.where(e == ck, float("-inf"), key)
            scores = tl.zeros((8,), tl.float32)
            for k in tl.static_range(6):
                ck = tl.sum(tl.where(j == k, cur, 0), 0)
                dk = tl.sum(tl.where(j == k, dropped.to(tl.int32), 0), 0)
                sk = tl.sum(tl.where(e == ck, score, 0.0), 0)
                if dk != 0:
                    sk = sk * 0.0
                    if SUBST:
                        best = tl.argmax(key, 0)
                        if tl.max(key, 0) > float("-inf"):
                            key = tl.where(e == best, float("-inf"), key)
                            tl.store(
                                out + t * 6 + k,
                                tl.load(table_row + best).to(out.dtype.element_ty),
                            )
                            tl.store(ids + t * 6 + k, best.to(ids.dtype.element_ty))
                            sk = tl.sum(tl.where(e == best, score, 0.0), 0)
                scores = tl.where(j == k, sk, scores)
            total = tl.sum(scores, 0)
            tl.store(weights + lane, scores / (total + 1e-20) * SCALE, mask=jv)
