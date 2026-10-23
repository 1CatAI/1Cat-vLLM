# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulate a per-layer GPU expert cache on recorded Flash-Next routing.

Input: the list saved by flashnext_expert_trace.py. Each verify round is
``--round-rows`` routed rows (5 for MTP4). For every cache fraction and policy
it reports the decode hit rate, unique missed experts per round and the
predicted extra time per round on each GPU's host link (the slowest rank
gates the round because experts are TP-sharded).

Policies (per layer, ``slots`` = fraction * experts):
  static   hot set ranked by routing frequency of the *other* prompts
  lru      least recently used
  adaptive static start; every ``--adapt-every`` rounds swap in up to
           ``--adapt-swaps`` experts by decayed heat (Strata-style); swaps
           land one round later
"""

import argparse
import collections
import json

import numpy as np
import torch


def decode_rounds(record, layer, rows_per_round):
    count = int(record["counter"][layer])
    ring = record["ring"][layer].numpy()
    rows = min(count, ring.shape[0])
    if count > ring.shape[0]:
        raise SystemExit(f"ring overflow in {record['label']} layer {layer}")
    ids = ring[:rows]
    usable = rows // rows_per_round * rows_per_round
    return ids[:usable].reshape(-1, rows_per_round * ids.shape[1])


def frequency(records, layer, experts, exclude):
    counts = np.zeros(experts, dtype=np.int64)
    for record in records:
        if record["label"] == exclude:
            continue
        counts += record["prefill"][layer].numpy()
        rounds = decode_rounds(record, layer, 5)
        np.add.at(counts, rounds[rounds >= 0].astype(np.int64), 1)
    return counts


def simulate(rounds, slots, policy, profile, adapt_every, adapt_swaps, decay):
    resident = set(np.argsort(-profile, kind="stable")[:slots].tolist())
    last_used = {}
    heat = collections.Counter()
    pending = []
    hits = misses = 0
    unique_misses = []
    for step, row in enumerate(rounds):
        if pending:
            for incoming, outgoing in pending:
                resident.discard(outgoing)
                resident.add(incoming)
            pending = []
        needed = [int(e) for e in row if e >= 0]
        missed = set()
        for expert in needed:
            if expert in resident:
                hits += 1
            else:
                misses += 1
                missed.add(expert)
        unique_misses.append(len(missed))
        for expert in set(needed):
            last_used[expert] = step
            heat[expert] += 1
        if policy == "lru":
            for expert in missed:
                if len(resident) >= slots:
                    victim = min(resident, key=lambda e: last_used.get(e, -1))
                    resident.discard(victim)
                resident.add(expert)
        elif policy == "adaptive" and (step + 1) % adapt_every == 0:
            candidates = [e for e, _ in heat.most_common() if e not in resident]
            victims = sorted(
                resident, key=lambda e: (heat.get(e, 0), last_used.get(e, -1))
            )
            swaps = []
            for incoming, outgoing in zip(candidates[:adapt_swaps], victims):
                if heat[incoming] > heat.get(outgoing, 0):
                    swaps.append((incoming, outgoing))
            pending = swaps
            for expert in list(heat):
                heat[expert] *= decay
    total = hits + misses
    return hits / total if total else 1.0, np.array(unique_misses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--round-rows", type=int, default=5)
    parser.add_argument("--fractions", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument(
        "--expert-mib",
        type=float,
        default=0.4675,
        help="per-rank bytes of one (layer, expert), MiB",
    )
    parser.add_argument(
        "--link-gbps", default="6,3,6,1.6", help="effective host->GPU GB/s per rank"
    )
    parser.add_argument("--adapt-every", type=int, default=4)
    parser.add_argument("--adapt-swaps", type=int, default=2)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--json")
    args = parser.parse_args()

    records = torch.load(args.trace)
    links = [float(x) for x in args.link_gbps.split(",")]
    results = []
    for fraction in (float(x) for x in args.fractions.split(",")):
        slots = int(round(fraction * args.experts))
        for policy in ("static", "lru", "adaptive"):
            hit_sum = hit_n = 0.0
            per_round = []
            for record in records:
                layer_misses = None
                for layer in range(args.layers):
                    rounds = decode_rounds(record, layer, args.round_rows)
                    if not len(rounds):
                        continue
                    profile = frequency(records, layer, args.experts, record["label"])
                    hit, unique = simulate(
                        rounds,
                        slots,
                        policy,
                        profile,
                        args.adapt_every,
                        args.adapt_swaps,
                        args.decay,
                    )
                    hit_sum += hit * len(rounds)
                    hit_n += len(rounds)
                    layer_misses = (
                        unique if layer_misses is None else layer_misses + unique
                    )
                if layer_misses is not None:
                    per_round.extend(layer_misses.tolist())
            per_round = np.array(per_round, dtype=np.float64)
            mib = per_round * args.expert_mib
            extra_ms = mib / 1024 / min(links) * 1000
            row = {
                "fraction": fraction,
                "policy": policy,
                "hit_rate": round(hit_sum / max(hit_n, 1), 4),
                "miss_experts_per_round_mean": round(float(per_round.mean()), 1),
                "miss_experts_per_round_p90": round(
                    float(np.percentile(per_round, 90)), 1
                ),
                "extra_ms_slowest_link_mean": round(float(extra_ms.mean()), 2),
                "extra_ms_slowest_link_p90": round(
                    float(np.percentile(extra_ms, 90)), 2
                ),
                "vram_freed_gib_per_rank": round(
                    (1 - fraction)
                    * args.layers
                    * args.experts
                    * args.expert_mib
                    / 1024,
                    2,
                ),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
