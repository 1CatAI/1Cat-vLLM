# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded topology-independent collective planning, costing and measured selection.

CPU only. Missing peer edges are never invented. A model shortlist is distinct
from a measured winner; disconnected peer graphs explicitly require a fallback.
"""

import hashlib
import heapq
import itertools
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .plan import Plan

SCHEMA = 1


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass
class Topology:
    nodes: list
    edges: list
    resources: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Distinct nodes required")
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.links: dict[tuple[Any, Any], dict[str, Any]] = {}
        for edge in self.edges:
            a, b = edge["source"], edge["dest"]
            if (
                a not in self.index
                or b not in self.index
                or a == b
                or (a, b) in self.links
            ):
                raise ValueError("Invalid/duplicate edge")
            if (
                not math.isfinite(edge["latency_us"])
                or edge["latency_us"] < 0
                or not math.isfinite(edge["bytes_per_us"])
                or edge["bytes_per_us"] <= 0
            ):
                raise ValueError("Invalid edge cost")
            if any(r not in self.resources for r in edge.get("resources", [])):
                raise ValueError("Unknown shared resource")
            self.links[a, b] = edge
        for resource in self.resources.values():
            if (
                not math.isfinite(resource["bytes_per_us"])
                or resource["bytes_per_us"] <= 0
                or not isinstance(resource.get("lanes", 1), int)
                or resource.get("lanes", 1) < 1
            ):
                raise ValueError("Invalid resource")
        # CUDA IPC mappings/signalling require supported edges both ways.
        self.labels = {
            (a, b): e.get("transport", "peer")
            for (a, b), e in self.links.items()
            if (b, a) in self.links
        }

    def export(self):
        return dict(
            nodes=self.nodes,
            edges=self.edges,
            resources=self.resources,
            identity=self.identity,
        )

    @property
    def key(self):
        return fingerprint(self.export())


def from_probe(record, profiles=None, identity=None):
    """Convert driver-supported links; supplied measured profiles override estimates.

    profiles is a list of directed edges with latency_us/bytes_per_us and optional
    resource identifiers. Explicit resources can describe upstream bottlenecks.
    Without profiles all costs are labelled estimates, never measurements.
    """
    from .planner_probe import matrix

    nodes, labels = matrix(record["topology"])
    positions = {node: i for i, node in enumerate(nodes)}
    edges = []
    resources: dict[str, dict[str, Any]] = {}
    overrides = {(e["source"], e["dest"]): e for e in (profiles or {}).get("edges", [])}
    resources.update((profiles or {}).get("resources", {}))
    for (a, b), label in sorted(labels.items()):
        if a == b or any(
            record[k][positions[x]][positions[y]] != "OK"
            for k in ("peer_read", "peer_write")
            for x, y in ((a, b), (b, a))
        ):
            continue
        if label.startswith("NV"):
            bw = 11294.1176470588 * int(label[2:]) / 2
            alpha = max(0.0, 2.050 - 5120 / bw) / 2
            shared = []
        else:
            bw = 5783.13253012048
            alpha = (3.392 - 5120 / bw) / 2
            shared = [f"pcie-tx-{a}", f"pcie-rx-{b}"]
            for name in shared:
                resources.setdefault(name, dict(bytes_per_us=bw, lanes=1))
        edge = dict(
            source=a,
            dest=b,
            transport=label,
            latency_us=alpha,
            bytes_per_us=bw,
            resources=shared,
            measured=False,
        )
        edge.update(overrides.get((a, b), {}))
        edges.append(edge)
    return Topology(
        nodes,
        edges,
        resources,
        dict(
            identity or {},
            cost_origin="profiles" if profiles else "explicit-initial-estimates",
        ),
    )


class DAG(Plan):
    def __init__(self, topology, nodes, members, payload, precision, name, family):
        super().__init__(nodes, topology.labels, name, family, payload)
        self.members = list(members)
        self.payload = payload
        self.precision = precision
        for rank, node in enumerate(nodes):
            if node not in members:
                self.values[rank]["mask"] = 0

    def add(self, rank, left, right, dtype=None):
        return super().add(rank, left, right, dtype or self.precision)

    def sum(self, rank, values, dtype=None):
        return super().sum(rank, values, dtype or self.precision)

    def reduce_groups(self, groups, dtype=None):
        return super().reduce_groups(groups, dtype or self.precision)

    def wave(self, transfers):
        before = len(self.messages)
        super().wave(transfers)
        for m in self.messages[before:]:
            m["bytes"] = self.payload * (
                2 if self.values[m["value"]]["dtype"] == "fp32" else 1
            )

    def finish(self):
        goal = sum(
            1 << r for r, node in enumerate(self.devices) if node in self.members
        )
        if len(set(self.current)) != 1 or any(
            self.values[v]["mask"] != goal for v in self.current
        ):
            raise ValueError("Incomplete common expression")
        self.outputs = self.current[:]
        return self

    def export(self):
        return dict(
            super().export(),
            members=self.members,
            payload_bytes=self.payload,
            precision=self.precision,
        )


def validate(data, topology):
    nodes = data["devices"]
    members = data["members"]
    n = len(nodes)
    values = data["values"]
    have = [{r} for r in range(n)]
    if (
        len(set(nodes)) != n
        or not set(nodes) <= set(topology.nodes)
        or not members
        or len(set(members)) != len(members)
        or not set(members) <= set(nodes)
    ):
        raise ValueError("Invalid membership")
    if len(data["programs"]) != n or len(data["outputs"]) != n or len(values) < n:
        raise ValueError("Rank count mismatch")
    if (
        data["precision"] not in ("fp16", "fp32")
        or data["payload_bytes"] <= 0
        or data["payload_bytes"] % 16
    ):
        raise ValueError("Invalid arithmetic/payload")
    for i, v in enumerate(values):
        if i < n:
            if v != dict(
                mask=1 << i if nodes[i] in members else 0, children=None, dtype="fp16"
            ):
                raise ValueError("Invalid input/helper")
        else:
            left, right = v["children"]
            if (
                not 0 <= left < i
                or not 0 <= right < i
                or not values[left]["mask"]
                or not values[right]["mask"]
                or values[left]["mask"] & values[right]["mask"]
            ):
                raise ValueError("Duplicate/invalid contribution")
            if (
                v["mask"] != values[left]["mask"] | values[right]["mask"]
                or v["dtype"] != data["precision"]
            ):
                raise ValueError("Invalid subtotal")
    sends = set()
    recvs = set()
    for rank, program in enumerate(data["programs"]):
        for op in program:
            kind = op["op"]
            v = op["value"]
            if not 0 <= v < len(values):
                raise ValueError("Invalid value")
            if kind == "add":
                if (
                    op["left"] not in have[rank]
                    or op["right"] not in have[rank]
                    or values[v]["children"] != [op["left"], op["right"]]
                ):
                    raise ValueError("Missing operands")
                have[rank].add(v)
            elif kind in ("send", "recv"):
                mid = op["message"]
                if not 0 <= mid < len(data["messages"]):
                    raise ValueError("Invalid message")
                m = data["messages"][mid]
                s, d = m["source"], m["dest"]
                if (
                    not 0 <= s < n
                    or not 0 <= d < n
                    or (nodes[s], nodes[d]) not in topology.labels
                ):
                    raise ValueError("Unsupported peer mapping")
                if m["value"] != v or m["bytes"] != data["payload_bytes"] * (
                    2 if values[v]["dtype"] == "fp32" else 1
                ):
                    raise ValueError("Invalid message width")
                if kind == "send":
                    if (
                        s != rank
                        or d != op["dest"]
                        or v not in have[rank]
                        or mid in sends
                    ):
                        raise ValueError("Invalid send")
                    sends.add(mid)
                else:
                    if d != rank or s != op["source"] or mid in recvs:
                        raise ValueError("Invalid receive")
                    recvs.add(mid)
                    have[rank].add(v)
            else:
                raise ValueError("Unknown operation")
        if data["outputs"][rank] not in have[rank]:
            raise ValueError("Missing output")
    if sends != recvs or len(sends) != len(data["messages"]):
        raise ValueError("Unpaired message")
    goal = sum(1 << r for r, node in enumerate(nodes) if node in members)
    if len(set(data["outputs"])) != 1 or any(
        values[v]["mask"] != goal for v in data["outputs"]
    ):
        raise ValueError("Inconsistent/incomplete output")
    return True


def simulate(data, topology, issue=0.02, receive=0.2, add=0.02, trace=False):
    n = len(data["devices"])
    positions = [0] * n
    clocks = [0.0] * n
    arrivals: dict[tuple[int, int], float] = {}
    resources: dict[str, list[float]] = {}
    events = []
    while any(positions[r] < len(data["programs"][r]) for r in range(n)):
        ready = []
        for r in range(n):
            if positions[r] == len(data["programs"][r]):
                continue
            op = data["programs"][r][positions[r]]
            t = clocks[r]
            if op["op"] == "recv":
                if op["message"] not in arrivals:
                    continue
                t = max(t, arrivals[op["message"]])
            ready.append((t, r))
        if not ready:
            raise ValueError("Deadlocked schedule")
        t, r = min(ready)
        op = data["programs"][r][positions[r]]
        start = t
        if op["op"] == "send":
            m = data["messages"][op["message"]]
            a, b = data["devices"][r], data["devices"][m["dest"]]
            edge = topology.links[a, b]
            needed = [
                (
                    f"link-{topology.index[a]}-{topology.index[b]}",
                    dict(bytes_per_us=edge["bytes_per_us"], lanes=1),
                )
            ]
            needed += [
                (key, topology.resources[key]) for key in edge.get("resources", [])
            ]
            selected = []
            for key, spec in needed:
                lanes = resources.setdefault(key, [0.0] * spec.get("lanes", 1))
                lane = min(range(len(lanes)), key=lanes.__getitem__)
                start = max(start, lanes[lane])
                selected.append((lanes, lane, m["bytes"] / spec["bytes_per_us"]))
            service = max(v[2] for v in selected)
            for lanes, lane, duration in selected:
                lanes[lane] = start + duration
            end = start + service + edge["latency_us"]
            arrivals[op["message"]] = end
            clocks[r] = t + issue
        else:
            end = t + (receive if op["op"] == "recv" else add)
            clocks[r] = end
        if trace:
            events.append(
                dict(
                    rank=r,
                    op=op["op"],
                    message=op.get("message"),
                    value=op["value"],
                    start_us=start,
                    end_us=end,
                )
            )
        positions[r] += 1
    return dict(us=max(clocks), rank_us=clocks, events=events if trace else None)


def shortest_tree(topology, nodes, root, payload, jitter=None):
    distance = [math.inf] * len(nodes)
    distance[root] = 0
    parents = [-1] * len(nodes)
    queue = [(0.0, root)]
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != distance[a]:
            continue
        for b in range(len(nodes)):
            if (nodes[a], nodes[b]) not in topology.labels:
                continue
            e = topology.links[nodes[a], nodes[b]]
            reverse = topology.links[nodes[b], nodes[a]]
            weight = (
                e["latency_us"]
                + reverse["latency_us"]
                + payload / e["bytes_per_us"]
                + payload / reverse["bytes_per_us"]
            )
            candidate = cost + weight * (
                jitter.get((min(a, b), max(a, b)), 1.0) if jitter else 1.0
            )
            if candidate < distance[b]:
                distance[b] = candidate
                parents[b] = a
                heapq.heappush(queue, (candidate, b))
    if any(not math.isfinite(v) for v in distance):
        raise ValueError("Disconnected peer graph")
    return parents


def tree(topology, nodes, members, payload, precision, root, parents):
    p = DAG(topology, nodes, members, payload, precision, "routed-tree", "tree")
    children: list[list[int]] = [[] for _ in nodes]
    for r, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(r)
    # Prune helper leaves, but retain every relay on a member-to-root path.
    keep = set(nodes.index(v) for v in members)
    for r in list(keep):
        while parents[r] >= 0:
            r = parents[r]
            keep.add(r)
    levels = [[root]]
    while levels[-1]:
        levels.append([c for r in levels[-1] for c in children[r] if c in keep])
    current = [r if node in members else None for r, node in enumerate(nodes)]
    for level in reversed(levels[1:-1]):
        p.wave([(r, parents[r], current[r]) for r in level])
        for parent in sorted({parents[r] for r in level}):
            operands = ([current[parent]] if current[parent] is not None else []) + [
                current[c] for c in children[parent] if c in keep
            ]
            current[parent] = p.sum(parent, operands)
    full = current[root]
    for level in levels[1:-1]:
        p.wave([(parents[r], r, full) for r in level])
    # Excluded helpers are omitted entirely from the resulting executable plan.
    active = sorted(keep)
    p.current = [full] * len(nodes)
    p.outputs = p.current[:]
    data = p.export()
    if len(active) != len(nodes):
        return None  # caller retries with the pruned node set
    data["root"] = root
    data["parents"] = parents
    return data


def fast_groups(topology, nodes):
    """Greedy complete fast-link islands; chains never become fake cliques."""
    remaining = set(range(len(nodes)))
    groups = []
    while remaining:
        group = [min(remaining)]
        remaining.remove(group[0])
        for r in sorted(remaining):
            if all(
                topology.labels.get((nodes[r], nodes[s]), "").startswith("NV")
                and topology.labels.get((nodes[s], nodes[r]), "").startswith("NV")
                for s in group
            ):
                group.append(r)
        remaining -= set(group)
        groups.append(tuple(group))
    return groups


def island_plan(topology, nodes, members, payload, precision, groups, assignment):
    p = DAG(
        topology,
        nodes,
        members,
        payload,
        precision,
        "balanced-islands",
        "island-delivery",
    )
    p.reduce_groups(groups)
    values = [p.current[g[0]] for g in groups]
    p.wave([(s, d, values[g]) for g, s, d in assignment])
    transfers = []
    for group in groups:
        for r in group:
            for value in values:
                if value not in p.have[r]:
                    source = next(s for s in group if value in p.have[s])
                    transfers.append((source, r, value))
    p.wave(transfers)
    for r in range(len(nodes)):
        p.current[r] = p.sum(r, values)
    return p.finish().export()


def solve(
    topology,
    members,
    *,
    helpers=(),
    payload_bytes=5120,
    precision="fp32",
    budget=4096,
    shortlist=8,
    seed=271828,
):
    """Deterministic bounded search, polynomial seeds + budgeted assignments.

    Returns a ranked model shortlist, not an unmeasured optimality claim.
    Native execution may impose a smaller hardware limit than this CPU solver.
    """
    start = time.monotonic()
    members = list(members)
    helpers = list(helpers)
    if (
        not members
        or len(set(members + helpers)) != len(members) + len(helpers)
        or not set(members + helpers) <= set(topology.nodes)
    ):
        raise ValueError("Invalid members/helpers")
    if (
        precision not in ("fp16", "fp32")
        or payload_bytes <= 0
        or payload_bytes % 16
        or not 16 <= budget <= 100000
        or not 1 <= shortlist <= 32
    ):
        raise ValueError("Invalid bounded planning request")
    key = fingerprint(
        dict(
            schema=SCHEMA,
            solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            topology=topology.export(),
            members=members,
            helpers=helpers,
            payload_bytes=payload_bytes,
            precision=precision,
            budget=budget,
            shortlist=shortlist,
            seed=seed,
        )
    )
    counts: dict[str, int] = {}
    best: dict[str, list[dict[str, Any]]] = {}
    seen = set()
    attempts = 0
    rng = random.Random(seed)

    def consider(data):
        nonlocal attempts
        if data is None or attempts >= budget:
            return
        attempts += 1
        family = data["family"]
        counts[family] = counts.get(family, 0) + 1
        signature = fingerprint(
            {
                k: data[k]
                for k in (
                    "devices",
                    "members",
                    "values",
                    "programs",
                    "messages",
                    "outputs",
                    "payload_bytes",
                    "precision",
                )
            }
        )
        if signature in seen:
            return
        seen.add(signature)
        validate(data, topology)
        result = simulate(data, topology)
        data["fingerprint"] = signature
        entry = dict(predicted_us=result["us"], plan=data)
        rows = best.setdefault(family, [])
        rows.append(entry)
        rows.sort(key=lambda v: (v["predicted_us"], v["plan"]["fingerprint"]))
        del rows[shortlist:]

    if len(members) == 1:
        p = DAG(
            topology, members, members, payload_bytes, precision, "local", "identity"
        )
        consider(p.finish().export())
    else:
        # Always keep a usable arbitrary connected-graph route, including relays.
        for nodes in (members, members + helpers) if helpers else (members,):
            for trial in range(min(16, len(nodes) + 8)):
                jitter = (
                    None
                    if trial < len(nodes)
                    else {
                        tuple(sorted((a, b))): rng.uniform(0.55, 1.8)
                        for a, b in itertools.combinations(range(len(nodes)), 2)
                    }
                )
                root = trial % len(nodes)
                try:
                    parents = shortest_tree(
                        topology,
                        nodes,
                        root,
                        payload_bytes * (2 if precision == "fp32" else 1),
                        jitter,
                    )
                except ValueError:
                    break
                keep = set(nodes.index(v) for v in members)
                for r in list(keep):
                    while parents[r] >= 0:
                        r = parents[r]
                        keep.add(r)
                chosen = [nodes[r] for r in sorted(keep)]
                if len(chosen) != len(nodes):
                    chosen_root = chosen.index(nodes[root])
                    parents = shortest_tree(
                        topology, chosen, chosen_root, payload_bytes
                    )
                    root = chosen_root
                consider(
                    tree(
                        topology,
                        chosen,
                        members,
                        payload_bytes,
                        precision,
                        root,
                        parents,
                    )
                )
        nodes = members
        n = len(nodes)
        groups = fast_groups(topology, nodes)
        if all((a, b) in topology.labels for a in nodes for b in nodes if a != b):
            p = DAG(
                topology,
                nodes,
                members,
                payload_bytes,
                precision,
                "direct-clique",
                "clique",
            )
            p.reduce_groups([tuple(range(n))])
            consider(p.finish().export())
        if n & (n - 1) == 0:
            # Rank permutations give canonical doubling trees; independent
            # random pairings could round the same shards differently by rank.
            for trial in range(min(128, max(0, (budget - attempts) // 4))):
                order = list(range(n))
                if trial:
                    rng.shuffle(order)
                rounds = [
                    [(order[i], order[i ^ bit]) for i in range(n) if i < (i ^ bit)]
                    for bit in (1 << k for k in range(n.bit_length() - 1))
                ]
                if any(
                    (nodes[a], nodes[b]) not in topology.labels
                    for matching in rounds
                    for a, b in matching
                ):
                    continue
                p = DAG(
                    topology,
                    nodes,
                    members,
                    payload_bytes,
                    precision,
                    "recursive-doubling",
                    "doubling",
                )
                for matching in rounds:
                    old = p.current[:]
                    p.wave(
                        [
                            (s, d, old[s])
                            for a, b in matching
                            for s, d in ((a, b), (b, a))
                        ]
                    )
                    for a, b in matching:
                        values = sorted(
                            (old[a], old[b]), key=lambda v: p.values[v]["mask"]
                        )
                        p.current[a] = p.sum(a, values)
                        p.current[b] = p.sum(b, values)
                consider(p.finish().export())
        # Fast-island delivery: Cartesian products are lazily bounded, never
        # materialize an exponential assignment space. Random samples supplement
        # a load-balanced beam when the complete product exceeds the budget.
        if 1 < len(groups) <= min(8, n):
            options: list[list[tuple[int, int, int]]] = []
            for i, g in enumerate(groups):
                for j, h in enumerate(groups):
                    if i == j:
                        continue
                    choices = [
                        (i, a, b)
                        for a in g
                        for b in h
                        if (nodes[a], nodes[b]) in topology.labels
                    ]
                    if not choices:
                        options = []
                        break
                    options.append(choices)
                if not options:
                    break
            expected = len(groups) * (len(groups) - 1)
            if len(options) == expected:
                space = math.prod(len(v) for v in options)
                limit = min(budget - attempts, max(64, budget // 2))
                if space <= limit:
                    assignments = itertools.product(*options)
                else:
                    # Rotate cheap / low-load endpoints, avoiding lexicographic
                    # concentration on the first GPU of an island.
                    def samples():
                        for trial in range(limit):
                            load = [0] * n
                            assignment = []
                            for choices in options:
                                pick = min(
                                    choices,
                                    key=lambda v: (
                                        load[v[1]]
                                        + load[v[2]]
                                        + rng.random() * (0.25 if trial == 0 else 2),
                                        topology.links[nodes[v[1]], nodes[v[2]]][
                                            "latency_us"
                                        ],
                                    ),
                                )
                                load[pick[1]] += 1
                                load[pick[2]] += 1
                                assignment.append(pick)
                            yield assignment

                    assignments = samples()
                for assignment in assignments:
                    consider(
                        island_plan(
                            topology,
                            nodes,
                            members,
                            payload_bytes,
                            precision,
                            groups,
                            assignment,
                        )
                    )
        # Bounded hierarchy seeds from clique covers, including the PCIe planes.
        if n <= 8:
            from .plan import clique_partitions

            covers = clique_partitions(nodes, topology.labels)
            for first in covers:
                if not 2 <= len(first) <= 4:
                    continue
                membership = {r: i for i, g in enumerate(first) for r in g}
                for second in covers:
                    if attempts >= budget:
                        break
                    if not all(
                        len({membership[r] for r in g}) == len(first) for g in second
                    ):
                        continue
                    p = DAG(
                        topology,
                        nodes,
                        members,
                        payload_bytes,
                        precision,
                        "clique-hierarchy",
                        "hierarchy",
                    )
                    p.reduce_groups(first)
                    p.reduce_groups(second)
                    consider(p.finish().export())
        # Refine independent local posting/polling order without changing sums.
        from .plan import reorder_runs

        beam = [e["plan"] for rows in best.values() for e in rows[:2]]
        while beam and attempts < budget:
            p = reorder_runs(rng.choice(beam), rng)
            p["family"] = "instruction-order"
            consider(p)
            if attempts % 128 == 0:
                beam = [e["plan"] for rows in best.values() for e in rows[:2]]
    ranked = sorted(
        (e for rows in best.values() for e in rows),
        key=lambda e: (e["predicted_us"], e["plan"]["fingerprint"]),
    )
    selected = []
    identities = set()
    # Preserve family diversity for hardware qualification, then fill by score.
    for entry in (
        sorted((rows[0] for rows in best.values()), key=lambda e: e["predicted_us"])
        + ranked
    ):
        if entry["plan"]["fingerprint"] in identities:
            continue
        selected.append(entry)
        identities.add(entry["plan"]["fingerprint"])
        if len(selected) == shortlist:
            break
    selected.sort(key=lambda e: (e["predicted_us"], e["plan"]["fingerprint"]))
    return dict(
        schema=SCHEMA,
        key=key,
        topology=topology.export(),
        members=members,
        helpers=helpers,
        payload_bytes=payload_bytes,
        precision=precision,
        status="shortlisted" if selected else "fallback-required",
        candidates=selected,
        evaluations=attempts,
        unique_plans=len(seen),
        counts=counts,
        elapsed_s=time.monotonic() - start,
        selection="model-only",
        optimality=(
            "Best within a bounded candidate search; hardware qualification required"
        ),
    )


def select_measured(result, measure, *, context):
    """Callback must qualify every candidate and return real complete-call samples.

    context should include UUIDs, driver/runtime/backend ABI, peer state and
    concurrent group layout. The caller synchronizes group participants.
    """
    measurements = []
    for candidate in result["candidates"]:
        receipt = measure(candidate["plan"])
        if receipt.get("correct") is not True:
            continue
        samples = receipt.get("samples_us", [])
        if len(samples) < 3 or any(
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or v <= 0
            for v in samples
        ):
            raise ValueError("Invalid measured samples")
        import statistics

        measurements.append(
            dict(
                fingerprint=candidate["plan"]["fingerprint"],
                median_us=statistics.median(samples),
                receipt=receipt,
            )
        )
    if not measurements:
        return dict(
            status="fallback-required", planner_key=result["key"], context=context
        )
    winner = min(measurements, key=lambda r: (r["median_us"], r["fingerprint"]))
    return dict(
        schema=SCHEMA,
        status="measured",
        planner_key=result["key"],
        context=context,
        measurements=measurements,
        winner=winner["fingerprint"],
        key=fingerprint(dict(planner_key=result["key"], context=context)),
    )


def save_selection(path, result, receipt):
    """Atomic cache with an exact configuration identity, never a shape wildcard."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if receipt.get("status") != "measured" or receipt["planner_key"] != result["key"]:
        raise ValueError("Unqualified cache entry")
    data = dict(result=result, receipt=receipt)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_selection(path, result, context):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    receipt = data.get("receipt", {})
    if not isinstance(receipt, dict):
        return None
    expected = fingerprint(dict(planner_key=result["key"], context=context))
    if (
        receipt.get("status") != "measured"
        or receipt.get("key") != expected
        or receipt.get("planner_key") != result["key"]
    ):
        return None
    candidates = {v["plan"]["fingerprint"]: v["plan"] for v in result["candidates"]}
    plan = candidates.get(receipt.get("winner"))
    if plan is None:
        return None
    evidence = next(
        (
            v
            for v in receipt.get("measurements", [])
            if v.get("fingerprint") == receipt.get("winner")
        ),
        None,
    )
    if evidence is None or evidence.get("receipt", {}).get("correct") is not True:
        return None
    samples = evidence["receipt"].get("samples_us", [])
    if len(samples) < 3 or any(
        isinstance(v, bool)
        or not isinstance(v, (float, int))
        or not math.isfinite(v)
        or v <= 0
        for v in samples
    ):
        return None
    validate(plan, Topology(**result["topology"]))
    return dict(plan=plan, receipt=receipt)
