# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contribution-tracking DAG construction and independent instruction ordering."""

import hashlib
import itertools
import json
from typing import Any


class Plan:
    def __init__(self, devices, labels, name, family, payload_bytes):
        self.payload = payload_bytes
        self.devices = tuple(devices)
        self.labels = labels
        self.name = name
        self.family = family
        self.values: list[dict[str, Any]] = [
            dict(mask=1 << r, children=None, dtype="fp16") for r in range(len(devices))
        ]
        self.cache = {}
        self.have = [{r} for r in range(len(devices))]
        self.programs: list[list[dict[str, Any]]] = [[] for _ in devices]
        self.messages = []
        self.current = list(range(len(devices)))
        self.outputs = None

    def edge(self, a, b):
        return a != b and (self.devices[a], self.devices[b]) in self.labels

    def add(self, rank, left, right, dtype="fp16"):
        if left not in self.have[rank] or right not in self.have[rank]:
            raise ValueError("Missing local operand")
        if self.values[left]["mask"] & self.values[right]["mask"]:
            raise ValueError("Duplicate contribution")
        key = left, right, dtype
        if key not in self.cache:
            self.cache[key] = len(self.values)
            self.values.append(
                dict(
                    mask=self.values[left]["mask"] | self.values[right]["mask"],
                    children=[left, right],
                    dtype=dtype,
                )
            )
        value = self.cache[key]
        if value not in self.have[rank]:
            self.programs[rank].append(
                dict(op="add", value=value, left=left, right=right)
            )
            self.have[rank].add(value)
        return value

    def sum(self, rank, values, dtype="fp16"):
        unique = list(dict.fromkeys(values))
        value = unique[0]
        for other in unique[1:]:
            value = self.add(rank, value, other, dtype)
        return value

    def wave(self, transfers):
        """Publish all independent sends before receiving; no phase barrier."""
        pending = []
        seen = set()
        for source, dest, value in transfers:
            if value in self.have[dest] or (dest, value) in seen:
                continue
            if not self.edge(source, dest) or value not in self.have[source]:
                raise ValueError("Unavailable source or peer edge")
            seen.add((dest, value))
            message = len(self.messages)
            size = self.payload * (2 if self.values[value]["dtype"] == "fp32" else 1)
            self.messages.append(
                dict(source=source, dest=dest, value=value, bytes=size)
            )
            self.programs[source].append(
                dict(op="send", value=value, dest=dest, message=message)
            )
            pending.append((source, dest, value, message))
        for source, dest, value, message in pending:
            self.programs[dest].append(
                dict(op="recv", value=value, source=source, message=message)
            )
            self.have[dest].add(value)

    def reduce_groups(self, groups, dtype="fp16"):
        transfers = []
        results = {}
        loads = [0] * len(self.devices)
        for group in groups:
            vals = sorted(
                set(self.current[r] for r in group),
                key=lambda v: self.values[v]["mask"],
            )
            mask = 0
            for value in vals:
                if mask & self.values[value]["mask"]:
                    raise ValueError("Overlapping clique partials")
                mask |= self.values[value]["mask"]
            for dest in group:
                for value in vals:
                    if value in self.have[dest]:
                        continue
                    donors = [
                        s for s in group if value in self.have[s] and self.edge(s, dest)
                    ]
                    if not donors:
                        raise ValueError("Clique source cannot reach receiver")
                    source = min(
                        donors,
                        key=lambda s: (
                            loads[s],
                            not self.labels[
                                self.devices[s], self.devices[dest]
                            ].startswith("NV"),
                            s,
                        ),
                    )
                    loads[source] += 1
                    transfers.append((source, dest, value))
            results[tuple(group)] = vals
        self.wave(transfers)
        for group, vals in results.items():
            for rank in group:
                self.current[rank] = self.sum(rank, vals, dtype)

    def finish(self):
        goal = (1 << len(self.devices)) - 1
        if (
            any(self.values[v]["mask"] != goal for v in self.current)
            or len(set(self.current)) != 1
        ):
            raise ValueError("Outputs do not share one complete expression")
        self.outputs = list(self.current)
        return self

    def export(self):
        return dict(
            name=self.name,
            family=self.family,
            devices=list(self.devices),
            values=self.values,
            programs=self.programs,
            messages=self.messages,
            outputs=self.outputs,
            labels=[[a, b, label] for (a, b), label in sorted(self.labels.items())],
            fingerprint=hashlib.sha256(
                json.dumps(
                    [self.programs, self.values, self.outputs], sort_keys=True
                ).encode()
            ).hexdigest(),
        )


def partitions(items):
    if not items:
        yield ()
        return
    first, *rest = items
    for partition in partitions(rest):
        yield ((first,),) + partition
        for i, group in enumerate(partition):
            yield partition[:i] + ((first,) + group,) + partition[i + 1 :]


def clique_partitions(devices, labels):
    def clique(group):
        return all(
            (devices[a], devices[b]) in labels and (devices[b], devices[a]) in labels
            for a, b in itertools.combinations(group, 2)
        )

    return [
        p
        for p in partitions(tuple(range(len(devices))))
        if all(clique(group) for group in p)
    ]


def reorder_runs(data, rng):
    """Change issue/poll order within independent runs; preserve arithmetic."""
    changed = dict(data)
    changed["programs"] = [list(p) for p in data["programs"]]
    options = []
    for rank, program in enumerate(changed["programs"]):
        begin = 0
        while begin < len(program):
            end = begin + 1
            kind = program[begin]["op"]
            while end < len(program) and program[end]["op"] == kind:
                end += 1
            if kind in ("send", "recv") and end - begin > 1:
                options.append((rank, begin, end))
            begin = end
    if options:
        rank, begin, end = rng.choice(options)
        run = changed["programs"][rank][begin:end]
        rng.shuffle(run)
        changed["programs"][rank][begin:end] = run
    changed["fingerprint"] = hashlib.sha256(
        json.dumps(
            [changed["programs"], changed["values"], changed["outputs"]], sort_keys=True
        ).encode()
    ).hexdigest()
    return changed
