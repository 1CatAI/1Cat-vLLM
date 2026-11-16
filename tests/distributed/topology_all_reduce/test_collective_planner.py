# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import itertools
import tempfile
import time
import unittest
from pathlib import Path

from _vllm_topology_all_reduce_tests.collective_planner import (
    Topology,
    load_selection,
    save_selection,
    select_measured,
    simulate,
    solve,
    validate,
)


def graph(nodes, edges, labels=None):
    return Topology(
        list(nodes),
        [
            dict(
                source=a,
                dest=b,
                transport=(labels or {}).get((a, b), "PIX"),
                latency_us=1.0,
                bytes_per_us=6000.0,
            )
            for a, b in edges
        ],
    )


class PlannerTests(unittest.TestCase):
    def test_local_and_arbitrary_sparse_sizes(self):
        for n in (1, 2, 3, 4, 5, 8, 12):
            nodes = [13 + i * 3 for i in range(n)]
            edges = [
                (a, b) for a, b in zip(nodes, nodes[1:]) for a, b in ((a, b), (b, a))
            ]
            topology = graph(nodes, edges)
            result = solve(topology, nodes, budget=32)
            self.assertEqual(result["status"], "shortlisted")
            self.assertLessEqual(result["evaluations"], 32)
            for entry in result["candidates"]:
                self.assertTrue(validate(entry["plan"], topology))

    def test_disconnected_and_unidirectional_fail_to_fallback(self):
        for edges in ([], [(0, 1)]):
            result = solve(graph([0, 1], edges), [0, 1], budget=32)
            self.assertEqual(result["status"], "fallback-required")
            self.assertFalse(result["candidates"])

    def test_external_relay_has_no_input_contribution(self):
        topology = graph(
            ["left", "relay", "right"],
            [
                ("left", "relay"),
                ("relay", "left"),
                ("right", "relay"),
                ("relay", "right"),
            ],
        )
        result = solve(topology, ["left", "right"], helpers=["relay"], budget=32)
        self.assertEqual(result["status"], "shortlisted")
        for candidate in result["candidates"]:
            p = candidate["plan"]
            validate(p, topology)
            helper = p["devices"].index("relay")
            self.assertEqual(p["values"][helper]["mask"], 0)
            self.assertEqual(p["values"][p["outputs"][0]]["mask"], 3)
            for m in p["messages"]:
                self.assertIn(
                    (p["devices"][m["source"]], p["devices"][m["dest"]]),
                    topology.labels,
                )

    def test_cache_requires_measured_finite_correct_receipt_and_exact_context(self):
        topology = graph([0, 1], [(0, 1), (1, 0)])
        result = solve(topology, [0, 1], budget=32)
        receipt = select_measured(
            result,
            lambda p: dict(correct=True, samples_us=[3.0, 2.0, 4.0]),
            context={"driver": "a", "concurrent_groups": [[0, 1]]},
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selection.json"
            save_selection(path, result, receipt)
            self.assertIsNotNone(load_selection(path, result, receipt["context"]))
            self.assertIsNone(load_selection(path, result, {"driver": "b"}))
            changed = solve(topology, [0, 1], payload_bytes=10240, budget=32)
            self.assertIsNone(load_selection(path, changed, receipt["context"]))
            changed = solve(topology, [1, 0], budget=32)
            self.assertIsNone(load_selection(path, changed, receipt["context"]))
        self.assertEqual(
            select_measured(
                result,
                lambda p: dict(correct=False, samples_us=[1.0, 2.0, 3.0]),
                context={},
            )["status"],
            "fallback-required",
        )
        with self.assertRaisesRegex(ValueError, "samples"):
            select_measured(
                result,
                lambda p: dict(correct=True, samples_us=[1.0, 2.0, float("nan")]),
                context={},
            )

    def test_current_graph_bounded_search_finds_balanced_delivery(self):
        # Synthetic crossed topology: NVLink 4+2+2, PCIe 4+4.
        islands = ((0, 1, 2, 3), (4, 5), (6, 7))
        planes = ((0, 1, 4, 6), (2, 3, 5, 7))
        labels = {
            (a, b): ("NV2" if any(a in g and b in g for g in islands) else "PIX")
            for a, b in itertools.permutations(range(8), 2)
            if any(a in g and b in g for g in islands + planes)
        }
        topology = graph(range(8), labels, labels)
        started = time.monotonic()
        result = solve(topology, range(8), precision="fp16", budget=2048, shortlist=8)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(result["evaluations"], 2048)
        balanced = [
            e for e in result["candidates"] if e["plan"]["family"] == "island-delivery"
        ]
        self.assertTrue(balanced)
        for entry in balanced:
            p = entry["plan"]
            payload = sum(
                m["bytes"]
                for m in p["messages"]
                if not topology.labels[
                    p["devices"][m["source"]], p["devices"][m["dest"]]
                ].startswith("NV")
            )
            self.assertEqual(payload, 30720)
        repeat = solve(topology, range(8), precision="fp16", budget=2048, shortlist=8)
        self.assertEqual(result["candidates"], repeat["candidates"])
        self.assertEqual(result["key"], repeat["key"])

    def test_explicit_shared_resources_and_duplex(self):
        resources = {"bus": dict(bytes_per_us=6000.0, lanes=1)}
        topology = Topology(
            [0, 1, 2, 3],
            [
                dict(
                    source=a,
                    dest=b,
                    transport="PIX",
                    latency_us=1.0,
                    bytes_per_us=6000.0,
                    resources=["bus"],
                )
                for a, b in itertools.permutations(range(4), 2)
            ],
            resources,
        )
        result = solve(topology, [0, 1, 2, 3], budget=32)
        for entry in result["candidates"]:
            trace = simulate(entry["plan"], topology, trace=True)["events"]
            sends = [v for v in trace if v["op"] == "send"]
            times = sorted(v["start_us"] for v in sends)
            self.assertEqual(len(times), len(set(times)))

    def test_invalid_profiles_and_duplicate_contributions_rejected(self):
        with self.assertRaises(ValueError):
            Topology(
                [0, 1],
                [dict(source=0, dest=1, latency_us=float("nan"), bytes_per_us=1)],
            )
        topology = graph([0, 1], [(0, 1), (1, 0)])
        result = solve(topology, [0, 1], budget=32)
        broken = copy.deepcopy(result["candidates"][0]["plan"])
        broken["values"][-1]["mask"] = 1
        with self.assertRaises(ValueError):
            validate(broken, topology)


if __name__ == "__main__":
    unittest.main()
