# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import itertools
import unittest

from _vllm_topology_all_reduce_tests.collective_planner import Topology, solve
from _vllm_topology_all_reduce_tests.planned_codegen import generate, layout


class PlannedCodeTests(unittest.TestCase):
    def topology(self, n):
        return Topology(
            list(range(n)),
            [
                dict(
                    source=a,
                    dest=b,
                    transport="NV2",
                    latency_us=1.0,
                    bytes_per_us=10000.0,
                )
                for a, b in itertools.permutations(range(n), 2)
            ],
        )

    def test_fp16_and_fp32_packet_width_and_native_arithmetic(self):
        topology = self.topology(3)
        for precision in ("fp16", "fp32"):
            plan = solve(topology, range(3), precision=precision, budget=32)[
                "candidates"
            ][0]["plan"]
            source = generate([plan], topology, 5120)
            self.assertEqual(source.count("__global__ void planned_"), 3)
            self.assertIn("int count", source)
            self.assertNotIn("i<320", source)
            self.assertIn("add.rn.f16x2", source)
            self.assertIn("__fadd_rn", source)
            self.assertIn(
                "add_wide(widen(", source
            ) if precision == "fp32" else self.assertIn("=add_half(", source)
            offsets, sizes = layout(plan, 5120)
            for dest in range(3):
                intervals = []
                for i, m in enumerate(plan["messages"]):
                    if m["dest"] != dest:
                        continue
                    length = (
                        5120
                        * (2 if plan["values"][m["value"]]["dtype"] == "fp32" else 1)
                        * 2
                    )
                    intervals.append((offsets[i], offsets[i] + length))
                    self.assertLessEqual(offsets[i] + length, sizes[dest])
                intervals.sort()
                self.assertTrue(
                    all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
                )

    def test_relay_leaf_never_loads_or_writes_member_input_output(self):
        topology = Topology(
            [10, 20, 30],
            [
                dict(
                    source=a,
                    dest=b,
                    transport="PIX",
                    latency_us=1.0,
                    bytes_per_us=6000.0,
                )
                for a, b in ((10, 20), (20, 10), (20, 30), (30, 20))
            ],
        )
        plan = solve(topology, [10, 30], helpers=[20], budget=32)["candidates"][0][
            "plan"
        ]
        source = generate([plan], topology, 5120)
        rank = plan["devices"].index(20)
        body = source.split(f"__global__ void planned_0_rank_{rank}(")[1].split(
            "__global__ void"
        )[0]
        self.assertIn("make_uint4(0,0,0,0)", body)
        self.assertNotIn("=x[i]", body)
        self.assertNotIn("out[i]=", body)

    def test_no_fixed_eight_gpu_assumption(self):
        for n in (1, 2, 4, 9):
            topology = self.topology(n)
            plan = solve(topology, range(n), budget=32)["candidates"][0]["plan"]
            source = generate([plan], topology, 10240)
            self.assertIn(f"char* data[{n}]", source)
            self.assertEqual(source.count("__global__ void planned_"), n)

    def test_reject_invalid_capacity_and_unsupported_mapping(self):
        topology = self.topology(2)
        plan = solve(topology, [0, 1], budget=32)["candidates"][0]["plan"]
        with self.assertRaises(ValueError):
            generate([plan], topology, 5130)
        broken = copy.deepcopy(plan)
        broken["messages"][0]["dest"] = broken["messages"][0]["source"]
        with self.assertRaises(ValueError):
            generate([broken], topology, 5120)


if __name__ == "__main__":
    unittest.main()
