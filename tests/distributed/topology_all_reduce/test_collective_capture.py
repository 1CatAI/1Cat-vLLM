# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise the real qualification/timing loops with CPU graph stand-ins."""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from _vllm_topology_all_reduce_tests.collective_setup import measure_channel
from _vllm_topology_all_reduce_tests.collective_tuner import BoundBackend, Existing
from _vllm_topology_all_reduce_tests.planned_collectives import PlannedChannel


class CaptureTests(unittest.TestCase):
    def exercise(self, timing, *, custom=True, world=3, corrupt=False, native=False):
        import torch

        events = []
        active_graph = None
        capturing = False
        registered = False
        gathered: list[torch.Tensor] = []
        graph_outputs = []
        tensor_gathers = 0
        divisor = 16384 if timing else 1048576

        @contextmanager
        def capture():
            nonlocal capturing, registered
            capturing = True
            events.append("enter")
            yield
            capturing = False
            registered = True
            events.append("register")

        @contextmanager
        def graph_context(graph):
            nonlocal active_graph
            if custom:
                self.assertTrue(capturing)
            active_graph = graph
            events.append("cuda-enter")
            yield
            events.append("cuda-exit")
            active_graph = None

        class Graph:
            def __init__(self):
                self.operations = []

            def replay(graph):
                if custom:
                    self.assertTrue(registered)
                    self.assertFalse(capturing)
                events.append("replay")
                for i, (source, output) in enumerate(graph.operations):
                    values = (
                        [source + r / divisor for r in range(world)]
                        if i == 0
                        else [source] * world
                    )
                    result = values[0].clone()
                    for value in values[1:]:
                        result = result + value
                    if corrupt and i == 1:
                        result = result + 0.125
                    output.copy_(result)

        def all_reduce(source):
            if active_graph is not None:
                output = torch.empty_like(source)
                active_graph.operations.append((source, output))
                graph_outputs.append(output)
                return output
            result = gathered[0].clone()
            for value in gathered[1:]:
                result = result + value
            return result

        def gather(output, value, group):
            nonlocal gathered, tensor_gathers
            if isinstance(value, torch.Tensor):
                tensor_gathers += 1
                # Eager inputs differ by rank; replay outputs are identical.
                values = (
                    [value + r / divisor for r in range(world)]
                    if not any(value is tensor for tensor in graph_outputs)
                    else [value] * world
                )
                output[:] = values
                gathered = values
            else:
                output[:] = [value] * world

        empty = torch.empty

        def allocate(*args, **kwargs):
            if custom:
                self.assertTrue(capturing)
            events.append("allocate")
            return empty(*args, **kwargs)

        members = list(range(world))
        plan = dict(
            payload_bytes=64,
            members=members,
            devices=members,
            precision="fp16",
            values=[dict(children=None, dtype="fp16") for _ in members]
            + [
                dict(children=[0 if r == 1 else world + r - 2, r], dtype="fp16")
                for r in range(1, world)
            ],
            outputs=[2 * world - 2] * world,
            fingerprint="cpu-test",
        )
        backend = SimpleNamespace(
            all_reduce=all_reduce,
            ca_comm=SimpleNamespace(capture=capture) if custom else None,
        )
        with ExitStack() as stack:
            for target, replacement in (
                ("torch.distributed.get_rank", lambda group: 0),
                ("torch.distributed.get_world_size", lambda group: world),
                ("torch.distributed.all_gather_object", gather),
                ("torch.distributed.barrier", lambda group: None),
                ("torch.accelerator.synchronize", lambda device: None),
                ("torch.cuda.CUDAGraph", Graph),
                ("torch.cuda.graph", graph_context),
                (
                    "torch.cuda.Event",
                    lambda **kw: SimpleNamespace(
                        record=lambda: None, elapsed_time=lambda end: 1.0
                    ),
                ),
                ("torch.empty", allocate),
            ):
                stack.enter_context(patch(target, replacement))
            adapter: Any
            if native:
                adapter = PlannedChannel.__new__(PlannedChannel)
                adapter.group, adapter.device, adapter.plan = (
                    "cpu-group",
                    torch.device("cpu"),
                    plan,
                )
                adapter.native, adapter.rank, adapter.capacity = (
                    True,
                    0,
                    plan["payload_bytes"],
                )
                adapter.all_reduce = all_reduce
            else:
                adapter = Existing(
                    BoundBackend(backend), "cpu-group", torch.device("cpu"), plan
                )
            receipt = (
                measure_channel(adapter, repeats=3, warmup=1)
                if timing
                else adapter.qualify(repeats=2)
            )
        if custom:
            self.assertLess(events.index("enter"), events.index("allocate"))
            self.assertLess(events.index("cuda-exit"), events.index("register"))
            self.assertLess(events.index("register"), events.index("replay"))
        self.assertFalse(torch.cuda.is_initialized())
        receipt["tensor_gathers"] = tensor_gathers
        return receipt

    def test_timing_registers_graph_inputs_before_replay(self):
        result = self.exercise(timing=True)
        self.assertTrue(result["correct"])
        self.assertEqual(len(result["samples_us"]), 3)

    def test_qualification_uses_same_capture_lifecycle(self):
        self.assertEqual(self.exercise(timing=False)["status"], "passed")

    def test_nccl_only_backend_needs_no_custom_capture_context(self):
        self.assertTrue(self.exercise(timing=True, custom=False)["correct"])

    def test_corrupt_dependent_result_fails_both_loops(self):
        for timing in (False, True):
            for native in (False, True):
                with (
                    self.subTest(timing=timing, native=native),
                    self.assertRaises(RuntimeError),
                ):
                    self.exercise(
                        timing=timing, corrupt=True, native=native, custom=not native
                    )

    def test_generated_checks_do_not_gather_intermediate_tensors(self):
        result = self.exercise(timing=False, custom=False, native=True)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["tensor_gathers"], 4
        )  # Eager and initial graph inputs only.
        result = self.exercise(timing=True, custom=False, native=True)
        self.assertTrue(result["correct"])
        self.assertEqual(result["tensor_gathers"], 0)

    def test_large_group_probes_remain_finite(self):
        self.assertEqual(self.exercise(timing=False, world=64)["status"], "passed")
        self.assertTrue(self.exercise(timing=True, world=64)["correct"])


if __name__ == "__main__":
    unittest.main()
