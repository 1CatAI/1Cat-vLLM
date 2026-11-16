# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from _vllm_topology_all_reduce_tests.collective_setup import PreparedCollective
from _vllm_topology_all_reduce_tests.collective_tuner import (
    BoundBackend,
    Existing,
    choose,
)
from _vllm_topology_all_reduce_tests.planned_collectives import PlannedChannel


class TunerTests(unittest.TestCase):
    def test_fp32_policy_upcasts_existing_winner_and_fallback(self):
        import torch

        calls = []

        def reduce(x):
            calls.append(x.dtype)
            return x * 2

        backend = SimpleNamespace(all_reduce=reduce)
        source = torch.tensor([0.125, -0.25], dtype=torch.float16)
        original = source.clone()
        bound = BoundBackend(backend, precision="fp32")
        for route in (bound, PreparedCollective(None, {}, bound)):
            output = route.all_reduce(source)
            self.assertEqual(output.dtype, torch.float16)
            self.assertTrue(torch.equal(output, original * 2))
            self.assertTrue(torch.equal(source, original))
        self.assertEqual(calls, [torch.float32, torch.float32])
        BoundBackend(backend).all_reduce(source)
        self.assertEqual(calls[-1], torch.float16)

    def test_fp32_qualification_does_not_allow_half_accumulation_error(self):
        import torch

        adapter = Existing.__new__(Existing)
        adapter.accumulation = "fp32"
        values = [torch.tensor([v], dtype=torch.float16) for v in (1.0, 2**-12, -1.0)]
        reference = torch.stack(values).double().sum(0).half()
        self.assertTrue(adapter.result_matches(reference, reference, values))
        self.assertFalse(
            adapter.result_matches(
                (values[0] + values[1]) + values[2], reference, values
            )
        )

    def test_existing_capture_matches_serving_context(self):
        events = []

        @contextmanager
        def capture():
            events.append("enter")
            yield
            events.append("register")

        backend = SimpleNamespace(
            all_reduce=lambda x: x, ca_comm=SimpleNamespace(capture=capture)
        )
        adapter = Existing.__new__(Existing)
        adapter.accumulation = "fp16"
        adapter.backend = BoundBackend(backend)
        with adapter.capture():
            events.append("capture")
        self.assertEqual(events, ["enter", "capture", "register"])
        # NCCL-only communicators and standalone harnesses need no CA context.
        for backend in (
            SimpleNamespace(all_reduce=lambda x: x, ca_comm=None),
            SimpleNamespace(all_reduce=lambda x: x),
        ):
            adapter.backend = BoundBackend(backend)
            with adapter.capture():
                pass
        adapter.backend = SimpleNamespace()
        with adapter.capture():
            pass

    def test_existing_accepts_half_rounding_with_three_or_more_ranks(self):
        import torch

        adapter = Existing.__new__(Existing)
        adapter.accumulation = "fp16"
        native = PlannedChannel.__new__(PlannedChannel)
        for world in (3, 4, 8, 64):
            generator = torch.Generator().manual_seed(723001)
            values = [
                (torch.randn(512, generator=generator) * 0.125).half()
                for _ in range(world)
            ]
            reference = torch.stack(values).double().sum(0).half()
            for order in (values, values[::-1], values[::2] + values[1::2]):
                output = order[0].clone()
                for value in order[1:]:
                    output = output + value
                self.assertFalse(torch.equal(output, reference))
                self.assertTrue(adapter.result_matches(output, reference, values))
                self.assertFalse(native.result_matches(output, reference, values))

    def test_existing_tolerance_handles_cancellation_and_subnormals(self):
        import torch

        adapter = Existing.__new__(Existing)
        adapter.accumulation = "fp16"
        for raw in ((1.0, 2**-12, -1.0), (2**-24, -(2**-24), 2**-24)):
            values = [torch.tensor([v], dtype=torch.float16) for v in raw]
            output = (values[0] + values[1]) + values[2]
            reference = torch.stack(values).double().sum(0).half()
            self.assertTrue(adapter.result_matches(output, reference, values))

    def test_existing_rejects_wrong_contributions_and_nonfinite_results(self):
        import torch

        adapter = Existing.__new__(Existing)
        adapter.accumulation = "fp16"
        values = [torch.full((16,), v, dtype=torch.float16) for v in (0.125, 0.25, 0.5)]
        reference = torch.stack(values).double().sum(0).half()
        for bad in (
            reference - values[-1],
            reference + values[0],
            reference + 0.01,
            torch.zeros_like(reference),
            torch.full_like(reference, float("nan")),
            torch.full_like(reference, float("inf")),
        ):
            self.assertFalse(adapter.result_matches(bad, reference, values))

    def test_existing_tolerance_does_not_relax_native_signed_zero_check(self):
        import torch

        native = PlannedChannel.__new__(PlannedChannel)
        zero = torch.zeros(16, dtype=torch.float16)
        self.assertFalse(native.result_matches(-zero, zero, [zero, zero]))
        self.assertTrue(native.result_matches(zero, zero, [zero, zero]))

    def test_installing_wrapper_cannot_recurse_into_old_communicator(self):
        class Communicator:
            def all_reduce(self, x):
                return x + 1

        old: Any = Communicator()
        bound = BoundBackend(old)
        old.all_reduce = lambda x: bound.all_reduce(x) * 2
        self.assertEqual(old.all_reduce(3), 8)

    def test_only_clear_native_improvement_replaces_existing(self):
        self.assertEqual(
            choose(dict(native=[6.0, 6.1, 5.9], existing=[9.0, 8.9, 9.1])), "native"
        )
        self.assertEqual(
            choose(dict(native=[9.0, 8.9, 9.1], existing=[6.0, 6.1, 5.9])), "existing"
        )
        self.assertEqual(
            choose(dict(native=[5.99, 6.0, 5.98], existing=[6.0, 6.0, 6.0])), "existing"
        )
        self.assertEqual(
            choose(dict(native=[5.99, 6.0, 5.98], existing=[6.0, 6.0, 6.0]), margin=0),
            "native",
        )

    def test_medians_resist_one_fast_outlier_and_reject_bad_receipts(self):
        self.assertEqual(
            choose(dict(native=[1.0, 10.0, 10.0], existing=[6.0, 6.0, 6.0])), "existing"
        )
        for values in (
            [1.0],
            [1.0, 2.0, float("inf")],
            [1.0, 2.0, float("nan")],
            [1.0, 2.0, True],
            [-1.0, 2.0, 3.0],
        ):
            with self.assertRaises(ValueError):
                choose(dict(native=values, existing=[6.0, 6.0, 6.0]))


if __name__ == "__main__":
    unittest.main()
