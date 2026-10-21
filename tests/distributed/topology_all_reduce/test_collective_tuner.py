# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from typing import Any

from _vllm_topology_all_reduce_tests.collective_tuner import BoundBackend, choose


class TunerTests(unittest.TestCase):
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
