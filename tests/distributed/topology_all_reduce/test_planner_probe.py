# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from _vllm_topology_all_reduce_tests.planner_probe import (
    from_text,
    matrix,
    refresh,
    visible_devices,
)


def table(nodes, diagonal, off):
    return (
        "\t"
        + "\t".join("GPU" + str(n) for n in nodes)
        + "\n"
        + "\n".join(
            "GPU"
            + str(a)
            + "\t"
            + "\t".join(diagonal if a == b else off for b in nodes)
            for a in nodes
        )
    )


class ProbeTests(unittest.TestCase):
    def test_nonconsecutive_ids_and_more_than_eight_cards(self):
        for nodes in ([0, 3, 11], list(range(12))):
            topo = from_text(
                table(nodes, "X", "NV2"),
                table(nodes, "X", "OK"),
                table(nodes, "X", "OK"),
                hardware="uuids",
            )
            self.assertEqual(topo.nodes, nodes)
            self.assertEqual(len(topo.labels), len(nodes) * (len(nodes) - 1))
            self.assertEqual(topo.identity["hardware"], "uuids")

    def test_single_gpu_header_with_cpu_affinity(self):
        nodes, labels = matrix(" GPU7 CPU Affinity NUMA Affinity\nGPU7 X 0-31 0\n")
        self.assertEqual(nodes, [7])
        self.assertEqual(labels[7, 7], "X")
        topo = from_text(" GPU7\nGPU7 X\n", " GPU7\nGPU7 X\n", " GPU7\nGPU7 X\n")
        self.assertEqual(topo.nodes, [7])
        self.assertFalse(topo.edges)

    def test_negotiated_pcie_width_scales_estimate_and_invalidates_profile(self):
        nodes = [0, 1]
        args = (
            table(nodes, "X", "PIX"),
            table(nodes, "X", "OK"),
            table(nodes, "X", "OK"),
        )
        slow = from_text(
            *args,
            hardware="0, GPU-a, bus0, driver, 3, 8\n1, GPU-b, bus1, driver, 3, 8\n",
        )
        fast = from_text(
            *args,
            hardware="0, GPU-a, bus0, driver, 4, 16\n1, GPU-b, bus1, driver, 4, 16\n",
        )
        self.assertAlmostEqual(
            fast.links[0, 1]["bytes_per_us"] / slow.links[0, 1]["bytes_per_us"],
            4.0,
            places=6,
        )
        with patch(
            "_vllm_topology_all_reduce_tests.planner_probe.discover", return_value=fast
        ):
            actual = refresh(slow)
        self.assertEqual(actual.key, fast.key)

    def test_unsupported_or_one_direction_only_not_invented(self):
        topo = from_text(
            table([0, 1], "X", "PIX"),
            table([0, 1], "X", "OK"),
            table([0, 1], "X", "NS"),
        )
        self.assertFalse(topo.labels)
        with self.assertRaises(ValueError):
            matrix(" GPU0 GPU1\nGPU0 X\nGPU1 OK X")

    def test_rewiring_or_uuid_change_discards_old_profiles(self):
        old = from_text(
            table([0, 1], "X", "NV2"),
            table([0, 1], "X", "OK"),
            table([0, 1], "X", "OK"),
            hardware="old",
        )
        old.edges[0]["latency_us"] = 99
        old.links[0, 1]["latency_us"] = 99
        for link, hardware in (("PIX", "old"), ("NV2", "new")):
            new = from_text(
                table([0, 1], "X", link),
                table([0, 1], "X", "OK"),
                table([0, 1], "X", "OK"),
                hardware=hardware,
            )
            with patch(
                "_vllm_topology_all_reduce_tests.planner_probe.discover",
                return_value=new,
            ):
                actual = refresh(old)
            self.assertNotEqual(actual.links[0, 1]["latency_us"], 99)
        same = from_text(
            table([0, 1], "X", "NV2"),
            table([0, 1], "X", "OK"),
            table([0, 1], "X", "OK"),
            hardware="old",
        )
        with patch(
            "_vllm_topology_all_reduce_tests.planner_probe.discover", return_value=same
        ):
            actual = refresh(old)
        self.assertEqual(actual.links[0, 1]["latency_us"], 99)

    def test_cuda_ordinal_mapping_uses_uuid_instead_of_environment_assumptions(self):
        cuda = SimpleNamespace(
            device_count=lambda: 2,
            get_device_properties=lambda d: SimpleNamespace(
                uuid=["GPU-bb", "GPU-aa"][d]
            ),
        )
        with (
            patch.dict(
                "sys.modules", {"torch": SimpleNamespace(cuda=cuda, accelerator=cuda)}
            ),
            patch(
                "_vllm_topology_all_reduce_tests.planner_probe.subprocess.check_output",
                return_value="0, GPU-aa\n3, GPU-bb\n",
            ),
        ):
            self.assertEqual(visible_devices(), [3, 0])


if __name__ == "__main__":
    unittest.main()
