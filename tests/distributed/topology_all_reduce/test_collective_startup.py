# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


"""Startup sizing follows model/speculation metadata instead of model names."""

import unittest
from types import SimpleNamespace

from _vllm_topology_all_reduce_tests.startup import payloads


def model(hidden):
    return SimpleNamespace(get_hidden_size=lambda: hidden)


class StartupTests(unittest.TestCase):
    def test_target_only(self):
        self.assertEqual(payloads(model(5120)), [10240])

    def test_different_draft_width(self):
        spec = SimpleNamespace(num_speculative_tokens=7, draft_model_config=model(4096))
        self.assertEqual(payloads(model(5120), spec), [8192, 10240, 65536, 81920])

    def test_mtp_and_shared_width_deduplication(self):
        spec = SimpleNamespace(num_speculative_tokens=4, draft_model_config=model(2560))
        self.assertEqual(payloads(model(2560), spec), [5120, 25600])

    def test_native_packet_alignment(self):
        with self.assertRaises(ValueError):
            payloads(model(1537))


class InstallTests(unittest.TestCase):
    """Exercise startup wiring with a communicator, without a CUDA runtime."""

    def run_install(self, *, world=2, remote=False, precision="fp16"):
        import tempfile
        from contextlib import ExitStack
        from unittest.mock import Mock, patch

        from _vllm_topology_all_reduce_tests import collective_tuner, startup

        calls = []
        comm = SimpleNamespace(
            all_reduce=lambda x: ("existing", x.nbytes),
            destroy=lambda: calls.append("destroy"),
        )
        old_reduce = comm.all_reduce
        lifecycle = dict(cpu_alive=True)
        tp = SimpleNamespace(
            world_size=world,
            rank_in_group=0,
            ranks=list(range(world)),
            device_communicator=comm,
            cpu_group="cpu",
        )

        def destroy_group():
            lifecycle["cpu_alive"] = False
            comm.destroy()

        tp.destroy = destroy_group

        def gather(output, value, **kwargs):
            output[:] = [value] * world
            if remote and isinstance(value, str):
                output[-1] = value + "-other-node"

        dist = SimpleNamespace(all_gather_object=gather)
        torch = SimpleNamespace(
            float16="fp16",
            distributed=dist,
            accelerator=SimpleNamespace(synchronize=lambda device: None),
        )
        routes = []

        def prepare(group, device, **kwargs):
            self.assertIs(comm.all_reduce, old_reduce)
            self.assertEqual(kwargs["precision"], precision)
            self.assertTrue(str(kwargs["cache"]).endswith(f"-{precision}.json"))
            width = kwargs["payload_bytes"]

            def close_route():
                self.assertTrue(lifecycle["cpu_alive"])
                calls.append(width)

            route = SimpleNamespace(
                receipt=dict(winner="native", cache_hit=True),
                all_reduce=lambda x: ("native", width),
                close=close_route,
            )
            routes.append(route)
            return route

        worker = SimpleNamespace(
            device="cuda:0",
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(
                    topology_all_reduce=SimpleNamespace(
                        precision=precision, cache_dir="unused"
                    )
                ),
                model_config=SimpleNamespace(
                    model="synthetic", get_hidden_size=lambda: 5120
                ),
                speculative_config=SimpleNamespace(
                    method="synthetic", num_speculative_tokens=7
                ),
                scheduler_config=SimpleNamespace(max_num_seqs=4),
            ),
        )
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            worker.vllm_config.kernel_config.topology_all_reduce.cache_dir = folder
            vllm = SimpleNamespace()
            stack.enter_context(
                patch.dict(
                    "sys.modules",
                    {
                        "torch": torch,
                        "torch.distributed": dist,
                        "vllm": vllm,
                        "vllm.distributed": SimpleNamespace(get_tp_group=lambda: tp),
                        "vllm.logger": SimpleNamespace(init_logger=lambda name: Mock()),
                    },
                )
            )
            mock = stack.enter_context(
                patch.object(collective_tuner, "prepare_best", side_effect=prepare)
            )
            result = startup.install(worker)
            return tp, result, calls, mock.call_count

    def test_local_and_multinode_leave_backend_unchanged(self):
        for kwargs in (dict(world=1), dict(remote=True)):
            tp, result, calls, count = self.run_install(**kwargs)
            comm = tp.device_communicator
            self.assertEqual(result, {})
            self.assertEqual(count, 0)
            self.assertEqual(
                comm.all_reduce(SimpleNamespace(nbytes=10240))[0], "existing"
            )
            self.assertEqual(calls, [])

    def test_freeze_exact_signatures_and_close_before_backend(self):
        tp, result, calls, count = self.run_install()
        comm = tp.device_communicator
        self.assertEqual(sorted(result), [10240, 81920])
        self.assertEqual(count, 2)
        tensor = SimpleNamespace(
            nbytes=10240, dtype="fp16", is_contiguous=lambda: True, data_ptr=lambda: 16
        )
        self.assertEqual(comm.all_reduce(tensor), ("native", 10240))
        tensor.nbytes = 16384
        self.assertEqual(comm.all_reduce(tensor)[0], "existing")
        tensor.nbytes = 10240
        tensor.dtype = "bf16"
        self.assertEqual(comm.all_reduce(tensor)[0], "existing")
        tensor.dtype = "fp16"
        tensor.data_ptr = lambda: 2
        self.assertEqual(comm.all_reduce(tensor)[0], "existing")
        tp.destroy()
        self.assertEqual(calls, [10240, 81920, "destroy"])

    def test_explicit_fp32_reaches_tuner_and_uses_separate_cache(self):
        tp, result, calls, count = self.run_install(precision="fp32")
        self.assertEqual(count, 2)
        self.assertEqual(sorted(result), [10240, 81920])
        tp.destroy()
        self.assertEqual(calls, [10240, 81920, "destroy"])

    def test_worker_installs_before_compile_and_capture(self):
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        source = (root / "vllm/v1/worker/gpu_worker.py").read_text()
        tree = ast.parse(source)
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "compile_or_warm_up_model"
        )
        text = ast.get_source_segment(source, method)
        assert text is not None
        self.assertIn("kernel_config.topology_all_reduce.enabled", text)
        self.assertIn('"_topology_all_reduce_installed", False', text)
        self.assertLess(text.index("startup.install(self)"), text.index("._dummy_run("))
        self.assertLess(
            text.index("startup.install(self)"), text.index(".capture_model(")
        )


if __name__ == "__main__":
    unittest.main()
