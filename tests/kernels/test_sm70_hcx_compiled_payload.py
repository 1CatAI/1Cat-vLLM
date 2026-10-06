# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-rank HCX payload ownership under Inductor and CUDA graph replay."""

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp


def _worker_run(rank, port, results):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce_sum2,
    )
    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
    from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual
    from vllm.models.qwen4_exp.nvidia.sm70_hcx import (
        get_hcx_runtime,
        register_moe_runner,
    )

    torch.accelerator.set_device_index(rank)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(4, rank, f"tcp://127.0.0.1:{port}", rank, "nccl")
        initialize_model_parallel(4, 1)
        config = HyperConnectionConfig(
            hc_count=4,
            hidden_size=2560,
            params_dtype=torch.float16,
            hc_lowrank=320,
            rms_norm_eps=1e-6,
            hc_per_branch_norm=True,
        )
        torch.manual_seed(0)
        module = GatedResidual(config, prefix="payload").cuda().half()
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.02)
        runtime = get_hcx_runtime(torch.device("cuda", rank))
        assert runtime.enabled, runtime.reason
        runtime.diagnostic = True
        module.enable_partial_inputs("payload", runtime)
        module._hcx_moe_payload = True
        register_moe_runner(
            "payload",
            SimpleNamespace(
                _maybe_sm70_moe_sum2_allreduce=lambda a, b, n: (
                    tensor_model_parallel_all_reduce_sum2(a, b)
                ),
            ),
        )

        @torch.compile(backend="inductor", dynamic=True, fullgraph=True)
        def compiled(x, hidden, injection):
            payload = torch.ops.vllm.qwen38_sm70_hcx_moe_output(
                x * 4, x * 3, "payload", 2560
            )
            scratch = x * 17
            outputs = module.combine_and_mix(hidden, payload, injection)
            return outputs, scratch

        report = []
        from vllm.models.qwen4_exp.nvidia import ple_layer

        @torch.compile(backend="inductor", dynamic=True, fullgraph=True)
        def ple_snapshots(x):
            first = ple_layer.snapshot_ple_diagnostic(x * 3, "before", True)
            second = ple_layer.snapshot_ple_diagnostic(first * 2, "after", True)
            return second, x * 17

        ple_input = torch.randn(5, 16, device="cuda").half()
        ple_snapshots(ple_input)
        ple_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ple_graph):
            ple_result, ple_scratch = ple_snapshots(ple_input)
        for multiplier in (1, 2):
            ple_input.mul_(multiplier)
            ple_graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                ple_layer._PLE_DIAGNOSTIC_BUFFERS["before"], ple_input * 3
            )
            torch.testing.assert_close(
                ple_layer._PLE_DIAGNOSTIC_BUFFERS["after"], ple_result
            )
            torch.testing.assert_close(ple_scratch, ple_input * 17)
        for rows in (5, 20):
            torch.manual_seed(1)
            hidden = (torch.randn(rows, 10240, device="cuda") * 0.5).half()
            injection = torch.randn(rows, 4, device="cuda").half()
            torch.manual_seed(100 + rank)
            x = (torch.randn(rows, 2560, device="cuda") * 0.02).half()
            compiled(x, hidden, injection)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs, scratch = compiled(x, hidden, injection)
            for _ in range(2):
                x.copy_(torch.randn_like(x) * 0.02)
                graph.replay()
                reduced = tensor_model_parallel_all_reduce_sum2(x * 4, x * 3)
                reference = module._combine_and_mix_reduced(hidden, reduced, injection)
                torch.accelerator.synchronize()
                errors = []
                for actual, expected in zip(outputs, reference):
                    error = float(
                        (actual.float() - expected.float()).norm()
                        / (expected.float().norm() + 1e-9)
                    )
                    assert error < 0.002, (rank, rows, error)
                    errors.append(error)
                torch.testing.assert_close(scratch, x * 17, rtol=0, atol=0)
                if rows == 5:
                    snapshot = runtime.snapshots["payload"]
                    torch.testing.assert_close(snapshot["partial"], x * 3)
                    torch.testing.assert_close(snapshot["secondary"], x * 4)
                    for key, value in zip(
                        ("hidden_out", "block_out", "injection_out"), outputs
                    ):
                        torch.testing.assert_close(snapshot[key], value, rtol=0, atol=0)
                report.append((rows, errors))
        results.put((rank, report))


def _worker(rank, port, results):
    import traceback

    try:
        _worker_run(rank, port, results)
    except BaseException:
        results.put((rank, {"error": traceback.format_exc()}))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hcx_owned_payload_compilation_and_graph_replay():
    if torch.cuda.device_count() != 4 or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Four visible SM70 devices required")
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_worker, args=(r, 29751, results)) for r in range(4)
    ]
    for worker in workers:
        worker.start()
    try:
        reports = [results.get(timeout=300) for _ in workers]
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
    assert all(isinstance(report, list) for _, report in reports), reports
    assert all(worker.exitcode == 0 for worker in workers)
    assert sorted(rank for rank, _ in reports) == list(range(4))
    print(sorted(reports))
