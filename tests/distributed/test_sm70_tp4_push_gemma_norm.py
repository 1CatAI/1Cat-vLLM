# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _check_tp2_norm_handoff(rank: int, rendezvous: str) -> None:
    from vllm import _custom_ops as ops
    from vllm import _sm70_ops as linear_ops
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.model_executor.layers.layernorm import (
        _sm70_dflash2_gemma_fused_add_rms_norm,
    )

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    ca = CustomAllreduce(
        dist.group.WORLD, torch.device("cuda", rank), max_size=1024 * 1024
    )
    assert not ca.disabled
    torch.manual_seed(817)
    matrix = torch.randn(2048, 5120, device="cuda").to(torch.float8_e4m3fn)
    codes, scales = linear_ops.fp8_qpn8_prepare_sm70(
        matrix, torch.full((2048, 1), 0.01, device="cuda")
    )
    try:
        for rows in (1, 5, 8, 9, 16, 17, 32, 40, 64):
            x = torch.zeros(rows, 5120, device="cuda", dtype=torch.float16)
            residual = torch.zeros_like(x, dtype=torch.float32)
            weight = torch.zeros(5120, device="cuda", dtype=torch.float32)
            ca.custom_all_reduce(x)
            ca.all_reduce_gemma_rms_norm(x, residual, weight, 1e-6)
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(graph):
                for _ in range(17):
                    normalized, state = ca.all_reduce_gemma_rms_norm(
                        x, residual, weight, 1e-6
                    )
                    ca.custom_all_reduce(x)
                if rows <= 40:  # Raw QPN8 GEMM's row capacity.
                    projected = torch.empty(rows, 2048, device="cuda", dtype=x.dtype)
                    linear_ops.fp8_qpn8_gemm_sm70_out(
                        projected, normalized, codes, scales, 16, 2, True, False
                    )
            for cycle in range(6):
                torch.manual_seed(819 + cycle * 2 + rank)
                x.normal_().mul_(0.125 if cycle < 3 else 32)
                torch.manual_seed(901 + cycle)
                residual.normal_()
                weight.normal_(0, 0.05)
                if cycle == 0:
                    x.fill_(65504 if rank == 0 else -65504)
                dist.barrier()
                graph.replay()
                reduced = ca.custom_all_reduce(x)
                expected, expected_state = _sm70_dflash2_gemma_fused_add_rms_norm(
                    reduced, residual, weight, 1e-6
                )
                expected_fp32 = torch.empty_like(expected_state)
                ops.rms_norm(expected_fp32, expected_state, weight + 1, 1e-6)
                assert torch.equal(state, expected_state)
                assert torch.equal(normalized, expected)
                torch.testing.assert_close(
                    normalized, expected_fp32.half(), atol=2e-6, rtol=0.001
                )
                if rows <= 40:
                    clone_projection = torch.empty_like(projected)
                    linear_ops.fp8_qpn8_gemm_sm70_out(
                        clone_projection,
                        normalized.clone(),
                        codes,
                        scales,
                        16,
                        2,
                        True,
                        False,
                    )
                    assert torch.equal(projected, clone_projection)
    finally:
        ca.close()
        dist.destroy_process_group()


def test_sm70_tp2_norm_publishes_compatible_k16_activations(tmp_path):
    if torch.accelerator.device_count() != 2 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires two isolated NVLink V100 GPUs")
    mp.spawn(
        _check_tp2_norm_handoff,
        args=((tmp_path / "tp2-norm").as_uri(),),
        nprocs=2,
        join=True,
    )


def _check_push_norm(rank: int, rendezvous: str, algorithm: str | None) -> None:
    from vllm import _sm70_ops as linear_ops
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.model_executor.layers.layernorm import (
        _sm70_dflash2_gemma_fused_add_rms_norm,
    )

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    ca = CustomAllreduce(
        dist.group.WORLD, torch.device("cuda", rank), max_size=1024 * 1024
    )
    assert not ca.disabled and ca.fully_connected and ca.sm70_tp4_push_buffer_ptrs
    torch.manual_seed(717)
    projection_weight = torch.randn(2048, 5120, device="cuda").to(torch.float8_e4m3fn)
    projection_scales = torch.full((2048, 1), 0.01, device="cuda")
    codes, scales = linear_ops.fp8_qpn8_prepare_sm70(
        projection_weight, projection_scales
    )
    row_counts = (
        (1, 3, 5, 8, 9, 16, 17, 25, 32, 33, 40, 51, 52, 63, 64)
        if algorithm is None
        else (1, 5, 52, 64)
    )
    try:
        # MTP and DFlash retain independent packets across changing batch sizes.
        for rows, dtype in [
            (rows, dtype)
            for rows in row_counts
            for dtype in (torch.float16, torch.float32)
        ]:
            storage = torch.zeros(
                max(rows, 8), 5120, device="cuda", dtype=torch.float16
            )
            residual_storage = torch.zeros_like(storage, dtype=torch.float32)
            x = storage[:rows]
            residual = residual_storage[:rows]
            weight = torch.zeros(5120, device="cuda", dtype=dtype)
            ca.custom_all_reduce(x)
            ca.all_reduce_gemma_rms_norm(x, residual, weight, 1e-6)
            torch.accelerator.synchronize()
            dist.barrier()
            graphs, outputs = [], []
            for fused in (False, True):
                graph = torch.cuda.CUDAGraph()
                with ca.capture(), torch.cuda.graph(graph):
                    for _ in range(140):
                        if fused:
                            output = ca.all_reduce_gemma_rms_norm(
                                x, residual, weight, 1e-6
                            )
                            # Separate packets must survive interleaved ordinary AR.
                            ca.custom_all_reduce(x)
                            if rows != 8:
                                ca.all_reduce_gemma_rms_norm(
                                    storage[:8], residual_storage[:8], weight, 1e-6
                                )
                        else:
                            output = _sm70_dflash2_gemma_fused_add_rms_norm(
                                ca.custom_all_reduce(x), residual, weight, 1e-6
                            )
                graphs.append(graph)
                outputs.append(output)
            for cycle in range(12):
                amplitude = (2**-8, 0.125, 4, 32, 128, 512)[cycle % 6]
                torch.manual_seed(3104 + cycle * 4 + rank)
                x.normal_().mul_(amplitude)
                torch.manual_seed(4104 + cycle)
                residual.normal_().mul_(amplitude)
                weight.normal_().mul_(0.05)
                if cycle == 0:
                    x.zero_()
                    residual.zero_()
                elif cycle == 1:
                    # Cancellation exposes the distinct FP32 addition order of
                    # each two-stage partition before the FP16 result is stored.
                    x.fill_((65504.0, 0.1, -65504.0, 0.1)[rank])
                    residual.zero_()
                dist.barrier()
                for graph in graphs:
                    if rank == cycle % 4:
                        torch.cuda._sleep(10000)
                    # Exercise many generation/epoch transitions after each
                    # live input update, including intentionally skewed ranks.
                    for _ in range(61):
                        graph.replay()
                torch.accelerator.synchronize()
                expected, expected_residual = outputs[0]
                actual, actual_residual = outputs[1]
                assert torch.equal(actual_residual, expected_residual)
                assert torch.equal(actual, expected)
                values = expected_residual.double()
                oracle = values * torch.rsqrt(
                    values.square().mean(-1, keepdim=True) + 1e-6
                )
                oracle = (oracle * (weight.double() + 1)).half()
                lo = torch.nextafter(oracle, torch.full_like(oracle, -float("inf")))
                hi = torch.nextafter(oracle, torch.full_like(oracle, float("inf")))
                assert bool(torch.all((actual >= lo) & (actual <= hi)))
                eager, eager_residual = ca.all_reduce_gemma_rms_norm(
                    x, residual, weight, 1e-6
                )
                torch.accelerator.synchronize()
                assert torch.equal(eager, actual)
                assert torch.equal(eager_residual, actual_residual)
                if rows in (16, 32):
                    # A fused producer may publish K16 activations for the next
                    # projection. A clone forces an independent pack, detecting
                    # a stale or unwritten published buffer after route changes.
                    projected = torch.empty(rows, 2048, device="cuda", dtype=x.dtype)
                    reference_projection = torch.empty_like(projected)
                    linear_ops.fp8_qpn8_gemm_sm70_out(
                        projected, eager, codes, scales, 16, 2, True, False
                    )
                    linear_ops.fp8_qpn8_gemm_sm70_out(
                        reference_projection,
                        eager.clone(),
                        codes,
                        scales,
                        16,
                        2,
                        True,
                        False,
                    )
                    assert torch.equal(projected, reference_projection)
            del graphs, outputs
    finally:
        ca.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("algorithm", [None, "1stage", "2stage"])
def test_sm70_tp4_push_norm_fp64_and_interleaved_replay(
    tmp_path, monkeypatch, algorithm
):
    if torch.accelerator.device_count() != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires an isolated full NVLink mesh of four V100 GPUs")
    if algorithm is None:
        monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raising=False)
    else:
        monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", algorithm)
    mp.spawn(
        _check_push_norm,
        args=((tmp_path / "push-norm-rendezvous").as_uri(), algorithm),
        nprocs=4,
        join=True,
    )
