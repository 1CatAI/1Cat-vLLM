# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest
import torch
import torch.multiprocessing as mp

from vllm.model_executor.kernels.ple.host_result import (
    HostResultRegion,
    host_result_capability,
    publish_host_flag,
    wait_host_resets,
)
from vllm.model_executor.layers.ple_offload_layer import CpuGpuSemaphore


def _producer(queue, acknowledgements, buffers, flags):
    acknowledgements.put("ready")
    while (item := queue.get()) is not None:
        count, sequence = item
        wait_host_resets(flags)
        time.sleep(0.005)
        for buffer in buffers:
            values = torch.arange(count * buffer.shape[1]).reshape(count, -1)
            buffer[:count].copy_((values + sequence * 19) % 127)
        for flag in flags:
            publish_host_flag(flag)
        acknowledgements.put(sequence)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_delayed_producer_changing_width_captured_bytes(dtype):
    regions, outputs, graphs = [], [], []
    process = None
    ctx = mp.get_context("spawn")
    queue, acknowledgements = ctx.Queue(), ctx.Queue()
    try:
        for rank in range(min(4, torch.accelerator.device_count())):
            device = torch.device("cuda", rank)
            with torch.accelerator.device_index(device.index):
                output = torch.empty(16, 97, dtype=dtype, device=device)
                reason = host_result_capability(device)
                if reason:
                    pytest.skip(reason)
                region = HostResultRegion.create(output)
                regions.append(region)
                sem = CpuGpuSemaphore(device, host_region=region)
                hidden = torch.zeros(1, 1, dtype=torch.float16, device=device)
                by_width = {}
                capture_stream = torch.cuda.Stream(device=device)
                for count in (1, 2, 4, 8, 16):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        torch.ops.vllm.ple_offload_wait(
                            sem.flag_tensor, output, hidden, region.result, count
                        )
                        sem.reset()
                    by_width[count] = graph
                outputs.append(output)
                graphs.append(by_width)
        process = ctx.Process(
            target=_producer,
            args=(
                queue,
                acknowledgements,
                [r.result for r in regions],
                [r.flag for r in regions],
            ),
        )
        process.start()
        assert acknowledgements.get(timeout=60) == "ready"
        for region in regions:
            region.validate_registration()
        for step, count in enumerate([1, 2, 4, 8, 16, 16, 8, 4, 2, 1] * 4):
            queue.put((count, step))
            for rank, by_width in enumerate(graphs):
                with torch.accelerator.device_index(rank):
                    by_width[count].replay()
            assert acknowledgements.get(timeout=30) == step
            expected = (
                (torch.arange(count * 97).reshape(count, 97) + step * 19) % 127
            ).to(dtype)
            for region, output in zip(regions, outputs):
                with torch.accelerator.device_index(output.device.index):
                    torch.accelerator.synchronize()
                    assert torch.equal(output[:count].cpu(), expected)
                wait_host_resets([region.flag], timeout_s=1)
    finally:
        if process is not None:
            queue.put(None)
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
        for region in regions:
            region.close()
        queue.close()
        acknowledgements.close()
