# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
import time

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_shared_protocol import (
    _sys_load_acquire,
    wait_shared_sequence,
)
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@triton.jit
def _acquire(flag, output):
    value = _sys_load_acquire(flag)
    tl.store(output, value)


def test_system_acquire_is_retained_in_generated_ptx():
    flag = torch.tensor([7], dtype=torch.int64, device="cuda")
    output = torch.zeros_like(flag)
    compiled = _acquire[(1,)](flag, output)
    assert "ld.acquire.sys.global.u64" in compiled.asm["ptx"]
    torch.testing.assert_close(output, flag, rtol=0, atol=0)


@triton.jit
def _wait_sequence(flags, output, TIMEOUT: tl.constexpr):
    wait_shared_sequence(flags, flags + 1, 7, TIMEOUT_NS=TIMEOUT)
    tl.store(output, 1)


def test_device_dead_owner_timeout_is_present_and_ready_sequence_passes():
    flags = torch.tensor([7, 0], device="cuda", dtype=torch.int64)
    output = torch.zeros(1, device="cuda", dtype=torch.int64)
    compiled = _wait_sequence[(1,)](flags, output, TIMEOUT=10_000_000)
    assert "%globaltimer" in compiled.asm["ptx"]
    assert "ld.acquire.sys.global.u64" in compiled.asm["ptx"]
    assert output.item() == 1


def _dead_owner_child(connection):
    try:
        flags = torch.zeros(2, device="cuda", dtype=torch.int64)
        output = torch.zeros(1, device="cuda", dtype=torch.int64)
        _wait_sequence.warmup(flags, output, TIMEOUT=10_000_000, grid=(1,))
        begin = time.monotonic()
        _wait_sequence[(1,)](flags, output, TIMEOUT=10_000_000)
        torch.accelerator.synchronize()
        connection.send((False, "GPU accepted a missing owner sequence"))
    except torch.AcceleratorError as error:
        connection.send(("launch failure" in str(error), time.monotonic() - begin))
    except BaseException as error:
        connection.send((False, repr(error)))
    finally:
        connection.close()


def test_missing_owner_traps_in_an_isolated_cuda_context():
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_dead_owner_child, args=(child,))
    process.start()
    child.close()
    try:
        assert parent.poll(60), "Missing owner left the GPU waiter stuck"
        passed, detail = parent.recv()
        assert passed, detail
        assert detail < 2.0, detail
        process.join(20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
