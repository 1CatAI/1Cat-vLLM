# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.ple.host_result import (
    publish_host_flag,
    wait_host_resets,
)
from vllm.v1.ple_offload.protocol import PleOffloadRequest
from vllm.v1.ple_offload.worker import (
    PleOffloadInputBuffers,
    PleOffloadOutputTarget,
    PleOffloadRunner,
)


def test_host_publication_requires_acknowledgement():
    flag = torch.zeros(16, dtype=torch.int32).share_memory_()
    wait_host_resets([flag])
    publish_host_flag(flag)
    assert flag[0].item() == 1
    with pytest.raises(TimeoutError):
        wait_host_resets([flag], timeout_s=0)
    with pytest.raises(ValueError):
        publish_host_flag(torch.zeros(1, dtype=torch.float32))


def test_worker_publishes_exact_rows_without_cuda_submission(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Mapped CPU producer must not submit CUDA work")

    monkeypatch.setattr(torch.cuda, "Stream", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.Tensor, "is_pinned", forbidden)

    class Layer:
        def forward_impl(self, hidden, ids, offsets, context, output_buffer):
            assert offsets.tolist() == [0, 3]
            assert context.tolist() == [[9, 10]]
            output_buffer[:3].copy_(ids[:, None].expand(3, 7))
            return output_buffer[:3]

    buffers = [torch.full((8, 7), -1, dtype=torch.int32) for _ in range(4)]
    flags = [torch.zeros(16, dtype=torch.int32) for _ in buffers]
    runner = PleOffloadRunner.__new__(PleOffloadRunner)
    runner._clamp_input_ids = False
    runner._layers = {"layer": Layer()}
    runner._worker_targets = {
        0: {
            "layer": [
                PleOffloadOutputTarget(
                    tp_rank=i,
                    gpu_output_buffer=None,
                    sem=SimpleNamespace(flag_tensor=flag),
                    copy_stream=None,
                    cpu_output_buffer=buffer,
                )
                for i, (buffer, flag) in enumerate(zip(buffers, flags))
            ]
        }
    }
    runner._input_bufs = {
        0: PleOffloadInputBuffers(
            input_ids_buf=torch.tensor([11, 12, 13], dtype=torch.int32),
            query_start_loc_buf=torch.tensor([0, 3], dtype=torch.int32),
            ngram_context_buf=torch.tensor([[9, 10]], dtype=torch.int32),
        )
    }
    runner._pinned_bufs = {0: {"layer": buffers[0]}}
    runner._handle_requests([PleOffloadRequest(0, 3, 1)])
    for buffer, flag in zip(buffers, flags):
        assert torch.equal(buffer[:3], torch.tensor([11, 12, 13])[:, None].expand(3, 7))
        assert (buffer[3:] == -1).all()
        assert flag[0].item() == 1
