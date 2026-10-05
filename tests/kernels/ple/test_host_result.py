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


@pytest.mark.parametrize(
    "method,expected",
    [(None, "mapped"), ("mtp", "mapped"), ("eagle", "cuda"), ("dflash", "cuda")],
)
def test_mtp_mapped_result_transport_admission(monkeypatch, method, expected):
    from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer
    from vllm.v1.ple_offload import connector as module

    class Layer(PleOffloadLayer):
        def forward_impl(self, *args, **kwargs):
            raise AssertionError("Layer execution is outside transport setup")

        def get_offload_output_dtype(self, dtype):
            return dtype

        def setup_cross_process_offload(self, output, sem):
            self.output = output

    layer = Layer()
    allocate = torch.empty
    monkeypatch.setattr(
        module.torch, "empty", lambda *a, **kw: allocate(*a, **{**kw, "device": "cpu"})
    )
    monkeypatch.setattr(module.envs, "VLLM_SM70_QWEN38_HYBRID_PLE", False)
    monkeypatch.setattr(
        module.HostResultRegion,
        "create",
        lambda output: SimpleNamespace(
            result=torch.empty_like(output),
            pinned_bytes=output.numel() * output.element_size(),
        ),
    )
    monkeypatch.setattr(module, "CpuGpuSemaphore", lambda *a, **kw: SimpleNamespace())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_embed_dim=7), dtype=torch.float16
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=10),
        kernel_config=SimpleNamespace(
            ple_result_transport="auto", ple_result_transports={}
        ),
        speculative_config=None if method is None else SimpleNamespace(method=method),
    )
    connector = module.PleOffloadConnector.__new__(module.PleOffloadConnector)
    connector.device = torch.device("cpu")
    model = SimpleNamespace(named_modules=lambda: [("ple", layer)])
    connector._setup_layers(config, model)
    transport = config.kernel_config.ple_result_transports["ple"]
    assert transport["mode"] == expected
    assert transport["reason"] == (
        None if expected == "mapped" else "speculative_transport_not_qualified"
    )
