# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch

import vllm.v1.ple_offload.connector as connector_module
from vllm.model_executor.layers import ple_offload_layer
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.v1.ple_offload.connector import PleOffloadConnector
from vllm.v1.ple_offload.protocol import PleCpuInputs, PleOffloadRequest
from vllm.v1.ple_offload.worker import (
    PleOffloadInputBuffers,
    PleOffloadOutputTarget,
    PleOffloadRunner,
)

EOS = 99


def _model_state(tokens: dict[int, np.ndarray]) -> Qwen4ExpModelState:
    state = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    state._ple_early_tokens = tokens
    state.ngram_context_len = 2
    state.ngram_eos_token_id = EOS
    state.ngram_context = torch.full((4, 2), EOS, dtype=torch.int32)
    state.ngram_context_offsets = torch.arange(-2, 0, dtype=torch.int64)
    return state


def _batch(**overrides):
    batch = dict(
        num_reqs=2,
        num_reqs_after_padding=2,
        num_draft_tokens=0,
        num_tokens=8,
        num_tokens_after_padding=8,
        idx_mapping_np=np.array([3, 1], dtype=np.int32),
        idx_mapping=torch.tensor([3, 1], dtype=torch.int32),
        num_computed_prefill_tokens_np=np.array([10, 0], dtype=np.int32),
        num_scheduled_tokens=np.array([5, 3], dtype=np.int32),
        prefill_len_np=np.array([30, 5], dtype=np.int32),
        query_start_loc_np=np.array([0, 5, 8], dtype=np.int32),
    )
    batch.update(overrides)
    return SimpleNamespace(**batch)


TOKENS = {
    3: np.arange(100, 130, dtype=np.int32),
    1: np.arange(200, 205, dtype=np.int32),
}


def test_cpu_inputs_follow_device_prefill_rules():
    state = _model_state(TOKENS)
    batch = _batch()
    inputs = state.ple_cpu_inputs(batch)
    assert inputs is not None
    assert inputs.input_ids.tolist() == [110, 111, 112, 113, 114, 200, 201, 202]
    assert inputs.query_start_loc.tolist() == [0, 5, 8]

    # The device path reads the same history from all_token_ids.
    all_token_ids = torch.full((4, 30), -1, dtype=torch.int32)
    for index, tokens in TOKENS.items():
        all_token_ids[index, : tokens.size] = torch.from_numpy(tokens)
    num_computed = torch.zeros(4, dtype=torch.int32)
    num_computed[3], num_computed[1] = 10, 0
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=num_computed),
        all_token_ids=SimpleNamespace(gpu=all_token_ids),
    )
    device_context = state._prepare_ngram_context(batch, req_states)
    assert inputs.ngram_context is not None
    assert inputs.ngram_context.tolist() == device_context[:2].tolist()
    assert inputs.ngram_context.tolist() == [[108, 109], [EOS, EOS]]


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_draft_tokens": 4},
        {"num_tokens_after_padding": 16},
        {"num_scheduled_tokens": np.array([25, 3], dtype=np.int32)},
        {"idx_mapping_np": np.array([3, 2], dtype=np.int32)},
        {"num_reqs": 0},
    ],
)
def test_cpu_inputs_admit_only_prompt_only_unpadded_batches(overrides):
    assert _model_state(TOKENS).ple_cpu_inputs(_batch(**overrides)) is None


def test_cpu_inputs_disabled_without_the_kernel_config_field():
    assert _model_state(None).ple_cpu_inputs(_batch()) is None  # type: ignore[arg-type]


def test_launch_queues_host_inputs_without_a_d2h_event():
    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector.tp_rank = 0
    connector.dp_rank = 0
    connector.fast_lane = None
    connector._uses_cuda_inputs = True
    connector._next_seq = 0
    connector._request_queue = connector_module.queue.Queue(maxsize=2)
    connector._d2h_event_pool = connector_module.queue.Queue(maxsize=1)
    inputs = PleCpuInputs(
        np.zeros(3, dtype=np.int32), np.array([0, 3], dtype=np.int32), None
    )

    connector._launch(num_reqs=1, num_tokens=3, cpu_inputs=inputs)
    connector._launch(num_reqs=1, num_tokens=3, cpu_inputs=inputs)

    first = connector._request_queue.get_nowait()
    second = connector._request_queue.get_nowait()
    assert first.cpu_inputs is inputs and first.d2h_done_event is None
    assert (first.request.seq, second.request.seq) == (1, 2)
    assert connector._d2h_event_pool.empty()


def test_early_request_waits_for_the_previous_request(monkeypatch):
    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector._uses_cuda_inputs = True
    connector._request_done = np.zeros(1, dtype=np.int64)
    connector._input_ids_buf = torch.zeros(8, dtype=torch.int32)
    connector._query_start_loc_buf = torch.zeros(3, dtype=torch.int32)
    connector._ngram_context_buf = torch.zeros(2, 2, dtype=torch.int32)
    monkeypatch.setattr(
        connector_module.torch.cuda.nvtx, "range", lambda *_: nullcontext()
    )
    request = PleOffloadRequest(dp_rank=0, num_tokens=3, num_reqs=1, seq=2)
    inputs = PleCpuInputs(
        np.array([5, 6, 7], dtype=np.int32),
        np.array([0, 3], dtype=np.int32),
        np.array([[1, 2]], dtype=np.int32),
    )
    sent: list[bytes] = []
    socket = SimpleNamespace(send=sent.append)
    pending = connector_module._PendingPleOffloadRequest(request, None, inputs)

    worker = threading.Thread(target=connector._process_request, args=(pending, socket))
    worker.start()
    time.sleep(0.05)
    # The worker still reads request 1 from the shared buffers.
    assert sent == []
    assert connector._input_ids_buf.tolist() == [0] * 8
    connector._request_done[0] = 1
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert sent == [msgspec.msgpack.encode(request)]
    assert connector._input_ids_buf[:3].tolist() == [5, 6, 7]
    assert connector._query_start_loc_buf.tolist() == [0, 3, 0]
    assert connector._ngram_context_buf[0].tolist() == [1, 2]


def test_release_skips_results_the_forward_already_released(monkeypatch):
    released, kept = torch.zeros(1), torch.zeros(1)
    resets = []

    def layer(flag):
        return SimpleNamespace(
            _is_cpu_offloaded=True,
            _sem=SimpleNamespace(flag_tensor=flag),
            release_offloaded_output=lambda stream: resets.append(flag),
        )

    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector.device = torch.device("cpu")
    connector._layers = {"a": layer(released), "b": layer(kept)}
    monkeypatch.setattr(
        connector_module.torch.cuda, "current_stream", lambda device: None
    )
    monkeypatch.setattr(
        connector_module,
        "finish_early_result_release",
        lambda: {released.data_ptr()},
    )

    connector.release_outputs()

    assert resets == [kept]


def test_worker_publishes_done_after_the_last_layer():
    done = torch.zeros(1, dtype=torch.int64)
    seen = []

    class Layer:
        def forward_impl(self, hidden, ids, offsets, context, output_buffer):
            seen.append(int(done[0]))
            output_buffer[:3].copy_(ids[:, None].expand(3, 2))
            return output_buffer[:3]

    names = ("first", "second")
    buffers = {name: torch.full((4, 2), -1, dtype=torch.int32) for name in names}
    runner = PleOffloadRunner.__new__(PleOffloadRunner)
    runner._clamp_input_ids = False
    runner._layers = {name: Layer() for name in names}
    runner._worker_targets = {
        0: {
            name: [
                PleOffloadOutputTarget(
                    tp_rank=0,
                    gpu_output_buffer=None,
                    sem=SimpleNamespace(flag_tensor=torch.zeros(16, dtype=torch.int32)),
                    copy_stream=None,
                    cpu_output_buffer=buffers[name],
                )
            ]
            for name in names
        }
    }
    runner._input_bufs = {
        0: PleOffloadInputBuffers(
            input_ids_buf=torch.tensor([11, 12, 13], dtype=torch.int32),
            query_start_loc_buf=torch.tensor([0, 3], dtype=torch.int32),
            ngram_context_buf=torch.tensor([[9, 10]], dtype=torch.int32),
            request_done_buf=done,
        )
    }
    runner._pinned_bufs = {0: buffers}

    runner._handle_requests([PleOffloadRequest(0, 3, 1, seq=7)])

    assert seen == [0, 0]
    assert int(done[0]) == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("armed", [False, True])
def test_wait_op_releases_mapped_result_only_when_armed(armed):
    flag = torch.ones(1, dtype=torch.int32, device="cuda")
    gpu_output = torch.zeros(4, 2, dtype=torch.float16, device="cuda")
    cpu_output = torch.arange(8, dtype=torch.float16).reshape(4, 2).pin_memory()
    hidden = torch.zeros(1, device="cuda")

    ple_offload_layer.begin_early_result_release(armed)
    torch.ops.vllm.ple_offload_wait(flag, gpu_output, hidden, cpu_output, 3)
    torch.accelerator.synchronize()
    released = ple_offload_layer.finish_early_result_release()

    torch.testing.assert_close(gpu_output[:3].cpu(), cpu_output[:3])
    assert flag.item() == (0 if armed else 1)
    assert released == ({flag.data_ptr()} if armed else set())
