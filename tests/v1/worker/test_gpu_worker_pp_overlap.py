# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker


def test_overlapped_send_owns_reusable_eager_output(monkeypatch):
    # Eager execution does not promise fresh storage: a model/provider may
    # return a view of its reusable workspace. Delay communication's read until
    # wait() to exercise that legitimate interleaving deterministically.
    workspace = torch.zeros(9, 4)
    received = []
    step = 0

    def forward(*args):
        nonlocal step
        step += 1
        workspace.fill_(step)
        return IntermediateTensors({"hidden_states": workspace})

    def send(tensors, **kwargs):
        def wait():
            received.append(tensors["hidden_states"].clone())

        return [SimpleNamespace(wait=wait)]

    group = SimpleNamespace(
        is_first_rank=True, is_last_rank=False, isend_tensor_dict=send
    )
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: None)
    worker = SimpleNamespace(
        _pp_send_work=[],
        _pp_send_tensors=None,
        _pp_send_overlaps_next_step=False,
        use_v2_model_runner=False,
        annotate_profile=lambda output: nullcontext(),
        _use_sm70_static_pp_hidden_transfer=lambda num_tokens: False,
        model_runner=SimpleNamespace(execute_model=forward),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                max_cudagraph_capture_size=8,
                pass_config=SimpleNamespace(enable_sp=False),
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, distributed_executor_backend="mp"
            ),
        ),
    )
    worker._wait_pp_sends = MethodType(Worker._wait_pp_sends, worker)
    worker._pp_send_overlaps = MethodType(Worker._pp_send_overlaps, worker)
    scheduled = SimpleNamespace(total_num_scheduled_tokens=9)
    Worker.execute_model(worker, scheduled)
    assert not received
    Worker.execute_model(worker, scheduled)
    assert len(received) == 1
    torch.testing.assert_close(received[0], torch.ones_like(workspace))
    assert worker._pp_send_tensors["hidden_states"].data_ptr() != workspace.data_ptr()
    Worker._wait_pp_sends(worker)
    torch.testing.assert_close(received[1], torch.full_like(workspace, 2))
    assert worker._pp_send_tensors is None
