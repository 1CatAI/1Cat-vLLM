# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the actual V2 decode transport implementation."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models.pp_decode_transfer import (
    StaticPPDecodeTransfer,
    _ReceiveSlot,
    is_complete_mtp_decode,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput

pytestmark = pytest.mark.cpu_test


def decode_output(n=2, query_len=5):
    output = SchedulerOutput.make_empty()
    ids = [f"r{i}" for i in range(n)]
    output.num_scheduled_tokens = dict.fromkeys(ids, query_len)
    output.total_num_scheduled_tokens = n * query_len
    output.scheduled_spec_decode_tokens = dict.fromkeys(ids, [-1] * (query_len - 1))
    cached = output.scheduled_cached_reqs
    cached.req_ids = ids
    cached.num_output_tokens = [1] * n
    return output


@pytest.mark.parametrize("requests", [1, 2, 4, 8])
@pytest.mark.parametrize("query_len", [2, 3, 4, 5])
def test_complete_decode_uses_common_scheduler_contract(requests, query_len):
    assert is_complete_mtp_decode(decode_output(requests, query_len), query_len, 40)


@pytest.mark.parametrize(
    "case",
    ["empty", "new", "encoder", "resume", "mixed", "budget", "prefill", "no_spec"],
)
def test_dynamic_fallback_for_prefill_mixed_and_unsupported_shapes(case):
    output = decode_output()
    capacity = 40
    if case == "empty":
        output = SchedulerOutput.make_empty()
    elif case == "new":
        output.scheduled_new_reqs = [object()]
    elif case == "encoder":
        output.scheduled_encoder_inputs = {"r0": [0]}
    elif case == "resume":
        output.scheduled_cached_reqs.resumed_req_ids = {"r0"}
    elif case == "mixed":
        output.num_scheduled_tokens["r0"] = 4
        output.total_num_scheduled_tokens = 9
    elif case == "budget":
        capacity = 9
    elif case == "prefill":
        output.scheduled_cached_reqs.num_output_tokens[0] = 0
    elif case == "no_spec":
        output.scheduled_spec_decode_tokens.clear()
    assert not is_complete_mtp_decode(output, 5, capacity)


def make_config():
    return SimpleNamespace(
        additional_config={"pp_decode_static_transfer": True},
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=7,
            tensor_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
            ubatch_size=0,
        ),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            hf_text_config=SimpleNamespace(
                model_type="qwen4_exp_text",
                hidden_size=8,
                hc_count=4,
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False),
            cudagraph_capture_sizes=[5, 40],
        ),
        max_concurrent_batches=4,
    )


def make_runner():
    return SimpleNamespace(
        device=torch.device("cuda:0"),
        max_num_tokens=1024,
        model=SimpleNamespace(
            make_empty_intermediate_tensors=lambda **kw: IntermediateTensors(
                {
                    "hidden_states": torch.empty(
                        (kw["batch_size"], 32),
                        dtype=kw["dtype"],
                        device=kw["device"],
                    )
                }
            )
        ),
    )


def fake_agreement(monkeypatch, change=None):
    reports_seen = []

    def gather(reports, local, *, group):
        reports_seen.append(local)
        reports[:] = [local] * len(reports)
        if change is not None:
            reports[1] = change(local)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    return reports_seen


def pp_first():
    return SimpleNamespace(world_size=7, is_first_rank=True, cpu_group=object())


def test_schema_agreement_and_graph_padding_capacity(monkeypatch):
    seen = fake_agreement(monkeypatch)
    config = make_config()
    config.compilation_config.cudagraph_capture_sizes = [5, 64]
    transfer = StaticPPDecodeTransfer.create(
        config, make_runner(), pp_first(), use_v2=True
    )
    assert transfer.capacity == 64
    assert transfer.shape == (32,)
    assert len(seen) == 1
    assert transfer.slots == []
    assert transfer.matches(decode_output())
    assert (transfer.static_steps, transfer.dynamic_steps) == (1, 0)
    assert not torch.cuda.is_initialized()


def test_rank_local_schema_failure_is_shared_before_raising(monkeypatch):
    seen = fake_agreement(monkeypatch)
    runner = make_runner()
    runner.model.make_empty_intermediate_tensors = Mock(
        side_effect=ValueError("bad schema")
    )
    with pytest.raises(RuntimeError, match="agreement failed"):
        StaticPPDecodeTransfer.create(make_config(), runner, pp_first(), use_v2=True)
    assert len(seen) == 1 and "bad schema" in seen[0][1]


def test_remote_schema_mismatch_fails_before_allocating(monkeypatch):
    fake_agreement(monkeypatch, lambda local: (((8, 4), *local[0][1:]), None))
    with pytest.raises(RuntimeError, match="agreement failed"):
        StaticPPDecodeTransfer.create(
            make_config(), make_runner(), pp_first(), use_v2=True
        )


@pytest.mark.parametrize(
    "field,value",
    [("tensor_parallel_size", 2), ("data_parallel_size", 2), ("enable_dbo", True)],
)
def test_unsupported_global_config_rejected_without_collective(
    monkeypatch, field, value
):
    gather = Mock()
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    config = make_config()
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError, match="requires V2"):
        StaticPPDecodeTransfer.create(config, make_runner(), pp_first(), use_v2=True)
    gather.assert_not_called()


def test_disabled_does_not_inspect_rank_local_state():
    assert (
        StaticPPDecodeTransfer.create(SimpleNamespace(), None, None, use_v2=False)
        is None
    )


@pytest.mark.parametrize("value", [0, 1, None, "true", "false"])
def test_enable_option_requires_an_explicit_boolean(value):
    config = SimpleNamespace(additional_config={"pp_decode_static_transfer": value})
    with pytest.raises(ValueError, match="must be a boolean"):
        StaticPPDecodeTransfer.create(config, None, None, use_v2=False)


def test_receive_pool_waits_for_initial_allocation_stream(monkeypatch):
    fake_agreement(monkeypatch)
    runner = make_runner()
    runner.intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": SimpleNamespace(
                shape=(1024, 32),
                dtype=torch.float16,
                device=runner.device,
                is_contiguous=lambda: True,
            ),
        }
    )
    pp = pp_first()
    pp.is_first_rank = False
    operations = []
    allocator_stream = object()
    receiver = SimpleNamespace(
        wait_stream=lambda stream: operations.append(("wait", stream))
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: receiver)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: allocator_stream)
    original_empty = torch.empty

    def empty(*args, **kwargs):
        if kwargs.get("device") == runner.device:
            operations.append(("allocate", None))
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    transfer = StaticPPDecodeTransfer.create(make_config(), runner, pp, use_v2=True)
    assert len(transfer.slots) == 4
    assert operations == [("allocate", None)] * 4 + [("wait", allocator_stream)]
    assert not torch.cuda.is_initialized()


def cpu_transfer(depth=2):
    transfer = StaticPPDecodeTransfer.__new__(StaticPPDecodeTransfer)
    transfer.device = torch.device("cpu")
    transfer.shape = (4,)
    transfer.capacity = 20
    transfer.depth = depth
    transfer.slots = [
        _ReceiveSlot({"hidden_states": torch.full((20, 4), 99.0, dtype=torch.float16)})
        for _ in range(depth)
    ]
    transfer.slot_index = 0
    transfer.active_slot = None
    return transfer


def test_real_rows_only_on_wire_and_sender_schema_guard():
    transfer = cpu_transfer()
    sent: list[dict[str, torch.Tensor]] = []

    def send(value):
        sent.append(value)
        return ["work"]

    transfer.pp = SimpleNamespace(isend_tensor_dict_static=send)
    padded = torch.arange(60, dtype=torch.float16).reshape(15, 4)
    assert transfer.send({"hidden_states": padded}, 10) == ["work"]
    assert sent[0]["hidden_states"].shape == (10, 4)
    assert sent[0]["hidden_states"].data_ptr() == padded.data_ptr()
    with pytest.raises(RuntimeError, match="schema"):
        transfer.send({"hidden_states": padded.float()}, 10)
    with pytest.raises(RuntimeError, match="schema"):
        transfer.send({"hidden_states": padded[:, :3]}, 10)
    with pytest.raises(RuntimeError, match="schema"):
        transfer.send({"hidden_states": padded}, 16)


def test_receive_ring_waits_for_consumption_and_zeroes_padding(monkeypatch):
    transfer = cpu_transfer()
    events = []
    waits: list[object] = []
    received = []
    transfer.receive_stream = SimpleNamespace(wait_event=waits.append)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    def record():
        event = object()
        events.append(event)
        return event

    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(record_event=record),
    )

    def recv(wire):
        received.append(wire["hidden_states"])
        wire["hidden_states"].fill_(len(received))
        return ["work"]

    transfer.pp = SimpleNamespace(irecv_tensor_dict_static=recv)
    for n in [10, 15, 5]:
        tensors, handles = transfer.receive(n)
        assert handles == ["work"]
        assert torch.all(tensors["hidden_states"][:n] == len(received))
        assert torch.count_nonzero(tensors["hidden_states"][n:]) == 0
        with pytest.raises(RuntimeError, match="not been consumed"):
            transfer.receive(n)
        transfer.mark_consumed()
    assert waits == [events[0]]
    assert received[0].data_ptr() == received[2].data_ptr()
    assert received[0].data_ptr() != received[1].data_ptr()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("last_rank", [True, False])
def test_worker_routes_both_peers_and_waits_before_output_reuse(
    monkeypatch, static, last_rank
):
    from vllm.v1.worker import gpu_worker

    operations = []
    pp = SimpleNamespace(
        is_first_rank=False,
        is_last_rank=last_rank,
        irecv_tensor_dict=Mock(
            return_value=(
                {"hidden_states": torch.zeros(10, 4)},
                [],
                [],
            )
        ),
        isend_tensor_dict=Mock(return_value=["dynamic_send"]),
    )
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: object())
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker._pp_send_work = [
        SimpleNamespace(wait=lambda: operations.append("old_send_wait"))
    ]
    worker._decode_static_transfer_initialized = True
    worker._decode_static_transfer = SimpleNamespace(
        matches=lambda output: static,
        receive=Mock(
            return_value=(
                {"hidden_states": torch.zeros(20, 4)},
                [SimpleNamespace(wait=lambda: operations.append("recv_wait"))],
            )
        ),
        mark_consumed=Mock(side_effect=lambda: operations.append("consumed")),
        send=Mock(return_value=["static_send"]),
    )
    worker._static_pp_hidden_recv_buffers = lambda count: None
    worker._use_sm70_static_pp_hidden_transfer = lambda count: False
    worker.annotate_profile = lambda output: nullcontext()
    worker.use_v2_model_runner = True
    worker.vllm_config = make_config()
    worker.vllm_config.parallel_config.distributed_executor_backend = "mp"

    def execute(output, intermediates):
        operations.append("model")
        assert intermediates.tensors["hidden_states"].shape[0] == (20 if static else 10)
        operations.append("input_copy")
        if last_rank:
            return None
        return IntermediateTensors({"hidden_states": torch.ones(10, 4)})

    worker.model_runner = SimpleNamespace(execute_model=execute, is_pooling_model=False)
    worker.execute_model(decode_output())
    assert operations[0] == "old_send_wait"
    transfer = worker._decode_static_transfer
    if static:
        assert operations == [
            "old_send_wait",
            "model",
            "recv_wait",
            "input_copy",
            "consumed",
        ]
        transfer.receive.assert_called_once_with(10)
        transfer.mark_consumed.assert_called_once()
        pp.irecv_tensor_dict.assert_not_called()
        pp.isend_tensor_dict.assert_not_called()
        assert transfer.send.call_count == (0 if last_rank else 1)
    else:
        assert operations == ["old_send_wait", "model", "input_copy"]
        transfer.receive.assert_not_called()
        transfer.send.assert_not_called()
        transfer.mark_consumed.assert_not_called()
        pp.irecv_tensor_dict.assert_called_once()
        assert pp.isend_tensor_dict.call_count == (0 if last_rank else 1)


def test_forward_failure_keeps_receive_slot_unavailable(monkeypatch):
    from vllm.v1.worker import gpu_worker

    pp = SimpleNamespace(is_first_rank=False, is_last_rank=True)
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp)
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker._pp_send_work = []
    worker._decode_static_transfer_initialized = True
    transfer = SimpleNamespace(
        matches=lambda output: True,
        receive=lambda count: ({"hidden_states": torch.zeros(20, 4)}, []),
        mark_consumed=Mock(),
    )
    worker._decode_static_transfer = transfer
    worker.annotate_profile = lambda output: nullcontext()
    worker.use_v2_model_runner = True
    worker.vllm_config = make_config()
    worker.model_runner = SimpleNamespace(
        execute_model=Mock(side_effect=RuntimeError("forward failed"))
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        worker.execute_model(decode_output())
    transfer.mark_consumed.assert_not_called()
