# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from contextlib import nullcontext
from threading import Condition
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
    P2pNcclConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)

pytestmark = pytest.mark.cpu_test


class ListenerDrained(BaseException):
    pass


@pytest.fixture
def listener(monkeypatch):
    messages: deque[tuple[bytes, bytes]] = deque()
    replies = []
    socket = SimpleNamespace(
        recv_multipart=lambda: messages.popleft(),
        send_multipart=lambda message: replies.append(message),
    )

    def poll():
        if not messages:
            raise ListenerDrained
        return [(socket_id, 1)]

    # The protocol loop uses the socket as a dictionary key.
    class Socket:
        recv_multipart = staticmethod(socket.recv_multipart)
        send_multipart = staticmethod(socket.send_multipart)

    socket_id = Socket()
    engine = P2pNcclEngine.__new__(P2pNcclEngine)
    engine.poller = SimpleNamespace(poll=poll)
    engine.router_socket = socket_id
    engine.comms = {"known": (object(), 0)}
    engine.device = torch.device("cpu")
    engine.recv_stream = None
    engine.send_stream = None
    engine.recv_store_cv = Condition()
    engine.send_store_cv = Condition()
    engine.recv_store = {}
    engine.send_store = {}
    engine.recv_request_id_to_tensor_ids = {}
    engine.send_request_id_to_tensor_ids = {}
    engine.buffer_size = 0
    engine.buffer_size_threshold = 1024
    engine.zmq_address = "local"
    engine.rank = 0
    engine.send_type = "PUT"
    engine.pool = SimpleNamespace(store_tensor=Mock(return_value=123), free=Mock())
    engine.recv = Mock()
    engine.send = Mock()
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    def run(*commands):
        for address, command in commands:
            messages.append((address, msgpack.dumps(command)))
        with pytest.raises(ListenerDrained):
            engine.listen_for_requests()
        return replies

    return engine, run


def put(tensor_id):
    return {"cmd": "PUT", "tensor_id": tensor_id, "shape": [4], "dtype": "float32"}


def test_unknown_put_does_not_allocate_and_listener_accepts_next_peer(
    listener, monkeypatch
):
    engine, run = listener
    allocate = Mock(wraps=torch.empty)
    monkeypatch.setattr(torch, "empty", allocate)
    replies = run(
        (b"stale", put("failed#layer")),
        (b"known", put("good#layer")),
    )
    assert replies == [[b"stale", b"1"], [b"known", b"0"]]
    assert allocate.call_count == 1
    assert engine.recv_tensor("failed#layer") is None
    assert "failed" not in engine.recv_request_id_to_tensor_ids
    assert engine.recv_tensor("good#layer").shape == (4,)
    assert engine.recv_request_id_to_tensor_ids == {"good": {"good#layer"}}


def test_unknown_get_is_rejected_without_losing_tensor_or_listener(listener):
    engine, run = listener
    tensor = torch.arange(4)
    engine.send_store["req#layer"] = tensor
    replies = run(
        (b"stale", {"cmd": "GET", "tensor_id": "req#layer"}),
        (b"known", {"cmd": "GET", "tensor_id": "req#layer"}),
    )
    assert msgpack.loads(replies[0][1]) == {"ret": 1}
    assert msgpack.loads(replies[1][1])["ret"] == 0
    assert engine.send_store["req#layer"] is tensor
    assert engine.send.call_count == 1


def test_allocation_failure_sends_one_rejection_without_receiving(
    listener, monkeypatch
):
    engine, run = listener
    monkeypatch.setattr(torch, "empty", Mock(side_effect=torch.cuda.OutOfMemoryError))
    assert run((b"known", put("failed#layer"))) == [[b"known", b"1"]]
    assert engine.recv_tensor("failed#layer") is None
    engine.recv.assert_not_called()
    assert engine.recv_request_id_to_tensor_ids == {}


@pytest.mark.parametrize("error", [ValueError, torch.cuda.OutOfMemoryError])
def test_spill_failure_does_not_send_an_extra_ack_or_mark_success(listener, error):
    engine, run = listener
    engine.buffer_size_threshold = 0
    engine.pool.store_tensor.side_effect = error
    assert run((b"known", put("failed#layer"))) == [[b"known", b"0"]]
    assert engine.recv_tensor("failed#layer") is None
    assert engine.recv.call_count == 1
    assert engine.recv_request_id_to_tensor_ids == {}
    assert engine.buffer_size == 0


@pytest.mark.parametrize("layers", [{}, {"layer": object()}])
def test_finished_producer_cleans_normalized_bookkeeping_without_recv_data(
    listener, layers
):
    engine, _ = listener
    engine.send_request_id_to_tensor_ids = {"req": {"req#layer"}, "other": set()}
    engine.recv_request_id_to_tensor_ids = {"req": {"req#layer"}}
    engine.get_finished({"req-deadbeef"}, layers)
    assert engine.send_request_id_to_tensor_ids == {"other": set()}
    assert engine.recv_request_id_to_tensor_ids == {}


@pytest.mark.parametrize("missing", [True, False])
def test_load_errors_reach_scheduler_without_poisoning_another_request(missing):
    connector = P2pNcclConnector.__new__(P2pNcclConnector)
    connector.is_producer = False
    connector._rank = 0
    connector._invalid_block_ids = set()
    metadata = P2pNcclConnectorMetadata()
    for name, blocks in [("first", [1, 3]), ("second", [4, 6])]:
        metadata.requests.append(
            ReqMeta(
                name + "___prefill_addr_127.0.0.1:19000___-deadbeef",
                torch.tensor(blocks),
                16,
            )
        )
    good = torch.ones(2, 2)
    connector.p2p_nccl_engine = SimpleNamespace(
        recv_tensor=Mock(side_effect=[None if missing else good, 2 * good])
    )
    connector._get_connector_metadata = lambda: metadata
    cache = torch.zeros(8, 2)
    context = SimpleNamespace(
        attn_metadata=object(),
        no_compile_layers={"layer": SimpleNamespace(kv_cache=cache)},
    )
    connector.start_load_kv(context)
    torch.testing.assert_close(cache[[4, 6]], 2 * good)
    torch.testing.assert_close(
        cache[[1, 3]], torch.zeros_like(good) if missing else good
    )
    assert connector.get_block_ids_with_load_errors() == ({1, 3} if missing else set())
    assert connector.get_block_ids_with_load_errors() == set()


def test_missing_prefix_cannot_feed_uninitialized_nan_kv_into_forward():
    connector = P2pNcclConnector.__new__(P2pNcclConnector)
    connector.is_producer = False
    connector._rank = 0
    connector._invalid_block_ids = set()
    metadata = P2pNcclConnectorMetadata()
    metadata.requests.append(
        ReqMeta("req___prefill_addr_127.0.0.1:19000___", torch.tensor([1, 3]), 16)
    )
    connector._get_connector_metadata = lambda: metadata
    connector.p2p_nccl_engine = SimpleNamespace(recv_tensor=Mock(return_value=None))
    caches = [torch.full((5, 2), float("nan")) for _ in range(2)]
    context = SimpleNamespace(
        attn_metadata=object(),
        no_compile_layers={
            "first": SimpleNamespace(kv_cache=caches[0]),
            "non_attention": object(),
            "later": SimpleNamespace(kv_cache=caches[1]),
        },
    )
    connector.start_load_kv(context)
    for cache in caches:
        assert torch.isfinite(cache[[1, 3]]).all()
        torch.testing.assert_close(cache[[1, 3]], torch.zeros(2, 2))
        assert torch.isnan(cache[[0, 2, 4]]).all()
    assert connector.get_block_ids_with_load_errors() == {1, 3}
    assert connector.p2p_nccl_engine.recv_tensor.call_count == 1


@pytest.mark.parametrize("chunked", [True, False])
def test_producer_exports_prompt_once_and_skips_later_decode(chunked):
    connector = P2pNcclConnector.__new__(P2pNcclConnector)
    connector.is_producer = True
    connector._block_size = 16
    connector._requests_need_load = {}
    connector.chunked_prefill = {}
    cached = SimpleNamespace(
        req_ids=[], num_computed_tokens=[], new_block_ids=[], resumed_req_ids=set()
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="req-deadbeef",
                num_computed_tokens=0,
                prompt_token_ids=list(range(32)),
                block_ids=([1] if chunked else [1, 2],),
            )
        ],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req-deadbeef": 16 if chunked else 32},
    )
    first = connector.build_connector_meta(output)
    assert len(first.requests) == (0 if chunked else 1)
    output.scheduled_new_reqs = []
    cached.req_ids = ["req-deadbeef"]
    if chunked:
        cached.num_computed_tokens = [16]
        cached.new_block_ids = [([2],)]
        output.num_scheduled_tokens = {"req-deadbeef": 16}
        final = connector.build_connector_meta(output)
        assert len(final.requests) == 1
        assert final.requests[0].block_ids.tolist() == [1, 2]
    else:
        assert first.requests[0].block_ids.tolist() == [1, 2]
    cached.num_computed_tokens = [32]
    cached.new_block_ids = [None]
    output.num_scheduled_tokens = {"req-deadbeef": 1}
    assert connector.build_connector_meta(output).requests == []
    assert connector.chunked_prefill == {}


@pytest.mark.parametrize("host", ["127.0.0.1", "worker.example", "[::1]"])
@pytest.mark.parametrize("suffix", ["", "-deadbeef"])
@pytest.mark.parametrize("tail", ["___", "_0123456789abcdef0123456789abcdef", ""])
def test_proxy_request_id_keeps_prefill_and_decode_addresses_separate(
    host, suffix, tail
):
    request_id = (
        f"req___prefill_addr_{host}:14590___decode_addr_{host}:14591{tail}{suffix}"
    )
    assert P2pNcclConnector.parse_request_id(request_id, True) == (host, 14591)
    assert P2pNcclConnector.parse_request_id(request_id, False) == (host, 14590)
