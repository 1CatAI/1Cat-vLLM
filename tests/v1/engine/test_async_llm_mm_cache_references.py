# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How AsyncLLM.add_request takes and gives back the shm processor-cache
references of a request (CPU only, the engine is mocked)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm import SamplingParams
from vllm.multimodal.cache import ShmCacheReference
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM, MMCacheReferenceLedger
from vllm.v1.engine.core_client import _is_for_output_queue


def _references(n: int) -> list[ShmCacheReference]:
    return [
        ShmCacheReference(i, i, mm_hash=f"image_{i}", generation=0) for i in range(n)
    ]


def _request(references: list[ShmCacheReference], n: int = 1) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="req",
        external_req_id="ext_req",
        prompt_token_ids=[1, 2, 3],
        mm_features=[
            MultiModalFeatureSpec(
                data=reference,
                modality="image",
                identifier=reference.mm_hash,
                mm_position=PlaceholderRange(offset=0, length=1),
                mm_hash=reference.mm_hash,
            )
            for reference in references
        ],
        sampling_params=SamplingParams(n=n, output_kind=RequestOutputKind.DELTA),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _engine_input(references: list[ShmCacheReference]) -> dict:
    return {
        "type": "multimodal",
        "prompt_token_ids": [1, 2, 3],
        "mm_kwargs": {"image": references},
        "mm_hashes": {"image": [r.mm_hash for r in references]},
        "mm_placeholders": {"image": []},
    }


def _async_llm() -> AsyncLLM:
    llm = object.__new__(AsyncLLM)
    llm.engine_core = MagicMock()
    llm.engine_core.resources.engine_dead = False
    llm.engine_core.add_request_async = AsyncMock()
    llm.output_handler = None
    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MagicMock()
    llm.input_processor = MagicMock()
    llm.output_processor = MagicMock()
    llm.renderer = MagicMock()
    llm.renderer.get_mm_items.side_effect = lambda engine_input: list(
        engine_input["mm_kwargs"]["image"]
    )
    llm.renderer.add_mm_cache_references_async.side_effect = _count_references
    llm.log_requests = False
    llm._mm_shadow_hits = {}
    llm._mm_cache_references = MMCacheReferenceLedger()
    llm._run_output_handler = MagicMock()
    llm.get_supported_tasks = AsyncMock(return_value=("generate",))
    return llm


def _done(result: object) -> asyncio.Future:
    future = asyncio.get_running_loop().create_future()
    future.set_result(result)
    return future


def _count_references(
    references: list[ShmCacheReference], sink: list[ShmCacheReference]
) -> asyncio.Future:
    """Stands in for the renderer thread counting one more reference each."""
    for reference in references:
        added = ShmCacheReference(
            reference["address"].data,
            reference["monotonic_id"].data,
            mm_hash=reference.mm_hash,
            generation=reference.generation,
        )
        added.taken = True
        sink.append(added)
    return _done(True)


def _added(llm: AsyncLLM) -> list[ShmCacheReference]:
    return [
        reference
        for call in llm.renderer.add_mm_cache_references_async.call_args_list
        for reference in call.args[1]
    ]


def _released(llm: AsyncLLM) -> list[ShmCacheReference]:
    return [
        reference
        for call in llm.renderer.release_mm_cache_references_async.call_args_list
        for reference in call.args[0]
    ]


def _ids(references: list[ShmCacheReference]) -> list[int]:
    """References to one item compare equal; tell them apart by identity."""
    return [id(reference) for reference in references]


@pytest.mark.asyncio
async def test_sent_request_holds_its_references():
    llm = _async_llm()
    references = _references(2)

    await llm.add_request("req", _request(references), SamplingParams())

    assert all(reference.taken for reference in references)
    assert _released(llm) == []
    # Given back only once EngineCore reports it never reached the workers.
    assert llm._mm_cache_references.finish("req", sent_to_workers=False) == (references)


@pytest.mark.asyncio
async def test_failure_before_the_send_gives_back_the_references():
    llm = _async_llm()
    references = _references(2)
    llm.input_processor.process_inputs.side_effect = ValueError("bad request")

    with pytest.raises(ValueError):
        await llm.add_request("req", _engine_input(references), SamplingParams())

    assert _released(llm) == references
    assert not llm._mm_cache_references


@pytest.mark.asyncio
async def test_failed_send_gives_back_the_references():
    llm = _async_llm()
    references = _references(1)
    llm.engine_core.add_request_async.side_effect = RuntimeError("socket closed")

    with pytest.raises(RuntimeError):
        await llm.add_request("req", _request(references), SamplingParams())

    assert _released(llm) == references
    assert not llm._mm_cache_references


@pytest.mark.asyncio
async def test_cancelled_send_leaves_the_references_to_the_engine_report():
    """The ADD may already be on its way: releasing here as well as on the
    workers' reads would free the item while it is still needed."""
    llm = _async_llm()
    references = _references(1)
    llm.engine_core.add_request_async.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await llm.add_request("req", _request(references), SamplingParams())

    assert _released(llm) == []
    assert llm._mm_cache_references.finish("req", sent_to_workers=True) == []


@pytest.mark.asyncio
async def test_request_resending_taken_references_counts_its_own():
    """Beam search re-sends the rendered items every step; only the first
    request that carries a reference owns it, every other one counts one of
    its own, since the workers fetch the item once per request."""
    llm = _async_llm()
    references = _references(1)
    await llm.add_request("req", _request(references), SamplingParams())
    llm.renderer.add_mm_cache_references_async.assert_not_called()

    second = _request(references)
    second.request_id = "req2"
    await llm.add_request("req2", second, SamplingParams())

    (call,) = llm.renderer.add_mm_cache_references_async.call_args_list
    assert _ids(call.args[0]) == _ids(references)
    added = _added(llm)
    assert len(added) == 1
    assert added[0] is not references[0]
    assert added[0].mm_hash == references[0].mm_hash
    ledger = llm._mm_cache_references
    assert _ids(ledger.finish("req2", sent_to_workers=False)) == _ids(added)
    assert _ids(ledger.finish("req", sent_to_workers=False)) == _ids(references)


@pytest.mark.asyncio
async def test_failed_count_gives_back_everything_and_sends_nothing():
    """An item that is no longer cached (reset, or the request that sent it
    has finished) must not be sent by address: its slot may hold another
    object by now."""
    llm = _async_llm()
    references = _references(2)
    counted: list[ShmCacheReference] = []

    def count_one(to_add, sink):
        _count_references(to_add[:1], sink)
        counted.extend(sink)
        return _done(False)

    llm.renderer.add_mm_cache_references_async.side_effect = count_one

    with pytest.raises(RuntimeError, match="no longer in the shm processor cache"):
        await llm.add_request("req", _request(references, n=2), SamplingParams(n=2))

    llm.engine_core.add_request_async.assert_not_called()
    assert len(counted) == 1
    assert sorted(_ids(_released(llm))) == sorted(_ids(references + counted))
    assert not llm._mm_cache_references


@pytest.mark.asyncio
async def test_cancelled_count_gives_back_what_it_counts_later():
    """The count runs on the renderer thread and may finish after the caller
    was cancelled; the release is submitted behind it, with the same list."""
    llm = _async_llm()
    references = _references(1)
    pending = asyncio.get_running_loop().create_future()
    sinks: list[list[ShmCacheReference]] = []

    def count_later(to_add, sink):
        sinks.append(sink)
        return pending

    llm.renderer.add_mm_cache_references_async.side_effect = count_later

    task = asyncio.create_task(
        llm.add_request("req", _request(references, n=2), SamplingParams(n=2))
    )
    for _ in range(100):
        if sinks or task.done():
            break
        await asyncio.sleep(0)
    assert sinks, "add_request did not count references for the second child"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The renderer thread finishes the count after the cancellation.
    _count_references(references, sinks[0])
    released_lists = [
        call.args[0]
        for call in llm.renderer.release_mm_cache_references_async.call_args_list
    ]
    assert any(released is sinks[0] for released in released_lists)
    assert sorted(_ids(_released(llm))) == sorted(_ids(references + sinks[0]))
    llm.engine_core.add_request_async.assert_not_called()
    assert not llm._mm_cache_references


@pytest.mark.asyncio
async def test_reused_request_id_is_not_held_twice():
    """A report for the first request must not give back the second's."""
    llm = _async_llm()
    first, second = _references(1), _references(2)[1:]
    await llm.add_request("req", _request(first), SamplingParams())
    await llm.add_request("req", _request(second), SamplingParams())

    assert llm._mm_cache_references.finish("req", sent_to_workers=False) == first
    assert llm._mm_cache_references.finish("req", sent_to_workers=False) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_step", ["output_processor", "send"])
async def test_reused_request_id_that_fails_leaves_the_first_alone(failing_step):
    """A second request under an id still in the ledger is not registered; if
    it fails before the send, it must not finish the first request's holder,
    whose references the workers may still read."""
    llm = _async_llm()
    first, second = _references(1), _references(2)[1:]
    await llm.add_request("req", _request(first), SamplingParams())

    if failing_step == "output_processor":
        llm.output_processor.add_request.side_effect = ValueError("duplicate")
    else:
        llm.engine_core.add_request_async.side_effect = RuntimeError("closed")
    with pytest.raises((ValueError, RuntimeError)):
        await llm.add_request("req", _request(second), SamplingParams())

    assert _released(llm) == []
    assert llm._mm_cache_references.finish("req", sent_to_workers=True) == []
    assert not llm._mm_cache_references


@pytest.mark.asyncio
async def test_parallel_sampling_children_hold_references_of_their_own():
    """Each child reaches the workers as a request of its own, and each
    worker fetches the item once per child. Every child holds one reference,
    and only a child that never reached the workers gives its own back."""
    llm = _async_llm()
    references = _references(1)

    await llm.add_request("req", _request(references, n=3), SamplingParams(n=3))

    sent = [
        call.args[0].request_id
        for call in llm.engine_core.add_request_async.call_args_list
    ]
    assert len(sent) == 3
    (call,) = llm.renderer.add_mm_cache_references_async.call_args_list
    assert _ids(call.args[0]) == _ids(references * 2)
    second_child, third_child = _added(llm)
    assert second_child is not third_child
    ledger = llm._mm_cache_references
    assert ledger.finish(sent[1], sent_to_workers=True) == []
    assert _ids(ledger.finish(sent[0], sent_to_workers=False)) == _ids(references)
    assert _ids(ledger.finish(sent[2], sent_to_workers=False)) == [id(third_child)]
    assert not ledger


@pytest.mark.asyncio
async def test_parallel_sampling_child_that_fails_to_send_gives_back_the_rest():
    llm = _async_llm()
    references = _references(1)
    llm.engine_core.add_request_async.side_effect = [None, RuntimeError("closed")]

    with pytest.raises(RuntimeError):
        await llm.add_request("req", _request(references, n=3), SamplingParams(n=3))

    # The first child was sent; the second failed, the third never went.
    assert len(_added(llm)) == 2
    assert _ids(_released(llm)) == _ids(_added(llm))
    sent = llm.engine_core.add_request_async.call_args_list[0].args[0].request_id
    ledger = llm._mm_cache_references
    assert _ids(ledger.finish(sent, sent_to_workers=False)) == _ids(references)
    assert not ledger


def test_core_client_passes_on_outputs_that_only_report_mm_requests():
    assert not _is_for_output_queue(EngineCoreOutputs())
    assert _is_for_output_queue(EngineCoreOutputs(finished_mm_requests={"req": False}))
