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
    llm.log_requests = False
    llm._mm_shadow_hits = {}
    llm._mm_cache_references = MMCacheReferenceLedger()
    llm._run_output_handler = MagicMock()
    llm.get_supported_tasks = AsyncMock(return_value=("generate",))
    return llm


def _released(llm: AsyncLLM) -> list[ShmCacheReference]:
    return [
        reference
        for call in llm.renderer.release_mm_cache_references_async.call_args_list
        for reference in call.args[0]
    ]


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
async def test_references_taken_once_are_not_taken_again():
    """Beam search re-sends the rendered items every step; only the first
    request that carries a reference owns it."""
    llm = _async_llm()
    references = _references(1)
    await llm.add_request("req", _request(references), SamplingParams())

    second = _request(references)
    second.request_id = "req2"
    await llm.add_request("req2", second, SamplingParams())

    assert llm._mm_cache_references.finish("req2", sent_to_workers=False) == []
    assert llm._mm_cache_references.finish("req", sent_to_workers=False) == (references)


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
async def test_parallel_sampling_children_share_the_references():
    llm = _async_llm()
    references = _references(1)

    await llm.add_request("req", _request(references, n=2), SamplingParams(n=2))

    sent = [
        call.args[0].request_id
        for call in llm.engine_core.add_request_async.call_args_list
    ]
    assert len(sent) == 2
    assert llm._mm_cache_references.finish(sent[0], sent_to_workers=False) == []
    assert llm._mm_cache_references.finish(sent[1], sent_to_workers=False) == (
        references
    )


def test_core_client_passes_on_outputs_that_only_report_mm_requests():
    assert not _is_for_output_queue(EngineCoreOutputs())
    assert _is_for_output_queue(EngineCoreOutputs(finished_mm_requests={"req": False}))
