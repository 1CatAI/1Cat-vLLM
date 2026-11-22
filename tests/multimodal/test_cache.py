# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import multiprocessing as mp
from typing import Any

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde,
    SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalCache,
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheItem,
    MultiModalProcessorCacheItemMetadata,
    MultiModalProcessorSenderCache,
    MultiModalReceiverCache,
    ShmObjectStoreReceiverCache,
    ShmObjectStoreSenderCache,
)
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.processing import PromptInsertion
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


def _dummy_elem(
    size: int,
    *,
    rng: np.random.RandomState | None = None,
):
    if rng is None:
        data = torch.empty((size,), dtype=torch.int8)
    else:
        data = torch.from_numpy(rng.randint(4, size=(size,), dtype=np.int8))

    return MultiModalFieldElem(
        data=data,
        field=MultiModalSharedField(batch_size=1),
    )


def _dummy_item(
    size_by_key: dict[str, int],
    *,
    rng: np.random.RandomState | None = None,
):
    return MultiModalKwargsItem(
        {key: _dummy_elem(size, rng=rng) for key, size in size_by_key.items()}
    )


def _dummy_items(
    size_by_key_modality: dict[str, dict[str, int]],
    *,
    rng: np.random.RandomState | None = None,
):
    return MultiModalKwargsItems(
        {
            modality: [_dummy_item(size_by_key, rng=rng)]
            for modality, size_by_key in size_by_key_modality.items()
        }
    )


@pytest.mark.parametrize(
    ("item", "expected_size"),
    [
        (_dummy_item({"a1": 100}), 100),
        (_dummy_item({"a1": 100, "a2": 110}), 210),
        (_dummy_items({"a": {"a1": 100, "a2": 110}, "b": {"b1": 120, "b2": 130}}), 460),  # noqa: E501
    ],
)
def test_cache_item_size(item, expected_size):
    cache = MultiModalCache.get_lru_cache(2048, type(item))

    cache[""] = item
    assert cache.currsize == expected_size

    prompt_update = PromptInsertion("dummy", "target", "insertion").resolve(0)

    cache[""] = MultiModalProcessorCacheItem(item, [prompt_update])
    assert cache.currsize == expected_size

    cache[""] = MultiModalProcessorCacheItemMetadata(item, [prompt_update])
    assert cache.currsize == expected_size

    cache[""] = item.get_data()
    assert cache.currsize == expected_size


def _create_vllm_config(
    *,
    mm_processor_cache_gb: float,
    enable_ipc: bool,
):
    return VllmConfig(
        model_config=ModelConfig(
            model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            mm_processor_cache_gb=mm_processor_cache_gb,
        ),
        parallel_config=ParallelConfig(data_parallel_size=1 if enable_ipc else 2),
    )


def _compare_caches(
    config_0: VllmConfig,
    config_1: VllmConfig,
    *,
    item_capacity: int = 8,
    hit_rate: float = 0.5,
    max_items_per_iter: int = 3,
    is_cached_calls_per_iter: int,
    n_iter: int = 100,
    seed: int = 0,
):
    cache_0_p0 = MULTIMODAL_REGISTRY.processor_cache_from_config(config_0)
    cache_0_p1 = MULTIMODAL_REGISTRY.engine_receiver_cache_from_config(config_0)
    cache_1_p0 = MULTIMODAL_REGISTRY.processor_cache_from_config(config_1)
    cache_1_p1 = MULTIMODAL_REGISTRY.engine_receiver_cache_from_config(config_1)

    cache_size_gb = max(
        config_0.model_config.multimodal_config.mm_processor_cache_gb,
        config_1.model_config.multimodal_config.mm_processor_cache_gb,
    )
    item_size_gb = int(cache_size_gb / item_capacity)

    rng = np.random.RandomState(seed)
    all_items = [
        _dummy_item({"key": item_size_gb}, rng=rng)
        for _ in range(int(item_capacity / hit_rate))
    ]
    all_hashes = [
        MultiModalHasher.hash_kwargs(item=item.get_data()) for item in all_items
    ]

    prompt_update = PromptInsertion("dummy", "target", "insertion").resolve(0)

    for it in range(n_iter):
        num_items_to_select = rng.randint(0, max_items_per_iter)
        item_idxs_to_select = rng.choice(len(all_items), num_items_to_select)

        selected_items = [all_items[idx] for idx in item_idxs_to_select]
        selected_hashes = [all_hashes[idx] for idx in item_idxs_to_select]

        if cache_0_p0 is None:
            cache_0_p0_out = selected_items
        else:
            for _ in range(is_cached_calls_per_iter):
                cache_0_p0.is_cached(selected_hashes)

            cache_0_p0_out = [
                item
                for item, _ in cache_0_p0.get_and_update(
                    [(item, [prompt_update]) for item in selected_items],
                    selected_hashes,
                )
            ]

        if cache_1_p0 is None:
            cache_1_p0_out = selected_items
        else:
            for _ in range(is_cached_calls_per_iter):
                cache_1_p0.is_cached(selected_hashes)

            cache_1_p0_out = [
                item
                for item, _ in cache_1_p0.get_and_update(
                    [(item, [prompt_update]) for item in selected_items],
                    selected_hashes,
                )
            ]

        if cache_0_p1 is None:
            cache_0_p1_out = cache_0_p0_out
        else:
            cache_0_p1_out = cache_0_p1.get_and_update(cache_0_p0_out, selected_hashes)

        if cache_1_p1 is None:
            cache_1_p1_out = cache_1_p0_out
        else:
            cache_1_p1_out = cache_1_p1.get_and_update(cache_1_p0_out, selected_hashes)

        assert cache_0_p1_out == cache_1_p1_out, f"Failed at {it=}"


@pytest.mark.parametrize("is_cached_calls_per_iter", [1, 2, 3])
def test_ipc_enable_disable_consistency(is_cached_calls_per_iter):
    cache_size_gb = 1 / (1 << 20)

    vllm_config_ipc_enabled = _create_vllm_config(
        mm_processor_cache_gb=cache_size_gb,
        enable_ipc=True,
    )
    vllm_config_ipc_disabled = _create_vllm_config(
        mm_processor_cache_gb=0,
        enable_ipc=False,
    )
    vllm_config_cache_disabled = _create_vllm_config(
        mm_processor_cache_gb=cache_size_gb,
        enable_ipc=True,
    )

    _compare_caches(
        vllm_config_ipc_enabled,
        vllm_config_ipc_disabled,
        is_cached_calls_per_iter=is_cached_calls_per_iter,
    )
    _compare_caches(
        vllm_config_ipc_disabled,
        vllm_config_cache_disabled,
        is_cached_calls_per_iter=is_cached_calls_per_iter,
    )
    _compare_caches(
        vllm_config_cache_disabled,
        vllm_config_ipc_enabled,
        is_cached_calls_per_iter=is_cached_calls_per_iter,
    )


def _run_test_cache_eviction_lru(
    p0_cache: BaseMultiModalProcessorCache,
    p1_cache: BaseMultiModalReceiverCache,
    base_item_size: int,
):
    request1_hashes = [
        "image_A",
        "image_B",
        "image_C",
    ]
    request1_items = {
        h: MultiModalKwargsItem.dummy(nbytes=2 * base_item_size)
        for h in request1_hashes
    }

    request2_hashes = ["image_D", "image_E", "image_A", "image_C"]
    request2_items = {
        h: MultiModalKwargsItem.dummy(nbytes=1 * base_item_size)
        for h in request2_hashes
    }

    ##########################
    # STEP 1: Request 1 send
    ##########################
    sender_is_cached_item_req1 = p0_cache.is_cached(request1_hashes)
    # Cache is empty
    assert sender_is_cached_item_req1 == [False, False, False]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request1_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 1 for P0 Cache
    ###########################
    item_tuple: MultiModalProcessorCacheInItem
    for i, h in enumerate(request1_hashes):
        # Use precomputed cache state
        is_cached = sender_is_cached_item_req1[i]
        item_tuple = (request1_items[h], []) if not is_cached else None
        print(f"Request 1: key={h} | cached={is_cached}")

        p0_cache.get_and_update_item(item_tuple, h)

    ###########################
    # Process request 1 for P1 Cache
    ###########################
    # Touch all mm hash for P1 Cache before process
    for mm_hash in request1_hashes:
        p1_cache.touch_receiver_cache_item(mm_hash)

    for h in request1_hashes:
        p1_cache.get_and_update_item(request1_items[h], h)

    expected_hashes = ["image_A", "image_B", "image_C"]
    assert list(p0_cache._cache.order) == expected_hashes

    ##########################
    # STEP 2: Request 2 send
    ##########################
    sender_is_cached_item_req2 = p0_cache.is_cached(request2_hashes)
    assert sender_is_cached_item_req2 == [False, False, True, True]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request2_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 2 for P0 Cache
    ###########################
    for i, h in enumerate(request2_hashes):
        # Use precomputed cache state again
        is_cached = sender_is_cached_item_req2[i]
        item_tuple = (request2_items[h], []) if not is_cached else None
        print(f"Request 2: key={h} | cached={is_cached}")

        p0_cache.get_and_update_item(item_tuple, h)

    ###########################
    # Process request 2 for P1 Cache
    ###########################

    # Touch all mm hash for P1 Cache before process
    for mm_hash in request2_hashes:
        p1_cache.touch_receiver_cache_item(mm_hash)

    for h in request2_hashes:
        p1_cache.get_and_update_item(request2_items[h], h)

    expected_hashes = ["image_D", "image_E", "image_A", "image_C"]
    assert list(p0_cache._cache.order) == expected_hashes


def test_cache_eviction_lru_cache():
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_gb=6 / GiB_bytes,
    )
    sender_cache = MultiModalProcessorSenderCache(model_config)
    receiver_cache = MultiModalReceiverCache(model_config)

    _run_test_cache_eviction_lru(sender_cache, receiver_cache, base_item_size=1)


# This test verifies shared-memory cache eviction behavior across processor (p0)
# and receiver (p1) caches.
# Flow summary:
# 1. Request 1 adds images A, B, C — completely filling the cache.
# 2. Request 2 tries to add image_G and image_A, but image_G cannot be added because
#    cache is full and A is protected from eviction — cache remains unchanged.
# 3. Request 3 adds image_G, image_H, image_I and image_B
#    this time, image_A is evicted, freeing 5MB space
#    and image_G, image_H successfully fits,
#    image_B is protected from eviction then image_i cannot be added.
#    This proving normal eviction and reuse behavior.
def _run_test_cache_eviction_shm(
    p0_cache: BaseMultiModalProcessorCache,
    p1_cache: BaseMultiModalReceiverCache,
    base_item_size: int,
):
    request1_hashes = ["image_A", "image_B", "image_C"]
    request1_items = {
        h: MultiModalKwargsItem.dummy(5 * base_item_size) for h in request1_hashes
    }
    request1_items_p0_result = []

    request2_hashes = ["image_G", "image_A"]
    request2_items = {
        h: MultiModalKwargsItem.dummy(
            (5 if h in request1_hashes else 2) * base_item_size
        )
        for h in request2_hashes
    }
    request2_items_p0_result = []

    request3_hashes = ["image_G", "image_H", "image_I", "image_B"]
    request3_items = {
        h: MultiModalKwargsItem.dummy(
            (5 if h in request1_hashes else 2) * base_item_size
        )
        for h in request3_hashes
    }
    request3_items_p0_result = []

    ##########################
    # STEP 1: Request 1 send
    # This will fill up the cache
    ##########################
    sender_is_cached_item_req1 = p0_cache.is_cached(request1_hashes)
    # Cache is empty
    assert sender_is_cached_item_req1 == [False, False, False]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request1_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 1 for P0 Cache
    ###########################
    item_tuple: MultiModalProcessorCacheInItem
    for i, h in enumerate(request1_hashes):
        # Use precomputed cache state
        is_cached = sender_is_cached_item_req1[i]
        item_tuple = (request1_items[h], []) if not is_cached else None
        print(f"Request 1: key={h} | cached={is_cached}")

        p0_result = p0_cache.get_and_update_item(item_tuple, h)
        # Only get mm item, ignore prompt update result
        request1_items_p0_result.append(p0_result[0])

    ###########################
    # Process request 1 for P1 Cache
    ###########################
    # Touch all mm hash for P1 Cache before process
    for mm_hash, mm_item in zip(request1_hashes, request1_items_p0_result):
        p1_cache.touch_receiver_cache_item(mm_hash, mm_item)

    for mm_hash, mm_item in zip(request1_hashes, request1_items_p0_result):
        p1_cache.get_and_update_item(mm_item, mm_hash)

    expected_hashes = ["image_A", "image_B", "image_C"]
    assert list(p0_cache._shm_cache.key_index.keys()) == expected_hashes

    ##########################
    # STEP 2: Request 2 send
    # There is no eviction because image_A is protected
    # No new item can add to cache
    ##########################
    sender_is_cached_item_req2 = p0_cache.is_cached(request2_hashes)
    assert sender_is_cached_item_req2 == [False, True]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request2_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 2 for P0 Cache
    ###########################
    for i, h in enumerate(request2_hashes):
        # Use precomputed cache state again
        is_cached = sender_is_cached_item_req2[i]
        item_tuple = (request2_items[h], []) if not is_cached else None
        print(f"Request 2: key={h} | cached={is_cached}")

        p0_result = p0_cache.get_and_update_item(item_tuple, h)
        # Only get mm item, ignore prompt update result
        request2_items_p0_result.append(p0_result[0])

    # image_A cannot be evict then
    # image_G will fail to allocate anyway and image_A still in cache
    assert p0_cache.is_cached(request2_hashes) == [False, True]

    ###########################
    # Process request 2 for P1 Cache
    ###########################

    # Touch all mm hash for P1 Cache before process
    for mm_hash, mm_item in zip(request2_hashes, request2_items_p0_result):
        p1_cache.touch_receiver_cache_item(mm_hash, mm_item)

    for mm_hash, mm_item in zip(request2_hashes, request2_items_p0_result):
        p1_cache.get_and_update_item(mm_item, mm_hash)

    # Prove that cache state is unchanged
    expected_hashes = ["image_A", "image_B", "image_C"]
    assert list(p0_cache._shm_cache.key_index.keys()) == expected_hashes

    ##########################
    # STEP 3: Request 3 send
    ##########################
    ##### Prove that cache eviction work normally
    sender_is_cached_item_req3 = p0_cache.is_cached(request3_hashes)
    assert sender_is_cached_item_req3 == [False, False, False, True]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request3_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 3 for P0 Cache
    ###########################
    for i, h in enumerate(request3_hashes):
        # Use precomputed cache state again
        is_cached = sender_is_cached_item_req3[i]
        item_tuple = (request3_items[h], []) if not is_cached else None
        print(f"Request 3: key={h} | cached={is_cached}")
        p0_result = p0_cache.get_and_update_item(item_tuple, h)
        # Only get mm item, ignore prompt update result
        request3_items_p0_result.append(p0_result[0])

    # image_A got evict and image_G add to cache
    # image_B is still protected
    # image_G, image_H fit but image_I cannot fit
    assert p0_cache.is_cached(request3_hashes) == [True, True, False, True]

    ###########################
    # Process request 3 for P1 Cache
    ###########################

    # Touch all mm hash for P1 Cache before process
    for mm_hash, mm_item in zip(request3_hashes, request3_items_p0_result):
        p1_cache.touch_receiver_cache_item(mm_hash, mm_item)

    for mm_hash, mm_item in zip(request3_hashes, request3_items_p0_result):
        p1_cache.get_and_update_item(mm_item, mm_hash)

    expected_hashes = ["image_B", "image_C", "image_G", "image_H"]
    assert list(p0_cache._shm_cache.key_index.keys()) == expected_hashes


def test_cache_eviction_shm_cache():
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            mm_processor_cache_type="shm",
            mm_shm_cache_max_object_size_mb=6,
            mm_processor_cache_gb=15.2 * MiB_bytes / GiB_bytes,
        ),
    )
    sender_cache = ShmObjectStoreSenderCache(vllm_config)
    receiver_cache = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())

    _run_test_cache_eviction_shm(sender_cache, receiver_cache, base_item_size=MiB_bytes)


def test_processor_cache_shared_across_loras():
    """Test that processor cache uses mm_hash to share data across LoRAs."""
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_gb=1,
    )
    receiver_cache = MultiModalReceiverCache(model_config)

    base_mm_hash = "image_hash_abc123"
    lora_a_identifier = f"12345:{base_mm_hash}"
    lora_b_identifier = f"67890:{base_mm_hash}"

    item_data = MultiModalKwargsItem.dummy(1024)

    feature_lora_a = MultiModalFeatureSpec(
        data=item_data,
        modality="image",
        identifier=lora_a_identifier,
        mm_position=PlaceholderRange(offset=0, length=100),
        mm_hash=base_mm_hash,
    )

    receiver_cache.get_and_update_features([feature_lora_a])
    assert base_mm_hash in receiver_cache._cache

    feature_lora_b = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=lora_b_identifier,
        mm_position=PlaceholderRange(offset=0, length=100),
        mm_hash=base_mm_hash,
    )

    receiver_cache.get_and_update_features([feature_lora_b])
    assert feature_lora_b.data == item_data


class _CountingShmStorage(SingleWriterShmObjectStorage):
    """A worker's shm reader that records every get()."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gets: list[tuple[int, int]] = []

    def get(self, address: int, monotonic_id: int) -> Any:
        self.gets.append((address, monotonic_id))
        return super().get(address, monotonic_id)


class _ShmReceiverCacheStub(ShmObjectStoreReceiverCache):
    """The worker-side shm receiver cache around a given reader."""

    def __init__(self, reader: SingleWriterShmObjectStorage) -> None:
        self._shm_cache = reader


class _ShmTPWorkers:
    """P0's shm writer and one WorkerWrapperBase per TP rank, each reading
    through its own shm receiver cache."""

    def __init__(self, tp_size: int) -> None:
        self.writer = SingleWriterShmObjectStorage(
            max_object_size=10 * 1024,
            n_readers=tp_size,
            ring_buffer=SingleWriterShmRingBuffer(
                data_buffer_size=100 * 1024, create=True
            ),
            serde_class=MsgpackSerde,
        )
        reader_lock = mp.Lock()
        self.readers: list[_CountingShmStorage] = []
        self.workers: list[WorkerWrapperBase] = []
        for rank in range(tp_size):
            reader = _CountingShmStorage(
                max_object_size=10 * 1024,
                n_readers=tp_size,
                ring_buffer=SingleWriterShmRingBuffer(
                    *self.writer.ring_buffer.handle()
                ),
                serde_class=MsgpackSerde,
                reader_lock=reader_lock,
            )
            worker = WorkerWrapperBase(rpc_rank=rank)
            worker.mm_receiver_cache = _ShmReceiverCacheStub(reader)
            self.readers.append(reader)
            self.workers.append(worker)

    def close(self) -> None:
        for reader in self.readers:
            reader.close()
        self.writer.close()

    def reference(
        self, mm_hash: str, item: MultiModalKwargsItem
    ) -> MultiModalKwargsItem:
        """What P0 sends for one reference to an item: one writer count."""
        if self.writer.is_cached(mm_hash):
            address, monotonic_id = self.writer.get_cached(mm_hash)
        else:
            address, monotonic_id = self.writer.put(mm_hash, item)
        return MultiModalKwargsItem(
            {
                "address": MultiModalFieldElem(
                    data=address, field=MultiModalBatchedField()
                ),
                "monotonic_id": MultiModalFieldElem(
                    data=monotonic_id, field=MultiModalBatchedField()
                ),
            }
        )

    def step(
        self,
        *,
        new_reqs: list[tuple[str, list[MultiModalFeatureSpec]]] | None = None,
        finished: set[str] | None = None,
        preempted: set[str] | None = None,
    ) -> list[SchedulerOutput]:
        """Send one scheduler output to every rank and return what each rank
        resolved. Each rank gets its own copy, as from the executor's
        broadcast, so the engine's features keep their shm references."""
        output = SchedulerOutput.make_empty()
        output.scheduled_new_reqs = [
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[0],
                mm_features=features,
                sampling_params=None,
                pooling_params=None,
                block_ids=([],),
                num_computed_tokens=0,
                lora_request=None,
            )
            for req_id, features in new_reqs or []
        ]
        output.finished_req_ids = finished or set()
        output.preempted_req_ids = preempted
        received = []
        for worker in self.workers:
            worker_output = copy.deepcopy(output)
            worker._apply_mm_cache(worker_output)
            received.append(worker_output)
        return received

    def num_gets(self) -> list[int]:
        return [len(reader.gets) for reader in self.readers]

    def is_evictable(self, mm_hash: str) -> bool:
        self.writer.free_unused()
        return not self.writer.is_cached(mm_hash)


def _image_feature(
    mm_hash: str, data: MultiModalKwargsItem, offset: int
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=data,
        modality="image",
        identifier=mm_hash,
        mm_position=PlaceholderRange(offset=offset, length=4),
        mm_hash=mm_hash,
    )


def _resolved_data(output: SchedulerOutput) -> list[MultiModalKwargsItem | None]:
    return [feature.data for feature in output.scheduled_new_reqs[0].mm_features]


def test_shm_worker_reads_each_reference_once_across_preemption():
    """Model Runner V2 sends a resumed request in scheduled_new_reqs again,
    with the shm references the engine kept for it. Reading them again
    over-counts the readers, and the slot may hold another item by then."""
    rng = np.random.RandomState(0)
    image_a, image_b, image_c = (
        _dummy_item({"pixel_values": 1024}, rng=rng) for _ in range(3)
    )
    shm = _ShmTPWorkers(tp_size=2)
    try:
        # image_a twice in one request: two writer counts, two reads per rank.
        features = [
            _image_feature("a", shm.reference("a", image_a), offset=0),
            _image_feature("a", shm.reference("a", image_a), offset=4),
            _image_feature("b", shm.reference("b", image_b), offset=8),
        ]
        first = shm.step(new_reqs=[("req", features)])
        assert shm.num_gets() == [3, 3]
        assert all(_resolved_data(out) == [image_a, image_a, image_b] for out in first)
        assert shm.is_evictable("a")
        assert shm.is_evictable("b")

        shm.step(preempted={"req"})
        # While the request waits, P0 resets its cache and stores another
        # image under the reference image_a had.
        shm.writer.clear()
        address_a = features[0].data["address"].data
        monotonic_id_a = features[0].data["monotonic_id"].data
        assert shm.writer.put("c", image_c) == (address_a, monotonic_id_a)

        resumed = shm.step(new_reqs=[("req", features)])
        assert shm.num_gets() == [3, 3]
        for before, after in zip(first, resumed):
            assert _resolved_data(after) == [image_a, image_a, image_b]
            # The very objects fetched before: no second copy.
            assert all(
                x is y for x, y in zip(_resolved_data(before), _resolved_data(after))
            )
        assert not shm.is_evictable("c")

        shm.step(finished={"req"})
        assert all(not worker._shm_mm_items for worker in shm.workers)
    finally:
        shm.close()


def test_shm_worker_streaming_chunk_reads_only_new_references():
    """A streaming-input session comes back in scheduled_new_reqs with every
    chunk. Only the new chunk's references are read, also one naming an
    image an earlier chunk had. An inline item is passed through as sent."""
    rng = np.random.RandomState(0)
    image_a, image_b, inline_item = (
        _dummy_item({"pixel_values": 1024}, rng=rng) for _ in range(3)
    )
    shm = _ShmTPWorkers(tp_size=1)
    try:
        features = [
            _image_feature("a", shm.reference("a", image_a), offset=0),
            _image_feature("x", inline_item, offset=4),
        ]
        shm.step(new_reqs=[("session", features)])
        assert shm.num_gets() == [1]

        features += [
            _image_feature("a", shm.reference("a", image_a), offset=8),
            _image_feature("b", shm.reference("b", image_b), offset=12),
        ]
        (output,) = shm.step(new_reqs=[("session", features)])
        assert shm.num_gets() == [3]
        assert _resolved_data(output) == [image_a, inline_item, image_a, image_b]
        assert shm.is_evictable("a")
        assert shm.is_evictable("b")
        # The engine sends the inline item with every chunk; it is not kept.
        assert shm.workers[0]._shm_mm_items["session"][1] is None
    finally:
        shm.close()


def test_shm_worker_drops_a_request_aborted_while_preempted():
    """A request aborted while preempted is reported in finished_req_ids.
    Its items are dropped before the step's new requests, so a new request
    reusing the id and the image reads its own reference."""
    image_a = _dummy_item({"pixel_values": 1024}, rng=np.random.RandomState(0))
    shm = _ShmTPWorkers(tp_size=1)
    try:
        shm.step(
            new_reqs=[("req", [_image_feature("a", shm.reference("a", image_a), 0)])]
        )
        shm.step(preempted={"req"})
        assert "req" in shm.workers[0]._shm_mm_items

        (output,) = shm.step(
            finished={"req"},
            new_reqs=[("req", [_image_feature("a", shm.reference("a", image_a), 0)])],
        )
        assert shm.num_gets() == [2]
        assert _resolved_data(output) == [image_a]
        assert shm.is_evictable("a")

        shm.step(finished={"req"})
        assert not shm.workers[0]._shm_mm_items
    finally:
        shm.close()


_SLEEP_VISION_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="sleep mode regression requires a CUDA GPU",
)
def test_sleep_wake_preserves_mm_cache_consistency():
    """Regression for vllm-project/vllm#42995."""
    from vllm import LLM, SamplingParams
    from vllm.assets.image import ImageAsset

    image = ImageAsset("stop_sign").pil_image
    prompt = {
        "prompt": _SLEEP_VISION_PROMPT,
        "multi_modal_data": {"image": image},
    }
    sampling_params = SamplingParams(temperature=0, max_tokens=8)

    llm = LLM(
        model="Qwen/Qwen2-VL-2B-Instruct",
        enable_sleep_mode=True,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
    )

    llm.generate([prompt], sampling_params)
    llm.sleep(level=1)
    llm.wake_up()
    output2 = llm.generate([prompt], sampling_params)
    assert output2[0].outputs[0].text
