# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU composition proof: existing tier managers and FS I/O behind group routing.

This deliberately does not enable grouped TieringOffloadingSpec in serving.
It validates the scheduler/storage seam, not CUDA registration or GPU transfers.
"""

import json
import multiprocessing
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.base import (
    OffloadingManager,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import GroupedCPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

GROUP_PAGES = {0: 8192, 2: 4096}
WORLD_SIZE = 2
CTX = ReqContext("grouped-fs")


def _key(group, number=1):
    return make_offload_key(number.to_bytes(8, "big"), group)


def _spec():
    # The existing FileMapper retains the original group IDs in OffloadKey.
    return SimpleNamespace(
        block_size_factor=1,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="test/grouped-hybrid"),
            cache_config=SimpleNamespace(block_size=16, cache_dtype="float16"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=WORLD_SIZE,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                rank=0,
            ),
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(block_size=16),
                    layer_names=[f"layer-{g}"],
                )
                for g in range(3)
            ]
        ),
    )


def _build(root):
    managers = {}
    for group, page in GROUP_PAGES.items():
        region = SharedOffloadRegion(
            instance_id=f"grouped-fs-test-{uuid.uuid4().hex}",
            total_size_bytes=2 * page * WORLD_SIZE,
            num_blocks=2,
            rank=None,
            num_workers=WORLD_SIZE,
            cpu_page_size=page,
        )
        primary = CPUPrimaryTierOffloadingManager(
            num_blocks=2, mmap_region=region, enable_events=True
        )
        secondary = SecondaryTierFactory.create_secondary_tier(
            {
                "type": "fs",
                "root_dir": str(root),
                "n_read_threads": 1,
                "n_write_threads": 1,
            },
            primary.get_kv_memoryview(),
            _spec(),
        )
        managers[group] = TieringOffloadingManager(
            primary_tier=primary, secondary_tiers=[secondary], enable_events=True
        )
    return GroupedCPUOffloadingManager(managers)


def _settle(manager):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        list(manager.take_events())
        if all(
            not m._transfer_jobs and not m._pending_load_submissions
            for m in manager.managers.values()
        ):
            return
        time.sleep(0.005)
    raise AssertionError("FS jobs did not finish")


def _payload(group):
    # Distinguish both groups and both TP rank slices within each stored row.
    return b"".join(
        bytes([31 + group + rank * 70]) * GROUP_PAGES[group]
        for rank in range(WORLD_SIZE)
    )


def _write_rows(manager, output, payloads):
    for key, block_id in zip(output.keys_to_store, output.store_spec.block_ids):
        group = int.from_bytes(key[-4:], "big")
        primary = manager.managers[group].primary_tier
        with primary.get_kv_memoryview().cast("B") as view:
            size = len(payloads[group])
            offset = int(block_id) * size
            view[offset : offset + size] = payloads[group]
    manager.complete_store(output.keys_to_store, CTX)
    _settle(manager)


def _phase(root, phase):
    """Runs in a fresh spawn process, so no cache index or mapping is inherited."""
    manager = _build(root)
    paths = []
    try:
        manager.on_new_request(CTX)
        keys = [_key(g) for g in GROUP_PAGES]
        if phase == "write":
            output = manager.prepare_store(keys, CTX)
            assert list(output.store_spec.block_ids) == [0, 0]
            _write_rows(manager, output, {g: _payload(g) for g in GROUP_PAGES})
            for g, child in manager.managers.items():
                path = Path(child.secondary_tiers[0].file_mapper.get_file_name(_key(g)))
                assert path.read_bytes() == _payload(g)
                paths.append(str(path))
            assert paths[0] != paths[1]
        else:
            assert all(
                child.primary_tier.lookup(_key(g), CTX) is False
                for g, child in manager.managers.items()
            )
            # Occupy slot 0: persisted identity must not depend on the old slot.
            filler = manager.prepare_store([_key(g, 2) for g in GROUP_PAGES], CTX)
            _write_rows(
                manager,
                filler,
                {g: bytes([99]) * len(_payload(g)) for g in GROUP_PAGES},
            )
            assert all(manager.lookup(key, CTX) is None for key in keys)
            _settle(manager)
            if phase == "corrupt":
                assert manager.lookup(_key(0), CTX) is False
                assert manager.managers[0].primary_tier.lookup(_key(0), CTX) is False
                keys = [_key(2)]
            assert all(manager.lookup(key, CTX) is True for key in keys)
            loaded = manager.prepare_load(keys, CTX)
            assert list(loaded.block_ids) == [1] * len(keys)
            for key, bid in zip(keys, loaded.block_ids):
                g = int.from_bytes(key[-4:], "big")
                with (
                    manager.managers[g]
                    .primary_tier.get_kv_memoryview()
                    .cast("B") as view
                ):
                    size = len(_payload(g))
                    assert bytes(
                        view[int(bid) * size : (int(bid) + 1) * size]
                    ) == _payload(g)
            manager.complete_load(keys, CTX)
            # Fill beyond capacity; old filler rows are evicted through standard LRU.
            manager.touch(keys, CTX)
            pressure = manager.prepare_store([_key(g, 3) for g in GROUP_PAGES], CTX)
            assert set(pressure.evicted_keys) == (
                {_key(2, 2)}
                if phase == "corrupt"
                else {_key(g, 2) for g in GROUP_PAGES}
            )
            assert list(pressure.store_spec.block_ids) == (
                [1, 0] if phase == "corrupt" else [0, 0]
            )
            _write_rows(
                manager,
                pressure,
                {g: bytes([100]) * len(_payload(g)) for g in GROUP_PAGES},
            )
        manager.on_request_finished(CTX)
        paths += [
            m.primary_tier._mmap_region.mmap_path for m in manager.managers.values()
        ]
    finally:
        manager.shutdown()
    assert all(not Path(p).exists() for p in paths if p.endswith(".mmap"))
    (Path(root) / f"{phase}.json").write_text(
        json.dumps({"phase": phase, "pass": True})
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_fs_groups_survive_process_restart(tmp_path, corrupt):
    for phase in ("write", "corrupt" if corrupt else "read"):
        if phase == "corrupt":
            files = list(tmp_path.glob("**/*_g0/*.bin"))
            assert len(files) == 1
            files[0].write_bytes(b"truncated")
        process = multiprocessing.get_context("spawn").Process(
            target=_phase, args=(str(tmp_path), phase)
        )
        process.start()
        process.join(60)
        if process.is_alive():
            process.terminate()
            process.join(10)
            pytest.fail(f"{phase} process did not finish")
        assert process.exitcode == 0
        assert json.loads((tmp_path / f"{phase}.json").read_text())["pass"]


def test_group_routing_forwards_all_lifecycle_hooks():
    children = {g: MagicMock(spec=OffloadingManager) for g in (0, 2, 5)}
    for g, child in children.items():
        child.on_new_request.return_value = RequestOffloadingContext(
            policy=OffloadPolicy.REQUEST_LEVEL if g == 0 else OffloadPolicy.BLOCK_LEVEL
        )
    manager = GroupedCPUOffloadingManager(children)
    assert manager.on_new_request(CTX).policy == OffloadPolicy.REQUEST_LEVEL
    manager.on_request_finished(CTX)
    manager.shutdown()
    for child in children.values():
        child.on_new_request.assert_called_once_with(CTX)
        child.on_request_finished.assert_called_once_with(CTX)
        child.shutdown.assert_called_once_with()
