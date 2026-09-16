# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import regex as re
import torch
from torch import nn
from torch.nn import functional as F

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.common.ple as ple_common
import vllm.models.qwen4_exp.nvidia.ple_layer as ple_module
from tests.utils import set_lazy_env
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.models.qwen4_exp.common.ple import (
    PLEDiskSegment,
    PLEPlacement,
    PLERemotePlacement,
    PLEShardOverlap,
    PLEStoreSegment,
    auto_ple_host_budget_bytes,
    available_host_bytes,
    cap_host_budget_bytes,
    check_ple_host_share,
    compute_ple_shard_overlap,
    copy_ple_embedding_shard_,
    copy_ple_embedding_shard_tiers_,
    plan_ple_placement,
    plan_ple_worker_segments,
    ple_disk_mask,
    ple_store_indices,
    total_host_bytes,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLELayer,
    _get_ple_embedding_quant_method,
)


def _patch_tp(monkeypatch: pytest.MonkeyPatch, rank: int, world_size: int) -> None:
    monkeypatch.setattr(ple_module, "is_pin_memory_available", lambda: True)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: world_size
    )


def _pinned_layer(
    num_embeddings: int = 32,
    embedding_dim: int = 8,
    prefix: str = "model.layers.2.ple.ngram_embedding",
):
    # The table registers itself in the config's static forward context so
    # the gather op can resolve it by name at run time.
    with set_current_vllm_config(VllmConfig()):
        return Qwen4ExpPinnedHostEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            params_dtype=torch.float16,
            padding_size=8,
            prefix=prefix,
            quant_method=Qwen4ExpPLEFp8EmbeddingMethod(),
        )


def _expose_to_gather_op(monkeypatch: pytest.MonkeyPatch, *layers) -> None:
    # qwen4_exp_ple_pinned_gather resolves the table through the forward
    # context, like qwen4_exp_compute_ple_ngram_ids does for its layer.
    by_name = {layer.layer_name: layer for layer in layers}
    monkeypatch.setattr(
        ple_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers=by_name),
    )


def _patch_cascade(
    monkeypatch: pytest.MonkeyPatch,
    *,
    host_bytes: int | None,
    store_bytes: int,
    store_device: int | None = 4,
    disk: bool = False,
) -> None:
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: host_bytes)
    monkeypatch.setattr(ple_module, "ple_store_device", lambda: store_device)
    monkeypatch.setattr(ple_module, "ple_store_budget_bytes", lambda: store_bytes)
    monkeypatch.setattr(ple_module, "ple_disk_tier_allowed", lambda: disk)
    monkeypatch.setattr(
        ple_module, "ple_cascade_configured", lambda: store_device is not None or disk
    )


def _patch_device_spill(monkeypatch: pytest.MonkeyPatch, layer, spill: int) -> None:
    # The real measurement reads the device, the KV specs and the config.
    monkeypatch.setattr(layer, "_device_spill_bytes", lambda device, table_bytes: spill)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_pinned_host_ple_allocates_nothing_before_the_first_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tp(monkeypatch, rank=2, world_size=4)

    layer = _pinned_layer()

    assert layer.tp_size == 4
    # The loader contract is kept by a row-less CPU placeholder; the TP shard
    # of 8 rows is only placed once the budget can see the real headroom.
    assert layer.weight.shape == (0, 8)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.device.type == "cpu"
    assert layer.weight._vllm_keep_on_cpu
    assert layer.ple_device_table is None and layer.ple_host_storage is None
    assert not layer.weight_scale.is_meta
    assert layer.weight_scale.dtype == torch.float16
    assert layer._accelerator_weight_views == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_pinned_host_ple_splits_the_shard_by_host_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tp(monkeypatch, rank=2, world_size=4)
    # 3 rows x 8 bytes fit the host budget, the other 5 rows stay on device.
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: 3 * 8)

    layer = _pinned_layer()
    layer.materialize_tables()

    assert layer.ple_device_table is not None and layer.ple_host_storage is not None
    assert layer.ple_device_table.shape == (5, 8)
    assert layer.ple_device_table.device.type == "cuda"
    assert layer.ple_host_storage.shape == (3, 8)
    assert layer.ple_host_storage.is_pinned()
    assert (layer._device_rows, layer._host_rows) == (5, 3)
    # Idempotent: a second call keeps the tables.
    device_table = layer.ple_device_table
    layer.materialize_tables()
    assert layer.ple_device_table is device_table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_pinned_host_ple_without_budget_keeps_everything_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tp(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: 0)

    layer = _pinned_layer(num_embeddings=8)
    # Preparing the accelerator view (process_weights_after_loading) must not
    # depend on a shard having been loaded, e.g. with dummy weights.
    layer.prepare_accelerator_weight()

    assert layer.ple_device_table is not None
    assert layer.ple_device_table.shape == (8, 8)
    assert layer.ple_host_storage is not None
    assert layer.ple_host_storage.shape == (0, 8)
    assert layer._host_rows == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)
@pytest.mark.parametrize("host_rows", [0, 4, 8])
def test_pinned_host_ple_fp8_rows_are_gatherable_across_the_split_on_sm70(
    monkeypatch: pytest.MonkeyPatch,
    host_rows: int,
) -> None:
    _patch_tp(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr(
        embedding_module, "tensor_model_parallel_all_reduce", lambda tensor: tensor
    )
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: host_rows * 8)
    layer = _pinned_layer(num_embeddings=8)
    _expose_to_gather_op(monkeypatch, layer)
    raw = torch.tensor(
        [0x00, 0x01, 0x08, 0x38, 0x7E, 0x80, 0xB8, 0xFE],
        dtype=torch.uint8,
    ).repeat(8, 1)
    # Distinct rows: row i carries the pattern rotated by i.
    raw = torch.stack([raw[i].roll(i) for i in range(8)])
    checkpoint = raw.view(torch.float8_e4m3fn)
    copied = layer.load_shard(checkpoint, checkpoint_start=0, tp_start=0, tp_end=8)
    assert copied == 8
    assert (layer._device_rows, layer._host_rows) == (8 - host_rows, host_rows)
    layer.weight_scale = nn.Parameter(
        torch.tensor([0.25], dtype=torch.float16, device="cuda"),
        requires_grad=False,
    )
    layer.prepare_accelerator_weight()

    ids = torch.tensor([0, 3, 4, 7, 2], dtype=torch.int64, device="cuda")
    output = layer(ids)
    torch.accelerator.synchronize()

    assert output.dtype == torch.float16
    expected = checkpoint.float() * 0.25
    torch.testing.assert_close(output.float().cpu(), expected[ids.cpu()])

    pointers = (layer._device_table_ptr, dict(layer._accelerator_weight_ptrs))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = layer(ids)
    for offset in range(4):
        # Reload both halves in-place; captured pointers must stay live.
        reloaded = raw.roll(offset, dims=0).view(torch.float8_e4m3fn)
        layer.load_shard(reloaded, checkpoint_start=0, tp_start=0, tp_end=8)
        layer.prepare_accelerator_weight()
        ids.copy_((ids + 1) % 8)
        graph.replay()
        expected = reloaded.float() * 0.25
        torch.testing.assert_close(graph_output.float().cpu(), expected[ids.cpu()])
        assert pointers == (layer._device_table_ptr, layer._accelerator_weight_ptrs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires pinned memory")
@pytest.mark.parametrize("host_rows", [0, 4, 8])
def test_dummy_ple_tables_do_not_retain_uninitialized_bytes(monkeypatch, host_rows):
    _patch_tp(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: host_rows * 8)
    layer = _pinned_layer(num_embeddings=8)
    empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        result = empty(*args, **kwargs)
        if result.dtype == torch.float8_e4m3fn:
            result.view(torch.uint8).fill_(0x7F)  # E4M3 NaN, not valid dummy data.
        return result

    monkeypatch.setattr(ple_module.torch, "empty", poisoned_empty)
    layer.prepare_accelerator_weight()
    for table in (layer.ple_device_table, layer.ple_host_storage):
        assert torch.all(table.view(torch.uint8) == 0)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("kind", ["HOST", "HOST_RESERVE", "VRAM_RESERVE", "STORE"])
def test_ple_budget_rejects_invalid_values(monkeypatch, kind, value):
    monkeypatch.setattr(ple_module.envs, f"VLLM_QWEN4EXP_PLE_{kind}_GIB", value)
    with pytest.raises(ValueError, match="finite and non-negative"):
        if kind == "HOST":
            ple_common.ple_host_budget_bytes()
        elif kind == "HOST_RESERVE":
            ple_common.ple_host_reserve_bytes(32 * 1024**3)
        elif kind == "VRAM_RESERVE":
            ple_common.ple_vram_reserve_bytes(32 * 1024**3)
        else:
            ple_common.ple_store_budget_bytes()


def test_ple_store_budget_is_required_and_in_bytes(monkeypatch) -> None:
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_STORE_GIB", None)
    with pytest.raises(ValueError, match="requires VLLM_QWEN4EXP_PLE_STORE_GIB"):
        ple_common.ple_store_budget_bytes()
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_STORE_GIB", "1.5")
    assert ple_common.ple_store_budget_bytes() == int(1.5 * 1024**3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_cascade_rank_leaves_the_overflow_to_the_store_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tp(monkeypatch, rank=0, world_size=1)
    _patch_cascade(monkeypatch, host_bytes=3 * 8, store_bytes=7 * 8)
    layer = _pinned_layer(num_embeddings=16)
    # 10.6 rows do not fit on the device: the partial row stays there as it
    # does without the cascade, the host takes its 3 rows, the store the rest.
    _patch_device_spill(monkeypatch, layer, 10 * 8 + 5)
    layer.materialize_tables()

    assert layer.ple_device_table is not None and layer.ple_host_storage is not None
    assert layer.ple_device_table.shape == (6, 8)
    assert layer.ple_host_storage.shape == (3, 8)
    assert (layer._device_rows, layer._host_rows, layer._store_rows) == (6, 3, 7)
    assert layer.local_rows == 9

    raw = torch.arange(16 * 8, dtype=torch.uint8).view(16, 8)
    copied = layer.load_shard(
        raw.view(torch.float8_e4m3fn), checkpoint_start=0, tp_start=0, tp_end=16
    )
    assert copied == 9
    assert torch.equal(layer.ple_device_table.view(torch.uint8).cpu(), raw[:6])
    assert torch.equal(layer.ple_host_storage.view(torch.uint8), raw[6:9])

    with pytest.raises(RuntimeError, match="no rows from the PLE offload worker"):
        layer.embedding_lookup(torch.zeros(2, dtype=torch.int64, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_cascade_rank_sends_the_last_rows_to_the_disk_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No spare card at all: device, host, then the mapped checkpoint.
    _patch_tp(monkeypatch, rank=0, world_size=1)
    _patch_cascade(
        monkeypatch, host_bytes=3 * 8, store_bytes=0, store_device=None, disk=True
    )
    layer = _pinned_layer(num_embeddings=16)
    _patch_device_spill(monkeypatch, layer, 10 * 8)
    layer.materialize_tables()
    assert (layer._device_rows, layer._host_rows) == (6, 3)
    assert (layer._store_rows, layer._disk_rows) == (0, 7)
    assert layer.local_rows == 9

    # With a store card in front of it the disk only takes what is left.
    _patch_cascade(monkeypatch, host_bytes=3 * 8, store_bytes=4 * 8, disk=True)
    other = _pinned_layer(num_embeddings=16)
    _patch_device_spill(monkeypatch, other, 10 * 8)
    other.materialize_tables()
    assert (other._store_rows, other._disk_rows) == (4, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_cascade_rank_refuses_rows_beyond_the_store_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tp(monkeypatch, rank=0, world_size=1)
    _patch_cascade(monkeypatch, host_bytes=3 * 8, store_bytes=6 * 8)
    layer = _pinned_layer(num_embeddings=16)
    _patch_device_spill(monkeypatch, layer, 10 * 8)
    with pytest.raises(ValueError, match="does not fit"):
        layer.materialize_tables()
    assert layer.ple_device_table is None

    # Nothing may stay resident: the always-running gathers would read row 0.
    _patch_cascade(monkeypatch, host_bytes=0, store_bytes=16 * 8)
    _patch_device_spill(monkeypatch, layer, 16 * 8)
    with pytest.raises(RuntimeError, match="at least one resident row"):
        layer.materialize_tables()


def test_configured_host_share_is_checked_once_before_the_ranks_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    ple_config = SimpleNamespace(ple_layer_ids=[1])
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", "6")
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    monkeypatch.setattr(ple_common, "total_host_bytes", lambda: 30 * gib)
    # 30 GiB host, 7.5 GiB reserve: two ranks of 6 GiB need 19.5 GiB available.
    monkeypatch.setattr(ple_common, "available_host_bytes", lambda: int(19.5 * gib))
    check_ple_host_share(ple_config, ranks_sharing_host=2)
    monkeypatch.setattr(ple_common, "available_host_bytes", lambda: 19 * gib)
    with pytest.raises(ValueError, match="asks for 6.0 GiB .* at most 5.75 GiB"):
        check_ple_host_share(ple_config, ranks_sharing_host=2)
    # A smaller reserve makes room again.
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", "4")
    check_ple_host_share(ple_config, ranks_sharing_host=2)

    # Nothing to check: models without PLE, a derived share, an unknown host.
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    check_ple_host_share(SimpleNamespace(), ranks_sharing_host=2)
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", None)
    check_ple_host_share(ple_config, ranks_sharing_host=2)
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", "6")
    monkeypatch.setattr(ple_common, "available_host_bytes", lambda: None)
    check_ple_host_share(ple_config, ranks_sharing_host=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_configured_host_share_is_used_as_given_by_the_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The rank does not read the host memory again: its siblings may already
    # pin their shares, which would count them twice.
    _patch_tp(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: 4 * 8)
    monkeypatch.setattr(ple_module, "available_host_bytes", lambda: 0)
    layer = _pinned_layer(num_embeddings=16)
    layer.materialize_tables()
    assert (layer._device_rows, layer._host_rows, layer._store_rows) == (12, 4, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_derived_host_share_cut_by_the_host_goes_to_the_store_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tp(monkeypatch, rank=0, world_size=1)
    _patch_cascade(monkeypatch, host_bytes=None, store_bytes=16 * 8)
    monkeypatch.setattr(ple_module, "available_host_bytes", lambda: 3 * 8)
    monkeypatch.setattr(ple_module, "total_host_bytes", lambda: 32 * 1024**3)
    monkeypatch.setattr(ple_module, "ple_host_reserve_bytes", lambda total: 0)
    layer = _pinned_layer(num_embeddings=16)
    _patch_device_spill(monkeypatch, layer, 10 * 8)
    layer.materialize_tables()
    assert (layer._device_rows, layer._host_rows, layer._store_rows) == (6, 3, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ple_host_allocation_failure_keeps_materialization_retryable(monkeypatch):
    _patch_tp(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: 4 * 8)
    layer = _pinned_layer(num_embeddings=8)
    empty = torch.empty

    def failing_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("injected pinned allocation failure")
        return empty(*args, **kwargs)

    monkeypatch.setattr(ple_module.torch, "empty", failing_empty)
    with pytest.raises(RuntimeError, match="injected pinned allocation failure"):
        layer.materialize_tables()
    assert layer.ple_device_table is None
    assert layer.ple_host_storage is None
    assert layer._device_table_ptr == 0
    monkeypatch.setattr(ple_module.torch, "empty", empty)
    layer.materialize_tables()
    assert (layer._device_rows, layer._host_rows) == (4, 4)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((7, 0), True), ((7, 5), True), ((8, 0), False), ((8, 9), False)],
)
def test_pinned_host_ple_decides_on_the_worker_device(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: bool
) -> None:
    from vllm.platforms.interface import DeviceCapability

    seen: list[int] = []

    def fake_capability(device_id: int = 0) -> DeviceCapability:
        seen.append(device_id)
        return DeviceCapability(*capability)

    monkeypatch.setattr(ple_module, "is_offload_process", lambda: False)
    monkeypatch.setattr(ple_module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        ple_module.current_platform, "get_device_capability", fake_capability
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)

    config = SimpleNamespace(ple_offload_embedding=None)
    assert ple_module._should_use_pinned_host_ple(config) is expected
    # The worker's own device, not device 0 of the visible list.
    assert seen == [3]
    # An explicit config choice wins over the capability.
    assert ple_module._should_use_pinned_host_ple(
        SimpleNamespace(ple_offload_embedding=not expected)
    ) is (not expected)


def _plan(
    total_rows: int,
    host_rows: int,
    vram_rows: int | None,
    store_rows: int,
    disk_allowed: bool = False,
):
    return plan_ple_placement(
        total_rows=total_rows,
        row_bytes=8,
        host_budget_bytes=host_rows * 8,
        vram_budget_bytes=None if vram_rows is None else vram_rows * 8,
        store_budget_bytes=store_rows * 8,
        disk_allowed=disk_allowed,
    )


def test_plan_ple_placement_spills_only_what_the_budget_holds() -> None:
    no_cascade = dict(vram_budget_bytes=None, store_budget_bytes=0)
    assert plan_ple_placement(
        total_rows=10, row_bytes=8, host_budget_bytes=0, **no_cascade
    ) == plan_ple_placement(
        total_rows=10, row_bytes=8, host_budget_bytes=7, **no_cascade
    )
    placement = plan_ple_placement(
        total_rows=10, row_bytes=8, host_budget_bytes=3 * 8 + 7, **no_cascade
    )
    assert placement == PLEPlacement(
        vram_rows=7, host_rows=3, store_rows=0, disk_rows=0
    )
    # A budget beyond the table never inflates the host part.
    big = plan_ple_placement(
        total_rows=10, row_bytes=8, host_budget_bytes=10**9, **no_cascade
    )
    assert big == PLEPlacement(vram_rows=0, host_rows=10, store_rows=0, disk_rows=0)
    with pytest.raises(ValueError):
        plan_ple_placement(
            total_rows=10, row_bytes=8, host_budget_bytes=-1, **no_cascade
        )


def test_plan_ple_placement_cascades_beyond_device_and_host() -> None:
    # The host takes its share, the device its measured budget, the store the rest.
    placement = _plan(total_rows=20, host_rows=5, vram_rows=9, store_rows=6)
    assert placement == PLEPlacement(
        vram_rows=9, host_rows=5, store_rows=6, disk_rows=0
    )
    assert (placement.local_rows, placement.total_rows) == (14, 20)
    # Fastest tier first: a device that holds the whole table leaves the host
    # share unused, so a fitting table pins no host memory.
    assert _plan(total_rows=20, host_rows=5, vram_rows=99, store_rows=0) == (
        PLEPlacement(vram_rows=20, host_rows=0, store_rows=0, disk_rows=0)
    )
    # The host takes only what the device could not hold.
    assert _plan(total_rows=20, host_rows=5, vram_rows=18, store_rows=0) == (
        PLEPlacement(vram_rows=18, host_rows=2, store_rows=0, disk_rows=0)
    )
    # Without a cascade the configured host share comes first, as before.
    assert plan_ple_placement(
        total_rows=20,
        row_bytes=8,
        host_budget_bytes=5 * 8,
        vram_budget_bytes=None,
        store_budget_bytes=0,
    ) == PLEPlacement(vram_rows=15, host_rows=5, store_rows=0, disk_rows=0)
    # A store budget larger than needed does not pull rows off the device.
    assert _plan(total_rows=20, host_rows=5, vram_rows=9, store_rows=99) == placement
    # Rows are never dropped: a remainder with no tier left to hold it is
    # refused, and the disk tier is the tier that takes it when allowed.
    with pytest.raises(ValueError, match="does not fit"):
        _plan(total_rows=20, host_rows=5, vram_rows=9, store_rows=5)


def test_plan_ple_placement_falls_through_to_the_disk_tier() -> None:
    # 20 rows, 5 on the host, 9 on the device, 5 on the store card: 1 remains.
    assert _plan(
        total_rows=20, host_rows=5, vram_rows=9, store_rows=5, disk_allowed=True
    ) == PLEPlacement(vram_rows=9, host_rows=5, store_rows=5, disk_rows=1)
    # Without a store card the disk takes the whole remainder (two-card host).
    placement = _plan(
        total_rows=20, host_rows=5, vram_rows=9, store_rows=0, disk_allowed=True
    )
    assert placement == PLEPlacement(
        vram_rows=9, host_rows=5, store_rows=0, disk_rows=6
    )
    assert (placement.local_rows, placement.remote_rows) == (14, 6)
    # A device budget that holds the rest leaves the disk empty.
    assert (
        _plan(
            total_rows=20, host_rows=5, vram_rows=99, store_rows=0, disk_allowed=True
        ).disk_rows
        == 0
    )
    with pytest.raises(ValueError, match="device budget"):
        plan_ple_placement(
            total_rows=20,
            row_bytes=8,
            host_budget_bytes=0,
            vram_budget_bytes=-8,
            store_budget_bytes=0,
        )


def test_copy_ple_embedding_shard_tiers_matches_the_single_copy() -> None:
    checkpoint = torch.arange(20 * 4, dtype=torch.int8).view(20, 4)
    # TP range [5, 15) of a 20-row table, checkpoint shards of 6 rows; the
    # last three rows belong to a tier another process holds.
    reference = torch.zeros(10, 4, dtype=torch.int8)
    vram = torch.zeros(4, 4, dtype=torch.int8)
    host = torch.zeros(3, 4, dtype=torch.int8)
    copied = 0
    for shard_index in range(4):
        start = shard_index * 6
        shard = checkpoint[start : start + 6]
        copy_ple_embedding_shard_(
            reference, shard, checkpoint_start=start, tp_start=5, tp_end=15
        )
        copied += copy_ple_embedding_shard_tiers_(
            [(4, vram), (3, host), (3, None)],
            shard,
            checkpoint_start=start,
            tp_start=5,
            tp_end=15,
        )
    assert copied == 7
    torch.testing.assert_close(torch.cat([vram, host]), reference[:7])
    with pytest.raises(ValueError, match="do not cover"):
        copy_ple_embedding_shard_tiers_(
            [(4, vram), (3, host), (2, None)],
            checkpoint[:6],
            checkpoint_start=0,
            tp_start=5,
            tp_end=15,
        )


@pytest.mark.parametrize("host_rows", [0, 4, 8, 12])
def test_tier_copy_accepts_tp_padding(host_rows: int) -> None:
    checkpoint = torch.arange(20 * 4, dtype=torch.int8).view(20, 4)
    device = torch.full((12 - host_rows, 4), -1, dtype=torch.int8)
    host = torch.full((host_rows, 4), -1, dtype=torch.int8)
    count = copy_ple_embedding_shard_tiers_(
        [(12 - host_rows, device), (host_rows, host)],
        checkpoint,
        checkpoint_start=0,
        tp_start=7,
        tp_end=17,
    )
    result = torch.cat([device, host])
    assert count == 10
    torch.testing.assert_close(result[:10], checkpoint[7:17])
    assert torch.all(result[10:] == -1)


def test_worker_segments_split_the_store_card_from_the_disk() -> None:
    placements = [
        PLERemotePlacement(tp_start=0, tp_end=100, local_rows=60, store_rows=10),
        PLERemotePlacement(tp_start=100, tp_end=200, local_rows=90, store_rows=0),
    ]
    store, disk = plan_ple_worker_segments(placements)
    assert store == [PLEStoreSegment(start=60, end=70, offset=0)]
    assert disk == [
        PLEDiskSegment(start=70, end=100),
        PLEDiskSegment(start=190, end=200),
    ]

    ids = torch.tensor([[59, 60, 69], [70, 99, 100], [189, 190, 199]])
    expected = torch.tensor(
        [[False, False, False], [True, True, False], [False, True, True]]
    )
    assert torch.equal(ple_disk_mask(ids, disk), expected)
    assert not ple_disk_mask(ids, []).any()
    # Rows the worker serves split into the two outer tiers, store tier first.
    assert placements[0].disk_rows == 30
    assert placements[1].disk_rows == 10
    with pytest.raises(ValueError, match="exceeds the rows left"):
        PLERemotePlacement(tp_start=0, tp_end=100, local_rows=90, store_rows=11)


def test_store_segments_lay_the_ranks_remote_rows_end_to_end() -> None:
    placements = [
        PLERemotePlacement(tp_start=0, tp_end=100, local_rows=60, store_rows=40),
        PLERemotePlacement(tp_start=100, tp_end=200, local_rows=128),
        PLERemotePlacement(tp_start=200, tp_end=300, local_rows=30, store_rows=70),
    ]
    segments, disk = plan_ple_worker_segments(placements)
    assert disk == []
    assert segments == [
        PLEStoreSegment(start=60, end=100, offset=0),
        PLEStoreSegment(start=230, end=300, offset=40),
    ]
    ids = torch.tensor([[59, 61, 99], [100, 229, 230], [299, 0, 150]])
    expected = torch.tensor([[0, 1, 39], [0, 0, 40], [109, 0, 0]])
    assert torch.equal(ple_store_indices(ids, segments), expected)
    assert torch.equal(ple_store_indices(ids, []), torch.zeros_like(ids))
    with pytest.raises(ValueError, match="overlap"):
        plan_ple_worker_segments(
            [
                PLERemotePlacement(
                    tp_start=0, tp_end=100, local_rows=0, store_rows=100
                ),
                PLERemotePlacement(
                    tp_start=90, tp_end=200, local_rows=0, store_rows=110
                ),
            ]
        )


def test_auto_ple_host_budget_spills_only_the_shortfall() -> None:
    gib = 1024**3
    common = dict(
        device_total_bytes=48 * gib,
        gpu_memory_utilization=0.95,
        reserve_bytes=2 * gib,
    )
    # 45.6 usable - 20 weights - 10 KV - 2 reserve = 13.6 GiB room: a 10 GiB
    # table fits entirely, a 20 GiB table spills 6.4 GiB.
    assert (
        auto_ple_host_budget_bytes(
            table_bytes=10 * gib,
            device_allocated_bytes=20 * gib,
            kv_cache_bytes=10 * gib,
            **common,
        )
        == 0
    )
    spill = auto_ple_host_budget_bytes(
        table_bytes=20 * gib,
        device_allocated_bytes=20 * gib,
        kv_cache_bytes=10 * gib,
        **common,
    )
    assert spill == 20 * gib - (int(48 * gib * 0.95) - 32 * gib)
    # No room at all: the whole table goes to the host, never more.
    assert (
        auto_ple_host_budget_bytes(
            table_bytes=20 * gib,
            device_allocated_bytes=46 * gib,
            kv_cache_bytes=10 * gib,
            **common,
        )
        == 20 * gib
    )


def test_available_host_bytes_reads_meminfo() -> None:
    available = available_host_bytes()
    assert available is None or available > 0
    total = total_host_bytes()
    assert total is None or total >= (available or 0)


def test_cap_host_budget_shares_the_host_between_ranks() -> None:
    gib = 1024**3
    # 30 GB host, 20 GiB available, 7.5 GiB reserve, two ranks: 6.25 GiB each.
    # The 7.09 GiB that double-booked the host on 2026-09-06 is cut to that.
    share = cap_host_budget_bytes(
        budget_bytes=int(7.09 * gib),
        available_bytes=20 * gib,
        reserve_bytes=int(7.5 * gib),
        ranks_sharing_host=2,
    )
    assert share == int(12.5 * gib) // 2
    # A budget below the share passes untouched.
    assert (
        cap_host_budget_bytes(
            budget_bytes=2 * gib,
            available_bytes=20 * gib,
            reserve_bytes=int(7.5 * gib),
            ranks_sharing_host=2,
        )
        == 2 * gib
    )
    # Reserve swallows everything: nothing may be pinned.
    assert (
        cap_host_budget_bytes(
            budget_bytes=2 * gib,
            available_bytes=6 * gib,
            reserve_bytes=8 * gib,
            ranks_sharing_host=2,
        )
        == 0
    )
    with pytest.raises(ValueError):
        cap_host_budget_bytes(
            budget_bytes=gib, available_bytes=gib, reserve_bytes=0, ranks_sharing_host=0
        )


def test_ple_host_reserve_defaults_to_a_quarter_of_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.qwen4_exp.nvidia import ple_layer

    monkeypatch.setattr(ple_layer.envs, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    assert ple_layer.ple_host_reserve_bytes(30 * 1024**3) == int(7.5 * 1024**3)
    monkeypatch.setattr(ple_layer.envs, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", 4.0)
    assert ple_layer.ple_host_reserve_bytes(30 * 1024**3) == 4 * 1024**3
    monkeypatch.setattr(ple_layer.envs, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", -1.0)
    with pytest.raises(ValueError):
        ple_layer.ple_host_reserve_bytes(30 * 1024**3)


def test_post_load_context_keeps_marked_parameter_on_cpu() -> None:
    module = nn.Module()
    host_weight = nn.Parameter(torch.ones(2))
    host_weight._vllm_keep_on_cpu = True
    module.register_parameter("weight", host_weight)

    with device_loading_context(module, torch.device("meta")):
        assert module.weight.device.type == "cpu"

    assert module.weight.device.type == "cpu"


def test_ngram_embedding_accepts_checkpoint_seed_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 4
    )
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        eos_token_id=2,
        vocab_size=64,
        split_ngram_parts=2,
        seed=None,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=128,
        ple_embedding_dtype="float8_e4m3fn",
        ple_offload_embedding=False,
    )

    with torch.device("meta"):
        layer = Qwen4ExpNGramEmbedding(
            config,
            embedding_dim=256,
            ple_dense_layer_id=0,
            max_total_tokens=8,
            max_num_reqs=2,
            prefix="model.layers.2.ple.ple_embedding",
            layer_name="model.layers.2.ple",
            params_dtype=torch.float16,
        )

    assert layer.ngram_heads == 16
    assert layer.head_dim == 16
    assert layer.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert layer.ngram_embedding.weight.is_meta


def test_ngram_embedding_disk_offload_allocates_only_meta_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(ple_module.envs, "VLLM_PLE_DISK_OFFLOAD", True)
    monkeypatch.setattr(ple_module.envs, "VLLM_PLE_DISK_OFFLOAD_NUM_THREADS", 0)
    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        eos_token_id=2,
        vocab_size=64,
        split_ngram_parts=2,
        seed=None,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=128,
        ple_embedding_dtype="float8_e4m3fn",
        ple_offload_embedding=False,
    )

    layer = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=256,
        ple_dense_layer_id=0,
        max_total_tokens=8,
        max_num_reqs=2,
        prefix="model.layers.2.ple.ple_embedding",
        layer_name="model.layers.2.ple",
        params_dtype=torch.float16,
    )

    assert layer._disk_offload
    assert len(layer._disk_shards) == 2
    assert layer.ngram_embedding.weight.is_meta
    assert layer.positions_buffer.device.type == "cpu"


def _make_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.split_ngram_parts = 2
    module.register_buffer("layer_multipliers", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_offsets", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long))
    module.ngram_embedding = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=2,
        weight=nn.Parameter(torch.full((4, 2), -1.0)),
        shard_indices=SimpleNamespace(
            org_vocab_start_index=2,
            org_vocab_end_index=6,
        ),
    )
    return module


def _make_fp8_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
    module = _make_ngram_embedding_for_load_test()
    embedding = nn.Module()
    embedding.org_vocab_size = 8
    embedding.embedding_dim = 2
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=2,
        org_vocab_end_index=6,
    )
    embedding.register_parameter(
        "weight",
        nn.Parameter(
            torch.full((4, 2), -1.0).to(torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    embedding.register_parameter(
        "weight_scale",
        nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False),
    )
    module.ngram_embedding = embedding
    return module


def _make_disk_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
    module = _make_fp8_ngram_embedding_for_load_test()
    module._disk_offload = True
    module._file_backed_shards = True
    module._disk_shards = [None, None]
    module._disk_mapped_paths = set()
    module._disk_shard_size = 4
    module._disk_shard_boundaries = torch.tensor([4], dtype=torch.int64)
    module.head_dim = 2
    return module


def test_ple_shard_overlap_and_copy() -> None:
    overlap = compute_ple_shard_overlap(
        checkpoint_start=2, checkpoint_rows=5, tp_start=4, tp_end=8
    )
    assert overlap == PLEShardOverlap(source_start=2, destination_start=0, row_count=3)

    destination = torch.full((4, 2), -1.0)
    loaded = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    copied = copy_ple_embedding_shard_(
        destination,
        loaded,
        checkpoint_start=2,
        tp_start=4,
        tp_end=8,
    )

    assert copied == 3
    torch.testing.assert_close(destination[:3], loaded[2:5].float())
    torch.testing.assert_close(destination[3], torch.tensor([-1.0, -1.0]))


def test_ple_shard_copy_is_a_noop_without_overlap() -> None:
    destination = torch.ones(4, 2)
    copied = copy_ple_embedding_shard_(
        destination,
        torch.zeros(2, 2),
        checkpoint_start=10,
        tp_start=4,
        tp_end=8,
    )

    assert copied == 0
    assert torch.equal(destination, torch.ones_like(destination))


def test_ngram_embedding_loads_shards_and_ignores_legacy_token_lookup() -> None:
    module = _make_ngram_embedding_for_load_test()
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("token_lookup", torch.tensor([2, 1, 0])),
        ]
    )

    assert loaded == {"ngram_embedding.weight"}
    torch.testing.assert_close(
        module.ngram_embedding.weight,
        torch.cat((shard_0[2:4], shard_1[0:2])),
    )


def test_ngram_embedding_rejects_mismatched_checkpoint_shard() -> None:
    module = _make_ngram_embedding_for_load_test()

    with pytest.raises(
        ValueError,
        match=r"Shape mismatch for PLE embedding shard 0",
    ):
        module.load_weights([("ngram_embedding.shard_0.weight", torch.zeros(3, 2))])


def test_ngram_embedding_loads_fp8_shards_and_global_scale() -> None:
    module = _make_fp8_ngram_embedding_for_load_test()
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    shard_1 = (
        torch.arange(8, 16, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    )
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    assert module.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(
        module.ngram_embedding.weight.float(),
        torch.cat((shard_0[2:4], shard_1[0:2])).float(),
    )
    assert torch.equal(module.ngram_embedding.weight_scale, weight_scale)
    assert module.get_offload_output_dtype(torch.bfloat16) == torch.uint8


def test_ngram_embedding_retains_and_gathers_disk_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _make_disk_ngram_embedding_for_load_test()
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    shard_1 = (
        torch.arange(8, 16, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    )
    monkeypatch.setattr(
        ple_module,
        "_advise_random_file_access",
        lambda _: "/tmp/test-ple.safetensors",
    )

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.weight_scale", torch.tensor([0.25])),
        ]
    )
    output = torch.empty(4, 2, dtype=torch.uint8)
    ngram_ids = torch.tensor([[7], [0], [7], [2]], dtype=torch.int64)
    with ThreadPoolExecutor(max_workers=2) as executor:
        module._disk_executor = executor
        module._disk_embedding_lookup(ngram_ids, output)

    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    assert module._disk_shards[0] is shard_0
    assert module._disk_shards[1] is shard_1
    expected = torch.cat((shard_0, shard_1))[ngram_ids.reshape(-1)]
    assert torch.equal(output, expected.view(torch.uint8))


def test_ngram_embedding_disk_offload_rejects_missing_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _make_disk_ngram_embedding_for_load_test()
    monkeypatch.setattr(
        ple_module,
        "_advise_random_file_access",
        lambda _: "/tmp/test-ple.safetensors",
    )

    with pytest.raises(RuntimeError, match=r"did not load shards: \[1\]"):
        module.load_weights(
            [
                (
                    "ngram_embedding.shard_0.weight",
                    torch.zeros(4, 2).to(torch.float8_e4m3fn),
                )
            ]
        )


def test_ngram_gpu_offload_retains_only_fp8_global_scale(monkeypatch) -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module._offload_model_dtype = torch.float16
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)
    monkeypatch.setattr(ple_module.envs, "VLLM_PLE_CPU_OFFLOAD", True)
    monkeypatch.setattr(ple_module, "is_offload_process", lambda: False)
    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        lambda: torch.device("cpu"),
    )

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", torch.empty(4, 2)),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert loaded == {"ngram_embedding.weight_scale"}
    assert module._offload_weight_scale.dtype == torch.float16
    assert torch.equal(module._offload_weight_scale, weight_scale.to(torch.float16))
    assert module.get_offload_output_dtype(torch.bfloat16) == torch.uint8

    ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(ple_layer)
    ple_layer.ple_embedding = module
    embeddings = torch.tensor([[4.0, 8.0]]).to(torch.float8_e4m3fn)
    output = ple_layer._dequantize_embeddings(embeddings, torch.bfloat16)
    torch.testing.assert_close(
        output,
        torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
    )


def _make_fp8_embedding_layer(
    monkeypatch: pytest.MonkeyPatch,
    params_dtype: torch.dtype = torch.bfloat16,
) -> embedding_module.VocabParallelEmbedding:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        embedding_module,
        "tensor_model_parallel_all_reduce",
        lambda tensor: tensor,
    )
    method = Qwen4ExpPLEFp8EmbeddingMethod()
    layer = embedding_module.VocabParallelEmbedding(
        3,
        2,
        params_dtype=params_dtype,
        padding_size=1,
        quant_method=method,
    )
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    layer.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    layer.weight_scale.data.copy_(torch.tensor([0.25], dtype=params_dtype))
    return layer


def test_ple_fp8_embedding_scale_matches_model_dtype(monkeypatch) -> None:
    layer = _make_fp8_embedding_layer(monkeypatch, params_dtype=torch.float16)

    assert layer.weight_scale.dtype == torch.float16


def test_ple_fp8_embedding_dequantizes_in_ple_layer(monkeypatch) -> None:
    layer = _make_fp8_embedding_layer(monkeypatch)
    quantized_output = layer(torch.tensor([2, 0]))
    ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(ple_layer)
    ple_layer.ple_embedding = nn.Module()
    ple_layer.ple_embedding.ngram_embedding = layer

    output = ple_layer._dequantize_embeddings(
        quantized_output,
        torch.bfloat16,
    )

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.bfloat16
    assert quantized_output.dtype == torch.float8_e4m3fn
    assert output.dtype == torch.bfloat16
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    torch.testing.assert_close(output, (weight[[2, 0]] * 0.25).bfloat16())


def test_ple_fp8_embedding_uses_int8_for_tp_reduce(monkeypatch) -> None:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        embedding_module,
        "get_masked_input_and_mask",
        lambda *args: (
            torch.tensor([0, 0]),
            torch.tensor([False, True]),
        ),
    )
    reduced_dtypes = []

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced_dtypes.append(tensor.dtype)
        return tensor.clone()

    monkeypatch.setattr(
        embedding_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )
    layer = embedding_module.VocabParallelEmbedding(
        4,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        quant_method=Qwen4ExpPLEFp8EmbeddingMethod(),
    )
    layer.weight.data.copy_(
        torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e4m3fn)
    )

    output = layer(torch.tensor([0, 2]))

    assert reduced_dtypes == [torch.int8]
    assert output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(output[0].float(), layer.weight[0].float())
    assert torch.count_nonzero(output[1].float()) == 0


def test_ple_fp8_embedding_respects_checkpoint_shard_exclusions() -> None:
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=[],
        weight_block_size=[128, 128],
    )
    assert isinstance(
        _get_ple_embedding_quant_method(quant_config, prefix),
        Qwen4ExpPLEFp8EmbeddingMethod,
    )

    quant_config.ignored_layers = [f"{prefix}.shard_0"]
    assert _get_ple_embedding_quant_method(quant_config, prefix) is None


def test_ple_ngram_ids_custom_op_uses_current_request_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RuntimeNGramEmbedding(nn.Module):
        def compute_ngram_ids(
            self,
            input_ids: torch.Tensor,
            query_start_loc: torch.Tensor,
            ngram_context: torch.Tensor,
        ) -> torch.Tensor:
            del input_ids, ngram_context
            num_reqs = query_start_loc.numel() - 1
            return torch.full((4, 2), num_reqs, dtype=torch.long)

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embedding = RuntimeNGramEmbedding()
    monkeypatch.setattr(
        ple_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    input_ids = torch.arange(4)
    ngram_context = torch.zeros(2, 2, dtype=torch.long)
    output = torch.empty(4, 2, dtype=torch.long)

    ple_module.qwen4_exp_compute_ple_ngram_ids(
        input_ids,
        torch.tensor([0, 4]),
        ngram_context,
        output,
        "ple",
    )
    assert torch.equal(output, torch.ones_like(output))

    ple_module.qwen4_exp_compute_ple_ngram_ids(
        input_ids,
        torch.tensor([0, 2, 4]),
        ngram_context,
        output,
        "ple",
    )
    assert torch.equal(output, torch.full_like(output, 2))


def test_ngram_cpu_offload_padding_does_not_overwrite_real_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.embedding_dim = 1
    module.head_dim = 1
    module.ngram_size = 2
    module.heads_per_ngram = 1
    module.eos_token_id = 99
    module.register_buffer("positions_buffer", torch.arange(4))
    module.register_buffer("padded_buffer", torch.empty(1, 4, dtype=torch.long))
    module.register_buffer("layer_multipliers", torch.tensor([3, 5]))
    module.register_buffer("ngram_heads_vocab_sizes", torch.tensor([101]))
    module.register_buffer("ngram_heads_offsets", torch.tensor([0]))
    module.ngram_embedding = nn.Embedding(101, 1)
    module.ngram_embedding.weight.requires_grad_(False)
    with torch.no_grad():
        module.ngram_embedding.weight.copy_(torch.arange(101).reshape(-1, 1))

    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    query_start_loc = torch.tensor([0, 2])
    ngram_context = torch.full((1, 1), 99, dtype=torch.long)
    expected = module.forward_impl(
        torch.empty(2, 0),
        torch.tensor([11, 13]),
        query_start_loc,
        ngram_context,
        output_buffer=torch.empty(2, 1),
    )
    actual = module.forward_impl(
        torch.empty(4, 0),
        torch.tensor([11, 13, 777, 888]),
        query_start_loc,
        ngram_context,
        output_buffer=torch.empty(4, 1),
    )

    torch.testing.assert_close(actual[:2], expected)


def test_ngram_fp8_cpu_offload_preserves_quantized_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.embedding_dim = 2
    module.head_dim = 2
    module.ngram_size = 2
    module.heads_per_ngram = 1
    module.eos_token_id = 99
    module.register_buffer("positions_buffer", torch.arange(2))
    module.register_buffer("padded_buffer", torch.empty(1, 2, dtype=torch.long))
    module.register_buffer("layer_multipliers", torch.tensor([1, 1]))
    module.register_buffer("ngram_heads_vocab_sizes", torch.tensor([3]))
    module.register_buffer("ngram_heads_offsets", torch.tensor([0]))
    module.ngram_embedding = _make_fp8_embedding_layer(monkeypatch)

    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    hidden_states = torch.empty(2, 0)
    input_ids = torch.tensor([0, 1])
    query_start_loc = torch.tensor([0, 2])
    ngram_context = torch.tensor([[99]])
    quantized = module.forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
    )
    output_buffer = torch.empty(2, 2, dtype=torch.uint8)

    output = module.forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
        output_buffer=output_buffer,
    )

    assert output.data_ptr() == output_buffer.data_ptr()
    assert output.dtype == torch.uint8
    assert torch.equal(output, quantized.view(torch.uint8))


def test_dilated_ple_spec_state_rolls_back_before_next_forward() -> None:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = 6
    module.short_conv_dilation = 2

    conv_weights = torch.tensor([[0.25, -0.5, 0.75, 1.0]])
    conv_state = torch.zeros(2, 1, 9)
    conv_state[1] = torch.arange(1, 10, dtype=torch.float32).reshape(1, 9)
    first_inputs = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    initial_state = conv_state[1:].clone()
    first_history = torch.cat(
        (initial_state[..., : module.conv_state_len], first_inputs.T.unsqueeze(0)),
        dim=-1,
    )

    graph_padded_inputs = F.pad(first_inputs, (0, 0, 0, 4))
    first_output = module._short_conv_dilated_spec_batched(
        graph_padded_inputs,
        conv_state,
        conv_weights,
        torch.tensor([1, 0]),
        torch.tensor([0, 4, 4]),
        torch.tensor([1, 0]),
        spec_query_len=4,
    )

    expected_first_output = F.silu(
        F.conv1d(
            first_history,
            conv_weights.unsqueeze(1),
            groups=1,
            dilation=module.short_conv_dilation,
        )
    ).transpose(1, 2)[0]
    expected_first_state = first_history[..., 1:10]
    torch.testing.assert_close(first_output[:4], expected_first_output)
    assert torch.count_nonzero(first_output[4:]) == 0
    assert torch.count_nonzero(conv_state[0]) == 0
    torch.testing.assert_close(conv_state[1:], expected_first_state)

    second_inputs = torch.tensor([[50.0], [60.0]])
    rollback_state = expected_first_state[..., 1:7]
    padded_second_inputs = F.pad(second_inputs.T.unsqueeze(0), (0, 2))
    second_history = torch.cat((rollback_state, padded_second_inputs), dim=-1)
    expected_second_state = expected_first_state.clone()
    expected_second_state[..., :7] = second_history[..., 1:8]

    second_output = module._short_conv_dilated_spec_batched(
        second_inputs,
        conv_state,
        conv_weights,
        torch.tensor([1]),
        torch.tensor([0, 2]),
        torch.tensor([2]),
        spec_query_len=4,
    )

    expected_second_output = F.silu(
        F.conv1d(
            second_history,
            conv_weights.unsqueeze(1),
            groups=1,
            dilation=module.short_conv_dilation,
        )
    ).transpose(1, 2)[0, :2]
    torch.testing.assert_close(second_output, expected_second_output)
    torch.testing.assert_close(conv_state[1:], expected_second_state)


def test_ple_state_shape_reserves_speculative_tokens() -> None:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.hc_hidden_size = 32
    module.conv_state_len = 9
    module.num_spec_tokens = 3

    assert module.get_state_shape()[0] in ((32, 12), (12, 32))


# ---------------------------------------------------------------------------
# PP gate: the pipeline partition decides, not the pipeline size (#479)
# ---------------------------------------------------------------------------


def _text_config(ple_layer_ids, num_hidden_layers=48):
    return SimpleNamespace(
        ple_layer_ids=ple_layer_ids, num_hidden_layers=num_hidden_layers
    )


@pytest.mark.parametrize(
    ("ple_layer_ids", "pp_size", "partition"),
    [
        pytest.param([2], 1, None, id="pp1"),
        pytest.param([2], 2, None, id="pp2-even-split"),
        pytest.param([2], 2, "2,46", id="pp2-custom-split-rank0-holds-layer1"),
        pytest.param([2, 24], 2, None, id="pp2-two-ple-layers-on-rank0"),
    ],
)
def test_ple_pp_gate_accepts_ple_layers_on_first_rank(
    monkeypatch: pytest.MonkeyPatch, ple_layer_ids, pp_size, partition
):
    from vllm.models.qwen4_exp.common.ple import check_ple_layers_on_first_pp_rank

    if partition is None:
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    else:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)

    check_ple_layers_on_first_pp_rank(_text_config(ple_layer_ids), pp_size)


@pytest.mark.parametrize(
    ("ple_layer_ids", "pp_size", "partition", "misplaced"),
    [
        pytest.param([2, 30], 2, None, "[29]", id="pp2-even-split"),
        # ple_layer_ids are 1-based: id 2 is decoder layer 1, which a 1,47
        # split puts on the second stage.
        pytest.param([2], 2, "1,47", "[1]", id="pp2-custom-split-off-by-one"),
        pytest.param([2, 20, 40], 4, None, "[19, 39]", id="pp4-two-misplaced"),
    ],
)
def test_ple_pp_gate_rejects_ple_layers_beyond_first_rank(
    monkeypatch: pytest.MonkeyPatch, ple_layer_ids, pp_size, partition, misplaced
):
    from vllm.models.qwen4_exp.common.ple import check_ple_layers_on_first_pp_rank

    if partition is None:
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    else:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)

    with pytest.raises(RuntimeError, match=re.escape(f"decoder layers {misplaced}")):
        check_ple_layers_on_first_pp_rank(_text_config(ple_layer_ids), pp_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
@pytest.mark.parametrize("host_rows", [0, 3, 8])
def test_pinned_host_ple_merges_the_workers_rows_bit_identically(
    monkeypatch: pytest.MonkeyPatch,
    host_rows: int,
) -> None:
    _patch_tp(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr(
        embedding_module, "tensor_model_parallel_all_reduce", lambda tensor: tensor
    )
    monkeypatch.setattr(
        ple_module, "tensor_model_parallel_all_reduce", lambda tensor: tensor
    )
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: host_rows * 8)
    layer = _pinned_layer(num_embeddings=8)
    _expose_to_gather_op(monkeypatch, layer)
    # Twelve distinct rows: the rank holds the first eight across its device
    # and host tiers, the offload worker serves the remaining four.
    raw = torch.tensor(
        [0x00, 0x01, 0x08, 0x38, 0x7E, 0x80, 0xB8, 0xFE],
        dtype=torch.uint8,
    ).repeat(12, 1)
    raw = torch.stack([raw[i].roll(i) for i in range(12)])
    resident = raw[:8].view(torch.float8_e4m3fn)
    assert layer.load_shard(resident, checkpoint_start=0, tp_start=0, tp_end=8) == 8
    layer.weight_scale = nn.Parameter(
        torch.tensor([0.25], dtype=torch.float16, device="cuda"),
        requires_grad=False,
    )
    layer.prepare_accelerator_weight()
    assert layer.local_rows == 8

    ids = torch.tensor([0, 9, 3, 11, 8, 7], dtype=torch.int64, device="cuda")
    ids_cpu = ids.cpu()
    expected = (raw.view(torch.float8_e4m3fn).float() * 0.25).to(torch.float16)
    # The worker delivers one row per id; slots the rank serves itself carry
    # NaN bytes that the merge has to ignore.
    remote = raw[ids_cpu].clone()
    remote[ids_cpu < 8] = 0xFF
    remote = remote.cuda()
    output = layer(ids, remote_rows=remote)
    torch.accelerator.synchronize()
    assert output.dtype == torch.float16
    assert torch.equal(output.cpu(), expected[ids_cpu])

    # With every id resident the merge changes nothing against the plain path.
    resident_ids = ids.clamp_max(7)
    plain = layer(resident_ids)
    merged = layer(resident_ids, remote_rows=torch.full_like(remote, 0xFF))
    assert torch.equal(plain, merged)

    # Inside a CUDA graph the merge reads the worker's buffer at replay time.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(ids, remote_rows=remote)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = layer(ids, remote_rows=remote)
    remote_slots = ids_cpu >= 8
    for offset in range(1, 4):
        rolled = raw.roll(offset, dims=0)
        remote.copy_(rolled[ids_cpu])
        graph.replay()
        torch.accelerator.synchronize()
        rolled_expected = (rolled.view(torch.float8_e4m3fn).float() * 0.25).to(
            torch.float16
        )
        replayed = graph_output.cpu()
        assert torch.equal(
            replayed[remote_slots], rolled_expected[ids_cpu][remote_slots]
        )
        assert torch.equal(replayed[~remote_slots], expected[ids_cpu][~remote_slots])


def _make_cascade_worker_embedding(
    monkeypatch: pytest.MonkeyPatch,
    store_device: int | None = 3,
    disk: bool = False,
) -> Qwen4ExpNGramEmbedding:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    set_lazy_env(monkeypatch, "VLLM_PLE_DISK_OFFLOAD", None)
    set_lazy_env(
        monkeypatch,
        "VLLM_QWEN4EXP_PLE_STORE_DEVICE",
        None if store_device is None else str(store_device),
    )
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_DISK", "1" if disk else None)
    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        eos_token_id=2,
        vocab_size=64,
        split_ngram_parts=2,
        seed=None,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=128,
        ple_embedding_dtype="float8_e4m3fn",
        ple_offload_embedding=False,
    )
    return Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=256,
        ple_dense_layer_id=0,
        max_total_tokens=8,
        max_num_reqs=2,
        prefix="model.layers.2.ple.ple_embedding",
        layer_name="model.layers.2.ple",
        params_dtype=torch.float16,
    )


def test_ngram_embedding_cascade_worker_keeps_shards_file_backed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_cascade_worker_embedding(monkeypatch)

    assert Qwen4ExpNGramEmbedding.offload_keeps_local_tables()
    assert layer._file_backed_shards
    assert not layer._disk_offload
    assert layer._disk_executor is None
    assert len(layer._disk_shards) == 2
    assert layer.ngram_embedding.weight.is_meta
    assert layer.get_offload_output_dtype(torch.float16) == torch.uint8


def test_cascade_worker_binds_resident_placements_and_serves_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_cascade_worker_embedding(monkeypatch, store_device=3)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 4)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (5 * 2**30, 8 * 2**30)
    )
    output = torch.full((8, 256), 7, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="before the ranks registered"):
        layer._remote_lookup(
            torch.zeros(8, 16, dtype=torch.int64), output.view(torch.float8_e4m3fn)
        )
    with pytest.raises(TypeError, match="PLERemotePlacement"):
        layer.bind_remote_placements([object()])

    resident = PLERemotePlacement(tp_start=0, tp_end=100, local_rows=128)
    layer.bind_remote_placements([resident])
    assert layer._remote_placements == [resident]
    assert layer._store_table is None

    input_ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    ngram_context = torch.full((2, 2), 2, dtype=torch.int32)
    result = layer.forward_impl(
        input_ids, input_ids, query_start_loc, ngram_context, output_buffer=output
    )
    assert result.shape == (5, 256)
    assert not output[:5].any()
    assert output[5:].eq(7).all()

    # The store card is only touched when rows actually go there.
    on_store = PLERemotePlacement(tp_start=0, tp_end=100, local_rows=60, store_rows=40)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 3)
    with pytest.raises(ValueError, match="not a visible CUDA device"):
        layer.bind_remote_placements([on_store])
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 4)
    # 40 store rows of 16 bytes need 640 bytes on the store device.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (639, 8 * 2**30))
    with pytest.raises(RuntimeError, match="needs .* only"):
        layer.bind_remote_placements([on_store])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_one_worker_buffer_merges_into_every_rank_bit_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker gathers the store rows of both ranks into one buffer. Each
    # rank merges the slots of its own store segment and masks the ids of the
    # other rank, so the reduced output equals a lookup in the complete table.
    worker = _make_cascade_worker_embedding(monkeypatch, store_device=0)
    rows, dim = worker.ngram_embedding.org_vocab_size, worker.head_dim
    shard_size = worker._disk_shard_size
    raw = torch.randint(0, 255, (rows, dim), dtype=torch.uint8)
    raw[(raw & 0x78) == 0x78] = 0x11  # keep NaN/Inf codes out of the comparison
    worker._disk_shards = [
        raw[index * shard_size : (index + 1) * shard_size].view(torch.float8_e4m3fn)
        for index in range(len(worker._disk_shards))
    ]

    scale = torch.tensor([0.0371], dtype=torch.float16, device="cuda")
    monkeypatch.setattr(ple_module, "tensor_model_parallel_all_reduce", lambda t: t)
    ranks = []
    for tp_rank, host_rows in ((0, 5), (1, 0)):
        _patch_tp(monkeypatch, rank=tp_rank, world_size=2)
        _patch_cascade(
            monkeypatch,
            host_bytes=host_rows * dim,
            store_bytes=rows * dim,
            store_device=0,
        )
        layer = _pinned_layer(
            num_embeddings=rows,
            embedding_dim=dim,
            prefix=f"model.layers.2.ple.rank{tp_rank}.ngram_embedding",
        )
        # Half of the rank's rows fit on the device.
        _patch_device_spill(monkeypatch, layer, rows // 4 * dim)
        for shard_index, shard in enumerate(worker._disk_shards):
            layer.load_shard(
                shard,
                checkpoint_start=shard_index * shard_size,
                tp_start=layer.shard_indices.org_vocab_start_index,
                tp_end=layer.shard_indices.org_vocab_end_index,
            )
        layer.weight_scale = nn.Parameter(scale.clone(), requires_grad=False)
        layer.prepare_accelerator_weight()
        assert layer._device_rows == rows // 4 and layer._store_rows > 0
        ranks.append(layer)
    _expose_to_gather_op(monkeypatch, *ranks)

    worker.bind_remote_placements(
        [
            PLERemotePlacement(
                tp_start=layer.shard_indices.org_vocab_start_index,
                tp_end=layer.shard_indices.org_vocab_end_index,
                local_rows=layer.local_rows,
                store_rows=layer.store_rows,
            )
            for layer in ranks
        ]
    )
    assert worker._store_table is not None
    assert worker._store_table.shape == (sum(r._store_rows for r in ranks), dim)

    table = (raw.view(torch.float8_e4m3fn).float() * scale.float().cpu()).to(
        torch.float16
    )
    for tokens in (1, 5, 37):
        ids = torch.randint(0, rows, (tokens, worker.ngram_heads))
        output = torch.full((tokens, worker.embedding_dim), 0x7F, dtype=torch.uint8)
        worker._remote_lookup(ids, output.view(torch.float8_e4m3fn))
        remote = output.cuda()
        ids_cuda = ids.cuda()
        reduced = ranks[0](ids_cuda, remote_rows=remote) + ranks[1](
            ids_cuda, remote_rows=remote
        )
        assert torch.equal(reduced.cpu(), table[ids])


def test_remote_placement_counts_the_rows_left_to_the_worker() -> None:
    placement = PLERemotePlacement(tp_start=100, tp_end=250, local_rows=100)
    assert placement.remote_rows == 50
    assert PLERemotePlacement(tp_start=0, tp_end=100, local_rows=128).remote_rows == 0
    with pytest.raises(ValueError, match="TP vocabulary range"):
        PLERemotePlacement(tp_start=10, tp_end=5, local_rows=0)
    with pytest.raises(ValueError, match="local_rows"):
        PLERemotePlacement(tp_start=0, tp_end=5, local_rows=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_cascade_rank_reports_its_resident_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tp(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: 3 * 8)
    embedding = _pinned_layer(num_embeddings=8)
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module._store_device = None
    module._cascade = False
    module.ngram_embedding = embedding
    assert module.remote_placement() is None

    module._store_device = 4
    module._cascade = True
    assert module.remote_placement() == PLERemotePlacement(
        tp_start=0, tp_end=8, local_rows=8, store_rows=0
    )
    assert (embedding._device_rows, embedding._host_rows) == (5, 3)

    module.ngram_embedding = nn.Module()
    with pytest.raises(RuntimeError, match="split device/host table"):
        module.remote_placement()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_pinned_host_ple_merge_stays_bit_identical_under_inductor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The resident gather writes the host tier and then the worker's rows into
    # the same output view. A wrong ordering of those two copies under
    # Inductor would lose host rows silently; eager tests cannot see it.
    rows, host_rows, dim, heads = 512, 128, 8, 16
    _patch_tp(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr(
        embedding_module, "tensor_model_parallel_all_reduce", lambda tensor: tensor
    )
    monkeypatch.setattr(
        ple_module, "tensor_model_parallel_all_reduce", lambda tensor: tensor
    )
    monkeypatch.setattr(ple_module, "ple_host_budget_bytes", lambda: host_rows * dim)
    layer = _pinned_layer(num_embeddings=rows, embedding_dim=dim)
    _expose_to_gather_op(monkeypatch, layer)
    raw = torch.randint(0, 255, (rows, dim), dtype=torch.uint8)
    raw[(raw & 0x78) == 0x78] = 0x11  # keep NaN/Inf codes out of the comparison
    layer.load_shard(
        raw.view(torch.float8_e4m3fn), checkpoint_start=0, tp_start=0, tp_end=rows
    )
    layer.weight_scale = nn.Parameter(
        torch.tensor([0.0371], dtype=torch.float16, device="cuda"),
        requires_grad=False,
    )
    layer.prepare_accelerator_weight()
    assert layer._host_rows == host_rows

    def merged(ids: torch.Tensor, remote: torch.Tensor) -> torch.Tensor:
        return layer(ids, remote_rows=remote)

    compiled = torch.compile(merged, dynamic=True, fullgraph=True)
    table = (raw.view(torch.float8_e4m3fn).float() * 0.0371).to(torch.float16).cuda()
    for tokens in (5, 333):
        ids = torch.randint(0, rows, (tokens, heads), device="cuda")
        remote = torch.full(
            (tokens, heads * dim), 0x7F, dtype=torch.uint8, device="cuda"
        )
        assert torch.equal(compiled(ids, remote), table[ids])


def _fill_worker_shards(layer: Qwen4ExpNGramEmbedding) -> torch.Tensor:
    """Give the worker mapped-looking shards and return the whole raw table."""
    rows, dim = layer.ngram_embedding.org_vocab_size, layer.head_dim
    raw = torch.randint(0, 255, (rows, dim), dtype=torch.uint8)
    raw[(raw & 0x78) == 0x78] = 0x11  # keep NaN/Inf codes out of the comparison
    shard_size = layer._disk_shard_size
    layer._disk_shards = [
        raw[index * shard_size : (index + 1) * shard_size].view(torch.float8_e4m3fn)
        for index in range(len(layer._disk_shards))
    ]
    return raw


def test_cascade_worker_reads_the_disk_tier_from_the_mapped_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A host without a spare card: everything beyond the resident tiers is
    # read from the checkpoint, and no store card is opened at all.
    layer = _make_cascade_worker_embedding(monkeypatch, store_device=None, disk=True)
    raw = _fill_worker_shards(layer)
    rows, dim, heads = raw.shape[0], layer.head_dim, layer.ngram_heads
    resident = rows // 2

    layer.bind_remote_placements(
        [PLERemotePlacement(tp_start=0, tp_end=rows, local_rows=resident)]
    )
    assert layer._store_table is None
    assert layer._store_segments == []
    assert layer._disk_segments == [PLEDiskSegment(start=resident, end=rows)]

    tokens = 9
    ids = torch.randint(0, rows, (tokens, heads))
    output = torch.full((tokens, layer.embedding_dim), 0x7F, dtype=torch.uint8)
    layer._remote_lookup(ids, output.view(torch.float8_e4m3fn))

    served = output.view(-1, dim)
    on_disk = ids.reshape(-1) >= resident
    assert torch.equal(served[on_disk], raw[ids.reshape(-1)][on_disk])
    # Rows the ranks hold themselves are not read, and not written either.
    assert served[~on_disk].eq(0x7F).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA store card")
def test_cascade_worker_serves_store_card_and_disk_in_one_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_cascade_worker_embedding(monkeypatch, store_device=0, disk=True)
    raw = _fill_worker_shards(layer)
    rows, dim, heads = raw.shape[0], layer.head_dim, layer.ngram_heads
    resident, store_rows = rows // 2, rows // 4

    layer.bind_remote_placements(
        [
            PLERemotePlacement(
                tp_start=0, tp_end=rows, local_rows=resident, store_rows=store_rows
            )
        ]
    )
    assert layer._store_table is not None
    assert layer._store_table.shape == (store_rows, dim)
    assert layer._disk_segments == [
        PLEDiskSegment(start=resident + store_rows, end=rows)
    ]

    ids = torch.randint(0, rows, (11, heads))
    output = torch.full((11, layer.embedding_dim), 0x7F, dtype=torch.uint8)
    layer._remote_lookup(ids, output.view(torch.float8_e4m3fn))

    flat = ids.reshape(-1)
    served = output.view(-1, dim)
    remote = flat >= resident
    # Both outer tiers land in the same buffer, each row from its own tier.
    assert torch.equal(served[remote], raw[flat][remote])
