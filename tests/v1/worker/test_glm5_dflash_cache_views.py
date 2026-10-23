# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct real PP7 cache layouts on CPU and check their byte boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.models.glm5next.nvidia.attention import Glm5NextTailCache
from vllm.models.glm5next.sm70.sparse import Glm5NextSM70SparseBackend
from vllm.v1.attention.backends import utils as backend_utils
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.core import kv_cache_utils as kv
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.attn_utils import _allocate_kv_cache, _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes

pytestmark = pytest.mark.cpu_test
PARTITION = (9, 5, 5, 2, 7, 7, 10)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _production_groups(monkeypatch, num_spec):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, PARTITION)))
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=8192, get_total_num_hidden_layers=lambda: 45, is_hybrid=True
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=7),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            num_gpu_blocks_override=None,
            mamba_ssm_cache_dtype="float32",
            mamba_cache_dtype="auto",
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=512, disable_hybrid_kv_cache_manager=False
        ),
        speculative_config=(
            SimpleNamespace(use_dflash=lambda: True, num_speculative_tokens=num_spec)
            if num_spec
            else None
        ),
    )
    q, k, v, recurrent = MambaStateShapeCalculator.kda_state_shape(
        1, 64, 128, conv_kernel_size=4, num_spec=num_spec
    )
    conv_dtype, _, _, recurrent_dtype = MambaStateDtypeCalculator.kda_state_dtype(
        torch.float16, "auto"
    )
    block = 4608 if num_spec else 4352
    state = MambaSpec(
        shapes=((q[0], q[1] + k[1] + v[1]), recurrent),
        dtypes=(conv_dtype, recurrent_dtype),
        block_size=block,
        page_size_padded=block * 1024,
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
    )
    workers = []
    start = 0
    for rank, count in enumerate(PARTITION):
        specs = {}
        for layer in range(start, start + count):
            prefix = f"language_model.model.layers.{layer}.self_attn"
            if layer % 4 != 3:
                specs[prefix] = state
            else:
                specs[prefix + ".mla_attn"] = MLAAttentionSpec(
                    block_size=block, num_kv_heads=1, head_size=512, dtype=torch.float16
                )
                specs[prefix + ".indexer.k_cache"] = MLAAttentionSpec(
                    block_size=block,
                    num_kv_heads=1,
                    head_size=132,
                    dtype=torch.uint8,
                    compress_ratio=4,
                )
                specs[prefix + ".indexer.tail_cache"] = (
                    Glm5NextTailCache.get_kv_cache_spec(
                        SimpleNamespace(
                            _index_kpool=4, head_dim=128, dtype=torch.float16
                        ),
                        cfg,
                    )
                )
        if num_spec and rank == 6:
            for layer in range(5):
                specs[f"speculator.model.layers.{layer}.self_attn.attn"] = (
                    SlidingWindowSpec(
                        block_size=block,
                        num_kv_heads=8,
                        head_size=128,
                        head_size_v=128,
                        dtype=torch.float16,
                        sliding_window=2048,
                    )
                )
        workers.append(specs)
        start += count
    groups = kv.get_kv_cache_groups(cfg, {n: s for w in workers for n, s in w.items()})
    return cfg, groups, workers


def _attention_groups(groups):
    result = []
    for gid, group in enumerate(groups):
        entries = []
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if isinstance(spec, MambaSpec):
                backend = GDNAttentionBackend
            elif isinstance(spec, KpoolTailSpec):
                backend = KpoolTailBackend
            elif isinstance(spec, MLAAttentionSpec):
                backend = (
                    DeepseekV32IndexerBackend
                    if spec.compress_ratio > 1
                    else Glm5NextSM70SparseBackend
                )
            else:
                backend = FlashAttentionBackend
            entries.append(AttentionGroup(backend, [name], spec, gid))
        result.append(entries)
    return result


@pytest.mark.parametrize(
    "rank,num_spec,layout",
    [(rank, num_spec, "NHD") for rank in range(7) for num_spec in (0, 7)]
    + [(6, 7, "HND")],
)
def test_real_pp7_pool_views_preserve_pages_and_shared_ownership(
    monkeypatch, rank, num_spec, layout
):
    monkeypatch.setattr(backend_utils, "_KV_CACHE_LAYOUT_OVERRIDE", layout)
    cfg, global_groups, workers = _production_groups(monkeypatch, num_spec)
    groups = kv._project_kv_cache_groups_to_worker(global_groups, workers[rank])
    pool_bytes = kv._pool_bytes_per_block(groups)
    cache_config = kv.get_kv_cache_config_from_groups(cfg, groups, pool_bytes * 3)
    attn_groups = _attention_groups(groups)
    with set_current_vllm_config(cfg):
        kernel_sizes = prepare_kernel_block_sizes(cache_config, attn_groups)
    flat_groups = [entry for group in attn_groups for entry in group]
    raw = _allocate_kv_cache(cache_config, {}, torch.device("cpu"))
    caches = _reshape_kv_cache(flat_groups, raw, "auto", kernel_sizes, {})
    assert set(caches) == set(workers[rank])
    assert sum(t.size for t in cache_config.kv_cache_tensors) == 3 * pool_bytes
    for tensor in cache_config.kv_cache_tensors:
        assert len({raw[n].data_ptr() for n in tensor.shared_by}) == 1
    # Every layer writes all of its payload in pool block 1. Pool blocks 0/2
    # and padding in block 1 must stay untouched, including for shared owners.
    for group in flat_groups:
        name = group.layer_names[0]
        spec = group.kv_cache_spec
        if isinstance(spec, KpoolTailSpec):
            assert spec.block_size == (12 if num_spec else 4)
            assert spec.real_page_size_bytes == (6144 if num_spec else 2048)
            assert spec.real_page_size_bytes < spec.page_size_bytes
        backing = raw[name]
        backing.zero_()
        cache = caches[name]
        views = cache if isinstance(cache, list) else [cache]
        for view in views:
            assert view.untyped_storage().data_ptr() == backing.data_ptr()
            subblocks = view.shape[0] // 3
            assert subblocks > 0 and view.shape[0] == 3 * subblocks
            assert (
                view.stride(0) * subblocks * view.element_size() == spec.page_size_bytes
            )
            view[subblocks : 2 * subblocks].view(torch.uint8).fill_(0xA5)
        pages = backing.view(torch.uint8).view(3, spec.page_size_bytes)
        assert not pages[0].any() and not pages[2].any()
        assert (pages[1, : spec.real_page_size_bytes] == 0xA5).all()
        assert not pages[1, spec.real_page_size_bytes :].any()
        if name.startswith("speculator."):
            assert spec.page_size_padded == spec.real_page_size_bytes == 4718592
            assert kernel_sizes[group.kv_cache_group_id] == 64
            assert cache.shape == (54, 2, 64, 8, 128)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_explicit_equal_page_size_has_identical_virtual_layout(
    monkeypatch, compressed, layout
):
    monkeypatch.setattr(backend_utils, "_KV_CACHE_LAYOUT_OVERRIDE", layout)
    if compressed:
        spec = MLAAttentionSpec(
            block_size=1152,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            compress_ratio=4,
        )
        backend = DeepseekV32IndexerBackend
    else:
        spec = SlidingWindowSpec(
            block_size=1152,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.float16,
            sliding_window=2048,
        )
        backend = FlashAttentionBackend
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    views = []
    for candidate in (spec, replace(spec, page_size_padded=spec.real_page_size_bytes)):
        group = AttentionGroup(backend, ["layer"], candidate, 0)
        views.append(
            _reshape_kv_cache([group], {"layer": raw}, "auto", [64], {})["layer"]
        )
    assert views[0].shape == views[1].shape
    assert views[0].stride() == views[1].stride()
    assert views[0].data_ptr() == views[1].data_ptr()
    views[0][-1].fill_(7)
    assert torch.equal(views[0], views[1])


@pytest.mark.parametrize("compressed", [False, True])
def test_real_padding_with_virtual_split_remains_rejected(compressed):
    if compressed:
        spec = MLAAttentionSpec(
            block_size=1152,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            compress_ratio=4,
        )
        backend = DeepseekV32IndexerBackend
    else:
        spec = SlidingWindowSpec(
            block_size=1152,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.float16,
            sliding_window=2048,
        )
        backend = FlashAttentionBackend
    spec = replace(spec, page_size_padded=spec.real_page_size_bytes + 128)
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    group = AttentionGroup(backend, ["layer"], spec, 0)
    with pytest.raises(ValueError, match="padded physical pages"):
        _reshape_kv_cache([group], {"layer": raw}, "auto", [64], {})


def test_tail_history_cannot_exceed_shared_physical_page(monkeypatch):
    cfg, _, workers = _production_groups(monkeypatch, 7)
    specs = {name: spec for worker in workers for name, spec in worker.items()}
    for name, spec in list(specs.items()):
        if isinstance(spec, KpoolTailSpec):
            specs[name] = replace(spec, block_size=300, sliding_window=300)
    with pytest.raises(ValueError, match="tail history exceeds"):
        kv.get_kv_cache_groups(cfg, specs)
