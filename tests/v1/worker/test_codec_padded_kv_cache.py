# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Both runner owners preserve data and scales across physical page padding."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.flash_v100.backend import FlashAttnV100Backend
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import FP16, INT8
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _cpu_engine_config():
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


@pytest.mark.parametrize("runner_version", [1, 2])
@pytest.mark.parametrize("layer_name,codec", [("target", INT8), ("draft", FP16)])
@pytest.mark.parametrize("configured_dtype", ["auto", "fp8", "int8"])
def test_padded_cache_regions_and_restore(
    runner_version, layer_name, codec, configured_dtype
):
    specs = unify_kv_cache_spec_page_size(
        {
            "target": FullAttentionSpec(
                block_size=2048,
                num_kv_heads=1,
                head_size=256,
                dtype=torch.int8,
                kv_quant_mode=get_kv_quant_mode("int8"),
            ),
            "draft": SlidingWindowSpec(
                block_size=1024,
                num_kv_heads=2,
                head_size=128,
                dtype=torch.float16,
                sliding_window=2048,
            ),
        }
    )
    spec = specs[layer_name]
    group = AttentionGroup(
        backend=FlashAttnV100Backend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    raw = torch.full((3 * spec.page_size_bytes,), 77, dtype=torch.int8)

    def bind(storage):
        if runner_version == 2:
            return _reshape_kv_cache(
                [group], {layer_name: storage}, configured_dtype, [spec.block_size], {}
            )[layer_name]
        runner = object.__new__(GPUModelRunner)
        runner.kv_cache_config = KVCacheConfig(
            num_blocks=3,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec([layer_name], spec)],
        )
        runner.attn_groups = [[group]]
        runner.runner_only_attn_layers = set()
        runner.cache_config = SimpleNamespace(cache_dtype=configured_dtype)
        return runner._reshape_kv_cache_tensors(
            {layer_name: storage}, [spec.block_size]
        )[layer_name]

    view = KVCacheView.bind(bind(raw), codec, spec.head_size)
    assert view.page_stride_bytes == spec.page_size_bytes
    assert view.key.untyped_storage().data_ptr() == raw.data_ptr()
    for page in range(3):
        view.key[page].fill_(page + 1)
        view.value[page].fill_(page + 4)
        if view.key_scale is not None:
            assert view.value_scale is not None
            view.key_scale[page].fill_(0.125 * (page + 1))
            view.value_scale[page].fill_(0.25 * (page + 1))
        gap = raw[
            page * spec.page_size_bytes + view.page_size_bytes : (page + 1)
            * spec.page_size_bytes
        ]
        assert torch.all(gap == 77)

    restored = raw.clone()
    rebound = KVCacheView.bind(bind(restored), codec, spec.head_size)
    assert torch.equal(raw, restored), "Binding must not overwrite restored scales"
    for name in ("key", "value", "key_scale", "value_scale"):
        before, after = getattr(view, name), getattr(rebound, name)
        if before is not None:
            assert torch.equal(before, after)
