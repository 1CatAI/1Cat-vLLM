# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.mamba_utils import MambaCopyBuffers, collect_mamba_copy_meta


def test_preprocess_uses_published_state_but_shifts_conv_history():
    conv = torch.arange(4 * 10 * 16, dtype=torch.float32).view(4, 10, 16)
    state = torch.arange(4 * 64, dtype=torch.float32).view(4, 64)
    spec = MambaSpec(
        block_size=256,
        shapes=((10, 16), (64,)),
        dtypes=(torch.float32, torch.float32),
        mamba_cache_mode="align",
        gdn_wy=True,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["gdn"], kv_cache_spec=spec)],
    )

    def buffer(n, dtype):
        arr = np.zeros(n, dtype=np.int64 if dtype == torch.int64 else np.int32)
        return SimpleNamespace(np=arr, cpu=torch.from_numpy(arr))

    functions = (get_conv_copy_spec, get_temporal_copy_spec)
    copies = MambaCopyBuffers.create(1, config, functions, buffer)
    collect_mamba_copy_meta(
        copies,
        config,
        functions,
        [0],
        0,
        1,
        3,
        SimpleNamespace(block_ids=[[0, 1, 2, 3]]),
        {"gdn": SimpleNamespace(kv_cache=(conv, state))},
        3,
    )
    assert copies.offset == 2
    assert copies.src_ptrs.np[0] == conv[0, 3:].data_ptr()
    assert copies.src_ptrs.np[1] == state[0].data_ptr()
    assert copies.dst_ptrs.np[1] == state[1].data_ptr()
    assert copies.sizes.np[0] == conv[0, 3:].numel() * conv.element_size()
    assert copies.sizes.np[1] == state[0].numel() * state.element_size()


def test_memory_admission_removes_seven_running_pages():
    spec = MambaSpec(
        block_size=256,
        shapes=((10, 2560), (12, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        mamba_cache_mode="align",
        num_speculative_blocks=7,
    )
    collapsed = replace(spec, gdn_wy=True, num_speculative_blocks=0)
    cfg = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    assert (
        spec.max_memory_usage_bytes(cfg) - collapsed.max_memory_usage_bytes(cfg)
        == 7 * spec.page_size_bytes
    )
    assert (
        collapsed.shapes == spec.shapes
    )  # prefill and convolution layouts remain valid
