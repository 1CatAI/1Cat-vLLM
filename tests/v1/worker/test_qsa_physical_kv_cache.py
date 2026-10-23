# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup


@pytest.mark.parametrize("stored_rows", [32, 64, 128, 204, 208, 288])
@pytest.mark.parametrize("padded", [False, True])
def test_qsa_physical_pages_match_in_both_worker_entry_points(stored_rows, padded):
    spec = MLAAttentionSpec(
        block_size=stored_rows * 4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        dcp_sharded=False,
        page_size_padded=stored_rows * 128 * 2 + 128 if padded else None,
    )
    name = "model.layers.3.self_attn.indexer.compressed_key_cache"
    group = AttentionGroup(QSAStateBackend, [name], spec, 0)
    runner = object.__new__(GPUModelRunner)
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([name], spec)],
    )
    runner.attn_groups = [[group]]
    runner.runner_only_attn_layers = set()
    runner.cache_config = SimpleNamespace(cache_dtype="float16")
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8)
    legacy = runner._reshape_kv_cache_tensors({name: raw}, [32])[name]
    modern = _reshape_kv_cache([group], {name: raw}, "float16", [32], {})[name]
    assert legacy.shape == modern.shape == (3, stored_rows, 1, 128)
    assert legacy.data_ptr() == modern.data_ptr() == raw.data_ptr()
    # Physical block ids must keep the same rows and cannot write a neighbour.
    legacy[1].fill_(7)
    assert torch.all(modern[1] == 7)
    assert torch.count_nonzero(modern[0]) == torch.count_nonzero(modern[2]) == 0
    modern[2, -1, 0, -1] = 11
    assert legacy[2, -1, 0, -1] == 11
    if padded:
        assert torch.count_nonzero(raw.reshape(3, -1)[:, stored_rows * 128 * 2 :]) == 0
