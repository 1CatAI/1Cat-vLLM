# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM70 sparse attention policy is read when the layers are built.

The worker runs the memory profile and execute_model outside a
set_current_vllm_config() context, so the forward pass must not look the
policy up there. These tests deliberately run without a config context.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.config.kernel import Sm70SparseConfig
from vllm.config.vllm import get_current_vllm_config


@pytest.mark.parametrize("prefill_bmm", [True, False])
def test_prefill_bmm_follows_the_layer_policy_without_config_context(prefill_bmm):
    from vllm.models.deepseek_v4.sm70 import sparse

    with pytest.raises(AssertionError, match="config is not set"):
        get_current_vllm_config()
    layer = SimpleNamespace(
        max_num_batched_tokens=2048,
        sm70_sparse=Sm70SparseConfig(prefill_bmm=prefill_bmm),
    )
    q = torch.empty((1, 64, 512), dtype=torch.float16)
    with (
        patch.object(sparse.current_platform, "is_cuda", return_value=True),
        patch.object(
            sparse.current_platform, "is_device_capability_family", return_value=True
        ),
    ):
        specs = sparse.DeepseekV4SM70SparseImpl._prefill_bmm_workspace_specs(
            layer, q, 640
        )
    assert bool(specs) is prefill_bmm


def test_indexer_caches_keep_the_engine_policy():
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache

    config = VllmConfig()
    config.kernel_config.sm70_sparse.indexer_decode_cublas = False
    with set_current_vllm_config(config):
        caches = [
            DeepseekV32IndexerCache(128, torch.uint8, "v32.k_cache", CacheConfig()),
            DeepseekV4IndexerCache(
                128, torch.uint8, "v4.k_cache", CacheConfig(), compress_ratio=4
            ),
        ]
    for cache in caches:
        assert cache.sm70_sparse is config.kernel_config.sm70_sparse
        assert not cache.sm70_sparse.indexer_decode_cublas
