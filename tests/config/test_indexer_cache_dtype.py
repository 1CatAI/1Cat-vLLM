# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest
from pydantic import TypeAdapter

from vllm.config.attention import AttentionConfig
from vllm.config.utils import replace


def test_indexer_selection_survives_worker_copy_and_hashes_effective_storage():
    implicit = AttentionConfig()
    explicit_fp8 = replace(implicit, indexer_cache_dtype="fp8")
    int8 = replace(implicit, indexer_cache_dtype="int8")
    assert implicit.resolved_indexer_cache_dtype == "fp8"
    assert implicit.compute_hash() == explicit_fp8.compute_hash()
    assert implicit.compute_hash() != int8.compute_hash()
    adapter = TypeAdapter(AttentionConfig)
    # API inputs contain configured fields, not derived native-policy state.
    parsed = adapter.validate_json(
        adapter.dump_json(int8, include={"indexer_cache_dtype"})
    )
    assert parsed.compute_hash() == int8.compute_hash()
    worker = deepcopy(parsed)
    assert worker.resolved_indexer_cache_dtype == "int8"
    assert worker.compute_hash() == int8.compute_hash()
    worker.indexer_cache_dtype = "fp8"
    assert int8.indexer_cache_dtype == "int8"
    assert implicit.indexer_cache_dtype == "auto"


@pytest.mark.parametrize("dtype", ["fp8", "int8"])
def test_conflicting_indexer_storage_is_rejected(dtype):
    with pytest.raises(ValueError, match="conflicts"):
        AttentionConfig(use_fp4_indexer_cache=True, indexer_cache_dtype=dtype)


def test_legacy_fp4_auto_keeps_its_effective_storage():
    config = AttentionConfig(use_fp4_indexer_cache=True)
    assert config.resolved_indexer_cache_dtype == "fp4"
