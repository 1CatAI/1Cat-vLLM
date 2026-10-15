# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Both indexer writers invalidate keys even when MTP skips selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import indexer_qsa
from vllm.models.qwen4_exp.nvidia.ops import qsa


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "skip_topk, select", [(False, True), (False, False), (True, True)]
)
def test_compressed_writes_precede_version_and_selection(
    monkeypatch, fused, skip_topk, select
):
    import flashinfer.norm

    indexer = indexer_qsa.QSAIndexer.__new__(indexer_qsa.QSAIndexer)
    torch.nn.Module.__init__(indexer)
    events = []
    slots = torch.tensor([-1, 0, -1, 2], dtype=torch.int64)
    raw = SimpleNamespace(
        num_actual_tokens=4,
        slot_mapping=slots,
        block_table=None,
        query_start_loc=None,
        logical_positions=torch.arange(4),
        token_to_req=None,
    )
    compressed = SimpleNamespace(slot_mapping=slots, k_work_metadata=None)
    indexer._metadata = lambda: (raw, compressed)
    indexer.index_qk_proj = SimpleNamespace(input_size=1)
    indexer.index_n_heads, indexer.index_kv_heads, indexer.index_head_dim = 4, 1, 128
    indexer.compress_ratio = 4
    indexer.use_fused_pre_indexer = fused
    indexer.skip_topk = skip_topk
    cache = torch.zeros(1, 4, 1, 128)
    indexer.raw_key_cache = SimpleNamespace(
        kv_cache=cache.clone(),
        key_cache=cache.clone(),
        rope_position_cache=None,
    )
    indexer.compressed_key_cache = SimpleNamespace(kv_cache=cache)
    indexer.rotary_emb = SimpleNamespace(cos_sin_cache=None)
    norm = SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6)
    indexer.q_layernorm, indexer.k_layernorm = norm, norm

    def fused_write(*args, **kwargs):
        assert args[13] is cache and args[14] is slots
        events.append("write")

    def store(target, mapping, rows):
        if target is cache:
            assert mapping is slots
            events.append("write")

    def touch(mapping):
        torch.testing.assert_close(mapping, slots, rtol=0, atol=0)
        events.append("version")

    output = torch.empty(4, 2051, dtype=torch.int32)

    def selection(*args):
        events.append("select")
        return output

    indexer._persistent_indexer_stage = SimpleNamespace(touch=touch)
    indexer._select_tokens = selection
    monkeypatch.setattr(indexer_qsa, "qsa_pre_indexer", fused_write)
    monkeypatch.setattr(qsa, "qsa_store_cache_rows", store)
    monkeypatch.setattr(
        qsa,
        "qsa_compress_groups_with_ratio",
        lambda *a: (torch.zeros(4, 1, 128), torch.zeros(4, 3, dtype=torch.int64)),
    )
    monkeypatch.setattr(flashinfer.norm, "gemma_rmsnorm", lambda x, *a: x)
    monkeypatch.setattr(indexer_qsa, "apply_qsa_rope", lambda rope, pos, x: x)
    assert indexer(torch.zeros(4, 640), torch.arange(4), output, select) is output
    expected = ["write", "version"]
    if select and not skip_topk:
        expected.append("select")
    assert events == expected
