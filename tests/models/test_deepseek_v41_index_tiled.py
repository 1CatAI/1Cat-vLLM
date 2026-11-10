# SPDX-License-Identifier: Apache-2.0
"""CPU unit contracts for scorer admission and packed caller dispatch."""

import pytest
import torch

from vllm.models.deepseek_v41 import attention_ops, index_topk
from vllm.models.deepseek_v41.index_tiled import IndexTiledScorer


@pytest.mark.parametrize("heads", [0, 3, 16])
def test_unsupported_heads_refuse_before_library_load(heads):
    with pytest.raises(ValueError, match="4 or 8"):
        IndexTiledScorer(None, heads)


@pytest.mark.parametrize("enabled", [False, True])
def test_packed_decode_dispatch_preserves_tensors(monkeypatch, enabled):
    q = torch.arange(1024, dtype=torch.float16).reshape(1, 8, 128)
    weights = torch.arange(8, dtype=torch.float16).reshape(1, 8)
    keys = torch.zeros(64, 68, dtype=torch.uint8)
    positions = torch.tensor([63], dtype=torch.int64)
    valid = torch.tensor([True])
    calls = []

    def score(out, actual_q, actual_w, actual_keys, actual_pos, heads, ratio):
        torch.testing.assert_close(actual_q, q.reshape(8, 128), rtol=0, atol=0)
        torch.testing.assert_close(actual_w, weights.reshape(8), rtol=0, atol=0)
        assert actual_keys is keys
        torch.testing.assert_close(actual_pos, positions)
        assert (heads, ratio) == (8, 1)
        out.copy_(torch.arange(64, dtype=torch.float16))

    class RecordingScorer:
        def scores(self, *args):
            calls.append("tiled")
            score(*args)

    def legacy_score(*args):
        calls.append("legacy")
        score(*args)

    # Only the adapter is under test: CPU sinks stand in for GPU-only providers.
    def topk(out, scores, actual_pos, actual_valid, ratio, count):
        assert actual_pos is positions and actual_valid is valid
        out.copy_(torch.topk(scores, count, dim=1).indices.to(torch.int32))

    monkeypatch.setattr(index_topk, "index_topk", topk)
    monkeypatch.setattr(attention_ops, "_cuda_index_scores", lambda: True)
    monkeypatch.setattr(
        torch.ops.v41k, "index_scores_decode", legacy_score, raising=False
    )
    result = attention_ops.select_packed_positions(
        q,
        weights,
        keys,
        positions,
        valid,
        1,
        lambda value: value,
        top_k=4,
        scorer=RecordingScorer() if enabled else None,
    )
    assert calls == ["tiled" if enabled else "legacy"]
    assert result.tolist() == [[63, 62, 61, 60]]
