# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense fallback must reuse one projection and retain exact sampling logits."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.model_executor.layers import logits_processor as logits_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.worker.gpu.spec_decode.dflash2 import sparse_rejection


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale,soft_cap", [(1.0, None), (0.5, None), (0.5, 3.0)])
@pytest.mark.parametrize("padding", [0, 3])
def test_cached_fallback_matches_full_logits(
    monkeypatch, dtype, scale, soft_cap, padding
):
    torch.manual_seed(21)
    local = torch.randn(8, 64, dtype=dtype)
    project = Mock(side_effect=lambda *args, **kwargs: local.clone())
    head = SimpleNamespace(
        quant_method=SimpleNamespace(apply=project),
        maybe_get_sm70_dflash2_top20=lambda *args: None,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=padding, org_vocab_start_index=0
        ),
    )
    processor = LogitsProcessor(
        64, org_vocab_size=64 - padding, scale=scale, soft_cap=soft_cap
    )
    monkeypatch.setattr(
        logits_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    gather = Mock(side_effect=lambda logits: logits)
    monkeypatch.setattr(processor, "_gather_logits", gather)
    hidden = torch.empty(8, 4, dtype=dtype)
    expected = processor.forward(head, hidden)
    expected_ids, expected_values = processor.get_topk_tokens_and_logits(
        head, hidden, 21
    )
    project.reset_mock()
    gather.reset_mock()

    ids, values, fallback = processor.get_topk_tokens_and_logits_with_fallback(
        head, hidden, 21
    )
    assert fallback is not None
    assert torch.equal(ids, expected_ids)
    assert torch.equal(values, expected_values)
    gather.assert_not_called()
    actual = fallback()
    assert torch.equal(actual, expected)
    project.assert_called_once()
    gather.assert_called_once()


def test_fused_candidates_do_not_require_dense_logits(monkeypatch):
    ids = torch.arange(21).reshape(1, 21)
    values = torch.arange(21, dtype=torch.float32).reshape(1, 21)
    head = SimpleNamespace(
        maybe_get_sm70_dflash2_top20=lambda *args: (values, ids),
        quant_method=SimpleNamespace(apply=Mock(side_effect=AssertionError("dense"))),
        weight=torch.empty(64, 4),
    )
    monkeypatch.setattr(
        logits_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    result = LogitsProcessor(64).get_topk_tokens_and_logits_with_fallback(
        head, torch.empty(1, 4), 21
    )
    assert result[2] is None
    assert torch.equal(result[0], ids)


@pytest.mark.parametrize("gather_rank", [False, True])
def test_cutoff_fallback_records_completed_projection_on_every_rank(
    monkeypatch, gather_rank
):
    class Speculator:
        def get_sparse_draft_logits(self):
            return None, None

    monkeypatch.setattr(sparse_rejection, "DFlash2Speculator", Speculator)
    monkeypatch.setattr(
        sparse_rejection.envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", True
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    monkeypatch.setattr(
        sparse_rejection, "_supports_sparse_sampling_contract", lambda *args: True
    )
    logits = torch.randn(8, 64) if gather_rank else None
    fallback = Mock(return_value=logits)
    probe = torch.ones(8, 21)  # Ties crossing top-k require exact full-vocab fallback.
    model = SimpleNamespace(
        get_topk_tokens_and_logits=Mock(side_effect=AssertionError("legacy call")),
        get_topk_tokens_and_logits_with_fallback=lambda *args: (None, probe, fallback),
    )
    states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array([0.7])),
        top_p=SimpleNamespace(np=np.array([0.8])),
    )
    batch = SimpleNamespace(
        has_structured_output_reqs=False,
        idx_mapping_np=np.array([0]),
        cu_num_logits_np=np.array([0, 8]),
    )
    result = sparse_rejection.try_dflash2_sparse_target_rejection(
        model,
        Speculator(),
        SimpleNamespace(sampler=SimpleNamespace(sampling_states=states)),
        SimpleNamespace(device=SimpleNamespace(type="cuda")),
        batch,
        None,
    )
    assert isinstance(result, sparse_rejection.DFlash2LogitsFallback)
    assert result.logits is logits
    fallback.assert_called_once()
