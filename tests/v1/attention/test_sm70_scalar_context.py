# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-query graph metadata enables device-resident wave selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import metadata as module


@pytest.mark.parametrize(
    "enabled,requests,query_len,variant,expected",
    [
        (True, 1, 1, None, True),
        (True, 1, 1, 0, True),
        (True, 1, 5, 0, False),
        (True, 8, 1, 0, True),
        (True, 3, 1, 0, True),
        (False, 1, 1, 0, False),
    ],
)
def test_scalar_capture_and_replay_admission(
    monkeypatch, enabled, requests, query_len, variant, expected
):
    monkeypatch.setattr(module, "as_flash_v100_metadata", lambda metadata: metadata)
    builder = SimpleNamespace(
        _batch_context_routing_enabled=False,
        _scalar_context_routing_enabled=enabled,
        spec_state=SimpleNamespace(attach_common=lambda metadata: None),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=262144)),
    )
    lengths = torch.full((requests,), 131072, dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=requests,
        max_query_len=query_len,
        cudagraph_graph_variant=variant,
        query_start_loc_cpu=torch.arange(requests + 1, dtype=torch.int32) * query_len,
        _seq_lens_cpu=None,
        seq_lens_cpu_upper_bound=lengths,
        causal=True,
    )
    result = SimpleNamespace()
    module.FlashAttnV100MetadataBuilder._attach_common_flash_metadata(
        builder, result, common
    )
    assert result.flash_v100_batch_context_routing is expected
    assert result.seq_lens_cpu is lengths
    assert result.max_model_len == 262144
