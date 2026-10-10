# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup resolves the split backend owner and covers real/padded requests."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.v1.attention.backends.flash_v100 import metadata
from vllm.v1.attention.backends.flash_v100.spec import smallq_metadata
from vllm.v1.worker.gpu import sm70_runner_ops as ops


def test_smallq_warmup_resolves_owned_backend_exports(monkeypatch):
    monkeypatch.setattr(ops, "capture_sm70_dflash2_config", lambda _: object())
    monkeypatch.setattr(ops, "sm70_dflash2_enabled", lambda *args: True)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    prepare = Mock()
    monkeypatch.setattr(
        smallq_metadata, "_sm70_prepare_grouped_smallq_decode_metadata", prepare
    )
    builder = object.__new__(metadata.FlashAttnV100MetadataBuilder)
    runner = SimpleNamespace(
        vllm_config=object(),
        attn_groups=[[SimpleNamespace(get_metadata_builder=lambda _: builder)]],
        block_tables=SimpleNamespace(
            block_tables=[SimpleNamespace(gpu=torch.empty((3, 7), dtype=torch.int32))]
        ),
        decode_query_len=5,
        max_num_reqs=3,
        device=torch.device("cpu"),
    )
    logger = Mock()
    assert ops.warmup_smallq_metadata(runner, logger)
    logger.warning_once.assert_not_called()
    assert {
        (c.kwargs["num_reqs"], c.kwargs["real_num_query_tokens"])
        for c in prepare.call_args_list
    } == {(1, 5), (2, 5), (2, 10), (3, 5), (3, 15)}
    for call in prepare.call_args_list:
        out_bt, _, _, in_bt, seq_lens, query_start_loc = call.args
        count = call.kwargs["num_reqs"]
        assert out_bt[0].shape == (count * 5, 7)
        assert in_bt[0].shape == (count, 7)
        assert torch.equal(seq_lens, torch.full((count,), 5, dtype=torch.int32))
        assert torch.equal(
            query_start_loc, torch.arange(0, (count + 1) * 5, 5, dtype=torch.int32)
        )
