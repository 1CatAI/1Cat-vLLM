# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for unused draft work, including collective/warmup guards."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as module

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    "incomplete,dp_size,dummy,profile,skip",
    [
        ([True], 1, False, False, True),
        ([True, True], 1, False, False, True),
        ([True, False], 1, False, False, False),
        ([False], 1, False, False, False),
        ([True], 2, False, False, False),
        ([True], 1, True, False, False),
        ([True], 1, False, True, False),
        (None, 1, False, False, False),
    ],
)
@pytest.mark.parametrize("disable_pruning", [False, True])
def test_partial_prefill_keeps_kv_warmup_but_skips_unused_decode(
    monkeypatch, incomplete, dp_size, dummy, profile, skip, disable_pruning
):
    num_reqs = len(incomplete) if incomplete is not None else 1
    num_tokens = num_reqs * 16
    spec = object.__new__(module.EagleSpeculator)
    spec.method = "mtp"
    spec.vllm_config = SimpleNamespace(
        additional_config={"disable_incomplete_prefill_draft_pruning": disable_pruning}
    )
    skip = skip and not disable_pruning
    spec.num_speculative_steps = 4
    spec.dp_size, spec.dp_rank = dp_size, 0
    spec.max_num_reqs, spec.max_model_len = num_reqs, 1024
    spec.hidden_states = torch.zeros(num_tokens, 4)
    spec.idx_mapping = torch.zeros(num_reqs, dtype=torch.int32)
    spec.temperature = torch.zeros(num_reqs)
    spec.seeds = torch.zeros(num_reqs, dtype=torch.int64)
    spec.draft_tokens = torch.full((num_reqs, 4), 99, dtype=torch.int64)
    spec.last_token_indices = torch.zeros(num_reqs, dtype=torch.int64)
    spec.current_draft_step = torch.tensor(0)
    spec.input_buffers = SimpleNamespace()
    spec.prefill_cudagraph_manager = object()
    spec.decode_cudagraph_manager = object()
    spec.share_mtp_topk_indices = False
    batch = SimpleNamespace(
        num_tokens_after_padding=num_tokens,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        num_scheduled_tokens=np.full(num_reqs, 16),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32),
        seq_lens=torch.full((num_reqs,), 16),
        is_prefilling_np=np.ones(num_reqs, dtype=bool),
        is_incomplete_prefilling_np=(
            np.array(incomplete, dtype=bool) if incomplete is not None else None
        ),
    )
    calls = []

    def prefill(*args, **kwargs):
        calls.append("prefill_kv_committed")
        spec.draft_tokens[:, 0] = 7

    def decode(*args, **kwargs):
        assert calls == ["prefill_kv_committed"]
        calls.append("decode")
        spec.draft_tokens[:, 1:] = 8

    spec.prefill = prefill
    spec.multi_step_decode = decode
    prepare_decode = Mock()
    dispatch = Mock(
        return_value=(
            SimpleNamespace(cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens),
            None,
        )
    )
    monkeypatch.setattr(module, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(module, "prepare_eagle_inputs", Mock())
    monkeypatch.setattr(module, "prepare_eagle_decode", prepare_decode)
    # Catch a regression that introduces a host wait to inspect sampled counts.
    monkeypatch.setattr(torch.cuda, "synchronize", Mock(side_effect=AssertionError))
    result = spec.propose(
        batch,
        {},
        {},
        torch.ones(num_tokens, 4),
        None,
        torch.zeros(num_reqs, dtype=torch.int32),
        torch.zeros(num_reqs, dtype=torch.int32),
        torch.zeros(num_reqs, dtype=torch.int64),
        torch.zeros(num_reqs, dtype=torch.int64),
        torch.ones(num_reqs),
        torch.zeros(num_reqs, dtype=torch.int64),
        dummy_run=dummy,
        is_profile=profile,
    )
    assert result.shape == (num_reqs, 4)
    if skip:
        assert calls == ["prefill_kv_committed"]
        assert dispatch.call_count == 1
        prepare_decode.assert_not_called()
        assert torch.count_nonzero(result) == 0
    else:
        assert calls == ["prefill_kv_committed", "decode"]
        assert dispatch.call_count == 2
        prepare_decode.assert_called_once()
        assert torch.all(result[:, 0] == 7)
        assert torch.all(result[:, 1:] == 8)
