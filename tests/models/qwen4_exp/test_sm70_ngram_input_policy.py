# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState


@pytest.mark.parametrize(
    "enabled,hardware,padded,reason",
    [
        (False, True, 1, "disabled_by_kernel_config"),
        (True, False, 1, "requires_sm70"),
        (True, True, 5, "input_shape_has_no_calibration"),
        (True, True, 1, "requires_contiguous_cuda_int32_inputs"),
    ],
)
def test_preparation_fallback_preserves_context(enabled, hardware, padded, reason):
    owner = object.__new__(Qwen4ExpModelState)
    owner.rope_state = None
    owner.uses_ngram_embedding = True
    owner.ngram_context_len = 2
    owner.ngram_eos_token_id = 9
    owner.ngram_context = torch.empty((padded, 2), dtype=torch.int32)
    owner.ngram_context_offsets = torch.arange(-2, 0, dtype=torch.int64)
    owner.ple_query_start_loc = torch.empty(padded + 1, dtype=torch.int32)
    owner._ple_input_hardware = hardware
    owner._ple_kernel_config = SimpleNamespace(
        ple_input_prepare=enabled, ple_input_preparations={}
    )
    batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=padded,
        idx_mapping=torch.zeros(padded, dtype=torch.int32),
        query_start_loc=torch.arange(padded + 1, dtype=torch.int32),
    )
    states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([1])),
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[11, 12, 13]])),
    )
    inputs = owner.prepare_inputs(batch, states)
    expected = torch.full((padded, 2), 9, dtype=torch.int32)
    expected[0, 1] = 11
    torch.testing.assert_close(inputs["ngram_context"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        inputs["query_start_loc"], batch.query_start_loc, rtol=0, atol=0
    )
    assert (
        owner._ple_kernel_config.ple_input_preparations["ngram_context"]["reason"]
        == reason
    )
