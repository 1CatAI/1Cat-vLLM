# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.models.qwen4_exp.nvidia import mtp


@pytest.mark.parametrize(
    "rows,accepted",
    [
        (5, False),
        (4095, False),
        (4096, True),
        (4097, True),
        (8192, True),
        (16384, True),
        (16385, False),
    ],
)
def test_mtp_row_admission_retains_decode_precision(monkeypatch, rows, accepted):
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    monkeypatch.setattr(mtp, "is_forward_context_available", lambda: True)
    context = NS(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(mtp, "get_forward_context", lambda: context)
    owner = NS(_hc_prefill_sequence_qualified=True)
    hidden = torch.empty(rows, 10240, dtype=torch.half, device="meta")
    positions = torch.empty(rows, device="meta")
    select = mtp.Qwen4ExpMTP._use_hc_prefill_sequence_rows
    assert select(owner, positions, hidden, None) is accepted
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    for mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
        context.cudagraph_runtime_mode = mode
        assert not select(owner, positions, hidden, None)
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    assert not select(owner, positions, None, None)
    assert not select(owner, positions, hidden.float(), None)
    assert not select(owner, positions, hidden[:, :-1], None)
    assert not select(owner, positions, hidden, object())
    owner._hc_prefill_sequence_qualified = False
    assert not select(owner, positions, hidden, None)


def test_mtp_prefill_view_borrows_weights_and_state():
    class Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("state", torch.zeros(1))

        def forward(self, ids, positions, hidden, *args, hc_sequence_parallel):
            assert hc_sequence_parallel
            self.state.add_(1)
            return ids * self.weight, hidden.clone()

    owner = Owner()
    cfg = NS(compilation_config=NS(mode=0))
    view = mtp._Qwen4ExpMTPPrefillSequenceModel(target_model=owner, vllm_config=cfg)
    assert not list(view.parameters()) and not list(view.buffers())
    hidden = torch.ones(2, 4)
    sample, multi = view(torch.ones(2), torch.arange(2), hidden)
    assert torch.equal(sample, torch.ones(2)) and torch.equal(multi, hidden)
    assert multi.data_ptr() != hidden.data_ptr()
    assert torch.equal(owner.state, torch.ones(1))
    assert view._target_model is owner
