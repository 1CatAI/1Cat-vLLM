# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import host_kv


@pytest.mark.parametrize(
    "budget,main_pages,draft_pages", [(262144, 321, 160), (525504, 644, 322)]
)
def test_shared_stage_budget_is_page_rounded_and_dtype_aware(
    monkeypatch, budget, main_pages, draft_pages
):
    monkeypatch.setattr(host_kv, "_STAGES", {})
    pointers = []
    for dtype, pages in [(torch.uint8, main_pages), (torch.float16, draft_pages)]:
        owner = host_kv.HostQSAKV.__new__(host_kv.HostQSAKV)
        owner.page_size, owner.dim = 816, 256
        owner.history, owner.host = torch.empty(0, dtype=dtype), None
        owner.fp8 = dtype == torch.uint8
        assert owner._stage_capacity(budget) == pages
        history, scales = owner._stage_views(budget, 0)
        assert history.shape == (pages, 2, 816, 1, 256)
        assert scales.shape == ((pages * 816, 2) if owner.fp8 else (1, 2))
        pointers.append(history.untyped_storage().data_ptr())
        assert history.untyped_storage().nbytes() == budget * 512
    assert pointers[0] == pointers[1]
    assert len(host_kv._STAGES) == 1


@pytest.mark.parametrize("tokens", [130560, 130561, 261936, 261937, 262144])
def test_full_draft_budget_covers_exact_256k_page_boundary(tokens):
    owner = host_kv.HostQSAKV.__new__(host_kv.HostQSAKV)
    owner.page_size, owner.dim = 816, 256
    owner.history = torch.empty(0, dtype=torch.float16)
    needed = (tokens + 815) // 816
    assert (needed <= owner._stage_capacity(262144)) is (tokens <= 130560)
    assert needed <= owner._stage_capacity(525504)
