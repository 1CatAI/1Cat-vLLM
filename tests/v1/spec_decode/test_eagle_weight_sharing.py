# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.v1.worker.gpu.spec_decode.eagle import utils


class ConditionalTarget(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def get_language_model(self):
        return self.language_model


def make_model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(8, 4)
    model.lm_head = nn.Linear(4, 8, bias=False)
    return model


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("own_head", ["absent", "identical", "different"])
def test_loader_shares_target_head_without_replacing_distinct_draft(
    monkeypatch, wrapped, own_head
):
    language_model = make_model()
    target = ConditionalTarget(language_model) if wrapped else language_model
    draft = make_model()
    if own_head == "absent":
        del draft.lm_head
    else:
        draft.has_own_lm_head = True
        with torch.no_grad():
            draft.lm_head.weight.copy_(language_model.lm_head.weight)
            if own_head == "different":
                draft.lm_head.weight.add_(1)
    original_head = getattr(draft, "lm_head", None)
    # Models using a per-layer shared head must receive the same alias too.
    layer = nn.Module()
    layer.shared_head = nn.Module()
    layer.shared_head.head = original_head
    draft.model.layers = nn.ModuleList([layer])
    monkeypatch.setattr(utils, "get_model", lambda **_kwargs: draft)
    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(
        "vllm.compilation.backends.set_model_tag", lambda _tag: nullcontext()
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object())
    )

    result = utils.load_eagle_model(target, config)

    expected = original_head if own_head == "different" else language_model.lm_head
    assert result is draft
    assert result.model.embed_tokens is language_model.model.embed_tokens
    assert result.lm_head is expected
    assert layer.shared_head.head is expected
    assert result.lm_head.weight.data_ptr() == expected.weight.data_ptr()


def test_target_head_resolution_prefers_language_model_and_keeps_fallback():
    language_model = make_model()
    wrapper = ConditionalTarget(language_model)
    wrapper.lm_head = nn.Linear(4, 8, bias=False)
    assert utils.get_target_lm_head(wrapper, language_model) is language_model.lm_head
    del language_model.lm_head
    assert utils.get_target_lm_head(wrapper, language_model) is wrapper.lm_head
    del wrapper.lm_head
    assert utils.get_target_lm_head(wrapper, language_model) is None
