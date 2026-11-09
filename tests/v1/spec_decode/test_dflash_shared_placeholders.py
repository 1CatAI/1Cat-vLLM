# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import qwen3_dflash as draft


@pytest.mark.parametrize("own_embed", [False, True])
@pytest.mark.parametrize("own_head", [False, True])
def test_only_absent_checkpoint_placeholders_are_released(
    monkeypatch, own_embed, own_head
):
    model = draft.DFlashQwen3ForCausalLM.__new__(draft.DFlashQwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(32, 16)
    model.lm_head = torch.nn.Linear(16, 32, bias=False)
    embed, head = model.model.embed_tokens, model.lm_head
    model.model.use_aux_hidden_state = False
    model.model.has_separate_mask_embedding = False
    model.model._build_fused_kv_buffers = lambda: None
    model._read_mask_embedding = lambda: None
    loaded = []
    monkeypatch.setattr(
        draft,
        "AutoWeightsLoader",
        lambda *args, **kwargs: SimpleNamespace(
            load_weights=lambda weights: loaded.extend(weights)
        ),
    )
    weights = []
    if own_embed:
        weights.append(("embed_tokens.weight", embed.weight))
    if own_head:
        weights.append(("lm_head.weight", head.weight))
    model.load_weights(weights)
    assert len(loaded) == len(weights)
    assert getattr(model.model, "embed_tokens", None) is (embed if own_embed else None)
    assert getattr(model, "lm_head", None) is (head if own_head else None)
    # The generic post-loader must only visit checkpoint-owned parameters;
    # target sharing will attach the already prepared target modules afterward.
    assert len(list(model.parameters())) == int(own_embed) + int(own_head)
