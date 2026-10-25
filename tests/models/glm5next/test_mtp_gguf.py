# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the native MTP weight loader with CPU-sized parameters."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.shared_weights import should_share
from vllm.models.glm5next.nvidia import mtp

pytestmark = pytest.mark.cpu_test


def test_mtp_loads_pp_embedding_and_preserves_indexer_fp32(monkeypatch):
    model = mtp.Glm5NextMTP.__new__(mtp.Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        n_routed_experts=1,
        mla_nope=True,
        qk_rope_head_dim=0,
    )
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 45
    model.model.num_mtp_layers = 1
    model.model.embed_tokens = nn.Embedding(3, 2, dtype=torch.float16)
    layer = nn.Module()
    layer.mtp_block = nn.Module()
    layer.mtp_block.self_attn = nn.Module()
    indexer = nn.Module()
    layer.mtp_block.self_attn.indexer = indexer
    indexer.wk_weights_proj = nn.Linear(2, 3, bias=False, dtype=torch.float16)
    indexer.register_buffer("_wp_fp32", None)
    model.model.layers = nn.ModuleDict({"45": layer})

    @torch.no_grad()
    def load_fused(param, value, shard):
        (param[:2] if shard == 0 else param[2:]).copy_(value)

    indexer.wk_weights_proj.weight.weight_loader = load_fused
    monkeypatch.setattr(mtp, "fused_moe_make_expert_params_mapping", lambda *a, **k: [])
    embed = torch.arange(6, dtype=torch.float16).reshape(3, 2)
    precise = torch.tensor([[1 + 2**-20, -1 - 2**-20]], dtype=torch.float32)
    loaded = model.load_weights(
        [
            ("model.embed_tokens.weight", embed),
            (
                "model.layers.45.mtp_block.self_attn.indexer.weights_proj.weight",
                precise,
            ),
        ]
    )
    assert loaded == {
        "model.embed_tokens.weight",
        "model.layers.45.mtp_block.self_attn.indexer.wk_weights_proj.weight",
    }
    torch.testing.assert_close(model.model.embed_tokens.weight, embed, rtol=0, atol=0)
    torch.testing.assert_close(indexer._wp_fp32, precise.t(), rtol=0, atol=0)
    assert not torch.equal(indexer._wp_fp32, precise.half().float().t())


def test_explicitly_owned_quantized_draft_head_is_not_replaced():
    draft = SimpleNamespace(has_own_lm_head=True)
    packed = SimpleNamespace(qweight=torch.tensor([1], dtype=torch.uint8))
    target = SimpleNamespace(qweight=torch.tensor([2], dtype=torch.uint8))
    assert not should_share(draft, "has_own_lm_head", packed, target)
    draft.has_own_lm_head = False
    assert should_share(draft, "has_own_lm_head", packed, target)
