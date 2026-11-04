# SPDX-License-Identifier: Apache-2.0
"""CPU model-adapter tests; provider arithmetic is qualified separately."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v41 import draft_projection, dspark, fp8_linear


@pytest.mark.parametrize("m", [1, 8, 9])
@pytest.mark.parametrize("enabled", [False, True])
def test_model_routes_small_rows_to_shards(monkeypatch, m, enabled):
    model = dspark.V41DraftModel.__new__(dspark.V41DraftModel)
    nn.Module.__init__(model)
    model.shard_main_projection = enabled
    model._shard_main_projection_logged = True
    model.main_norm = nn.Identity()
    calls = []

    class Projection(nn.Module):
        weight = torch.empty(128, 384, dtype=torch.float8_e4m3fn)
        weight_scale_inv = torch.empty(4, 12, dtype=torch.float8_e8m0fnu)

        def forward(self, x):
            calls.append("replicated")
            return x[:, :128] + 1

    model.main_proj = Projection()

    def sharded(x, weight, scales, rank, world, gather):
        assert weight is model.main_proj.weight
        assert scales is model.main_proj.weight_scale_inv
        assert (rank, world) == (1, 4)
        calls.append("sharded")
        return x[:, :128] + 2

    monkeypatch.setattr(draft_projection, "sharded_projection", sharded)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    result = model.combine_hidden_states(torch.zeros(m, 384, dtype=torch.float16))
    use_shard = enabled and m <= 8
    assert calls == ["sharded" if use_shard else "replicated"]
    assert torch.equal(result, torch.full_like(result, 2 if use_shard else 1))


@pytest.mark.parametrize("kind", ["wrong_weight", "missing_scales", "wrong_scales"])
def test_model_construction_refuses_non_raw_fp8(monkeypatch, kind):
    class Projection(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.bias = None
            self.weight = nn.Parameter(
                torch.empty(
                    128,
                    384,
                    dtype=(
                        torch.float16 if kind == "wrong_weight" else torch.float8_e4m3fn
                    ),
                ),
                requires_grad=False,
            )
            if kind != "missing_scales":
                self.weight_scale_inv = nn.Parameter(
                    torch.empty(
                        4,
                        12,
                        dtype=(
                            torch.float16
                            if kind == "wrong_scales"
                            else torch.float8_e8m0fnu
                        ),
                    ),
                    requires_grad=False,
                )

    monkeypatch.setattr(dspark, "ReplicatedLinear", Projection)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(fp8_linear, "_gemv_ready", lambda: True)
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(
            v41_draft_projection=SimpleNamespace(enabled=True)
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=128)),
        quant_config=None,
    )
    with pytest.raises(ValueError, match="raw E4M3 weights and UE8M0 scales"):
        dspark.V41DraftModel(config, "draft")
