# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.glm5next.nvidia import model as glm_model

_SCALE = "model.layers.3.self_attn.q_b_proj.weight_scale"
_WEIGHT = "model.layers.3.self_attn.q_b_proj.weight"
_TARGET = "model.layers.3.self_attn.q_b_proj.weight"


def _fp8_weight() -> torch.Tensor:
    return torch.full((256, 128), 2.0).to(torch.float8_e4m3fn)


def _block_scale() -> torch.Tensor:
    # ModelOpt FP8_PB_WO: float32 [N/128, 1, K/128, 1].
    return torch.full((2, 1, 1, 1), 0.5)


def _load(model, name, tensor, buf, params, loaded):
    return glm_model._try_load_fp8_attn_proj(
        model, name, tensor, buf, params, loaded, kv_a_pad_size=0
    )


def test_block_fp8_projection_loads_with_its_scale_first(monkeypatch):
    monkeypatch.setattr(glm_model, "is_pp_missing_parameter", lambda name, model: False)
    received = []
    param = SimpleNamespace(weight_loader=lambda p, tensor: received.append(tensor))
    params = {_TARGET: param}
    loaded: set[str] = set()
    buf: dict[str, dict] = {}

    assert _load(None, _SCALE, _block_scale(), buf, params, loaded)
    assert not received
    assert _load(None, _WEIGHT, _fp8_weight(), buf, params, loaded)

    assert loaded == {_TARGET}
    expected = torch.full((256, 128), 1.0, dtype=received[0].dtype)
    assert torch.equal(received[0], expected)
    assert not any(buf.values())


def test_block_fp8_projection_of_another_stage_is_skipped(monkeypatch):
    # Mapped loading under pipeline parallelism streams every stage's layers;
    # a projection this stage does not hold must be dropped, not looked up.
    monkeypatch.setattr(glm_model, "is_pp_missing_parameter", lambda name, model: True)
    buf: dict[str, dict] = {}
    loaded: set[str] = set()

    assert _load(None, _SCALE, _block_scale(), buf, {}, loaded)
    assert _load(None, _WEIGHT, _fp8_weight(), buf, {}, loaded)

    assert not loaded
    assert buf == {}
