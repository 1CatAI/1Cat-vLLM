# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""§3.8 golden-dump hook: off by default, per-step L{nn}/final safetensors with §3.8 names, dummy runs skipped."""

from __future__ import annotations

import importlib
import json

import pytest
import torch
from safetensors import safe_open

from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v41.common import dump as ds41_dump

from . import core_stubs
from .test_core_model import _vllm_config, loaded_model  # noqa: F401  (fixture)

LAYER_NAMES = {"stream_in", "pre_in", "hc.attn_pre", "hc.attn_post", "hc.attn_comb", "hc.attn_x", "attn.x",
               "attn.out", "stream_attn", "hc.ffn_pre", "hc.ffn_post", "hc.ffn_comb", "hc.ffn_x", "moe.x", "moe.out",
               "stream_out", "pre_out"}


@pytest.fixture
def dump_env(monkeypatch, tmp_path):
    def enable(layers: str, steps: str) -> None:
        monkeypatch.setenv(ds41_dump.DIR_KNOB, str(tmp_path))
        monkeypatch.setenv(ds41_dump.LAYERS_KNOB, layers)
        monkeypatch.setenv(ds41_dump.STEPS_KNOB, steps)
        importlib.reload(ds41_dump)

    yield enable
    for knob in (ds41_dump.DIR_KNOB, ds41_dump.LAYERS_KNOB, ds41_dump.STEPS_KNOB):
        monkeypatch.delenv(knob, raising=False)
    importlib.reload(ds41_dump)
    assert not ds41_dump.ENABLED


def _names(path) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    with safe_open(str(path), framework="pt") as f:
        return {k: (tuple(f.get_tensor(k).shape), f.get_tensor(k).dtype) for k in f.keys()}


def test_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(ds41_dump.DIR_KNOB, raising=False)
    importlib.reload(ds41_dump)
    assert ds41_dump.ENABLED is False and ds41_dump._STATE is None
    ds41_dump.dump(0, "stream_in", torch.zeros(1))   # no-op, no distributed state needed


def test_parse_ids() -> None:
    assert ds41_dump.parse_ids("all", "K") is None and ds41_dump.parse_ids("none", "K") == frozenset()
    assert ds41_dump.parse_ids("0-3,14", "K") == frozenset({0, 1, 2, 3, 14})
    with pytest.raises(ValueError):
        ds41_dump.parse_ids("3-1", "K")


def test_dump_steps_layers_and_dummy(loaded_model, dump_env, tmp_path) -> None:  # noqa: F811
    model, vc, _ = loaded_model
    dump_env("0,2", "0-1")
    ids, pos = torch.tensor([0, 671, 6102]), torch.arange(3)
    with set_current_vllm_config(vc), torch.inference_mode():
        model.compute_logits(model(ids, pos, None))                       # step 0
        with set_forward_context(None, vc, num_tokens=3, is_dummy_run=True):
            model(ids, pos, None)                                         # dummy: not a step
        model.compute_logits(model(ids[:1], pos[:1] + 3, None))           # step 1 (decode-like)
        model(ids, pos, None)                                             # step 2: outside STEPS
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step00000", "step00001"]
    step0 = tmp_path / "step00000"
    assert sorted(p.name for p in step0.iterdir()) == ["L00.safetensors", "L02.safetensors", "final.safetensors",
                                                       "meta_pp0.json"]
    l0 = _names(step0 / "L00.safetensors")
    assert set(l0) == LAYER_NAMES
    assert l0["stream_in"] == ((3, 4, 5120), torch.bfloat16) and l0["pre_in"] == ((3, 4), torch.float32)
    assert l0["hc.attn_comb"] == ((3, 4, 4), torch.float32) and l0["attn.x"] == ((3, 5120), torch.float16)
    assert l0["attn.out"] == ((3, 5120), torch.float32) and l0["moe.out"] == ((3, 5120), torch.float32)
    final = _names(step0 / "final.safetensors")
    assert final["logits"] == ((3, 129280), torch.float32) and final["final.h"] == ((3, 5120), torch.float16)
    assert final["final.stream_in"][0] == (3, 4, 5120) and final["final.hc"] == ((3, 5120), torch.bfloat16)
    meta = json.loads((step0 / "meta_pp0.json").read_text())
    assert meta["step"] == 0 and meta["num_tokens"] == 3 and meta["positions"] == [0, 1, 2]
    assert _names(tmp_path / "step00001" / "L02.safetensors")["stream_out"][0] == (1, 4, 5120)
    # the dumped final hidden is exactly what the model returned
    with safe_open(str(step0 / "final.safetensors"), framework="pt") as f:
        with set_current_vllm_config(vc), torch.inference_mode():
            assert torch.equal(f.get_tensor("final.h"), model(ids, pos, None))


def test_dump_requires_eager(ds41_dist_single, ds41_checkpoint_dir, monkeypatch, dump_env) -> None:
    core_stubs.install(monkeypatch.setitem)
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41Model

    dump_env("all", "0")
    vc = _vllm_config(ds41_checkpoint_dir)
    vc.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    with set_current_vllm_config(vc), pytest.raises(ValueError, match="enforce-eager"):
        DeepseekV41Model(vllm_config=vc, prefix="model")
