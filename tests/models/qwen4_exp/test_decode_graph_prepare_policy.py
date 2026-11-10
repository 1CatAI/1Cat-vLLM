# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-compiler preparation must read the engine's own graph policy.

Graph capture calls the preparation hook outside any forward context and
without a current vLLM config. Reading the standalone policy there resolved
``dual_compile=False`` and skipped preparation, after which the decode-graph
forward (inside a context carrying the engine policy) raised.
"""

from types import SimpleNamespace

import pytest

from vllm.models.qwen4_exp.nvidia import model as target_module
from vllm.models.qwen4_exp.nvidia import mtp as mtp_module


@pytest.mark.parametrize(
    "module,owner",
    [
        (target_module, target_module.Qwen4ExpForCausalLM),
        (mtp_module, mtp_module.Qwen4ExpMTP),
    ],
)
def test_prepare_reads_engine_policy(monkeypatch, module, owner):
    engine_config = object()
    seen = []

    def policy(cfg=None):
        seen.append(cfg)
        # Only the engine-owned policy enables the dual-compile lane.
        return SimpleNamespace(dual_compile=cfg is engine_config)

    monkeypatch.setattr(module, "graph_policy", policy)
    prepared = object()
    model = SimpleNamespace(
        vllm_config=engine_config,
        _sm70_decode_graph_model=prepared,
        prepare_sm70_draft_head=lambda: None,
    )
    assert owner.prepare_sm70_decode_graph_model(model) is True
    assert seen == [engine_config]
    assert model._sm70_decode_graph_model is prepared
