# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM70 ModelOpt NVFP4 MoE gate must step aside for an explicit backend.

Without ``--moe-backend`` the gate keeps binding routed experts to the
TurboMind method, which is the behaviour every existing deployment relies on,
unless the skinny SM70 MoE kernel admits the shape.
With an explicit pick the generic path runs and its backend oracle decides.
"""

import pytest

from vllm.model_executor.layers.fused_moe.experts import skinny_sm70_moe
from vllm.model_executor.layers.quantization.modelopt import (
    _sm70_moe_backend_requested_explicitly,
)


class _MoeConfig:
    def __init__(self, moe_backend):
        self.moe_backend = moe_backend


class _Layer:
    def __init__(self, moe_backend):
        self.moe_config = _MoeConfig(moe_backend)


@pytest.mark.parametrize("backend", ["auto", "AUTO", ""])
def test_default_backend_keeps_the_sm70_gate(backend, monkeypatch):
    # An admitted skinny MoE kernel also releases the gate for "auto"; this
    # case covers the shapes it does not admit.
    monkeypatch.setattr(
        skinny_sm70_moe, "skinny_backend_admitted", lambda *a, **k: False
    )
    assert not _sm70_moe_backend_requested_explicitly(_Layer(backend))


@pytest.mark.parametrize("backend", ["sm70_skinny", "marlin", "SM70_SKINNY"])
def test_explicit_backend_releases_the_sm70_gate(backend):
    assert _sm70_moe_backend_requested_explicitly(_Layer(backend))


def test_layer_without_moe_config_keeps_the_sm70_gate():
    assert not _sm70_moe_backend_requested_explicitly(object())
