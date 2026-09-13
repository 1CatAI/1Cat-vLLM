# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN prefill backend resolution across pre-Ampere capabilities.

Turing (sm75) must not resolve to ``flashqla_sm70``: the path's default
TileLang kernel does not compile there with the pinned tilelang, and its
VLK CUDA alternative is slower than Triton/FLA at prefill chunk sizes.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _resolve_gdn_prefill_backend,
)
from vllm.platforms import current_platform


def _config(dtype=torch.float16, head_k_dim=128, requested="auto"):
    model_config = SimpleNamespace(
        dtype=dtype,
        hf_text_config=SimpleNamespace(linear_key_head_dim=head_k_dim),
        hf_config=None,
    )
    return SimpleNamespace(
        model_config=model_config,
        additional_config={"gdn_prefill_backend": requested},
    )


@pytest.fixture
def pre_ampere(monkeypatch):
    """Pin the platform to a pre-Ampere CUDA device of the given capability."""

    def _apply(major: int, minor: int):
        capability = SimpleNamespace(major=major, minor=minor)
        monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
        monkeypatch.setattr(
            current_platform, "get_device_capability", lambda *_a, **_kw: capability
        )
        monkeypatch.setattr(
            current_platform, "is_device_capability", lambda *_a, **_kw: False
        )
        monkeypatch.setattr(
            current_platform, "is_device_capability_family", lambda *_a, **_kw: False
        )

    return _apply


def test_turing_resolves_to_triton(pre_ampere):
    pre_ampere(7, 5)
    _, backend = _resolve_gdn_prefill_backend(_config())
    assert backend == "triton"


def test_turing_resolves_to_triton_when_flashqla_requested(pre_ampere):
    # An explicit request must not route Turing into a path that cannot run.
    pre_ampere(7, 5)
    _, backend = _resolve_gdn_prefill_backend(_config(requested="flashqla_sm70"))
    assert backend == "triton"


def test_volta_keeps_flashqla(pre_ampere):
    pytest.importorskip(
        "flash_qla.ops.gated_delta_rule.chunk.sm70",
        reason="FlashQLA-SM70 kernels are not installed.",
    )
    pre_ampere(7, 0)
    _, backend = _resolve_gdn_prefill_backend(_config())
    assert backend == "flashqla_sm70"


def test_volta_non_fp16_falls_back(pre_ampere):
    # Unchanged behaviour: the path is fp16-only on Volta as well.
    pre_ampere(7, 0)
    _, backend = _resolve_gdn_prefill_backend(_config(dtype=torch.bfloat16))
    assert backend == "triton"
