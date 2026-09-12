# SPDX-License-Identifier: AGPL-3.0-only
"""The QSA E4M3 scale gate must degrade to a warning, not refuse to start.

A checkpoint that was never calibrated for the E4M3 QSA KV cache carries no
k_scale/v_scale entries. The loader already ignores their absence, so the gate
has to agree with it instead of turning that into a hard failure.
"""

import pytest

from vllm.models.qwen4_exp.nvidia.model import _validate_qsa_e4m3_scale_load

REQUIRED = {
    "language_model.model.layers.3.self_attn.k_scale",
    "language_model.model.layers.3.self_attn.v_scale",
}


@pytest.fixture(autouse=True)
def _unit_scales(monkeypatch):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES", False)


def test_uncalibrated_checkpoint_degrades_to_a_warning():
    missing = _validate_qsa_e4m3_scale_load(REQUIRED, set(), "fp8_e4m3")
    assert missing == REQUIRED


def test_strict_mode_keeps_the_hard_failure(monkeypatch):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES", True)
    with pytest.raises(ValueError, match="refusing to start"):
        _validate_qsa_e4m3_scale_load(REQUIRED, set(), "fp8_e4m3")


def test_calibrated_checkpoint_passes_without_a_warning():
    assert _validate_qsa_e4m3_scale_load(REQUIRED, set(REQUIRED), "fp8_e4m3") == set()


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_e5m2"])
def test_other_cache_dtypes_are_not_gated(cache_dtype):
    # Always a set, never None: the caller iterates the result.
    assert _validate_qsa_e4m3_scale_load(REQUIRED, set(), cache_dtype) == set()
