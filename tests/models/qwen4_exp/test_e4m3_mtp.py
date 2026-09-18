# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2 CPU tests: E4M3 QSA main KV cache together with MTP speculative
decoding (D1 gate opt-in + D2 strict draft-side scale finalization + D3 draft
scale visibility). All tests are CPU-only; the live SM70 kernel/selection path
is exercised in the W3 GPU window and by tests/models/qwen4_exp/test_qsa_e4m3.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import envs
from vllm.models.qwen4_exp.nvidia import model as model_mod
from vllm.models.qwen4_exp.nvidia import qsa as qsa_mod
from vllm.models.qwen4_exp.nvidia.model import (
    _finalize_qsa_e4m3_scale_load,
    _validate_qsa_e4m3_scale_load,
)
from vllm.models.qwen4_exp.nvidia.mtp import _remap_mtp_weight_name
from vllm.models.qwen4_exp.nvidia.qsa import (
    Qwen4ExpQSAAttention,
    _verify_e4m3_kv_requirements,
)

E4M3 = "fp8_e4m3"
_MTP_SPEC = object()
pytestmark = pytest.mark.skip_global_cleanup


def _configs(*, dtype=torch.float16, tp=4, spec=_MTP_SPEC):
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=spec,
    )
    model_config = SimpleNamespace(dtype=dtype)
    cache_config = SimpleNamespace(cache_dtype=E4M3)
    return vllm_config, model_config, cache_config


def _force_sm70(monkeypatch):
    monkeypatch.setattr(
        qsa_mod,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda cap: cap == 70),
    )


# --------------------------------------------------------------------------- #
# D1: the MTP0 gate is opt-in; default keeps the phase-1 error verbatim.
# --------------------------------------------------------------------------- #
def test_gate_default_off_rejects_e4m3_plus_mtp(monkeypatch):
    _force_sm70(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_MTP", False)
    vllm_config, model_config, cache_config = _configs(spec=object())
    with pytest.raises(NotImplementedError, match="requires MTP0"):
        _verify_e4m3_kv_requirements(vllm_config, model_config, cache_config)


def test_gate_switch_on_allows_e4m3_plus_mtp(monkeypatch):
    _force_sm70(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_MTP", True)
    vllm_config, model_config, cache_config = _configs(spec=object())
    # Must not raise: MTP is permitted when the opt-in is set.
    _verify_e4m3_kv_requirements(vllm_config, model_config, cache_config)


def test_gate_switch_on_without_spec_is_unaffected(monkeypatch):
    _force_sm70(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_MTP", False)
    vllm_config, model_config, cache_config = _configs(spec=None)
    _verify_e4m3_kv_requirements(vllm_config, model_config, cache_config)


def test_gate_opt_in_does_not_loosen_sm70_fp16_tp4(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_MTP", True)
    # Non-SM70 still rejected even with the opt-in on.
    monkeypatch.setattr(
        qsa_mod,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda cap: False),
    )
    vc, mc, cc = _configs()
    with pytest.raises(NotImplementedError, match="requires SM70"):
        _verify_e4m3_kv_requirements(vc, mc, cc)
    _force_sm70(monkeypatch)
    # Non-FP16 rejected.
    vc, mc, cc = _configs(dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="FP16 activations"):
        _verify_e4m3_kv_requirements(vc, mc, cc)
    # TP != 4 rejected.
    vc, mc, cc = _configs(tp=2)
    with pytest.raises(NotImplementedError, match="requires TP4"):
        _verify_e4m3_kv_requirements(vc, mc, cc)


def test_gate_non_e4m3_cache_is_noop(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_MTP", False)
    vc, mc, _ = _configs(spec=object())
    _verify_e4m3_kv_requirements(vc, mc, SimpleNamespace(cache_dtype="auto"))


# --------------------------------------------------------------------------- #
# D2: strict draft-side scale validation lists the missing tensor names.
# --------------------------------------------------------------------------- #
def test_validate_scale_overlay_lists_missing_names(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES", True)
    required = {
        "model.layers.0.self_attn.k_scale",
        "model.layers.0.self_attn.v_scale",
    }
    # Nothing loaded -> both sorted names are listed.
    with pytest.raises(ValueError, match="Missing:.*k_scale.*v_scale"):
        _validate_qsa_e4m3_scale_load(required, set(), E4M3)
    # Partial load -> only the genuinely missing name is listed.
    with pytest.raises(ValueError, match="v_scale"):
        _validate_qsa_e4m3_scale_load(
            required, {"model.layers.0.self_attn.k_scale"}, E4M3
        )
    # Complete -> no raise; non-e4m3 -> no-op even when incomplete.
    _validate_qsa_e4m3_scale_load(required, required, E4M3)
    _validate_qsa_e4m3_scale_load(required, set(), "auto")


def _make_qsa_stub(k_value, v_value, *, dtype=E4M3):
    """Real Qwen4ExpQSAAttention instance with only the scale slots populated."""
    stub = Qwen4ExpQSAAttention.__new__(Qwen4ExpQSAAttention)
    nn.Module.__init__(stub)
    stub.kv_cache_dtype = dtype
    stub.layer_name = "model.layers.0.self_attn"
    stub._qsa_kv_scales_finalized = dtype not in ("fp8", E4M3)
    stub.register_buffer("_k_scale", torch.tensor(0.0))
    stub.register_buffer("_v_scale", torch.tensor(0.0))
    if not stub._qsa_kv_scales_finalized:
        stub.k_scale = nn.Parameter(torch.tensor(float(k_value)), requires_grad=False)
        stub.v_scale = nn.Parameter(torch.tensor(float(v_value)), requires_grad=False)
    return stub


def test_validate_loaded_kv_scales_finalizes_floats():
    stub = _make_qsa_stub(0.125, 0.25)
    stub.validate_loaded_kv_scales()
    assert stub._qsa_kv_scales_finalized is True
    assert stub._k_scale_float == pytest.approx(0.125)
    assert stub._v_scale_float == pytest.approx(0.25)
    assert float(stub._k_scale) == pytest.approx(0.125)
    assert not hasattr(stub, "k_scale") and not hasattr(stub, "v_scale")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_validate_loaded_kv_scales_rejects_invalid(bad):
    stub = _make_qsa_stub(bad, 0.25)
    with pytest.raises(ValueError, match="calibrated scales are required"):
        stub.validate_loaded_kv_scales()


def test_validate_loaded_kv_scales_noop_for_fp16():
    stub = _make_qsa_stub(0.0, 0.0, dtype="float16")
    stub.validate_loaded_kv_scales()  # returns immediately, no finalize needed


# --------------------------------------------------------------------------- #
# D2 end-to-end: _finalize over a fake model holding real QSA instances.
# --------------------------------------------------------------------------- #
class _FakeInner(nn.Module):
    def __init__(self, stub):
        super().__init__()
        self.self_attn = stub


class _FakeModel(nn.Module):
    def __init__(self, stub):
        super().__init__()
        layer = nn.Module()
        layer.self_attn = stub
        self.layers = nn.ModuleList([layer])


def test_finalize_qsa_scale_load_success_and_missing(monkeypatch):
    monkeypatch.setattr(model_mod, "is_offload_process", lambda: False)
    stub = _make_qsa_stub(0.1, 0.2)
    container = _FakeModel(stub)
    # named_modules() -> "layers.0.self_attn"; required = {...k_scale, ...v_scale}
    loaded = {"layers.0.self_attn.k_scale", "layers.0.self_attn.v_scale"}
    _finalize_qsa_e4m3_scale_load(container, loaded, E4M3)
    assert stub._qsa_kv_scales_finalized is True

    stub2 = _make_qsa_stub(0.1, 0.2)
    container2 = _FakeModel(stub2)
    with pytest.raises(ValueError, match="Missing:.*self_attn"):
        _finalize_qsa_e4m3_scale_load(
            container2,
            {"layers.0.self_attn.k_scale"},
            E4M3,
            require_calibrated_speculative_draft=True,
        )


def test_finalize_qsa_scale_load_skips_offload_process(monkeypatch):
    monkeypatch.setattr(model_mod, "is_offload_process", lambda: True)
    stub = _make_qsa_stub(0.1, 0.2)
    container = _FakeModel(stub)
    # Missing scales, but offload process must skip entirely (no raise, no finalize).
    _finalize_qsa_e4m3_scale_load(container, set(), E4M3)
    assert stub._qsa_kv_scales_finalized is False


def test_finalize_noop_for_fp16_cache(monkeypatch):
    monkeypatch.setattr(model_mod, "is_offload_process", lambda: False)
    stub = _make_qsa_stub(0.0, 0.0, dtype="float16")
    container = _FakeModel(stub)
    _finalize_qsa_e4m3_scale_load(container, set(), "float16")


# --------------------------------------------------------------------------- #
# D3: draft weight-name remap + draft-visible shard selection.
# --------------------------------------------------------------------------- #
def test_remap_mtp_scale_names_to_draft_module_paths():
    assert (
        _remap_mtp_weight_name("mtp.layers.0.self_attn.k_scale")
        == "model.layers.0.self_attn.k_scale"
    )
    assert (
        _remap_mtp_weight_name("mtp.layers.0.self_attn.v_scale")
        == "model.layers.0.self_attn.v_scale"
    )
    # Target scales never start with "mtp." and are not rerouted by the drafter.
    assert _remap_mtp_weight_name("model.layers.5.self_attn.k_scale") is None
