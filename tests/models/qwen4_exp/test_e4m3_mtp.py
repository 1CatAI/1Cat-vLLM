# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for calibrated E4M3 cache with MTP.

Native attention and capture controls live in the neighboring GPU suites.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import model as model_mod
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


@pytest.mark.parametrize(
    "capability,dtype",
    [
        (70, torch.float16),
        (75, torch.float16),
        (80, torch.bfloat16),
        (90, torch.float16),
    ],
)
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("spec", [None, _MTP_SPEC])
def test_e4m3_admission_uses_operator_capability(
    monkeypatch, capability, dtype, tp, spec
):
    from vllm.models.qwen4_exp.nvidia.ops import qsa as ops

    monkeypatch.setattr(
        ops,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            has_device_capability=lambda minimum: capability >= minimum,
        ),
    )
    vc, mc, cc = _configs(dtype=dtype, tp=tp, spec=spec)
    _verify_e4m3_kv_requirements(vc, mc, cc)


@pytest.mark.parametrize(
    "capability,dtype",
    [
        (60, torch.float16),
        (70, torch.bfloat16),
        (75, torch.bfloat16),
        (80, torch.float32),
    ],
)
def test_e4m3_unsupported_tensor_core_dtype_reports_reason(
    monkeypatch, capability, dtype
):
    from vllm.models.qwen4_exp.nvidia.ops import qsa as ops

    monkeypatch.setattr(
        ops,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            has_device_capability=lambda minimum: capability >= minimum,
        ),
    )
    vc, mc, cc = _configs(dtype=dtype)
    with pytest.raises(NotImplementedError, match="QSA E4M3 cache unavailable"):
        _verify_e4m3_kv_requirements(vc, mc, cc)


def test_e4m3_gate_non_e4m3_cache_is_noop():
    vc, mc, _ = _configs(spec=object())
    _verify_e4m3_kv_requirements(vc, mc, SimpleNamespace(cache_dtype="auto"))


# --------------------------------------------------------------------------- #
# D2: strict draft-side scale validation lists the missing tensor names.
# --------------------------------------------------------------------------- #
def test_validate_scale_overlay_lists_missing_names(monkeypatch):
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES", "1")
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


def test_automatic_e4m3_refuses_unit_scale_fallback(monkeypatch):
    monkeypatch.setattr(model_mod, "is_offload_process", lambda: False)
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES", "0")
    stub = _make_qsa_stub(0.1, 0.2)
    container = _FakeModel(stub)
    with pytest.raises(ValueError, match="Automatically selected QSA E4M3"):
        _finalize_qsa_e4m3_scale_load(
            container,
            {"layers.0.self_attn.k_scale"},
            E4M3,
            require_calibrated_target=True,
        )
    assert stub._qsa_kv_scales_finalized is False
    _finalize_qsa_e4m3_scale_load(
        container,
        {"layers.0.self_attn.k_scale", "layers.0.self_attn.v_scale"},
        E4M3,
        require_calibrated_target=True,
    )
    assert stub._qsa_kv_scales_finalized is True
