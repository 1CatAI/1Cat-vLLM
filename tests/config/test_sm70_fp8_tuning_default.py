# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.sm70_native import NATIVE_FIELDS, UNSET, Sm70NativeConfig

ALIAS = "VLLM_SM70_FP8_TUNE_SMALL_SHAPES"


def _resolved(monkeypatch, legacy=None, **typed):
    monkeypatch.delenv(ALIAS, raising=False)
    if legacy is not None:
        monkeypatch.setenv(ALIAS, legacy)
    native = Sm70NativeConfig(**typed)
    native.resolve("fp8")
    return native


def test_effective_default_matches_explicit_disable(monkeypatch):
    default = _resolved(monkeypatch)
    typed = _resolved(monkeypatch, fp8_tune_small_shapes=False)
    legacy = _resolved(monkeypatch, "0")
    assert default.values == typed.values == legacy.values
    assert default.hash_options() == typed.hash_options() == legacy.hash_options()
    assert default.hash_options()["fp8_tune_small_shapes"] == "0"
    assert default.sources["fp8_tune_small_shapes"] == "default"


def test_explicit_and_legacy_opt_in_remain_available(monkeypatch):
    legacy = _resolved(monkeypatch, "1")
    typed = _resolved(monkeypatch, fp8_tune_small_shapes=True)
    disabled = _resolved(monkeypatch, "1", fp8_tune_small_shapes=False)
    assert legacy.values == typed.values
    assert legacy.hash_options()["fp8_tune_small_shapes"] == "1"
    assert disabled.hash_options()["fp8_tune_small_shapes"] == "0"
    assert disabled.sources["fp8_tune_small_shapes"] == "configuration"


def test_engines_keep_independent_captured_defaults(monkeypatch):
    first = _resolved(monkeypatch)
    second = _resolved(monkeypatch, "1")
    first.resolve("fp8")
    assert first.hash_options()["fp8_tune_small_shapes"] == "0"
    assert second.hash_options()["fp8_tune_small_shapes"] == "1"


def test_fp8_default_does_not_change_other_families(monkeypatch):
    monkeypatch.delenv(ALIAS, raising=False)
    native = Sm70NativeConfig()
    native.resolve("nvfp4")
    index = next(
        i
        for i, field in enumerate(NATIVE_FIELDS)
        if field[0] == "fp8_tune_small_shapes"
    )
    assert native.values[index] == UNSET
    assert "fp8_tune_small_shapes" not in native.hash_options()
