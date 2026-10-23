# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
from unittest.mock import Mock

import pytest

from tests.config.runtime_policy_utils import make_policy_defaults
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.config.sm70_moe import bind_moe_diagnostics
from vllm.config.sm70_native import NATIVE_FIELDS
from vllm.model_executor.layers.quantization import sm70_turbomind as provider


def engine(backend):
    defaults = make_policy_defaults()
    defaults.cfg.kernel_config.layer_execution.quant_backend = backend
    defaults.finish()
    bind_moe_diagnostics(
        defaults.cfg.kernel_config, defaults.cfg.observability_config.runtime_trace
    )
    return defaults.cfg


@pytest.mark.parametrize("order", [("marlin", "turbomind"), ("turbomind", "marlin")])
def test_loader_worker_inputs_are_frozen_and_typed_backend_wins(monkeypatch, order):
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "bad-overridden")
    monkeypatch.setenv("VLLM_SM70_AWQ_TURBOMIND", "0")
    monkeypatch.setenv("VLLM_SM70_FP8_TURBOMIND", "0")
    monkeypatch.setenv("VLLM_SM70_NVFP4_TURBOMIND", "0")
    engines = [engine(backend) for backend in order]
    for cfg in engines:
        # Transfer the actual component configuration using the worker serializer.
        recv, send = multiprocessing.Pipe(duplex=False)
        send.send(cfg.kernel_config)
        cfg.kernel_config = recv.recv()
        send.close()
        recv.close()
    aliases = set()
    for cfg in engines:
        for family in ("awq", "fp8", "nvfp4"):
            aliases.update(getattr(cfg.kernel_config, "sm70_" + family).legacy.values)
        for family in ("awq", "fp8", "nvfp4", "mxfp4"):
            aliases.update(getattr(cfg.kernel_config.sm70_moe, family).legacy.values)
    for alias in aliases:
        monkeypatch.setenv(alias, "changed-after-initialization")
        monkeypatch.setitem(
            envs.environment_variables, alias, Mock(side_effect=AssertionError(alias))
        )
    for cfg, backend in list(zip(engines, order)) * 2:
        with set_current_vllm_config(cfg):
            assert provider.quant_backend() == backend
            for family in ("awq", "fp8", "nvfp4"):
                assert provider.format_enabled(family) == (backend == "turbomind")
            cfg.kernel_config.sm70_awq.resolve()
            cfg.kernel_config.sm70_fp8.resolve()
            cfg.kernel_config.sm70_moe.awq.resolve("awq")
            cfg.kernel_config.sm70_moe.fp8.resolve("fp8")
            cfg.kernel_config.sm70_moe.nvfp4.resolve()
            cfg.kernel_config.sm70_moe.mxfp4.resolve()


def test_native_lazy_format_bind_does_not_read_worker_environment(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8_M32_NATIVE", "1")
    kernel = KernelConfig()
    kernel.capture_provider_inputs()
    assert not kernel.sm70_fp8.resolved
    assert not kernel.sm70_fp8.native.values
    before = kernel.compute_hash()
    for _, alias, _, _ in NATIVE_FIELDS:
        monkeypatch.setenv(alias, "mutated-worker-input")
    native = kernel.sm70_fp8.native
    native.resolve("fp8")
    values = dict(zip((field for field, *_ in NATIVE_FIELDS), native.values))
    assert values["fp8_qpn8_m32_native"] == "1"
    assert (
        kernel.compute_hash() != before
    )  # Format activation, not capture, changes hash.


def test_loader_error_remains_behind_its_original_gate(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_MOE_DISABLE", "")
    cfg = engine("auto")
    cfg.kernel_config.sm70_awq.resolve()  # A dense-only layer does not consume this.
    with set_current_vllm_config(cfg):
        assert provider.format_enabled("awq")
        with pytest.raises(ValueError):
            provider.format_option("awq", "moe_disable")
    cfg.kernel_config.sm70_awq.moe_disable = False
    with set_current_vllm_config(cfg):
        assert not provider.format_option("awq", "moe_disable")


@pytest.mark.parametrize("family", ["awq", "fp8", "nvfp4"])
def test_explicit_format_flag_wins_but_raw_gate_keeps_legacy_meaning(
    monkeypatch, family
):
    monkeypatch.setenv("VLLM_SM70_" + family.upper() + "_TURBOMIND", "0")
    cfg = engine("turbomind")
    with set_current_vllm_config(cfg):
        assert not provider.format_option(family, "enabled")
        assert provider.format_enabled(family)
        getattr(cfg.kernel_config, "sm70_" + family).enabled = False
        assert not provider.format_enabled(family)
