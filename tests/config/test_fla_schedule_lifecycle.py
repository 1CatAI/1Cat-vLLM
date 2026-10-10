# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from vllm.config import KernelConfig, set_current_vllm_config
from vllm.config.gdn_schedule import resolve_schedule
from vllm.model_executor.layers.fla.ops import gdn_chunk_kernels as factory


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_fla_launches_bind_once_per_engine_without_import_snapshots(monkeypatch, order):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 0))
    monkeypatch.setattr(
        factory,
        "_retune",
        lambda template, configs, key, **kw: SimpleNamespace(
            configs=configs, cache={}, key=key
        ),
    )
    owners = []
    for enabled in order:
        monkeypatch.setenv("VLLM_SM70_KDA_PREFILL_SCHEDULE", str(int(enabled)))
        cfg = SimpleNamespace(kernel_config=KernelConfig())
        cfg.kernel_config.capture_provider_inputs()
        cfg.kernel_config.resolve_gdn(
            SimpleNamespace(architecture="KimiLinearForCausalLM"), {}
        )
        owners.append(cfg)
    getter = os.getenv
    environ_get = os.environ.get

    def forbidden(name, *args):
        assert not name.startswith(
            (
                "VLLM_SM70_KDA_",
                "VLLM_SM70_GDN_",
                "VLLM_SM70_FLA_",
                "VLLM_SM70_FUSED_SIGMOID_",
            )
        ), name
        return getter(name, *args)

    def forbidden_environ(name, *args):
        assert not name.startswith(
            (
                "VLLM_SM70_KDA_",
                "VLLM_SM70_GDN_",
                "VLLM_SM70_FLA_",
                "VLLM_SM70_FUSED_SIGMOID_",
            )
        ), name
        return environ_get(name, *args)

    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(os.environ, "get", forbidden_environ)
    for module in (
        "chunk_delta_h",
        "chunk_o",
        "chunk_scaled_dot_kkt",
        "fused_recurrent",
        "fused_sigmoid_gating",
        "kda",
    ):
        importlib.import_module("vllm.model_executor.layers.fla.ops." + module)
    bindings = []
    for cfg, enabled in zip(owners, order):
        bound = factory.bind_kda_kernels(cfg)
        bindings.append(bound)
        assert len(bound.recompute.configs) == (2 if enabled else 9)
        assert bound is factory.bind_kda_kernels(cfg)
        assert (
            bound.delta_h
            is factory.bind_chunk_kernels(cfg, cfg.kernel_config.gdn.schedule).delta_h
        )
        with set_current_vllm_config(cfg):
            assert resolve_schedule() is cfg.kernel_config.gdn.schedule
            assert factory.resolve_kda_kernels() is bound
    bindings[0].recompute.cache["shape"] = "winner"
    assert bindings[1].recompute.cache == {}
    assert (
        owners[0].kernel_config.compute_hash() != owners[1].kernel_config.compute_hash()
    )


def test_inactive_schedule_fields_do_not_change_other_models_hash(monkeypatch):
    hashes = {}
    for architecture in (
        "KimiLinearForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "LlamaForCausalLM",
    ):
        results = []
        for enabled in (False, True):
            kernel = KernelConfig()
            kernel.gdn.schedule.kda_prefill_enabled = enabled
            kernel.capture_provider_inputs()
            model = SimpleNamespace(architecture=architecture)
            if architecture.startswith("Qwen"):
                model.hf_text_config = SimpleNamespace(linear_key_head_dim=128)
            kernel.resolve_gdn(model, {})
            results.append(kernel.compute_hash())
        hashes[architecture] = results
    assert hashes["KimiLinearForCausalLM"][0] != hashes["KimiLinearForCausalLM"][1]
    assert (
        hashes["Qwen3_5ForConditionalGeneration"][0]
        == hashes["Qwen3_5ForConditionalGeneration"][1]
    )
    assert hashes["LlamaForCausalLM"][0] == hashes["LlamaForCausalLM"][1]


def test_fla_import_has_no_legacy_schedule_snapshot():
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import os
            import torch
            original = os.environ.get
            def check(name, *args):
                prefixes = ("VLLM_SM70_KDA_", "VLLM_SM70_GDN_", "VLLM_SM70_FLA_",
                            "VLLM_SM70_FUSED_SIGMOID_")
                assert not name.startswith(prefixes), name
                return original(name, *args)
            os.environ.get = check
            from vllm.model_executor.layers.fla.ops import (
                chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, fused_recurrent,
                fused_sigmoid_gating, kda,
            )
        """),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_recurrent_and_sigmoid_launch_overrides_remain_effective(monkeypatch):
    import torch

    from vllm.config.gdn_schedule import GdnScheduleConfig
    from vllm.model_executor.layers.fla.ops import fused_recurrent as recurrent
    from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as sigmoid

    monkeypatch.setattr(
        recurrent.triton,
        "next_power_of_2",
        lambda n: 1 << (n - 1).bit_length(),
        raising=False,
    )
    policy = GdnScheduleConfig(
        recurrent_bv=7,
        recurrent_warps=3,
        sigmoid_bv=15,
        sigmoid_warps=7,
        sigmoid_stages=2,
    )
    policy.resolve()
    device = torch.device("cpu")
    assert recurrent._select_sm70_bv(128, 1, 12, device, policy) == 8
    assert recurrent._select_sm70_num_warps(8, 1, 12, policy) == 4
    assert sigmoid._select_fused_sigmoid_schedule(128, 1, 12, device, policy) == (
        16,
        8,
        2,
    )
