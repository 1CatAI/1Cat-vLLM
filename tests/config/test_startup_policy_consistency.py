# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup checkpoints must consume the same policy as execution. No weights."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import vllm.config.vllm as config_module
from vllm import envs
from vllm.config import CompilationConfig, DeviceConfig, KernelConfig, VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.execution_policy import GraphPolicy
from vllm.config.policy_defaults import engine_policy_aliases
from vllm.config.sm70_dflash2 import DFlashLookupPolicy, Sm70DFlash2Config
from vllm.config.speculative import uses_adaptive_dflash_lookup

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/config/vllm.py"
TREE = ast.parse(SOURCE.read_text())
CLASS = next(
    n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "VllmConfig"
)
POST = next(
    n
    for n in CLASS.body
    if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def checkpoint(nodes, **bindings):
    """Execute unchanged source nodes, substituting only device/model metadata."""
    namespace = dict(vars(config_module), **bindings)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize(
    "typed,legacy", [(True, "0"), (True, "1"), (False, "0"), (False, "1")]
)
def test_compile_range_obeys_final_graph_policy(monkeypatch, typed, legacy):
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", legacy)
    c = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        runtime=GraphPolicy(compile_graph=typed),
    )
    c.pass_config.fuse_allreduce_rms = False
    c.pass_config.enable_sp = False
    c.pass_config.fuse_rope_kvcache = False
    cfg = NS(compilation_config=c, scheduler_config=NS(max_num_batched_tokens=2048))
    VllmConfig._set_compile_ranges(cfg)
    assert c.compile_ranges_endpoints == [2048 + int(typed)]


@pytest.mark.parametrize(
    "typed,legacy", [(True, "0"), (True, "1"), (False, "0"), (False, "1")]
)
def test_dflash_startup_guard_matches_execution_policy(monkeypatch, typed, legacy):
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", legacy)
    policy = Sm70DFlash2Config(lookup=DFlashLookupPolicy(adaptive=typed))
    spec = NS(
        method="dflash",
        ngram_assist=True,
        num_speculative_tokens=15,
        disable_padded_drafter_batch=False,
        draft_model_config=NS(
            hf_config=NS(dflash_config={"block_size": 8, "selector_top_k": 16})
        ),
        sm70_dflash2=policy,
    )
    policy.resolve_lookup(spec)
    assert uses_adaptive_dflash_lookup(spec)
    assert policy.lookup.adaptive == typed
    cfg = NS(speculative_config=spec, scheduler_config=NS(async_scheduling=True))
    index = next(
        i
        for i, n in enumerate(POST.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "adaptive_dflash_lookup"
            for t in n.targets
        )
    )
    try:
        checkpoint(
            POST.body[index : index + 2],
            self=cfg,
            executor_supports_async_sched=True,
            dflash_ddtree_tree_verify=False,
            executor_backend="mp",
        )
        rejected = False
    except ValueError as error:
        assert "adaptive q8/q16 lookup verification" in str(error)
        rejected = True
    assert rejected == policy.lookup.adaptive


@pytest.mark.parametrize(
    "typed,legacy",
    [(True, "0"), (True, "1"), (False, "0"), (False, "1"), (False, "invalid")],
)
def test_fp8_custom_op_gate_obeys_resolved_policy(monkeypatch, typed, legacy):
    monkeypatch.setenv("VLLM_SM70_FP8_TURBOMIND", legacy)
    kernel = KernelConfig(sm70_fp8={"enabled": typed})
    kernel.sm70_fp8.resolve()
    assert kernel.sm70_fp8.enabled == typed
    cfg = NS(
        kernel_config=kernel,
        model_config=NS(quantization="fp8"),
        quant_config=NS(weight_block_size=[128, 128]),
    )
    nodes = [
        n
        for n in POST.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in ("has_blocked_weights", "enable_quant_fp8_custom_op_for_blocked_weights")
    ]
    ns = checkpoint(
        nodes,
        self=cfg,
        current_platform=NS(is_cuda=lambda: True),
        _any_participating_device_is_capability=lambda *a: True,
    )
    assert ns["enable_quant_fp8_custom_op_for_blocked_weights"]() == (not typed)


def test_environment_cache_respects_overridden_alias(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FP8_TURBOMIND", "invalid")
    kernel = KernelConfig(sm70_fp8={"enabled": False})
    kernel.sm70_fp8.resolve()
    assert kernel.sm70_fp8.enabled is False
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"), kernel_config=kernel)
    name = "VLLM_SM70_FP8_TURBOMIND"
    # Keep the real invalid parser and a process-scoped getter. Isolating the
    # registry avoids unrelated optional feature imports, not policy parsing.
    monkeypatch.setattr(
        envs,
        "environment_variables",
        {
            name: envs.environment_variables[name],
            "VLLM_HOST_IP": envs.environment_variables["VLLM_HOST_IP"],
        },
    )
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    envs.enable_envs_cache(exclude=engine_policy_aliases(cfg))
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.2")
    assert envs.VLLM_HOST_IP == "127.0.0.1"
    assert cfg.kernel_config.sm70_fp8.enabled is False
    # Only a deliberate standalone compatibility call may evaluate the alias.
    with pytest.raises(ValueError, match="invalid literal"):
        envs.__getattr__(name)


def test_failed_environment_cache_is_not_published(monkeypatch):
    name = "VLLM_SM70_FP8_TURBOMIND"
    monkeypatch.setattr(
        envs,
        "environment_variables",
        {
            name: envs.environment_variables[name],
        },
    )
    monkeypatch.setenv(name, "invalid")
    with pytest.raises(ValueError, match="invalid literal"):
        envs.enable_envs_cache()
    assert not envs._is_envs_cache_enabled()
    monkeypatch.setenv(name, "0")
    envs.enable_envs_cache()
    assert envs.__getattr__(name) is False


def test_engine_cache_never_reparses_owned_aliases(monkeypatch):
    from unittest.mock import Mock

    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    aliases = engine_policy_aliases(cfg)
    for name in aliases & envs.environment_variables.keys():
        monkeypatch.setitem(
            envs.environment_variables,
            name,
            Mock(side_effect=AssertionError(f"re-read engine alias: {name}")),
        )
    envs.enable_envs_cache(exclude=aliases)
    assert envs._is_envs_cache_enabled()


def test_nested_process_getters_share_one_cached_value(monkeypatch):
    from unittest.mock import Mock

    getter = Mock(side_effect=[1, 2])
    monkeypatch.setattr(
        envs,
        "environment_variables",
        {"BASE": getter, "DEPENDENT": lambda: envs.__getattr__("BASE")},
    )
    envs.enable_envs_cache()
    assert envs.__getattr__("BASE") == envs.__getattr__("DEPENDENT") == 1
    getter.assert_called_once()


def test_lookup_startup_binds_only_scheduling_constraint(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", "0")
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_NSTRONG", "invalid")
    policy = DFlashLookupPolicy()
    assert not policy.resolve_adaptive()
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", "1")
    assert not policy.resolve_adaptive()
    with pytest.raises(ValueError, match="invalid literal"):
        policy.resolve()
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_NSTRONG", "6")
    policy.resolve()
    assert not policy.adaptive
    assert policy.sources["adaptive"] == "VLLM_DFLASH2_LOOKUP_ADAPTIVE"
