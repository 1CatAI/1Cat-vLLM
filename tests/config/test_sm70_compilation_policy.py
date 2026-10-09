# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of SM70 policy defaults, before generic config validation.

Run without the model-loading parent fixtures or an installed vLLM build:
pytest --confcutdir=tests/config tests/config/test_sm70_compilation_policy.py
"""

import ast
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPILE_POLICY = "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH"
DECODE_POLICY = "VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE"


@pytest.fixture(scope="module")
def policy_module():
    from vllm import envs
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.config.policy_defaults import PolicyDefaults
    from vllm.model_executor.models.runtime_defaults import (
        _is_sm70_dflash2_verifier_contract,
        _is_sm70_qwen38_decode_compile_contract,
    )
    from vllm.platforms import runtime_defaults

    path = ROOT / "vllm/platforms/runtime_defaults.py"
    tree = ast.parse(path.read_text())
    post_init = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_runtime_policy_defaults"
    )
    start = next(
        i
        for i, node in enumerate(post_init.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "sm70_compile_disabled_by_user"
            for t in node.targets
        )
    )
    end = next(
        i
        for i, node in enumerate(post_init.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "sm70_flash_no_compile_graph"
    )
    policy = ast.parse("def apply_policy(self):\n    pass\n").body[0]
    assert isinstance(policy, ast.FunctionDef)
    policy.body = [
        *ast.parse("defaults = PolicyDefaults(self)").body,
        *post_init.body[start : end + 1],
        *ast.parse("defaults.finish()").body,
    ]
    module = ModuleType("_sm70_test_policy")
    module.__dict__.update(runtime_defaults.__dict__)
    module.__dict__.update(
        envs=envs,
        logger=Mock(),
        PolicyDefaults=PolicyDefaults,
        CompilationMode=CompilationMode,
        CUDAGraphMode=CUDAGraphMode,
        sm70_glm5_dflash_tp8_pp1_verifier=False,
        _is_sm70_dflash2_verifier_contract=_is_sm70_dflash2_verifier_contract,
        _is_sm70_qwen38_decode_compile_contract=(
            _is_sm70_qwen38_decode_compile_contract
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[policy], type_ignores=[])),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


@pytest.fixture
def policy(policy_module, monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability=lambda capability, device_id=0: capability == (7, 0),
    )
    monkeypatch.setattr(policy_module, "current_platform", platform, raising=False)
    monkeypatch.setattr(
        policy_module,
        "_any_participating_device_is_capability",
        lambda cfg, capability: platform.is_device_capability(capability),
        raising=False,
    )
    monkeypatch.setattr(
        policy_module,
        "_any_participating_device_is_pre_ampere",
        lambda cfg: True,
        raising=False,
    )
    monkeypatch.setattr(
        policy_module, "_participating_cuda_device_ids", lambda cfg: (0,), raising=False
    )
    policy_module.logger.reset_mock()
    return policy_module


def _config(**overrides: object) -> SimpleNamespace:
    compilation: dict[str, object] = dict(
        mode=None,
        cudagraph_mode=None,
        cudagraph_capture_sizes=None,
        max_cudagraph_capture_size=None,
        use_inductor_graph_partition=None,
        pass_config=SimpleNamespace(eliminate_noops=False),
        inductor_compile_config={},
    )
    compilation.update(overrides)
    cfg = SimpleNamespace(
        model_config=None,
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=SimpleNamespace(backend=None),
        scheduler_config=SimpleNamespace(max_num_seqs=10, max_num_batched_tokens=2048),
        kernel_config=SimpleNamespace(ir_op_priority=SimpleNamespace()),
        compilation_config=SimpleNamespace(**compilation),
        use_v2_model_runner=False,
    )
    from types import MethodType

    from vllm.config import AttentionConfig, KernelConfig, OffloadConfig, VllmConfig
    from vllm.config.execution_policy import CommunicationPolicy, GraphPolicy

    cfg.kernel_config = KernelConfig()
    cfg.attention_config = AttentionConfig()
    cfg.offload_config = OffloadConfig()
    cfg.compilation_config.runtime = GraphPolicy()
    cfg.parallel_config.communication = CommunicationPolicy()
    cfg.runtime_default_sources = {}
    cfg.apply_model_runtime_defaults = MethodType(
        VllmConfig.apply_model_runtime_defaults, cfg
    )
    return cfg


@pytest.mark.parametrize("policy_name", [COMPILE_POLICY, DECODE_POLICY])
@pytest.mark.parametrize(
    "mode", [None, "NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE", "VLLM_COMPILE"]
)
@pytest.mark.parametrize(
    "graph_mode",
    [None, "NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"],
)
def test_explicit_fields_survive_policy(
    policy, monkeypatch, policy_name, mode, graph_mode
):
    monkeypatch.setenv(policy_name, "1")
    compile_mode = getattr(policy.CompilationMode, mode) if mode is not None else None
    cudagraph_mode = (
        getattr(policy.CUDAGraphMode, graph_mode) if graph_mode is not None else None
    )
    config = _config(mode=compile_mode, cudagraph_mode=cudagraph_mode)
    policy.apply_policy(config)

    default_mode, default_graph = (
        ("VLLM_COMPILE", "FULL_AND_PIECEWISE")
        if policy_name == COMPILE_POLICY
        else ("NONE", "FULL_DECODE_ONLY")
    )
    assert config.compilation_config.mode == getattr(
        policy.CompilationMode, default_mode if mode is None else mode
    )
    assert config.compilation_config.cudagraph_mode == getattr(
        policy.CUDAGraphMode, default_graph if graph_mode is None else graph_mode
    )


def test_baseline_still_auto_enables_unset_fields(policy):
    assert COMPILE_POLICY not in os.environ
    config = _config()
    policy.apply_policy(config)
    assert config.compilation_config.runtime.compile_graph is True
    assert config.compilation_config.mode == policy.CompilationMode.VLLM_COMPILE
    assert (
        config.compilation_config.cudagraph_mode
        == policy.CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert config.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 10]
    assert config.compilation_config.max_cudagraph_capture_size == 10


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "NONE"),
        ("mode", "STOCK_TORCH_COMPILE"),
        ("cudagraph_mode", "NONE"),
        ("cudagraph_mode", "PIECEWISE"),
        ("cudagraph_mode", "FULL_AND_PIECEWISE"),
    ],
)
def test_automatic_baseline_preserves_explicit_fields(policy, field, value):
    assert COMPILE_POLICY not in os.environ
    modes = policy.CompilationMode if field == "mode" else policy.CUDAGraphMode
    explicit = getattr(modes, value)
    config = _config(**{field: explicit})
    policy.apply_policy(config)
    assert config.compilation_config.runtime.compile_graph is True
    assert getattr(config.compilation_config, field) == explicit


@pytest.mark.parametrize("policy_name", [COMPILE_POLICY, DECODE_POLICY])
def test_explicit_capture_settings_survive_policy(policy, monkeypatch, policy_name):
    monkeypatch.setenv(policy_name, "1")
    config = _config(
        cudagraph_capture_sizes=[7],
        max_cudagraph_capture_size=7,
        use_inductor_graph_partition=True,
    )
    policy.apply_policy(config)
    assert config.compilation_config.cudagraph_capture_sizes == [7]
    assert config.compilation_config.max_cudagraph_capture_size == 7
    assert config.compilation_config.use_inductor_graph_partition is True


@pytest.mark.parametrize("policy_name", [COMPILE_POLICY, DECODE_POLICY])
def test_policy_log_reports_preserved_modes(policy, monkeypatch, policy_name):
    monkeypatch.setenv(policy_name, "1")
    config = _config(
        mode=policy.CompilationMode.STOCK_TORCH_COMPILE,
        cudagraph_mode=policy.CUDAGraphMode.NONE,
    )
    policy.apply_policy(config)
    messages = [
        call.args[0] % call.args[1:] for call in policy.logger.info_once.call_args_list
    ]
    assert any(
        "policy:" in message
        and "mode=STOCK_TORCH_COMPILE, cudagraph_mode=NONE" in message
        for message in messages
    )


@pytest.mark.parametrize("policy_name", [COMPILE_POLICY, DECODE_POLICY])
def test_policy_requires_cuda(policy, monkeypatch, policy_name):
    monkeypatch.setenv(policy_name, "1")
    monkeypatch.setattr(policy.current_platform, "is_cuda", lambda: False)
    config = _config()
    policy.apply_policy(config)
    assert config.compilation_config.mode is None
    assert config.compilation_config.cudagraph_mode is None


@pytest.mark.parametrize("policy_name", [COMPILE_POLICY, DECODE_POLICY])
def test_policy_requires_sm70(policy, monkeypatch, policy_name):
    monkeypatch.setenv(policy_name, "1")
    monkeypatch.setattr(
        policy.current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability == (8, 0),
    )
    monkeypatch.setattr(
        policy, "_any_participating_device_is_pre_ampere", lambda cfg: False
    )
    config = _config()
    policy.apply_policy(config)
    assert config.compilation_config.mode is None
    assert config.compilation_config.cudagraph_mode is None


def test_explicit_baseline_opt_out_remains_effective(policy, monkeypatch):
    monkeypatch.setenv(COMPILE_POLICY, "0")
    config = _config()
    policy.apply_policy(config)
    assert os.environ[COMPILE_POLICY] == "0"
    assert config.compilation_config.mode is None
    assert config.compilation_config.cudagraph_mode is None


@pytest.mark.parametrize("policy_name", [COMPILE_POLICY, DECODE_POLICY])
@pytest.mark.parametrize("budget", [1, 2, 8, 50, 2048])
def test_policy_capture_cap_respects_token_budget(
    policy, monkeypatch, policy_name, budget
):
    monkeypatch.setenv(policy_name, "1")
    config = _config()
    config.scheduler_config.max_num_batched_tokens = budget
    policy.apply_policy(config)
    sizes = config.compilation_config.cudagraph_capture_sizes
    eligible = [size for size in sizes if size <= budget]
    assert config.compilation_config.max_cudagraph_capture_size == max(eligible)


@pytest.mark.parametrize("policy_name", [COMPILE_POLICY, DECODE_POLICY])
@pytest.mark.parametrize("budget,expected", [(3, 2), (7, 4), (50, 8)])
def test_policy_infers_cap_from_explicit_capture_sizes(
    policy, monkeypatch, policy_name, budget, expected
):
    monkeypatch.setenv(policy_name, "1")
    config = _config(cudagraph_capture_sizes=[2, 4, 8])
    config.scheduler_config.max_num_batched_tokens = budget
    policy.apply_policy(config)
    assert config.compilation_config.cudagraph_capture_sizes == [2, 4, 8]
    assert config.compilation_config.max_cudagraph_capture_size == expected
