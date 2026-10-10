# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered engine defaults, worker transfer and execution isolation without weights."""

import multiprocessing
import os
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    KernelConfig,
    ObservabilityConfig,
    OffloadConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.execution_policy import GraphPolicy, graph_policy, layer_policy
from vllm.config.policy_defaults import (
    finalize_runtime_policy_hashes,
    runtime_policy_report,
)
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.platforms import runtime_defaults
from vllm.platforms.interface import DeviceCapability
from vllm.runtime_resources import runtime_resources_for


@pytest.fixture(autouse=True)
def clean_legacy(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name)
    envs.disable_envs_cache()


def engine(*, model="qwen", tp=4, pp=1, spec=None, **graph):
    parallel = ParallelConfig(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        distributed_executor_backend="mp",
    )
    text = NS(
        model_type="glm5_next" if model == "glm" else "qwen4_exp",
        num_hidden_layers=45,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        hidden_size=5120,
        ple_layer_ids=[0, 1],
    )
    architecture = "Glm5NextForCausalLM" if model == "glm" else "Qwen4ExpForCausalLM"
    model_cfg = NS(
        architecture=architecture,
        architectures=[architecture],
        hf_text_config=text,
        multimodal_config=None,
        enforce_eager=False,
        dtype=torch.float16,
        quantization="modelopt_fp4" if model == "glm" else None,
        is_nvfp4_quantized=lambda: model == "glm",
    )
    if spec is not None:
        spec = NS(
            method=spec,
            draft_sample_method="probabilistic",
            num_speculative_tokens=7,
            num_speculative_state_tokens=lambda: 7,
            sm70_dflash2=Sm70DFlash2Config(),
            draft_model_config=NS(hf_config=NS(dflash_config={"selector_top_k": 16})),
        )
    cfg = NS(
        model_config=model_cfg,
        speculative_config=spec,
        parallel_config=parallel,
        kernel_config=KernelConfig(),
        offload_config=OffloadConfig(),
        compilation_config=CompilationConfig(runtime=GraphPolicy(**graph)),
        attention_config=AttentionConfig(),
        observability_config=ObservabilityConfig(),
        cache_config=CacheConfig(),
        device_config=DeviceConfig(device="cuda"),
        runtime_default_sources={},
        scheduler_config=NS(
            enable_chunked_prefill=True, max_num_seqs=8, max_num_batched_tokens=2048
        ),
        use_v2_model_runner=True,
    )
    cfg.apply_model_runtime_defaults = MethodType(
        VllmConfig.apply_model_runtime_defaults, cfg
    )
    return cfg


def apply(cfg, monkeypatch, capability=70):
    import vllm.platforms

    cap = DeviceCapability(capability // 10, capability % 10)
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        NS(
            is_cuda=lambda: True,
            uses_host_device_handling=lambda: False,
            device_count=lambda: 8,
            get_device_capability=lambda: cap,
            is_device_capability=lambda value, device_id=0: (value == tuple(cap)),
        ),
    )
    runtime_defaults.apply_runtime_policy_defaults(cfg)
    finalize_runtime_policy_hashes(cfg)


@pytest.mark.parametrize("reverse", [False, True])
def test_engines_do_not_inherit_auto_defaults(monkeypatch, reverse):
    fast = engine(spec="mtp")
    disabled = engine(compile_graph=False, dual_compile=False)
    original = dict(os.environ)
    for cfg in [disabled, fast] if reverse else [fast, disabled]:
        apply(cfg, monkeypatch)
    assert os.environ == original
    assert fast.compilation_config.runtime.compile_graph
    assert fast.compilation_config.runtime.dual_compile
    assert fast.compilation_config.runtime.split_draft_graphs
    assert fast.kernel_config.layer_execution.fp16_gemv
    assert fast.offload_config.ple.disk
    assert not disabled.compilation_config.runtime.compile_graph
    assert not disabled.kernel_config.layer_execution.fp16_gemv
    assert not disabled.offload_config.ple.disk
    assert fast.compilation_config.cudagraph_capture_sizes == [8, 16, 24, 32, 48, 64]


def test_explicit_typed_choices_override_legacy_without_writing_env(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    cfg = engine(compile_graph=False, sm70_breakable=True, breakable=False)
    original = dict(os.environ)
    apply(cfg, monkeypatch)
    assert not cfg.compilation_config.runtime.compile_graph
    assert not cfg.compilation_config.runtime.breakable
    assert os.environ == original
    assert (
        runtime_policy_report(cfg)["owners"]["compilation_config.runtime"]["sources"][
            "compile_graph"
        ]
        == "typed"
    )


@pytest.mark.parametrize(
    "tp,pp,temp,partition", [(8, 1, 0.9, None), (4, 2, 0.8, "24,21")]
)
def test_glm_defaults_preserve_checkpoint_order(monkeypatch, tp, pp, temp, partition):
    cfg = engine(model="glm", tp=tp, pp=pp, spec="dflash")
    original = dict(os.environ)
    apply(cfg, monkeypatch)
    assert cfg.speculative_config.sm70_dflash2.proposal_temperature_scale == temp
    assert cfg.speculative_config.sm70_dflash2.proposal_top_p == 0.95
    assert cfg.parallel_config.communication.pp_layer_partition == partition
    if tp == 8:
        assert cfg.parallel_config.communication.tp8_hierarchical
        assert not cfg.compilation_config.runtime.aot_compile
        assert not cfg.kernel_config.sm70_moe.nvfp4.glm53_qpn_w13
        assert cfg.kernel_config.layer_execution.mhc_pre_threads == 1024
        assert not cfg.speculative_config.sm70_dflash2.sparse_target_rejection
    else:
        assert not cfg.parallel_config.communication.tp4_push
    assert os.environ == original


def _worker_roundtrip(connection, cfg):
    connection.send(cfg)
    connection.close()


def test_worker_serialization_and_forward_borrowing(monkeypatch):
    from vllm.forward_context import set_forward_context

    cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(runtime=GraphPolicy(dual_compile=True)),
    )
    other = VllmConfig(device_config=DeviceConfig(device="cpu"))
    parent_resources = runtime_resources_for(cfg)
    # A worker handle cannot be pickled; the transfer must not even visit it.
    parent_resources["live_handle"] = lambda: None
    parent_resources["diagnostics"].histories["attention"] = {"decode": 7}
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker_roundtrip, args=(sender, cfg))
    process.start()
    sender.close()
    try:
        assert receiver.poll(60), "worker configuration transfer timed out"
        transferred = receiver.recv()
    finally:
        receiver.close()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    worker_resources = runtime_resources_for(transferred)
    assert "live_handle" not in worker_resources
    assert "attention" not in worker_resources["diagnostics"].histories
    assert parent_resources["diagnostics"].histories["attention"] == {"decode": 7}
    assert transferred.compilation_config.runtime.dual_compile
    assert (
        transferred.compilation_config.runtime.sources
        == cfg.compilation_config.runtime.sources
    )
    with set_current_vllm_config(other), set_forward_context(None, transferred):
        assert graph_policy() is transferred.compilation_config.runtime
        assert layer_policy() is transferred.kernel_config.layer_execution
    assert runtime_resources_for(other) is not runtime_resources_for(transferred)


def test_unused_graph_and_resource_controls_do_not_salt_hash(monkeypatch):
    first = engine(estimate_graph_memory=False, split_draft_graphs=False)
    second = engine(estimate_graph_memory=True, split_draft_graphs=True)
    apply(first, monkeypatch, capability=80)
    apply(second, monkeypatch, capability=80)
    assert (
        first.compilation_config.compute_hash()
        == second.compilation_config.compute_hash()
    )
    second.compilation_config.runtime.aot_compile = (
        not first.compilation_config.runtime.aot_compile
    )
    assert (
        first.compilation_config.compute_hash()
        != second.compilation_config.compute_hash()
    )


def test_compile_factors_use_effective_policy_not_legacy_input(monkeypatch):
    from vllm.config.utils import hash_factors

    typed = engine(compile_graph=False, aot_compile=False)
    apply(typed, monkeypatch)
    before = hash_factors(envs.compile_factors(vllm_config=typed))
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "0")
    legacy = engine()
    apply(legacy, monkeypatch)
    assert (
        typed.compilation_config.compute_hash()
        == legacy.compilation_config.compute_hash()
    )
    assert before == hash_factors(envs.compile_factors(vllm_config=legacy))
    # Mutating a legacy input cannot change a prepared engine's cache identity.
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")
    assert before == hash_factors(envs.compile_factors(vllm_config=typed))
    assert "VLLM_USE_AOT_COMPILE" in envs.compile_factors()


@pytest.mark.parametrize("raw,expected", [("", True), ("false", True), ("0", False)])
def test_glm_projection_parser_preserves_nonzero_string_semantics(
    monkeypatch, raw, expected
):
    from vllm.config.execution_policy import LayerExecutionPolicy

    monkeypatch.setenv("VLLM_SM70_GLM53_TP8_CUBLASLT", raw)
    policy = LayerExecutionPolicy()
    policy.resolve()
    assert policy.glm_cublaslt is expected


@pytest.mark.parametrize("raw,expected", [(None, True), (" yes ", True), ("2", False)])
def test_bf16_emulation_retains_model_parser(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("VLLM_SM70_DFLASH2_BF16_EMULATION", raw)
    policy = Sm70DFlash2Config()
    policy.resolve(qualified=False)
    assert policy.bf16_emulation is expected


@pytest.mark.parametrize(
    "raw,expected", [("128tail", 128), ("garbage", 256), ("17", 256)]
)
def test_native_thread_policy_retains_atoi_fallback(monkeypatch, raw, expected):
    from vllm.config.execution_policy import LayerExecutionPolicy

    monkeypatch.setenv("VLLM_SM70_GLM_MHC_PRE_THREADS", raw)
    policy = LayerExecutionPolicy()
    policy.resolve()
    assert policy.mhc_pre_threads == expected


@pytest.mark.parametrize("raw,expected", [("128tail", 128), ("garbage", 0), ("-1", 0)])
def test_dense_native_tuning_preserves_parser(monkeypatch, raw, expected):
    from vllm.config.execution_policy import read_execution_legacy

    monkeypatch.setenv("VLLM_SM70_FP8_DENSE_TUNE_MAX_M", raw)
    assert read_execution_legacy("VLLM_SM70_FP8_DENSE_TUNE_MAX_M") == expected


def test_warmup_consumes_engine_limit_after_defaults(monkeypatch):
    from vllm.model_executor.warmup.awq_sm70_warmup import _get_decode_m_values

    cfg = engine()
    apply(cfg, monkeypatch)
    monkeypatch.setenv("VLLM_SM70_AWQ_WARMUP_MAX_M", "16")
    assert 64 in _get_decode_m_values(NS(vllm_config=cfg))


def test_fullgraph_policy_compatibility_and_engine_isolation(monkeypatch):
    from vllm.forward_context import set_forward_context

    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")

    def run(value):
        return value + (1 if graph_policy().compile_graph else 0)

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    value = torch.zeros(1)
    assert torch.equal(compiled(value), value + 1)
    first = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(runtime=GraphPolicy(compile_graph=True)),
    )
    second = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(runtime=GraphPolicy(compile_graph=False)),
    )
    for cfg, expected in ((first, value + 1), (second, value), (first, value + 1)):
        with set_current_vllm_config(cfg), set_forward_context(None, cfg):
            assert torch.equal(compiled(value), expected)


def test_fp16_decode_partition_is_computation_and_fp8_wave_is_unused(monkeypatch):
    first = engine(decode_partition_size=256, e4m3_p512_begin=100)
    second = engine(decode_partition_size=512, e4m3_p512_begin=200)
    apply(first, monkeypatch)
    apply(second, monkeypatch)
    assert (
        first.compilation_config.compute_hash()
        != second.compilation_config.compute_hash()
    )
    second.compilation_config.runtime.decode_partition_size = 256
    assert (
        first.compilation_config.compute_hash()
        == second.compilation_config.compute_hash()
    )


def test_diagnostic_report_is_json_safe_without_environment_getters(monkeypatch):
    import json

    from vllm.config.observability import ObservabilityConfig

    cfg = engine()
    cfg.observability_config = ObservabilityConfig()
    apply(cfg, monkeypatch)

    def forbidden():
        raise AssertionError("report must not evaluate environment getters")

    for name in envs.environment_variables:
        monkeypatch.setitem(envs.environment_variables, name, forbidden)
    report = runtime_policy_report(cfg)
    assert report["diagnostic_channels"]["qwen_layer"]["policy"]["filters"][
        "layers"
    ] == [0, 1]
    json.dumps(report)


@pytest.mark.parametrize("capability,active", [(70, True), (75, False), (80, False)])
def test_only_sm70_hashes_its_triton_schedule(monkeypatch, capability, active):
    first, second = engine(), engine()
    second.attention_config.sm70_triton.num_warps = 2
    apply(first, monkeypatch, capability=capability)
    apply(second, monkeypatch, capability=capability)
    assert first.attention_config.sm70_triton.active is active
    assert (
        first.attention_config.compute_hash() != second.attention_config.compute_hash()
    ) is active


def test_unused_model_and_quantization_policies_do_not_salt_other_engines(monkeypatch):
    first, second = engine(), engine()
    for cfg in (first, second):
        cfg.model_config.architecture = "LlamaForCausalLM"
        cfg.model_config.quantization = None
        cfg.model_config.hf_text_config.ple_layer_ids = []
    second.kernel_config.layer_execution.glm_fused_fg_b = True
    second.kernel_config.layer_execution.ple_spec_conv = True
    second.kernel_config.layer_execution.mxfp4_turbomind = False
    apply(first, monkeypatch)
    apply(second, monkeypatch)
    assert first.kernel_config.compute_hash() == second.kernel_config.compute_hash()


def test_all_captured_gdn_and_diagnostic_aliases_leave_environment_hash(monkeypatch):
    from unittest.mock import Mock

    cfg = engine()
    apply(cfg, monkeypatch)
    # Dormant GDN has no computation hash; an admitted GDN hashes its own policy.
    aliases = set()
    for owner in (
        cfg.kernel_config.gdn,
        cfg.kernel_config.sm70_runtime,
        cfg.observability_config.runtime_trace,
        cfg.observability_config.step_profiler,
        cfg.observability_config.spec_decode_trace,
        cfg.observability_config.gdn_profile,
        cfg.observability_config.gdn_state,
    ):
        aliases.update(owner.compile_ignored_aliases())
    readers = {}
    for name in aliases & envs.environment_variables.keys():
        readers[name] = Mock(side_effect=AssertionError(name))
        monkeypatch.setitem(envs.environment_variables, name, readers[name])
    before = envs.compile_factors(cfg.kernel_config, vllm_config=cfg)
    for name in aliases:
        monkeypatch.setenv(name, "1")
    assert envs.compile_factors(cfg.kernel_config, vllm_config=cfg) == before
    for reader in readers.values():
        reader.assert_not_called()


def test_native_shared_stage_aliases_use_captured_owner_hash(monkeypatch):
    from unittest.mock import Mock

    from vllm.config.sm70_native import NATIVE_FIELDS

    cfg = engine()
    apply(cfg, monkeypatch)
    aliases = {alias for _, alias, _, _ in NATIVE_FIELDS}
    readers = {}
    for name in aliases & envs.environment_variables.keys():
        readers[name] = Mock(side_effect=AssertionError(name))
        monkeypatch.setitem(envs.environment_variables, name, readers[name])
    before = envs.compile_factors(cfg.kernel_config, vllm_config=cfg)
    for name in aliases:
        monkeypatch.setenv(name, "1")
    assert envs.compile_factors(cfg.kernel_config, vllm_config=cfg) == before
    for reader in readers.values():
        reader.assert_not_called()


def test_inactive_speculation_and_gated_norm_do_not_salt_other_engines(monkeypatch):
    from unittest.mock import Mock

    from vllm.config.sm70_dflash2 import speculation_compile_ignored_aliases

    cfg = engine()
    apply(cfg, monkeypatch)
    aliases = speculation_compile_ignored_aliases(None)
    aliases.update(cfg.kernel_config.sm70_rmsnorm_gated_aliases.values())
    readers = {}
    for name in aliases & envs.environment_variables.keys():
        readers[name] = Mock(side_effect=AssertionError(name))
        monkeypatch.setitem(envs.environment_variables, name, readers[name])
    before = envs.compile_factors(cfg.kernel_config, vllm_config=cfg)
    for name in aliases:
        monkeypatch.setenv(name, "1")
    assert envs.compile_factors(cfg.kernel_config, vllm_config=cfg) == before
    for reader in readers.values():
        reader.assert_not_called()
