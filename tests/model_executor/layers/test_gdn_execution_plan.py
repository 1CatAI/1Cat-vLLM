# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for initialized GDN policy, provider order and resource isolation."""

import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from vllm.config.gdn import GdnConfig, GdnProfileConfig
from vllm.config.gdn_schedule import GdnScheduleConfig
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.fla.ops import gdn_prefill as prefill
from vllm.model_executor.layers.fla.ops import gdn_stages as stages
from vllm.model_executor.layers.fla.ops.gdn_profiling import bind_gdn_profiler
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    GDN_BACKEND_STAGES,
    GdnExecutionPlan,
)


def test_captured_aliases_typed_precedence_and_two_engines(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL", "off")
    monkeypatch.setenv("FLASH_QLA_SM70_USE_ORIGINAL_TILELANG", "yes")
    monkeypatch.setenv("VLLM_SM70_FLA_BV", "16")
    first = GdnConfig(prefill_backend="TRITON")
    first.resolve(additional_config={"gdn_prefill_backend": "flashqla_sm70"})
    monkeypatch.setenv("VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL", "on")
    monkeypatch.setenv("VLLM_SM70_FLA_BV", "8")
    second = GdnConfig()
    second.resolve()
    assert first.prefill_backend == "triton"
    assert first.original_prefill is False and second.original_prefill is True
    assert (first.schedule.recurrent_bv, second.schedule.recurrent_bv) == (16, 8)
    # The serialized resolved policy is authoritative on a worker with a
    # different process environment; no worker-side re-resolution occurs.
    payload = json.loads(json.dumps(asdict(first)))
    assert payload["resolved"] and payload["schedule"]["recurrent_bv"] == 16
    first.resolve()
    assert first.schedule.recurrent_bv == 16


def test_defaults_do_not_write_environment_or_override_explicit(monkeypatch):
    import os

    monkeypatch.delenv("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", raising=False)
    monkeypatch.delenv("VLLM_SM70_GDN_DECODE_FLASHQLA", raising=False)
    before = dict(os.environ)
    policy = GdnConfig(flashqla_decode=False)
    policy.apply_platform_defaults()
    policy.resolve()
    assert policy.packed_recurrent_decode is True
    assert policy.flashqla_decode is False
    assert policy.sources["packed_recurrent_decode"] == "platform baseline"
    assert dict(os.environ) == before


def test_schedule_ignored_bad_overrides_keep_admission(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLA_RECURRENT_SCHEDULE", "0")
    monkeypatch.setenv("VLLM_SM70_FLA_BV", "bad")
    monkeypatch.setenv("VLLM_SM70_FLA_BV_CANDIDATES", "-1, 32, bad, 8")
    policy = GdnScheduleConfig()
    policy.resolve()
    assert not policy.recurrent_enabled
    assert policy.recurrent_override
    assert policy.recurrent_bv is None
    assert policy.recurrent_bv_candidates == [32, 8]


def test_active_computation_hash_and_inactive_models():
    first, second = KernelConfig(), KernelConfig(gdn=GdnConfig(original_prefill=False))
    assert first.compute_hash() == second.compute_hash()
    for kernel in (first, second):
        kernel.gdn.resolve()
    assert first.compute_hash() != second.compute_hash()
    first_hash = first.compute_hash()
    first.gdn.decode_warmup = not first.gdn.decode_warmup
    first.gdn.sources["original_prefill"] = "other provenance"
    assert first.compute_hash() == first_hash


def test_profiler_engine_budget_capture_guard_and_inactive_limits(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_GDN_PREFILL_PROFILE_MAX_LOGS", "bad")
    disabled = GdnProfileConfig(enabled=False)
    disabled.resolve()
    assert disabled.max_logs == 256
    first = SimpleNamespace(
        observability_config=SimpleNamespace(
            gdn_profile=GdnProfileConfig(enabled=True, max_logs=2, max_per_stage=1)
        )
    )
    second = SimpleNamespace(
        observability_config=SimpleNamespace(
            gdn_profile=GdnProfileConfig(enabled=True, max_logs=2, max_per_stage=1)
        )
    )
    owner = bind_gdn_profiler(first)
    other = bind_gdn_profiler(second)
    assert owner is bind_gdn_profiler(first) and owner is not other
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    for layer in ("a", "a", "b", "c"):
        owner.end(layer, "kernel", 0.0)
    assert owner.counts["__total__"] == 2
    assert not other.counts
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert owner.start() is None


@pytest.mark.parametrize("backend", ["flashinfer", "cutedsl", "triton"])
def test_prefill_keeps_gate_rounding_and_normalization_order(monkeypatch, backend):
    calls = []
    q = torch.randn(1, 2, 1, 4, dtype=torch.float16)
    gate = torch.full((1, 2, 1), 0.1234, dtype=torch.float16)
    state = torch.ones(1, 1, 4, 4, dtype=torch.float16)
    original_log = torch.log

    def log(value):
        calls.append("log")
        return original_log(value)

    def norm(value):
        calls.append("norm")
        return value

    def operator(**kwargs):
        calls.append("kernel")
        expected = original_log(gate)
        if backend == "flashinfer":
            assert kwargs["initial_state"].dtype == torch.float32
            assert kwargs["beta"].dtype == torch.float32
            expected = torch.exp(expected.squeeze(0).float())
        assert torch.equal(kwargs["g"], expected)
        return kwargs["v"], kwargs["initial_state"]

    monkeypatch.setattr(torch, "log", log)
    monkeypatch.setattr(stages, "l2norm_fwd", norm)
    if backend == "flashinfer":
        monkeypatch.setitem(
            sys.modules,
            "flashinfer.gdn_prefill",
            SimpleNamespace(chunk_gated_delta_rule=operator),
        )
    elif backend == "cutedsl":
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl",
            SimpleNamespace(chunk_gated_delta_rule_cutedsl=operator),
        )
    else:
        monkeypatch.setattr(prefill, "fla_chunk_gated_delta_rule", operator)
    plan = GdnExecutionPlan(
        backend, GDN_BACKEND_STAGES[backend], True, False, True, None
    )
    owner = bind_gdn_profiler(
        SimpleNamespace(
            observability_config=SimpleNamespace(
                gdn_profile=GdnProfileConfig(enabled=False)
            )
        )
    )
    provider = prefill.GdnPrefill(plan, owner)
    out, final = provider._forward_method(
        q=q,
        k=q,
        v=q,
        g=gate,
        beta=gate,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 2]),
        chunk_indices=torch.tensor([[0, 0]]),
        chunk_offsets=torch.tensor([0, 1]),
        gate_is_exp=True,
    )
    assert out.shape == q.shape and final is not None
    assert (
        calls
        == {
            "flashinfer": ["log", "norm", "norm", "kernel"],
            "cutedsl": ["norm", "norm", "log", "kernel"],
            "triton": ["log", "kernel"],
        }[backend]
    )


@pytest.mark.parametrize("validate", [False, True])
def test_decode_convolution_retains_state_views_and_validation(monkeypatch, validate):
    x, state, weight = torch.randn(2, 4), torch.randn(8, 4, 3), torch.randn(4, 1, 4)
    indices = torch.tensor([4, -1], dtype=torch.int32)

    def conv(x_arg, state_arg, w, bias, activation, **kwargs):
        assert state_arg is state and x_arg is x
        assert w.shape == (4, 4) and w.data_ptr() == weight.data_ptr()
        assert kwargs == {"conv_state_indices": indices, "validate_data": validate}
        return x_arg

    monkeypatch.setattr(stages, "causal_conv1d_update", conv)
    assert (
        stages.convolve_decode(
            x,
            state,
            weight,
            None,
            "silu",
            state_indices=indices,
            validate_data=validate,
        )
        is x
    )


def test_flashqla_dtype_fallback_retains_engine_chunk_policy(monkeypatch):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as layer

    plan = GdnExecutionPlan(
        "flashqla_sm70", GDN_BACKEND_STAGES["flashqla_sm70"], True, False, True, None
    )
    config = SimpleNamespace(model_config=None)
    kernels = object()
    profiler = SimpleNamespace()
    monkeypatch.setattr(layer.CustomOp, "__init__", lambda self: None)
    monkeypatch.setattr(layer, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(layer, "select_gdn_execution", lambda config: plan)
    monkeypatch.setattr(layer, "bind_gdn_profiler", lambda config: profiler)
    monkeypatch.setattr(
        layer, "resolve_gdn_config", lambda config: SimpleNamespace(schedule=object())
    )
    monkeypatch.setattr(layer, "bind_chunk_kernels", lambda config, schedule: kernels)
    provider = layer.ChunkGatedDeltaRule()
    q = torch.randn(1, 2, 1, 4, dtype=torch.float32)
    state = torch.zeros(1, 1, 4, 4)

    def native(**kwargs):
        assert kwargs["kernels"] is kernels
        assert kwargs["initial_state"] is state
        return kwargs["v"], state

    monkeypatch.setattr(prefill, "fla_chunk_gated_delta_rule", native)
    out, final = provider.forward_flashqla_sm70(
        q, q, q, q[..., 0], q[..., 0], state, True
    )
    assert out is q and final is state


@pytest.mark.parametrize("capability", [(7, 0), (7, 5), (8, 0), None])
def test_decode_admission_keeps_empty_dtype_and_device_fallback(
    monkeypatch, capability
):
    from vllm.model_executor.layers.fla.ops.sm70 import gdn_decode

    heads = stages.GdnHeadContract(16, 48, 128, 128, 4)
    admission = gdn_decode.FlashQlaDecodeAdmission(heads, True, capability, 0)
    mixed = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float16,
        shape=(1, 2560),
        device=torch.device("cuda:0"),
        dim=lambda: 2,
        stride=lambda axis: (4096, 1)[axis],
    )
    indices = torch.tensor([0], dtype=torch.int32)
    monkeypatch.setattr(gdn_decode, "_flashqla_sm70_decode_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *_: pytest.fail("bound-device capability must not be read per token"),
    )
    assert admission.rejection(mixed, indices, 0) == "no_decode_tokens_or_state_indices"
    assert admission.rejection(mixed, indices.long(), 1) == "state_indices_dtype"
    expected = None if capability in ((7, 0), (7, 5)) else "device_capability"
    assert admission.rejection(mixed, indices, 1) == expected
    if expected is None:
        monkeypatch.setattr(
            gdn_decode, "_flashqla_sm70_decode_available", lambda: False
        )
        assert admission.rejection(mixed, indices, 1) == "flashqla_decode_import"


def test_old_prefill_import_still_forwards_to_native():
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as old

    assert old.fla_chunk_gated_delta_rule is chunk_gated_delta_rule


def test_bound_backend_hash_uses_effective_strategy():
    policies = [
        GdnConfig(prefill_backend="auto"),
        GdnConfig(prefill_backend="triton", original_prefill=False),
    ]
    for policy in policies:
        policy.resolve()
        policy.active_prefill_backend = "triton"
    assert policies[0].compute_hash() == policies[1].compute_hash()
    policies[1].active_prefill_backend = "flashinfer"
    assert policies[0].compute_hash() != policies[1].compute_hash()
    before = policies[1].compute_hash()
    policies[1].schedule.kkt_bk = [128]
    assert policies[1].compute_hash() == before


@pytest.mark.parametrize(
    "cap,requested,expected",
    [
        (90, "auto", "flashinfer"),
        (90, "unknown", "triton"),
        (100, "auto", "triton"),
        (100, "flashinfer", "flashinfer"),
        (100, "cutedsl", "cutedsl"),
        (80, "flashinfer", "triton"),
    ],
)
def test_backend_plan_preserves_candidate_order(monkeypatch, cap, requested, expected):
    from vllm.model_executor.layers.fla.ops import gdn_selector as selector

    platform = selector.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    monkeypatch.setattr(platform, "is_device_capability", lambda value: cap == value)
    monkeypatch.setattr(
        platform, "is_device_capability_family", lambda value: cap == value
    )
    monkeypatch.setattr(
        platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=cap // 10, minor=cap % 10),
    )
    monkeypatch.setattr(platform, "get_cuda_runtime_major", lambda: 13)
    monkeypatch.setattr(selector, "_is_libs_cu13_install_intact", lambda: True)
    config = SimpleNamespace(
        kernel_config=KernelConfig(gdn=GdnConfig(prefill_backend=requested)),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            hf_text_config=SimpleNamespace(linear_key_head_dim=128),
            hf_config=None,
        ),
    )
    plan = selector.select_gdn_execution(config)
    assert plan.prefill.backend == expected
    assert config.kernel_config.gdn.active_prefill_backend == expected
    assert plan.explain()["evidence"] == "static selection; not a native launch"


def test_native_binding_uses_current_weights_and_keeps_missing_op_fallback(monkeypatch):
    from vllm.model_executor.layers.fla.ops.sm70.gdn_verify import bind_native_verifier

    heads = stages.GdnHeadContract(1, 1, 128, 128, 1)
    seen = []
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())
    assert bind_native_verifier(heads, enabled=True) is None
    torch.ops._C.sm70_gdn_verify_out = lambda *args: seen.append(args[3:5])
    verify = bind_native_verifier(heads, enabled=True)
    assert verify is not None
    qkv = torch.zeros(1, 384, dtype=torch.float16)
    a = torch.zeros(1, 1, dtype=torch.float16)
    state = torch.zeros(2, 1, 128, 128)
    out = torch.empty(1, 1, 128, dtype=torch.float16)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    indices = torch.tensor([[1]], dtype=torch.int32)
    for _ in range(2):
        weight, bias = torch.randn(1), torch.randn(1)
        verify(weight, bias, 1, qkv, a, a, state, out, 1, True, cu, indices, None)
        assert seen[-1][0] is weight and seen[-1][1] is bias
    assert seen[0][0] is not seen[1][0]


@pytest.mark.parametrize(
    "options", [{"recurrent_bv": 0}, {"kkt_bk": []}, {"chunk_o_bv": [-1]}]
)
def test_invalid_typed_geometry_is_rejected(options):
    with pytest.raises(ValueError):
        GdnScheduleConfig(**options)
