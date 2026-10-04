# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU loader/dispatch regressions; no GPU arithmetic or capacity claim."""

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
from vllm.sm70_profiles.acceleration import loaded_sm70_preparations


@pytest.fixture(autouse=True)
def restore_precision():
    backend = torch.backends.cuda.matmul
    names = (
        "allow_fp16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    )
    before = [getattr(backend, name) for name in names]
    envs.disable_envs_cache()
    yield
    for name, value in zip(names, before):
        setattr(backend, name, value)
    envs.disable_envs_cache()


def make_layer(role):
    layer = torch.nn.Module()
    rows = 512 if role == "router" else 320
    # Exactly representable values let packing tests inspect every FP16 bit.
    weight = (torch.arange(rows * 2560) % 257 - 128).to(torch.float16)
    layer.weight = torch.nn.Parameter(weight.reshape(rows, 2560), requires_grad=False)
    setattr(layer, f"_sm70_mtp_prepare_{role}_batch", True)
    if role == "shared":
        layer.forward_fused_silu_and_mul = MethodType(
            gemv._forward_shared_batch_silu, layer
        )
    return layer


def prepare_on_cpu(layer, monkeypatch, *, native_available=True):
    # Exercise the real loader and packing on CPU storage. Only placement and
    # the unrelated base linear preparation are stubbed; no native op executes.
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        m.setattr(
            UnquantizedLinearMethod, "process_weights_after_loading", lambda *args: None
        )
        if not native_available:
            m.setattr(torch.ops, "_C", SimpleNamespace())
        for name in (
            "qwen38_router_batch_sm70_out",
            "qwen38_shared_up_batch_sm70_out",
        ):
            if native_available:
                m.setattr(torch.ops._C, name, lambda *args: None, raising=False)
        gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)


@pytest.mark.parametrize(
    "reduced,accumulation,reason",
    [
        (False, False, "fp16_reduced_precision_reduction_disabled"),
        (True, True, "fp16_accumulation_enabled"),
        (True, False, None),
    ],
)
def test_precision_skip_info_once_per_role(reduced, accumulation, reason, monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
    torch.backends.cuda.matmul.allow_fp16_accumulation = accumulation
    # A dedicated logger exercises the real info_once deduplication without
    # depending on earlier loader tests or an initialized distributed group.
    logger = init_logger(f"test_mtp_packing_policy.{reduced}.{accumulation}")
    messages = []
    monkeypatch.setattr(
        logger, "info", lambda msg, *args, **kwargs: messages.append(msg % args)
    )
    monkeypatch.setattr(gemv, "logger", logger)
    for role in ("router", "shared"):
        # Target and draft layers share the process/role precision policy.
        for _ in range(2):
            layer = torch.nn.Module()
            assert gemv._mtp_batch_packing_allowed(layer, role) == (reason is None)
            assert getattr(layer, f"_sm70_mtp_{role}_batch_reason") == reason
    expected = (
        []
        if reason is None
        else [
            f"Skipping SM70 MTP {role} packed weights due to precision policy: "
            f"{reason}."
            for role in ("router", "shared")
        ]
    )
    assert messages == expected


@pytest.mark.parametrize("role", ["router", "shared"])
@pytest.mark.parametrize(
    "reduced,accumulation", [(False, False), (False, True), (True, False), (True, True)]
)
def test_loader_skips_only_precision_rejected_packs(
    role, reduced, accumulation, monkeypatch
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
    torch.backends.cuda.matmul.allow_fp16_accumulation = accumulation
    layer = make_layer(role)
    weight = layer.weight
    before = weight.detach().clone()
    hook = getattr(layer, "forward_fused_silu_and_mul", None)
    prepare_on_cpu(layer, monkeypatch)

    admitted = reduced and not accumulation
    buffer_name = f"_sm70_mtp_{role}_packed"
    assert hasattr(layer, buffer_name) == admitted
    assert layer.weight is weight and torch.equal(layer.weight, before)
    assert getattr(layer, f"_sm70_mtp_prepare_{role}_batch")
    assert getattr(layer, "forward_fused_silu_and_mul", None) is hook
    reason_name = f"_sm70_mtp_{role}_batch_reason"
    report = loaded_sm70_preparations(torch.nn.Sequential(layer))["variants"]["0"]
    if admitted:
        packed = getattr(layer, buffer_name)
        if role == "router":
            restored = packed.permute(0, 4, 3, 1, 2, 5).reshape_as(weight)
        else:
            restored = packed.permute(0, 3, 1, 2, 4).reshape_as(weight)
        assert torch.equal(restored.view(torch.int16), before.view(torch.int16))
        assert buffer_name not in layer.state_dict()
        assert report["prepared_buffers"] == [buffer_name]
        assert reason_name not in report["reasons"]
    else:
        expected = (
            "fp16_reduced_precision_reduction_disabled"
            if not reduced
            else "fp16_accumulation_enabled"
        )
        assert report["reasons"][reason_name] == expected
        assert report["prepared_buffers"] == []


@pytest.mark.parametrize("role", ["router", "shared"])
def test_rejected_pack_does_not_require_native_operator(role, monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    layer = make_layer(role)
    prepare_on_cpu(layer, monkeypatch, native_available=False)
    assert not hasattr(layer, f"_sm70_mtp_{role}_packed")


@pytest.mark.parametrize("role", ["router", "shared"])
def test_admitted_pack_still_requires_native_operator(role, monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    with pytest.raises(RuntimeError, match="Rebuild the SM70 extension"):
        prepare_on_cpu(make_layer(role), monkeypatch, native_available=False)


@pytest.mark.parametrize("role", ["router", "shared"])
def test_admitted_pack_remains_reloadable(role, monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    layer = make_layer(role)
    prepare_on_cpu(layer, monkeypatch)
    old = getattr(layer, f"_sm70_mtp_{role}_packed").clone()
    layer.weight.data.mul_(0.5)
    prepare_on_cpu(layer, monkeypatch)
    assert torch.equal(getattr(layer, f"_sm70_mtp_{role}_packed"), old * 0.5)


@pytest.mark.parametrize("role", ["router", "shared"])
@pytest.mark.parametrize("rows", [5, 10])
def test_rejected_policy_fallback_is_unchanged_with_or_without_pack(
    role, rows, monkeypatch
):
    # Model the pre-fix resident pack, then compare against the skipped pack at
    # the same precision policy. Assert the original F.linear fallback executes.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    layer = make_layer(role)
    prepare_on_cpu(layer, monkeypatch)
    packed = getattr(layer, f"_sm70_mtp_{role}_packed")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    monkeypatch.setenv("VLLM_SM70_MTP_ROUTER_BATCH", "1")
    monkeypatch.setenv("VLLM_SM70_MTP_SHARED_BATCH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    x = torch.linspace(-0.01, 0.01, rows * 2560, dtype=torch.float16).reshape(
        rows, 2560
    )
    linear = F.linear
    calls = []

    def record_linear(*args, **kwargs):
        calls.append(args[0].shape)
        return linear(*args, **kwargs)

    def cpu_silu_and_mul(out, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        out.copy_(F.silu(gate) * up)

    monkeypatch.setattr(F, "linear", record_linear)
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", cpu_silu_and_mul, raising=False)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        if role == "router":
            assert not gemv._router_batch_runtime_ok(x, packed)
            before = gemv._qwen38_sm70_fp16_gemv(
                x, layer.weight, "model.layers.0.mlp.gate", packed
            )
            after = gemv._qwen38_sm70_fp16_gemv(
                x, layer.weight, "model.layers.0.mlp.gate", None
            )
        else:
            assert not gemv._shared_batch_runtime_ok(x)
            before = gemv._qwen38_sm70_shared_up(x, layer.weight, packed)
            after = gemv._qwen38_sm70_shared_up(x, layer.weight, None)
    assert calls == [x.shape, x.shape]
    assert torch.equal(before.view(torch.int16), after.view(torch.int16))
