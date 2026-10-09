# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import io
import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re
import torch

from vllm import _sm70_ops
from vllm._sm70 import policy as binding
from vllm.config.kernel import KernelConfig
from vllm.config.sm70_moe import Sm70MoEFormatConfig
from vllm.config.sm70_native import NATIVE_FIELDS, UNSET, Sm70NativeConfig

pytestmark = pytest.mark.cpu_test


def test_packaged_native_policy_schemas_parse_before_build():
    # Torch rejects str[] defaults written as [], even though C++ compiles.
    # Parse the actual concatenated registration literals before a GPU build.
    root = Path(__file__).parents[2]
    names = set()
    for source in ("csrc/torch_bindings.cpp", "csrc/moe/torch_bindings.cpp"):
        text = (root / source).read_text()
        for match in re.finditer(
            r'\b(?:ops|m)\.def\(\s*((?:"[^"\\]*(?:\\.[^"\\]*)*"\s*)+)', text
        ):
            schema = "".join(
                ast.literal_eval(item)
                for item in re.findall(r'"[^"\\]*(?:\\.[^"\\]*)*"', match[1])
            )
            if "native_policy=" not in schema:
                continue
            parsed = torch._C.parse_schema(schema)
            names.add(parsed.name)
            assert parsed.arguments[-1].default_value is None
            assert str(parsed.arguments[-1].type) == "Optional[str]"
    assert names == set(binding.CONFIGURED_OPERATORS + binding.ROUTING_OPERATORS)


def values(policy):
    return {entry[0]: value for entry, value in zip(NATIVE_FIELDS, policy.values)}


@pytest.fixture(autouse=True)
def clean_native_environment(monkeypatch):
    for _, alias, _, _ in NATIVE_FIELDS:
        monkeypatch.delenv(alias, raising=False)


def test_capture_is_frozen_and_explicit_values_win(monkeypatch):
    alias = "VLLM_SM70_FP8_PREFILL_FAST_SELECTOR"
    monkeypatch.setenv(alias, "1")
    before = dict(os.environ)
    explicit = Sm70NativeConfig(fp8_prefill_fast_selector=False)
    explicit.resolve("fp8")
    legacy = Sm70NativeConfig()
    legacy.resolve("fp8")
    assert values(explicit)["fp8_prefill_fast_selector"] == "0"
    assert values(legacy)["fp8_prefill_fast_selector"] == "1"
    assert explicit.sources["fp8_prefill_fast_selector"] == "configuration"
    assert legacy.sources["fp8_prefill_fast_selector"] == alias
    assert dict(os.environ) == before
    monkeypatch.setenv(alias, "0")
    legacy.resolve("fp8")
    assert values(legacy)["fp8_prefill_fast_selector"] == "1"


def test_unused_format_and_diagnostics_do_not_change_kernel_hash(monkeypatch):
    first, second = KernelConfig(), KernelConfig()
    first.sm70_fp8.resolve()
    first.sm70_fp8.native.resolve("fp8")
    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL", "1")
    monkeypatch.setenv("TM_GEMM_TRACE", "1")
    second.sm70_fp8.resolve()
    second.sm70_fp8.native.resolve("fp8")
    assert first.compute_hash() == second.compute_hash()
    assert values(second.sm70_fp8.native)["nvfp4_moe_grouped_prefill"] == UNSET
    changed = KernelConfig()
    changed.sm70_fp8.resolve()
    changed.sm70_fp8.native.fp8_dense_tune_max_m = 7
    changed.sm70_fp8.native.resolve("fp8")
    assert changed.compute_hash() != first.compute_hash()


def test_parent_native_typed_conflict_is_rejected(monkeypatch):
    monkeypatch.setattr(binding, "native_policy_abi_available", lambda: True)
    policy = Sm70MoEFormatConfig(
        active_exact_w2=True,
        native=Sm70NativeConfig(awq_moe_batched_active_exact_w2=False),
    )
    with pytest.raises(ValueError, match="Conflicting typed requests"):
        policy.resolve("awq")


@pytest.mark.parametrize("legacy", list(itertools.product((False, True), repeat=3)))
@pytest.mark.parametrize(
    "explicit", list(itertools.product((None, False, True), repeat=2))
)
def test_single_token_stage_overrides_preserve_the_other_stage(
    monkeypatch, legacy, explicit
):
    combined, permute, unpermute = legacy
    fields = (
        "moe_single_token_fastpath",
        "moe_single_token_permute_fastpath",
        "moe_single_token_unpermute_fastpath",
    )
    for field, enabled in zip(fields, legacy):
        alias = next(entry[1] for entry in NATIVE_FIELDS if entry[0] == field)
        monkeypatch.setenv(alias, str(int(enabled)))
    policy = Sm70NativeConfig(**dict(zip(fields[1:], explicit)))
    policy.resolve("fp8")
    captured = values(policy)
    actual = tuple(
        bool(int(captured[field])) or bool(int(captured[fields[0]]))
        for field in fields[1:]
    )
    expected = tuple(
        request if request is not None else stage or combined
        for request, stage in zip(explicit, (permute, unpermute))
    )
    assert actual == expected


def test_bindings_forward_captured_values_without_reading_environment(monkeypatch):
    monkeypatch.setattr(binding, "native_policy_abi_available", lambda: True)
    monkeypatch.setattr(torch.ops, "_C_qwen38", SimpleNamespace())
    first = Sm70NativeConfig(fp8_dense_tune_max_m=8)
    first.resolve("fp8")
    second = Sm70NativeConfig(fp8_dense_tune_max_m=16)
    second.resolve("fp8")
    observed = []
    monkeypatch.setattr(
        _sm70_ops,
        "fp8_gemm_sm70_out",
        lambda *a, **kw: observed.append(kw["native_policy"]),
    )
    owners = [
        binding.NativeBindings(first.values),
        binding.NativeBindings(second.values),
    ]
    monkeypatch.setattr(
        os, "getenv", lambda *a: pytest.fail("execution read environment")
    )
    for owner in (*owners, owners[0]):
        owner.fp8_gemm_sm70_out(None)
    assert observed == [owner.arguments for owner in (*owners, owners[0])]


def test_prepared_argument_is_registered_once_and_preserves_utf8(monkeypatch):
    monkeypatch.setattr(binding, "native_policy_abi_available", lambda: True)
    registered: list[str] = []
    for namespace in ("_C", "_moe_C"):
        monkeypatch.setattr(
            torch.ops,
            namespace,
            SimpleNamespace(sm70_prepare_native_policy_token=registered.append),
        )
    monkeypatch.setattr(torch.ops, "_C_qwen38", SimpleNamespace())
    policy = Sm70NativeConfig(tm_gemm_trace_filter="路径:1")
    policy.resolve("fp8")
    observed = []
    monkeypatch.setattr(
        _sm70_ops,
        "fp8_gemm_sm70_out",
        lambda *a, native_policy: observed.append(native_policy),
    )
    owner = binding.NativeBindings(policy.values)
    assert owner.values == policy.values
    assert len(owner.arguments) == 1 and registered == [owner.arguments[0]] * 2
    encoded = owner.arguments[0].encode("utf-8")
    assert encoded.startswith(b"sm70:1:")
    remaining = encoded[7:]
    decoded = []
    while remaining:
        size, remaining = remaining.split(b":", 1)
        length = int(size)
        decoded.append(remaining[:length].decode("utf-8"))
        remaining = remaining[length:]
    assert tuple(decoded) == policy.values
    for _ in range(3):
        owner.fp8_gemm_sm70_out(None)
    assert observed == [owner.arguments] * 3
    assert len(registered) == 2


def test_old_native_abi_accepts_legacy_but_rejects_silent_typed_override(monkeypatch):
    monkeypatch.setattr(binding, "native_policy_abi_available", lambda: False)
    monkeypatch.setattr(torch.ops, "_C_qwen38", SimpleNamespace())
    monkeypatch.setenv("VLLM_SM70_FP8_DENSE_TUNE_MAX_M", "8")
    legacy = Sm70NativeConfig()
    legacy.resolve("fp8")
    assert binding.NativeBindings(legacy.values).values == ()
    explicit = Sm70NativeConfig(fp8_dense_tune_max_m=16)
    explicit.resolve("fp8")
    with pytest.raises(RuntimeError, match="policy-argument ABI"):
        binding.NativeBindings(explicit.values)


def test_compile_key_uses_effective_policy_not_overridden_env(monkeypatch):
    from vllm import envs
    from vllm.compilation.caching import aot_compile_hash_factors

    kernel = KernelConfig()
    kernel.sm70_moe.fp8.native.fp8_dense_tune_max_m = 8
    kernel.sm70_moe.fp8.resolve("fp8")
    cfg = SimpleNamespace(kernel_config=kernel, compute_hash=kernel.compute_hash)
    before = aot_compile_hash_factors(cfg)
    monkeypatch.setenv("VLLM_SM70_FP8_DENSE_TUNE_MAX_M", "16")
    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE", "1")
    envs.disable_envs_cache()
    assert aot_compile_hash_factors(cfg) == before
    # Callers without an engine config retain the legacy safety key.
    assert "VLLM_SM70_FP8_DENSE_TUNE_MAX_M" in envs.compile_factors()
    envs.disable_envs_cache()


def test_unused_linear_families_do_not_perturb_loaded_moe_hash():
    first, second = KernelConfig(), KernelConfig()
    for kernel in (first, second):
        # VllmConfig resolves qualification before it knows which providers load.
        kernel.sm70_nvfp4.resolve(qualified=True, active=False)
        kernel.sm70_moe.awq.resolve("awq")
    second.sm70_nvfp4.qpn2 = False
    second.sm70_gguf.small_m_dp4a = False
    assert first.compute_hash() == second.compute_hash()
    second.sm70_nvfp4.active = True
    assert first.compute_hash() != second.compute_hash()


def test_block_qpn8_owns_native_policy_even_with_its_independent_constructor(
    monkeypatch,
):
    from vllm.model_executor.kernels.linear.qpn import fp8_block
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        ScaledMMLinearKernel,
    )

    monkeypatch.setattr(ScaledMMLinearKernel, "__init__", lambda *args: None)
    monkeypatch.setattr(binding, "native_policy_abi_available", lambda: True)
    monkeypatch.setattr(torch.ops, "_C_qwen38", SimpleNamespace())
    config = KernelConfig()
    config.sm70_fp8.native.fp8_dense_tune_max_m = 8
    kernel = fp8_block.QPN8Fp8BlockScaledMMLinearKernel(
        SimpleNamespace(policy=config.sm70_fp8)
    )
    assert kernel.native_ops.values == config.sm70_fp8.native.values
    assert values(config.sm70_fp8.native)["fp8_dense_tune_max_m"] == "8"


def test_explicit_native_option_for_a_different_format_is_not_silently_ignored():
    policy = Sm70NativeConfig(nvfp4_qpn2_m16_native=True)
    with pytest.raises(ValueError, match="does not apply to fp8"):
        policy.resolve("fp8")


@pytest.mark.parametrize("compact", [False, True])
def test_opaque_native_policy_survives_export_with_dynamic_rows(monkeypatch, compact):
    # The optional string-list schema must retain all captured values through
    # fake dispatch and serialization, while M stays dynamic inside the op.
    from vllm.model_executor.kernels.linear.qpn import nvfp4_dequant

    policy = Sm70NativeConfig(nvfp4_qpn2_m16_native=False)
    policy.resolve("nvfp4")
    arguments = policy.values
    if compact:
        monkeypatch.setattr(binding, "native_policy_abi_available", lambda: True)
        monkeypatch.setattr(torch.ops, "_C_qwen38", SimpleNamespace())
        for namespace in (torch.ops._C, torch.ops._moe_C):
            monkeypatch.setattr(
                namespace,
                "sm70_prepare_native_policy_token",
                lambda token: None,
                raising=False,
            )
        arguments = binding.NativeBindings(policy.values).arguments
        assert len(arguments) == 1
    observed: list[tuple[int, tuple[str, ...] | str]] = []

    def native(operation, native_policy, out, x, *args):
        observed.append((x.shape[0], tuple(native_policy)))
        out.zero_()

    def dense(x, codes, scales, global_scale, n, k):
        observed.append((x.shape[0], "dense"))
        return x.new_zeros((x.shape[0], n))

    monkeypatch.setattr(binding, "call_native", native)
    monkeypatch.setattr(nvfp4_dequant, "_nvfp4_qpn2_dense_linear", dense)

    class Projection(torch.nn.Module):
        def forward(self, x):
            return torch.ops.vllm.nvfp4_qpn2_dispatch_linear(
                x, x, x, 1.0, 8, 4, 16, 2, list(arguments)
            )

    lib = None
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "vllm::nvfp4_qpn2_dispatch_linear", "CPU"
    ):
        lib = torch.library.Library("vllm", "IMPL", "CPU")
        lib.impl(
            "nvfp4_qpn2_dispatch_linear", nvfp4_dequant._nvfp4_qpn2_dispatch_linear
        )
    try:
        exported = torch.export.export(
            Projection(),
            (torch.zeros(2, 4),),
            dynamic_shapes={"x": {0: torch.export.Dim("rows", min=1, max=128)}},
        )
        artifact = io.BytesIO()
        torch.export.save(exported, artifact)
        artifact.seek(0)
        reloaded = torch.export.load(artifact).module()
        for rows in (1, 32, 33, 65):
            assert reloaded(torch.zeros(rows, 4)).shape == (rows, 8)
        assert observed == [
            (1, arguments),
            (32, arguments),
            (33, "dense"),
            (65, "dense"),
        ]
    finally:
        if lib is not None:
            lib._destroy()


@pytest.mark.parametrize("direct_stage", [False, True])
def test_native_token_survives_direct_operator_export_reload(direct_stage):
    if not hasattr(torch.ops._C, "sm70_prepare_native_policy_token"):
        pytest.skip("requires the packaged native token ABI")
    policy = Sm70NativeConfig(tm_gemm_trace_filter="路径:1")
    policy.resolve("awq")
    owner = binding.NativeBindings(policy.values)
    assert len(owner.arguments) == 1
    token = owner.arguments[0]
    observed: list[tuple[int, str]] = []

    def native(out, x, weight, scales, group, k, n, gated, native_policy=None):
        # This CPU implementation checks transport and mutation semantics;
        # numerical CUDA parity is covered by the artifact comparison.
        assert isinstance(native_policy, str)
        assert native_policy == token
        observed.append((x.shape[0], native_policy))
        out.copy_(x)

    def reduce(x, weights, indices, out, top_k, hidden, native_policy=None):
        native(out, x, x, x, 16, 4, 4, False, native_policy)

    class Projection(torch.nn.Module):
        def forward(self, x):
            out = torch.empty_like(x)
            if direct_stage:
                owner.awq_moe_single_token_weighted_reduce_out(x, x, x, out, 1, 4)
            else:
                owner.awq_gemm_sm70_out(out, x, x, x, 16, 4, 4, False)
            return out

    lib = torch.library.Library("_C", "IMPL", "CPU")
    lib.impl("awq_gemm_sm70_out", native)
    lib.impl("awq_moe_single_token_weighted_reduce_out", reduce)
    try:
        exported = torch.export.export(
            Projection(),
            (torch.zeros(2, 4),),
            dynamic_shapes={"x": {0: torch.export.Dim("rows", min=1, max=128)}},
        )
        artifact = io.BytesIO()
        torch.export.save(exported, artifact)
        artifact.seek(0)
        reloaded = torch.export.load(artifact).module()
        for rows in (1, 33, 65):
            x = torch.full((rows, 4), float(rows))
            assert torch.equal(reloaded(x), x)
        assert observed == [(rows, token) for rows in (1, 33, 65)]
    finally:
        lib._destroy()


@pytest.mark.parametrize("explicit_qpn8", [None, False, True])
def test_channel_fp8_keeps_qualified_default_but_typed_linear_request_wins(
    monkeypatch, explicit_qpn8
):
    from vllm.config.kernel import Sm70Fp8Config
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a16_fp8 as channel,
    )

    linear = Sm70Fp8Config(qpn8=explicit_qpn8)
    linear.resolve()
    monkeypatch.setattr(channel, "capture_sm70_fp8_linear_config", lambda: linear)
    monkeypatch.setattr(
        channel,
        "capture_sm70_dflash2_config",
        lambda: SimpleNamespace(
            resolved=True,
            qualified=True,
            target_fp8_qpn8=True,
            explicit_fields=(),
        ),
    )
    assert channel._sm70_fp8_qpn8_enabled(False) is (
        True if explicit_qpn8 is None else explicit_qpn8
    )


def test_prepared_stage_bindings_preserve_arguments_and_instrumentation(monkeypatch):
    import functools
    import inspect

    from vllm._sm70 import moe

    monkeypatch.setattr(binding, "native_policy_abi_available", lambda: True)
    monkeypatch.setattr(torch.ops, "_C_qwen38", SimpleNamespace())
    calls = []
    namespaces = {}
    for name in ("_C", "_moe_C"):
        namespaces[name] = SimpleNamespace(
            sm70_prepare_native_policy_token=lambda s: None
        )
        monkeypatch.setattr(torch.ops, name, namespaces[name])
    stages = []
    for module in (moe, binding):
        for name, operation in vars(module).items():
            marker = getattr(operation, "_sm70_direct", None)
            if marker and marker[0] is operation:
                native = lambda *a, **kw: calls.append((a, kw))
                setattr(namespaces[marker[1]], name, native)
                stages.append((name, operation, native))
    policy = Sm70NativeConfig()
    policy.resolve("fp8")
    owner = binding.NativeBindings(policy.values)
    for name, operation, native in stages:
        if name in binding.ROUTING_OPERATORS:
            arguments: tuple[object, ...] = (object(), object())
        else:
            arguments = tuple(
                object()
                for p in inspect.signature(operation).parameters.values()
                if p.name != "native_policy"
            )
        operation(*arguments, native_policy=owner.arguments)
        getattr(owner, name)(*arguments)
        assert calls[-2:] == [(arguments, {"native_policy": owner.arguments[0]})] * 2
        assert getattr(owner, name).func is native

    # functools.wraps copies attributes. An observer remains a public wrapper,
    # not an invitation to bypass it while resolving the prepared owner.
    name, operation, native = stages[0]
    observed = []

    @functools.wraps(operation)
    def instrument(*a, **kw):
        observed.append(name)
        return operation(*a, **kw)

    monkeypatch.setattr(_sm70_ops, name, instrument)
    instrumented = binding.NativeBindings(policy.values)
    assert getattr(instrumented, name).func is instrument
    arguments = tuple(
        object()
        for p in inspect.signature(operation).parameters.values()
        if p.name != "native_policy"
    )
    getattr(instrumented, name)(*arguments)
    assert observed == [name]
