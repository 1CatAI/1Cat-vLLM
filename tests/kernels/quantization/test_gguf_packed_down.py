# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU load/dispatch contracts; native numerical and graph proof is separate."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.quantization import gguf_moe as module
from vllm.model_executor.layers.quantization.gguf import GGUFConfig


class CpuCudaMetadata(torch.Tensor):
    """Real CPU storage with CUDA admission metadata; never runs a CUDA op."""

    @property
    def is_cuda(self):
        return True


@pytest.fixture(scope="module")
def banks():
    # Empty CPU virtual storage is never read or filled. The real parent load
    # must retain these contiguous, K-aligned banks, not copy/materialize them.
    return {
        "w1": torch.empty((288, 2048, 1760), dtype=torch.uint8).as_subclass(
            CpuCudaMetadata
        ),
        "w3": torch.empty((288, 2048, 1760), dtype=torch.uint8).as_subclass(
            CpuCudaMetadata
        ),
        "w2": torch.empty((288, 4096, 1088), dtype=torch.uint8).as_subclass(
            CpuCudaMetadata
        ),
    }


def prepare(monkeypatch, banks, *, enabled=True, arch=70, operator=True):
    assert not torch.cuda.is_initialized()
    config = KernelConfig(gguf_moe_packed_down=enabled)
    monkeypatch.setattr(
        module,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(kernel_config=config, model_config=None),
    )
    method = module.GGUFNativeMoEMethod(
        GGUFConfig(), SimpleNamespace(dp_size=1, pcp_size=1, ep_size=1)
    )
    method.num_experts, method.hidden_size, method.intermediate_size = 288, 4096, 2048
    method.params_dtype = torch.float16
    method.weight_types = {"w1": 21, "w3": 21, "w2": 23}
    method.loaded_experts = {key: set(range(288)) for key in banks}
    layer = SimpleNamespace(
        tp_size=1,
        ep_size=1,
        top_k=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        swiglu_limit=None,
        **{"gguf_" + key: value for key, value in banks.items()},
    )
    calls: list[Any] = []
    probes: list[Any] = []
    messages: list[Any] = []
    device_probes: list[Any] = []

    def capability(weight, x, kind, rows):
        probes.append((weight.shape, x.shape, kind, rows))
        return 12  # MMVQ and MMQ; actual admission/preparation remains real.

    def run(name):
        def operation(x, weight, ids, kind, rows, routes, count, *extra):
            calls.append(
                (name, x.shape, ids.shape, kind, rows, routes, count, len(extra))
            )
            return torch.ones((count * routes, rows), dtype=x.dtype).as_subclass(
                CpuCudaMetadata
            )

        return operation

    namespace = SimpleNamespace(
        ggml_dense_upstream_capabilities=capability,
        ggml_moe_mmvq=run("old"),
        ggml_moe_mmq=run("mmq"),
    )
    if operator:
        namespace.ggml_moe_mmvq_packed = run("packed")
    monkeypatch.setattr(torch.ops, "_C_gguf", namespace)
    monkeypatch.setattr(module, "native_available", lambda: True)
    # Deliberately report a different current-device policy. The new load
    # admission must query the actual weight device rather than this policy.
    monkeypatch.setattr(
        module.current_platform, "is_device_capability", lambda *_: False
    )
    monkeypatch.setattr(
        module.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(to_int=lambda: 86),
    )

    def actual_device(device):
        device_probes.append(device)
        assert device == layer.gguf_w2.device
        return divmod(arch, 10)

    monkeypatch.setattr(torch.cuda, "get_device_capability", actual_device)
    monkeypatch.setattr(
        module.logger,
        "info_once",
        lambda message, *args, **kwargs: messages.append((message, args, kwargs)),
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "gguf_original_expert_reduce",
        lambda down, weights: (down.float() * weights[..., None]).sum(1).to(down.dtype),
    )
    before = {
        k: (v.data_ptr(), v.untyped_storage().nbytes(), v.stride(), v.shape)
        for k, v in banks.items()
    }
    # Do not replace process_weights_after_loading, pad_weight_tail,
    # admit_moe_fallback or the prefill capability functions.
    method.process_weights_after_loading(layer)
    for key, original in banks.items():
        prepared = getattr(layer, "gguf_" + key)
        assert prepared is original
        assert (
            prepared.data_ptr(),
            prepared.untyped_storage().nbytes(),
            prepared.stride(),
            prepared.shape,
        ) == before[key]
    assert len(probes) == 6
    assert not torch.cuda.is_initialized()
    return method, layer, calls, messages, device_probes


def apply(method, layer, m=5, topk=8, dtype=torch.float16):
    x = torch.ones((m, 4096), dtype=dtype).as_subclass(CpuCudaMetadata)
    ids = torch.zeros((m, topk), dtype=torch.int32).as_subclass(CpuCudaMetadata)
    weights = torch.full((m, topk), 1 / topk)
    return method.apply(layer, x, weights, ids, None, None)


@pytest.mark.parametrize("active", [False, True])
def test_default_and_hash_always_bind_typed_flag(active):
    off, on = KernelConfig(), KernelConfig(gguf_moe_packed_down=True)
    off.sm70_gguf.active = on.sm70_gguf.active = active
    assert not off.gguf_moe_packed_down
    assert off.compute_hash() != on.compute_hash()


@pytest.mark.parametrize(
    "enabled,arch,operator",
    [(False, 70, True), (True, 70, True), (True, 86, True), (True, 70, False)],
)
def test_real_load_caches_actual_bank_and_constructor_config(
    monkeypatch, banks, enabled, arch, operator
):
    method, layer, calls, messages, devices = prepare(
        monkeypatch, banks, enabled=enabled, arch=arch, operator=operator
    )
    admitted = enabled and arch == 70 and operator
    assert method.packed_down_admitted == admitted
    assert len(devices) == int(enabled)
    assert len(messages) == int(enabled)
    if messages:
        assert messages[0][1][0] == admitted and messages[0][2] == {"scope": "process"}
    monkeypatch.setattr(
        module,
        "get_current_vllm_config_or_none",
        lambda: (_ for _ in ()).throw(AssertionError("Runtime config lookup")),
    )
    apply(method, layer)
    assert [call[0] for call in calls] == [
        "old",
        "old",
        "packed" if admitted else "old",
    ]
    assert all(
        call[-1] == 0 for call in calls
    )  # All native calls retain seven arguments.
    if admitted:
        assert messages[-1][0].startswith("GGUF MoE down dispatch:")
        assert messages[-1][2] == {"scope": "process"}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "m,topk,dtype",
    [
        (1, 8, torch.float16),
        (4, 8, torch.float16),
        (6, 8, torch.float16),
        (10, 4, torch.float16),
        (5, 8, torch.bfloat16),
    ],
)
def test_unmatched_forward_calls_original_complete_wrapper(
    monkeypatch, banks, m, topk, dtype
):
    method, layer, calls, _, _ = prepare(monkeypatch, banks)
    apply(method, layer, m, topk, dtype)
    assert [call[0] for call in calls] == ["old"] * 3
    assert calls[-1][1:7] == ((m * topk, 2048), (m * topk, 1), 23, 4096, 1, m * topk)


@pytest.mark.parametrize(
    "change",
    ["type", "padding", "bank", "cached_device", "ep", "cache", "old_capability"],
)
def test_late_metadata_mismatch_falls_back_without_calling_packed(
    monkeypatch, banks, change
):
    method, layer, calls, _, _ = prepare(monkeypatch, banks)
    if change == "type":
        method.weight_types["w2"] = 14
    elif change == "padding":
        method.input_padding["w2"] = (0, 2560)
    elif change == "bank":
        layer.gguf_w2 = layer.gguf_w2[:287]
    elif change == "cached_device":
        method.packed_down_device = torch.device("cuda:1")
    elif change == "ep":
        layer.expert_map = torch.arange(288, dtype=torch.int32)
    elif change == "cache":
        method.expert_cache = SimpleNamespace(
            tm=None,
            prefill=lambda _, ids: (layer.gguf_w1, layer.gguf_w3, layer.gguf_w2, ids),
            prefill_done=lambda _: None,
        )
    else:
        from dataclasses import replace

        method.projection_capabilities["w2"] = replace(
            method.projection_capabilities["w2"], operator="ggml_moe_mmq"
        )
    apply(method, layer)
    assert not any(call[0] == "packed" for call in calls)
    assert len(calls) == 3 and calls[-1][-1] == 0


def test_reload_failure_clears_cached_admission(monkeypatch, banks):
    method, layer, _, _, _ = prepare(monkeypatch, banks)
    assert method.packed_down_admitted
    method.loaded_experts["w1"].remove(0)
    with pytest.raises(ValueError, match="Incomplete GGUF"):
        method.process_weights_after_loading(layer)
    assert not method.packed_down_admitted and method.packed_down_device is None


def test_instances_and_legacy_new_fixtures_keep_default_off(monkeypatch, banks):
    on, _, _, _, _ = prepare(monkeypatch, banks)
    off, _, _, _, _ = prepare(monkeypatch, banks, enabled=False)
    assert on.packed_down_admitted and not off.packed_down_admitted
    legacy = object.__new__(module.GGUFNativeMoEMethod)
    assert not legacy.packed_down_enabled and not legacy.packed_down_admitted


def test_registered_native_schema_is_old_seven_arguments_plus_new_name():
    # Imports the actual packaged candidate module on CPU; no fake numerical op.
    import vllm._C_gguf  # noqa: F401

    old = str(
        torch._C._dispatch_find_schema_or_throw("_C_gguf::ggml_moe_mmvq", "").schema()
    )
    new = str(
        torch._C._dispatch_find_schema_or_throw(
            "_C_gguf::ggml_moe_mmvq_packed", ""
        ).schema()
    )
    assert old.replace("ggml_moe_mmvq(", "ggml_moe_mmvq_packed(") == new
    assert old.count(",") == 6
    assert not torch.cuda.is_initialized()
