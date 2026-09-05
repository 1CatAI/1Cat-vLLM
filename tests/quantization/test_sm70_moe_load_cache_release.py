# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle tests; native repacking and GPU workspaces are stubbed.

These exercise the real loading methods, not the numerical CUDA converters.
Model-level GPU A/B validation remains necessary for peak-memory measurements.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization import awq_sm70_moe as awq
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as nvfp4


@pytest.fixture
def should_do_global_cleanup_after_test():
    # All tensors and native-op stubs in this module are CPU-only.
    return False


def _parameter(layer, name, shape, dtype):
    value = torch.ones(shape, dtype=torch.float32).to(dtype)
    setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))


def _ptrs(weight, scales, k_ld, q_ld, experts):
    # Native pointer construction is not under test on CPU.
    return torch.zeros(experts, 8, dtype=torch.uint8), torch.zeros(
        experts, 8, dtype=torch.uint8
    )


def _awq_case(monkeypatch):
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(tp_size=1)
    for prefix, k, n in (("w13", 64, 64), ("w2", 32, 64)):
        _parameter(layer, f"{prefix}_qweight", (2, k, n // 8), torch.int32)
        _parameter(layer, f"{prefix}_scales", (2, k // 32, n), torch.float16)
        _parameter(layer, f"{prefix}_qzeros", (2, k // 32, n // 8), torch.int32)
    monkeypatch.setattr(
        awq,
        "envs",
        SimpleNamespace(
            VLLM_SM70_AWQ_MOE_COMPACT_METADATA=False,
            VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT=False,
            VLLM_SM70_AWQ_QWEN38_MOE_INDEXED_PREFILL=False,
            VLLM_SM70_AWQ_MOE_BATCHED_LAYER_ALLOWLIST=None,
            VLLM_SM70_AWQ_MOE_BATCHED_LAYER_DENYLIST=None,
        ),
    )
    monkeypatch.setattr(awq, "_batched_gemm_enabled_for_layer", lambda *_: False)
    monkeypatch.setattr(awq, "_get_layer_id", lambda _: 0)

    def prepare(weight, scales, zeros, group_size, interleaved):
        return weight.clone(), scales.clone(), torch.tensor([64, 64])

    monkeypatch.setattr(awq.sm70_ops, "awq_sm70_prepare", prepare)
    method = SimpleNamespace(
        group_size=32,
        pack_factor=8,
        use_batched_gemm=False,
        _allocate_buffers=lambda layer: setattr(layer, "test_buffer", torch.ones(4)),
    )
    source_names = tuple(layer._parameters)
    return (
        awq.AWQSM70MoEMethod.process_weights_after_loading,
        method,
        layer,
        source_names,
    )


def _nvfp4_case(monkeypatch):
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(
        tp_size=1,
        hidden_dim=64,
        intermediate_size_per_partition=32,
        experts_per_token=1,
    )
    layer.local_num_experts = layer.global_num_experts = 2
    layer.activation = nvfp4.MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.expert_map = layer.swiglu_limit = None
    for prefix, n, k in (("w13", 64, 64), ("w2", 64, 32)):
        _parameter(layer, f"{prefix}_weight", (2, n, k // 2), torch.uint8)
        _parameter(
            layer, f"{prefix}_weight_scale", (2, n, k // 16), torch.float8_e4m3fn
        )
        shape = (2, 2) if prefix == "w13" else (2,)
        _parameter(layer, f"{prefix}_weight_scale_2", shape, torch.float32)
        _parameter(layer, f"{prefix}_input_scale", (2,), torch.float32)
    flags = (
        "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE",
        "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE",
        "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE",
        "VLLM_SM70_NVFP4_QWEN38_MOE_W2_DIRECT_REDUCE",
        "VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL",
        "VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL",
        "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE",
        "VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8",
        "VLLM_SM70_GLM53_MOE_QPN_W13_Q8",
        "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE",
    )
    monkeypatch.setattr(nvfp4, "envs", SimpleNamespace(**dict.fromkeys(flags, False)))
    for name in (
        "nvfp4_sm70_prepare",
        "nvfp4_moe_dense_stage_sm70_out",
        "awq_moe_build_strided_ptrs",
    ):
        monkeypatch.setattr(torch.ops._C, name, Mock(), raising=False)
    monkeypatch.setattr(
        torch.ops._moe_C, "moe_permute_with_scratch", Mock(), raising=False
    )
    # Use small tensors while retaining the real packed-weight layout validator.
    monkeypatch.setattr(nvfp4, "validate_nvfp4_sm70_moe_contract", lambda _: None)

    def prepare(weight, scales, group_size, **kwargs):
        return weight.clone(), scales.clone(), torch.tensor([64, 64])

    monkeypatch.setattr(nvfp4.sm70_ops, "nvfp4_sm70_prepare", prepare)
    method = SimpleNamespace(
        moe=SimpleNamespace(has_bias=False),
        _allocate_graph_safe_decode_buffers=lambda layer: setattr(
            layer, "test_buffer", torch.ones(4)
        ),
    )
    source_names = tuple(layer._parameters)
    return (
        nvfp4.ModelOptNvFp4SM70MoEMethod.process_weights_after_loading,
        method,
        layer,
        source_names,
    )


@pytest.mark.parametrize("make_case", [_awq_case, _nvfp4_case], ids=["awq", "nvfp4"])
def test_release_once_after_source_removal_preserves_prepared_tensors(
    monkeypatch, make_case
):
    monkeypatch.setattr(awq.sm70_ops, "awq_moe_build_strided_ptrs", _ptrs)
    load, method, reference, _ = make_case(monkeypatch)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    load(method, reference)
    expected = {name: value.clone() for name, value in reference.named_parameters()}
    assert {"w13_tm_weight", "w2_tm_weight", "w13_tm_scales", "w2_tm_scales"} <= (
        expected.keys()
    )

    # Two layer invocations ensure this is per-layer, not process-once cleanup.
    for _ in range(2):
        load, method, layer, source_names = make_case(monkeypatch)

        def check_release(layer=layer, source_names=source_names):
            assert all(not hasattr(layer, name) for name in source_names)
            assert dict(layer.named_parameters()).keys() == expected.keys()
            for name, tensor in layer.named_parameters():
                assert torch.equal(tensor, expected[name])
            assert torch.equal(layer.test_buffer, reference.test_buffer)

        release = Mock(side_effect=check_release)
        monkeypatch.setattr(torch.accelerator, "empty_cache", release)
        load(method, layer)
        release.assert_called_once_with()
        check_release()


@pytest.mark.parametrize("make_case", [_awq_case, _nvfp4_case], ids=["awq", "nvfp4"])
def test_failed_conversion_does_not_run_success_cleanup(monkeypatch, make_case):
    load, method, layer, source_names = make_case(monkeypatch)
    failing_prepare = Mock(side_effect=RuntimeError("conversion failed"))
    monkeypatch.setattr(awq.sm70_ops, "awq_sm70_prepare", failing_prepare)
    monkeypatch.setattr(nvfp4.sm70_ops, "nvfp4_sm70_prepare", failing_prepare)
    release = Mock()
    monkeypatch.setattr(torch.accelerator, "empty_cache", release)
    with pytest.raises(RuntimeError, match="conversion failed"):
        load(method, layer)
    release.assert_not_called()
    assert all(hasattr(layer, name) for name in source_names)
