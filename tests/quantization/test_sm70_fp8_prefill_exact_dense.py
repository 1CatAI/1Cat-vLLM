# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

import vllm.envs as envs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
    CompressedTensorsW8A16Fp8,
    _sm70_channel_fp8_qpn8_config,
    _sm70_fp8_qpn8_enabled,
)
from vllm.model_executor.layers.quantization.fp8 import (
    _SM70_FP8_EXACT_8K_PREFILL_M,
    _SM70_FP8_PREFILL_DENSE_MIN_M,
    _SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES,
    Fp8LinearMethod,
    _get_sm70_fp8_prefill_exact_dense_workspace,
    _is_sm70_dsv4_fp8_prescaled_decode_layer,
    _is_sm70_fp8_exact_8k_prefill_layer,
    _is_sm70_fp8_prefill_exact_dense_layer,
    _is_sm70_fp8_qpn8_layer,
    _is_sm70_fp8_qpn8_runtime_contract,
    _sm70_fp8_prefill_dense_workspaces,
    _sm70_fp8_prefill_visible_dense_mm,
)


def _make_channel_fp8_scheme(
    monkeypatch,
    *,
    static_input: bool = False,
    enable_qpn8_by_default: bool = False,
):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a16_fp8.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float16)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a16_fp8.sm70_tm.is_exact_sm70_cuda_platform",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a16_fp8.sm70_tm.use_turbomind",
        lambda enabled: enabled,
    )
    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.CHANNEL,
        dynamic=False,
    )
    return CompressedTensorsW8A16Fp8(
        weight_quant,
        static_input,
        enable_sm70_qpn8_by_default=enable_qpn8_by_default,
    )


def test_compressed_tensors_channel_fp8_selects_turbomind_on_sm70(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FP8_TURBOMIND", "1")
    envs.disable_envs_cache()
    try:
        assert _make_channel_fp8_scheme(monkeypatch).use_sm70_fp8_turbomind
        assert not _make_channel_fp8_scheme(
            monkeypatch, static_input=True
        ).use_sm70_fp8_turbomind
    finally:
        envs.disable_envs_cache()


def test_channel_fp8_qpn8_uses_mixed_nvfp4_default_only(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8", raising=False)
    envs.disable_envs_cache()
    try:
        assert not _make_channel_fp8_scheme(monkeypatch).use_sm70_fp8_qpn8
        assert _make_channel_fp8_scheme(
            monkeypatch,
            enable_qpn8_by_default=True,
        ).use_sm70_fp8_qpn8

        monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "0")
        envs.disable_envs_cache()
        assert not _make_channel_fp8_scheme(
            monkeypatch,
            enable_qpn8_by_default=True,
        ).use_sm70_fp8_qpn8
    finally:
        envs.disable_envs_cache()


def test_compressed_tensors_channel_fp8_prepares_and_dispatches_turbomind(
    monkeypatch,
):
    scheme = object.__new__(CompressedTensorsW8A16Fp8)
    scheme.use_sm70_fp8_turbomind = True
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((6, 4), dtype=torch.float8_e4m3fn), requires_grad=False
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((6, 1)), requires_grad=False),
    )
    layer.orig_dtype = torch.float16
    layer.output_size_per_partition = 6
    prepare_calls = []

    monkeypatch.setattr(torch.ops._C, "fp8_sm70_prepare", object(), raising=False)

    def fake_prepare(weight, scales, group_size, gated_silu):
        prepare_calls.append(
            (tuple(weight.shape), tuple(scales.shape), group_size, gated_silu)
        )
        return (
            torch.empty((4, 6), dtype=torch.uint8),
            torch.ones((1, 6), dtype=torch.float16),
            torch.tensor([4, 6]),
        )

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a16_fp8.sm70_ops.fp8_sm70_prepare",
        fake_prepare,
    )
    scheme.process_weights_after_loading(layer)

    assert prepare_calls == [((6, 4), (6, 1), 128, False)]
    assert layer.sm70_fp8_turbomind
    assert layer.sm70_fp8_channel_scale
    assert tuple(layer.weight.shape) == (4, 6)
    assert tuple(layer.weight_scale_inv.shape) == (1, 6)
    assert not hasattr(layer, "weight_scale")

    gemm_calls = []

    def fake_gemm(out, x, weight, scales, group_size, k_ld, q_ld, gated_silu):
        gemm_calls.append(
            (
                tuple(out.shape),
                tuple(x.shape),
                tuple(weight.shape),
                tuple(scales.shape),
                group_size,
                k_ld,
                q_ld,
                gated_silu,
            )
        )
        out.fill_(2)

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a16_fp8.sm70_ops.fp8_gemm_sm70_out",
        fake_gemm,
    )
    output = scheme.apply_weights(
        layer,
        torch.ones((2, 4), dtype=torch.float16),
        torch.ones(6, dtype=torch.float16),
    )

    assert torch.equal(output, torch.full((2, 6), 3, dtype=torch.float16))
    assert gemm_calls == [((2, 6), (2, 4), (4, 6), (1, 6), 128, 4, 6, False)]


def test_compressed_tensors_channel_fp8_qpn8_shape_gate_is_exact():
    layer = SimpleNamespace(
        tp_size=4,
        prefix="model.language_model.layers.0.linear_attn.in_proj_qkvz",
        weight=SimpleNamespace(shape=(4096, 5120)),
    )
    assert _sm70_channel_fp8_qpn8_config(layer) == (16, 2, False)

    layer.prefix = "model.language_model.layers.3.self_attn.qkv_proj"
    layer.weight = SimpleNamespace(shape=(3584, 5120))
    assert _sm70_channel_fp8_qpn8_config(layer) == (16, 2, False)

    layer.prefix = "model.language_model.layers.3.self_attn.o_proj"
    layer.weight = SimpleNamespace(shape=(5120, 1536))
    assert _sm70_channel_fp8_qpn8_config(layer) == (12, 2, False)

    layer.prefix = "model.language_model.lm_head"
    layer.weight = SimpleNamespace(shape=(62080, 5120))
    assert _sm70_channel_fp8_qpn8_config(layer) is None

    layer.tp_size = 2
    layer.prefix = "model.language_model.layers.3.self_attn.o_proj"
    layer.weight = SimpleNamespace(shape=(5120, 1536))
    assert _sm70_channel_fp8_qpn8_config(layer) is None


def test_compressed_tensors_channel_fp8_qpn8_prepares_and_dispatches(monkeypatch):
    scheme = object.__new__(CompressedTensorsW8A16Fp8)
    scheme.use_sm70_fp8_turbomind = True
    scheme.use_sm70_fp8_qpn8 = True
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((6, 4), dtype=torch.float8_e4m3fn), requires_grad=False
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((6, 1)), requires_grad=False),
    )
    layer.orig_dtype = torch.float16
    layer.output_size_per_partition = 6
    layer.prefix = "model.language_model.layers.63.mlp.gate_up_proj"
    workspace = torch.empty(1, dtype=torch.float16)
    prepare_calls = []

    module = (
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a16_fp8"
    )
    monkeypatch.setattr(torch.ops._C, "fp8_sm70_prepare", object(), raising=False)
    monkeypatch.setattr(
        f"{module}._sm70_channel_fp8_qpn8_config", lambda layer: (16, 2, False)
    )
    monkeypatch.setattr(f"{module}._is_sm70_fp8_qpn8_runtime_contract", lambda: True)
    monkeypatch.setattr(f"{module}._missing_sm70_fp8_qpn8_ops", lambda: [])
    monkeypatch.setattr(
        f"{module}._get_sm70_fp8_prefill_exact_dense_workspace",
        lambda weight: workspace,
    )

    def fake_prepare(weight, scales):
        prepare_calls.append((tuple(weight.shape), tuple(scales.shape)))
        return (
            torch.empty((4, 6), dtype=torch.uint8),
            torch.ones((1, 6), dtype=torch.float16),
        )

    monkeypatch.setattr(f"{module}.sm70_ops.fp8_qpn8_prepare_sm70", fake_prepare)
    scheme.process_weights_after_loading(layer)

    assert prepare_calls == [((6, 4), (6, 1))]
    assert layer.sm70_fp8_turbomind
    assert layer.sm70_fp8_qpn8
    assert layer.sm70_fp8_channel_scale
    assert layer.sm70_fp8_qpn8_split_k == 16
    assert layer.sm70_fp8_qpn8_nacc == 2
    assert not layer.sm70_fp8_qpn8_prefetch
    assert layer.sm70_fp8_gated_silu
    assert layer.sm70_fp8_qpn8_gated_split_k == 8
    assert layer.sm70_fp8_qpn8_gated_nacc == 2
    assert not layer.sm70_fp8_qpn8_gated_prefetch
    assert tuple(layer.weight.shape) == (4, 6)
    assert tuple(layer.weight_scale_inv.shape) == (1, 6)
    assert not hasattr(layer, "weight_scale")

    dispatch_calls = []

    def fake_dispatch(
        out, workspace_ptr, x, weight, scales, split_k, nacc, prefetch, gated_silu
    ):
        dispatch_calls.append(
            (
                tuple(out.shape),
                tuple(x.shape),
                workspace_ptr,
                tuple(weight.shape),
                tuple(scales.shape),
                split_k,
                nacc,
                prefetch,
                gated_silu,
            )
        )
        out.fill_(2)

    monkeypatch.setattr(f"{module}.sm70_ops.fp8_qpn8_dispatch_sm70_out", fake_dispatch)
    output = scheme.apply_weights(
        layer,
        torch.ones((2, 4), dtype=torch.float16),
        torch.ones(6, dtype=torch.float16),
    )

    assert torch.equal(output, torch.full((2, 6), 3, dtype=torch.float16))
    fused_output = scheme.apply_fused_silu_and_mul(
        layer, torch.ones((2, 4), dtype=torch.float16)
    )
    assert fused_output is not None
    assert torch.equal(fused_output, torch.full((2, 3), 2, dtype=torch.float16))
    assert dispatch_calls == [
        (
            (2, 6),
            (2, 4),
            workspace.data_ptr(),
            (4, 6),
            (1, 6),
            16,
            2,
            False,
            False,
        ),
        (
            (2, 3),
            (2, 4),
            workspace.data_ptr(),
            (4, 6),
            (1, 6),
            8,
            2,
            False,
            True,
        ),
    ]


def test_compressed_tensors_linear_method_delegates_fused_apply():
    expected = torch.ones((1, 3), dtype=torch.float16)
    layer = SimpleNamespace(
        scheme=SimpleNamespace(
            apply_fused_silu_and_mul=lambda layer, x: expected,
        )
    )
    method = object.__new__(CompressedTensorsLinearMethod)

    assert (
        method.apply_fused_silu_and_mul(layer, torch.ones((1, 4), dtype=torch.float16))
        is expected
    )


def test_fp8_prefill_exact_dense_is_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_PREFILL_EXACT_DENSE", raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_FP8_PREFILL_EXACT_DENSE
    finally:
        envs.disable_envs_cache()


def test_fp8_exact_8k_prefill_controls_are_generic_defaults(monkeypatch):
    for name in (
        "VLLM_SM70_FP8_PREFILL_FAST_SELECTOR",
        "VLLM_SM70_FP8_PREFILL_PRESCALED",
        "VLLM_SM70_FP8_PREFILL_CUTLASS",
        "VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM",
    ):
        monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_FP8_PREFILL_FAST_SELECTOR
        assert envs.VLLM_SM70_FP8_PREFILL_PRESCALED
        assert envs.VLLM_SM70_FP8_PREFILL_CUTLASS
        assert not envs.VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM
    finally:
        envs.disable_envs_cache()


def test_fp8_qpn8_is_opt_in_except_for_mixed_nvfp4(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8", raising=False)
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8_LIBRARY", raising=False)
    envs.disable_envs_cache()
    try:
        assert not envs.VLLM_SM70_FP8_QPN8
        assert envs.VLLM_SM70_FP8_QPN8_LIBRARY is None
        assert not _sm70_fp8_qpn8_enabled(False)
        assert _sm70_fp8_qpn8_enabled(True)
        monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "0")
        envs.disable_envs_cache()
        assert not envs.VLLM_SM70_FP8_QPN8
        assert not _sm70_fp8_qpn8_enabled(True)
        monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "1")
        envs.disable_envs_cache()
        assert envs.VLLM_SM70_FP8_QPN8
        assert _sm70_fp8_qpn8_enabled(False)
    finally:
        envs.disable_envs_cache()


def test_fp8_prefill_exact_dense_shape_gate_is_narrow():
    layer = SimpleNamespace(
        tp_size=4,
        weight_block_size=[128, 128],
        prefix="model.language_model.layers.1.mlp.gate_up_proj",
        weight=SimpleNamespace(shape=(5120, 8704)),
    )

    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)

    layer.tp_size = 2
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.tp_size = 4
    layer.prefix = "model.language_model.layers.1.self_attn.qkv_proj"
    layer.weight = SimpleNamespace(shape=(5120, 3584))
    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.linear_attn.in_proj_qkvz"
    layer.weight = SimpleNamespace(shape=(5120, 4096))
    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.mlp.gate_up_proj"
    layer.weight = SimpleNamespace(shape=(5120, 8192))
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.weight_block_size = [64, 128]
    layer.weight = SimpleNamespace(shape=(5120, 8704))
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)


def test_fp8_exact_8k_prefill_gate_uses_operator_contract_only():
    layer = SimpleNamespace(
        tp_size=4,
        weight_block_size=[128, 128],
        prefix="engine.layers.1.linear_attn.in_proj_qkvz",
        weight=SimpleNamespace(shape=(5120, 4096)),
    )

    assert _is_sm70_fp8_exact_8k_prefill_layer(layer)
    layer.prefix = "engine.layers.3.self_attn.qkv_proj"
    layer.weight = SimpleNamespace(shape=(5120, 3584))
    assert _is_sm70_fp8_exact_8k_prefill_layer(layer)

    layer.tp_size = 2
    assert not _is_sm70_fp8_exact_8k_prefill_layer(layer)
    layer.tp_size = 4
    layer.weight_block_size = [64, 128]
    assert not _is_sm70_fp8_exact_8k_prefill_layer(layer)
    layer.weight_block_size = [128, 128]
    layer.weight = SimpleNamespace(shape=(5120, 4096))
    assert not _is_sm70_fp8_exact_8k_prefill_layer(layer)


def test_fp8_qpn8_shape_gate_uses_checkpoint_native_layout():
    layer = SimpleNamespace(
        tp_size=4,
        weight_block_size=[128, 128],
        prefix="model.language_model.layers.1.mlp.gate_up_proj",
        input_size_per_partition=5120,
        output_size_per_partition=8704,
        weight=SimpleNamespace(shape=(8704, 5120)),
    )

    assert _is_sm70_fp8_qpn8_layer(layer)

    layer.tp_size = 2
    assert not _is_sm70_fp8_qpn8_layer(layer)
    layer.tp_size = 4
    layer.prefix = "model.language_model.layers.1.linear_attn.in_proj_qkvz"
    layer.output_size_per_partition = 4096
    layer.weight = SimpleNamespace(shape=(4096, 5120))
    assert _is_sm70_fp8_qpn8_layer(layer)
    layer.prefix = "model.language_model.layers.3.self_attn.qkv_proj"
    layer.output_size_per_partition = 3584
    layer.weight = SimpleNamespace(shape=(3584, 5120))
    assert _is_sm70_fp8_qpn8_layer(layer)
    layer.output_size_per_partition = 4096
    assert not _is_sm70_fp8_qpn8_layer(layer)
    layer.output_size_per_partition = 3584
    layer.weight_block_size = [64, 128]
    assert not _is_sm70_fp8_qpn8_layer(layer)


def test_fp8_qpn8_runtime_gate_uses_engine_contract_only(monkeypatch):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        speculative_config=None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.get_current_vllm_config",
        lambda: vllm_config,
    )

    assert _is_sm70_fp8_qpn8_runtime_contract()
    vllm_config.scheduler_config.max_num_seqs = 16
    assert not _is_sm70_fp8_qpn8_runtime_contract()
    vllm_config.scheduler_config.max_num_seqs = 8
    vllm_config.speculative_config = object()
    assert not _is_sm70_fp8_qpn8_runtime_contract()
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "1")
    assert _is_sm70_fp8_qpn8_runtime_contract()


def test_fp8_prefill_exact_dense_workspace_is_bounded():
    assert _SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES == 85 * 1024**2


def test_fp8_prefill_exact_dense_workspace_is_reused(monkeypatch):
    workspace = torch.empty(1, dtype=torch.float16)
    allocations = []

    def fake_empty(shape, *, dtype, device):
        allocations.append((shape, dtype, device))
        return workspace

    _sm70_fp8_prefill_dense_workspaces.clear()
    monkeypatch.setattr(torch, "empty", fake_empty)
    weight = SimpleNamespace(device=torch.device("cuda:0"))

    try:
        first = _get_sm70_fp8_prefill_exact_dense_workspace(weight)
        second = _get_sm70_fp8_prefill_exact_dense_workspace(weight)

        assert first is workspace
        assert second is workspace
        assert len(allocations) == 1
    finally:
        _sm70_fp8_prefill_dense_workspaces.clear()


def test_fp8_prefill_visible_dense_mm_is_long_prefill_only(monkeypatch):
    workspace = torch.empty(24, dtype=torch.float16)
    weight = torch.empty((4, 6), dtype=torch.uint8)
    scales = torch.empty((1, 6), dtype=torch.float16)
    calls = []

    def fake_dequantize(out, qweight, factors, group_size):
        calls.append((qweight, factors, group_size))
        out.fill_(1)

    monkeypatch.setenv("VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM", "1")
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops.fp8_sm70_dequantize_out",
        fake_dequantize,
    )
    envs.disable_envs_cache()
    _sm70_fp8_prefill_dense_workspaces[(0, torch.float16)] = workspace
    try:
        long_prefill = _sm70_fp8_prefill_visible_dense_mm(
            torch.ones((_SM70_FP8_PREFILL_DENSE_MIN_M, 4), dtype=torch.float16),
            weight,
            scales,
            workspace.data_ptr(),
            gated_silu=False,
            min_prefill_m=_SM70_FP8_PREFILL_DENSE_MIN_M,
        )
        tail = _sm70_fp8_prefill_visible_dense_mm(
            torch.ones((1, 4), dtype=torch.float16),
            weight,
            scales,
            workspace.data_ptr(),
            gated_silu=False,
            min_prefill_m=_SM70_FP8_PREFILL_DENSE_MIN_M,
        )
    finally:
        _sm70_fp8_prefill_dense_workspaces.clear()
        envs.disable_envs_cache()

    assert long_prefill is not None
    assert long_prefill.shape == (_SM70_FP8_PREFILL_DENSE_MIN_M, 6)
    assert tail is None
    assert calls == [(weight, scales, 128)]


def test_fp8_prefill_dispatch_reaches_runtime_op_for_small_and_large_m(monkeypatch):
    calls = []

    def fake_dispatch(
        out,
        dense_weight_ptr,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
        min_prefill_m,
    ):
        assert dense_weight_ptr == 42
        calls.append((input.shape[0], min_prefill_m, gated_silu))
        out.zero_()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops."
        "fp8_gemm_sm70_prefill_dispatch_out",
        fake_dispatch,
    )
    layer = SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_bmm=False,
        output_size_per_partition=6,
        weight=torch.empty((4, 6), dtype=torch.uint8),
        weight_scale_inv=torch.empty((1, 6), dtype=torch.float16),
        sm70_fp8_k_ld=4,
        sm70_fp8_q_ld=6,
        sm70_fp8_prefill_exact_dense_workspace_ptr=42,
    )
    method = SimpleNamespace()

    for m in (1, _SM70_FP8_PREFILL_DENSE_MIN_M):
        output = Fp8LinearMethod.apply(
            method, layer, torch.empty((m, 4), dtype=torch.float16)
        )
        assert output.shape == (m, 6)

    assert calls == [
        (1, _SM70_FP8_PREFILL_DENSE_MIN_M, False),
        (
            _SM70_FP8_PREFILL_DENSE_MIN_M,
            _SM70_FP8_PREFILL_DENSE_MIN_M,
            False,
        ),
    ]


def test_fp8_prefill_prescaled_scales_only_reach_exact_8k_route(monkeypatch):
    calls = []

    def fake_default(out, input, qweight, scales, *args):
        calls.append(("default", input.shape[0], scales))
        out.zero_()

    def fake_prescaled(out, input, qweight, scales, *args):
        calls.append(("prescaled", input.shape[0], scales))
        out.zero_()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops.fp8_gemm_sm70_out",
        fake_default,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops."
        "fp8_gemm_sm70_prefill_prescaled_out",
        fake_prescaled,
    )
    normal_scales = torch.empty((1, 6), dtype=torch.float16)
    prescaled_scales = torch.empty((1, 6), dtype=torch.float16)
    layer = SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_bmm=False,
        output_size_per_partition=6,
        weight=torch.empty((4, 6), dtype=torch.uint8),
        weight_scale_inv=normal_scales,
        sm70_fp8_prefill_prescaled_scales=prescaled_scales,
        sm70_fp8_k_ld=4,
        sm70_fp8_q_ld=6,
    )
    method = SimpleNamespace()

    monkeypatch.setenv("VLLM_SM70_FP8_PREFILL_FAST_SELECTOR", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_PREFILL_PRESCALED", "1")
    envs.disable_envs_cache()
    try:
        Fp8LinearMethod.apply(method, layer, torch.empty((1, 4), dtype=torch.float16))
        Fp8LinearMethod.apply(
            method,
            layer,
            torch.empty((_SM70_FP8_EXACT_8K_PREFILL_M, 4), dtype=torch.float16),
        )
        monkeypatch.setenv("VLLM_SM70_FP8_PREFILL_PRESCALED", "0")
        envs.disable_envs_cache()
        Fp8LinearMethod.apply(
            method,
            layer,
            torch.empty((_SM70_FP8_EXACT_8K_PREFILL_M, 4), dtype=torch.float16),
        )
    finally:
        envs.disable_envs_cache()

    assert [(route, m) for route, m, _ in calls] == [
        ("default", 1),
        ("prescaled", _SM70_FP8_EXACT_8K_PREFILL_M),
        ("default", _SM70_FP8_EXACT_8K_PREFILL_M),
    ]
    assert calls[0][2] is normal_scales
    assert calls[1][2] is prescaled_scales
    assert calls[2][2] is normal_scales


def test_dsv4_fp8_prescaled_decode_admission_is_exact():
    layer = SimpleNamespace(
        prefix="model.layers.2.attn.fused_wqa_wkv",
        tp_size=1,
        weight_block_size=[128, 128],
        input_size_per_partition=4096,
        output_size_per_partition=1536,
        weight=torch.empty((1536, 4096), device="meta"),
    )
    assert _is_sm70_dsv4_fp8_prescaled_decode_layer(layer)

    layer.tp_size = 4
    assert not _is_sm70_dsv4_fp8_prescaled_decode_layer(layer)
    layer.tp_size = 1
    layer.prefix = "model.layers.2.attn.indexer.fused_wqa_wkv"
    assert _is_sm70_dsv4_fp8_prescaled_decode_layer(layer)
    layer.prefix = "model.layers.2.attn.wq_b"
    assert not _is_sm70_dsv4_fp8_prescaled_decode_layer(layer)


def test_dsv4_fp8_prescaled_decode_only_handles_m1(monkeypatch):
    calls = []

    def fake_default(out, input, qweight, scales, *args):
        calls.append(("default", input.shape[0], scales))
        out.zero_()

    def fake_prescaled(out, input, qweight, scales, *args):
        calls.append(("prescaled", input.shape[0], scales))
        out.zero_()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops.fp8_gemm_sm70_out",
        fake_default,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops."
        "fp8_gemm_sm70_prefill_prescaled_out",
        fake_prescaled,
    )
    normal_scales = torch.empty((1, 6), dtype=torch.float16)
    prescaled_scales = torch.empty((1, 6), dtype=torch.float16)
    layer = SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_bmm=False,
        output_size_per_partition=6,
        weight=torch.empty((4, 6), dtype=torch.uint8),
        weight_scale_inv=normal_scales,
        sm70_fp8_decode_prescaled_scales=prescaled_scales,
        sm70_fp8_k_ld=4,
        sm70_fp8_q_ld=6,
    )
    method = SimpleNamespace()

    monkeypatch.setenv("VLLM_SM70_DSV4_FP8_PRESCALED_DECODE", "1")
    envs.disable_envs_cache()
    try:
        Fp8LinearMethod.apply(method, layer, torch.empty((1, 4), dtype=torch.float16))
        Fp8LinearMethod.apply(method, layer, torch.empty((2, 4), dtype=torch.float16))
        monkeypatch.setenv("VLLM_SM70_DSV4_FP8_PRESCALED_DECODE", "0")
        envs.disable_envs_cache()
        Fp8LinearMethod.apply(method, layer, torch.empty((1, 4), dtype=torch.float16))
    finally:
        envs.disable_envs_cache()

    assert [(route, m) for route, m, _ in calls] == [
        ("prescaled", 1),
        ("default", 2),
        ("default", 1),
    ]
    assert calls[0][2] is prescaled_scales
    assert calls[1][2] is normal_scales
    assert calls[2][2] is normal_scales


def test_fp8_qpn8_dispatches_small_m_and_workspace_fallback(monkeypatch):
    calls = []

    def fake_dispatch(
        out, workspace, input, codes, scales, split_k, nacc, prefetch, gated_silu
    ):
        calls.append(
            ("dispatch", input.shape[0], workspace, split_k, nacc, prefetch, gated_silu)
        )
        out.zero_()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops."
        "fp8_qpn8_dispatch_sm70_out",
        fake_dispatch,
    )
    layer = SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_qpn8=True,
        output_size_per_partition=6,
        weight=torch.empty((4, 6), dtype=torch.uint8),
        weight_scale_inv=torch.empty((1, 1), dtype=torch.float16),
        sm70_fp8_qpn8_split_k=16,
        sm70_fp8_qpn8_nacc=1,
        sm70_fp8_qpn8_prefetch=False,
        sm70_fp8_prefill_exact_dense_workspace_ptr=42,
    )
    method = SimpleNamespace()

    for m in (1, 9):
        output = Fp8LinearMethod.apply(
            method, layer, torch.empty((m, 4), dtype=torch.float16)
        )
        assert output.shape == (m, 6)

    assert calls == [
        ("dispatch", 1, 42, 16, 1, False, False),
        ("dispatch", 9, 42, 16, 1, False, False),
    ]


def test_fp8_qpn8_fused_gate_dispatches_without_intermediate(monkeypatch):
    calls = []

    def fake_dispatch(
        out, workspace, input, codes, scales, split_k, nacc, prefetch, gated_silu
    ):
        calls.append(
            ("dispatch", input.shape[0], workspace, split_k, nacc, prefetch, gated_silu)
        )
        out.zero_()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops."
        "fp8_qpn8_dispatch_sm70_out",
        fake_dispatch,
    )
    layer = SimpleNamespace(
        sm70_fp8_qpn8=True,
        sm70_fp8_gated_silu=True,
        output_size_per_partition=12,
        weight=torch.empty((4, 12), dtype=torch.uint8),
        weight_scale_inv=torch.empty((1, 1), dtype=torch.float16),
        sm70_fp8_qpn8_gated_split_k=8,
        sm70_fp8_qpn8_gated_nacc=2,
        sm70_fp8_qpn8_gated_prefetch=True,
        sm70_fp8_prefill_exact_dense_workspace_ptr=42,
    )
    method = SimpleNamespace()

    for m in (8, 16):
        output = Fp8LinearMethod.apply_fused_silu_and_mul(
            method, layer, torch.empty((m, 4), dtype=torch.float16)
        )
        assert output is not None
        assert output.shape == (m, 6)

    assert calls == [
        ("dispatch", 8, 42, 8, 2, True, True),
        ("dispatch", 16, 42, 8, 2, True, True),
    ]
