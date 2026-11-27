# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import skinny_sm70_moe
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a4_nvfp4 as ct_nvfp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
    CompressedTensorsW8A16Fp8,
    _sm70_channel_fp8_qpn8_config,
)


@pytest.mark.parametrize(
    ("override", "a16", "native_enabled", "skinny", "use_turbomind"),
    [
        ({}, True, True, False, True),
        ({"moe_backend": "marlin"}, True, True, False, False),
        ({"intermediate_size_per_partition": 320}, True, True, False, False),
        ({"num_local_experts": 256}, True, True, False, True),
        ({"in_dtype": torch.bfloat16}, True, True, False, True),
        ({}, False, True, False, True),
        ({}, True, False, False, False),
        ({}, True, True, True, False),
        (
            {
                "hidden_dim": 2048,
                "intermediate_size_per_partition": 128,
                "num_experts": 256,
                "num_local_experts": 256,
                "experts_per_token": 8,
            },
            True,
            True,
            False,
            True,
        ),
    ],
)
def test_ct_nvfp4_sm70_backend_selection(
    monkeypatch, override, a16, native_enabled, skinny, use_turbomind
):
    monkeypatch.setattr(
        nvfp4_sm70_moe, "should_use_nvfp4_moe_turbomind", lambda: native_enabled
    )
    monkeypatch.setattr(skinny_sm70_moe, "skinny_backend_admitted", lambda _: skinny)
    monkeypatch.setattr(
        ct_nvfp4,
        "select_nvfp4_moe_backend",
        lambda **_: (ct_nvfp4.NvFp4MoeBackend.MARLIN, None),
    )
    weight = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        group_size=16,
        dynamic=False,
    )
    fmt = CompressionFormat.nvfp4_pack_quantized.value
    scheme = {"weights": weight, "input_activations": None if a16 else weight}
    config = CompressedTensorsConfig(
        {f"experts.0.{proj}_proj": scheme for proj in ("gate", "up", "down")},
        [],
        fmt,
    )
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(
        **(
            dict(
                tp_size=4,
                dp_size=1,
                pcp_size=1,
                use_ep=False,
                is_sequence_parallel=False,
                moe_parallel_config=SimpleNamespace(
                    enable_eplb=False, use_all2all_kernels=False
                ),
                device=torch.device("cuda"),
                in_dtype=torch.float16,
                moe_backend="auto",
                num_experts=512,
                num_local_experts=512,
                experts_per_token=10,
                hidden_dim=2560,
                intermediate_size_per_partition=160,
                activation=MoEActivation.SILU,
                is_act_and_mul=True,
                swiglu_limit=None,
                has_bias=False,
                is_lora_enabled=False,
                apply_router_weight_on_input=False,
            )
            | override
        )
    )
    method = CompressedTensorsMoEMethod.get_moe_method(config, layer, "experts")
    cls = (
        nvfp4_sm70_moe.CompressedTensorsNvfp4SM70MoEMethod
        if use_turbomind
        else ct_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod
    )
    assert type(method) is cls


def _make_channel_fp8_scheme(monkeypatch, *, static_input: bool = False):
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
    return CompressedTensorsW8A16Fp8(weight_quant, static_input)


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

    # process_weights_after_loading binds the native ops with the engine policy.
    def fake_prepare(weight, scales, group_size, gated_silu, native_policy=()):
        prepare_calls.append(
            (tuple(weight.shape), tuple(scales.shape), group_size, gated_silu)
        )
        return (
            torch.empty((4, 6), dtype=torch.uint8),
            torch.ones((1, 6), dtype=torch.float16),
            torch.tensor([4, 6]),
        )

    module = (
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a16_fp8"
    )
    monkeypatch.setattr(
        f"{module}.capture_linear_native_config",
        lambda family: SimpleNamespace(values=()),
    )
    monkeypatch.setattr(f"{module}.sm70_ops.fp8_sm70_prepare", fake_prepare)
    # Isolate dispatch from the real packing geometry in this tiny mocked case.
    monkeypatch.setattr(
        f"{module}._sm70_channel_fp8_shape_is_validated", lambda _: True
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

    # The bound native ops were captured during process_weights_after_loading.
    monkeypatch.setattr(scheme.native_ops, "fp8_gemm_sm70_out", fake_gemm)
    output = scheme.apply_weights(
        layer,
        torch.ones((2, 4), dtype=torch.float16),
        torch.ones(6, dtype=torch.float16),
    )

    assert torch.equal(output, torch.full((2, 6), 3, dtype=torch.float16))
    assert gemm_calls == [((2, 6), (2, 4), (4, 6), (1, 6), 128, 4, 6, False)]


def test_compressed_tensors_channel_fp8_qpn8_shape_gate_requires_alignment():
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
    # QPN8 now supports aligned local projections independently of TP size.
    assert _sm70_channel_fp8_qpn8_config(layer) == (12, 2, False)

    layer.weight = SimpleNamespace(shape=(5121, 1536))
    assert _sm70_channel_fp8_qpn8_config(layer) is None
    layer.weight = SimpleNamespace(shape=(5120, 1537))
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
    monkeypatch.setattr(
        f"{module}.capture_linear_native_config",
        lambda family: SimpleNamespace(values=()),
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
    # The runtime dispatch resolves the shared workspace again inside the QPN8
    # kernel module; it must be the one weight preparation reserved.
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.qpn.fp8."
        "_get_sm70_fp8_prefill_exact_dense_workspace",
        lambda weight: workspace,
    )
    # An SM70 VllmConfig built earlier in the process exports
    # VLLM_SM70_BATCH_GEMM_LAYOUTS=1; this test covers the QPN8 prepare only.
    monkeypatch.setattr(f"{module}.sm70_tm.use_batched_gemm_layouts", lambda: False)

    # process_weights_after_loading binds the native ops with the engine policy.
    def fake_prepare(weight, scales, native_policy=()):
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
        out,
        workspace_ptr,
        x,
        weight,
        scales,
        split_k,
        nacc,
        prefetch,
        gated_silu,
        native_policy=(),
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
    # Exercise the runtime workspace resolver with the CPU native-op stand-in.
    import importlib

    monkeypatch.setattr(
        torch.ops.vllm,
        "sm70_ct_fp8_qpn8_dispatch",
        importlib.import_module(module)._sm70_ct_fp8_qpn8_dispatch,
    )
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


def test_tp2_channel_fp8_shapes_resolve_qpn8_configs():
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
        _sm70_channel_fp8_qpn8_config,
    )

    cases = [
        ("model.layers.0.linear_attn.in_proj_qkvz", (8192, 5120), (16, 2, False)),
        ("model.layers.3.self_attn.qkv_proj", (7168, 5120), (16, 2, False)),
        ("model.layers.0.linear_attn.out_proj", (5120, 3072), (16, 2, False)),
        ("model.layers.0.mlp.down_proj", (5120, 8704), (16, 2, False)),
        ("model.layers.0.mlp.gate_up_proj", (17408, 5120), (16, 2, False)),
    ]
    for prefix, shape, expected in cases:
        layer = SimpleNamespace(prefix=prefix, weight=SimpleNamespace(shape=shape))
        assert _sm70_channel_fp8_qpn8_config(layer) == expected


@torch.inference_mode()
def test_mixed_gdn_unquantized_projections_convert_to_qpn8_and_fuse_ba_tp2():
    import pytest

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA required")

    from vllm.model_executor.kernels.linear.qpn.fp8 import clear_sm70_fp8_workspaces
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.sm70_gdn_ba_verify import (
        apply_gdn_ba_verify,
    )

    def _make_fp16_linear(in_features: int, out_features: int, prefix: str):
        mod = LinearBase(
            input_size=in_features,
            output_size=out_features,
            bias=False,
            params_dtype=torch.float16,
            prefix=prefix,
            disable_tp=True,
        )
        mod.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(
                    (out_features, in_features), device="cuda", dtype=torch.float16
                ),
                requires_grad=False,
            ),
        )
        mod.weight.data.normal_(0, 0.02)
        return mod

    try:
        torch.manual_seed(20261115)
        qkvz = _make_fp16_linear(5120, 8192, "test.layers.0.linear_attn.in_proj_qkvz")
        ba = _make_fp16_linear(5120, 48, "test.layers.0.linear_attn.in_proj_ba")
        out_proj = _make_fp16_linear(3072, 5120, "test.layers.0.linear_attn.out_proj")

        qkvz_ref = qkvz.weight.detach().float().clone()
        ba_ref = ba.weight.detach().float().clone()
        out_ref = out_proj.weight.detach().float().clone()

        qkvz._sm70_gdn_mixed_qpn8 = True
        out_proj._sm70_gdn_mixed_qpn8 = True
        assert isinstance(qkvz.quant_method, UnquantizedLinearMethod)
        qkvz.quant_method.process_weights_after_loading(qkvz)
        ba.quant_method.process_weights_after_loading(ba)
        out_proj.quant_method.process_weights_after_loading(out_proj)

        assert qkvz.sm70_fp8_qpn8
        assert qkvz.weight.dtype == torch.uint8 and qkvz.weight.shape == (5120, 8192)
        assert qkvz.sm70_fp8_prefill_exact_dense_workspace_ptr > 0
        assert not hasattr(out_proj, "_sm70_dense_state")
        assert out_proj.quant_method.supports_direct_output(out_proj)

        # Verify direct-output write on out_proj.
        x_out = torch.randn(5, 3072, device="cuda", dtype=torch.float16) * 0.1
        direct_buf = torch.empty((5, 5120), device="cuda", dtype=torch.float16)
        ret = out_proj.quant_method.apply_weights(out_proj, x_out, output=direct_buf)
        assert ret.data_ptr() == direct_buf.data_ptr()
        expected_out = x_out.float() @ out_ref.t()
        rel_err = (
            (direct_buf.float() - expected_out).norm() / expected_out.norm()
        ).item()
        assert rel_err < 3.5e-2

        # Verify TP=2 fused qkvz + FP16 ba verification dispatch. The layer
        # binds the engine's FP8 policy at build time (#1508); do the same.
        from vllm.config import KernelConfig, set_current_vllm_config
        from vllm.model_executor.layers.quantization import (
            sm70_gdn_ba_verify as ba_verify,
        )
        from vllm.runtime_resources import release_runtime_resources

        verify_cfg = SimpleNamespace(kernel_config=KernelConfig(), lora_config=None)
        verify_cfg.kernel_config.capture_provider_inputs()
        with set_current_vllm_config(verify_cfg):  # type: ignore[arg-type]
            verify_native = ba_verify.bind_gdn_ba_verify_policy(verify_cfg)
        assert verify_native is not None
        gdn_stub = SimpleNamespace(
            in_proj_qkvz=qkvz,
            in_proj_ba=ba,
            enable_sm70_dflash2_fused_gdn_verify=True,
            sm70_gdn_ba_verify_native=verify_native,
            num_v_heads=48,
            tp_size=2,
            head_k_dim=128,
            head_v_dim=128,
            key_dim=2048,
            value_dim=6144,
            gqa_interleaved_layout=False,
            disable_tp_for_ba_proj=False,
        )
        x_in = torch.randn(5, 5120, device="cuda", dtype=torch.float16) * 0.1
        with verify_native.owner.activate():
            fused = apply_gdn_ba_verify(gdn_stub, x_in)
        release_runtime_resources(verify_cfg)  # type: ignore[arg-type]
        assert fused is not None
        qkv_actual, z_actual, b_actual, a_actual = fused
        qkvz_expected = x_in.float() @ qkvz_ref.t()
        ba_expected = x_in.float() @ ba_ref.t()
        assert (
            (qkv_actual.float() - qkvz_expected[:, :5120]).norm()
            / qkvz_expected[:, :5120].norm()
        ).item() < 3.5e-2
        assert (
            (z_actual.float() - qkvz_expected[:, 5120:]).norm()
            / qkvz_expected[:, 5120:].norm()
        ).item() < 3.5e-2
        torch.testing.assert_close(
            b_actual.float(), ba_expected[:, :24], rtol=5e-3, atol=5e-3
        )
        torch.testing.assert_close(
            a_actual.float(), ba_expected[:, 24:], rtol=5e-3, atol=5e-3
        )
    finally:
        clear_sm70_fp8_workspaces()


def test_channel_fp8_qpn8_skips_duplicate_batch_tm_by_default_and_honors_override(
    monkeypatch,
):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a16_fp8 as ct_fp8,
    )

    scheme = object.__new__(ct_fp8.CompressedTensorsW8A16Fp8)
    scheme.strategy = ct_fp8.QuantizationStrategy.CHANNEL
    scheme.is_static_input_scheme = False
    scheme.use_sm70_fp8_turbomind = True
    scheme.use_sm70_fp8_qpn8 = True
    tm_calls: list[bool] = []

    def _mock_fp8_sm70_prepare(w, s, g, gated):
        tm_calls.append(bool(gated))
        return (
            torch.empty((4, 6), dtype=torch.uint8),
            torch.ones((1, 6), dtype=torch.float16),
            torch.tensor([4, 6], dtype=torch.int32),
        )

    mock_native_ops = SimpleNamespace(
        fp8_qpn8_prepare_sm70=lambda w, s: (
            torch.empty((4, 6), dtype=torch.uint8),
            torch.ones((1, 6), dtype=torch.float16),
        ),
        fp8_sm70_prepare=_mock_fp8_sm70_prepare,
        arguments=(),
    )
    monkeypatch.setattr(ct_fp8, "NativeBindings", lambda *_: mock_native_ops)
    monkeypatch.setattr(ct_fp8, "_is_sm70_fp8_qpn8_runtime_contract", lambda: True)
    monkeypatch.setattr(ct_fp8.sm70_tm, "is_exact_sm70_cuda", lambda *_: True)
    monkeypatch.setattr(ct_fp8.sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(ct_fp8.sm70_tm, "use_batched_gemm_layouts", lambda: True)
    monkeypatch.setattr(
        ct_fp8.current_platform, "is_device_capability", lambda cap: cap == 70
    )
    monkeypatch.setattr(torch.ops._C, "fp8_sm70_prepare", object(), raising=False)
    monkeypatch.setattr(
        ct_fp8, "_sm70_channel_fp8_qpn8_config", lambda layer: (16, 2, False)
    )
    monkeypatch.setattr(ct_fp8, "_missing_sm70_fp8_qpn8_ops", lambda: [])
    workspace = torch.empty((4, 6), dtype=torch.float16)
    monkeypatch.setattr(
        ct_fp8, "_get_sm70_fp8_prefill_exact_dense_workspace", lambda weight: workspace
    )

    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=5120),
            quantization="compressed-tensors",
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
        kernel_config=SimpleNamespace(
            sm70_fp8=SimpleNamespace(
                enabled=True,
                qpn8=True,
                batch_prescaled=True,
                prefill_exact_dense=True,
                sources={"qpn8": "default", "batch_prescaled": "default"},
                native=SimpleNamespace(values=(), resolve=lambda *_: None),
                resolve=lambda: None,
            )
        ),
    )

    def _make_layer():
        mod = torch.nn.Module()
        mod.prefix = "model.layers.0.mlp.down_proj"
        mod.output_size_per_partition = 6
        mod.orig_dtype = torch.float16
        mod.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.zeros((6, 4), dtype=torch.float8_e4m3fn), requires_grad=False
            ),
        )
        mod.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.ones((6, 1), dtype=torch.float32), requires_grad=False
            ),
        )
        return mod

    with set_current_vllm_config(cfg):  # type: ignore[arg-type]
        default_layer = _make_layer()
        scheme.process_weights_after_loading(default_layer)
        assert default_layer.sm70_fp8_qpn8
        assert not getattr(default_layer, "sm70_fp8_batch_tm", False)
        assert not hasattr(default_layer, "sm70_fp8_batch_tm_weight")
        assert tm_calls == []

        cfg.kernel_config.sm70_fp8.sources["batch_prescaled"] = "env"
        explicit_layer = _make_layer()
        scheme.process_weights_after_loading(explicit_layer)
        assert explicit_layer.sm70_fp8_qpn8
        assert explicit_layer.sm70_fp8_batch_tm
        assert hasattr(explicit_layer, "sm70_fp8_batch_tm_weight")
        assert len(tm_calls) == 1
