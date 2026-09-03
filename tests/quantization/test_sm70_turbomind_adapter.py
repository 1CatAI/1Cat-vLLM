# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import torch


def _load_adapter():
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "vllm"
        / "model_executor"
        / "layers"
        / "quantization"
        / "sm70_turbomind.py"
    )
    spec = importlib.util.spec_from_file_location("sm70_turbomind_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gptq_unpack_weight_and_zeros():
    tm = _load_adapter()
    packed = torch.tensor([[0x76543210]], dtype=torch.int32)

    weight = tm.unpack_gptq_weight(packed)
    zeros = tm.unpack_gptq_zeros(packed)

    assert weight.dtype == torch.uint8
    assert weight.tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert zeros.dtype == torch.float16
    assert zeros.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]


def test_compressed_unpack_transposes_to_turbomind_layout():
    tm = _load_adapter()
    weight_packed = torch.tensor([[0x3210], [0x7654]], dtype=torch.int32)
    zeros_packed = torch.tensor([[0x76543210]], dtype=torch.int32)

    weight = tm.unpack_compressed_weight(weight_packed)
    zeros = tm.unpack_compressed_zeros(zeros_packed)

    expected_weight = [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert weight.dtype == torch.uint8
    assert weight.tolist() == expected_weight
    assert zeros.dtype == torch.float16
    assert zeros.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_mxfp4_unpack_uint8_blocks_transposes_to_turbomind_layout():
    tm = _load_adapter()
    packed = torch.tensor([[0x10, 0x32], [0x54, 0x76]], dtype=torch.uint8)

    weight = tm.unpack_mxfp4_weight(packed)

    assert weight.dtype == torch.uint8
    assert weight.tolist() == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_mxfp4_unpack_flattens_last_two_block_dims_like_lmdeploy():
    tm = _load_adapter()
    packed = torch.tensor([[[0x10], [0x32]], [[0x54], [0x76]]], dtype=torch.uint8)

    weight = tm.unpack_mxfp4_weight(packed)

    assert weight.dtype == torch.uint8
    assert weight.tolist() == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_nvfp4_prepare_pads_output_to_converter_alignment(monkeypatch):
    tm = _load_adapter()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((24, 16), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((24, 2), dtype=torch.float16), requires_grad=False
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor(0.25, dtype=torch.float32), requires_grad=False
    )

    from vllm import _sm70_ops as sm70_ops

    prepared = []

    def fake_prepare(qweight, scales, group_size, interleave_gated_silu):
        prepared.append(
            (
                tuple(qweight.shape),
                tuple(scales.shape),
                group_size,
                interleave_gated_silu,
            )
        )
        return (
            torch.empty((32, 4), dtype=torch.int32),
            torch.empty((2, 32), dtype=torch.float16),
            torch.tensor([32, 32], dtype=torch.int64),
        )

    monkeypatch.setattr(sm70_ops, "nvfp4_sm70_prepare", fake_prepare)

    tm.prepare_nvfp4_linear(layer)

    state = getattr(layer, tm.STATE_ATTR)
    assert prepared == [((32, 32), (2, 32), 16, False)]
    assert state.output_size == 24


def test_nvfp4_apply_crops_converter_padding(monkeypatch):
    tm = _load_adapter()
    layer = torch.nn.Module()
    state = tm.SM70TurboMindLinearState(
        weight=torch.empty((32, 4), dtype=torch.int32),
        scales=torch.empty((2, 32), dtype=torch.float16),
        group_size=16,
        k_ld=32,
        q_ld=32,
        output_size=24,
        op_kind="nvfp4",
        padded_output_size=32,
    )
    setattr(layer, tm.STATE_ATTR, state)

    from vllm import _sm70_ops as sm70_ops

    def fake_gemm(out, *args):
        del args
        out.copy_(torch.arange(32, dtype=out.dtype).view(1, 32))

    monkeypatch.setattr(sm70_ops, "nvfp4_gemm_sm70_out", fake_gemm)

    output = tm.apply_prepared_linear(
        layer, torch.ones((1, 32), dtype=torch.float16), bias=None
    )

    assert output.shape == (1, 24)
    assert output.tolist() == [list(map(float, range(24)))]


def test_symmetric_int4_zero_points_are_eight():
    tm = _load_adapter()
    scales = torch.ones((2, 3), dtype=torch.float32)

    zeros = tm.symmetric_int4_zeros_like(scales)

    assert zeros.dtype == torch.float16
    assert zeros.tolist() == [[8, 8, 8], [8, 8, 8]]


def _make_nvfp4_state(tm, *, gated_silu: bool):
    return tm.SM70TurboMindLinearState(
        weight=torch.empty((3, 1), dtype=torch.uint8),
        scales=torch.empty((1, 4), dtype=torch.float16),
        group_size=16,
        k_ld=3,
        q_ld=4,
        output_size=4,
        op_kind="nvfp4",
        gated_silu=gated_silu,
    )


def _make_nvfp4_qpn4_state(tm, *, gated_silu: bool):
    output_size = 4
    return tm.SM70TurboMindLinearState(
        weight=torch.empty((3, output_size // 2), dtype=torch.uint8),
        scales=torch.empty(
            (1, output_size),
            dtype=torch.uint8 if gated_silu else torch.float16,
        ),
        group_size=16,
        k_ld=0,
        q_ld=0,
        output_size=output_size,
        op_kind="nvfp4_qpn4",
        gated_silu=gated_silu,
        dense_weight_ptr=1234,
        global_scale=0.25 if gated_silu else 0.0,
        use_scale_code=gated_silu,
    )


def test_nvfp4_gated_layout_is_deinterleaved_for_unfused_apply(monkeypatch):
    tm = _load_adapter()
    layer = torch.nn.Module()
    setattr(layer, tm.STATE_ATTR, _make_nvfp4_state(tm, gated_silu=True))

    from vllm import _sm70_ops as sm70_ops

    calls = []

    def fake_gemm(
        out,
        x,
        weight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu=False,
    ):
        del x, weight, scales, group_size, k_ld, q_ld
        calls.append(gated_silu)
        out.copy_(torch.tensor([[1.0, 10.0, 2.0, 20.0]], dtype=out.dtype))

    monkeypatch.setattr(sm70_ops, "nvfp4_gemm_sm70_out", fake_gemm)

    output = tm.apply_prepared_linear(
        layer,
        torch.ones((1, 3), dtype=torch.float16),
        bias=None,
    )

    assert calls == [False]
    assert output.tolist() == [[1.0, 2.0, 10.0, 20.0]]


def test_nvfp4_gated_layout_uses_fused_silu_epilogue(monkeypatch):
    tm = _load_adapter()
    layer = torch.nn.Module()
    setattr(layer, tm.STATE_ATTR, _make_nvfp4_state(tm, gated_silu=True))

    from vllm import _sm70_ops as sm70_ops

    calls = []

    def fake_gemm(
        out,
        x,
        weight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu=False,
    ):
        del weight, scales
        calls.append(
            {
                "out_shape": tuple(out.shape),
                "x_shape": tuple(x.shape),
                "group_size": group_size,
                "k_ld": k_ld,
                "q_ld": q_ld,
                "gated_silu": gated_silu,
            }
        )
        out.fill_(3.0)

    monkeypatch.setattr(sm70_ops, "nvfp4_gemm_sm70_out", fake_gemm)

    output = tm.apply_prepared_fused_silu_and_mul(
        layer,
        torch.ones((2, 1, 3), dtype=torch.float16),
    )

    assert output is not None
    assert output.shape == (2, 1, 2)
    assert output.tolist() == [[[3.0, 3.0]], [[3.0, 3.0]]]
    assert calls == [
        {
            "out_shape": (2, 2),
            "x_shape": (2, 3),
            "group_size": 16,
            "k_ld": 3,
            "q_ld": 4,
            "gated_silu": True,
        }
    ]


def test_nvfp4_fused_silu_rejects_non_gated_state():
    tm = _load_adapter()
    layer = torch.nn.Module()
    setattr(layer, tm.STATE_ATTR, _make_nvfp4_state(tm, gated_silu=False))

    output = tm.apply_prepared_fused_silu_and_mul(
        layer,
        torch.ones((1, 3), dtype=torch.float16),
    )

    assert output is None


def test_nvfp4_qpn4_regular_apply_uses_opaque_dynamic_m_dispatch(monkeypatch):
    tm = _load_adapter()
    layer = torch.nn.Module()
    setattr(layer, tm.STATE_ATTR, _make_nvfp4_qpn4_state(tm, gated_silu=False))

    from vllm import _sm70_ops as sm70_ops

    calls = []

    def fake_dispatch(
        out,
        dense_weight_ptr,
        x,
        weight,
        scales,
        global_scale,
        use_scale_code,
        gated_silu,
    ):
        del weight, scales
        calls.append(
            (
                tuple(out.shape),
                dense_weight_ptr,
                tuple(x.shape),
                global_scale,
                use_scale_code,
                gated_silu,
            )
        )
        out.fill_(2.0)

    monkeypatch.setattr(sm70_ops, "nvfp4_qpn4_dispatch_sm70_out", fake_dispatch)
    output = tm.apply_prepared_linear(
        layer,
        torch.ones((2, 3), dtype=torch.float16),
        bias=None,
    )

    assert output.tolist() == [[2.0] * 4, [2.0] * 4]
    assert calls == [((2, 4), 1234, (2, 3), 0.0, False, False)]


def test_nvfp4_qpn4_fused_gate_uses_scale_code_dispatch(monkeypatch):
    tm = _load_adapter()
    layer = torch.nn.Module()
    setattr(layer, tm.STATE_ATTR, _make_nvfp4_qpn4_state(tm, gated_silu=True))

    from vllm import _sm70_ops as sm70_ops

    calls = []

    def fake_dispatch(
        out,
        dense_weight_ptr,
        x,
        weight,
        scales,
        global_scale,
        use_scale_code,
        gated_silu,
    ):
        del weight, scales
        calls.append(
            (
                tuple(out.shape),
                dense_weight_ptr,
                tuple(x.shape),
                global_scale,
                use_scale_code,
                gated_silu,
            )
        )
        out.fill_(4.0)

    monkeypatch.setattr(sm70_ops, "nvfp4_qpn4_dispatch_sm70_out", fake_dispatch)
    output = tm.apply_prepared_fused_silu_and_mul(
        layer,
        torch.ones((1, 3), dtype=torch.float16),
    )

    assert output is not None
    assert output.tolist() == [[4.0, 4.0]]
    assert calls == [((1, 2), 1234, (1, 3), 0.25, True, True)]
