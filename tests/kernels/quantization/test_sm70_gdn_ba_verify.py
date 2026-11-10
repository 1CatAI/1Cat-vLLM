# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


def test_wider_gdn_shards_require_the_native_geometry_contract(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import sm70_gdn_ba_verify as module

    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())
    assert module._gdn_ba_shard_supported(4096, 12)
    assert not module._gdn_ba_shard_supported(8192, 24)
    torch.ops._C.fp8_qpn8_ba_split_geometry_revision = lambda: 1
    assert module._gdn_ba_shard_supported(8192, 24)
    assert not module._gdn_ba_shard_supported(8192, 12)


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 9, 16, 17, 32])
def test_tp2_fused_projection_covers_both_ba_column_tiles(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    from vllm import _sm70_ops as ops

    if not hasattr(torch.ops._C, "fp8_qpn8_ba_split_geometry_revision"):
        pytest.skip("requires the dynamic projection geometry")
    torch.manual_seed(9851)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16) * 0.125
    weight = (torch.randn(8192, 5120, device="cuda") * 0.01).to(torch.float8_e4m3fn)
    scales = torch.ones(8192, 1, device="cuda")
    codes, packed_scales = ops.fp8_qpn8_prepare_sm70(weight, scales)
    ba = (torch.randn(48, 5120, device="cuda") * 0.01).to(torch.bfloat16).half()
    q, z = x.new_empty(rows, 5120), x.new_empty(rows, 3072)
    b, a = x.new_empty(rows, 24), x.new_empty(rows, 24)
    operation = (
        torch.ops._C.fp8_qpn8_dispatch_ba_masked_sm70_out
        if rows < 8
        else torch.ops._C.fp8_qpn8_dispatch_ba_split_sm70_out
    )
    staging, ba_staging = x.new_empty(rows, 8192), x.new_empty(rows, 48)

    def run():
        operation(q, z, b, a, staging, ba_staging, 0, x, codes, packed_scales, ba)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for amplitude in (0.0, 0.125, 2.0):
        x.normal_().mul_(amplitude)
        graph.replay()
        ordinary = x.new_empty(rows, 8192)
        ops.fp8_qpn8_gemm_sm70_out(
            ordinary, x, codes, packed_scales, 16, 2, True, False
        )
        assert torch.equal(q, ordinary[:, :5120])
        assert torch.equal(z, ordinary[:, 5120:])
        reference = x.double() @ ba.double().T
        # FP32 split accumulation followed by one FP16 store: two half ULPs
        # plus an absolute term around zero cover the unchanged reduction.
        torch.testing.assert_close(
            torch.cat((b, a), 1).double(), reference, rtol=0.002, atol=3e-5
        )


@pytest.mark.parametrize("boundary", ["core", "projection"])
def test_verify_fusion_reaches_opaque_input_boundaries(monkeypatch, boundary):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.model_executor.layers.quantization import sm70_gdn_ba_verify as ba

    layer = SimpleNamespace(
        num_v_heads=48, tp_size=4, key_dim=2048, value_dim=6144, head_v_dim=128
    )
    monkeypatch.setattr(gdn, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"test.layer": layer}),
    )
    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda name: False)
    monkeypatch.setattr(
        gdn, "_sm70_dump_gdn_projection_tensor", lambda label, name, value: value
    )
    calls = []

    def fused(actual_layer, x, outputs=None):
        assert actual_layer is layer
        assert x.shape == (8, 5120)
        calls.append(outputs)
        for index, value in enumerate(outputs):
            value.fill_(index + 1)
        return outputs

    monkeypatch.setattr(ba, "apply_gdn_ba_verify", fused)
    x = torch.zeros(8, 5120, dtype=torch.float16)
    z = x.new_empty((8, 12, 128))
    if boundary == "core":
        core = torch.zeros_like(z)

        def recurrent(actual_layer, **kwargs):
            assert actual_layer is layer
            assert torch.all(kwargs["mixed_qkv"] == 1)
            assert torch.all(kwargs["b"] == 3)
            assert torch.all(kwargs["a"] == 4)

        monkeypatch.setattr(gdn, "_qwen_gdn_run_recurrent_core", recurrent)
        result = gdn.qwen_gdn_input_projection_core(
            x, z, core, torch.empty(0), torch.empty(0), "test.layer"
        )
        assert result[0] is z and result[1] is core
    else:
        q, b, a = x.new_empty((8, 2560)), x.new_empty((8, 12)), x.new_empty((8, 12))
        gdn.qwen_gdn_input_projection(x, q, z, b, a, "test.layer")
        assert torch.all(q == 1) and torch.all(b == 3) and torch.all(a == 4)
    assert len(calls) == 1
    assert calls[0][1].data_ptr() == z.data_ptr()
    assert torch.all(z == 2)


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("boundary", ["core", "projection"])
def test_strided_output_keeps_the_original_storage(monkeypatch, tp_size, boundary):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.model_executor.layers.quantization import sm70_gdn_ba_verify as ba
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as split

    rows, heads = 5, 48 // tp_size
    qkv_width = 10240 // tp_size
    x = torch.zeros(rows, 5120, dtype=torch.float16)
    qkv = x.new_full((rows, qkv_width), 1)
    expected_z = x.new_full((rows, heads, 128), 2)
    b, a = x.new_full((rows, heads), 3), x.new_full((rows, heads), 4)
    z = x.new_full((rows, 128, heads), -1).transpose(1, 2)
    assert not z.is_contiguous()
    layer = SimpleNamespace(
        num_v_heads=48,
        tp_size=tp_size,
        key_dim=2048,
        value_dim=6144,
        head_k_dim=128,
        head_v_dim=128,
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
        gdn_policy=SimpleNamespace(
            projection=SimpleNamespace(mixed_qkv_contiguous=False, z_contiguous=False)
        ),
        in_proj_qkvz=lambda value: (
            torch.cat((qkv, expected_z.flatten(1)), dim=1),
            None,
        ),
        in_proj_ba=lambda value: (torch.cat((b, a), dim=1), None),
    )
    monkeypatch.setattr(gdn, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"test.layer": layer}),
    )
    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda name: False)
    monkeypatch.setattr(
        gdn, "_sm70_dump_gdn_projection_tensor", lambda label, name, value: value
    )
    monkeypatch.setattr(gdn, "_sm70_gdn_qpn8_ba_split_eligible", lambda *args: False)
    monkeypatch.setattr(split, "_can_fuse_gdn_projection_split", lambda *args: False)

    def fused(actual_layer, hidden, outputs=None):
        # A copied flatten would accept these writes without updating z.
        for index, value in enumerate(outputs):
            value.fill_(index + 1)
        return outputs

    monkeypatch.setattr(ba, "apply_gdn_ba_verify", fused)

    def recurrent(actual_layer, **kwargs):
        assert actual_layer is layer
        torch.testing.assert_close(kwargs["mixed_qkv"], qkv)
        torch.testing.assert_close(kwargs["b"], b)
        torch.testing.assert_close(kwargs["a"], a)

    monkeypatch.setattr(gdn, "_qwen_gdn_run_recurrent_core", recurrent)
    original_pointer = z.data_ptr()
    if boundary == "core":
        core = torch.zeros_like(expected_z)
        result = gdn.qwen_gdn_input_projection_core(
            x, z, core, torch.empty(0), torch.empty(0), "test.layer"
        )
        assert result[0] is z and result[1] is core
    else:
        actual_q, actual_b, actual_a = (
            torch.empty_like(qkv),
            torch.empty_like(b),
            torch.empty_like(a),
        )
        gdn.qwen_gdn_input_projection(x, actual_q, z, actual_b, actual_a, "test.layer")
        torch.testing.assert_close(actual_q, qkv)
        torch.testing.assert_close(actual_b, b)
        torch.testing.assert_close(actual_a, a)
    assert z.data_ptr() == original_pointer
    torch.testing.assert_close(z, expected_z)


def test_verify_fusion_reaches_qwen35_override(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.model_executor.layers.quantization import sm70_gdn_ba_verify as ba
    from vllm.model_executor.models import qwen3_5

    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda name: False)
    monkeypatch.setattr(qwen3_5, "_encode_layer_name", lambda name: name)
    monkeypatch.setattr(
        qwen3_5, "_sm70_dump_gdn_projection_tensor", lambda label, name, value: value
    )
    monkeypatch.setattr(
        qwen3_5, "_resolve_qwen_gdn_kv_cache_args", lambda *args: (None, None)
    )
    x = torch.zeros(8, 5120, dtype=torch.float16)
    q, z = x.new_full((8, 2560), 1), x.new_full((8, 1536), 2)
    b, a = x.new_full((8, 12), 3), x.new_full((8, 12), 4)
    layer = SimpleNamespace(
        key_dim=2048,
        value_dim=6144,
        tp_size=4,
        num_v_heads=48,
        head_v_dim=128,
        prefix="test.layer",
        # Keep the fused projection's QKV view: the recurrent stub checks its identity.
        gdn_policy=SimpleNamespace(
            projection=SimpleNamespace(mixed_qkv_contiguous=False)
        ),
    )
    calls = []

    def fused(actual_layer, hidden):
        assert actual_layer is layer and hidden is x
        calls.append(True)
        return q, z, b, a

    def recurrent(actual_layer, **kwargs):
        assert actual_layer is layer
        assert kwargs["mixed_qkv"] is q
        assert kwargs["b"] is b and kwargs["a"] is a
        return kwargs["core_attn_out"]

    def project(core, actual_z, output, rows):
        assert actual_z.data_ptr() == z.data_ptr()
        assert actual_z.shape == (8, 12, 128) and rows == 8
        return core

    layer._output_projection = project
    layer._allocate_core_attn_out = lambda mixed_qkv, a, b, hidden: x.new_zeros(
        8, 12, 128
    )
    monkeypatch.setattr(ba, "apply_gdn_ba_verify", fused)
    monkeypatch.setattr(qwen3_5, "_qwen_gdn_run_recurrent_core", recurrent)
    result = qwen3_5.Qwen3_5GatedDeltaNet.forward_cuda(layer, x, None)
    assert result.shape == (8, 12, 128) and calls == [True]


@pytest.mark.parametrize("amplitude", [0.0, 0.125, 1.0])
@torch.inference_mode()
def test_m8_qkvz_is_bitwise_and_ba_matches_dense64(amplitude):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    from vllm import _sm70_ops as ops

    torch.manual_seed(891)
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * amplitude
    weight = (torch.randn(4096, 5120, device="cuda") * 0.01).to(torch.float8_e4m3fn)
    scales = torch.ones(4096, 1, device="cuda")
    codes, packed_scales = ops.fp8_qpn8_prepare_sm70(weight, scales)
    ba = (torch.randn(24, 5120, device="cuda") * 0.01).to(torch.bfloat16).half()
    ordinary = x.new_empty((8, 4096))
    q, z = x.new_empty((8, 2560)), x.new_empty((8, 1536))
    b, a = x.new_empty((8, 12)), x.new_empty((8, 12))
    ops.fp8_qpn8_gemm_sm70_out(ordinary, x, codes, packed_scales, 16, 2, True, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.fp8_qpn8_dispatch_ba_split_sm70_out(
            q,
            z,
            b,
            a,
            x.new_empty((8, 4096)),
            x.new_empty((8, 24)),
            0,
            x,
            codes,
            packed_scales,
            ba,
        )
    for _ in range(4):
        graph.replay()
    assert torch.equal(q, ordinary[:, :2560])
    assert torch.equal(z, ordinary[:, 2560:])
    reference = x.double() @ ba.double().T
    torch.testing.assert_close(
        torch.cat((b, a), dim=1).double(), reference, atol=0.002, rtol=0.001
    )
