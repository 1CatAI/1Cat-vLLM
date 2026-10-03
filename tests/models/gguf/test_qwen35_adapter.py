# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.model_executor.model_loader.gguf_adapters.qwen35 import Qwen35Adapter


def config():
    return SimpleNamespace(
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )


def ggml_tile(weight, dim, groups=8, repeat=2, head_dim=128):
    # llama.cpp conversion/qwen.py: _LinearAttentionVReorderBase._reorder_v_heads.
    shape = list(weight.shape)
    tiled = weight.reshape(*shape[:dim], groups, repeat, head_dim, *shape[dim + 1 :])
    return tiled.transpose(dim, dim + 1).reshape(shape)


def test_restore_gdn_rows_norms_and_convolution():
    adapter = Qwen35Adapter(config())
    qk = torch.randn(2048, 4)
    value = torch.arange(2048 * 4).reshape(2048, 4).float()
    packed = torch.cat([qk, ggml_tile(value, 0)])
    torch.testing.assert_close(
        adapter.restore("linear_attn.in_proj_qkv.weight", packed),
        torch.cat([qk, value]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        adapter.restore("linear_attn.conv1d.weight", packed),
        torch.cat([qk, value]).unsqueeze(1),
        rtol=0,
        atol=0,
    )
    a = torch.linspace(-3, 2, 16)
    tiled = ggml_tile(-a.exp(), 0, head_dim=1)
    torch.testing.assert_close(adapter.restore("linear_attn.A_log", tiled), a)
    norm = torch.ones(128) * 1.5
    assert torch.equal(adapter.restore("input_layernorm.weight", norm), norm - 1)
    assert torch.equal(adapter.restore("linear_attn.norm.weight", norm), norm)


def test_tp4_tiled_out_projection_matches_unsharded():
    layout = GGUFHeadTilingLayout(2, 128)
    x, weight = torch.randn(7, 2048), torch.randn(32, 2048)
    stored = ggml_tile(weight, 1)
    terms = []
    for rank in range(4):
        w = layout.shard_weight(
            stored, dim=1, logical_size=2048, block_size=256, tp_rank=rank, tp_size=4
        )
        local_input = layout.input_to_gguf(x[:, rank * 512 : (rank + 1) * 512])
        terms.append(local_input @ w.T)
    torch.testing.assert_close(sum(terms), x @ weight.T, rtol=1e-4, atol=1e-4)


def test_reject_unsupported_tensor_instead_of_silently_skipping():
    adapter = Qwen35Adapter(config())
    with pytest.raises(ValueError, match="Unmapped Qwen3.5 GGUF tensor"):
        adapter.build_name_map({"blk.0.hc_attn_down.weight": None})
    mapping = adapter.build_name_map(
        {"blk.0.ssm_a": None, "blk.1.nextn.eh_proj.weight": None}
    )
    assert mapping == {"blk.0.ssm_a": "model.layers.0.linear_attn.A_log"}


def test_misaligned_tiled_shard_is_rejected():
    layout = GGUFHeadTilingLayout(2, 128)
    with pytest.raises(ValueError, match="not aligned"):
        layout.shard_weight(
            torch.zeros(4, 1024),
            dim=1,
            logical_size=1024,
            block_size=256,
            tp_rank=0,
            tp_size=4,
        )


def test_unaligned_k_quant_row_is_converted_before_tp():
    import gguf
    import numpy as np

    # 0.8B FFN K=3584 gives local K=896 under TP4. Give each superblock a
    # nonzero scale and codes so incorrect byte slicing cannot pass by zero.
    blocks = np.zeros((4 * 14, 144), dtype=np.uint8)
    blocks[:, :2] = np.frombuffer(np.float16(0.125).tobytes(), dtype=np.uint8)
    blocks[:, 4:16] = 1
    blocks[:, 16:] = 0x21
    tensor = SimpleNamespace(
        tensor_type=gguf.GGMLQuantizationType.Q4_K,
        shape=[3584, 4],
        data=blocks.reshape(4, -1),
    )
    adapter = Qwen35Adapter(config(), tp_size=4)
    name = "model.layers.0.mlp.down_proj.weight"
    weights = list(adapter.weights({"down": tensor}, {"down": name}, torch.float16))
    assert weights[0][0].endswith(".qweight_type")
    assert weights[0][1].item() == int(gguf.GGMLQuantizationType.F16)
    assert weights[1][1].shape == (4, 3584)
    expected = torch.from_numpy(gguf.quants.dequantize(tensor.data, tensor.tensor_type))
    torch.testing.assert_close(weights[1][1], expected.half(), rtol=0, atol=0)
    assert "local_K=896" in adapter.fallback_reasons[name.removesuffix(".weight")]


@pytest.mark.parametrize("rank", range(4))
def test_gguf_kv_heads_are_replicated_when_tp_exceeds_head_count(rank):
    from vllm.model_executor.layers.linear import QKVParallelLinear

    # Eight Q heads and two KV heads, distributed over four ranks.
    layer = object.__new__(QKVParallelLinear)
    layer.tp_rank, layer.tp_size = rank, 4
    layer.num_heads, layer.num_kv_heads = 2, 1
    layer.head_size = layer.v_head_size = 4
    layer.num_kv_head_replicas = 2
    parameter = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    parameter.is_gguf_weight = True
    parameter.output_dim = 0
    parameter.shard_id, parameter.shard_id_map = [], {}
    parameter.data_container = []
    for shard, rows in (("q", 32), ("k", 8), ("v", 8)):
        weight = torch.arange(rows * 3).reshape(rows, 3)
        layer.weight_loader(parameter, weight, shard)
        local_rows = 8 if shard == "q" else 4
        source_rank = rank if shard == "q" else rank // 2
        expected = weight[source_rank * local_rows : (source_rank + 1) * local_rows]
        assert torch.equal(parameter.data_container[-1], expected)
    assert sum(weight.shape[0] for weight in parameter.data_container) == 16


def test_mixed_float_and_quantized_projections_keep_logical_order(monkeypatch):
    from vllm.model_executor.layers.quantization import gguf as quant

    layer = torch.nn.Module()
    method = quant.GGUFLinearMethod(quant.GGUFConfig())
    method.create_weights(layer, 4, [2, 3], 4, 5, torch.float16)
    # File order is reversed. A homogeneous buffer cannot hold these dtypes.
    up = torch.full((3, 4), 7, dtype=torch.uint8)
    gate = torch.arange(8, dtype=torch.float16).reshape(2, 4)
    layer.qweight.data_container.extend([up, gate])
    layer.qweight.shard_id.extend([1, 0])
    layer.qweight.shard_id_map.update({1: 0, 0: 1})
    layer.qweight_type.shard_weight_type.update({1: 2, 0: 1})
    method.process_weights_after_loading(layer)
    assert layer.gguf_shard_weights[0].dtype == torch.float16
    assert layer.gguf_shard_weights[1].dtype == torch.uint8
    calls = []

    def dispatch(x, weight, weight_type, enabled, prefill_min_m):
        assert enabled == method.native_enabled
        assert prefill_min_m == method.prefill_min_m
        calls.append(weight_type)
        return x @ weight.to(x.dtype).T

    monkeypatch.setattr(quant, "fused_mul_mat_gguf", dispatch)
    x = torch.arange(4, dtype=torch.float16).reshape(1, 4)
    expected = torch.cat([x @ gate.T, x @ up.half().T], dim=-1)
    assert torch.equal(method.apply(layer, x), expected)
    assert calls == [1, 2]


def test_inverse_a_log_keeps_the_model_fp32_parameter_contract():
    import gguf
    import numpy as np

    tensor = SimpleNamespace(
        tensor_type=gguf.GGMLQuantizationType.F32,
        data=-np.exp(np.linspace(-3, 2, 16, dtype=np.float32)),
    )
    adapter = Qwen35Adapter(config())
    name = "model.layers.0.linear_attn.A_log"
    weights = dict(adapter.weights({"a": tensor}, {"a": name}, torch.float16))
    expected = adapter.restore(name, torch.from_numpy(tensor.data))
    assert weights[name].dtype == torch.float32
    torch.testing.assert_close(weights[name], expected, rtol=0, atol=0)


def test_text_model_supplies_three_identical_mrope_position_axes():
    from vllm.model_executor.models.interfaces import supports_mrope
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM

    model = object.__new__(Qwen3_5ForCausalLM)
    torch.nn.Module.__init__(model)
    assert supports_mrope(model)
    positions, delta = model.get_mrope_input_positions([1, 2, 3, 4], [])
    assert positions.shape == (3, 4)
    assert positions.device.type == "cpu" and delta == 0
    assert torch.equal(positions, torch.arange(4).expand(3, -1))


def test_bf16_overflow_is_detected_even_with_an_existing_nan():
    import gguf

    source = torch.tensor([1e10, float("nan")], dtype=torch.bfloat16)
    tensor = SimpleNamespace(
        tensor_type=gguf.GGMLQuantizationType.BF16,
        data=source.view(torch.uint16).numpy(),
    )
    adapter = Qwen35Adapter(config())
    name = "model.layers.0.input_layernorm.weight"
    with pytest.raises(ValueError, match="overflow"):
        list(adapter.weights({"norm": tensor}, {"norm": name}, torch.float16))
