# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader.gguf_adapters import get_gguf_adapter
from vllm.model_executor.model_loader.gguf_adapters.qwen35_moe import Qwen35MoeAdapter


def config():
    return SimpleNamespace(
        model_type="qwen3_5_moe_text",
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        num_experts=3,
        linear_num_key_heads=1,
        linear_num_value_heads=3,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )


def test_registry_router_shared_expert_and_nextn_mapping():
    cfg = config()
    adapter = get_gguf_adapter(SimpleNamespace(get_text_config=lambda: cfg), tp_size=4)
    assert isinstance(adapter, Qwen35MoeAdapter)
    assert adapter.tp_size == 4 and adapter.native_expert_storage
    names = {
        "blk.0.ffn_gate_inp.weight": "mlp.gate.weight",
        "blk.0.ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
        "blk.0.ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
        "blk.0.ffn_down_exps.weight": "mlp.experts.down_proj.weight",
        "blk.0.ssm_a": "linear_attn.A_log",
    }
    mapping = adapter.build_name_map({**names, "blk.1.nextn.eh_proj.weight": None})
    assert mapping == {raw: "model.layers.0." + name for raw, name in names.items()}
    assert not adapter.is_linear(mapping["blk.0.ffn_gate_inp.weight"])
    assert not adapter.is_linear(mapping["blk.0.ffn_gate_inp_shexp.weight"])
    assert adapter.is_linear(mapping["blk.0.ffn_gate_shexp.weight"])
    assert adapter.restore(
        mapping["blk.0.ffn_gate_inp_shexp.weight"], torch.ones(64)
    ).shape == (1, 64)


def test_mixed_stacked_experts_keep_source_bytes_and_descriptors():
    rng = np.random.default_rng(41)
    tensors = {}
    kinds = {
        "gate": gguf.GGMLQuantizationType.Q4_0,
        "up": gguf.GGMLQuantizationType.Q8_0,
        "down": gguf.GGMLQuantizationType.F32,
    }
    for projection, kind in kinds.items():
        values = rng.normal(size=(3, 32, 64)).astype(np.float32)
        data = (
            values
            if kind == gguf.GGMLQuantizationType.F32
            else gguf.quants.quantize(values, kind)
        )
        tensors[f"blk.0.ffn_{projection}_exps.weight"] = SimpleNamespace(
            tensor_type=kind, shape=[64, 32, 3], data=data
        )
    adapter = Qwen35MoeAdapter(config(), tp_size=4)
    mapping = adapter.build_name_map(tensors)
    output = dict(adapter.weights(tensors, mapping, torch.float16))
    assert len(output) == 18
    for projection, kind in kinds.items():
        tensor = tensors[f"blk.0.ffn_{projection}_exps.weight"]
        expected = torch.from_numpy(gguf.quants.dequantize(tensor.data, kind))
        for expert in range(3):
            name = f"model.layers.0.mlp.experts.{expert}.{projection}_proj"
            storage_type = int(
                gguf.GGMLQuantizationType.F16
                if kind == gguf.GGMLQuantizationType.F32
                else kind
            )
            assert output[name + ".qweight_type"].item() == storage_type
            weight = output[name + ".qweight"]
            actual = (
                weight
                if kind == gguf.GGMLQuantizationType.F32
                else torch.from_numpy(gguf.quants.dequantize(weight.numpy(), kind))
            )
            torch.testing.assert_close(
                actual.half(), expected[expert].half(), rtol=0, atol=0
            )
            if kind != gguf.GGMLQuantizationType.F32:
                assert np.shares_memory(weight.numpy(), tensor.data)
                assert not adapter.needs_dense_fallback(
                    mapping[f"blk.0.ffn_{projection}_exps.weight"], tensor
                )
    assert adapter.fallback_reasons == {}


def test_reject_bad_expert_count_and_fp16_overflow():
    adapter = Qwen35MoeAdapter(config())
    raw = "blk.0.ffn_gate_exps.weight"
    tensor = SimpleNamespace(
        tensor_type=gguf.GGMLQuantizationType.F32,
        shape=[64, 32, 4],
        data=np.zeros((4, 32, 64), dtype=np.float32),
    )
    mapping = adapter.build_name_map({raw: tensor})
    with pytest.raises(ValueError, match="Invalid stacked GGUF expert shape"):
        list(adapter.weights({raw: tensor}, mapping, torch.float16))
    tensor.shape = [64, 32, 3]
    tensor.data = np.full((3, 32, 64), 1e8, dtype=np.float32)
    with pytest.raises(ValueError, match="overflows target dtype"):
        list(adapter.weights({raw: tensor}, mapping, torch.float16))
