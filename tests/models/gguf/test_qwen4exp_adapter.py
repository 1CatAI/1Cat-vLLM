# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader.gguf_adapters.qwen4exp import Qwen4ExpAdapter
from vllm.transformers_utils.gguf_config import gguf_config_from_metadata


def metadata():
    fixture = json.loads(
        Path(__file__).with_name("flashnext_metadata.json").read_text()
    )
    result = fixture["metadata"]
    # Keep the real header fixture small: configuration needs the vocabulary
    # length, not its 248,320 strings.
    result["tokenizer.ggml.tokens"] = range(fixture["vocab_size"])
    return result


def test_real_flashnext_metadata_keeps_integer_hash_constants():
    config = gguf_config_from_metadata(metadata())
    assert config.architectures == ["Qwen4ExpForCausalLM"]
    assert (config.hidden_size, config.head_dim, config.num_hidden_layers) == (
        2560,
        256,
        48,
    )
    assert (config.num_experts, config.num_experts_per_tok) == (512, 10)
    assert (config.linear_num_key_heads, config.linear_num_value_heads) == (16, 48)
    assert (config.hc_count, config.hc_lowrank) == (4, 320)
    assert config.ple_layer_ids == [2]
    assert config.ple_embed_dim == 2560
    assert config.indexer_compress_ratio == 4
    assert config.gguf_ple_constants["layer_multipliers"][0] == 23703573157769
    assert config.gguf_ple_eos_token_id == 248044
    assert config.layer_types.count("full_attention") == 12


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            {"qwen4exp.ple.layer_multipliers": [float(23703573157769), 2, 3]},
            "integer constants",
        ),
        ({"qwen4exp.ple.head_offsets": [1] * 16}, "contiguous"),
        ({"qwen4exp.attention.compress_ratios": [4]}, "length"),
        ({"qwen4exp.ple.layers": [48]}, "outside backbone"),
    ],
)
def test_reject_invalid_flashnext_metadata(mutation, error):
    m = metadata()
    m.update(mutation)
    with pytest.raises(ValueError, match=error):
        gguf_config_from_metadata(m)


def small_config():
    return SimpleNamespace(
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        ple_layer_ids=[2],
        num_experts=2,
        hidden_size=4,
        indexer_n_heads=2,
        indexer_head_dim=1,
        ngram_size=2,
        heads_per_ngram=2,
        ple_embed_dim=64,
        gguf_ple_constants={
            "layer_multipliers": [23703573157769, 20109073645365],
            "ngram_heads_offsets": [0, 50],
            "ngram_heads_vocab_sizes": [50, 60],
        },
    )


def test_hc_and_ple_mapping_and_inverse_norm_convolution():
    adapter = Qwen4ExpAdapter(small_config())
    mapping = adapter.build_name_map(
        {
            "blk.0.hc_attn_inject.weight": None,
            "blk.1.ple_norm_key.weight": None,
            "per_layer_token_embd.weight": None,
        }
    )
    assert mapping["blk.0.hc_attn_inject.weight"].endswith(
        "attn_hyper_connection.block_inject_weight.weight"
    )
    assert mapping["per_layer_token_embd.weight"].startswith("model.layers.1.ple.")
    assert not adapter.is_linear(mapping["blk.0.hc_attn_inject.weight"])
    assert not adapter.is_linear("model.layers.0.mlp.gate.weight")
    assert not adapter.is_linear("model.layers.0.mlp.shared_expert_gate.weight")
    assert adapter.is_linear("lm_head.weight")
    for norm in ("norm_key", "norm_query", "norm_conv"):
        assert not adapter.is_linear(f"model.layers.1.ple.{norm}.weight")
    assert adapter.restore(
        "model.layers.0.mlp.shared_expert_gate.weight", torch.ones(4)
    ).shape == (1, 4)
    assert torch.equal(
        adapter.restore(mapping["blk.1.ple_norm_key.weight"], torch.ones(24)),
        torch.zeros(24),
    )
    conv = torch.arange(24 * 4).reshape(24, 4)
    # PLE is HC-wide convolution, and must not receive GDN value-head reorder.
    assert torch.equal(
        adapter.restore("model.layers.1.ple.conv1d.weight", conv), conv.unsqueeze(1)
    )


def test_indexer_split_projections_restore_dense_concatenation():
    adapter = Qwen4ExpAdapter(small_config())
    tensors = {
        "blk.1.indexer.q_proj.weight": SimpleNamespace(
            tensor_type=gguf.GGMLQuantizationType.F16,
            data=np.arange(8, dtype=np.float16).reshape(2, 4),
        ),
        "blk.1.indexer.k_proj.weight": SimpleNamespace(
            tensor_type=gguf.GGMLQuantizationType.F32,
            data=np.arange(4, dtype=np.float32).reshape(1, 4),
        ),
    }
    weights = dict(
        adapter.weights(tensors, adapter.build_name_map(tensors), torch.float16)
    )
    prefix = "model.layers.1.self_attn.indexer.index_qk_proj"
    assert weights[prefix + ".qweight_type"].item() == 1
    expected = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3]], dtype=torch.float16
    )
    assert torch.equal(weights[prefix + ".qweight"], expected)


def test_ple_table_remains_packed_mmap_and_hash_constants_int64(tmp_path):
    path = tmp_path / "ple.bin"
    data = np.memmap(path, mode="w+", dtype=np.uint8, shape=(128, 18))
    data[:] = 0
    tensors = {
        "per_layer_token_embd.weight": SimpleNamespace(
            tensor_type=gguf.GGMLQuantizationType.IQ4_NL,
            data=data,
            shape=[32, 128],
        )
    }
    adapter = Qwen4ExpAdapter(small_config())
    weights = dict(
        adapter.weights(tensors, adapter.build_name_map(tensors), torch.float16)
    )
    prefix = "model.layers.1.ple.ple_embedding"
    packed = weights[prefix + ".ngram_embedding.qweight"]
    assert packed.dtype == torch.uint8 and packed.shape == (128, 18)
    assert packed.data_ptr() == data.ctypes.data
    constants = weights[prefix + ".layer_multipliers"]
    assert constants.dtype == torch.int64
    assert constants[0].item() == 23703573157769


def test_stacked_experts_keep_independent_projection_type():
    adapter = Qwen4ExpAdapter(small_config())
    tensors = {
        "blk.0.ffn_gate_exps.weight": SimpleNamespace(
            tensor_type=gguf.GGMLQuantizationType.Q4_0,
            shape=[32, 4, 2],
            data=np.zeros((2, 4, 18), dtype=np.uint8),
        ),
        "blk.0.ffn_up_exps.weight": SimpleNamespace(
            tensor_type=gguf.GGMLQuantizationType.Q8_0,
            shape=[32, 4, 2],
            data=np.zeros((2, 4, 34), dtype=np.uint8),
        ),
    }
    weights = dict(
        adapter.weights(tensors, adapter.build_name_map(tensors), torch.float16)
    )
    prefix = "model.layers.0.mlp.experts"
    for expert in range(2):
        assert weights[f"{prefix}.{expert}.gate_proj.qweight_type"].item() == 2
        assert weights[f"{prefix}.{expert}.up_proj.qweight_type"].item() == 8
        assert weights[f"{prefix}.{expert}.gate_proj.qweight"].shape == (4, 18)
        assert weights[f"{prefix}.{expert}.up_proj.qweight"].shape == (4, 34)


def test_quantized_vocabulary_head_keeps_packed_payload_and_type():
    adapter = Qwen4ExpAdapter(small_config())
    raw = np.arange(3 * 210, dtype=np.uint16).astype(np.uint8).reshape(3, 210)
    tensor = SimpleNamespace(tensor_type=gguf.GGMLQuantizationType.Q6_K, data=raw)
    tensors = {"output.weight": tensor}
    weights = dict(
        adapter.weights(tensors, adapter.build_name_map(tensors), torch.float16)
    )
    assert set(weights) == {"lm_head.qweight_type", "lm_head.qweight"}
    assert weights["lm_head.qweight_type"].item() == int(gguf.GGMLQuantizationType.Q6_K)
    assert weights["lm_head.qweight"].dtype == torch.uint8
    torch.testing.assert_close(weights["lm_head.qweight"], torch.from_numpy(raw))
