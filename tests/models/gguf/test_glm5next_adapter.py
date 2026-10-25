# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM tensor directory coverage and exporter inversions without a GPU."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader.gguf_adapters import get_gguf_adapter
from vllm.model_executor.model_loader.gguf_adapters.common import decode_dense_tensor
from vllm.model_executor.model_loader.gguf_adapters.glm5next import (
    Glm5NextAdapter,
    Glm5NextMTPAdapter,
)
from vllm.transformers_utils.gguf_config import (
    gguf_config_dict,
    gguf_config_from_metadata,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def inventory():
    return json.loads(Path(__file__).with_name("glm5next_inventory.json").read_text())


@pytest.fixture
def config(inventory):
    return gguf_config_from_metadata(inventory["metadata"])


def directory(inventory):
    return {
        row["name"]: SimpleNamespace(
            shape=row["shape"],
            tensor_type=gguf.GGMLQuantizationType[row["tensor_type"]],
        )
        for row in inventory["tensors"]
    }


def test_native_mtp_config_and_real_inventory(config, inventory):
    from vllm.config.speculative import SpeculativeConfig

    draft = SpeculativeConfig.hf_config_override(copy.deepcopy(config))
    assert draft.architectures == ["Glm5NextMTPModel"]
    assert draft.hc_mult == 1 and draft.mhc_num_residual_streams == 4
    assert draft.is_mtp_draft and draft.n_predict == 1
    assert SpeculativeConfig.hf_config_override(draft) is draft
    adapter = get_gguf_adapter(draft)
    assert isinstance(adapter, Glm5NextMTPAdapter)
    mapping = adapter.build_name_map(directory(inventory))
    assert len(mapping) == 31
    assert len(adapter.excluded_tensor_names) == 1381
    assert mapping["token_embd.weight"] == "model.embed_tokens.weight"
    assert mapping["output.weight"] == "model.layers.45.shared_head.head.weight"
    assert "output_norm.weight" not in mapping
    assert mapping["blk.45.nextn.eh_proj.weight"] == "model.layers.45.eh_proj.weight"
    assert mapping["blk.45.ffn_gate_exps.weight"] == (
        "model.layers.45.mtp_block.mlp.experts.gate_proj.weight"
    )
    assert mapping["blk.45.attn_k_b.weight"] == mapping["blk.45.attn_v_b.weight"]
    assert adapter.is_linear(mapping["output.weight"])
    assert not adapter.has_own_embed_tokens and not adapter.has_own_lm_head


def test_mtp_optional_owned_weights_replace_shared_fallback(config, inventory):
    tensors = directory(inventory)
    for raw in ("embed_tokens", "shared_head_head"):
        tensors[f"blk.45.nextn.{raw}.weight"] = SimpleNamespace(shape=[4096, 154880])
    adapter = Glm5NextMTPAdapter(config)
    mapping = adapter.build_name_map(tensors)
    assert "token_embd.weight" not in mapping and "output.weight" not in mapping
    assert adapter.has_own_embed_tokens and adapter.has_own_lm_head
    head = object()
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers={"45": SimpleNamespace(shared_head=SimpleNamespace(head=head))}
        )
    )
    adapter.configure_model(model)
    assert model.has_own_embed_tokens and model.has_own_lm_head
    assert model.lm_head is head


@pytest.mark.parametrize("fault", ["target_shape", "draft_missing", "norm_missing"])
def test_mtp_fails_closed_on_incomplete_or_corrupt_checkpoint(config, inventory, fault):
    tensors = directory(inventory)
    if fault == "target_shape":
        tensors["blk.0.attn_q.weight"].shape = [1, 1]
    else:
        raw = "nextn.shared_head_norm" if fault == "norm_missing" else "attn_q_a"
        del tensors[f"blk.45.{raw}.weight"]
    with pytest.raises(ValueError, match="expected shape|missing|requires"):
        Glm5NextMTPAdapter(config).build_name_map(tensors)


def tensor(values, kind=gguf.GGMLQuantizationType.F32):
    values = np.asarray(values, dtype=np.float32)
    data = gguf.quants.quantize(values, kind)
    return SimpleNamespace(shape=list(values.shape[::-1]), tensor_type=kind, data=data)


def test_real_inventory_config(config):
    assert config.model_type == "glm5_next_text"
    assert config.architectures == ["Glm5NextForCausalLM"]
    assert config.num_hidden_layers == 45
    assert config.num_nextn_predict_layers == 1
    assert config.vocab_size == 154880
    assert config.hidden_size == 4096
    assert config.max_position_embeddings == 1048576
    assert config.num_attention_heads == config.num_key_value_heads == 64
    assert config.kv_lora_rank == 512
    assert config.qk_nope_head_dim == config.v_head_dim == 256
    assert config.mla_nope and config.qk_rope_head_dim == 0
    assert [i for i in range(45) if not config.is_kda_layer(i)] == list(range(3, 45, 4))
    assert config.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 42
    assert config.n_routed_experts == 288
    assert config.num_experts_per_token == 8
    assert config.routed_scaling_factor == 2.5
    assert config.scoring_func == "sigmoid"
    assert config.moe_router_dtype == "float32"
    assert config.topk_method == "noaux_tc"
    assert config.swiglu_limit == 10
    assert config.index_kpool == 4 and config.index_topk == 2048
    assert config.mhc_num_residual_streams == 4
    assert config.eos_token_id == [154820, 154827, 154829]
    assert not hasattr(config, "quantization_config")


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("attention.head_count_kv", [1] * 45, "KV-head layout"),
        ("attention.head_count_kv", [True] * 46, "KV-head layout"),
        ("rope.dimension_count", 64, "NoPE"),
        ("attention.key_length", 256, "latent cache"),
        ("expert_gating_func", 1, "sigmoid"),
        ("expert_group_count", 5, "routing dimensions"),
        ("attention.indexer.kpool", 3, "divisible"),
        ("attention.indexer.types", 17, "full indexer"),
        ("attention.layer_norm_epsilon", 1e-5, "norm eps"),
        ("swiglu_clamp_exp", [10] * 45 + [7], "one shared/expert"),
    ],
)
def test_unsupported_metadata_fails(inventory, key, value, error):
    metadata = copy.deepcopy(inventory["metadata"])
    metadata["glm5-next." + key] = value
    with pytest.raises(ValueError, match=error):
        gguf_config_dict(metadata)


def test_unknown_architecture_and_missing_metadata(inventory):
    metadata = dict(inventory["metadata"])
    metadata["general.architecture"] = "future-unknown-model"
    with pytest.raises(ValueError, match="No native GGUF config adapter"):
        gguf_config_dict(metadata)
    metadata = dict(inventory["metadata"])
    del metadata["glm5-next.attention.head_count_kv"]
    with pytest.raises(ValueError, match="Missing required GGUF metadata"):
        gguf_config_dict(metadata)


def test_real_get_config_dispatches_declared_builder(tmp_path, inventory, monkeypatch):
    from transformers import PretrainedConfig

    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.gguf_config import supports_native_gguf_config

    assert supports_native_gguf_config("qwen35")
    assert supports_native_gguf_config("glm5-next")
    assert not supports_native_gguf_config("unknown")
    assert not supports_native_gguf_config(None)
    path = tmp_path / "glm.gguf"
    writer = gguf.GGUFWriter(path, "glm5-next")
    for key, value in inventory["metadata"].items():
        if key == "general.architecture":
            continue
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        else:
            writer.add_string(key, value)
    writer.add_tensor("sentinel.weight", np.zeros((1,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    def forbidden(*args, **kwargs):
        pytest.fail("registered GGUF builder fell through to HF parser")

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", forbidden)
    config = get_config(
        path, trust_remote_code=False, hf_overrides_kw={"max_position_embeddings": 8192}
    )
    assert config.model_type == "glm5_next_text"
    assert config.num_hidden_layers == 45
    assert config.max_position_embeddings == 8192
    assert config.eos_token_id == [154820, 154827, 154829]


def test_registry_inventory_and_explicit_nextn_exclusion(config, inventory):
    adapter = get_gguf_adapter(config)
    assert isinstance(adapter, Glm5NextAdapter)
    assert adapter.native_expert_storage and not adapter.packed_embeddings
    tensors = directory(inventory)
    mapping = adapter.build_name_map(tensors)
    assert len(tensors) == 1412
    assert len(mapping) == 1383
    assert len(adapter.excluded_tensor_names) == 29
    assert set(mapping) | set(adapter.excluded_tensor_names) == tensors.keys()
    assert all(raw.startswith("blk.45.") for raw in adapter.excluded_tensor_names)
    assert mapping["blk.3.attn_k_b.weight"] == mapping["blk.3.attn_v_b.weight"]
    assert mapping["blk.0.hc_attn_fn.weight"] == "model.layers.0.hc_attn_fn"
    assert not adapter.is_linear(mapping["blk.0.attn_q.weight"])
    assert not adapter.is_linear(mapping["blk.3.ffn_gate_inp.weight"])
    assert adapter.is_linear(mapping["blk.3.ffn_gate_shexp.weight"])
    assert (
        get_gguf_adapter(
            SimpleNamespace(
                get_text_config=lambda: SimpleNamespace(model_type="unknown")
            )
        )
        is None
    )


@pytest.mark.parametrize("fault", ["missing", "unknown", "shape", "nextn_shape"])
def test_inventory_errors_are_never_silently_dropped(config, inventory, fault):
    tensors = directory(inventory)
    if fault == "missing":
        del tensors["blk.0.attn_q.weight"]
    elif fault == "unknown":
        tensors["blk.0.unknown.weight"] = SimpleNamespace(shape=[1])
    elif fault == "shape":
        tensors["blk.3.attn_k_b.weight"].shape = [512, 256, 64]
    else:
        tensors["blk.45.nextn.eh_proj.weight"].shape = [4096, 8192]
    with pytest.raises(ValueError, match="mismatch|expected shape"):
        Glm5NextAdapter(config).build_name_map(tensors)


def test_hf_config_without_shared_adapter_alias_is_supported(config):
    del config.num_experts
    adapter = Glm5NextAdapter(config)
    assert adapter.config.num_experts == config.n_routed_experts
    assert not hasattr(config, "num_experts")


def test_mla_kv_split_transpose_exact_inverse(config):
    config.kv_lora_rank = 5
    original = np.arange(3 * (4 + 2) * 5, dtype=np.float32).reshape(3, 6, 5)
    key, value = np.split(original, [4], axis=1)
    tensors = {
        "blk.3.attn_k_b.weight": tensor(key.transpose(0, 2, 1)),
        "blk.3.attn_v_b.weight": tensor(value),
    }
    name = "model.layers.3.self_attn.kv_b_proj.weight"
    # V arrives first in an arbitrary shard directory; still emit exactly once.
    mapping = {raw: name for raw in reversed(tensors)}
    output = list(Glm5NextAdapter(config).weights(tensors, mapping, torch.float32))
    assert len(output) == 1 and output[0][0] == name
    torch.testing.assert_close(
        output[0][1], torch.from_numpy(original.reshape(-1, 5)), rtol=0, atol=0
    )


@pytest.mark.parametrize("part", ["k", "v"])
def test_pp_filter_must_keep_both_mla_halves(config, part):
    raw = f"blk.3.attn_{part}_b.weight"
    tensors = {raw: tensor(np.zeros((2, 4, 4)))}
    with pytest.raises(ValueError, match="filtered incompletely"):
        list(
            Glm5NextAdapter(config).weights(
                tensors,
                {raw: "model.layers.3.self_attn.kv_b_proj.weight"},
                torch.float16,
            )
        )


def test_kda_and_mhc_precision_preserving_inverses(config):
    config.linear_num_heads, config.linear_head_dim = 2, 4
    config.linear_conv_kernel_dim = 3
    source_log = np.array([-0.25, 0.75], dtype=np.float32)
    conv = np.arange(24, dtype=np.float32).reshape(1, 8, 1, 3) / 7
    precise = np.array([1 + 2**-20, 70000, -70000], dtype=np.float32)
    tensors = {
        "blk.0.ssm_a": tensor(-np.exp(source_log)),
        "blk.0.ssm_conv1d_q.weight": tensor(conv),
        "blk.0.hc_attn_scale.weight": tensor(precise),
    }
    mapping = {
        "blk.0.ssm_a": "model.layers.0.self_attn.A_log",
        "blk.0.ssm_conv1d_q.weight": "model.layers.0.self_attn.q_conv1d.weight",
        "blk.0.hc_attn_scale.weight": "model.layers.0.hc_attn_scale",
    }
    output = dict(Glm5NextAdapter(config).weights(tensors, mapping, torch.float16))
    assert all(weight.dtype == torch.float32 for weight in output.values())
    torch.testing.assert_close(
        output[mapping["blk.0.ssm_a"]], torch.from_numpy(source_log).reshape(1, 1, 2, 1)
    )
    torch.testing.assert_close(
        output[mapping["blk.0.ssm_conv1d_q.weight"]],
        torch.from_numpy(conv.reshape(8, 1, 3)),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output[mapping["blk.0.hc_attn_scale.weight"]],
        torch.from_numpy(precise),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "value,error",
    [(float("nan"), "non-finite"), (float("inf"), "non-finite"), (70000, "overflow")],
)
def test_dense_decode_rejects_corrupt_or_overflowed_weights(value, error):
    with pytest.raises(ValueError, match=error):
        decode_dense_tensor(tensor([value]), torch.float16, "test.weight")


def test_dense_decode_checks_directory_without_guessing_transpose():
    value = tensor(np.zeros((2, 3)))
    value.shape = [2, 3]
    with pytest.raises(ValueError, match="decoded shape"):
        decode_dense_tensor(value, torch.float16, "test.weight")


@pytest.mark.parametrize(
    "dtype,kind",
    [(torch.float16, "F16"), (torch.float32, "F32"), (torch.bfloat16, "BF16")],
)
def test_floating_linear_descriptors_match_payload(config, dtype, kind):
    raw, name = "output.weight", "lm_head.weight"
    output = dict(
        Glm5NextAdapter(config).weights(
            {raw: tensor(np.ones((3, 4)))}, {raw: name}, dtype
        )
    )
    assert output["lm_head.qweight_type"].item() == int(gguf.GGMLQuantizationType[kind])
    assert output["lm_head.qweight"].dtype == dtype


def test_iq3_iq4_expert_blocks_are_preserved_without_dequantization(
    config, monkeypatch
):
    config.n_routed_experts = config.num_experts = 2
    tensors, mapping = {}, {}
    for projection, kind in (("gate", "IQ3_S"), ("up", "IQ3_S"), ("down", "IQ4_XS")):
        qtype = gguf.GGMLQuantizationType[kind]
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        payload = (
            np.arange(2 * 8 * size, dtype=np.int64).astype(np.uint8).reshape(2, 8, size)
        )
        raw = f"blk.3.ffn_{projection}_exps.weight"
        tensors[raw] = SimpleNamespace(
            shape=[block, 8, 2], tensor_type=qtype, data=payload
        )
        mapping[raw] = f"model.layers.3.mlp.experts.{projection}_proj.weight"
    adapter = Glm5NextAdapter(config)
    monkeypatch.setattr(
        adapter, "_dense", lambda *args: pytest.fail("IQ experts must remain packed")
    )
    output = dict(adapter.weights(tensors, mapping, torch.float16))
    assert len(output) == 12
    for raw, value in tensors.items():
        projection = raw.split("ffn_")[1].split("_exps")[0]
        for expert in range(2):
            prefix = f"model.layers.3.mlp.experts.{expert}.{projection}_proj"
            assert output[prefix + ".qweight_type"].item() == int(value.tensor_type)
            assert np.shares_memory(output[prefix + ".qweight"].numpy(), value.data)
            np.testing.assert_array_equal(
                output[prefix + ".qweight"].numpy(), value.data[expert]
            )


def test_glm_router_retains_fp32_weights_and_compute(config, monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.fused_moe.router import gate_linear
    from vllm.models.glm5next.nvidia import model as glm

    config.hidden_size = 2
    config.n_routed_experts = config.num_experts = 2
    config.n_shared_experts = None
    group = SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0
    )
    monkeypatch.setattr(glm, "get_ep_group", lambda: group)
    for module in (glm, linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    # Keep native GateLinear construction, loading and forward; only the
    # unneeded experts and hardware-specific matmul backend are replaced.
    monkeypatch.setattr(glm, "FusedMoE", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(
        gate_linear,
        "current_platform",
        SimpleNamespace(
            is_device_capability=lambda *args: False,
            is_device_capability_family=lambda *args: False,
            is_cuda=lambda: False,
        ),
    )
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=False,
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    vconfig = VllmConfig(device_config=DeviceConfig(device="cpu"))
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with set_current_vllm_config(vconfig):
            moe = glm.Glm5NextMoE(config, parallel)
            gate = moe.gate
            assert gate.weight.dtype == torch.float32
            weight = torch.tensor([[1.0001, -1], [-1, 1.0002]], dtype=torch.float32)
            gate.weight.weight_loader(gate.weight, weight)
            monkeypatch.setattr(
                gate.quant_method,
                "apply",
                lambda layer, x, bias: torch.nn.functional.linear(
                    x, layer.weight, bias
                ),
            )
            x = torch.ones((1, 2), dtype=torch.float16)
            actual, _ = gate(x)
    finally:
        torch.set_default_dtype(old_dtype)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, x.float() @ weight.T, rtol=0, atol=0)
    assert not torch.equal(actual, (x @ weight.half().T).float())


@pytest.mark.parametrize(
    "kind", [gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.BF16]
)
def test_dense_decode_real_gguf_payload(tmp_path, kind):
    from vllm.transformers_utils.gguf_tensor_reader import GGUFReader

    values = np.linspace(-2, 2, 3 * 256, dtype=np.float32).reshape(3, 256)
    payload = gguf.quants.quantize(values, kind)
    path = tmp_path / "dense.gguf"
    writer = gguf.GGUFWriter(path, "glm5-next")
    writer.add_tensor("test.weight", payload, raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    source = GGUFReader(path).tensors[0]
    actual = decode_dense_tensor(source, torch.float32, "test.weight")
    expected = gguf.quants.dequantize(payload, kind)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_native_text_loader_preserves_indexer_fp32_before_fusion(config, monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.glm5next.nvidia.model import Glm5NextModel

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    model = Glm5NextModel.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.config = copy.copy(config)
    model.config.n_routed_experts = None
    model._replicate_dflash_embedding = False
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    indexer = layer.self_attn.indexer = torch.nn.Module()
    indexer.register_buffer("_wp_fp32", None, persistent=False)
    model.layers = torch.nn.ModuleList([layer])
    vconfig = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(vconfig):
        indexer.wk_weights_proj = linear.MergedColumnParallelLinear(
            4, [3, 2], bias=False, params_dtype=torch.float16, disable_tp=True
        )
    weight = torch.tensor([[1 + 2**-20, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
    raw = "blk.0.indexer.proj.weight"
    adapter = Glm5NextAdapter(config)
    decoded = dict(
        adapter.weights(
            {raw: tensor(weight.numpy())},
            {raw: "layers.0.self_attn.indexer.weights_proj.weight"},
            torch.float16,
        )
    )
    assert (
        decoded["layers.0.self_attn.indexer.weights_proj.weight"].dtype == torch.float32
    )
    loaded = model.load_weights(
        [
            (
                "layers.0.self_attn.indexer.wk.weight",
                torch.ones((3, 4), dtype=torch.float16),
            ),
            *decoded.items(),
        ]
    )
    assert loaded == {"layers.0.self_attn.indexer.wk_weights_proj.weight"}
    torch.testing.assert_close(indexer._wp_fp32, weight.T.contiguous(), rtol=0, atol=0)
    assert not torch.equal(
        indexer._wp_fp32, indexer.wk_weights_proj.weight[3:].T.float()
    )
    adapter.validate_loaded_parameters(model, loaded)
    indexer._wp_fp32 = None
    with pytest.raises(ValueError, match="missing FP32 indexer gate"):
        adapter.validate_loaded_parameters(model, loaded)


def test_loaded_guard_checks_dense_and_all_bulk_expert_payloads(config):
    from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
    from vllm.model_executor.models.utils import PPMissingLayer

    model = torch.nn.Module()
    model.norm = torch.nn.Linear(2, 2, bias=False)
    model.remote_stage = PPMissingLayer()
    model.visual = torch.nn.Linear(2, 2)  # Checked separately by the vision owner.
    experts = model.experts = torch.nn.Module()
    method = GGUFNativeMoEMethod.__new__(GGUFNativeMoEMethod)
    method.num_experts = 2
    method.weight_types = {shard: 21 for shard in ("w1", "w3", "w2")}
    method.loaded_experts = {shard: {0, 1} for shard in ("w1", "w3", "w2")}
    experts.quant_method = method
    for name in ("w13_qweight", "w2_qweight", "w13_qweight_type", "w2_qweight_type"):
        experts.register_parameter(name, torch.nn.Parameter(torch.empty(0)))
    adapter = Glm5NextAdapter(config)
    adapter.validate_loaded_parameters(model, {"norm.weight"})
    with pytest.raises(ValueError, match="norm.weight"):
        adapter.validate_loaded_parameters(model, set())
    method.loaded_experts["w2"] = {0}
    with pytest.raises(ValueError, match="incomplete expert payloads: experts.w2"):
        adapter.validate_loaded_parameters(model, set(dict(model.named_parameters())))
    with pytest.raises(ValueError, match="loaded-parameter set"):
        adapter.validate_loaded_parameters(model, None)
