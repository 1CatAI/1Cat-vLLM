# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.gguf import GGUFConfig, GGUFLinearMethod
from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout


def make_layer(types, *, layout=None):
    layer = torch.nn.Module()
    method = GGUFLinearMethod(GGUFConfig(), layout)
    method.create_weights(layer, 8, [3] * len(types), 8, 3 * len(types), torch.float16)
    layer.quant_method = method
    for i, source_type in reversed(list(enumerate(types))):
        layer.qweight.shard_id.append(i)
        layer.qweight.shard_id_map[i] = len(layer.qweight.data_container)
        layer.qweight.data_container.append(
            torch.full((3, 8), i + 1, dtype=torch.float16)
        )
        layer.qweight_type.shard_weight_type[i] = source_type
    layer.qweight.data = torch.empty(0)
    return layer, method


@pytest.mark.parametrize("types", [(1,), (30,), (0,), (1, 30), (30, 0, 1)])
def test_floating_gguf_shards_restore_dense_method(types, monkeypatch):
    monkeypatch.setattr(
        UnquantizedLinearMethod, "process_weights_after_loading", lambda *_: None
    )
    layer, method = make_layer(types)
    method.process_weights_after_loading(layer)
    assert type(layer.quant_method) is UnquantizedLinearMethod
    assert not hasattr(layer, "qweight")
    expected = torch.cat(
        [torch.full((3, 8), i + 1, dtype=torch.float16) for i in range(len(types))]
    )
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
    x = torch.arange(16, dtype=torch.float16).reshape(2, 8)
    torch.testing.assert_close(
        torch.nn.functional.linear(x, layer.weight), x @ expected.T, rtol=0, atol=0
    )


@pytest.mark.parametrize("types", [(1, 12), (12,), (30, 2)])
def test_mixed_and_quantized_shards_retain_quantized_method(types):
    layer, method = make_layer(types)
    assert not method._prepare_dense_weight(layer)
    assert layer.quant_method is method
    assert not hasattr(layer, "weight")


def test_dense_input_layout_cannot_be_bypassed():
    layer, method = make_layer((1,), layout=GGUFHeadTilingLayout(2, 2))
    assert not method._prepare_dense_weight(layer)
    assert layer.quant_method is method


def test_incomplete_dense_shard_is_rejected():
    layer, method = make_layer((1, 30))
    layer.qweight.data_container[0] = torch.empty(2, 8, dtype=torch.float16)
    assert not method._prepare_dense_weight(layer)


def test_single_dense_weight_preserves_shared_checkpoint_storage(monkeypatch):
    monkeypatch.setattr(
        UnquantizedLinearMethod, "process_weights_after_loading", lambda *_: None
    )
    layer = torch.nn.Module()
    method = GGUFLinearMethod(GGUFConfig())
    method.create_weights(layer, 8, [3], 8, 3, torch.float16)
    layer.quant_method = method
    layer.qweight.materialize((3, 8), dtype=torch.float16)
    layer.qweight.data.fill_(2)
    layer.qweight_type.weight_type = 1
    shared = layer.qweight
    method.process_weights_after_loading(layer)
    assert layer.weight.data_ptr() == shared.data_ptr()
    torch.testing.assert_close(shared, torch.full((3, 8), 2, dtype=torch.float16))


def test_model_dense_preparation_runs_after_all_linear_methods(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    events = []
    model = torch.nn.Module()
    for name in ("a", "b"):
        layer = torch.nn.Module()
        layer.quant_method = UnquantizedLinearMethod()
        model.add_module(name, layer)

    monkeypatch.setattr(
        UnquantizedLinearMethod,
        "process_weights_after_loading",
        lambda _, layer: events.append(layer),
    )
    model.prepare_loaded_linear_weights = lambda: events.append("prepare")
    process_weights_after_loading(
        model, SimpleNamespace(quantization="gguf"), torch.device("cpu")
    )
    assert events == [model.a, model.b, "prepare"]


@pytest.mark.parametrize("reduced", [False, True])
def test_fp32_router_does_not_require_fp16_partial_reduction(reduced, monkeypatch):
    from types import SimpleNamespace

    from vllm import envs
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
        _router_batch_runtime_ok,
    )

    monkeypatch.setattr(envs, "VLLM_SM70_MTP_ROUTER_BATCH", True)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", reduced
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    common = dict(
        dtype=torch.float16,
        device=torch.device("cuda:0"),
        is_cuda=True,
        is_contiguous=lambda: True,
        data_ptr=lambda: 16,
    )
    x = SimpleNamespace(ndim=2, shape=(5, 2560), **common)
    packed = SimpleNamespace(shape=(64, 40, 2, 4, 8, 8), **common)
    assert _router_batch_runtime_ok(x, packed)


def test_loaded_dense_projection_prepares_existing_fp16_method(monkeypatch):
    from types import SimpleNamespace

    from vllm import envs
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
        Qwen38SM70FP16LinearMethod,
        enable_qwen38_sm70_fp16_gemv,
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        UnquantizedLinearMethod, "process_weights_after_loading", lambda *_: None
    )
    monkeypatch.setattr(current_platform, "is_device_capability", lambda *_: True)
    for name, value in {
        "VLLM_SM70_QWEN38_FP16_GEMV": True,
        "VLLM_SM70_QWEN38_BATCH_FASTPATH": False,
        "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16": False,
        "VLLM_SM70_QWEN38_FUSED_HC_FP16": False,
        "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8": False,
    }.items():
        monkeypatch.setattr(envs, name, value)
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.linear_attn.in_proj_ba"
    method = GGUFLinearMethod(GGUFConfig())
    method.create_weights(layer, 8, [24], 8, 24, torch.float16)
    layer.quant_method = method
    layer.qweight.materialize((24, 8), dtype=torch.float16)
    layer.qweight.data.fill_(2)
    layer.qweight_type.weight_type = 30
    model = torch.nn.Module()
    model.add_module("ba", layer)
    model.model_config = SimpleNamespace(
        dtype=torch.float16, architectures=["Qwen4ExpForCausalLM"]
    )
    model.vllm_config = SimpleNamespace(
        model_config=model.model_config,
        parallel_config=SimpleNamespace(),
        speculative_config=SimpleNamespace(method="mtp"),
    )
    method.process_weights_after_loading(layer)
    enable_qwen38_sm70_fp16_gemv(model, torch.float16, model.vllm_config)
    assert isinstance(layer.quant_method, Qwen38SM70FP16LinearMethod)


@pytest.mark.parametrize("embedding_storage", ["dense", "original"])
@pytest.mark.parametrize("embedding_type", ["F16", "IQ4_XS"])
def test_flashnext_loader_configures_embedding_and_floating_modules(
    monkeypatch, embedding_storage, embedding_type
):
    from types import SimpleNamespace

    from vllm.model_executor.kernels.ple import gguf_pinned
    from vllm.model_executor.model_loader import gguf_loader
    from vllm.model_executor.model_loader.gguf_adapters.qwen4exp import Qwen4ExpAdapter

    hf_config = SimpleNamespace(
        num_hidden_layers=0,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        tie_word_embeddings=False,
    )
    adapter = Qwen4ExpAdapter(hf_config, tp_size=4)
    loader = gguf_loader.GGUFModelLoader.__new__(gguf_loader.GGUFModelLoader)
    loader._native_adapter = adapter
    loader._native_tensors = {"output.weight": SimpleNamespace()}
    types = {
        "model.embed_tokens.weight": embedding_type,
        "model.layers.0.mlp.gate.weight": "F32",
        "model.layers.0.hyper_connection_attn.input_mix_weight_down.weight": "BF16",
        "model.layers.0.linear_attn.in_proj_a.weight": "BF16",
    }
    monkeypatch.setattr(loader, "_prepare_weights", lambda *_: "fixture.gguf")
    monkeypatch.setattr(loader, "_get_gguf_weights_map", lambda *_: {})
    monkeypatch.setattr(loader, "_get_all_gguf_files", lambda *_: [])
    monkeypatch.setattr(loader, "_get_gguf_weight_type", lambda *_: types)
    monkeypatch.setattr(gguf_pinned, "prepare_pinned_gguf_ple", lambda *_: None)

    class ModelConstructionReached(Exception):
        pass

    def check_config(**kwargs):
        config = kwargs["vllm_config"]
        assert adapter.packed_embeddings == (embedding_storage == "original")
        expected = {
            "model.layers.0.mlp.gate",
            "model.layers.0.hyper_connection_attn.input_mix_weight_down",
        }
        if embedding_type == "F16":
            expected.add("model.embed_tokens")
        assert set(config.quant_config.unquantized_modules) == expected
        raise ModelConstructionReached

    monkeypatch.setattr(gguf_loader, "initialize_model", check_config)
    config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        kernel_config=SimpleNamespace(
            sm70_gguf=SimpleNamespace(
                embedding_storage=embedding_storage, enabled=False
            )
        ),
        quant_config=GGUFConfig(),
    )
    model_config = SimpleNamespace(hf_config=hf_config, dtype=torch.float16)
    with pytest.raises(ModelConstructionReached):
        loader.load_model(config, model_config)
