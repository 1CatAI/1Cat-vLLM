# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small real GGUF files exercise vision inversion and the loader on CPU."""

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import gguf_loader
from vllm.model_executor.model_loader.gguf_adapters.qwen3_vision import Qwen3VisionGGUF
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    StageMissingLayer,
)
from vllm.transformers_utils import gguf_config
from vllm.transformers_utils.gguf_files import gguf_tensor_index

F16 = gguf.GGMLQuantizationType.F16
F32 = gguf.GGMLQuantizationType.F32
BF16 = gguf.GGMLQuantizationType.BF16


@pytest.fixture(autouse=True)
def no_cuda_context():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def write_gguf(path, arrays, tensor_type=F16):
    writer = gguf.GGUFWriter(path, "clip")
    for name, array in arrays.items():
        data = gguf.quants.quantize(np.ascontiguousarray(array), tensor_type)
        writer.add_tensor(name, data, raw_dtype=tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def vision_fixture():
    """HF-style parameters and their converter-format names, independent of adapter.

    Deliberately distinct values expose transposes, slice swaps and index swaps.
    Unsorted deepstack indexes test that checkpoint block IDs do not become the
    ordered HF merger indexes. All values are exactly representable in BF16.
    """
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=6,
        spatial_merge_size=2,
        temporal_patch_size=2,
        depth=2,
        deepstack_visual_indexes=[1, 0],
        out_hidden_size=8,
        in_channels=3,
        patch_size=2,
        num_position_embeddings=9,
    )
    specifications = [
        ("v.position_embd.weight", "pos_embed.weight", (9, 4)),
        ("v.patch_embd.bias", "patch_embed.proj.bias", (4,)),
        ("v.post_ln.weight", "merger.norm.weight", (4,)),
        ("v.post_ln.bias", "merger.norm.bias", (4,)),
        ("mm.0.weight", "merger.linear_fc1.weight", (16, 16)),
        ("mm.0.bias", "merger.linear_fc1.bias", (16,)),
        ("mm.2.weight", "merger.linear_fc2.weight", (8, 16)),
        ("mm.2.bias", "merger.linear_fc2.bias", (8,)),
    ]
    block_parameters = [
        ("attn_qkv.weight", "attn.qkv.weight", (12, 4)),
        ("attn_qkv.bias", "attn.qkv.bias", (12,)),
        ("attn_out.weight", "attn.proj.weight", (4, 4)),
        ("attn_out.bias", "attn.proj.bias", (4,)),
        ("ffn_up.weight", "mlp.linear_fc1.weight", (6, 4)),
        ("ffn_up.bias", "mlp.linear_fc1.bias", (6,)),
        ("ffn_down.weight", "mlp.linear_fc2.weight", (4, 6)),
        ("ffn_down.bias", "mlp.linear_fc2.bias", (4,)),
        ("ln1.weight", "norm1.weight", (4,)),
        ("ln1.bias", "norm1.bias", (4,)),
        ("ln2.weight", "norm2.weight", (4,)),
        ("ln2.bias", "norm2.bias", (4,)),
    ]
    for block in (0, 1):
        specifications.extend(
            (f"v.blk.{block}.{raw}", f"blocks.{block}.{target}", shape)
            for raw, target, shape in block_parameters
        )
    shape: tuple[int, ...]
    for raw_block, merger_index in ((1, 0), (0, 1)):
        for raw, target, shape in (
            ("norm.weight", "norm.weight", (16,)),
            ("norm.bias", "norm.bias", (16,)),
            ("fc1.weight", "linear_fc1.weight", (16, 16)),
            ("fc1.bias", "linear_fc1.bias", (16,)),
            ("fc2.weight", "linear_fc2.weight", (8, 16)),
            ("fc2.bias", "linear_fc2.bias", (8,)),
        ):
            specifications.append(
                (
                    f"v.deepstack.{raw_block}.{raw}",
                    f"deepstack_merger_list.{merger_index}.{target}",
                    shape,
                )
            )
    raw_arrays = {}
    expected = {}
    for index, (raw, target, shape) in enumerate(specifications):
        values = ((np.arange(np.prod(shape)) + index * 7) % 127 - 63) / 32
        array = values.astype(np.float32).reshape(shape)
        raw_arrays[raw] = array
        expected["visual." + target] = torch.from_numpy(array.copy())
    patch = np.arange(4 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2, 2)
    patch = (patch - 96) / 32
    raw_arrays["v.patch_embd.weight"] = patch[:, :, 0].copy()
    raw_arrays["v.patch_embd.weight.1"] = patch[:, :, 1].copy()
    expected["visual.patch_embed.proj.weight"] = torch.from_numpy(patch.copy())
    return config, raw_arrays, expected


@pytest.mark.parametrize("tensor_type", [F16, F32, BF16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_real_gguf_roundtrip_preserves_all_vision_values(tmp_path, tensor_type, dtype):
    config, raw_arrays, expected = vision_fixture()
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, raw_arrays, tensor_type)
    tensors = gguf_tensor_index([path])
    adapter = Qwen3VisionGGUF(config, tensors)
    actual = dict(adapter.weights(dtype))
    assert actual.keys() == expected.keys()
    assert len(tensors) == len(actual) + 1
    for name, value in expected.items():
        assert actual[name].device.type == "cpu"
        assert actual[name].dtype == dtype
        torch.testing.assert_close(actual[name], value.to(dtype), rtol=0, atol=0)
    assert set(adapter.unquantized_modules) == {
        name.removesuffix(".weight") for name in expected if name.endswith(".weight")
    }
    # The output must own its memory; parameter preparation cannot modify mmap.
    actual["visual.pos_embed.weight"].fill_(0)
    again = dict(adapter.weights(dtype))
    torch.testing.assert_close(
        again["visual.pos_embed.weight"], expected["visual.pos_embed.weight"].to(dtype)
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape"])
def test_vision_rejects_invalid_tensor_directory(tmp_path, mutation):
    config, arrays, _ = vision_fixture()
    name = "v.blk.0.attn_qkv.bias"
    if mutation == "missing":
        del arrays[name]
        error = "missing=.*attn_qkv.bias"
    elif mutation == "extra":
        arrays["unexpected.weight"] = np.ones(4, np.float32)
        error = "unexpected=.*unexpected.weight"
    else:
        arrays[name] = np.ones(13, np.float32)
        error = "attn_qkv.bias: shape"
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays)
    with pytest.raises(ValueError, match=error):
        Qwen3VisionGGUF(config, gguf_tensor_index([path]))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1e10])
def test_vision_rejects_nonfinite_and_fp16_overflow(tmp_path, invalid):
    config, arrays, _ = vision_fixture()
    arrays["v.position_embd.weight"][0, 0] = invalid
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays, F32)
    adapter = Qwen3VisionGGUF(config, gguf_tensor_index([path]))
    with pytest.raises(ValueError, match="nonfinite or torch.float16 overflow"):
        list(adapter.weights(torch.float16))


def test_vision_rejects_quantized_tensor_and_unsupported_temporal_geometry(tmp_path):
    config, arrays, _ = vision_fixture()
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays)
    tensors = gguf_tensor_index([path])
    name = "v.position_embd.weight"
    tensors[name] = SimpleNamespace(
        shape=tensors[name].shape, tensor_type=gguf.GGMLQuantizationType.Q8_0
    )
    with pytest.raises(ValueError, match="dense mmproj required"):
        Qwen3VisionGGUF(config, tensors)
    config.temporal_patch_size = 1
    with pytest.raises(ValueError, match="temporal_patch_size=2"):
        Qwen3VisionGGUF(config, tensors)


class UnreadablePayload:
    def __init__(self, tensor):
        self.shape = tensor.shape
        self.tensor_type = tensor.tensor_type

    @property
    def data(self):
        raise AssertionError("A skipped vision payload was read")


def test_vision_filter_reads_only_selected_complete_temporal_pair(tmp_path):
    config, arrays, expected = vision_fixture()
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays)
    tensors = gguf_tensor_index([path])
    selected = {
        "v.patch_embd.weight": "visual.patch_embed.proj.weight",
        "v.patch_embd.weight.1": "visual.patch_embed.proj.weight",
    }
    guarded = {
        name: tensor if name in selected else UnreadablePayload(tensor)
        for name, tensor in tensors.items()
    }
    adapter = Qwen3VisionGGUF(config, guarded)
    actual = dict(adapter.weights(torch.float16, selected))
    assert list(actual) == ["visual.patch_embed.proj.weight"]
    torch.testing.assert_close(
        actual[next(iter(actual))], expected[next(iter(actual))].half()
    )
    assert list(adapter.weights(torch.float16, {})) == []
    with pytest.raises(ValueError, match="retain both patch slices"):
        list(adapter.weights(torch.float16, {next(iter(selected)): next(iter(actual))}))
    with pytest.raises(ValueError, match="unknown/changed names"):
        list(adapter.weights(torch.float16, {"v.patch_embd.bias": "visual.wrong"}))


def loader_fixture(tmp_path, monkeypatch, *, language_model_only):
    vision_config, arrays, expected_vision = vision_fixture()
    text_config = SimpleNamespace(
        model_type="qwen4_exp_text",
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_value_head_dim=2,
        gguf_architecture="qwen4exp",
        gguf_ple_constants={},
        gguf_ple_eos_token_id=0,
    )
    config = SimpleNamespace(
        vision_config=vision_config, get_text_config=lambda: text_config
    )
    embedding = np.arange(32, dtype=np.float32).reshape(8, 4) / 8
    path = tmp_path / "text.gguf"
    write_gguf(path, {"token_embd.weight": embedding})
    # Only the large native-config metadata header is stubbed. File discovery,
    # indexing, native text/vision adapters and tensor loading remain real.
    monkeypatch.setattr(gguf_config, "load_gguf_config", lambda _: text_config)
    model_config = SimpleNamespace(
        model=str(path),
        hf_config=config,
        dtype=torch.float16,
        multimodal_config=SimpleNamespace(language_model_only=language_model_only),
    )
    loader = GGUFModelLoader(LoadConfig(load_format="gguf"))
    return loader, model_config, arrays, expected_vision, torch.from_numpy(embedding)


class TinyLanguageWrapper(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = nn.Embedding(8, 4, dtype=torch.float16)
        self.visual = visual

    def load_weights(self, weights):
        self.loaded = AutoWeightsLoader(self).load_weights(weights)
        return self.loaded


@pytest.mark.parametrize("present_mmproj", [False, True])
def test_language_only_loader_never_discovers_or_reads_mmproj(
    tmp_path, monkeypatch, present_mmproj
):
    loader, config, _, _, embedding = loader_fixture(
        tmp_path, monkeypatch, language_model_only=True
    )
    if present_mmproj:
        (tmp_path / "mmproj.gguf").write_bytes(b"invalid and intentionally unreadable")

    def reject_discovery(*args, **kwargs):
        raise AssertionError("language_model_only must not discover mmproj")

    monkeypatch.setattr(gguf_loader, "detect_gguf_multimodal", reject_discovery)
    names = loader._get_gguf_weights_map(config)
    name = "language_model.model.embed_tokens.weight"
    assert names == {"token_embd.weight": name}
    assert loader._native_vision_adapter is None
    assert loader._get_gguf_weight_type(config, config.model, names) == {name: "F16"}
    model = TinyLanguageWrapper(StageMissingLayer("vision_tower"))
    loader.load_weights(model, config)
    assert model.loaded == {name}
    torch.testing.assert_close(
        model.language_model.model.embed_tokens.weight, embedding.half()
    )
    # Exercise the loader's prepared-map cache as well as initial discovery.
    assert dict(loader.get_all_weights(config, model)).keys() == {name}


def test_multimodal_loader_requires_mmproj(tmp_path, monkeypatch):
    loader, config, _, _, _ = loader_fixture(
        tmp_path, monkeypatch, language_model_only=False
    )
    with pytest.raises(ValueError, match="requires mmproj"):
        loader._get_gguf_weights_map(config)


def test_multimodal_loader_reads_dense_vision_without_text_quantization(
    tmp_path, monkeypatch
):
    loader, config, arrays, expected, embedding = loader_fixture(
        tmp_path, monkeypatch, language_model_only=False
    )
    write_gguf(tmp_path / "mmproj.gguf", arrays, BF16)
    model = TinyLanguageWrapper(nn.Module())
    actual = dict(loader.get_all_weights(config, model))
    expected["language_model.model.embed_tokens.weight"] = embedding
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value.half(), rtol=0, atol=0)
    assert loader._get_gguf_weight_type(
        config, config.model, loader._gguf_prepared_weights_map
    ) == {"language_model.model.embed_tokens.weight": "F16"}


@pytest.mark.parametrize("missing_tower", [PPMissingLayer, StageMissingLayer])
def test_pipeline_stage_skips_all_vision_payloads(tmp_path, monkeypatch, missing_tower):
    loader, config, arrays, _, embedding = loader_fixture(
        tmp_path, monkeypatch, language_model_only=False
    )
    write_gguf(tmp_path / "mmproj.gguf", arrays)
    loader._get_gguf_weights_map(config)
    original = Qwen3VisionGGUF.weights

    def reject_payloads(*args, **kwargs):
        raise AssertionError("Downstream PP ranks must not load vision payloads")

    monkeypatch.setattr(Qwen3VisionGGUF, "weights", reject_payloads)
    tower = (
        StageMissingLayer("vision_tower")
        if missing_tower is StageMissingLayer
        else PPMissingLayer()
    )
    model = TinyLanguageWrapper(tower)
    weights = dict(loader.get_all_weights(config, model))
    assert weights.keys() == {"language_model.model.embed_tokens.weight"}
    torch.testing.assert_close(next(iter(weights.values())), embedding.half())
    # A first-stage wrapper with skip_weight must also filter before decoding.
    monkeypatch.setattr(Qwen3VisionGGUF, "weights", original)
    loader._native_vision_adapter.tensors = {
        name: UnreadablePayload(tensor)
        for name, tensor in loader._native_vision_adapter.tensors.items()
    }
    filtered = dict(
        loader.get_all_weights(
            config,
            TinyLanguageWrapper(nn.Module()),
            skip_weight=lambda n: n.startswith("visual."),
        )
    )
    assert filtered.keys() == weights.keys()


def test_generic_weight_paths_honor_language_only_without_native_adapter(
    tmp_path, monkeypatch
):
    loader, config, _, _, embedding = loader_fixture(
        tmp_path, monkeypatch, language_model_only=True
    )
    names = {"token_embd.weight": "language_model.model.embed_tokens.weight"}

    def reject_discovery(*args, **kwargs):
        raise AssertionError("text-only metadata/payload paths must not open mmproj")

    monkeypatch.setattr(gguf_loader, "detect_gguf_multimodal", reject_discovery)
    assert loader._get_gguf_weight_type(config, config.model, names) == {
        next(iter(names.values())): "F16"
    }
    weights = dict(loader._get_weights_iterator(config, config.model, names))
    torch.testing.assert_close(next(iter(weights.values())), embedding.half())
