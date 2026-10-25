# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real dense GGUF files exercise GLM vision inversion without CUDA."""

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader.gguf_adapters.glm5next_vision import (
    Glm5NextVisionGGUF,
)
from vllm.transformers_utils.configs.glm5_next import Glm5NextVisionConfig
from vllm.transformers_utils.gguf_files import gguf_tensor_index

F16 = gguf.GGMLQuantizationType.F16
F32 = gguf.GGMLQuantizationType.F32
BF16 = gguf.GGMLQuantizationType.BF16


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_vision_config_preserves_checkpoint_norm_epsilon(eps):
    assert Glm5NextVisionConfig(rms_norm_eps=eps).rms_norm_eps == eps
    assert Glm5NextVisionConfig().rms_norm_eps == 1e-5


def test_adapter_loads_every_native_vision_parameter(tmp_path, monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.models.glm5next.nvidia import multimodal as vision

    config, arrays, expected = vision_fixture()
    config.rms_norm_eps = 1e-5
    config.swiglu_limit = 10
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    # Keep native parameter-owning layers and their real weight loaders.
    # Only the parameter-free device/backend setup is replaced for CPU testing.
    monkeypatch.setattr(vision, "is_vit_use_data_parallel", lambda: True)
    monkeypatch.setattr(vision, "get_vit_attn_backend", lambda **kwargs: None)
    monkeypatch.setattr(vision, "get_rope", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(
        vision, "MMEncoderAttention", lambda **kwargs: torch.nn.Identity()
    )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        tower = vision.Glm5NextVisionTransformer(
            SimpleNamespace(), config, norm_eps=config.rms_norm_eps
        )
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays)
    adapter = Glm5NextVisionGGUF(config, gguf_tensor_index([path]))
    parameters = dict(tower.named_parameters())
    assert {"visual." + name: tuple(p.shape) for name, p in parameters.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }
    loaded = tower.load_weights(
        (name.removeprefix("visual."), value)
        for name, value in adapter.weights(torch.float32)
    )
    assert loaded == set(parameters)
    for name, parameter in parameters.items():
        torch.testing.assert_close(
            parameter, expected["visual." + name], rtol=0, atol=0
        )


@pytest.fixture(autouse=True)
def no_cuda_context():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def write_gguf(path, arrays, tensor_type=F16):
    writer = gguf.GGUFWriter(path, "clip")
    writer.add_string("clip.projector_type", "glm5v")
    for name, array in arrays.items():
        data = gguf.quants.quantize(np.ascontiguousarray(array), tensor_type)
        writer.add_tensor(name, data, raw_dtype=tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def vision_fixture():
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=12,
        projection_intermediate_size=24,
        out_hidden_size=16,
        num_heads=2,
        depth=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    specifications = [
        ("v.patch_embd.bias", "patch_embed.proj.bias", (8,)),
        ("v.post_ln.weight", "post_layernorm.weight", (8,)),
        ("mm.patch_merger.weight", "downsample.weight", (16, 8, 2, 2)),
        ("mm.patch_merger.bias", "downsample.bias", (16,)),
        ("mm.model.fc.weight", "merger.proj.weight", (16, 16)),
        ("mm.post_norm.weight", "merger.post_projection_norm.weight", (16,)),
        ("mm.post_norm.bias", "merger.post_projection_norm.bias", (16,)),
        ("mm.down.weight", "merger.down_proj.weight", (16, 24)),
        ("mm.gate.weight", "merger.gate_proj.weight", (24, 16)),
        ("mm.up.weight", "merger.up_proj.weight", (24, 16)),
    ]
    per_block = [
        ("attn_qkv.weight", "attn.qkv.weight", (24, 8)),
        ("attn_qkv.bias", "attn.qkv.bias", (24,)),
        ("attn_out.weight", "attn.proj.weight", (8, 8)),
        ("attn_out.bias", "attn.proj.bias", (8,)),
        ("attn_q_norm.weight", "attn.q_norm.weight", (4,)),
        ("attn_k_norm.weight", "attn.k_norm.weight", (4,)),
        ("ln1.weight", "norm1.weight", (8,)),
        ("ln2.weight", "norm2.weight", (8,)),
        ("ffn_gate.weight", "mlp.gate_proj.weight", (12, 8)),
        ("ffn_gate.bias", "mlp.gate_proj.bias", (12,)),
        ("ffn_up.weight", "mlp.up_proj.weight", (12, 8)),
        ("ffn_up.bias", "mlp.up_proj.bias", (12,)),
        ("ffn_down.weight", "mlp.down_proj.weight", (8, 12)),
        ("ffn_down.bias", "mlp.down_proj.bias", (8,)),
    ]
    for index in (0, 1):
        specifications.extend(
            (f"v.blk.{index}.{raw}", f"blocks.{index}.{target}", shape)
            for raw, target, shape in per_block
        )
    arrays = {}
    expected = {}
    # Distinct, exactly BF16-representable values reveal permutation errors.
    for index, (raw, target, shape) in enumerate(specifications):
        values = ((np.arange(np.prod(shape)) + 11 * index) % 127 - 63) / 32
        array = values.astype(np.float32).reshape(shape)
        arrays[raw] = array
        expected["visual." + target] = torch.from_numpy(array.copy())
    for prefix in ("visual.blocks.0.mlp", "visual.blocks.1.mlp", "visual.merger"):
        for suffix in ("weight",) if prefix.endswith("merger") else ("weight", "bias"):
            gate = expected.pop(f"{prefix}.gate_proj.{suffix}")
            up = expected.pop(f"{prefix}.up_proj.{suffix}")
            expected[f"{prefix}.gate_up_proj.{suffix}"] = torch.cat([gate, up])
    patch = np.arange(8 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(8, 3, 2, 2, 2)
    patch = (patch % 127 - 63) / 32
    arrays["v.patch_embd.weight"] = patch[:, :, 0].copy()
    arrays["v.patch_embd.weight.1"] = patch[:, :, 1].copy()
    expected["visual.patch_embed.proj.weight"] = torch.from_numpy(patch.copy())
    return config, arrays, expected


@pytest.mark.parametrize("tensor_type", [F16, F32, BF16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_real_gguf_preserves_all_packed_and_convolution_values(
    tmp_path, tensor_type, dtype
):
    config, arrays, expected = vision_fixture()
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays, tensor_type)
    adapter = Glm5NextVisionGGUF(config, gguf_tensor_index([path]))
    # Input/filter order must not swap gate/up or the two temporal slices.
    selected = dict(reversed(list(adapter.name_map.items())))
    actual = dict(adapter.weights(dtype, selected))
    assert actual.keys() == expected.keys()
    assert len(arrays) == 40 and len(actual) == 34
    for name, value in expected.items():
        assert actual[name].device.type == "cpu"
        assert actual[name].dtype == dtype
        torch.testing.assert_close(actual[name], value.to(dtype), rtol=0, atol=0)
    assert set(adapter.unquantized_modules) == {
        name.removesuffix(".weight") for name in expected if name.endswith(".weight")
    }
    actual["visual.blocks.0.attn.qkv.weight"].fill_(0)
    again = dict(adapter.weights(dtype))
    torch.testing.assert_close(
        again["visual.blocks.0.attn.qkv.weight"],
        expected["visual.blocks.0.attn.qkv.weight"].to(dtype),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape"])
def test_invalid_tensor_directory_fails_before_payload_read(tmp_path, mutation):
    config, arrays, _ = vision_fixture()
    name = "v.blk.0.attn_q_norm.weight"
    if mutation == "missing":
        del arrays[name]
        message = "missing=.*attn_q_norm"
    elif mutation == "extra":
        arrays["v.position_embd.weight"] = np.zeros((4, 8), np.float32)
        message = "unexpected=.*position_embd"
    else:
        arrays[name] = np.zeros(8, np.float32)
        message = "attn_q_norm.weight: expected shape"
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays)
    with pytest.raises(ValueError, match=message):
        Glm5NextVisionGGUF(config, gguf_tensor_index([path]))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1e10])
def test_nonfinite_and_destination_overflow_are_rejected(tmp_path, invalid):
    config, arrays, _ = vision_fixture()
    arrays["mm.gate.weight"][0, 0] = invalid
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays, F32)
    adapter = Glm5NextVisionGGUF(config, gguf_tensor_index([path]))
    with pytest.raises(ValueError, match="non-finite|overflow destination"):
        list(adapter.weights(torch.float16))


@pytest.mark.parametrize(
    "field,value", [("temporal_patch_size", 1), ("num_heads", 3), ("depth", 0)]
)
def test_unsupported_geometry_is_rejected(tmp_path, field, value):
    config, arrays, _ = vision_fixture()
    setattr(config, field, value)
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays)
    with pytest.raises(ValueError, match="GLM5Next"):
        Glm5NextVisionGGUF(config, gguf_tensor_index([path]))


def test_quantized_projector_is_rejected(tmp_path):
    config, arrays, _ = vision_fixture()
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays)
    tensors = gguf_tensor_index([path])
    name = "mm.down.weight"
    tensors[name] = SimpleNamespace(
        shape=tensors[name].shape, tensor_type=gguf.GGMLQuantizationType.Q8_0
    )
    with pytest.raises(ValueError, match="dense mmproj required"):
        Glm5NextVisionGGUF(config, tensors)


class UnreadablePayload:
    def __init__(self, tensor):
        self.shape = tensor.shape
        self.tensor_type = tensor.tensor_type

    @property
    def data(self):
        raise AssertionError("Skipped vision payload was touched")


@pytest.mark.parametrize(
    "names,target",
    [
        (
            ("v.patch_embd.weight", "v.patch_embd.weight.1"),
            "visual.patch_embed.proj.weight",
        ),
        (
            ("v.blk.1.ffn_gate.bias", "v.blk.1.ffn_up.bias"),
            "visual.blocks.1.mlp.gate_up_proj.bias",
        ),
        (("mm.gate.weight", "mm.up.weight"), "visual.merger.gate_up_proj.weight"),
    ],
)
def test_filtered_loading_preserves_complete_groups_only(tmp_path, names, target):
    config, arrays, expected = vision_fixture()
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, arrays)
    tensors = gguf_tensor_index([path])
    guarded = {
        name: tensor if name in names else UnreadablePayload(tensor)
        for name, tensor in tensors.items()
    }
    adapter = Glm5NextVisionGGUF(config, guarded)
    selected = dict.fromkeys(names, target)
    actual = dict(adapter.weights(torch.float32, selected))
    assert list(actual) == [target]
    torch.testing.assert_close(actual[target], expected[target], rtol=0, atol=0)
    assert list(adapter.weights(torch.float32, {})) == []
    with pytest.raises(ValueError, match="retain complete group"):
        list(adapter.weights(torch.float32, {names[0]: target}))
    with pytest.raises(ValueError, match="unknown/changed names"):
        list(adapter.weights(torch.float32, {names[0]: "visual.wrong"}))
