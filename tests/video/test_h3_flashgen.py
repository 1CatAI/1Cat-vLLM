# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashGen metadata, native TP algebra and original AdaLN restoration."""

import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch.nn.functional import linear

from vllm.model_executor.models.minimax_h3 import flashgen
from vllm.model_executor.models.minimax_h3.config import (
    H3Config,
    H3InputError,
    sampling_for_deployment,
)
from vllm.model_executor.models.minimax_h3.lora import (
    inspect_adapter,
    select_adapter_file,
    validate_adapter_sampling,
)
from vllm.model_executor.models.minimax_h3.pipeline import MiniMaxH3Pipeline

METADATA = {
    "key_format": "minimax-h3-native",
    "qkv_layout": "grouped",
    "lora_rank": "64",
    "lora_alpha": "64",
    "tasks": "t2va",
    "base_schedule": "1.0,0.7,0.4,0.15,0.0",
}


def flashgen_manifest(tmp_path, *, metadata=None, omit=False, bad_shape=False):
    header = {"__metadata__": METADATA if metadata is None else metadata}
    offset = 0
    for name, (inputs, outputs) in flashgen._TARGETS.items():
        for side, shape in (("A", [64, inputs]), ("B", [outputs, 64])):
            if omit and name == "blocks.0.attn.qkv_proj" and side == "B":
                continue
            if bad_shape and name == "blocks.0.adaln_proj.linear" and side == "A":
                shape[1] = 8
            size = shape[0] * shape[1] * 2
            header[f"transformer.{name}.lora_{side}.default.weight"] = {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            offset += size
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    path = tmp_path / flashgen.FLASHGEN_FILENAME
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        stream.truncate(8 + len(data) + offset)
    return path


def test_flashgen_defaults_are_four_intervals_and_metadata_schedule(tmp_path):
    path = flashgen_manifest(tmp_path)
    config = H3Config(lora_path=str(path))
    spec = inspect_adapter(path, "fl2va")
    assert len(flashgen._TARGETS) == 259
    assert spec.base_schedule == (1.0, 0.7, 0.4, 0.15, 0.0)
    params = sampling_for_deployment(config)
    assert params.num_inference_steps == 4
    pipeline = SimpleNamespace(turbo_spec=spec, _base_schedule_for_task=lambda _: None)
    assert MiniMaxH3Pipeline._resolve_sigma_positions(pipeline, "t2va", params) == (
        spec.base_schedule,
        4,
    )
    base = sampling_for_deployment(config, lora_scale=0)
    assert base.num_inference_steps == 50
    assert MiniMaxH3Pipeline._resolve_sigma_positions(pipeline, "fl2va", base) == (
        None,
        50,
    )
    with pytest.raises(H3InputError, match="actual denoiser calls"):
        validate_adapter_sampling(spec, "t2va", replace(params, num_inference_steps=5))
    with pytest.raises(H3InputError, match="supports"):
        validate_adapter_sampling(spec, "fl2va", params)
    with pytest.raises(H3InputError, match="FL2VA"):
        inspect_adapter(path, "ref2va")


@pytest.mark.parametrize(
    "key,value",
    [
        ("key_format", "minimax-h3-diffusers"),
        ("qkv_layout", "qkv"),
        ("lora_rank", "128"),
        ("lora_alpha", "nan"),
        ("tasks", "fl2va"),
        ("base_schedule", "1,0.5,0"),
        ("base_schedule", "1,0.7,0.4,0.15,nan"),
        ("base_schedule", "1,0.4,0.7,0.15,0"),
    ],
)
def test_flashgen_rejects_wrong_metadata(tmp_path, key, value):
    path = flashgen_manifest(tmp_path, metadata={**METADATA, key: value})
    with pytest.raises(H3InputError):
        inspect_adapter(path, "fl2va")


@pytest.mark.parametrize("option", ["omit", "bad_shape"])
def test_flashgen_requires_complete_unpruned_adapter_shape(tmp_path, option):
    path = flashgen_manifest(tmp_path, **{option: True})
    with pytest.raises(H3InputError):
        inspect_adapter(path, "fl2va")


def test_mixed_adapter_directory_requires_explicit_file(tmp_path):
    from test_h3_lora import _manifest

    flash = flashgen_manifest(tmp_path)
    assert select_adapter_file(tmp_path) == flash
    _manifest(tmp_path)
    with pytest.raises(H3InputError, match="exactly one"):
        select_adapter_file(tmp_path)


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize(
    "name",
    ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2", "adaln_proj.linear"],
)
def test_native_delta_matches_global_algebra_after_tp_layout(tp, name):
    generator = torch.Generator().manual_seed(317)
    inputs, outputs, rank = (8, 24, 3) if name == "attn.qkv_proj" else (8, 16, 3)
    a = torch.randn(rank, inputs, generator=generator)
    b = torch.randn(outputs, rank, generator=generator)
    x = torch.randn(7, inputs, generator=generator)
    native_b = b
    if name == "attn.qkv_proj":
        native_b = torch.cat(
            [part.reshape(8, rank) for part in b.reshape(4, 3, 2, rank).unbind(1)]
        )
    expected = linear(linear(x, a), native_b)
    local_outputs = []
    for tp_rank in range(tp):
        shard_a, shard_b = flashgen.shard_native_pair(
            a, b, name, tp_rank, tp, heads=4, head_dim=2
        )
        local_x = (
            x.chunk(tp, dim=1)[tp_rank] if name in ("attn.out_proj", "mlp.fc2") else x
        )
        local_outputs.append(linear(linear(local_x, shard_a), shard_b))
    if name in ("attn.out_proj", "mlp.fc2"):
        actual = sum(local_outputs)
    elif name in ("attn.qkv_proj", "mlp.fc1"):
        parts = 3 if name == "attn.qkv_proj" else 2
        actual = torch.cat(
            [
                torch.cat(
                    [out.chunk(parts, dim=1)[part] for out in local_outputs], dim=1
                )
                for part in range(parts)
            ],
            dim=1,
        )
    else:
        actual = torch.cat(local_outputs, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_restore_keeps_int8_backbone_and_uses_original_adaln(tmp_path, monkeypatch):
    monkeypatch.setattr(flashgen, "_TARGETS", {"blocks.0.adaln_proj.linear": (12, 24)})
    originals = {
        "blocks.0.adaln_proj.linear.weight": torch.randn(24, 12),
        "blocks.0.adaln_proj.linear.bias": torch.randn(24),
        **{
            f"time_embedder.{name}.{side}": torch.randn(4)
            for name in ("proj_in", "proj_out")
            for side in ("weight", "bias")
        },
        "blocks.0.attn.qkv_proj.weight": torch.randn(24, 12),
    }
    save_file(originals, tmp_path / "model.safetensors")
    int8 = torch.randint(-128, 128, (24, 12), dtype=torch.int8)
    scale = torch.randn(24)
    base = [
        ("adaln_t_table", torch.randn(7, 8)),
        ("blocks.0.adaln_proj.linear.weight", torch.randn(24, 8)),
        ("blocks.0.adaln_proj.linear.bias", torch.randn(24)),
        ("blocks.0.attn.qkv_proj.weight", int8),
        ("blocks.0.attn.qkv_proj.weight_scale", scale),
    ]
    restored = dict(flashgen.restore_dense_adaln_weights(iter(base), tmp_path))
    assert "adaln_t_table" not in restored
    assert restored["blocks.0.attn.qkv_proj.weight"] is int8
    assert restored["blocks.0.attn.qkv_proj.weight_scale"] is scale
    for name, weight in originals.items():
        if name != "blocks.0.attn.qkv_proj.weight":
            torch.testing.assert_close(restored[name], weight, rtol=0, atol=0)
    originals.pop("time_embedder.proj_in.weight")
    save_file(originals, tmp_path / "model.safetensors")
    with pytest.raises(H3InputError, match="missing AdaLN"):
        dict(flashgen.restore_dense_adaln_weights(iter(base), tmp_path))


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_installer_binds_every_native_target_and_preserves_zero_scale(
    tp, tmp_path, monkeypatch
):
    from torch import nn

    from vllm.model_executor.models.minimax_h3.lora import lora_scale

    dims = {
        name: (8, 24 if name.endswith("qkv_proj") else 16) for name in flashgen._TARGETS
    }
    monkeypatch.setattr(flashgen, "_TARGETS", dims)
    generator = torch.Generator().manual_seed(73)
    tensors = {
        f"transformer.{name}.lora_{side}.default.weight": torch.randn(
            shape, generator=generator
        )
        * 0.01
        for name, (inputs, outputs) in dims.items()
        for side, shape in (("A", (64, inputs)), ("B", (outputs, 64)))
    }
    path = tmp_path / flashgen.FLASHGEN_FILENAME
    save_file(tensors, path, metadata=METADATA)

    class BaseMethod:
        def apply(self, layer, x, bias=None):
            return linear(x.float(), layer.weight.float())

    layers = {}
    for name, (inputs, outputs) in dims.items():
        layer = nn.Module()
        is_row = name.endswith(("attn.out_proj", "mlp.fc2"))
        layer.weight = nn.Parameter(
            torch.randn(
                outputs if is_row else outputs // tp,
                inputs // tp if is_row else inputs,
                generator=generator,
            )
        )
        layer.tp_rank, layer.tp_size = tp - 1, tp
        layer.quant_method = BaseMethod()
        layers[name] = layer
    model = SimpleNamespace(
        named_modules=lambda: iter(layers.items()),
        arch=SimpleNamespace(num_attention_heads=4, attention_head_dim=2),
    )
    flashgen.install_flashgen_lora(model, path, "fl2va")
    assert len(layers) == 259
    for name, layer in layers.items():
        assert set(dict(layer.named_buffers())) == {"h3_lora_a_0", "h3_lora_b_0"}
        assert layer._sm70_f16_forbidden
        x = torch.randn(3, layer.weight.shape[1], generator=generator)
        base = linear(x, layer.weight)
        token = lora_scale.set(0)
        try:
            torch.testing.assert_close(
                layer.quant_method.apply(layer, x), base, rtol=0, atol=0
            )
            lora_scale.set(0.5)
            expected = base + 0.5 * linear(
                linear(x, layer.h3_lora_a_0.float()), layer.h3_lora_b_0.float()
            )
            torch.testing.assert_close(layer.quant_method.apply(layer, x), expected)
        finally:
            lora_scale.reset(token)


def test_flashgen_http_accepts_four_and_rejects_lightx2v_count(tmp_path):
    from fastapi.testclient import TestClient
    from test_h3_omni_api import RecordingEngine

    from vllm.video.server import create_app

    path = flashgen_manifest(tmp_path)
    config = H3Config(lora_path=str(path))
    engine = RecordingEngine(config)
    with TestClient(
        create_app(config, tmp_path / "outputs", engine_factory=lambda _: engine)
    ) as client:
        assert client.post("/v1/videos/sync", json={"task": "t2va"}).status_code == 200
        assert engine.requests[-1].sampling.num_inference_steps == 4
        assert (
            client.post("/v1/videos", json={"num_inference_steps": 5}).status_code
            == 422
        )
        assert client.post("/v1/videos/sync", json={"lora_scale": 0}).status_code == 200
        assert engine.requests[-1].sampling.num_inference_steps == 50
