# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract and algebra gates for native H3 Turbo, without model downloads."""

import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.functional import linear

from vllm.model_executor.models.minimax_h3.config import (
    H3Config,
    H3InputError,
    H3SamplingParams,
    sampling_for_deployment,
)
from vllm.model_executor.models.minimax_h3.lora import (
    TurboLinearMethod,
    _shard_pair,
    inspect_turbo_lora,
    lora_scale,
    parse_turbo_filename,
    select_turbo_file,
    validate_turbo_sampling,
)
from vllm.model_executor.models.minimax_h3.ops import convrot_reference
from vllm.model_executor.models.minimax_h3.pipeline import MiniMaxH3Pipeline

PUBLISHED = [
    ("fl2v_turbo_4step_v0.1", 4, 12),
    ("fl2v_turbo_4step_v1.0_768p_bf16", 4, 6),
    ("fl2v_turbo_4step_v1.1_768p_bf16", 4, 6),
    ("fl2v_turbo_4step_v1.2_768p_bf16", 4, 6),
    ("fl2v_turbo_8step_v1.0_bf16", 8, 12),
    ("fl2v_turbo_8step_v1.0_768p_bf16", 8, 6),
    ("ref2v_turbo_4step_v0.1_bf16", 4, 12),
    ("ref2v_turbo_8step_v1.0_768p_bf16", 8, 6),
]


@pytest.mark.parametrize("name,steps,shift", PUBLISHED)
def test_published_artifact_sampling(name, steps, shift):
    spec = parse_turbo_filename(f"minimax_h3_{name}.safetensors")
    assert spec.denoise_steps == steps
    assert spec.video_shift == shift
    task = "ref2va" if name.startswith("ref2v") else "t2va"
    params = H3SamplingParams(num_inference_steps=steps + 1)
    validate_turbo_sampling(spec, task, params)
    with pytest.raises(H3InputError, match="actual denoiser calls"):
        validate_turbo_sampling(spec, task, replace(params, num_inference_steps=steps))
    with pytest.raises(H3InputError, match="flow_shift"):
        validate_turbo_sampling(
            spec, task, replace(params, extra_args={"flow_shift": 9})
        )
    with pytest.raises(H3InputError, match="supports"):
        validate_turbo_sampling(spec, "t2va" if task == "ref2va" else "ref2va", params)


def _manifest(tmp_path, *, alpha="8", bad_shape=False, missing=False):
    # Sparse file: validate a real safetensors header with production dimensions
    # without allocating or reading 1.3 GB of synthetic weights.
    dims = {
        "attn.to_q": (5376, 7168),
        "attn.to_k": (5376, 7168),
        "attn.to_v": (5376, 7168),
        "attn.to_out.0": (7168, 5376),
        "ff.net.0.proj": (5376, 28672),
        "ff.net.2": (14336, 5376),
    }
    metadata = {} if alpha is None else {"alpha": alpha}
    header = {"__metadata__": metadata}
    offset = 0
    for prefix, count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    ):
        for index in range(count):
            for suffix, (inputs, outputs) in dims.items():
                for side, shape in (("A", [128, inputs]), ("B", [outputs, 128])):
                    if missing and index == 0 and suffix == "attn.to_q" and side == "A":
                        continue
                    if bad_shape and index == 0 and suffix == "attn.to_q":
                        shape[0] += 1
                    size = shape[0] * shape[1] * 2
                    header[f"{prefix}.{index}.{suffix}.lora_{side}.default.weight"] = {
                        "dtype": "BF16",
                        "shape": shape,
                        "data_offsets": [offset, offset + size],
                    }
                    offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
    with path.open("wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        out.truncate(8 + len(raw) + offset)
    return path


@pytest.mark.parametrize("alpha,expected", [(None, 8), ("8", 8), ("128", 128)])
def test_metadata_alpha_and_deployment_defaults(tmp_path, alpha, expected):
    path = _manifest(tmp_path, alpha=alpha)
    spec = inspect_turbo_lora(path, "fl2va")
    assert spec.alpha == expected
    config = H3Config(lora_path=str(path))
    assert sampling_for_deployment(config).num_inference_steps == 5
    assert sampling_for_deployment(config, lora_scale=0).num_inference_steps == 50
    assert (
        sampling_for_deployment(config, num_inference_steps=4).num_inference_steps == 4
    )
    with pytest.raises(H3InputError, match="partition=fl2va"):
        inspect_turbo_lora(path, "ref2va")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"alpha": "nan"}, "alpha"),
        ({"alpha": "-1"}, "alpha"),
        ({"bad_shape": True}, "shape"),
        ({"missing": True}, "tensor set"),
    ],
)
def test_bad_artifact_fails_before_tensor_loading(tmp_path, kwargs, match):
    with pytest.raises(H3InputError, match=match):
        inspect_turbo_lora(_manifest(tmp_path, **kwargs), "fl2va")


def test_ambiguous_directory_and_comfy_layout_rejected(tmp_path):
    path = _manifest(tmp_path)
    other = tmp_path / "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
    other.touch()
    with pytest.raises(H3InputError, match="exactly one"):
        select_turbo_file(tmp_path)
    path = path.rename(tmp_path / path.name.replace("_bf16", "_comfyui_bf16"))
    with pytest.raises(H3InputError, match="different layouts"):
        select_turbo_file(path)


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize(
    "suffix",
    [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ],
)
def test_tp_shards_reconstruct_global_lora_delta(tp, suffix):
    torch.manual_seed(19)
    x = torch.randn(7, 16, dtype=torch.float64)
    a = torch.randn(4, 16, dtype=torch.float64)
    b = torch.randn(24, 4, dtype=torch.float64)
    expected = linear(linear(x, a), b)
    outputs = []
    row_parallel = suffix in ("attn.to_out.0", "ff.net.2")
    for rank in range(tp):
        local_a, local_b, offset = _shard_pair(a, b, suffix, rank, tp)
        local_x = x.chunk(tp, dim=-1)[rank] if row_parallel else x
        outputs.append(linear(linear(local_x, local_a), local_b))
        if suffix in ("attn.to_q", "attn.to_k", "attn.to_v"):
            assert offset == ("attn.to_q", "attn.to_k", "attn.to_v").index(suffix) * (
                24 // tp
            )
    if row_parallel:
        actual = torch.stack(outputs).sum(0)
    elif suffix == "ff.net.0.proj":
        gate = torch.cat([o.chunk(2, dim=-1)[0] for o in outputs], dim=-1)
        value = torch.cat([o.chunk(2, dim=-1)[1] for o in outputs], dim=-1)
        actual = torch.cat((value, gate), dim=-1)
    else:
        actual = torch.cat(outputs, dim=-1)
    torch.testing.assert_close(actual, expected)


def test_int8_convrot_delta_uses_unrotated_input_and_scale_zero_is_exact():
    from vllm.model_executor.models.minimax_h3.quantization import (
        DiffusionInt8ConvRotConfig,
        Int8ConvRotLayerConfig,
        Int8ConvRotLinearMethod,
    )

    torch.manual_seed(42)
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.randint(-128, 128, (16, 256), dtype=torch.int8), requires_grad=False
    )
    layer.weight_scale = nn.Parameter(torch.full((16,), 0.001), requires_grad=False)
    layer.h3_output_fp32 = True
    a, b = torch.randn(8, 256).half() * 0.1, torch.randn(16, 8).half() * 0.1
    layer.register_buffer("h3_lora_a_0", a)
    layer.register_buffer("h3_lora_b_0", b)
    base = Int8ConvRotLinearMethod(
        DiffusionInt8ConvRotConfig(), Int8ConvRotLayerConfig(True), prefix="test"
    )
    method = TurboLinearMethod(base, [(0, 0, 16)], 8 / 128)
    x = torch.randn(3, 256).half()
    baseline = base.apply(layer, x)
    token = lora_scale.set(1.5)
    try:
        actual = method.apply(layer, x)
        expected = baseline + 1.5 * (8 / 128) * linear(
            linear(x.float(), a.float()), b.float()
        )
        torch.testing.assert_close(actual, expected)
        wrong = baseline + 1.5 * (8 / 128) * linear(
            linear(convrot_reference(x).float(), a.float()), b.float()
        )
        assert not torch.allclose(actual, wrong)
        lora_scale.set(0)
        assert torch.equal(method.apply(layer, x), baseline)
    finally:
        lora_scale.reset(token)


def test_four_updates_use_five_sigma_positions():
    spec = parse_turbo_filename(
        "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
    )
    deployment = SimpleNamespace(turbo_spec=spec)
    assert MiniMaxH3Pipeline._resolve_sigma_positions(
        deployment, "t2va", H3SamplingParams(num_inference_steps=5)
    ) == (None, 5)


@pytest.mark.parametrize("key", ["flow_shift", "audio_flow_shift"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_shifts_rejected(key, value):
    with pytest.raises(H3InputError, match="finite and positive"):
        H3SamplingParams(extra_args={key: value})
