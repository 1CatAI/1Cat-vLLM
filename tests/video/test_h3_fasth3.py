# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastH3 reconstruction, native TP loading, staging and serving contracts."""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.model_executor.models.minimax_h3.config import (
    H3Config,
    H3InputError,
    sampling_for_deployment,
)
from vllm.model_executor.models.minimax_h3.fasth3 import FastH3Fusion, FastH3Spec
from vllm.model_executor.models.minimax_h3.lora import (
    inspect_adapter,
    select_adapter_file,
    validate_adapter_sampling,
)
from vllm.model_executor.models.minimax_h3.pipeline import MiniMaxH3Pipeline


def artifact(tmp_path, *, mutate=None, metadata_updates=None, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(519)

    def rand(*shape):
        return (torch.randn(*shape, generator=generator) * 0.03).to(dtype)

    tensors, base, deltas = {}, {}, {}
    for prefix, native, count in (
        ("transformer_blocks", "blocks", 50),
        ("token_refiner.refiner_blocks", "token_refiner.blocks", 2),
    ):
        for index in range(count):
            key = f"{native}.{index}.norm1.weight"
            base[key] = rand(8)
            tensors[f"{prefix}.{index}.norm1.diff"] = rand(8)
            deltas[key] = tensors[f"{prefix}.{index}.norm1.diff"].float()
    qkv = torch.empty(24, 8)
    for slot, label in enumerate(("q", "k", "v")):
        a, b = rand(64, 8), rand(8, 64)
        tensors[f"transformer_blocks.0.attn.to_{label}.lora_A.weight"] = a
        tensors[f"transformer_blocks.0.attn.to_{label}.lora_B.weight"] = b
        projection = b.float() @ a.float()
        for head in range(4):
            qkv[6 * head + 2 * slot : 6 * head + 2 * slot + 2] = projection[
                2 * head : 2 * head + 2
            ]
    key = "blocks.0.attn.qkv_proj.weight"
    base[key], deltas[key] = rand(24, 8), qkv
    for exported, native, outputs in (
        ("transformer_blocks.0.ff.net.0.proj", "blocks.0.mlp.fc1", 16),
        ("transformer_blocks.0.ff.net.2", "blocks.0.mlp.fc2", 8),
        ("transformer_blocks.0.attn.to_out.0", "blocks.0.attn.out_proj", 8),
        ("transformer_blocks.0.adaln_proj.linear", "blocks.0.adaln_proj.linear", 16),
        ("proj_in", "video_patch_proj", 8),
    ):
        a, b = rand(64, 8), rand(outputs, 64)
        tensors[exported + ".lora_A.weight"] = a
        tensors[exported + ".lora_B.weight"] = b
        delta = b.float() @ a.float()
        if native.endswith("fc1"):
            delta = delta.roll(outputs // 2, dims=0)
        key = native + ".weight"
        base[key], deltas[key] = rand(outputs, 8), delta
    # The projection combines a low-rank update with a full-rank residual.
    tensors["proj_in.diff"] = rand(8, 8)
    deltas["video_patch_proj.weight"] += tensors["proj_in.diff"].float()
    tensors["proj_in.diff_b"] = rand(8)
    base["video_patch_proj.bias"] = rand(8)
    deltas["video_patch_proj.bias"] = tensors["proj_in.diff_b"].float()
    base["untouched.weight"] = rand(3)
    expected = {
        name: (weight.float() + deltas[name]).to(dtype) if name in deltas else weight
        for name, weight in base.items()
    }
    if mutate:
        mutate(tensors)
    metadata = {
        "format": "fastvideo-lora-v2",
        "finetuned_model": "FastVideo/FastVideo-FastH3-Dense-4-step-v1",
        "base_model": "MiniMaxAI/MiniMax-H3",
        "rank": "64",
        "low_rank_tensors": str(
            sum(name.endswith((".lora_A.weight", ".lora_B.weight")) for name in tensors)
        ),
        "diff_tensors": str(
            sum(name.endswith((".diff", ".diff_b")) for name in tensors)
        ),
        "set_weight_tensors": "0",
        **(metadata_updates or {}),
    }
    path = tmp_path / "adapter_model.safetensors"
    save_file(tensors, path, metadata=metadata)
    return path, base, expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fusion_preserves_release_rounding_and_consumes_all_edits(tmp_path, dtype):
    path, base, expected = artifact(tmp_path, dtype=dtype)
    fusion = FastH3Fusion(path, head_dim=2)
    actual = dict(fusion.apply(base.items()))
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
    assert actual["untouched.weight"] is base["untouched.weight"]
    fusion.validate_fully_applied(actual)
    with pytest.raises(H3InputError, match="single-use"):
        dict(fusion.apply(base.items()))
    with pytest.raises(H3InputError, match="did not reach"):
        fusion.validate_fully_applied(set(actual) - {"video_patch_proj.bias"})


def test_fusion_does_not_modify_mapped_fp32_delta(tmp_path):
    from safetensors import safe_open

    path, base, _ = artifact(tmp_path, dtype=torch.float32)
    fusion = FastH3Fusion(path, head_dim=2)
    key = "transformer_blocks.1.norm1.diff"
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        original = checkpoint.get_tensor(key).clone()
        fusion._fuse(checkpoint, "blocks.1.norm1.weight", base["blocks.1.norm1.weight"])
        torch.testing.assert_close(checkpoint.get_tensor(key), original, atol=0, rtol=0)


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires GPU")
def test_cuda_fusion_matches_cpu_oracle_and_returns_host_weights(tmp_path):
    path, base, expected = artifact(tmp_path)
    actual = dict(FastH3Fusion(path, head_dim=2, device="cuda").apply(base.items()))
    for name, value in actual.items():
        assert value.device.type == "cpu"
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "update",
    [
        {"format": "peft"},
        {"finetuned_model": "FastVideo/FastH3-VSA"},
        {"base_model": "another/model"},
        {"rank": "128"},
        {"low_rank_tensors": "0"},
        {"diff_tensors": "bad"},
        {"set_weight_tensors": "50"},
    ],
)
def test_bad_metadata_fails_before_fusion(tmp_path, update):
    path, _, _ = artifact(tmp_path, metadata_updates=update)
    with pytest.raises(H3InputError):
        inspect_adapter(path, "fl2va")


@pytest.mark.parametrize(
    "failure",
    ["unpaired", "block", "qkv", "role", "vsa", "dtype", "rank", "fused_diff"],
)
def test_invalid_tensor_inventory_is_rejected(tmp_path, failure):
    def mutate(tensors):
        prefix = "transformer_blocks.0.attn.to_q"
        if failure == "unpaired":
            tensors.pop(prefix + ".lora_A.weight")
        elif failure == "block":
            tensors.pop("transformer_blocks.49.norm1.diff")
        elif failure == "qkv":
            tensors.pop(prefix + ".lora_A.weight")
            tensors.pop(prefix + ".lora_B.weight")
        elif failure == "role":
            tensors["unknown.weight"] = torch.zeros(8)
        elif failure == "vsa":
            tensors["transformer_blocks.0.attn.to_gate_compress.set_weight"] = (
                torch.zeros(8, 8)
            )
        elif failure == "dtype":
            tensors["transformer_blocks.49.norm1.diff"] = torch.zeros(
                8, dtype=torch.int8
            )
        elif failure == "rank":
            tensors[prefix + ".lora_A.weight"] = torch.zeros(32, 8)
        else:
            tensors[prefix + ".diff"] = torch.zeros(8, 8)

    path, _, _ = artifact(tmp_path, mutate=mutate)
    with pytest.raises(H3InputError):
        inspect_adapter(path, "fl2va")


@pytest.mark.parametrize("failure", ["missing", "shape", "int8", "nan", "duplicate"])
def test_base_stream_errors_fail_closed(tmp_path, failure):
    path, base, _ = artifact(tmp_path)
    key = "blocks.0.adaln_proj.linear.weight"
    if failure == "missing":
        base.pop(key)
    elif failure == "shape":
        base[key] = torch.zeros(16, 4)
    elif failure == "int8":
        base[key] = torch.zeros(16, 8, dtype=torch.int8)
    elif failure == "nan":
        base[key].fill_(float("nan"))
    stream = list(base.items())
    if failure == "duplicate":
        stream.append((key, base[key]))
    with pytest.raises(H3InputError):
        dict(FastH3Fusion(path, head_dim=2).apply(stream))


def test_deployment_schedule_and_fused_request_restrictions(tmp_path):
    path, _, _ = artifact(tmp_path)
    assert select_adapter_file(tmp_path) == path
    config = H3Config(lora_path=str(path))
    params = sampling_for_deployment(config)
    spec = inspect_adapter(path, "fl2va")
    assert isinstance(spec, FastH3Spec)
    assert params.num_inference_steps == 4
    assert params.extra_args == {"flow_shift": 12.0, "audio_flow_shift": 3.0}
    pipeline = SimpleNamespace(turbo_spec=spec, _base_schedule_for_task=lambda _: None)
    assert MiniMaxH3Pipeline._resolve_sigma_positions(pipeline, "t2va", params) == (
        spec.base_schedule,
        4,
    )
    for updates in (
        {"num_inference_steps": 5},
        {"lora_scale": 0},
        {"lora_scale": 0.5},
        {"extra_args": {"flow_shift": 6}},
    ):
        with pytest.raises(H3InputError):
            MiniMaxH3Pipeline._resolve_sigma_positions(
                pipeline, "t2va", replace(params, **updates)
            )
    with pytest.raises(H3InputError, match="supports"):
        validate_adapter_sampling(spec, "fl2va", params)
    with pytest.raises(H3InputError, match="FL2VA"):
        inspect_adapter(path, "ref2va")
    with pytest.raises(H3InputError, match="original weights"):
        sampling_for_deployment(replace(config, transformer_path="int8.safetensors"))
    with pytest.raises(H3InputError, match="fused"):
        sampling_for_deployment(config, lora_scale=0)


def test_fasth3_api_defaults_and_unavailable_dynamic_lora(tmp_path):
    from fastapi.testclient import TestClient
    from test_h3_omni_api import RecordingEngine

    from vllm.video.server import create_app

    path, _, _ = artifact(tmp_path)
    config = H3Config(lora_path=str(path))
    engine = RecordingEngine(config)
    with TestClient(
        create_app(config, tmp_path / "outputs", engine_factory=lambda _: engine)
    ) as client:
        assert client.post("/v1/videos/sync", json={"task": "t2va"}).status_code == 200
        assert engine.requests[-1].sampling.num_inference_steps == 4
        for payload in (
            {"lora_scale": 0},
            {"num_inference_steps": 5},
            {"task": "fl2va"},
            {"lora": {"path": str(path), "scale": 1}},
        ):
            response = client.post("/v1/videos", json=payload)
            assert response.status_code == 422, response.text


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_fused_weights_enter_native_tp_loaders_and_host_snapshot(tmp_path, tp):
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.models.minimax_h3.residency import PinnedModuleStager
    from vllm.model_executor.models.minimax_h3.transformer import MiniMaxH3DiTModel

    path, base, expected = artifact(tmp_path)
    fused = dict(FastH3Fusion(path, head_dim=2).apply(base.items()))
    # The native loader converts serialized [head, Q/K/V, channel] to Q/K/V.
    qkv = expected["blocks.0.attn.qkv_proj.weight"]
    runtime_qkv = torch.cat(
        [
            torch.cat(
                [
                    qkv[head * 6 + slot * 2 : head * 6 + slot * 2 + 2]
                    for head in range(4)
                ]
            )
            for slot in range(3)
        ]
    )
    for rank in range(tp):
        with ExitStack() as stack:
            stack.enter_context(
                set_current_vllm_config(
                    VllmConfig(
                        parallel_config=ParallelConfig(
                            tensor_parallel_size=tp, distributed_executor_backend="mp"
                        )
                    )
                )
            )
            for module in (
                "vllm.model_executor.layers.linear",
                "vllm.model_executor.parameter",
            ):
                stack.enter_context(
                    patch(module + ".get_tensor_model_parallel_rank", return_value=rank)
                )
                stack.enter_context(
                    patch(
                        module + ".get_tensor_model_parallel_world_size",
                        return_value=tp,
                    )
                )
            model = nn.Module()
            model.arch = SimpleNamespace(num_attention_heads=4, attention_head_dim=2)
            model.blocks = nn.ModuleList([nn.Module() for _ in range(50)])
            model.token_refiner = nn.Module()
            model.token_refiner.blocks = nn.ModuleList([nn.Module() for _ in range(2)])
            for block in list(model.blocks) + list(model.token_refiner.blocks):
                block.norm1 = nn.LayerNorm(8, bias=False, dtype=torch.float16)
            block = model.blocks[0]
            block.attn, block.mlp, block.adaln_proj = (
                nn.Module(),
                nn.Module(),
                nn.Module(),
            )
            block.attn.qkv_proj = MergedColumnParallelLinear(
                8, [8, 8, 8], bias=False, params_dtype=torch.float16
            )
            block.attn.out_proj = RowParallelLinear(
                8, 8, bias=False, params_dtype=torch.float16
            )
            block.mlp.fc1 = MergedColumnParallelLinear(
                8, [8, 8], bias=False, params_dtype=torch.float16
            )
            block.mlp.fc2 = RowParallelLinear(
                8, 8, bias=False, params_dtype=torch.float16
            )
            block.adaln_proj.linear = ColumnParallelLinear(
                8, 16, bias=False, params_dtype=torch.float16
            )
            model.video_patch_proj = ColumnParallelLinear(
                8, 8, bias=True, params_dtype=torch.float32
            )
            loaded = MiniMaxH3DiTModel.load_weights(model, fused.items())
            assert loaded == set(fused) - {"untouched.weight"}
            for name, param in model.named_parameters():
                value = expected[name]
                if name.endswith("qkv_proj.weight"):
                    value = torch.cat(
                        [part.chunk(tp)[rank] for part in runtime_qkv.chunk(3)]
                    )
                elif name.endswith("fc1.weight"):
                    value = torch.cat([part.chunk(tp)[rank] for part in value.chunk(2)])
                elif name.endswith(("out_proj.weight", "fc2.weight")):
                    value = value.chunk(tp, dim=1)[rank]
                elif ".norm1." not in name:
                    value = value.chunk(tp)[rank]
                torch.testing.assert_close(param, value.to(param.dtype), atol=0, rtol=0)
            # Exercise the actual CPU snapshot and restore without CUDA streams.
            stager = object.__new__(PinnedModuleStager)
            stager._groups = stager._snapshot_groups((model,), pin_memory=False)
            before = {
                name: param.detach().clone() for name, param in model.named_parameters()
            }
            for param in model.parameters():
                param.data = torch.zeros_like(param)
            stager._restore_masters()
            for name, param in model.named_parameters():
                torch.testing.assert_close(param, before[name], atol=0, rtol=0)
