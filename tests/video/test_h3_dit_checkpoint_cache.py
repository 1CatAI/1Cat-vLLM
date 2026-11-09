# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact checkpoint reuse through H3's actual QKV loader and post-load methods."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.minimax_h3.prepared_weights import (
    load_transformer_checkpoint,
)
from vllm.model_executor.models.minimax_h3.quantization import (
    DiffusionInt8ConvRotConfig,
    FP16LinearMethod,
    Int8ConvRotLayerConfig,
    Int8ConvRotLinearMethod,
)
from vllm.model_executor.models.minimax_h3.transformer import MiniMaxH3DiTModel
from vllm.model_executor.parameter import ChannelQuantScaleParameter

PREFIX = "blocks.0.attn.qkv_proj"


@pytest.fixture(autouse=True)
def single_rank_parameters(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )


class CheckpointModel(nn.Module):
    load_weights = MiniMaxH3DiTModel.load_weights
    post_load_weights = MiniMaxH3DiTModel.post_load_weights

    def __init__(self, dtype, layout):
        super().__init__()
        self.arch = SimpleNamespace(
            num_attention_heads=4, attention_head_dim=64, adaln_curve_grid=None
        )
        block = nn.Module()
        block.attn = nn.Module()
        layer = nn.Module()
        layer.register_parameter(
            "weight", nn.Parameter(torch.empty(768, 256, dtype=dtype), False)
        )
        if dtype == torch.int8:
            layer.register_parameter(
                "weight_scale",
                ChannelQuantScaleParameter(
                    data=torch.empty(768, 1),
                    output_dim=0,
                    weight_loader=default_weight_loader,
                ),
            )
            layer.quant_method = Int8ConvRotLinearMethod(
                DiffusionInt8ConvRotConfig(weight_layout=layout),
                Int8ConvRotLayerConfig(True),
                prefix=PREFIX,
            )
        else:
            layer.quant_method = FP16LinearMethod()
        layer.h3_fp16_weight_layout = layout
        block.attn.qkv_proj = layer
        self.blocks = nn.ModuleList([block])
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(64, dtype=torch.float32))

    @property
    def projection(self):
        return self.blocks[0].attn.qkv_proj

    def prepare(self):
        self.projection.quant_method.process_weights_after_loading(self.projection)
        self.post_load_weights()


def checkpoint(tmp_path, dtype):
    path = tmp_path / "model.safetensors"
    values = torch.arange(768 * 256).reshape(768, 256) % 127 - 63
    weights = {
        PREFIX + ".weight": values.to(dtype)
        if dtype == torch.int8
        else values.to(dtype) / 128,
        "rope.inv_freq": torch.arange(64, dtype=torch.float32) / 123,
    }
    if dtype == torch.int8:
        # The real loader normalizes checkpoint [N] scales to constructor [N,1]
        # and reorders them with the grouped QKV weights.
        weights[PREFIX + ".weight_scale"] = torch.linspace(0.001, 0.02, 768)
    save_file(weights, path)
    return path


def load(target, path, root, *, enabled=True, **kwargs):
    def weights():
        target.checkpoint_reads = getattr(target, "checkpoint_reads", 0) + 1
        yield from load_file(path).items()

    return load_transformer_checkpoint(
        target,
        weights(),
        [path],
        enabled=enabled,
        root=root,
        rank=0,
        world_size=1,
        options={"layout": target.projection.h3_fp16_weight_layout},
        reserve_bytes=0,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["row", "column"])
def test_dit_cold_and_warm_preserve_qkv_layout_scales_output_and_rng(
    tmp_path, dtype, layout
):
    path = checkpoint(tmp_path, dtype)
    root = tmp_path / "cache"
    control = CheckpointModel(dtype, layout)
    assert load(control, path, root, enabled=False) is None
    control.prepare()
    inputs = torch.arange(512, dtype=torch.float16).reshape(2, 256) / 512
    expected = control.projection.quant_method.apply(control.projection, inputs)

    for cold in (True, False):
        target = CheckpointModel(dtype, layout)
        rng = torch.get_rng_state().clone()
        cache = load(target, path, root)
        assert cache is not None
        assert torch.equal(torch.get_rng_state(), rng)
        assert getattr(target, "checkpoint_reads", 0) == int(cold)
        if dtype == torch.int8:
            assert target.projection.weight_scale.shape == (768, 1)
        target.prepare()  # Required on both arms, including the warm cache hit.
        for name, tensor in target.state_dict().items():
            reference = control.state_dict()[name]
            assert tensor.dtype == reference.dtype
            assert tensor.stride() == reference.stride()
            assert torch.equal(tensor, reference)
        assert torch.equal(
            target.projection.quant_method.apply(target.projection, inputs), expected
        )
        with torch.no_grad():
            target.projection.weight.add_(1)  # Cannot modify the shared snapshot.
        cache.close()


@pytest.mark.parametrize("adapter", ["fusion", "adaln"])
def test_dit_adapter_transforms_use_ordinary_loader_and_validate(tmp_path, adapter):
    path = checkpoint(tmp_path, torch.int8)
    root = tmp_path / "cache"
    target = CheckpointModel(torch.int8, "column")
    fusion = Mock() if adapter == "fusion" else None
    assert (
        load(target, path, root, fusion=fusion, restore_adaln=adapter == "adaln")
        is None
    )
    assert target.checkpoint_reads == 1
    if fusion is not None:
        fusion.validate_fully_applied.assert_called_once()
        assert "rope.inv_freq" in fusion.validate_fully_applied.call_args.args[0]
    target.prepare()
    assert not root.exists()


def test_dit_missing_required_buffer_never_publishes(tmp_path):
    path = checkpoint(tmp_path, torch.int8)
    weights = load_file(path)
    del weights["rope.inv_freq"]
    save_file(weights, path)
    root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="missing tensors.*rope.inv_freq"):
        load(CheckpointModel(torch.int8, "row"), path, root)
    assert not list(root.glob("entry-*/ready.json"))


def test_dit_warm_hit_still_rejects_invalid_quantization_scales(tmp_path):
    path = checkpoint(tmp_path, torch.int8)
    weights = load_file(path)
    weights[PREFIX + ".weight_scale"].zero_()
    save_file(weights, path)
    root = tmp_path / "cache"
    for cold in (True, False):
        target = CheckpointModel(torch.int8, "column")
        cache = load(target, path, root)
        assert getattr(target, "checkpoint_reads", 0) == int(cold)
        try:
            with pytest.raises(ValueError, match="non-positive INT8 weight scales"):
                target.prepare()
        finally:
            cache.close()
