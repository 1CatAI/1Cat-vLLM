# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact ConvRot-before-gather regression on TP4, with real INT8 projections."""

from types import SimpleNamespace

import pytest
import torch

from tests.video import test_h3_residual_parallel as residual_tests
from vllm.model_executor.models.minimax_h3 import transformer
from vllm.model_executor.models.minimax_h3.quantization import (
    DiffusionInt8ConvRotConfig,
    H3QKVParallelLinear,
    Int8ConvRotLayerConfig,
    Int8ConvRotLinearMethod,
    rotate_local_fp16,
)

tp4_group = residual_tests.tp4_group


def test_cpu_rotation_preserves_original_path_and_rejects_pre_rotated_input():
    method = Int8ConvRotLinearMethod(
        DiffusionInt8ConvRotConfig(), Int8ConvRotLayerConfig(True), prefix="blocks.0"
    )
    layer = SimpleNamespace(quant_method=method, bias=None, gather_output=False)
    x = torch.randn(17, 256, dtype=torch.float16)
    actual, rotated = rotate_local_fp16(layer, x)
    assert actual is x and not rotated
    with pytest.raises(ValueError, match="Pre-rotated H3"):
        H3QKVParallelLinear.forward(layer, x, input_is_rotated=True)


@torch.inference_mode()
def test_gpu_local_rotation_preserves_tp4_blocks_and_hooks(tp4_group, monkeypatch):
    # Keep both backends in one distributed fixture lifetime, as with the
    # residual regression: repository fixtures tear down state after each test.
    for backend in ("FLASH_ATTN_V100", "FLASHINFER_SM70"):
        _check_local_rotation(tp4_group, monkeypatch, backend)


def _check_local_rotation(tp4_group, monkeypatch, backend):
    from vllm.model_executor.models.minimax_h3.attention import attention_backend
    from vllm.model_executor.models.minimax_h3.cuda_ops import w8a16_extension
    from vllm.video.metrics import DenoiseWorkCounter

    group = tp4_group
    names = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    quant = DiffusionInt8ConvRotConfig(
        layer_configs={
            f"blocks.0.{name}": {"format": "int8_tensorwise", "convrot": True}
            for name in names
        }
    )
    arch = transformer.MiniMaxH3DiTArchConfig(
        hidden_size=512,
        num_attention_heads=8,
        ffn_hidden_size=1024,
        adaln_curve_grid=2,
        adaln_out_features=18 * 512,
    )
    token = attention_backend.set(backend)
    try:
        block = (
            transformer.MiniMaxH3DiTBlock(
                arch, quant, prefix="blocks.0", residual_sequence_parallel=True
            )
            .cuda()
            .eval()
        )
    finally:
        attention_backend.reset(token)
    torch.manual_seed(1234 + group.rank_in_group)
    for name, parameter in block.named_parameters():
        if parameter.dtype == torch.int8:
            parameter.random_(-7, 8)
        elif name.endswith("weight_scale"):
            parameter.fill_(0.005)
        elif "norm" in name:
            parameter.fill_(1)
        else:
            parameter.normal_(0, 0.03)
    for layer in block.modules():
        method = getattr(layer, "quant_method", None)
        if isinstance(method, Int8ConvRotLinearMethod):
            method.process_weights_after_loading(layer)
    original = rotate_local_fp16
    observed = []

    def checked_rotation(layer, values):
        result, selected = original(layer, values)
        assert selected
        observed.append(tuple(values.shape))
        return result, selected

    for cached in (False, True):
        if cached:
            for layer in block.modules():
                if isinstance(
                    getattr(layer, "quant_method", None), Int8ConvRotLinearMethod
                ):
                    layer.h3_fp16_weight = w8a16_extension().dequantize(
                        layer.weight, layer.weight_scale
                    )
        for valid in (33, 131):
            torch.manual_seed(42)
            total = (valid + 3) // 4 * 4
            full = torch.randn(total, 512, device="cuda", dtype=torch.float32)
            full[::5, 0] = 70000
            local = full.chunk(4, dim=0)[group.rank_in_group].clone()
            kwargs = dict(
                t_emb=torch.randn(1, 8, device="cuda"),
                combined_indices=torch.arange(total, device="cuda") % 3,
                rope_table=torch.randn(total, 96, device="cuda"),
                cu_seqlens=torch.tensor([0, valid], device="cuda", dtype=torch.int32),
                max_seqlen=valid,
                packed_total=total,
            )
            outputs, counts = [], []
            for enabled in (False, True):
                observed.clear()
                monkeypatch.setattr(
                    transformer,
                    "rotate_local_fp16",
                    checked_rotation if enabled else lambda layer, x: (x, False),
                )
                counter = DenoiseWorkCounter(
                    block, used_length=valid, video_outputs=valid, audio_outputs=0
                )
                value = local.clone()
                try:
                    for _ in range(2):
                        value = block(value, **kwargs)
                    counts.append((counter.flops, dict(counter.by_layer)))
                finally:
                    counter.close()
                outputs.append(value)
                if enabled:
                    assert observed == [(total // 4, 512)] * 4
            assert counts[0] == counts[1]
            assert torch.isfinite(outputs[1]).all()
            assert outputs[1].abs().max() > torch.finfo(torch.float16).max
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
