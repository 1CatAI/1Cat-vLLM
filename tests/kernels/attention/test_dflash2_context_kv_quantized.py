# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused context K/V must survive a quantized draft checkpoint.

``_build_context_kv_buffers`` slices the K/V rows out of ``qkv_proj.weight``.
That attribute only holds the dense ``[N, K]`` matrix for an unquantized
layer; a quantized draft head keeps packed codes there (NVFP4 packs two
values per byte, so the tensor is ``[N, K // 2]``), and the fused matrix then
comes out half as wide -- ``F.linear`` fails with a shape mismatch and the
engine never finishes its profile run.

These tests use a minimal fake quantization method so they stay CPU-only and
do not need a checkpoint: what matters is that ``.weight`` is packed and that
the dense weight is reachable only through ``quant_method.apply``.
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

HIDDEN = 32
Q_SIZE = 16
KV_SIZE = 8
LAYERS = 3


class _FakePackedQuantMethod:
    """Stores the weight transposed and halved, reachable only via apply()."""

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.dense_weight, bias)


def _make_attn(dense_weight: torch.Tensor, quantized: bool) -> nn.Module:
    qkv = nn.Module()
    qkv.input_size_per_partition = HIDDEN
    qkv.dense_weight = dense_weight
    if quantized:
        # Packed codes under the same attribute name: half as wide, uint8.
        qkv.weight = torch.zeros(dense_weight.shape[0], HIDDEN // 2, dtype=torch.uint8)
        qkv.quant_method = _FakePackedQuantMethod()
    else:
        qkv.weight = dense_weight
        qkv.quant_method = UnquantizedLinearMethod()
    qkv.bias = None
    attn = nn.Module()
    attn.qkv_proj = qkv
    attn.q_size = Q_SIZE
    return attn


def _make_model(quantized: bool):
    torch.manual_seed(1234)
    weights = [
        torch.randn(Q_SIZE + 2 * KV_SIZE, HIDDEN, dtype=torch.float32)
        for _ in range(LAYERS)
    ]
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = nn.Module()
    model.hidden_norm.weight = nn.Parameter(torch.ones(HIDDEN), requires_grad=False)
    layers_attn = [_make_attn(w, quantized) for w in weights]
    for attn in layers_attn:
        attn.k_norm = nn.Module()
        attn.k_norm.weight = nn.Parameter(torch.ones(4), requires_grad=False)
    return model, layers_attn, weights


def _expected(weights: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([w[Q_SIZE:] for w in weights], dim=0)


def test_unquantized_draft_head_fuses_eagerly():
    """The dense path must keep fusing at build time, exactly as before."""
    model, layers_attn, weights = _make_model(quantized=False)
    model._build_context_kv_buffers(layers_attn, has_bias=False)
    assert model._fused_kv_weight is not None
    torch.testing.assert_close(model._fused_kv_weight, _expected(weights))


def test_quantized_draft_head_defers_then_fuses_dense():
    """The packed path defers, then rebuilds the SAME matrix via apply()."""
    model, layers_attn, weights = _make_model(quantized=True)
    model._build_context_kv_buffers(layers_attn, has_bias=False)
    # Deferred: quant_method is not usable until process_weights_after_loading.
    assert model._fused_kv_weight is None

    model._fuse_dense_kv_weight(torch.float32, torch.device("cpu"))
    torch.testing.assert_close(model._fused_kv_weight, _expected(weights))


def test_quantized_and_unquantized_fusions_agree():
    """Both paths must produce a bit-identical fused matrix."""
    dense_model, dense_layers, weights = _make_model(quantized=False)
    dense_model._build_context_kv_buffers(dense_layers, has_bias=False)

    quant_model, quant_layers, _ = _make_model(quantized=True)
    quant_model._build_context_kv_buffers(quant_layers, has_bias=False)
    quant_model._fuse_dense_kv_weight(torch.float32, torch.device("cpu"))

    assert torch.equal(dense_model._fused_kv_weight, quant_model._fused_kv_weight)
    assert quant_model._fused_kv_weight.shape == (LAYERS * 2 * KV_SIZE, HIDDEN)
