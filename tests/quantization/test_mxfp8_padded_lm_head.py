# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An MXFP8 LM head whose vocabulary is padded: the padding rows load zero
codes and the unit scale, so they stay zero and inside QPN8's exponent range."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm.model_executor import parameter
from vllm.model_executor.kernels.linear.qpn import pre_ampere  # noqa: F401
from vllm.model_executor.layers import vocab_parallel_embedding as embedding
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_mxfp8 as ct_mxfp8,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_ONE,
    dequant_mxfp8_to_bf16,
)

# 100 rows pad to 128 (padding size 64); K = 128 has eight 16-wide code groups,
# which QPN8's split-K 8 divides.
VOCAB, HIDDEN = 100, 128


@pytest.fixture
def single_rank(monkeypatch):
    for module in (embedding, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)


def _mxfp8_config() -> modelopt.ModelOptMixedPrecisionConfig:
    return modelopt.ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
                "quantized_layers": {"lm_head": {"quant_algo": "MXFP8"}},
            }
        }
    )


def _padded_head(device: str) -> embedding.ParallelLMHead:
    """A loaded MXFP8 LM head with VOCAB checkpoint rows."""
    # The kernel choice is not under test; the QPN8 preparation is called below.
    with patch.object(modelopt, "init_mxfp8_linear_kernel"), torch.device(device):
        head = embedding.ParallelLMHead(
            VOCAB,
            HIDDEN,
            params_dtype=torch.float16,
            quant_config=_mxfp8_config(),
            prefix="lm_head",
        )
    assert isinstance(head.quant_method, modelopt.ModelOptMxFp8LinearMethod)
    assert head.num_embeddings_per_partition == 128

    torch.manual_seed(4511)
    low, high = tm.MXFP8_QPN8_EXPONENT_RANGE
    weight = (torch.randn(VOCAB, HIDDEN) * 8).to(torch.float8_e4m3fn)
    scale = torch.randint(
        low, high + 1, (VOCAB, HIDDEN // MXFP8_BLOCK_SIZE), dtype=torch.uint8
    )
    head.weight_loader(head.weight, weight)
    head.weight_loader(head.weight_scale, scale)
    return head


def test_padding_rows_load_zero_codes_and_the_unit_scale(
    default_vllm_config, single_rank, monkeypatch
):
    head = _padded_head("cpu")

    assert (head.weight[VOCAB:].view(torch.uint8) == 0).all()
    assert (head.weight_scale[VOCAB:] == MXFP8_SCALE_ONE).all()
    dequantized = dequant_mxfp8_to_bf16(head.weight, head.weight_scale)
    assert (dequantized[VOCAB:] == 0).all()

    # QPN8 checks the exponent range, then packs natively (CUDA); stop there.
    packed: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(
        tm,
        "_prepare_fp8_qpn8_linear",
        lambda layer, weight, scales: packed.update(scales=scales),
    )
    tm.prepare_mxfp8_qpn8_linear(head, head.weight, head.weight_scale)
    assert (packed["scales"][VOCAB:] == 1.0).all()

    # Zero codes, as the padding held before, would leave QPN8's range.
    zero_padded = head.weight_scale.clone()
    zero_padded[VOCAB:] = 0
    with pytest.raises(RuntimeError, match="exact QPN8 range"):
        tm.prepare_mxfp8_qpn8_linear(nn.Module(), head.weight, zero_padded)


@pytest.mark.parametrize("rows", [1, 9])
def test_padded_head_logits_through_qpn8(default_vllm_config, single_rank, rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (7, 0),
        (7, 5),
    ):
        pytest.skip("requires SM70 or SM75")
    import vllm._C  # noqa: F401

    head = _padded_head("cuda")
    reference_weight = dequant_mxfp8_to_bf16(head.weight, head.weight_scale).float()
    tm.prepare_mxfp8_qpn8_linear(head, head.weight, head.weight_scale)

    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.float16) * 0.1
    logits = tm.apply_prepared_fp8_qpn8_linear(head, x, None)
    reference = x.float() @ reference_weight.t()

    assert logits.shape == (rows, 128)
    assert (logits[:, VOCAB:] == 0).all()
    error = (logits[:, :VOCAB].float() - reference[:, :VOCAB]).abs().max()
    assert error <= 2e-3 * reference.abs().max()


def test_compressed_tensors_mxfp8_scale_pads_with_the_unit_scale(single_rank):
    layer = nn.Module()
    with patch.object(ct_mxfp8, "init_mxfp8_linear_kernel"):
        ct_mxfp8.CompressedTensorsW8A8Mxfp8().create_weights(
            layer,
            output_partition_sizes=[128],
            input_size_per_partition=HIDDEN,
            params_dtype=torch.float16,
            weight_loader=embedding.VocabParallelEmbedding.weight_loader,
        )

    assert layer.weight_scale.padding_value == MXFP8_SCALE_ONE
    assert not hasattr(layer.weight, "padding_value")
