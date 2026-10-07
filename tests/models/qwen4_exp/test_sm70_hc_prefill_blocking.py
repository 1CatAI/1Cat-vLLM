# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Large HC rows retain materialization boundaries with bounded scratch."""

import pytest
import torch

from vllm.config import get_current_vllm_config
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import (
    _blocked_hc_projection,
    _dense_hc_projection,
    _hc_prefill_chunk_size,
)


@pytest.mark.parametrize("rows", [1, 5, 20, 512, 4096])
def test_decode_and_one_block_skip_chunking(default_vllm_config, rows):
    config = get_current_vllm_config()
    config.kernel_config.prefill_hc_chunk_size = 4096
    assert _hc_prefill_chunk_size(rows) == 0
    assert _hc_prefill_chunk_size(4097) == 4096
    config.kernel_config.prefill_hc_chunk_size = 0
    assert _hc_prefill_chunk_size(16384) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [4097, 8209])
def test_projection_short_tail_preserves_fp16_boundaries(rows):
    backend = torch.backends.cuda.matmul
    before = (
        backend.allow_fp16_reduced_precision_reduction,
        backend.allow_fp16_accumulation,
    )
    backend.allow_fp16_reduced_precision_reduction = False
    backend.allow_fp16_accumulation = False
    try:
        torch.manual_seed(1023)
        x = torch.randn(rows, 10240, device="cuda", dtype=torch.float16) * 0.125
        down = torch.randn(336, 10240, device="cuda", dtype=torch.float16) * 0.003
        up = torch.randn(10240, 320, device="cuda", dtype=torch.float16) * 0.03
        expected = _dense_hc_projection(x, down, up)
        actual = _blocked_hc_projection(x, down, up, 4096)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=1e-3, atol=1e-4)
        assert actual[1].stride() == (4, 1)
    finally:
        (
            backend.allow_fp16_reduced_precision_reduction,
            backend.allow_fp16_accumulation,
        ) = before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_combine_short_tail_matches_full_rows(default_vllm_config):
    from types import SimpleNamespace

    from torch import nn

    from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual

    config = get_current_vllm_config()
    backend = torch.backends.cuda.matmul
    before = (
        backend.allow_fp16_reduced_precision_reduction,
        backend.allow_fp16_accumulation,
    )
    backend.allow_fp16_reduced_precision_reduction = False
    backend.allow_fp16_accumulation = False
    try:
        torch.manual_seed(1023)
        module = object.__new__(GatedResidual)
        nn.Module.__init__(module)
        module.config = SimpleNamespace(rms_norm_eps=1e-6)
        module.use_combine = True
        module.hc_count, module.hidden_size = 4, 2560
        module.lora_rank, module.pad_size = 320, 12
        module._partial_inputs = True
        module.input_mix_weight_down_block_inject = nn.Linear(
            10240, 336, bias=False, device="cuda", dtype=torch.float16
        )
        module.input_mix_weight_up = nn.Linear(
            320, 10240, bias=False, device="cuda", dtype=torch.float16
        )
        module.hc_norm = nn.Linear(
            10240, 1, bias=False, device="cuda", dtype=torch.float16
        )
        module.hc_norm.weight = nn.Parameter(
            torch.zeros(10240, device="cuda", dtype=torch.float16), False
        )
        hidden = torch.randn(4097, 10240, device="cuda", dtype=torch.float16) * 0.125
        block = torch.randn(4097, 2560, device="cuda", dtype=torch.float16) * 0.03
        inject = torch.randn(4097, 4, device="cuda", dtype=torch.float16) * 0.125
        config.kernel_config.prefill_hc_chunk_size = 0
        expected = module._combine_and_mix_reduced(hidden, block, inject)
        config.kernel_config.prefill_hc_chunk_size = 4096
        actual = module._combine_and_mix_reduced(hidden, block, inject)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for value, reference in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(value, reference, rtol=1e-3, atol=1e-4)
    finally:
        (
            backend.allow_fp16_reduced_precision_reduction,
            backend.allow_fp16_accumulation,
        ) = before
