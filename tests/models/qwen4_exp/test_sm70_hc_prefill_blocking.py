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


def _projection_module(device):
    from torch import nn

    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual

    module = object.__new__(GatedResidual)
    nn.Module.__init__(module)
    module.use_combine = True
    module._sm70_qwen38_fp16_fused_hc = True
    module.lora_rank, module.hc_count, module.pad_size = 320, 4, 12
    for name, shape in (
        ("input_mix_weight_down_block_inject", (336, 10240)),
        ("input_mix_weight_up", (10240, 320)),
    ):
        linear = object.__new__(LinearBase)
        nn.Module.__init__(linear)
        linear.weight = nn.Parameter(
            torch.empty(shape, dtype=torch.float16, device=device), False
        )
        linear.quant_method = UnquantizedLinearMethod()
        module.add_module(name, linear)
    return module


def test_first_hc_prefill_export_keeps_runtime_boundary(
    default_vllm_config, monkeypatch
):
    from torch import nn

    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc

    monkeypatch.setattr(hc, "use_sm70_decode_graph_semantics", lambda: False)

    class FirstHC(nn.Module):
        def __init__(self):
            super().__init__()
            self.hc = _projection_module("meta")

        def forward(self, x):
            return self.hc._project(x)

    exported = torch.export.export(
        FirstHC(), (torch.empty(16384, 10240, device="meta", dtype=torch.float16),)
    )
    calls = [
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.vllm.qwen38_sm70_fp16_prefill_hc.default
    ]
    assert len(calls) == 1
    assert not any(
        node.target == torch.ops.aten.linear.default for node in exported.graph.nodes
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_first_hc_dynamic_prefill_dispatch(default_vllm_config, monkeypatch):
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc

    monkeypatch.setattr(hc, "use_sm70_decode_graph_semantics", lambda: False)
    backend = torch.backends.cuda.matmul
    before = (
        backend.allow_fp16_reduced_precision_reduction,
        backend.allow_fp16_accumulation,
    )
    backend.allow_fp16_reduced_precision_reduction = False
    backend.allow_fp16_accumulation = False
    try:
        torch.manual_seed(1023)
        module = _projection_module("cuda")
        with torch.no_grad():
            module.input_mix_weight_down_block_inject.weight.normal_(std=0.003)
            module.input_mix_weight_up.weight.normal_(std=0.03)
        calls = []
        original = hc._blocked_hc_projection

        def record(*args):
            calls.append(args[0].shape[0])
            return original(*args)

        monkeypatch.setattr(hc, "_blocked_hc_projection", record)
        compiled = torch.compile(module._project, fullgraph=True, dynamic=True)
        for rows in (4097, 513):
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.float16) * 0.125
            expected = _dense_hc_projection(
                x,
                module.input_mix_weight_down_block_inject.weight,
                module.input_mix_weight_up.weight,
            )
            with torch.no_grad():
                actual = compiled(x)
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference, rtol=1e-3, atol=1e-4)
            assert actual[1].stride() == (4, 1)
        assert calls == [4097]
    finally:
        (
            backend.allow_fp16_reduced_precision_reduction,
            backend.allow_fp16_accumulation,
        ) = before


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
