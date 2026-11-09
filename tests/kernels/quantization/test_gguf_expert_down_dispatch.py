# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_turbomind_moe import _expert_down


@pytest.mark.parametrize("source,decoder", [(20, 0), (42, 2)])
@pytest.mark.parametrize("rows", [10, 50, 100, 150, 200, 512])
def test_actual_routed_rows_select_measured_vectors(monkeypatch, source, decoder, rows):
    calls = []

    def vector(out, x, offsets, wp, sp, source_type, experts, group):
        calls.append(("vector", source_type))
        out.fill_(3)

    def fallback(out, x, offsets, wp, sp, actual_decoder, experts, group):
        calls.append(("gemm", actual_decoder))
        out.fill_(3)

    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            gguf_small_grouped_vec_sm70_out=vector,
            gguf_lut4_grouped_gemm_sm70_out=fallback,
            gguf_affine_grouped_gemm_sm70_out=fallback,
        ),
    )
    x = torch.empty((rows, 160), dtype=torch.float16)
    out = torch.empty((rows, 2560), dtype=x.dtype)
    empty = torch.empty(0)
    _expert_down(out, x, empty, empty, empty, source, decoder, 512, 32, [10, 50])
    expected = ("vector", source) if rows in (10, 50) else ("gemm", decoder)
    assert calls == [expected]
    assert torch.all(out == 3)
