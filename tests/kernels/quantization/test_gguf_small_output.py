# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.gguf import small_output_capability
from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.model_executor.layers.quantization.gguf_small_output import _small_output


@pytest.mark.parametrize("source_type", [18, 21, 23])
def test_measured_shape_m_and_precision_admission(monkeypatch, source_type):
    monkeypatch.setattr(
        torch.ops._C, "gguf_small_output_sm70_out", lambda *args: None, raising=False
    )
    cap = small_output_capability(source_type, 1536, 5120, torch.float16)
    assert cap.reason is None
    assert [m for m in (1, 5, 8, 16, 32, 512) if cap.supports_m(m)] == [8]
    for arguments in (
        (12, 1536, 5120, torch.float16),
        (17, 1536, 5120, torch.float16),
        (source_type, 4352, 5120, torch.float16),
        (source_type, 1536, 2560, torch.float16),
        (source_type, 1536, 5120, torch.bfloat16),
        (source_type, 1536, 5120, torch.float16, False),
        (source_type, 1536, 5120, torch.float16, True, 80),
    ):
        assert small_output_capability(*arguments).reason


@pytest.mark.parametrize("head_tiling", [False, True])
@pytest.mark.parametrize("fallback_tiling", [False, True])
def test_runtime_m_and_fallback_head_order(monkeypatch, head_tiling, fallback_tiling):
    calls = []

    def native(out, x, records, partials, counters, kind, splits, tiled):
        calls.append(("native", x.clone(), kind, splits, tiled))
        out.fill_(7)

    def canonical(
        x,
        codes,
        stats,
        cache,
        family,
        decoder,
        group,
        kld,
        qld,
        n,
        logical,
        cache_bands,
        blas_bands,
    ):
        calls.append(("canonical", x.clone(), decoder, cache_bands, blas_bands))
        return x.new_full((x.shape[0], logical), 3)

    monkeypatch.setattr(
        torch.ops._C, "gguf_small_output_sm70_out", native, raising=False
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.gguf_small_output."
        "_prepared_gguf_projection",
        canonical,
    )
    empty = torch.empty(0)
    # Last two serialized fields count bands; they are not single-op arguments.
    descriptors = [2, 21, 32, 0, 0, 5120, 5120, 2, 2]
    for m in (512, 8, 1, 5, 16, 20, 32, 8):
        x = (torch.arange(m * 1536).remainder(97).half()).reshape(m, 1536)
        out = _small_output(
            x,
            empty,
            empty,
            empty,
            21,
            head_tiling,
            fallback_tiling,
            [empty],
            [empty],
            [None],
            descriptors,
            [32, 512],
            [512, -1],
        )
        call = calls[-1]
        if m == 8:
            assert call[0] == "native" and call[2:] == (21, 1, head_tiling)
            torch.testing.assert_close(call[1], x, rtol=0, atol=0)
            assert torch.all(out == 7)
        else:
            assert call[0] == "canonical" and call[2:] == (21, [32, 512], [512, -1])
            expected = (
                GGUFHeadTilingLayout(3, 128).input_to_gguf(x) if fallback_tiling else x
            )
            torch.testing.assert_close(call[1], expected, rtol=0, atol=0)
            assert torch.all(out == 3)


def test_dynamic_export_keeps_one_opaque_output_boundary():
    class Projection(torch.nn.Module):
        def forward(self, x):
            empty = torch.empty(0)
            return torch.ops.vllm.gguf_small_output(
                x,
                empty,
                empty,
                empty,
                21,
                True,
                True,
                [empty],
                [empty],
                [None],
                [2, 21, 32, 0, 0, 5120, 5120, 0, 0],
                [],
                [],
            )

    exported = torch.export.export(
        Projection(),
        (torch.empty(512, 1536, dtype=torch.float16),),
        dynamic_shapes=({0: torch.export.Dim("tokens", min=1, max=8192)},),
    )
    targets = [str(n.target) for n in exported.graph.nodes if n.op == "call_function"]
    assert sum("gguf_small_output" in name for name in targets) == 1
    assert not any("dequant" in name or "mm.default" in name for name in targets)
