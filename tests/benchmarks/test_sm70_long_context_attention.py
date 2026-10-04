# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest
import torch


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "benchmarks/kernels" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_respects_request_page_order_and_causal_lengths():
    bench = load_script("benchmark_sm70_long_context_attention.py")
    raw = torch.zeros(4, 2, 8, 1, 256)
    for page in range(4):
        raw[page, 1] = page + 1
    cache = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    key, value = cache.unbind(1)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    query = torch.zeros(16, 6, 256, dtype=torch.float16)
    lengths = torch.arange(1, 9, dtype=torch.int32).repeat(2)
    lengths[7] = lengths[15] = 12
    out = bench.dense_reference((query, key, value, table, lengths))
    expected = torch.empty_like(out)
    for request, first, second in ((0, 3, 1), (1, 4, 2)):
        expected[request * 8 : (request + 1) * 8] = first * 1.25
        expected[request * 8 + 7] = (8 * first + 4 * second) / 12 * 1.25
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-12)


def test_source_generation_preserves_reference_and_rejects_layout_drift():
    build = load_script("build_sm70_long_context_attention.py")
    source = (
        Path(__file__).resolve().parents[2]
        / "csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu"
    ).read_text()
    assert build.candidate_source(source, "reference") == source
    changed = build.candidate_source(source, "bit-decode")
    assert changed != source
    assert "void flash_attention_grouped_verify_e5m2_combine_kernel(" in changed
    with pytest.raises(ValueError, match="lookup table"):
        build.candidate_source(
            source.replace("e4m3_lut[256]", "e4m3_lut[257]"), "bit-decode"
        )
