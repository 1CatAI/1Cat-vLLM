# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical contracts of offline candidates, not format-selection evidence."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[2] / "tools/kv_codec/evaluate.py"
SPEC = importlib.util.spec_from_file_location("kv_codec_evaluate", PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)


def test_symmetric_int8_uses_the_stored_fp16_scale():
    x = torch.tensor([[[1.0, 0.001, 2.0]]], dtype=torch.float16)
    result = evaluate.symmetric_int8(x, group=3, scale_dtype=torch.float16)
    stored_scale = torch.tensor(2 / 127, dtype=torch.float16).float()
    expected = torch.tensor([[[64.0, 0.0, 127.0]]]) * stored_scale
    torch.testing.assert_close(result.values, expected, rtol=0, atol=0)
    assert result.values[0, 0, 2] != 2.0
    assert result.storage_bytes == 5


@pytest.mark.parametrize("constant", [0.0, 100.0, -100.0])
def test_affine_int8_keeps_constant_heads_exact(constant):
    x = torch.full((2, 1, 16), constant, dtype=torch.float16)
    result = evaluate.asymmetric_int8(x)
    torch.testing.assert_close(result.values, x.float(), rtol=0, atol=0)
    assert result.storage_bytes == 48


def test_gqa_oracle_obeys_head_specific_masks():
    q = torch.ones(1, 4, 2, dtype=torch.float16)
    k = torch.ones(2, 2, 2, dtype=torch.float16)
    v = torch.tensor([[[3.0], [5.0]], [[7.0], [11.0]]], dtype=torch.float16)
    mask = torch.tensor([[[True, False], [False, True], [False, True], [True, False]]])
    output = evaluate.attention(q, k, v, mask, 1.0)
    torch.testing.assert_close(output, torch.tensor([[[3.0], [7.0], [11.0], [5.0]]]))


def test_e4m3_reports_saturation_and_layer_scale_cost():
    result = evaluate.e4m3(torch.tensor([500.0, -500.0, 1.0]), 1.0)
    torch.testing.assert_close(result.values, torch.tensor([448.0, -448.0, 1.0]))
    assert result.clipped_elements == 2
    assert result.storage_bytes == 3


def test_encoded_input_cannot_be_an_fp16_oracle():
    sample = {
        "q": torch.ones(1, 1, 16, dtype=torch.float16),
        "k": torch.ones(2, 1, 16, dtype=torch.int8),
        "v": torch.ones(2, 1, 16, dtype=torch.float16),
        "allowed": torch.ones(1, 2, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="unquantized FP16"):
        evaluate.evaluate_sample(sample)


def test_error_percentiles_do_not_use_the_quantile_size_limit(monkeypatch):
    def forbidden_quantile(*args, **kwargs):
        raise AssertionError("torch.quantile cannot handle 128K/256K KV arrays")

    monkeypatch.setattr(torch, "quantile", forbidden_quantile)
    result = evaluate.error_metrics(torch.arange(4.0), torch.zeros(4))
    assert result["p50_abs"] == 1.5
    assert result["p99_abs"] == pytest.approx(2.97)
    assert result["p999_abs"] == pytest.approx(2.997)
