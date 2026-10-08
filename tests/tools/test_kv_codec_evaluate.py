# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical contracts of offline candidates, not format-selection evidence."""

import importlib.util
import json
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


def test_incomplete_corpus_is_rejected_before_evaluation(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "complete": False, "samples": []}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--manifest", str(manifest), "--out", str(tmp_path / "out.json")],
    )
    with pytest.raises(ValueError, match="incomplete request corpus"):
        evaluate.main()
    assert not (tmp_path / "out.json").exists()


def test_symmetric_int8_uses_the_stored_fp16_scale():
    x = torch.tensor([[[1.0, 0.001, 2.0]]], dtype=torch.float16)
    result = evaluate.symmetric_int8(x, group=3, scale_dtype=torch.float16)
    stored_scale = torch.tensor(2 / 127, dtype=torch.float16).float()
    expected = torch.tensor([[[64.0, 0.0, 127.0]]]) * stored_scale
    torch.testing.assert_close(result.values, expected, rtol=0, atol=0)
    assert result.values[0, 0, 2] != 2.0
    assert result.storage_bytes == 5


def test_legacy_int8_truncation_is_separate_from_nearest_even_candidate():
    x = torch.tensor([[[127.0, 1.75, -1.75, 2.5, -2.5]]], dtype=torch.float16)
    legacy = evaluate.legacy_token_head_int8(x)
    nearest = evaluate.symmetric_int8(x, group=5)
    torch.testing.assert_close(
        legacy.values, torch.tensor([[[127.0, 1.0, -1.0, 2.0, -2.0]]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        nearest.values, torch.tensor([[[127.0, 2.0, -2.0, 2.0, -2.0]]]), rtol=0, atol=0
    )
    assert legacy.storage_bytes == nearest.storage_bytes == 9


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


def test_kv_ablation_distinguishes_the_softmax_from_value_error():
    sample = {
        "q": torch.ones(1, 2, 4, dtype=torch.float16),
        "k": torch.ones(2, 1, 4, dtype=torch.float16),
        "v": torch.tensor([[[17.0] * 4], [[30.0] * 4]], dtype=torch.float16),
        "allowed": torch.ones(1, 2, dtype=torch.bool),
        "attention_scale": 0.5,
        "k_scale": 1.0,
        "v_scale": 1.0,
    }
    plain = evaluate.evaluate_sample(sample)
    ablated = evaluate.evaluate_sample(sample, ablate_kv=True)
    for old, new in zip(plain, ablated):
        assert old == {
            key: value
            for key, value in new.items()
            if key not in ("attention_k_only", "attention_v_only")
        }
    e4m3 = next(row for row in ablated if row["scheme"] == "e4m3_layer_scale")
    assert e4m3["attention_k_only"]["max_abs"] == 0
    assert e4m3["attention_v_only"]["max_abs"] == 0.5
    assert e4m3["attention"] == e4m3["attention_v_only"]

    # Constant V makes every softmax distribution yield the same output.
    sample["k"] = sample["v"]
    sample["v"] = torch.full_like(sample["v"], 17.0)
    e4m3 = next(
        row
        for row in evaluate.evaluate_sample(sample, ablate_kv=True)
        if row["scheme"] == "e4m3_layer_scale"
    )
    assert e4m3["attention_k_only"]["max_abs"] == 0
    assert e4m3["attention_v_only"]["max_abs"] == 1


def test_e4m3_reports_saturation_and_layer_scale_cost():
    result = evaluate.e4m3(torch.tensor([500.0, -500.0, 1.0]), 1.0)
    torch.testing.assert_close(result.values, torch.tensor([448.0, -448.0, 1.0]))
    assert result.clipped_elements == 2
    assert result.storage_bytes == 3


def test_fp16_tile_staging_reports_overflow_from_rounded_scale():
    sample = {
        "q": torch.ones(1, 1, 4, dtype=torch.float16),
        "k": torch.ones(2, 1, 4, dtype=torch.float16),
        "v": torch.tensor([[[65504.0, -65504.0, 0.0, 1.0]]] * 2, dtype=torch.float16),
        "allowed": torch.ones(1, 2, dtype=torch.bool),
        "attention_scale": 0.5,
        "k_scale": 1.0,
        "v_scale": 1.0,
    }
    results = {
        row["scheme"]: row
        for row in evaluate.evaluate_sample(sample, simulate_fp16_staging=True)
    }
    # 65504/127 rounds to stored FP16 scale516;127*516=65532 is finite
    # in FP32 but becomes infinity when a reader stages the decoded tile in FP16.
    bad = results["int8_token_head_fp16"]["fp16_tile_staging"]
    assert bad == {"nonfinite_k": 0, "nonfinite_v": 4}
    assert "attention" not in bad
    for name in ("fp16", "int8_token_head_fp32"):
        staged = results[name]["fp16_tile_staging"]
        assert staged["nonfinite_k"] == staged["nonfinite_v"] == 0
        assert staged["attention"]["max_abs"] == (0 if name == "fp16" else 1)


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
