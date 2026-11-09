# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.attention.kv_codecs import (
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    canonical_kv_cache_dtype,
    resolve_kv_codec,
)


@pytest.mark.parametrize(
    "spelling,codec",
    [
        ("auto", FP16),
        ("float16", FP16),
        ("bfloat16", BF16),
        ("fp8", FP8_E4M3),
        ("fp8_e4m3", FP8_E4M3),
        ("fp8_e5m2", FP8_E5M2),
    ],
)
def test_aliases_resolve_to_one_codec(spelling, codec):
    assert resolve_kv_codec(spelling) is codec
    assert canonical_kv_cache_dtype(spelling) == codec.name


def test_unknown_formats_are_not_guessed():
    assert resolve_kv_codec(None) is None
    assert resolve_kv_codec("int8_per_token_head") is None
    assert canonical_kv_cache_dtype("int8_per_token_head") == "int8_per_token_head"


def test_storage_admission_checks_both_tensors():
    k = torch.zeros(2, 16, 1, 8, dtype=torch.uint8)
    assert FP8_E4M3.stores(k, k.clone())
    assert not FP8_E4M3.stores(k, k.to(torch.float16))
    assert not FP16.stores(k, k)


@pytest.mark.parametrize("codec", [FP8_E4M3, FP8_E5M2])
def test_dequantize_matches_bit_cast(codec):
    ref = torch.linspace(-3, 3, 64).to(codec.element_dtype)
    raw = ref.view(torch.uint8)
    out = codec.dequantize(raw, 0.5, torch.float16)
    assert torch.equal(out, ref.to(torch.float16) * 0.5)
    assert FP16.dequantize(raw, 0.5, torch.float16) is raw
