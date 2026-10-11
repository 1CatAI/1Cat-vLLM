# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.attention.kv_codecs import (
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    INT8,
    INT8_PER_TOKEN_HEAD,
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
        ("int8", INT8),
        ("int8_per_token_head", INT8_PER_TOKEN_HEAD),
    ],
)
def test_aliases_resolve_to_one_codec(spelling, codec):
    assert resolve_kv_codec(spelling) is codec
    assert canonical_kv_cache_dtype(spelling) == codec.name


def test_unknown_formats_are_not_guessed():
    assert resolve_kv_codec(None) is None
    assert resolve_kv_codec("int8_unregistered") is None
    assert canonical_kv_cache_dtype("int8_unregistered") == "int8_unregistered"


def test_int8_rounding_and_cache_identity_are_not_legacy_aliases():
    x = torch.tensor([[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]])
    q, scale = INT8.encode(x)
    legacy, old_scale = INT8_PER_TOKEN_HEAD.encode(x)
    assert q.tolist() == [[127, 0, 2, 2, 0, -2, -2]]
    assert legacy.tolist() == [[127, 0, 1, 2, 0, -1, -2]]
    assert torch.equal(scale, old_scale)
    assert INT8.cache_identity != INT8_PER_TOKEN_HEAD.cache_identity


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_finite_extremes_and_reconstruction_error(dtype):
    limit = torch.finfo(dtype).max
    x = torch.tensor(
        [[0, 0, 0, 0], [limit, -limit, limit / 3, -limit / 7]], dtype=dtype
    )
    q, scale = INT8.encode(x)
    decoded = INT8.dequantize(q, scale, dtype)
    assert scale.dtype == torch.float32
    assert torch.isfinite(scale).all()
    assert torch.isfinite(decoded).all()
    assert torch.equal(decoded[0], x[0])
    # Quantizer error plus the destination dtype's final rounding error.
    error = (decoded.double() - x.double()).abs()
    allowance = scale.double()[:, None] / 2 + x.double().abs() * torch.finfo(dtype).eps
    assert torch.all(error <= allowance)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_int8_nonfinite_rows_propagate(invalid):
    q, scales = INT8.encode(torch.tensor([[1.0, invalid], [2.0, 3.0]]))
    result = INT8.dequantize(q, scales, torch.float32)
    assert torch.isnan(result[0]).all()
    assert torch.isfinite(result[1]).all()


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
