# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.attention.kv_codecs import (
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    INT8_PER_TOKEN_HEAD,
    canonical_kv_cache_dtype,
    per_token_head_scales_for_kv_views,
    resolve_kv_codec,
    split_per_token_head_kv_cache,
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
        ("int8", INT8_PER_TOKEN_HEAD),
        ("int8_per_token_head", INT8_PER_TOKEN_HEAD),
    ],
)
def test_aliases_resolve_to_one_codec(spelling, codec):
    assert resolve_kv_codec(spelling) is codec
    assert canonical_kv_cache_dtype(spelling) == codec.name


def test_unknown_formats_are_not_guessed():
    assert resolve_kv_codec(None) is None
    assert resolve_kv_codec("fp8_per_token_head") is None
    assert canonical_kv_cache_dtype("fp8_per_token_head") == "fp8_per_token_head"


def test_per_token_head_block_planar_views_preserve_alignment_and_head_slices():
    raw = torch.zeros((4, 2 * 32 * 2 * 260 + 128), dtype=torch.int8)
    carrier = torch.as_strided(
        raw,
        size=(4, 2, 32, 2, 260),
        stride=(raw.shape[1], 32 * 2 * 260, 2 * 260, 260, 1),
        storage_offset=64,
    )
    kc, vc, ks, vs = split_per_token_head_kv_cache(carrier, 256)
    assert kc.shape == vc.shape == (4, 32, 2, 256)
    assert ks.shape == vs.shape == (4, 32, 2)
    assert kc.data_ptr() % 16 == 0 and vc.data_ptr() % 16 == 0
    ks.copy_(torch.arange(4 * 32 * 2, dtype=torch.float32).reshape(4, 32, 2))
    vs.copy_(ks + 1000.0)
    for h in range(2):
        ks_h, vs_h = per_token_head_scales_for_kv_views(
            kc[:, :, h : h + 1], vc[:, :, h : h + 1]
        )
        assert torch.equal(ks_h, ks[:, :, h : h + 1])
        assert torch.equal(vs_h, vs[:, :, h : h + 1])


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
