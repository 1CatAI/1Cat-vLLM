# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subpages preserve physical owner addresses for payload and FP32 scales."""

import pytest
import torch

from vllm.v1.attention.kv_cache_view import KVCacheView, KVPageMapping
from vllm.v1.attention.kv_codecs import FP8_E4M3, FP16, INT8

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("codec", [FP16, FP8_E4M3, INT8], ids=lambda c: c.name)
@pytest.mark.parametrize("page", [16, 64, 2048])
def test_subpages_cover_payload_and_separate_scale_addresses(codec, page):
    width = 256 + (4 if codec.per_token_head else 0)
    shape = (3, 2, page, 1, width)
    storage = torch.zeros(3 * (2 * page * width + 1024), dtype=codec.storage_dtype)
    carrier = storage.as_strided(
        shape, (2 * page * width + 1024, page * width, width, width, 1)
    )
    view = KVCacheView.bind(carrier, codec, 256)
    mapping = view.subpage_mapping(4)
    assert mapping.owner_page_span == page // 4
    for owner, subpage, token in ((2, 0, 0), (0, page // 4 - 1, 3), (1, 1, 2)):
        page_id = owner * mapping.owner_page_span + subpage
        mapped_owner, mapped_subpage = divmod(page_id, mapping.owner_page_span)
        for payload in (view.key, view.value):
            expected = payload[owner, subpage * 4 + token, 0, 0].data_ptr()
            actual = payload[
                mapped_owner, mapped_subpage * mapping.subpage_tokens + token, 0, 0
            ].data_ptr()
            assert actual == expected
        if view.key_scale is not None:
            for plane in (view.key_scale, view.value_scale):
                assert (
                    plane[mapped_owner, mapped_subpage * 4 + token, 0].data_ptr()
                    == plane[owner, subpage * 4 + token, 0].data_ptr()
                )
    if codec is INT8 and page < 128:
        # These legal pages cannot be flattened into an affine page4 payload.
        # The mapped owner strides are essential; no rounded byte ratio is valid.
        assert view.page_stride_bytes % (4 * 256) != 0


def test_subpage_mapping_rejects_unsupported_row_strides():
    data = torch.empty((3, 64, 1, 264))[:, :, :, :256]
    with pytest.raises(ValueError, match="contiguous payload rows"):
        KVPageMapping.bind_payload(data, data, 4)


def test_qsa_old_native_binding_does_not_admit_int8():
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia.ops import qsa

    old = SimpleNamespace(
        grouped_sparse_page4_fwd=lambda *args: None,
        grouped_sparse_page4_plan_fwd=lambda *args: None,
        grouped_sparse_page4_abi_version=lambda: 2,
    )
    assert qsa._qsa_grouped_page4_supported(old, "fp8_e4m3")
    assert not qsa._qsa_grouped_page4_supported(old, "int8")


@pytest.mark.parametrize("padding,expected", [(0, True), (128, False), (1024, True)])
def test_qsa_legacy_mapping_rejects_fractional_physical_microtiles(padding, expected):
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    shape = (3, 64, 1, 256)
    stride = 64 * 256 + padding
    data = torch.empty(3 * stride, dtype=torch.uint8).as_strided(
        shape, (stride, 256, 256, 1)
    )
    assert qsa._qsa_page4_legacy_mapping_supported(data) == expected


@pytest.mark.parametrize("codec", [FP16, FP8_E4M3, INT8], ids=lambda c: c.name)
@pytest.mark.parametrize("history", [False, True])
def test_qsa_engine_spec_keeps_payload_owner_and_scale_owner_distinct(
    monkeypatch, codec, history
):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import qsa
    from vllm.v1.kv_cache_interface import KVQuantMode

    monkeypatch.setattr(qsa, "qsa_dcp_block_geometry", lambda *args: (2048, 1, True))
    layer = SimpleNamespace(
        layer_name="attention",
        host_kv_enabled=history,
        num_kv_heads=1,
        head_dim=256,
        kv_cache_torch_dtype=codec.storage_dtype,
        kv_cache_dtype=codec.name,
    )
    spec = qsa.Qwen4ExpQSAAttention.get_kv_cache_spec(layer, None)
    assert spec.host_backed == history
    if history:
        assert spec.kv_quant_mode == KVQuantMode.NONE
        assert spec.layout_cache_dtype("int8") == "auto"
        assert spec.page_size_bytes == 2048 * 2 * 256 * codec.storage_dtype.itemsize
    else:
        assert spec.layout_cache_dtype(codec.name) == codec.name
        assert spec.page_size_bytes == 2048 * 2 * (
            260 if codec.per_token_head else 256 * codec.storage_dtype.itemsize
        )


@pytest.mark.parametrize("codec", [FP16, FP8_E4M3, INT8], ids=lambda c: c.name)
def test_qsa_constructor_admits_cache_codec_and_retains_weight_quality_guard(
    monkeypatch, codec
):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import qsa

    monkeypatch.setattr(qsa, "_verify_quantized_kv_requirements", lambda *args: None)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=codec.name),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    with pytest.raises(NotImplementedError, match="does not support KV quantization"):
        qsa.Qwen4ExpQSAAttention(
            vllm_config=config,
            config=None,
            layer_id=0,
            quant_config=SimpleNamespace(kv_cache_scheme=object()),
        )
