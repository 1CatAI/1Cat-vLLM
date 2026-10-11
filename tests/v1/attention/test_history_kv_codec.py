# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import INT8, resolve_kv_codec


@pytest.mark.parametrize("name", ["auto", "fp8_e4m3", "int8"])
def test_history_view_borrows_payload_and_scale_regions(name):
    codec = resolve_kv_codec(name, representation="history")
    assert codec is not None
    history = torch.zeros((3, 2, 16, 1, 256), dtype=codec.storage_dtype)
    scales = torch.zeros((48 if codec.quantized else 1, 2), dtype=torch.float32)
    view = KVCacheView.bind_history(history, scales, codec)
    assert view.key.untyped_storage().data_ptr() == history.data_ptr()
    assert (
        view.page_size_bytes == view.page_stride_bytes == 8192 * history.element_size()
    )
    view.key[2, 7, 0, 3] = 12
    view.value[1, 9, 0, 2] = 13
    assert history[2, 0, 7, 0, 3] == 12
    assert history[1, 1, 9, 0, 2] == 13
    if codec.quantized:
        assert view.key_scale is not None and view.value_scale is not None
        view.key_scale[2, 7, 0] = 1.25
        view.value_scale[1, 9, 0] = 2.5
        assert scales[39, 0] == 1.25 and scales[25, 1] == 2.5
    else:
        assert view.key_scale is view.value_scale is None


def test_history_identity_and_scale_capacity_are_checked_before_binding():
    history = torch.zeros((1, 2, 16, 1, 256), dtype=torch.int8)
    with pytest.raises(ValueError, match="representation"):
        KVCacheView.bind_history(history, torch.ones(16, 2), INT8)
    codec = resolve_kv_codec("int8", representation="history")
    assert codec is not None and codec.cache_identity != INT8.cache_identity
    with pytest.raises(ValueError, match="every physical token"):
        KVCacheView.bind_history(history, torch.ones(15, 2), codec)


def test_history_int8_reference_retains_token_head_rne():
    codec = resolve_kv_codec("int8", representation="history")
    assert codec is not None
    values = torch.tensor([[127.0, 2.5, 3.5, -2.5], [0.0, 0.0, 0.0, 0.0]])
    payload, scales = codec.encode(values)
    assert payload.tolist() == [[127, 2, 4, -2], [0, 0, 0, 0]]
    assert scales.tolist() == [1.0, pytest.approx(1e-6)]


def test_old_native_history_binary_rejects_int8_before_allocating(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.ops import device_kv_attention as attention

    monkeypatch.setattr(attention, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(
        attention.current_platform, "is_device_capability", lambda _: True
    )
    monkeypatch.setattr(attention.importlib, "import_module", lambda _: object())
    monkeypatch.setattr(torch.ops, "vllm_sm70_qsa_device", SimpleNamespace())
    state = SimpleNamespace(
        codec=resolve_kv_codec("int8", representation="history"),
        device_reference=True,
        width=64,
    )
    attention.initialize_device_history_attention(state, True)
    assert state.device_history_reason == "native_codec_binding_unavailable"
    assert state.device_history_workspace is None
