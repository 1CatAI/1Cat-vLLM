# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unscaled operands require both encoder and actual native reader capability."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import dense_prefill, ops, prefill
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import INT8, INT8_PER_TOKEN_HEAD

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("abi", [None, 0, 1, 2])
def test_scale_binding_uses_the_loaded_binary_probe(monkeypatch, abi):
    def operation(*args, key_scale=None, value_scale=None):
        raise AssertionError("Binding must not execute an attention kernel")

    namespace = SimpleNamespace()
    if abi is not None:
        namespace.contract = lambda: abi
    monkeypatch.setattr(
        ops, "torch", SimpleNamespace(ops=SimpleNamespace(_vllm_fa2_C=namespace))
    )
    assert ops._bind_prefill_scale_capability(operation, "contract") is operation
    assert ops.prefill_row_scales_supported(operation) is (abi == 1)


def owner(codec, bridge, paged=None):
    carrier = torch.zeros((3, 2, 32, 1, 260), dtype=codec.storage_dtype)
    view = KVCacheView.bind(carrier, codec, 256)
    return prefill.PrefillExecutor(
        prefill.PrefillConfig(
            SimpleNamespace(
                use_fp8_prefill_bridge=True, use_flash_v100_prefill_paged=True
            ),
            0.0625,
            codec.name,
        ),
        prefill.PrefillDriverOps(cache_view=view, bridge=bridge, paged=paged),
        SimpleNamespace(),
    ), view


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
@pytest.mark.parametrize(
    "encoder,reader", [(False, False), (False, True), (True, False), (True, True)]
)
def test_row_scaled_bridge_checks_encoder_and_reader(
    monkeypatch, codec, encoder, reader
):
    class Bridge:
        kv_cache_codecs = (codec.name,)
        unscaled_kv_cache_codecs = (codec.name,) if encoder else ()

        def __call__(self, *args, **kwargs):
            raise AssertionError("Admission must not perform a conversion")

    bridge = Bridge()
    native = SimpleNamespace(kv_scale_abi=1 if reader else 0)
    monkeypatch.setattr(
        ops, "get_sm70_splitd_d256_ops", lambda: (native, native, native)
    )
    executor, view = owner(codec, bridge)
    assert executor._should_use_fp8_prefill_bridge(
        q_len=64,
        head_dim=256,
        key_cache=view.key,
        value_cache=view.value,
        causal=True,
        window_size=(-1, -1),
    ) is (encoder and reader)


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
def test_unscaled_payload_never_reaches_a_legacy_half_consumer(monkeypatch, codec):
    seen = []

    def bridge(k, v, table, lengths, ko, vo, *args, **kwargs):
        assert kwargs["kv_cache_dtype"] == codec.name
        assert "key_scale_out" in kwargs and "value_scale_out" in kwargs
        seen.append("unscaled bridge")
        ko.fill_(127)
        vo.fill_(-128)
        kwargs["key_scale_out"].fill_(0.01)
        kwargs["value_scale_out"].fill_(0.02)

    def legacy(*args, **kwargs):
        raise AssertionError("An unscaled INT8 payload is not reconstructed FP16 KV")

    def decline(*args, **kwargs):
        assert kwargs["key_scale"] is not None and kwargs["value_scale"] is not None
        seen.append("qualified reader declined")
        return None

    monkeypatch.setattr(dense_prefill, "try_sm70_fa2_d256_prefill", decline)
    monkeypatch.setattr(
        dense_prefill,
        "uniform_cu_seqlens",
        lambda *args, **kwargs: (torch.tensor([0, 64]), torch.tensor([0, 96])),
    )
    executor, view = owner(codec, bridge, legacy)
    q = torch.zeros((1, 64, 6, 256), dtype=torch.float16)
    out = torch.full_like(q, 19)
    assert (
        executor._run_fp8_prefill_bridge(
            query=q,
            key_cache=view.key,
            value_cache=view.value,
            block_table=torch.tensor([[2, 0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([96], dtype=torch.int32),
            seq_len=96,
            k_scale=1.0,
            v_scale=1.0,
            causal=True,
            window_size=(-1, -1),
            out=out,
        )
        is None
    )
    assert seen == [
        "unscaled bridge",
        "qualified reader declined",
        "qualified reader declined",
    ]
    assert torch.all(out == 19)
