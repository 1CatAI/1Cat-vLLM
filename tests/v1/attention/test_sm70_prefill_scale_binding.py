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


@pytest.mark.parametrize(
    "available", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_exact_prefill_loader_retains_independent_native_capabilities(
    monkeypatch, available
):
    import sys
    import types

    interface = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    interface.__dict__["ensure_fa2_library_loaded"] = lambda: None
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    names = (
        "sm70_d256_splitd_n32_dense_fwd",
        "sm70_d256_splitd_n32_paged_fwd",
        "sm70_d256_splitd_n32_dense_splitkv3_fwd",
    )
    operations = tuple(object() if present else None for present in available)
    namespace = SimpleNamespace(
        **{name: operation for name, operation in zip(names, operations) if operation}
    )
    monkeypatch.setattr(
        ops, "torch", SimpleNamespace(ops=SimpleNamespace(_vllm_fa2_C=namespace))
    )
    monkeypatch.setattr(ops, "_sm70_splitd_d256_ops_checked", False)
    monkeypatch.delenv("VLLM_SM70_FA2_D256_LIBRARY", raising=False)
    assert ops.get_sm70_splitd_d256_ops() == operations


@pytest.mark.parametrize("query_length", [8000, 8001, 8192])
def test_79t_capability_does_not_require_an_unused_exact_kernel(
    monkeypatch, query_length
):
    monkeypatch.setattr(ops, "get_sm70_splitd_d256_ops", lambda: None)
    monkeypatch.setattr(ops, "get_sm70_d256_gqa_architecture_op", lambda: object())
    monkeypatch.setattr(
        ops, "get_sm70_d256_gqa_architecture_q8192_op", lambda: object()
    )
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", "0")
    q = SimpleNamespace(ndim=4, shape=(1, query_length, 6, 256))
    _, dense = dense_prefill._splitd_admission(q, False, query_length, 8224)
    _, paged = dense_prefill._splitd_admission(q, True, query_length, 8224)
    assert dense and not paged


@pytest.mark.parametrize("neutral_entry", [False, True])
def test_79t_loaded_binding_checks_entry_and_abi_together(monkeypatch, neutral_entry):
    def legacy(*args):
        raise AssertionError("Admission must not launch")

    def neutral(*args):
        raise AssertionError("Admission must not launch")

    namespace = SimpleNamespace(sm70_d256_gqa_architecture_late_scale_abi=lambda: 1)
    if neutral_entry:
        namespace.sm70_d256_gqa_architecture_kv_fwd = neutral
    monkeypatch.setattr(
        ops, "torch", SimpleNamespace(ops=SimpleNamespace(_vllm_fa2_C=namespace))
    )
    bound = ops._architecture_operand_entry("sm70_d256_gqa_architecture_fwd", legacy)
    assert bound is (neutral if neutral_entry else legacy)
    assert ops.prefill_row_scales_supported(bound) is neutral_entry


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
