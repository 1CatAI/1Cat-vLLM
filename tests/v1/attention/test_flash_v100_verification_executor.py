# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier execution consumes explicit policy and operators, without Impl."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import routing, state, verify
from vllm.v1.attention.backends.flash_v100.config import V100AttnConfig

pytestmark = pytest.mark.cpu_test


def test_ordinary_decode_contract_does_not_construct_verifier():
    def unexpected(*args, **kwargs):
        raise AssertionError("ordinary attention needs no verifier allocation")

    receiver = SimpleNamespace(
        _new_verification_executor=unexpected, _flash_v100_window_size=unexpected
    )
    verify._validate_dflash_attention_contract(
        receiver, SimpleNamespace(), SimpleNamespace()
    )


def policy():
    return verify.VerificationConfig(
        policy=cast(
            V100AttnConfig,
            SimpleNamespace(
                use_flash_v100_decode=True,
                use_smallq_decode_xqa=True,
                smallq_decode_max_query_len=8,
                smallq_decode_max_model_len=32768,
            ),
        ),
        scale=0.125,
        kv_cache_dtype="auto",
        alibi_slopes=None,
        logits_soft_cap=0.0,
        grouped_enabled=False,
        grouped_batch_enabled=False,
        grouped_max_query=16,
        grouped_request_major_abi=1,
        grouped_min_model_len=32768,
    )


def operators(native):
    return verify.VerificationOps(
        grouped=native("legacy"),
        fp16_grouped=native("fp16"),
        e4m3_grouped=native("e4m3"),
        xqa=native("xqa"),
        window_size=lambda causal: (-1, -1),
        layer_info=lambda layer: {},
        xqa_codec=lambda *args: None,
        decode=native("scalar"),
    )


@pytest.mark.parametrize("choice", ["fp16", "e4m3", "xqa", "scalar"])
def test_verifier_operator_order_and_native_contract(monkeypatch, choice):
    calls = []
    routes: list[str] = []
    query = torch.zeros((8, 6, 256), dtype=torch.float16)
    cache = torch.zeros((1, 832, 1, 256), dtype=torch.float16)
    table = torch.zeros((8, 1), dtype=torch.int32)
    lengths = torch.arange(9, 17, dtype=torch.int32)
    metadata = SimpleNamespace(block_table=table[:1], seq_lens=lengths[-1:])
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    output = torch.zeros_like(query)

    def native(name):
        def run(q, k, v, bt, seq, **kwargs):
            calls.append(name)
            assert q is query and k is cache and v is cache
            assert bt is (metadata.block_table if name in ("fp16", "e4m3") else table)
            assert seq is lengths
            assert kwargs["softmax_scale"] == 0.125
            kwargs["out"].fill_(7)

        return run

    ops = operators(native)

    def grouped_admit(contract, *args, **kwargs):
        calls.append("admit_fp16")
        assert isinstance(contract, verify.GroupedAdmission)
        assert contract.flash_attn_grouped_fp16_fp32_paged is ops.fp16_grouped
        assert contract.flash_attn_grouped_e4m3_fp32_paged is ops.e4m3_grouped
        assert contract.kv_cache_dtype == "auto" and contract.use_smallq_decode_xqa
        assert contract._flash_v100_window_size(causal=True) == (-1, -1)
        return None if choice == "fp16" else "test_decline"

    def e4m3_admit(*args, **kwargs):
        calls.append("admit_e4m3")
        return choice == "e4m3"

    def xqa_admit(*args, **kwargs):
        calls.append("admit_xqa")
        return choice == "xqa"

    monkeypatch.setattr(verify, "grouped_fp16_fp32_reason", grouped_admit)
    monkeypatch.setattr(verify, "grouped_e4m3_fp32_allowed", e4m3_admit)
    monkeypatch.setattr(routing, "_record_route", routes.append)
    monkeypatch.setattr(state, "_logged_prefill_smallq_decode_xqa", False)
    executor = verify.VerificationExecutor(
        policy(), replace(ops, admit_xqa_override=xqa_admit)
    )
    executor.call_smallq_decode_paged(
        layer,
        query,
        cache,
        cache,
        table,
        lengths,
        metadata,
        out=output,
        max_seq_len_hint=16,
        workspace_seq_capacity_hint=32,
        partition_size_hint=None,
    )
    expected = ["admit_fp16"]
    if choice != "fp16":
        expected.append("admit_e4m3")
    if choice in ("xqa", "scalar"):
        expected.append("admit_xqa")
    assert calls == expected + [choice]
    assert output.eq(7).all()
    assert routes == [
        {
            "fp16": "prefill_smallq_fp16_grouped_fp32",
            "e4m3": "prefill_smallq_e4m3_grouped_fp32",
            "xqa": "prefill_smallq_decode_xqa",
            "scalar": "prefill_smallq_decode_scalar",
        }[choice]
    ]


def test_persistent_small_query_rows_and_falsey_injection():
    calls = []
    query = torch.ones((3, 2, 4))
    output = torch.zeros_like(query)
    table = torch.zeros((3, 1), dtype=torch.int32)
    lengths = torch.tensor([8, 9, 10], dtype=torch.int32)
    qsl = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        seq_lens=torch.tensor([10]),
        query_start_loc=torch.tensor([0, 3]),
        smallq_decode_block_table=table,
        smallq_decode_seq_lens=lengths,
        smallq_query_start_loc=qsl,
    )

    class FalseyOperator:
        def __bool__(self):
            return False

        def __call__(self, layer, q, k, v, bt, seq, attn, **kwargs):
            calls.append((bt.data_ptr(), seq.data_ptr()))
            assert attn is metadata
            kwargs["out"].fill_(5)

    executor = verify.VerificationExecutor(
        policy(),
        replace(operators(lambda name: None), run_smallq_override=FalseyOperator()),
    )
    for _ in range(2):
        assert (
            executor.small_query_prefill(
                None,
                query,
                query,
                query,
                metadata,
                output,
                metadata.query_start_loc,
                metadata.seq_lens,
            )
            is output
        )
    assert calls == [(table.data_ptr(), lengths.data_ptr())] * 2
    assert output.eq(5).all()


@pytest.mark.parametrize("causal", [False, True])
def test_contract_validation_without_backend(monkeypatch, causal):
    monkeypatch.setattr(state, "_logged_dflash_attention_contracts", set())
    executor = verify.VerificationExecutor(policy(), operators(lambda name: None))
    layer = SimpleNamespace(is_dflash_draft_attn=True, dflash_expected_causal=False)
    metadata = SimpleNamespace(causal=causal)
    if causal:
        with pytest.raises(RuntimeError, match="causality mismatch"):
            executor.validate_contract(layer, metadata)
    else:
        executor.validate_contract(layer, metadata)
        executor.validate_contract(layer, metadata)
        assert len(state._logged_dflash_attention_contracts) == 1
