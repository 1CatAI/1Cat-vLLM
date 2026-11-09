# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature policies remain dynamic at the injected execution boundary."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import impl
from vllm.v1.attention.backends.flash_v100.spec import policy

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "variable,owner,field,default",
    [
        ("VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN", "verify", "branch_enabled", True),
        (
            "VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT",
            "verify",
            "branch_strict",
            False,
        ),
        ("VLLM_FLASH_V100_DFLASH_PREFIX_DUMP", "prefill", "prefix_dump_enabled", False),
        (
            "VLLM_DFLASH_DDTREE_WORKER_PROFILE",
            "policy",
            "worker_profile_enabled",
            False,
        ),
    ],
)
def test_dynamic_switch_changes_after_owner_construction(
    monkeypatch, variable, owner, field, default
):
    monkeypatch.delenv(variable, raising=False)
    instance = object.__new__(impl.FlashAttnV100Impl)
    dependencies = {
        "verify": instance._new_verification_executor().ops,
        "prefill": instance._new_prefill_executor().ops,
        "policy": policy,
    }
    read = getattr(dependencies[owner], field)
    assert read() is default
    monkeypatch.setenv(variable, "0")
    assert read() is False
    monkeypatch.setenv(variable, "1")
    assert read() is True


def test_partition_validation_is_lazy_and_keeps_error_contract(monkeypatch):
    instance = object.__new__(impl.FlashAttnV100Impl)
    read = instance._new_verification_executor().ops.partition_hint
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "invalid")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "0")
    assert read() is None
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "1")
    with pytest.raises(ValueError, match="got 'invalid'") as error:
        read()
    assert isinstance(error.value.__cause__, ValueError)
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "512")
    assert read() == 512
    variable = "VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE"
    monkeypatch.delenv(variable, raising=False)
    assert policy.context_bucket_partition_size_hint() is None
    monkeypatch.setenv(variable, "64")
    with pytest.raises(ValueError, match=r"\(256, 512, 1024\), got 64"):
        policy.context_bucket_partition_size_hint()


def test_trace_destination_and_payload_are_read_at_emission(monkeypatch, tmp_path):
    instance = object.__new__(impl.FlashAttnV100Impl)
    ops = instance._new_verification_executor().ops
    variable = "VLLM_DFLASH_DDTREE_TRACE_JSONL"
    monkeypatch.delenv(variable, raising=False)
    assert not ops.tree_trace_enabled()
    ops.tree_trace_event("disabled", {})
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(variable, str(path))
    assert ops.tree_trace_enabled()
    ops.tree_trace_event("original", {"event": "payload", "pid": 7, "text": "树"})
    assert json.loads(path.read_text()) == {"event": "payload", "pid": 7, "text": "树"}
    assert "\\u6811" in path.read_text()
    messages = []
    monkeypatch.setattr(policy.logger, "exception", lambda *args: messages.append(args))
    monkeypatch.setenv(variable, str(tmp_path / "missing" / "trace.jsonl"))
    ops.tree_trace_event("unwritable", {})
    assert len(messages) == 1
    assert messages[0][1].endswith("missing/trace.jsonl")


def test_legacy_partition_patch_reaches_real_xqa_call(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as legacy

    calls = []

    def hint():
        calls.append("hint")
        return 512

    def native(*args, **kwargs):
        calls.append(kwargs["partition_size_hint"])
        kwargs["out"].fill_(3)

    monkeypatch.setattr(legacy, "_mtp5_xqa_dual_cta_partition_size_hint", hint)
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.kv_cache_dtype = "fp8_e5m2"
    instance.flash_attn_decode_paged_xqa = native
    instance._smallq_decode_xqa_allowed = lambda *args, **kwargs: True
    instance._flash_v100_window_size = lambda causal: (-1, -1)
    query = torch.zeros((5, 6, 256), dtype=torch.float16)
    cache = torch.zeros((1, 1616, 1, 256), dtype=torch.uint8)
    table = torch.zeros((5, 1), dtype=torch.int32)
    lengths = torch.arange(100, 105, dtype=torch.int32)
    output = torch.zeros_like(query)
    instance._new_verification_executor().call_smallq_decode_paged(
        SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0),
        query,
        cache,
        cache,
        table,
        lengths,
        SimpleNamespace(),
        out=output,
        max_seq_len_hint=104,
        workspace_seq_capacity_hint=None,
        partition_size_hint=None,
    )
    assert calls == ["hint", 512]
    assert torch.all(output == 3)
