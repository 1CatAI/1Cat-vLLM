# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native selection inputs survive worker serialization and engine interleaving."""

import os
from multiprocessing import Pipe
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import regex as re

from vllm.config.collective import CollectiveNativeConfig
from vllm.config.execution_policy import CommunicationPolicy
from vllm.distributed.device_communicators.collective_provider import (
    NativeCollectiveBindings,
)


@pytest.mark.parametrize("order", [("1stage", "2stage"), ("2stage", "1stage")])
def test_native_policy_isolated_serialized_and_no_environment_write(monkeypatch, order):
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", "legacy")
    before = dict(os.environ)
    policies = []
    for mode in order:
        policy = CollectiveNativeConfig(custom_allreduce_algo=mode)
        policy.resolve()
        policies.append(_transfer(policy))
    assert os.environ == before
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", "changed")
    for policy, mode in zip(policies, order):
        policy.resolve()
        assert policy.raw("custom_allreduce_algo") == mode
    assert policies[0].compute_hash() != policies[1].compute_hash()


def _transfer(value):
    sender, receiver = Pipe()
    try:
        sender.send(value)
        return receiver.recv()
    finally:
        sender.close()
        receiver.close()


def test_native_sentinel_and_diagnostics(monkeypatch):
    monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raising=False)
    unset = CollectiveNativeConfig()
    unset.resolve()
    empty = CollectiveNativeConfig(custom_allreduce_algo="")
    empty.resolve()
    assert unset.raw("custom_allreduce_algo") is None
    assert empty.raw("custom_allreduce_algo") == ""
    traced = CollectiveNativeConfig(sm70_profile_trace=True)
    traced.resolve()
    assert traced.compute_hash() == unset.compute_hash()
    assert unset.raw("sm70_tp4_push_allreduce_small_messages") == "1"
    assert unset.raw("sm70_tp4_push_allreduce_concurrency") == "1"
    unset.active = empty.active = False
    assert unset.compute_hash() == empty.compute_hash()


def test_parent_and_native_hierarchy_have_one_effective_value():
    policy = CommunicationPolicy(
        native=CollectiveNativeConfig(sm70_tp8_hierarchical_custom_ar=True)
    )
    policy.resolve()
    assert policy.tp8_hierarchical
    with pytest.raises(ValueError, match="Conflicting typed requests"):
        CommunicationPolicy(
            tp8_hierarchical=False,
            native=CollectiveNativeConfig(sm70_tp8_hierarchical_custom_ar=True),
        ).resolve()


def test_configured_native_abi_order_matches_python():
    root = Path(__file__).resolve().parents[2]
    fields = re.findall(
        r'COLLECTIVE_POLICY_FIELD\((\w+),\s*"([^"]+)"\)',
        (root / "csrc/custom_all_reduce_policy_fields.inc").read_text(),
    )
    assert fields == list(CollectiveNativeConfig.aliases.items())


def test_binding_keeps_original_pointer_owner(monkeypatch):
    import vllm._custom_ops as ops

    original = SimpleNamespace(
        init_custom_ar_configured=Mock(return_value=123), dispose=Mock()
    )
    replacement = SimpleNamespace(dispose=Mock())
    monkeypatch.setattr(ops, "_custom_ar_owner_namespace", lambda: original)
    policy = CollectiveNativeConfig()
    policy.resolve()
    binding = NativeCollectiveBindings(policy)
    assert binding.available
    pointer = binding.init_custom_ar([1, 2], "rank_data", 0, True)
    original.init_custom_ar_configured.assert_called_once_with(
        [1, 2], "rank_data", 0, True, list(policy.values)
    )
    monkeypatch.setattr(ops, "_custom_ar_owner_namespace", lambda: replacement)
    binding.dispose(pointer)
    original.dispose.assert_called_once_with(123)
    replacement.dispose.assert_not_called()
    assert not NativeCollectiveBindings(policy).available


def test_missing_configured_abi_falls_back_before_allocating(monkeypatch):
    import vllm.distributed.device_communicators.custom_all_reduce as module

    monkeypatch.setattr(module.NativeCollectiveBindings, "available", False)
    allocate = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(module.CustomAllreduce, "create_shared_buffer", allocate)
    comm = module.CustomAllreduce(None, "cpu")
    assert comm.disabled
    allocate.assert_not_called()


def test_topology_and_diagnostic_fields_do_not_split_other_caches():
    default = CollectiveNativeConfig()
    different = CollectiveNativeConfig(
        sm70_tp8_hierarchical_push_blocks=4, sm70_profile_trace=True
    )
    for policy in (default, different):
        policy.resolve()
        policy.finalize_hash(4, True)
    assert default.compute_hash() == different.compute_hash()
    for policy in (default, different):
        policy.finalize_hash(8, True)
    assert default.compute_hash() != different.compute_hash()
    for policy in (default, different):
        policy.finalize_hash(8, False)
    assert default.compute_hash() == different.compute_hash()


def test_trace_seen_state_belongs_to_each_owner(monkeypatch):
    import torch

    from vllm.distributed.device_communicators import collective_provider as module

    log = Mock()
    monkeypatch.setattr(module.logger, "warning", log)
    first = module.CollectiveTrace(True, "tp:0", True, False, False)
    second = module.CollectiveTrace(True, "tp:0", True, False, False)
    for trace in (first, second, first, second):
        trace.record("custom", torch.empty(1, 8))
    assert log.call_count == 2
    assert first.seen is not second.seen


def test_legacy_initializer_supplies_defaults_without_environment_writes(monkeypatch):
    import vllm._custom_ops as ops

    for field in (
        "sm70_tp4_push_allreduce_small_messages",
        "sm70_tp4_push_allreduce_concurrency",
    ):
        monkeypatch.delenv(CollectiveNativeConfig.aliases[field], raising=False)
    before = dict(os.environ)
    native = SimpleNamespace(init_custom_ar_configured=Mock(return_value=321))
    monkeypatch.setattr(ops, "_custom_ar_owner_namespace", lambda: native)
    assert ops.init_custom_ar([], "rank_data", 0, True) == 321
    values = native.init_custom_ar_configured.call_args.args[-1]
    assert values[:2] == ["1", "1"]
    assert os.environ == before
