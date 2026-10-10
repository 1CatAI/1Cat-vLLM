# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config.gdn import GdnConfig
from vllm.model_executor.layers.fla.ops.sm70.gdn_prefill import (
    bind_flashqla_native_policy,
)


def worker_roundtrip(value):
    recv, send = multiprocessing.Pipe(duplex=False)
    try:
        send.send(value)
        return recv.recv()
    finally:
        send.close()
        recv.close()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, -1),
        ("", -1),
        (" 2junk", 2),
        ("bad", 0),
        ("0", 0),
        ("-1", 0),
        ("4294967295", 0),
        ("8", 8),
    ],
)
def test_flashqla_native_legacy_parser_and_worker_snapshot(monkeypatch, raw, expected):
    name = "FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    policy = GdnConfig()
    policy.resolve()
    assert policy.flashqla_column_groups == expected
    restored = worker_roundtrip(policy)
    monkeypatch.setenv(name, "4")
    restored.resolve()
    assert restored.flashqla_column_groups == expected
    typed = GdnConfig(flashqla_column_groups=1)
    typed.resolve()
    assert typed.flashqla_column_groups == 1


def test_inactive_flashqla_override_does_not_change_other_backend_hash(monkeypatch):
    policies = []
    for value in (1, 4):
        policy = GdnConfig(flashqla_column_groups=value, flashqla_decode=False)
        policy.resolve()
        policy.active_prefill_backend = "triton"
        policies.append(policy)
    assert policies[0].compute_hash() == policies[1].compute_hash()
    for policy in policies:
        policy.flashqla_decode = True
    assert policies[0].compute_hash() != policies[1].compute_hash()


def test_flashqla_owner_is_bound_once_and_old_binary_fails_at_initialization(
    monkeypatch,
):
    from flash_qla.ops.gated_delta_rule.chunk.sm70 import fused_fwd

    create = Mock(side_effect=lambda groups: SimpleNamespace(groups=groups))
    monkeypatch.setattr(
        fused_fwd,
        "_load_ext",
        lambda: SimpleNamespace(gdn_policy_abi_version=lambda: 1, GdnPolicy=create),
    )
    configs = [SimpleNamespace(), SimpleNamespace()]
    policies = [GdnConfig(flashqla_column_groups=value) for value in (1, 4)]
    owners = [
        bind_flashqla_native_policy(cfg, policy, needed=True)
        for cfg, policy in zip(configs, policies)
    ]
    for _ in range(2):
        for cfg, policy, owner in zip(configs, policies, owners):
            assert bind_flashqla_native_policy(cfg, policy, needed=True) is owner
    assert [owner.groups for owner in owners] == [1, 4]
    assert create.call_count == 2
    assert not worker_roundtrip(configs[0])._runtime_resources
    monkeypatch.setattr(fused_fwd, "_load_ext", lambda: SimpleNamespace())
    assert (
        bind_flashqla_native_policy(SimpleNamespace(), policies[0], needed=False)
        is None
    )
    with pytest.raises(RuntimeError, match="policy ABI 1"):
        bind_flashqla_native_policy(SimpleNamespace(), policies[0], needed=True)
