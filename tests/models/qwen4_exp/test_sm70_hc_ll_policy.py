# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.kernel import KernelConfig
from vllm.distributed.device_communicators import sm70_hc_ll


def test_hc_runtime_reports_do_not_change_graph_hash():
    cfg = KernelConfig()
    before = cfg.compute_hash()
    cfg.collective_kernel_selections["hc_ll:tp"] = {
        "enabled": True,
        "rank_order": [0, 1, 2, 3],
    }
    assert cfg.compute_hash() == before
    cfg.hc_ll_shard = False
    assert cfg.compute_hash() != before


@pytest.mark.parametrize(
    "enabled,reason", [(False, "disabled_by_kernel_config"), (True, "requires_tp4")]
)
def test_rejected_owner_does_not_allocate(monkeypatch, enabled, reason):
    cfg = SimpleNamespace(kernel_config=KernelConfig(hc_ll_shard=enabled))
    monkeypatch.setattr(sm70_hc_ll, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(sm70_hc_ll.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(sm70_hc_ll.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        sm70_hc_ll.Sm70HcLLCommunicator,
        "_prepare",
        lambda self: pytest.fail("unsupported owner must not allocate"),
    )
    owner = sm70_hc_ll.Sm70HcLLCommunicator(None, None, "tp")
    assert owner.status["reason"] == reason
    assert not owner.status["enabled"]
    assert owner.pointers is None
    owner.close()
