# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch limits are typed engine policy and survive worker serialization."""

import copy
import multiprocessing

import pytest

from vllm.config.kernel import KernelConfig


def _send_worker_policy(connection, policy):
    connection.send(policy)
    connection.close()


def test_mtp_batch_policy_is_owned_hashed_and_serializable():
    legacy = KernelConfig()
    tiled = KernelConfig(
        sm70_hcx=True,
        sm70_hcx_batch_max_m=40,
        qsa_host_kv_state_compact=True,
        sm70_qsa_shared_key_batch=True,
        sm70_gguf={"small_m_dp4a": True, "small_m_dp4a_max_m": 40},
        qsa_host_kv=True,
        qsa_host_indexer_history=True,
        sm70_qsa_indexer_single_rank=True,
    )
    tiled.sm70_gguf.active = True
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_send_worker_policy, args=(sender, tiled))
    worker.start()
    sender.close()
    try:
        assert receiver.poll(60), "Worker policy transfer timed out"
        transferred = receiver.recv()
    finally:
        receiver.close()
        worker.join(10)
        if worker.is_alive():
            worker.terminate()
            worker.join()
    assert worker.exitcode == 0
    assert legacy.sm70_hcx_batch_max_m == 8
    assert legacy.sm70_gguf.small_m_dp4a_max_m == 20
    assert not legacy.qsa_host_kv_state_compact
    assert transferred == tiled
    assert transferred.compute_hash() == tiled.compute_hash()
    for field in (
        "sm70_hcx_batch_max_m",
        "qsa_host_kv_state_compact",
        "sm70_qsa_shared_key_batch",
        "sm70_qsa_indexer_single_rank",
    ):
        changed = copy.deepcopy(tiled)
        setattr(changed, field, getattr(legacy, field))
        assert changed.compute_hash() != tiled.compute_hash()
    transferred.sm70_gguf.small_m_dp4a_max_m = 20
    assert transferred.compute_hash() != tiled.compute_hash()
    assert tiled.sm70_gguf.small_m_dp4a_max_m == 40


def test_disabled_parents_keep_inactive_batch_options_out_of_hash():
    a = KernelConfig(sm70_hcx=False, sm70_qsa_shared_key=False)
    b = KernelConfig(
        sm70_hcx=False,
        sm70_qsa_shared_key=False,
        sm70_hcx_batch_max_m=40,
        sm70_qsa_shared_key_batch=True,
    )
    assert a.compute_hash() == b.compute_hash()


@pytest.mark.parametrize(
    "host, history", [(False, False), (False, True), (True, False)]
)
def test_single_rank_indexer_is_inactive_without_host_history(host, history):
    a = KernelConfig(qsa_host_kv=host, qsa_host_indexer_history=history)
    b = KernelConfig(
        qsa_host_kv=host,
        qsa_host_indexer_history=history,
        sm70_qsa_indexer_single_rank=True,
    )
    assert a.compute_hash() == b.compute_hash()


@pytest.mark.parametrize("limit", [0, 41])
def test_mtp_batch_policy_rejects_unsupported_limits(limit):
    with pytest.raises(ValueError):
        KernelConfig(sm70_hcx_batch_max_m=limit)
    with pytest.raises(ValueError):
        KernelConfig(sm70_gguf={"small_m_dp4a_max_m": limit})
