# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing

import pytest

from vllm.config import KernelConfig


def test_qsa_stage_slots_are_bounded_serialized_and_hashed():
    assert KernelConfig().qsa_host_kv_prefill_stage_slots == 2
    policies = [
        KernelConfig(
            qsa_host_kv=True,
            qsa_host_kv_prefill_stage_tokens=262144,
            qsa_host_kv_prefill_stage_slots=slots,
        )
        for slots in (1, 2)
    ]
    for policy in policies:
        receiver, sender = multiprocessing.Pipe(duplex=False)
        try:
            sender.send(policy)
            copied = receiver.recv()
        finally:
            sender.close()
            receiver.close()
        assert (
            copied.qsa_host_kv_prefill_stage_slots
            == policy.qsa_host_kv_prefill_stage_slots
        )
        assert copied.compute_hash() == policy.compute_hash()
    assert policies[0].compute_hash() != policies[1].compute_hash()
    for inactive in (
        {},
        {"qsa_host_kv": True},
        {"qsa_host_kv_prefill_stage_tokens": 262144},
    ):
        assert (
            KernelConfig(**inactive, qsa_host_kv_prefill_stage_slots=1).compute_hash()
            == KernelConfig(
                **inactive, qsa_host_kv_prefill_stage_slots=2
            ).compute_hash()
        )
    for slots in (0, 3):
        with pytest.raises(ValueError):
            KernelConfig(qsa_host_kv_prefill_stage_slots=slots)
