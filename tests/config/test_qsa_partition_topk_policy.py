# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing

from vllm.config.sm70_sparse import Sm70SparseConfig


def test_partition_policy_transfer_and_qualified_hash():
    control = Sm70SparseConfig()
    candidate = Sm70SparseConfig(qsa_decode_partition_topk=True)
    assert not control.qsa_decode_partition_topk
    assert candidate.compute_hash() == control.compute_hash()
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        sender.send(candidate)
        copied = receiver.recv()
    finally:
        sender.close()
        receiver.close()
    assert copied.qsa_decode_partition_topk
    for policy in (control, candidate):
        policy.active = True
        policy.hash_fields = ("qsa_mtp_topk",)
    assert candidate.compute_hash() != control.compute_hash()
    for policy in (control, candidate):
        policy.hash_fields = ("indexer_decode_cublas",)
    assert candidate.compute_hash() == control.compute_hash()
