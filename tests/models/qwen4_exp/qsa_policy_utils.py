# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind explicit policies in tests that previously changed import snapshots."""

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops


def set_qsa_option(monkeypatch, field, value):
    policy = qsa_ops.sparse_policy()
    setattr(policy, field, value)
    policy.sources[field] = "typed"
    monkeypatch.setattr(qsa_ops, "sparse_policy", lambda: policy)
