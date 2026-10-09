# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.sm70_graph_observer import GraphParityWorkerExtension


def test_host_prefill_policy_reaches_target_draft_and_deduplicates_shared_owners():
    target = SimpleNamespace(host_kv_enabled=True, host_kv_prefill_enabled=True)
    draft = SimpleNamespace(host_kv_enabled=True, host_kv_prefill_enabled=True)
    resident = SimpleNamespace(host_kv_enabled=False, host_kv_prefill_enabled=True)

    def config(context):
        return SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context),
            kernel_config=SimpleNamespace(qsa_host_kv_prefill=True),
        )

    main = config({"target": target, "resident": resident})
    speculative = config({"target": target, "draft": draft})
    worker = GraphParityWorkerExtension()
    worker.rank = 0
    worker.model_runner = SimpleNamespace(
        vllm_config=main, speculator=SimpleNamespace(vllm_config=speculative)
    )
    for enabled in (False, True):
        result = worker.set_qsa_host_prefill_policy(enabled)
        assert result == {"rank": 0, "enabled": enabled, "owners": 2}
        assert target.host_kv_prefill_enabled is enabled
        assert draft.host_kv_prefill_enabled is enabled
        assert resident.host_kv_prefill_enabled is True
        assert main.kernel_config.qsa_host_kv_prefill is enabled
        assert speculative.kernel_config.qsa_host_kv_prefill is enabled
    with pytest.raises(TypeError):
        worker.set_qsa_host_prefill_policy(1)
