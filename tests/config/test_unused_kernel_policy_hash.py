# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.kernel import KernelConfig


@pytest.mark.parametrize("policy", ["sm70_skinny_moe", "fused_fp16_aux_gemv"])
def test_unused_policy_does_not_partition_compile_cache(policy):
    enabled = KernelConfig()
    disabled = KernelConfig(**{policy: False})
    assert enabled.compute_hash() == disabled.compute_hash()


@pytest.mark.parametrize("policy", ["sm70_skinny_moe", "fused_fp16_aux_gemv"])
def test_loaded_policy_keeps_enabled_and_disabled_graphs_distinct(policy):
    enabled = KernelConfig()
    disabled = KernelConfig(**{policy: False})
    setattr(enabled, f"{policy}_applicable", True)
    setattr(disabled, f"{policy}_applicable", True)
    assert enabled.compute_hash() != disabled.compute_hash()
