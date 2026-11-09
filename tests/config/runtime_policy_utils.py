# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-only fixtures for default checkpoints; never load a model."""

from types import SimpleNamespace

from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    KernelConfig,
    ObservabilityConfig,
    OffloadConfig,
)
from vllm.config.execution_policy import CommunicationPolicy
from vllm.config.policy_defaults import PolicyDefaults
from vllm.config.sm70_dflash2 import Sm70DFlash2Config


def make_policy_defaults():
    cfg = SimpleNamespace(
        compilation_config=CompilationConfig(),
        kernel_config=KernelConfig(),
        observability_config=ObservabilityConfig(),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=AttentionConfig(),
        offload_config=OffloadConfig(),
        parallel_config=SimpleNamespace(communication=CommunicationPolicy()),
        speculative_config=SimpleNamespace(sm70_dflash2=Sm70DFlash2Config()),
        runtime_default_sources={},
    )
    return PolicyDefaults(cfg)
