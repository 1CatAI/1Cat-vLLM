# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On Turing the FA2 build is tried first; FlashInfer is not offered there.

FlashInfer's paged prefill fails with "invalid argument" on sm75, so the
priority list for (7, 5) is FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION.
"""

from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_turing_tries_the_fa2_build_first():
    priorities = _get_backend_priorities(
        use_mla=False, device_capability=DeviceCapability(7, 5)
    )
    assert priorities[0] is AttentionBackendEnum.FLASH_ATTN
    assert AttentionBackendEnum.FLASHINFER not in priorities
    assert AttentionBackendEnum.TRITON_ATTN in priorities


def test_ampere_priorities_are_untouched():
    priorities = _get_backend_priorities(
        use_mla=False, device_capability=DeviceCapability(8, 0)
    )
    assert AttentionBackendEnum.FLASH_ATTN in priorities
    assert AttentionBackendEnum.FLASHINFER in priorities


def test_backend_priority_cache_isolated_between_engines(monkeypatch):
    from types import SimpleNamespace

    from vllm.config import execution_policy
    from vllm.platforms import cuda

    for enabled in (True, False, True, False):
        monkeypatch.setattr(
            execution_policy,
            "flash_v100_policy",
            lambda enabled=enabled: SimpleNamespace(enabled=enabled),
        )
        choices = cuda._get_backend_priorities(False, DeviceCapability(7, 0))
        assert (AttentionBackendEnum.FLASH_ATTN_V100 in choices) == enabled
