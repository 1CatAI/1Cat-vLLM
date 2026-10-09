# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility of shared bindings after the Flash-V100 package move."""

import pytest

pytestmark = pytest.mark.cpu_test


def test_legacy_public_patch_reaches_new_package(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as legacy
    from vllm.v1.attention.backends import flash_v100 as package
    from vllm.v1.attention.backends.flash_v100 import dense_prefill

    replacement = object()
    monkeypatch.setattr(legacy, "flash_v100_dense_prefill_lse", replacement)
    assert package.flash_v100_dense_prefill_lse is replacement
    assert dense_prefill.flash_v100_dense_prefill_lse is replacement


def test_legacy_route_state_and_logger_patch_reach_shutdown(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.v1.attention.backends import flash_attn_v100 as legacy
    from vllm.v1.attention.backends.flash_v100 import routing
    from vllm.v1.worker.gpu.shutdown import log_loaded_attention_route_summaries

    logger = MagicMock()
    counts = {"decode_xqa_paged": 2}
    monkeypatch.setattr(legacy, "logger", logger)
    monkeypatch.setattr(legacy, "_route_counts", counts)
    log_loaded_attention_route_summaries()
    log_loaded_attention_route_summaries()
    logger.info.assert_called_once_with(
        "FLASH_ATTN_V100 route summary: %s", '{"decode_xqa_paged": 2}'
    )
    assert routing._route_counts is counts
    assert counts == {}


def test_prefill_log_flag_has_one_owner(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as legacy
    from vllm.v1.attention.backends.flash_v100 import dense_prefill, impl

    monkeypatch.setattr(legacy, "_logged_prefill_fa2_d256", True)
    assert dense_prefill._logged_prefill_fa2_d256
    assert "_logged_prefill_fa2_d256" not in vars(impl)
    dense_prefill._logged_prefill_fa2_d256 = False
    assert not legacy._logged_prefill_fa2_d256


def test_registry_and_legacy_share_backend_classes():
    from vllm.v1.attention.backends import flash_attn_v100 as legacy
    from vllm.v1.attention.backends import flash_v100 as package
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert (
        AttentionBackendEnum.FLASH_ATTN_V100.get_class() is legacy.FlashAttnV100Backend
    )
    assert package.FlashAttnV100Impl is legacy.FlashAttnV100Impl
    assert package.FlashAttnV100MetadataBuilder is legacy.FlashAttnV100MetadataBuilder


def test_legacy_star_import_exports_backend_api():
    from vllm.v1.attention.backends import flash_v100 as package

    namespace: dict[str, object] = {}
    exec("from vllm.v1.attention.backends.flash_attn_v100 import *", namespace)
    for name in package.__all__:
        assert namespace[name] is getattr(package, name)
