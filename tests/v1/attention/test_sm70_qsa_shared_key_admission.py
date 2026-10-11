# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for public QSA admission and installed-binary compatibility."""

from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops import sm70_qsa_shared_key as scorer


def descriptor(shape, dtype=torch.float16, device="cuda:0", contiguous=True):
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=torch.device(device),
        is_cuda=device.startswith("cuda"),
        numel=lambda: prod(shape),
        is_contiguous=lambda: contiguous,
    )


def inputs(rows=9):
    return [
        descriptor((rows, 4, 128)),
        descriptor((128, 16, 1, 128)),
        descriptor((1, 128), torch.int32),
        descriptor((rows,), torch.int32),
        descriptor((rows,), torch.int64),
        descriptor((1,), torch.int32),
    ]


@pytest.fixture
def native(monkeypatch):
    module = SimpleNamespace(MAX_QUERY_ROWS=65535 * 8)
    monkeypatch.setattr(scorer.current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setattr(scorer, "load_operator", lambda: True)
    monkeypatch.setattr(scorer.logger, "info_once", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        scorer, "importlib", SimpleNamespace(import_module=lambda _: module)
    )
    scorer._native_query_limit.cache_clear()
    yield module
    scorer._native_query_limit.cache_clear()


@pytest.mark.parametrize("rows", [2, 8, 9, 16, 17, 32, 65535 * 8])
def test_row_tiles_cover_interiors_and_native_grid_boundary(native, rows):
    assert scorer.shared_key_reason(*inputs(rows)) is None


def test_legacy_binary_keeps_its_fallback_instead_of_receiving_extra_rows(native):
    del native.MAX_QUERY_ROWS
    assert scorer.shared_key_reason(*inputs(8)) is None
    assert scorer.shared_key_reason(*inputs(9)) == "native_query_row_limit"
    # Admission rejects before allocating CUDA outputs or invoking the operator.
    assert scorer.maybe_shared_key_scores(*inputs(9), 4, 2048, 1.0) is None


def test_native_grid_limit_is_not_a_model_limit(native):
    assert (
        scorer.shared_key_reason(*inputs(native.MAX_QUERY_ROWS + 1))
        == "native_query_row_limit"
    )


@pytest.mark.parametrize(
    "index,replacement,reason",
    [
        (0, descriptor((9, 3, 128)), "requires_H4_D128"),
        (0, descriptor((1, 4, 128)), "requires_multiple_query_rows"),
        (1, descriptor((128, 16, 2, 128)), "requires_single_head_D128_cache"),
        (
            1,
            descriptor((128, 16, 1, 128), torch.float32),
            "requires_FP16_query_and_indexer_cache",
        ),
        (2, descriptor((2, 64), torch.int32), "multiple_requests"),
        (2, descriptor((1, 128), torch.int64), "requires_int32_request_metadata"),
        (3, descriptor((8,), torch.int32), "insufficient_row_metadata"),
        (4, descriptor((9,), torch.float32), "requires_integer_positions"),
        (
            4,
            descriptor((9,), torch.int64, contiguous=False),
            "noncontiguous_row_metadata",
        ),
        (5, descriptor((0,), torch.int32), "missing_request_length"),
        (
            5,
            descriptor((1,), torch.int32, device="cuda:1"),
            "requires_same_CUDA_device",
        ),
    ],
)
def test_other_capability_and_metadata_constraints_remain(
    native, index, replacement, reason
):
    args = inputs()
    args[index] = replacement
    assert scorer.shared_key_reason(*args) == reason


def test_missing_binary_and_wrong_device_remain_fallbacks(native, monkeypatch):
    monkeypatch.setattr(scorer, "load_operator", lambda: False)
    assert scorer.shared_key_reason(*inputs()) == "native_extension_unavailable"
    monkeypatch.setattr(
        scorer.current_platform, "is_device_capability", lambda _: False
    )
    assert scorer.shared_key_reason(*inputs()) == "requires_SM70"


def test_model_compatibility_module_uses_the_same_provider():
    from vllm.models.qwen4_exp.nvidia.ops import qsa_shared_key as legacy

    assert legacy.load_operator is scorer.load_operator
    assert legacy.shared_key_reason is scorer.shared_key_reason
    assert legacy.maybe_shared_key_scores is scorer.maybe_shared_key_scores
