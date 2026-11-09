# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as ops


@pytest.fixture(autouse=True)
def workspace_cache(monkeypatch):
    saved = dict(ops._SM70_QSA_XQA_PAGE4_WORKSPACES)
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=123)
    )
    yield
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.update(saved)


def _query(rows=3):
    return torch.zeros(rows, 6, 256, dtype=torch.float16)


@pytest.mark.parametrize(
    "kv_dtype,expected",
    [("fp8_e4m3", torch.float32), ("auto", torch.float16), ("float16", torch.float16)],
)
def test_workspace_matches_native_output_contract_on_cpu(kv_dtype, expected):
    temporary, maximum, sums, active = ops._qsa_xqa_page4_workspace(
        _query(), 8, kv_dtype
    )
    assert temporary.dtype == expected
    assert temporary.shape == (3, 6, 8, 256)
    assert maximum.dtype == sums.dtype == torch.float32
    assert active.dtype == torch.int32 and active.tolist() == [8]


def test_workspace_cache_separates_incompatible_dtypes():
    query = _query()
    e4m3 = ops._qsa_xqa_page4_workspace(query, 8, "fp8_e4m3")[0]
    fp16 = ops._qsa_xqa_page4_workspace(query, 8, "auto")[0]
    assert e4m3.data_ptr() != fp16.data_ptr()
    cached_e4m3 = ops._qsa_xqa_page4_workspace(query, 8, "fp8_e4m3")[0]
    cached_fp16 = ops._qsa_xqa_page4_workspace(query, 8, "float16")[0]
    assert cached_e4m3.data_ptr() == e4m3.data_ptr()
    assert cached_fp16.data_ptr() == fp16.data_ptr()


def test_workspace_cache_separates_streams(monkeypatch):
    first = ops._qsa_xqa_page4_workspace(_query(), 8, "fp8_e4m3")[0]
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=124)
    )
    second = ops._qsa_xqa_page4_workspace(_query(), 8, "fp8_e4m3")[0]
    assert first.data_ptr() != second.data_ptr()


def test_workspace_grows_without_reusing_an_undersized_allocation():
    small = ops._qsa_xqa_page4_workspace(_query(), 8, "fp8_e4m3")[0]
    large = ops._qsa_xqa_page4_workspace(_query(7), 8, "fp8_e4m3")[0]
    assert large.shape == (7, 6, 8, 256)
    assert large.data_ptr() != small.data_ptr()
    again = ops._qsa_xqa_page4_workspace(_query(5), 8, "fp8_e4m3")[0]
    assert again.data_ptr() == large.data_ptr()


def test_workspace_partition_count_is_part_of_the_cache_key():
    first = ops._qsa_xqa_page4_workspace(_query(), 8, "fp8_e4m3")
    second = ops._qsa_xqa_page4_workspace(_query(), 16, "fp8_e4m3")
    assert second[0].shape == (3, 6, 16, 256)
    assert second[0].data_ptr() != first[0].data_ptr()
    assert second[3].tolist() == [16]
