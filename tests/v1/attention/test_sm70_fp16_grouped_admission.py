# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.sm70_fp16_grouped import grouped_fp16_fp32_reason


def descriptor(shape, dtype=torch.float16, strides=None):
    if strides is None:
        strides = (832 * 256, 256, 256, 1)
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=torch.device("cuda:0"),
        is_contiguous=lambda: True,
        data_ptr=lambda: 16,
        stride=lambda dim=None: strides if dim is None else strides[dim],
    )


def scenario(groups=1, rows=8):
    instance = SimpleNamespace(
        flash_attn_grouped_fp16_fp32_paged=lambda: None,
        kv_cache_dtype="auto",
        use_smallq_decode_xqa=True,
        _flash_v100_window_size=lambda **kwargs: (-1, -1),
    )
    q = descriptor((rows, 6, 256))
    k, v = (
        descriptor((316 * groups, 832, 1, 256)),
        descriptor((316 * groups, 832, 1, 256)),
    )
    table = descriptor((rows, 316), torch.int32)
    lengths = descriptor((rows,), torch.int32)
    metadata = SimpleNamespace(
        block_table=descriptor((groups, 316), torch.int32),
        seq_lens=descriptor((groups,), torch.int32),
        causal=True,
    )
    return [instance, q, k, v, table, lengths, metadata]


def reason(inputs):
    return grouped_fp16_fp32_reason(
        *inputs, out=descriptor(inputs[1].shape), partition_size_hint=None
    )


@pytest.mark.parametrize("groups,rows", [(1, 2), (1, 8), (2, 16), (4, 32)])
def test_fp16_grouped_request_major_admission(groups, rows):
    assert reason(scenario(groups, rows)) is None


@pytest.mark.parametrize(
    "change,expected",
    [
        ("missing", "operator_missing:sm70_grouped_fp16_fwd"),
        ("fp8", "kv_dtype"),
        ("head", "head_shape"),
        ("page", "kv_shape_or_unmeasured_page"),
        ("capacity", "context_capacity"),
        ("cpu", "device_not_cuda"),
        ("row_dtype", "metadata_layout"),
        ("window", "causal_or_window"),
    ],
)
def test_fp16_grouped_declines_incompatible_descriptor(change, expected):
    inputs = scenario()
    if change == "missing":
        inputs[0].flash_attn_grouped_fp16_fp32_paged = None
    elif change == "fp8":
        inputs[0].kv_cache_dtype = "fp8_e4m3"
    elif change == "head":
        inputs[1].shape = (8, 8, 256)
    elif change == "page":
        inputs[2].shape = inputs[3].shape = (1, 1648, 1, 256)
    elif change == "capacity":
        inputs[6].block_table.shape = (1, 400)
    elif change == "cpu":
        inputs[1].device = torch.device("cpu")
    elif change == "row_dtype":
        inputs[5].dtype = torch.int64
    elif change == "window":
        inputs[0]._flash_v100_window_size = lambda **kwargs: (2048, 0)
    assert reason(inputs) == expected


def test_fp16_grouped_rejects_unqualified_batch_and_row_counts():
    assert reason(scenario(5, 40)) == "request_group_count"
    assert reason(scenario(4, 20)) == "query_rows"
    assert reason(scenario(1, 1)) == "query_rows"
