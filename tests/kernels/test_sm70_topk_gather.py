# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.sm70_topk_gather import (
    _pack,
    _unpack,
    packed_topk_reason,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_wire_preserves_float_bits_and_signed_ids(dtype: torch.dtype) -> None:
    bits_dtype = torch.int16 if dtype == torch.float16 else torch.int32
    # Include subnormal values, signed zero and distinct NaN payloads.
    patterns = (
        [0, -32768, 1, 1023, 31744, -1024, 32257, 31745]
        if dtype == torch.float16
        else [0, -(2**31), 1, 8388607, 2139095040, -8388608, 2143289345, 2139095041]
    )
    values = torch.tensor(patterns, dtype=bits_dtype, device="cuda").view(dtype)
    ids = torch.tensor([-1, 0, 1, 65536, 248319, 2**31 - 1, 9, 42], device="cuda")
    wire = torch.empty(values.numel() * 2, dtype=torch.int32, device="cuda")
    restored_values = torch.empty_like(values)
    restored_ids = torch.empty_like(ids)
    half = dtype == torch.float16
    _pack[(1,)](values, ids, wire, values.numel(), half, 256)
    _unpack[(1,)](wire, restored_values, restored_ids, values.numel(), half, 256)
    assert torch.equal(values.view(bits_dtype), restored_values.view(bits_dtype))
    assert torch.equal(ids, restored_ids)


@pytest.mark.parametrize("tp", [2, 4])
def test_candidate_guards_reject_unsupported_layouts(tp: int) -> None:
    values = torch.zeros((8, 64), device="cuda", dtype=torch.float32)
    ids = torch.zeros_like(values, dtype=torch.int64)
    kwargs = dict(vocab_size=248320, tp_size=tp)
    assert packed_topk_reason(values, ids, **kwargs) is None
    assert packed_topk_reason(values, ids, vocab_size=2**31, tp_size=tp) == (
        "vocabulary_exceeds_int32"
    )
    assert packed_topk_reason(values.bfloat16(), ids, **kwargs) == "value_dtype"
    assert packed_topk_reason(values, ids.int(), **kwargs) == "token_id_layout"
    assert packed_topk_reason(values[:, ::2], ids[:, ::2], **kwargs) == (
        "unmeasured_candidate_count"
    )
    assert packed_topk_reason(values.expand(2, -1, -1), ids, **kwargs) == (
        "unmeasured_row_count"
    )
    assert packed_topk_reason(values, ids, vocab_size=248320, tp_size=8) == (
        "unmeasured_tp_size"
    )


def test_explicit_policy_survives_sampling_without_config() -> None:
    from vllm.config import get_current_vllm_config_or_none

    assert get_current_vllm_config_or_none() is None
    values = torch.zeros((8, 64), device="cuda", dtype=torch.float32)
    ids = torch.zeros_like(values, dtype=torch.int64)
    assert (
        packed_topk_reason(values, ids, vocab_size=248320, tp_size=4, enabled=False)
        == "disabled_by_policy"
    )
    assert (
        packed_topk_reason(values, ids, vocab_size=248320, tp_size=4, enabled=True)
        is None
    )
