# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


@pytest.mark.parametrize("supports_split", (False, True))
def test_hc_cta_default_and_old_extension_fallback(monkeypatch, supports_split):
    calls = []

    class Packet:
        _schemas = {"": "cta_split_warps" if supports_split else "fused_chain"}

        def __call__(self, *args):
            calls.append(args)

    monkeypatch.setattr(
        ops,
        "_custom_ar_owner_namespace",
        lambda: SimpleNamespace(sm70_qwen38_hc_batch=Packet()),
    )
    buffers = [object() for _ in range(8)]
    CustomAllreduce.sm70_qwen38_hc_batch(
        SimpleNamespace(_ptr=123), *buffers, fused_chain=True
    )
    assert calls[0][:9] == (123, *buffers)
    assert calls[0][9:13] == (False, False, False, True)
    assert calls[0][13:] == ((8,) if supports_split else ())
