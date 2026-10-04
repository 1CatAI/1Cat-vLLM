# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.distributed import parallel_state
from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc


def test_local_batch_hc_uses_tp_collective_without_full_mesh(monkeypatch):
    calls = []
    group = SimpleNamespace(
        world_size=4,
        rank_in_group=1,
        device_communicator=SimpleNamespace(ca_comm=None),
    )

    def all_reduce(x):
        calls.append(tuple(x.shape))
        return x

    group.all_reduce = all_reduce
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(hc, "_batch_runtime_ok", lambda *_: True)
    monkeypatch.setattr(ops, "supports_sm70_qwen38_hc_local", lambda: True)

    def down(x, packed, partials, output, rank):
        assert partials.dtype == torch.float32
        assert partials.shape == (20, 5, 96)
        assert rank == 1
        output.fill_(2)

    def up(lora, packed, x, output, rank):
        assert lora.is_contiguous() and lora.shape == (5, 320)
        assert torch.all(lora == 2)
        output.fill_(3)

    monkeypatch.setattr(ops, "sm70_qwen38_hc_down_local", down)
    monkeypatch.setattr(ops, "sm70_qwen38_hc_up_local", up)
    x = torch.empty(5, 10240, dtype=torch.float16)
    block, injection = hc._qwen38_sm70_fp16_fused_hc(
        x, None, None, None, None, concurrent_batch=True
    )
    assert calls == [(5, 336), (5, 2560)]
    assert torch.all(block == 3)
    assert injection.shape == (5, 4) and torch.all(injection == 2)


def test_legacy_fp16_partial_policy_is_not_replaced(monkeypatch):
    group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=None))
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(hc, "_batch_runtime_ok", lambda *_: True)
    monkeypatch.setattr(hc, "_runtime_ok", lambda *_: False)

    def unexpected(*_):
        raise AssertionError("local FP32 route must not replace FP16 partial policy")

    def dense(*_):
        raise RuntimeError("dense fallback")

    monkeypatch.setattr(ops, "sm70_qwen38_hc_down_local", unexpected)
    monkeypatch.setattr(torch.nn.functional, "linear", dense)
    with pytest.raises(RuntimeError, match="dense fallback"):
        hc._qwen38_sm70_fp16_fused_hc(
            torch.empty(5, 10240), None, None, None, None, concurrent_batch=False
        )


def test_replicated_hc_packing_preserves_dense_weights():
    generator = torch.Generator().manual_seed(20261004)
    down = torch.randn(336, 10240, generator=generator, dtype=torch.float16)
    up = torch.randn(10240, 320, generator=generator, dtype=torch.float16)
    packed_down = hc._pack_hc_batch_weight(down, "down", None)
    restored_down = packed_down.permute(0, 3, 1, 2, 4).reshape(352, 10240)
    torch.testing.assert_close(restored_down[:336], down, rtol=0, atol=0)
    assert torch.count_nonzero(restored_down[336:]) == 0
    packed_up = hc._pack_hc_batch_weight(up, "up", None)
    restored_up = packed_up.permute(3, 0, 4, 1, 2, 5).reshape(10240, 320)
    torch.testing.assert_close(restored_up, up, rtol=0, atol=0)
