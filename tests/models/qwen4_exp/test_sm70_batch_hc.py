# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm import envs
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.forward_context import is_uniform_decode_metadata
from vllm.models.qwen4_exp.nvidia import sm70_batch_hc as hc

pytestmark = pytest.mark.skip_global_cleanup


def test_batch_hc_is_opt_in(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_SM70_QWEN38_BATCH_HC_FP16", raising=False)
    assert not envs.VLLM_SM70_QWEN38_BATCH_HC_FP16


@pytest.mark.parametrize("rows", (1, 2, 4, 8, 16, 32))
def test_cpu_metadata_is_independent_of_scheduler_and_kv(rows):
    assert is_uniform_decode_metadata(
        {
            "attn": NS(max_query_len=1),
            "gdn": NS(
                num_prefills=0,
                num_decodes=rows,
                num_decode_tokens=rows,
            ),
        }
    )


@pytest.mark.parametrize(
    "metadata",
    (
        None,
        {},
        [],
        {"unknown": NS()},
        {"attn": NS(max_query_len=5)},
        {"attn": NS(max_query_len=1), "gdn": NS(num_prefills=1)},
        {"gdn": NS(num_decodes=4, num_decode_tokens=8)},
        {"attn": NS(max_query_len=torch.tensor(1))},
    ),
)
def test_mixed_prefill_verify_unknown_context_falls_back(metadata):
    assert not is_uniform_decode_metadata(metadata)


@pytest.mark.parametrize("rows", (1, 17, 32))
def test_unsupported_width_delegates_to_original_fused_hc(monkeypatch, rows):
    x = torch.empty(rows, 10240, dtype=torch.float16)
    expected = (torch.empty(rows, 2560), torch.empty(rows, 4))
    monkeypatch.setattr(hc, "_channel", lambda: pytest.fail("unexpected batch channel"))
    monkeypatch.setattr(hc, "_qwen38_sm70_fp16_fused_hc", lambda *a: expected)
    dummy = torch.empty(0)
    assert hc._batch_hc(x, dummy, dummy, dummy, True, False) is expected


def test_original_gemv_fallback_is_not_replaced_by_linear(monkeypatch):
    x = torch.empty(1, 10240, dtype=torch.float16)
    seen = []

    def gemv(x, weight):
        seen.append("gemv")
        return torch.zeros(1, 336, dtype=x.dtype)

    monkeypatch.setattr(hc, "_qwen38_sm70_fp16_gemv", gemv)
    monkeypatch.setattr(hc, "hc_silu", lambda a, n: a)
    monkeypatch.setattr(hc, "hc_gate_mix", lambda a, b, n: b[:, :2560])
    hc._batch_hc(
        x,
        torch.empty(0),
        torch.zeros(10240, 320, dtype=x.dtype),
        torch.empty(0),
        False,
        True,
    )
    assert seen == ["gemv"]


@pytest.mark.parametrize("rank", range(4))
def test_up_shard_reload_keeps_captured_pointer_and_is_nonpersistent(rank):
    child = nn.Module()
    weight = torch.arange(10240, dtype=torch.float16)[:, None].expand(-1, 320).clone()
    hc._copy_up_shard(child, weight, rank)
    pointer = child._sm70_batch_hc_up.data_ptr()
    assert not child.state_dict()
    expected = weight.view(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640]
    torch.testing.assert_close(child._sm70_batch_hc_up, expected.reshape(2560, 320))
    weight.zero_()
    hc._copy_up_shard(child, weight, rank)
    assert child._sm70_batch_hc_up.data_ptr() == pointer
    assert torch.count_nonzero(child._sm70_batch_hc_up).item() == 0


def test_channel_destroy_is_idempotent_and_does_not_close_ordinary_channel():
    closed = []
    communicator = object.__new__(CudaCommunicator)
    communicator.sm70_hc_batch_comm = NS(close=lambda: closed.append("hc"))
    communicator.pynccl_comm = communicator.ca_comm = None
    communicator.fi_ar_comm = communicator.all2all_manager = None
    communicator.destroy()
    communicator.destroy()
    assert closed == ["hc"]


def test_native_capabilities_resolve_only_from_the_pointer_owner(monkeypatch):
    monkeypatch.setattr(ops, "_custom_ar_owner_namespace", lambda: NS())
    assert not ops.supports_sm70_qwen38_hc_batch()
    called = []
    owner = NS(
        sm70_qwen38_hc_batch_down=lambda *a: called.append("down"),
        sm70_qwen38_hc_batch_mix=lambda *a: called.append("mix"),
    )
    monkeypatch.setattr(ops, "_custom_ar_owner_namespace", lambda: owner)
    assert ops.supports_sm70_qwen38_hc_batch()
    dummy = torch.empty(0)
    ops.sm70_qwen38_hc_batch_down(1, dummy, dummy, dummy)
    ops.sm70_qwen38_hc_batch_mix(1, dummy, dummy, dummy)
    assert called == ["down", "mix"]


def test_fake_hc_never_touches_communicator(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    monkeypatch.setattr(hc, "_channel", lambda: pytest.fail("fake cannot use channel"))
    with FakeTensorMode():
        x = torch.empty(8, 10240, dtype=torch.float16)
        block, injection = torch.ops.vllm.qwen38_sm70_batch_hc(
            x,
            torch.empty(336, 10240),
            torch.empty(10240, 320),
            torch.empty(2560, 320),
            True,
            False,
        )
        assert block.shape == (8, 2560) and injection.shape == (8, 4)


def test_derived_weight_hook_runs_after_quant_postprocessing(monkeypatch):
    from vllm.model_executor.model_loader import utils

    events = []

    class Quant:
        def process_weights_after_loading(self, layer):
            events.append("quant")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = Quant()

        def prepare_sm70_batch_hc(self):
            assert events == ["quant"]
            events.append("hc")

    monkeypatch.setattr(utils, "QuantizeMethodBase", Quant)
    utils.process_weights_after_loading(
        Model(),
        NS(dtype=torch.float16, quantization=None),
        torch.device("cpu"),
    )
    assert events == ["quant", "hc"]
