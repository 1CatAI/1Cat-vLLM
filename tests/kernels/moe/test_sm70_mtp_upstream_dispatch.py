# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing checks: tuning must retain upstream M1/M5 native admission."""

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe import fused_moe as moe
from vllm.triton_utils import tl


def test_legacy_warmup_context_restores_native_eligible_tiles(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_MTP_MOE_TUNED_CONFIG", "1")
    args = (1, 512, 160, 2560, 10)
    before = moe._get_sm70_mtp_moe_decode_config(*args)
    assert before is not None
    assert before["BLOCK_SIZE_N"] == 128
    with moe.force_sm70_mtp_moe_legacy_config():
        assert moe._get_sm70_mtp_moe_decode_config(*args) is None
    assert moe._get_sm70_mtp_moe_decode_config(*args) == before


class _SM70Platform:
    @staticmethod
    def is_cuda():
        return True

    @staticmethod
    def has_device_capability(capability):
        return capability == 70

    @staticmethod
    def is_device_capability(capability):
        return capability == 70


class _CudaMetadataTensor:
    """Only metadata is read; native/Triton calls are recorded, never launched."""

    is_cuda = True
    device = torch.device("cuda", 0)

    def __init__(self, shape, dtype):
        self._metadata = torch.empty(shape, dtype=dtype, device="meta")

    def __getattr__(self, name):
        return getattr(self._metadata, name)

    def data_ptr(self):
        return 16


@pytest.mark.parametrize("rows", [1, 5])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("native_enabled", [False, True])
def test_selected_mtp_config_preserves_upstream_projection_dispatch(
    monkeypatch, rows, weighted, native_enabled
):
    """Exercise the real tile selector and dispatcher for both expert GEMMs.

    The dispatcher's tile guard is part of the upstream native-op contract.
    A fast standalone Triton tile can accidentally make M1 ineligible, even
    when the native operator and its shape guard remain available.
    """
    monkeypatch.setattr(moe, "current_platform", _SM70Platform())
    monkeypatch.setenv("VLLM_SM70_MTP_MOE_TUNED_CONFIG", "1")
    monkeypatch.setenv("VLLM_SM70_MTP_MOE_FP16_EXACT", str(int(native_enabled)))
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    calls = []
    monkeypatch.setattr(
        torch.ops._C,
        "sm70_mtp_moe_fp16_out",
        lambda *args: calls.append(("native", args)),
        raising=False,
    )
    monkeypatch.setattr(
        moe,
        "invoke_fused_moe_triton_kernel",
        lambda *args: calls.append(("triton", args)),
    )

    n, k = (2560, 160) if weighted else (320, 2560)
    source = _CudaMetadataTensor((rows * 10 if weighted else rows, k), torch.float16)
    weight = _CudaMetadataTensor((512, n, k), torch.float16)
    output = _CudaMetadataTensor((rows, 10, n), torch.float16)
    topk_weights = _CudaMetadataTensor((rows, 10), torch.float32)
    expert_ids = _CudaMetadataTensor((rows * 10,), torch.int32)
    padded = _CudaMetadataTensor((1,), torch.int32)
    config = moe.get_default_config(rows, 512, 160, 2560, 10, None)

    moe.dispatch_fused_moe_kernel(
        source,
        weight,
        output,
        None,
        None,
        None,
        topk_weights,
        None,
        expert_ids,
        padded,
        weighted,
        1 if weighted else 10,
        config,
        tl.float16,
        False,
        False,
        False,
        False,
        False,
    )

    assert len(calls) == 1
    assert calls[0][0] == ("native" if native_enabled else "triton")
    if native_enabled:
        assert calls[0][1] == (
            output,
            source,
            weight,
            expert_ids,
            topk_weights,
            padded,
            weighted,
        )
