# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

import vllm.envs as envs
import vllm.models.qwen4_exp.nvidia.ops.hc as hc


@pytest.mark.parametrize(
    "rows,hidden,dtype,sm70,expected",
    [
        (1, 2560, torch.float16, True, True),
        (2, 2560, torch.float16, True, False),
        (1, 2560, torch.float32, True, False),
        (1, 1280, torch.float16, True, False),
        (1, 2560, torch.float16, False, False),
    ],
)
def test_hc_norm_prefetch_is_exact_decode_shape_only(
    monkeypatch, rows, hidden, dtype, sm70, expected
) -> None:
    monkeypatch.setenv("VLLM_SM70_QWEN38_HC_BATCH_NORM_PREFETCH", "0")
    envs.disable_envs_cache()
    kernel = MagicMock()
    monkeypatch.setattr(hc, "_hc_combine_norm_kernel", kernel)
    monkeypatch.setattr(hc.current_platform, "is_device_capability", lambda cap: sm70)
    monkeypatch.setattr(hc.current_platform, "is_arch_support_pdl", lambda: False)
    residual = torch.empty(rows, 4 * hidden, dtype=dtype)
    combined, normalized = hc._hc_combine_norm(
        residual,
        torch.empty(rows, hidden, dtype=dtype),
        torch.empty(rows, 4, dtype=dtype),
        torch.empty(hidden, dtype=dtype),
        1e-6,
        4,
    )
    assert combined.shape == normalized.shape == residual.shape
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    assert launch.call_args.kwargs["PREFETCH_WEIGHT"] is expected


@pytest.mark.parametrize("rows", [2, 4, 8, 16, 17, 32])
@pytest.mark.parametrize("enabled", [False, True])
def test_hc_batch_prefetch_preserves_reduction_tile(monkeypatch, rows, enabled):
    monkeypatch.setenv("VLLM_SM70_QWEN38_HC_BATCH_NORM_PREFETCH", str(int(enabled)))
    envs.disable_envs_cache()
    kernel = MagicMock()
    monkeypatch.setattr(hc, "_hc_combine_norm_kernel", kernel)
    monkeypatch.setattr(hc.current_platform, "is_device_capability", lambda cap: True)
    monkeypatch.setattr(hc.current_platform, "is_arch_support_pdl", lambda: False)
    residual = torch.empty(rows, 10240, dtype=torch.float16)
    hc._hc_combine_norm(
        residual,
        torch.empty(rows, 2560, dtype=torch.float16),
        torch.empty(rows, 4, dtype=torch.float16),
        torch.empty(2560, dtype=torch.float16),
        1e-6,
        4,
    )
    launch = kernel.__getitem__.return_value
    assert launch.call_args.kwargs["BLOCK_SIZE"] == 512
    assert launch.call_args.kwargs["PREFETCH_WEIGHT"] == (enabled and rows <= 16)
