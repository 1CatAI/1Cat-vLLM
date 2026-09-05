# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

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
