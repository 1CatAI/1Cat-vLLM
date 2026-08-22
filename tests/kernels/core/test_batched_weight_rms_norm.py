# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the batched-weight RMS norm kernel.

The outermost input dimension selects the corresponding weight row. DFlash
uses this for its fused per-layer K normalization.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="rms_norm requires a CUDA/ROCm device",
)


@pytest.mark.parametrize(
    "shape",
    [
        (28, 17, 128),
        (1, 5, 2, 128),
        (28, 13, 8, 128),
        (6, 3, 4, 769),
    ],
)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_rms_norm_matches_loop(
    shape: tuple[int, ...], dtype: torch.dtype, seed: int
) -> None:
    set_random_seed(seed)
    torch.set_default_device("cuda")

    num_rows, hidden = shape[0], shape[-1]
    eps = 1e-6
    x = torch.randn(*shape, dtype=dtype) * 0.1
    weight = torch.randn(num_rows, hidden, dtype=dtype) * 0.1 + 1.0

    out_ref = torch.empty_like(x)
    for i in range(x.shape[0]):
        ops.rms_norm(out_ref[i], x[i], weight[i], eps)

    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, eps)
    torch.testing.assert_close(out, out_ref, atol=0, rtol=0)


@torch.inference_mode()
def test_rms_norm_validates_shapes() -> None:
    torch.set_default_device("cuda")

    x = torch.randn(4, 8, 128, dtype=torch.float)
    out = torch.empty_like(x)
    with pytest.raises(RuntimeError):
        ops.rms_norm(out, x, torch.randn(3, 128), 1e-6)
    with pytest.raises(RuntimeError):
        ops.rms_norm(out, x, torch.randn(4, 64), 1e-6)
