# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 decode GEMV of k_group_partials vs the mma tile kernel (V100 only)."""

import pytest
import torch

from vllm.models.inkling2.nvidia.ops import is_sm70
from vllm.models.inkling2.nvidia.ops.linear import (
    SM70_GEMV_MAX_TOKENS,
    _k_group_kernel,
    k_group_partials,
)
from vllm.triton_utils import triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm70(), reason="SM70 GPU"
)


def _tile_partials(x, w):
    tokens, hidden = x.shape
    n = w.shape[0]
    groups = triton.cdiv(hidden, 1024)
    out = torch.empty((groups, tokens, n), dtype=torch.float32, device=x.device)
    _k_group_kernel[(triton.cdiv(n, 32), triton.cdiv(tokens, 16), groups)](
        x,
        w,
        out,
        tokens,
        n,
        hidden,
        16,
        32,
        True,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return out


@pytest.mark.parametrize("tokens", [1, 2, 4, SM70_GEMV_MAX_TOKENS])
@pytest.mark.parametrize("n,hidden", [(128, 2304), (32, 2304), (64, 4608)])
def test_gemv_partials_match_tile_kernel(tokens, n, hidden):
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16)
    w = (torch.randn(n, hidden, device="cuda") * 0.02).half()
    got = k_group_partials(x, w)
    ref = _tile_partials(x, w)
    assert got.shape == ref.shape == (triton.cdiv(hidden, 1024), tokens, n)
    # same fp32 K-group partials; only the summation order inside a group differs
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-4)
    full = x.float() @ w.float().t()
    torch.testing.assert_close(got.sum(0), full, rtol=1e-4, atol=1e-3)


def test_larger_batches_keep_the_tile_kernel():
    x = torch.randn(SM70_GEMV_MAX_TOKENS + 1, 2304, device="cuda", dtype=torch.float16)
    w = (torch.randn(128, 2304, device="cuda") * 0.02).half()
    torch.testing.assert_close(
        k_group_partials(x, w), _tile_partials(x, w), rtol=0, atol=0
    )
