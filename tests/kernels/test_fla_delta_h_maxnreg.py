# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The maxnreg=255 delta_h configs only change register allocation, not results."""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
    _delta_h_configs,
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from vllm.model_executor.layers.fla.ops.gdn_chunk_kernels import _retune
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.triton_utils import triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _inputs(T=256, Hg=2, H=6, K=128, V=128, BT=64):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    k = torch.nn.functional.normalize(torch.randn(1, T, Hg, K, device=dev), dim=-1)
    cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)
    return dict(
        k=k.half(),
        w=(torch.randn(1, T, H, K, device=dev) * 0.05).half(),
        u=(torch.randn(1, T, H, V, device=dev) * 0.5).half(),
        g=torch.cumsum(-torch.rand(1, T, H, device=dev) * 0.1, dim=1),
        initial_state=torch.randn(1, H, V, K, device=dev) * 0.1,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=prepare_chunk_indices(cu_seqlens, BT),
        chunk_offsets=prepare_chunk_offsets(cu_seqlens, BT),
    )


def _run_one_config(config, inputs):
    kernel = _retune(
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64, [config], ["H", "K", "V", "BT"]
    )
    return chunk_gated_delta_rule_fwd_h(**inputs, kernel=kernel)


def test_every_config_also_has_a_maxnreg_variant():
    def shape(config):
        return (tuple(config.kwargs.items()), config.num_warps, config.num_stages)

    plain = {shape(c) for c in _delta_h_configs if c.maxnreg is None}
    lifted = {shape(c) for c in _delta_h_configs if c.maxnreg == 255}
    assert plain and plain == lifted


@pytest.mark.parametrize("bv", [32, 64])
@pytest.mark.parametrize("num_warps", [2, 4])
def test_maxnreg_variant_is_bit_identical(bv, num_warps):
    inputs = _inputs()
    outputs = [
        _run_one_config(
            triton.Config(
                {"BV": bv}, num_warps=num_warps, num_stages=2, maxnreg=maxnreg
            ),
            inputs,
        )
        for maxnreg in (None, 255)
    ]
    for plain, lifted in zip(*outputs):
        assert torch.equal(plain, lifted)
