# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-shared-memory FlashQLA TileLang GDN prefill matches Triton/FLA."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("needs CUDA", allow_module_level=True)

hopper_fused_fwd = pytest.importorskip(
    "flash_qla.ops.gated_delta_rule.chunk.hopper.fused_fwd",
    reason="FlashQLA TileLang kernels are not installed.",
)
from flash_qla.ops.gated_delta_rule.chunk import (  # noqa: E402
    chunk_gated_delta_rule_fwd_sm70_tilelang,
)
from vllm.model_executor.layers.fla.ops.chunk import (  # noqa: E402
    chunk_gated_delta_rule_fwd,
)

if (
    torch.cuda.get_device_properties().shared_memory_per_block_optin
    >= hopper_fused_fwd.SM70_FUSED_FWD_SHARED_BYTES
):
    # Such a device runs the SM70 kernel; this test covers the Turing variant.
    pytest.skip("device fits the SM70 kernel", allow_module_level=True)


@pytest.mark.parametrize("lengths", [[2048], [17, 2031], [63, 1, 1984]])
@pytest.mark.parametrize("heads", [(8, 24), (16, 48)])
@torch.inference_mode()
def test_turing_fused_fwd_matches_fla(lengths, heads):
    num_key_heads, num_value_heads = heads
    head_dim = 128
    torch.manual_seed(0)
    dev = torch.device("cuda")
    tokens = sum(lengths)
    q = torch.randn(1, tokens, num_key_heads, head_dim, device=dev)
    k = torch.randn(1, tokens, num_key_heads, head_dim, device=dev)
    q = torch.nn.functional.normalize(q, dim=-1).half()
    k = torch.nn.functional.normalize(k, dim=-1).half()
    v = (torch.randn(1, tokens, num_value_heads, head_dim, device=dev) * 0.5).half()
    g = -torch.rand(1, tokens, num_value_heads, device=dev) * 0.1
    beta = torch.rand(1, tokens, num_value_heads, device=dev).half()
    initial_state = (
        torch.randn(len(lengths), num_value_heads, head_dim, head_dim, device=dev) * 0.1
    )
    offsets = [0] + torch.tensor(lengths).cumsum(0).tolist()
    cu_seqlens = torch.tensor(offsets, device=dev, dtype=torch.int32)
    scale = head_dim**-0.5

    _, expected_out, _, expected_state, *_ = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale, initial_state.clone(), True, cu_seqlens=cu_seqlens
    )
    _, _, out, _, state = chunk_gated_delta_rule_fwd_sm70_tilelang(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens.long(),
        output_final_state=True,
        state_layout_vlk=True,
    )
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(state, expected_state, atol=2e-3, rtol=2e-3)
