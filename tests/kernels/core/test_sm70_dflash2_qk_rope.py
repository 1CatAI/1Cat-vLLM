# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.sm70_dflash2_qk_rope import qk_norm_rope_cache
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="Requires SM70",
)


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 2), (16, 4)])
@pytest.mark.parametrize("rows,context", [(8, 1024), (32, 2047), (8, 262136)])
@torch.inference_mode()
def test_qk_rope_cache_matches_original_fp16(num_q_heads, num_kv_heads, rows, context):
    torch.manual_seed(123)
    q_width = num_q_heads * 128
    kv_width = num_kv_heads * 128
    raw = (
        torch.randn(rows, q_width + 2 * kv_width, device="cuda", dtype=torch.float16)
        * 3
    )
    qw = torch.randn(128, device="cuda", dtype=torch.float16)
    kw = torch.randn_like(qw)
    positions = torch.arange(context, context + rows, device="cuda")
    with set_current_vllm_config(VllmConfig()):
        rope = RotaryEmbedding(128, 128, 262144, 1e7, True, raw.dtype).cuda()
    kc = torch.full((1, 2048, num_kv_heads, 128), 13.0, device="cuda", dtype=raw.dtype)
    vc = torch.full_like(kc, 17.0)
    slots = torch.arange(rows, device="cuda")
    slots[-1] = -1  # A graph-padding row must not alter cache storage.
    scale = torch.ones(1, device="cuda")
    q = torch.empty(rows, num_q_heads, 128, device="cuda", dtype=raw.dtype)
    k = torch.empty(rows, num_kv_heads, 128, device="cuda", dtype=raw.dtype)
    ops.rms_norm(q, raw[:, :q_width].reshape_as(q), qw, 1e-6)
    ops.rms_norm(k, raw[:, q_width : q_width + kv_width].reshape_as(k), kw, 1e-6)
    q, k = rope.forward_cuda(positions, q.view(rows, q_width), k.view(rows, kv_width))
    ops.reshape_and_cache_flash(
        k.view(rows, num_kv_heads, 128),
        raw[:, q_width + kv_width :].view(rows, num_kv_heads, 128),
        kc,
        vc,
        slots,
        "auto",
        scale,
        scale,
    )
    expected_q, expected_k, expected_kc, expected_vc = (
        t.clone() for t in (q, k, kc, vc)
    )
    kc.fill_(13.0)
    vc.fill_(17.0)
    actual_q, actual_k = qk_norm_rope_cache(
        raw,
        qw,
        kw,
        positions,
        rope.cos_sin_cache,
        kc,
        vc,
        slots,
    )
    # Value cache and masked padding slot must match bit-for-bit; Q/K/KC allow
    # at most 1 FP16 ULP from FP32 associative sum rounding in RMSNorm variance.
    assert torch.equal(expected_vc, vc)
    assert torch.equal(expected_kc[:, rows - 1 :], kc[:, rows - 1 :])
    for expected, actual in zip(
        (expected_q, expected_k, expected_kc),
        (actual_q, actual_k, kc),
    ):
        ulp = (
            (expected.view(torch.int16).int() - actual.view(torch.int16).int())
            .abs()
            .max()
        )
        assert int(ulp.item()) <= 1

    # Replay must consume live input and slot values, including padding.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qk_norm_rope_cache(
            raw,
            qw,
            kw,
            positions,
            rope.cos_sin_cache,
            kc,
            vc,
            slots,
            q_out=actual_q,
            k_out=actual_k,
        )
    raw.zero_()
    slots.fill_(-1)
    kc.fill_(13.0)
    vc.fill_(17.0)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.count_nonzero(actual_q) == torch.count_nonzero(actual_k) == 0
    assert torch.all(kc == 13.0) and torch.all(vc == 17.0)
