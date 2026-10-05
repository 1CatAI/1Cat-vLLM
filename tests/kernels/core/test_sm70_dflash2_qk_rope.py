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


@pytest.mark.parametrize("rows,context", [(8, 1024), (32, 2047), (8, 262136)])
@torch.inference_mode()
def test_qk_rope_cache_matches_original_fp16(rows, context):
    torch.manual_seed(123)
    raw = torch.randn(rows, 1536, device="cuda", dtype=torch.float16) * 3
    qw = torch.randn(128, device="cuda", dtype=torch.float16)
    kw = torch.randn_like(qw)
    positions = torch.arange(context, context + rows, device="cuda")
    with set_current_vllm_config(VllmConfig()):
        rope = RotaryEmbedding(128, 128, 262144, 1e7, True, raw.dtype).cuda()
    kc = torch.full((1, 2048, 2, 128), 13.0, device="cuda", dtype=raw.dtype)
    vc = torch.full_like(kc, 17.0)
    slots = torch.arange(rows, device="cuda")
    slots[-1] = -1  # A graph-padding row must not alter cache storage.
    scale = torch.ones(1, device="cuda")
    q = torch.empty(rows, 8, 128, device="cuda", dtype=raw.dtype)
    k = torch.empty(rows, 2, 128, device="cuda", dtype=raw.dtype)
    ops.rms_norm(q, raw[:, :1024].reshape_as(q), qw, 1e-6)
    ops.rms_norm(k, raw[:, 1024:1280].reshape_as(k), kw, 1e-6)
    q, k = rope.forward_cuda(positions, q.view(rows, 1024), k.view(rows, 256))
    ops.reshape_and_cache_flash(
        k.view(rows, 2, 128),
        raw[:, 1280:].view(rows, 2, 128),
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
    for expected, actual in zip(
        (expected_q, expected_k, expected_kc, expected_vc),
        (actual_q, actual_k, kc, vc),
    ):
        assert torch.equal(expected, actual)

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
