# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retained FP16 schedules across two engines and changed-input graph replay."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import AttentionConfig, set_current_vllm_config
from vllm.config.sm70_triton_attention import Sm70TritonAttentionPolicy
from vllm.runtime_resources import release_runtime_resources
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)
@pytest.mark.parametrize("rows", [1, 4, 33])
def test_changed_input_capture_uses_initialized_schedule(monkeypatch, rows):
    torch.manual_seed(451)
    cache = torch.randn(4, 16, 2, 128, device="cuda", dtype=torch.float16) * 0.1
    values = torch.randn_like(cache) * 0.1
    pages = torch.arange(4, device="cuda", dtype=torch.int32)[None, :]
    lengths = torch.tensor([64], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    inputs = [
        torch.randn(rows, 8, 128, device="cuda", dtype=torch.float16) * 0.1
        for _ in range(3)
    ]
    records = []
    for warps in (4, 8):
        cfg = SimpleNamespace(attention_config=AttentionConfig())
        cfg.attention_config.sm70_triton.num_warps = warps
        cfg.attention_config.sm70_triton.resolve()
        q, out = torch.empty_like(inputs[0]), torch.empty_like(inputs[0])

        def run(q=q, out=out):
            unified_attention(
                q=q,
                k=cache,
                v=values,
                out=out,
                cu_seqlens_q=starts,
                max_seqlen_q=rows,
                seqused_k=lengths,
                max_seqlen_k=64,
                softmax_scale=128**-0.5,
                causal=True,
                window_size=(-1, -1),
                block_table=pages,
                softcap=0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                seq_threshold_3D=None,
                num_par_softmax_segments=None,
                softmax_segm_output=None,
                softmax_segm_max=None,
                softmax_segm_expsum=None,
            )

        with set_current_vllm_config(cfg):
            expected = []
            for source in inputs:
                q.copy_(source)
                run()
                expected.append(out.clone())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
        records.append((cfg, q, out, graph, expected))
    for alias in Sm70TritonAttentionPolicy.aliases.values():
        monkeypatch.setenv(alias, "invalid-after-init")
    for cfg, q, out, graph, expected in records:
        for source, reference in zip(inputs, expected):
            q.copy_(source)
            graph.replay()
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
        graph.reset()
        release_runtime_resources(cfg)
