# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("m", [5, 20])
def test_default_forward_tail_route(monkeypatch, m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as split

    q = torch.randn(m, 4096, device="cuda", dtype=torch.float16)
    ba = torch.randn(m, 24, device="cuda", dtype=torch.float16)
    x = torch.empty(m, 2560, device="cuda", dtype=torch.float16)
    operands, outputs = [], []

    def recurrent(_, **kwargs):
        operands.append((kwargs["mixed_qkv"], kwargs["b"], kwargs["a"]))
        return kwargs["core_attn_out"]

    from vllm.config.gdn import GdnConfig

    policy = GdnConfig()
    policy.resolve()
    layer = SimpleNamespace(
        gdn_policy=policy,
        prefix="test",
        in_proj_qkvz=lambda _: (q, None),
        in_proj_ba=lambda _: (ba, None),
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
        key_dim=2048,
        value_dim=6144,
        tp_size=4,
        num_v_heads=48,
        head_v_dim=128,
        _output_projection=lambda core, z, out, count: outputs.append(z),
    )
    for name in (
        "VLLM_SM70_QWEN_GDN_INPUT_CORE_OP",
        "VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP",
        "VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP",
        "VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS",
    ):
        monkeypatch.setattr(gdn.envs, name, False)
    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_Z_CONTIGUOUS", True)
    monkeypatch.setattr(gdn, "_sm70_gdn_graph_buffer_copy", lambda *args: None)
    monkeypatch.setattr(
        gdn, "_sm70_dump_gdn_projection_tensor", lambda label, name, value: value
    )
    monkeypatch.setattr(gdn, "_qwen_gdn_run_recurrent_core", recurrent)
    monkeypatch.setattr(gdn, "_resolve_qwen_gdn_kv_cache_args", lambda *args: (x, x))
    run = lambda: gdn.QwenGatedDeltaNetAttention.forward_cuda(layer, x, None)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    actual, actual_z = operands[-1], outputs[-1]
    for _ in range(3):
        q.normal_()
        ba.normal_()
        monkeypatch.setattr(
            split, "_can_fuse_gdn_projection_split", lambda *args: False
        )
        run()
        expected, expected_z = operands[-1], outputs[-1]
        graph.replay()
        assert actual[0].data_ptr() == expected[0].data_ptr() == q.data_ptr()
        assert actual[0].stride() == expected[0].stride() == (4096, 1)
        assert torch.equal(actual_z.view(torch.int16), expected_z.view(torch.int16))
        for result, reference in zip(actual, expected, strict=True):
            assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
