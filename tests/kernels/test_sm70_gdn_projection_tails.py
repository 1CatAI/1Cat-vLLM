# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
    _split_gdn_projection_tails,
)


@pytest.mark.parametrize("m", [5, 20])
def test_tail_split_preserves_half_payloads_and_qkv_alias(m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    qkvz = torch.empty(m, 4096, device="cuda", dtype=torch.float16)
    ba = torch.empty(m, 24, device="cuda", dtype=torch.float16)
    z = torch.empty(m, 12, 128, device="cuda", dtype=torch.float16)
    bits = torch.arange(m * 4096, device="cuda", dtype=torch.int32).to(torch.int16)
    qkvz.view(torch.int16).copy_(bits.view(m, 4096))
    ba.view(torch.int16).copy_(bits[: m * 24].view(m, 24))
    qkv, b, a = _split_gdn_projection_tails(qkvz, ba, z)
    assert qkv.data_ptr() == qkvz.data_ptr() and qkv.stride() == (4096, 1)
    assert torch.equal(
        z.view(m, 1536).view(torch.int16), qkvz[:, 2560:].view(torch.int16)
    )
    assert torch.equal(b.view(torch.int16), ba[:, :12].view(torch.int16))
    assert torch.equal(a.view(torch.int16), ba[:, 12:].view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qkv, b, a = _split_gdn_projection_tails(qkvz, ba, z)
        qkv.zero_()
    for offset in (23456, -12345):
        qkvz.view(torch.int16).copy_((bits + offset).view(m, 4096))
        ba.view(torch.int16).copy_((bits[: m * 24] + offset).view(m, 24))
        expected_z = qkvz[:, 2560:].clone()
        expected_b, expected_a = ba[:, :12].clone(), ba[:, 12:].clone()
        graph.replay()
        assert torch.equal(
            z.view(m, 1536).view(torch.int16), expected_z.view(torch.int16)
        )
        assert torch.equal(b.view(torch.int16), expected_b.view(torch.int16))
        assert torch.equal(a.view(torch.int16), expected_a.view(torch.int16))
        assert torch.count_nonzero(qkv) == 0


@pytest.mark.parametrize("m", [5, 20])
def test_input_core_tail_route_keeps_recurrent_operands(monkeypatch, m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as split

    q = torch.randn(m, 4096, device="cuda", dtype=torch.float16)
    ba = torch.randn(m, 24, device="cuda", dtype=torch.float16)
    x = torch.empty(m, 2560, device="cuda", dtype=torch.float16)
    z = torch.empty(m, 12, 128, device="cuda", dtype=torch.float16)
    core = torch.empty_like(z)
    layer = SimpleNamespace(
        in_proj_qkvz=lambda _: (q, None),
        in_proj_ba=lambda _: (ba, None),
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
        key_dim=2048,
        value_dim=6144,
        tp_size=4,
        head_v_dim=128,
    )
    monkeypatch.setattr(gdn, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"test": layer}),
    )
    monkeypatch.setattr(gdn, "_sm70_gdn_qpn8_ba_split_eligible", lambda *args: False)
    monkeypatch.setattr(
        gdn, "_sm70_dump_gdn_projection_tensor", lambda label, name, value: value
    )
    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS", False)
    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_Z_CONTIGUOUS", False)
    operands = []

    def recurrent(_, **kwargs):
        operands.append((kwargs["mixed_qkv"], kwargs["b"], kwargs["a"]))

    monkeypatch.setattr(gdn, "_qwen_gdn_run_recurrent_core", recurrent)
    run = lambda: gdn.qwen_gdn_input_projection_core(
        x, z, core, x.new_empty(0), x.new_empty(0), "test"
    )
    monkeypatch.setattr(split, "_can_fuse_gdn_projection_split", lambda *args: True)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    actual = operands[-1]
    for _ in range(3):
        q.normal_()
        ba.normal_()
        monkeypatch.setattr(
            split, "_can_fuse_gdn_projection_split", lambda *args: False
        )
        run()
        expected = operands[-1]
        expected_z = z.clone()
        graph.replay()
        assert actual[0].data_ptr() == expected[0].data_ptr() == q.data_ptr()
        assert actual[0].stride() == expected[0].stride() == (4096, 1)
        assert torch.equal(z.view(torch.int16), expected_z.view(torch.int16))
        for result, reference in zip(actual, expected, strict=True):
            assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
