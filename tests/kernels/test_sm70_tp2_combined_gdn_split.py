# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.models import qwen3_5 as model

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "override,expected",
    [
        ({}, True),
        ({"tp_size": 4}, False),
        ({"key_dim": 1024}, False),
        ({"value_dim": 3072}, False),
        ({"num_v_heads": 24}, False),
        ({"enable_sm70_dflash2_fused_gdn_split": False}, False),
    ],
)
def test_combined_split_constructor_keeps_unverified_shapes_off(
    monkeypatch, override, expected
):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_DFLASH2_TP2_COMBINED_GDN_SPLIT", "1")

    def init(self):
        torch.nn.Module.__init__(self)
        for key, value in (
            dict(
                quant_config=None,
                tp_size=2,
                key_dim=2048,
                value_dim=6144,
                num_v_heads=48,
                enable_sm70_dflash2_fused_gdn_split=True,
            )
            | override
        ).items():
            setattr(self, key, value)

    monkeypatch.setattr(model.QwenGatedDeltaNetAttention, "__init__", init)
    monkeypatch.setattr(model, "_uses_split_gdn_input_projections", lambda _: False)
    layer = model.Qwen3_5GatedDeltaNet()
    assert layer.enable_sm70_dflash2_tp2_combined_gdn_split is expected
    monkeypatch.delenv("VLLM_SM70_DFLASH2_TP2_COMBINED_GDN_SPLIT")
    assert not model.Qwen3_5GatedDeltaNet().enable_sm70_dflash2_tp2_combined_gdn_split
    monkeypatch.setenv("VLLM_SM70_DFLASH2_TP2_COMBINED_GDN_SPLIT", "1")
    monkeypatch.setattr(model, "_uses_split_gdn_input_projections", lambda _: True)
    assert not model.Qwen3_5GatedDeltaNet().enable_sm70_dflash2_tp2_combined_gdn_split


@pytest.mark.parametrize(
    "rows,stride,offset",
    [
        (1, 8240, 0),
        (7, 8256, 17),
        (8, 8256, 0),
        (8, 8320, 17),
        (9, 8256, 0),
        (32, 8256, 0),
        (4096, 8256, 0),
    ],
)
def test_combined_split_forward_preserves_bits_and_convolution_ownership(
    monkeypatch, rows, stride, offset
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires an owned SM70 GPU")
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS", "0")
    monkeypatch.setattr(model, "_sm70_gdn_qpn8_ba_dispatch_eligible", lambda *a: False)
    monkeypatch.setattr(model, "_sm70_dump_gdn_projection_tensor", lambda *a: a[-1])
    monkeypatch.setattr(
        model, "_resolve_qwen_gdn_kv_cache_args", lambda *a: (None, None)
    )
    # Force materialized reference slices, as in the compiled q8 control.
    monkeypatch.setattr(
        model,
        "_sm70_compile_graph_slice_dim",
        lambda x, dim, start, size: x.index_select(
            dim, torch.arange(start, start + size, device=x.device)
        ),
    )
    arena = torch.full(
        (offset + rows * stride + 64,), 16977, device="cuda", dtype=torch.int16
    )
    projection = torch.as_strided(
        arena.view(torch.float16), (rows, 8240), (stride, 1), offset
    )
    hidden = torch.empty((rows, 5120), device="cuda", dtype=torch.float16)
    observed: dict[str, torch.Tensor] = {}

    def recurrent(self, *, mixed_qkv, b, a, core_attn_out, **kwargs):
        observed.update(b=b, a=a, qkv=mixed_qkv.clone())
        # The real convolution writes QKV in place. Tail materialization must
        # neither detach this view nor let those writes corrupt z/b/a.
        mixed_qkv.zero_()
        return core_attn_out

    monkeypatch.setattr(model, "_qwen_gdn_run_recurrent_core", recurrent)
    layer = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        tp_size=2,
        key_dim=2048,
        value_dim=6144,
        num_v_heads=48,
        head_v_dim=128,
        use_split_input_projections=False,
        enable_sm70_dflash2_tp2_combined_gdn_split=False,
        in_proj_qkvz=lambda _: (projection, None),
        _output_projection=lambda core, z, output, n: (
            z.flatten(1),
            observed["b"],
            observed["a"],
            observed["qkv"],
        ),
    )
    helper = model._sm70_materialize_qwen35_gdn_splits
    hits = []

    def tracked(qkvz, ba, *sizes):
        hits.append((qkvz.stride(), ba.storage_offset() - qkvz.storage_offset()))
        return helper(qkvz, ba, *sizes)

    monkeypatch.setattr(model, "_sm70_materialize_qwen35_gdn_splits", tracked)
    graphs, outputs = [], []
    for enabled in (False, True):
        layer.enable_sm70_dflash2_tp2_combined_gdn_split = enabled
        model.Qwen3_5GatedDeltaNet.forward_cuda(layer, hidden, None)
        torch.cuda.synchronize()
        hits.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model.Qwen3_5GatedDeltaNet.forward_cuda(layer, hidden, None)
        assert hits == ([((stride, 1), 8192)] if enabled else [])
        graphs.append(graph)
        outputs.append(output)

    # Four changing patterns cover every FP16 payload at q8, including NaNs
    # and signed zero, without numerical comparison that would hide bit flips.
    for shift in (0, 16384, 32768, 49152):
        bits = (torch.arange(rows * 8240, device="cuda") + shift).to(torch.int16)
        before = arena.clone()
        reference = bits.view(rows, 8240)
        expected = [
            reference[:, lo:hi]
            for lo, hi in ((5120, 8192), (8192, 8216), (8216, 8240), (0, 5120))
        ]
        for graph, output in zip(graphs, outputs, strict=True):
            projection.view(torch.int16).copy_(reference)
            graph.replay()
            assert all(
                torch.equal(a.view(torch.int16), b)
                for a, b in zip(output, expected, strict=True)
            )
            assert torch.count_nonzero(projection[:, :5120].view(torch.int16)) == 0
            assert torch.equal(
                projection[:, 5120:].view(torch.int16), reference[:, 5120:]
            )
            # Exclude the projection itself when checking padding and canaries.
            before_view = torch.as_strided(before, (rows, 8240), (stride, 1), offset)
            before_view.copy_(projection.view(torch.int16))
            assert torch.equal(arena, before)
