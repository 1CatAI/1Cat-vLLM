# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import hyperconnection as hc
from vllm.models.qwen4_exp.nvidia import sm70_hcx as hcx


@pytest.fixture(autouse=True)
def isolate_registries(monkeypatch):
    monkeypatch.setattr(hcx, "_OUTPUT_PROJECTIONS", {})
    monkeypatch.setattr(hcx, "_MOE_RUNNERS", {})
    monkeypatch.setattr(hcx, "_MOE_INPUTS", {})
    monkeypatch.setattr(hc, "_PARTIAL_MODULES", {})


@pytest.mark.parametrize("rows", [5, 8, 9, 20, 512])
def test_projection_dispatch_retains_large_batch_reduction(rows, monkeypatch):
    from vllm import distributed

    calls = []

    class Projection:
        output_size = 4

        def __call__(self, x):
            calls.append("projection")
            return torch.nn.functional.pad(x * 2, (0, 2)), None

    def reduce(x):
        calls.append("reduce")
        return x * 4

    monkeypatch.setattr(distributed, "tensor_model_parallel_all_reduce", reduce)
    hcx.register_output_projection("p", Projection(), defer=True)
    x = torch.ones(rows, 2)
    out = hcx._output_projection(x, "p")
    if rows <= 8:
        assert calls == []
        torch.testing.assert_close(out[:, :2], x)
    else:
        assert calls == ["projection", "reduce"]
        torch.testing.assert_close(out[:, :2], x * 8)
    assert out.shape == (rows, 4)


@pytest.mark.parametrize("rows", [9, 20, 512])
def test_large_m_moe_keeps_original_fp32_sum2(rows):
    calls = []
    fused = torch.full((rows, 4), 40000, dtype=torch.float16)
    shared = torch.full_like(fused, 40000)

    def sum2(a, b, trunc):
        calls.append((a, b, trunc))
        return a.float() + b.float()

    hcx.register_moe_runner("p", SimpleNamespace(_maybe_sm70_moe_sum2_allreduce=sum2))
    out = hcx._moe_output(shared, fused, "p", 4)
    assert calls == [(shared, fused, 4)]
    assert out.dtype == torch.float32 and torch.all(out == 80000)
    assert not hcx._MOE_INPUTS


def test_small_m_hc_consumes_unrounded_moe_pair():
    # A half-precision sum loses the shared contribution; HCX must receive
    # both originals and sum in FP32 instead of consuming that rounded sum.
    fused = torch.full((5, 4), 2048, dtype=torch.float16)
    shared = torch.ones_like(fused)
    partial = hcx._moe_output(shared, fused, "p", 4)
    seen = []

    def run(first, *args, secondary):
        seen.append((first, secondary))
        return args[0], first, args[1]

    module = SimpleNamespace(
        _hcx=SimpleNamespace(run=run),
        _hcx_moe_name="p",
        hc_norm=SimpleNamespace(weight=torch.ones(4)),
        config=SimpleNamespace(rms_norm_eps=1e-6),
        _hcx_down=None,
        _hcx_up=None,
    )
    hc._PARTIAL_MODULES["consumer"] = module
    hc._hcx_combine_and_mix(torch.ones(5, 16), partial, torch.ones(5, 4), "consumer")
    assert seen[0][0] is fused and seen[0][1] is shared
    assert torch.all(partial == 2048)
    assert torch.all(seen[0][0].float() + seen[0][1].float() == 2049)


def test_large_m_hc_does_not_reduce_or_recompute_projection():
    calls = []

    def original(hidden, block, injection):
        calls.append(block)
        return hidden, block, injection

    hc._PARTIAL_MODULES["consumer"] = SimpleNamespace(
        _hcx=SimpleNamespace(run=lambda *a, **k: pytest.fail("small-M HCX")),
        _hcx_oproj=(lambda x: pytest.fail("duplicate projection"), 2),
        _combine_and_mix_reduced=original,
    )
    block = torch.ones(20, 4)
    hc._hcx_combine_and_mix(torch.ones(20, 16), block, torch.ones(20, 4), "consumer")
    assert calls == [block]


def test_compiled_projection_resolves_actual_rows(monkeypatch):
    from vllm import distributed

    name = "qwen38_sm70_hcx_output_projection"
    library = None
    if not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::" + name, "CPU"):
        library = torch.library.Library("vllm", "IMPL", "CPU")
        library.impl(name, hcx._output_projection)

    class Projection:
        output_size = 4

        def __call__(self, x):
            return torch.nn.functional.pad(x * 2, (0, 2)), None

    monkeypatch.setattr(
        distributed, "tensor_model_parallel_all_reduce", lambda x: x * 4
    )
    hcx.register_output_projection("p", Projection(), defer=True)

    @torch.compile(backend="eager", dynamic=True, fullgraph=True)
    def project(x):
        return torch.ops.vllm.qwen38_sm70_hcx_output_projection(x, "p")

    for rows in [5, 20, 8, 9]:
        out = project(torch.ones(rows, 2))
        assert torch.all(out[:, :2] == (1 if rows <= 8 else 8))
    # Keep the CPU implementation registered throughout compiled execution.
    del library


def test_hcx_only_consumes_partial_producers(monkeypatch):
    from vllm.v1.attention.backends import fa_utils

    monkeypatch.setattr(fa_utils, "get_flash_attn_version", lambda *a, **k: 2)
    from vllm.models.qwen4_exp.nvidia import model as model_module

    class HC:
        def __init__(self):
            self.input_mix_weight_down_block_inject = SimpleNamespace(
                weight=torch.empty(336, 10240, dtype=torch.float16, device="meta")
            )
            self.input_mix_weight_up = SimpleNamespace(
                weight=torch.empty(10240, 320, dtype=torch.float16, device="meta")
            )
            self.hc_norm = SimpleNamespace(
                weight=torch.empty(2560, dtype=torch.float16, device="meta")
            )
            self.partial = False

        def enable_partial_inputs(self, name, runtime=None):
            self.partial = True

    class MoE:
        def __init__(self):
            self.experts = SimpleNamespace(runner=SimpleNamespace())

    class Decoder:
        def __init__(self, index, moe):
            self.layer_idx = index
            self.layer_type = "linear_attention"
            self.linear_attn = SimpleNamespace(
                out_proj=SimpleNamespace(reduce_results=True)
            )
            self.mlp = MoE() if moe else SimpleNamespace()
            self.mlp_hyper_connection = HC()
            self.attn_hyper_connection = HC()

    monkeypatch.setattr(model_module, "Qwen4ExpDecoderLayer", Decoder)
    monkeypatch.setattr(model_module, "Qwen4ExpSparseMoeBlock", MoE)
    monkeypatch.setattr(
        hcx, "get_hcx_runtime", lambda _: SimpleNamespace(enabled=True, reason=None)
    )
    monkeypatch.setattr(hcx, "pack_output_projection", lambda _: None)
    model = SimpleNamespace(
        layers=[Decoder(0, True), Decoder(1, False), Decoder(2, True)],
        hyper_connection_mixer=HC(),
    )
    assert model_module.enable_sm70_hcx(model, torch.device("cpu"))
    assert not model.layers[0].attn_hyper_connection.partial
    assert model.layers[1].attn_hyper_connection._hcx_moe_name == "layer0"
    assert not model.layers[2].attn_hyper_connection.partial
    assert model.hyper_connection_mixer._hcx_moe_name == "layer2"
    assert model.sm70_hcx_status["prepared_modules"] == 4
