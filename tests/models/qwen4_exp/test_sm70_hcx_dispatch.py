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
        return (a.float() + b.float() - 80000).half()

    hcx.register_moe_runner("p", SimpleNamespace(_maybe_sm70_moe_sum2_allreduce=sum2))
    out = hcx._moe_output(shared, fused, "p", 4)
    assert calls == [(shared, fused, 4)]
    assert out.shape == (2 * rows, 4)
    assert torch.all(out[:rows] == 0)


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
        _partial_pair=lambda x: (x[:5], x[5:]),
        hc_norm=SimpleNamespace(weight=torch.ones(4)),
        config=SimpleNamespace(rms_norm_eps=1e-6),
        _hcx_down=None,
        _hcx_up=None,
    )
    hc._PARTIAL_MODULES["consumer"] = module
    hc._hcx_combine_and_mix(torch.ones(5, 16), partial, torch.ones(5, 4), "consumer")
    torch.testing.assert_close(seen[0][0], fused, rtol=0, atol=0)
    torch.testing.assert_close(seen[0][1], shared, rtol=0, atol=0)
    assert partial.shape == (10, 4)
    assert torch.all(seen[0][0].float() + seen[0][1].float() == 2049)


def test_ple_diagnostic_snapshot_owns_both_outputs(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import ple_layer

    monkeypatch.setattr(ple_layer, "_PLE_DIAGNOSTIC_BUFFERS", {})
    source = torch.arange(20).reshape(5, 4)
    output = ple_layer._ple_diagnostic_snapshot(source, "test")
    saved = ple_layer._PLE_DIAGNOSTIC_BUFFERS["test"]
    assert len({source.data_ptr(), output.data_ptr(), saved.data_ptr()}) == 3
    output.zero_()
    source.add_(100)
    torch.testing.assert_close(saved, torch.arange(20).reshape(5, 4))
    pointer = saved.data_ptr()
    ple_layer._ple_diagnostic_snapshot(source, "test")
    assert ple_layer._PLE_DIAGNOSTIC_BUFFERS["test"].data_ptr() == pointer
    torch.testing.assert_close(saved, source)


def test_ple_diagnostic_gate_uses_module_flag(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import ple_layer

    calls = []
    captured = object()

    def snapshot(tensor, label):
        calls.append((tensor, label))
        return captured

    monkeypatch.setattr(torch.ops.vllm, "qwen4_exp_ple_diagnostic_snapshot", snapshot)
    cuda_input = SimpleNamespace(is_cuda=True)
    cpu_input = SimpleNamespace(is_cuda=False)
    assert ple_layer.snapshot_ple_diagnostic(cuda_input, "enabled", True) is captured
    assert (
        ple_layer.snapshot_ple_diagnostic(cuda_input, "disabled", False) is cuda_input
    )
    assert ple_layer.snapshot_ple_diagnostic(cpu_input, "cpu", True) is cpu_input
    assert calls == [(cuda_input, "enabled")]


@pytest.mark.parametrize("rows", [5, 20])
def test_materialization_reduces_small_payloads_at_runtime(rows, monkeypatch):
    from vllm import distributed

    calls = []

    def sum2(first, second):
        calls.append("reduce")
        return (first.float() + second.float()).mul(4).half()

    monkeypatch.setattr(distributed, "tensor_model_parallel_all_reduce_sum2", sum2)
    monkeypatch.setattr(hc, "hc_combine", lambda h, b, i, count: b.clone())
    module = SimpleNamespace(
        _partial_inputs=True,
        _hcx_moe_payload=True,
        hc_count=4,
        _partial_pair=lambda payload: (payload[:rows], payload[rows:]),
        _combine_and_mix_reduced=lambda h, b, i: (b.clone(), b.clone(), None),
    )
    module._reduce_partial = lambda payload: hc.GatedResidual._reduce_partial(
        module, payload
    )
    hc._PARTIAL_MODULES["materialize"] = module
    first = torch.ones(rows, 4, dtype=torch.float16)
    # The second plane is unused for an already-reduced large batch.
    second = torch.full_like(first, 2 if rows == 5 else float("nan"))
    payload = torch.cat((first, second))
    expected = first * (12 if rows == 5 else 1)
    hidden, injection = torch.ones_like(first), torch.ones_like(first)
    actual = hc._hcx_combine(hidden, payload, injection, "materialize")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    final = hc._hcx_final_mix(hidden, payload, injection, "materialize")
    for output in final:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert calls == (["reduce", "reduce"] if rows == 5 else [])


def test_large_m_hc_does_not_reduce_or_recompute_projection():
    calls = []

    def original(hidden, block, injection):
        calls.append(block)
        return hidden, block, injection

    hc._PARTIAL_MODULES["consumer"] = SimpleNamespace(
        _hcx=SimpleNamespace(run=lambda *a, **k: pytest.fail("small-M HCX")),
        _partial_pair=lambda x: (x, None),
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
    assert model.layers[1].attn_hyper_connection._hcx_moe_payload
    assert not model.layers[2].attn_hyper_connection.partial
    assert model.hyper_connection_mixer._hcx_moe_payload
    assert model.sm70_hcx_status["prepared_modules"] == 4


def test_compiled_moe_payload_keeps_both_dependencies_live():
    names = {
        "qwen38_sm70_hcx_moe_output": hcx._moe_output,
        "qwen38_sm70_hcx_combine_and_mix": hc._hcx_combine_and_mix,
    }
    library = torch.library.Library("vllm", "IMPL", "CPU")
    for name, function in names.items():
        if not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::" + name, "CPU"):
            library.impl(name, function)

    def native(first, hidden, injection, *args, secondary):
        block = ((first.float() + secondary.float()) * 0.5).half()
        return hidden.clone(), block, injection.clone()

    def original(hidden, block, injection):
        return hidden.clone(), block * 0.5, injection.clone()

    hc._PARTIAL_MODULES["consumer"] = SimpleNamespace(
        _hcx=SimpleNamespace(run=native),
        _partial_pair=lambda x: (x[: x.shape[0] // 2], x[x.shape[0] // 2 :]),
        hc_norm=SimpleNamespace(weight=torch.ones(4)),
        config=SimpleNamespace(rms_norm_eps=1e-6),
        _hcx_down=None,
        _hcx_up=None,
        _combine_and_mix_reduced=original,
    )
    hcx.register_moe_runner(
        "producer",
        SimpleNamespace(
            _maybe_sm70_moe_sum2_allreduce=lambda a, b, n: (
                a.float() + b.float()
            ).half(),
        ),
    )

    @torch.compile(backend="inductor", dynamic=True, fullgraph=True)
    def model(x, hidden, injection):
        # These producer temporaries can be reused after packing. Their values
        # must survive in the explicit payload, not in hidden Python refs.
        payload = torch.ops.vllm.qwen38_sm70_hcx_moe_output(x * 4, x * 3, "producer", 4)
        scratch = x * 100
        _, block, _ = torch.ops.vllm.qwen38_sm70_hcx_combine_and_mix(
            hidden, payload, injection, "consumer"
        )
        return block, scratch

    for rows in [5, 20, 8, 9, 5]:
        x = torch.arange(rows * 4).reshape(rows, 4).half() * 0.125
        hidden, injection = torch.ones(rows, 16).half(), torch.ones(rows, 4).half()
        block, scratch = model(x, hidden, injection)
        torch.testing.assert_close(block, x * 3.5, rtol=0, atol=0)
        torch.testing.assert_close(scratch, x * 100, rtol=0, atol=0)
    del library


def test_large_m_injection_matches_contiguous_fake_contract():
    hidden, block = torch.ones(20, 16), torch.ones(20, 4)
    padded_injection = torch.ones(20, 336)
    injection = padded_injection[:, :4]
    assert injection.stride(0) == 336
    hc._PARTIAL_MODULES["consumer"] = SimpleNamespace(
        _partial_pair=lambda x: (x, None),
        _combine_and_mix_reduced=lambda h, b, i: (h, b, i),
    )
    outputs = hc._hcx_combine_and_mix(hidden, block, injection, "consumer")
    assert all(t.is_contiguous() for t in outputs)
    torch.testing.assert_close(outputs[2], injection, rtol=0, atol=0)


def test_diagnostic_snapshots_own_storage_and_reuse_addresses(monkeypatch):
    runtime = hcx.Sm70HcxRuntime.__new__(hcx.Sm70HcxRuntime)
    runtime.diagnostic, runtime.snapshots = True, {}
    runtime.xn = runtime.sq = runtime.dpart = runtime.bar = None
    runtime.seq = torch.zeros(1, dtype=torch.int32)
    runtime.ar = runtime.lora = runtime.hb = []
    runtime.logical_rank, runtime.full = 0, False

    def native(*args):
        args[8].copy_(args[2])
        args[9].copy_(args[0] + args[1])
        args[10].copy_(args[3])
        args[15].add_(1)

    monkeypatch.setattr(torch.ops._C, "sm70_hcx_out", native, raising=False)
    partial, secondary = torch.ones(5, 2560), torch.full((5, 2560), 2.0)
    hidden, injection = torch.ones(5, 10240), torch.ones(5, 4)
    arguments = (partial, hidden, injection, torch.ones(2560), 1e-6, None, None)
    runtime.run(*arguments, secondary=secondary, snapshot_name="layer0.mlp")
    snapshot = runtime.snapshots["layer0.mlp"]
    pointers = {name: value.data_ptr() for name, value in snapshot.items()}
    assert snapshot["partial"].data_ptr() != partial.data_ptr()
    assert snapshot["epoch"].item() == 0
    partial.add_(5)
    secondary.add_(7)
    runtime.run(*arguments, secondary=secondary, snapshot_name="layer0.mlp")
    assert pointers == {name: value.data_ptr() for name, value in snapshot.items()}
    torch.testing.assert_close(snapshot["partial"], partial)
    torch.testing.assert_close(snapshot["block_out"], partial + secondary)
    assert snapshot["epoch"].item() == 1
