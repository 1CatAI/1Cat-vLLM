# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QPN2 must preserve the effective NVFP4 weights used by TurboMind."""

import pytest
import torch


@pytest.mark.parametrize("rows", [32, 33, 40, 64, 65])
@pytest.mark.parametrize("capacity", [None, 64])
def test_native_decode_follows_the_loaded_row_capacity(monkeypatch, rows, capacity):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.utils import (
        sm70_nvfp4_native as module,
    )

    native = SimpleNamespace()
    if capacity is not None:
        native.nvfp4_qpn2_max_rows_sm70 = lambda: capacity
    monkeypatch.setattr(torch.ops, "_C", native)
    calls = []
    monkeypatch.setattr(
        module.sm70_ops, "nvfp4_qpn2_gemm_sm70_out", lambda *a, **k: calls.append(True)
    )
    monkeypatch.setattr(
        module, "_get_sm70_fp8_prefill_exact_dense_workspace", lambda _: None
    )
    x, out = torch.empty(rows, 128, device="meta"), torch.empty(rows, 32, device="meta")
    codes = torch.empty(32, 64, dtype=torch.uint8, device="meta")
    if rows <= (capacity or 32):
        module._dispatch(out, x, codes, codes, 0.001, 8, 2, False)
        assert calls == [True]
    else:
        with pytest.raises(RuntimeError, match="prefill workspace"):
            module._dispatch(out, x, codes, codes, 0.001, 8, 2, False)
        assert not calls


@pytest.mark.parametrize(
    "k,n", [(5120, 7168), (5120, 8256), (3072, 5120), (4352, 5120), (8704, 5120)]
)
@pytest.mark.parametrize("global_scale", [2**-20, 0.000502813432831, 0.5])
def test_wider_shared_projection_basis_keeps_original_weights(k, n, global_scale):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm import _sm70_ops as ops

    torch.manual_seed(195)
    logical = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    raw = (
        torch.tensor([0, 2**-9, 0.5, 1.5, 7, 24, 192, 448], device="cuda")
        .to(torch.float8_e4m3fn)
        .repeat(n, k // 128)
    )
    compact = torch.ops._C.nvfp4_qpn2_prepare_scales_sm70(raw)
    tm, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        logical.T.contiguous(),
        (raw.float() * global_scale).half().T.contiguous(),
        16,
        False,
    )
    x = torch.zeros(5, k, device="cuda", dtype=torch.float16)
    output = x.new_empty(5, n)

    def run():
        ops.nvfp4_qpn2_tm_dispatch_sm70_out(
            output,
            x,
            tm,
            compact,
            global_scale,
            16,
            2,
            tm_scales,
            16,
            int(meta[0]),
            int(meta[1]),
            False,
            0,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    for shift in (0, 17):
        columns = (
            torch.tensor([0, k // 4 - 1, k // 2, k // 2 + 1, k - 1], device="cuda")
            + shift
        ) % k
        x.zero_()
        x[torch.arange(5, device="cuda"), columns] = 1
        graph.replay()
        code = logical[:, columns].long()
        scale = (raw.float() * global_scale).half()[:, columns // 16]
        expected = (
            (magnitudes[code & 7] * torch.where(code & 8 != 0, -1, 1) * scale.float())
            .half()
            .T
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed_k", [2176, 4352, 768])
def test_bundled_down_selection_uses_local_geometry(packed_k):
    from types import SimpleNamespace

    from vllm.model_executor.kernels.linear.qpn.nvfp4 import _bundled_down_layout

    config = SimpleNamespace(
        policy=SimpleNamespace(bundled_down_decode=True), gated_silu=False
    )
    layer = SimpleNamespace(
        prefix="model.layers.0.mlp.down_proj",
        weight=torch.empty(5120, packed_k, device="meta"),
    )
    assert _bundled_down_layout(layer, config)
    config.gated_silu = True
    assert not _bundled_down_layout(layer, config)
    config.gated_silu = False
    layer.prefix = "model.layers.0.mlp.other_proj"
    assert not _bundled_down_layout(layer, config)


@pytest.mark.parametrize("k", [4352, 8704])
@pytest.mark.parametrize("rows", [1, 5, 8])
def test_shared_down_projection_matches_native_weight_schedule(k, rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm import _sm70_ops as ops

    torch.manual_seed(950)
    n, global_scale = 5120, 0.000502813432831
    raw = (
        torch.tensor([0, 2**-9, 0.5, 1.5, 7, 24, 192, 448], device="cuda")
        .to(torch.float8_e4m3fn)
        .repeat(n, k // 128)
    )
    logical = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    packed = logical[:, ::2] | (logical[:, 1::2] << 4)
    codes, scales = ops.nvfp4_qpn2_prepare_sm70(packed, raw)
    tm, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        logical.T.contiguous(),
        (raw.float() * global_scale).half().T.contiguous(),
        16,
        False,
    )
    k_ld, q_ld = int(meta[0]), int(meta[1])
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    actual, expected = x.new_empty(rows, n), x.new_empty(rows, n)

    def run():
        ops.nvfp4_qpn2_tm_dispatch_sm70_out(
            actual,
            x,
            tm,
            scales,
            global_scale,
            16,
            2,
            tm_scales,
            16,
            k_ld,
            q_ld,
            False,
            0,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for amplitude in (2**-20, 0.125, 2):
        x.normal_().mul_(amplitude)
        graph.replay()
        ops.nvfp4_qpn2_gemm_sm70_out(expected, x, codes, scales, global_scale, 16, 2)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "rows,k", [(1, 4352), (5, 4352), (1, 8704), (5, 8704), (8, 8704)]
)
def test_bundled_down_geometry_preserves_effective_weights_and_replay(rows, k):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm import _sm70_ops as ops

    torch.manual_seed(508)
    n = 5120
    packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    raw_scales = torch.randint(1, 24, (n, k // 16), device="cuda").to(
        torch.float8_e4m3fn
    )
    codes, scales = ops.nvfp4_qpn2_prepare_sm70(packed, raw_scales)
    bundled, bundled_scales = ops.nvfp4_qpn2_bundle_sm70(codes, scales)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    actual, expected = [x.new_empty(rows, n) for _ in range(2)]

    def run():
        ops.nvfp4_qpn2_gemm_sm70_out(
            actual, x, bundled, bundled_scales, 0.000502813432831, 16, 2
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for amplitude in (0.0, 0.125, 2.0):
        x.normal_().mul_(amplitude)
        ops.nvfp4_qpn2_gemm_sm70_out(
            expected, x, codes, scales, 0.000502813432831, 16, 2
        )
        graph.replay()
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("rows", [5, 40])
@pytest.mark.parametrize("native_policy", [("policy-token",), ["policy-token"]])
def test_bundled_down_preserves_wrapper_policy(rows, native_policy, monkeypatch):
    from vllm.model_executor.layers.quantization.utils import sm70_nvfp4_native

    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["native_policy"])

    name = (
        "nvfp4_qpn2_gemm_sm70_out" if rows <= 8 else "nvfp4_qpn2_tm_dispatch_sm70_out"
    )
    monkeypatch.setattr(sm70_nvfp4_native.sm70_ops, name, record)
    x = torch.empty(rows, 4352, dtype=torch.float16)
    out = torch.empty(rows, 5120, dtype=torch.float16)
    codes = torch.empty(0, dtype=torch.uint8)
    scales = torch.empty(0, dtype=torch.float16)
    sm70_nvfp4_native._bundled_down_dispatch(
        out,
        x,
        codes,
        codes,
        codes,
        codes,
        0.001,
        16,
        2,
        scales,
        16,
        0,
        0,
        False,
        0,
        False,
        native_policy,
    )
    assert len(calls) == 1 and calls[0] is native_policy


@pytest.mark.parametrize("global_scale", [2**-20, 0.000502813432831, 0.5])
@pytest.mark.parametrize("n", [8704, 17408])
def test_shared_mtp_gate_preserves_scale_range(global_scale, n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    k = 5120
    torch.manual_seed(905)
    logical = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    packed = logical[:, ::2] | (logical[:, 1::2] << 4)
    raw = torch.tensor([0, 2**-9, 0.5, 1.5, 7, 24, 192, 448], device="cuda")
    raw = raw.to(torch.float8_e4m3fn).repeat(n, k // 128)
    _, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw)
    tm, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        logical.T.contiguous(),
        (raw.float() * global_scale).half().T.contiguous(),
        16,
        False,
    )
    k_ld, q_ld = int(meta[0]), int(meta[1])
    padded = torch.zeros(8, k, device="cuda", dtype=torch.float16)
    padded[torch.arange(5), torch.arange(5) * 17] = 1
    outputs = []
    for rows in (8, 5):
        out = torch.empty(rows, n // 2, device="cuda", dtype=torch.float16)
        torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
            out,
            padded[:rows],
            tm,
            scales,
            global_scale,
            8,
            2,
            tm_scales,
            16,
            k_ld,
            q_ld,
            True,
            0,
        )
        outputs.append(out)
    assert torch.equal(outputs[0][:5].view(torch.int16), outputs[1].view(torch.int16))
    # A basis input selects original effective weights directly, independently
    # of the projection's packing, row stride and FP32 reduction schedule.
    columns = torch.arange(5, device="cuda") * 17
    code = logical[:, columns].long()
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    scale = (raw.float() * global_scale).half()[:, columns // 16]
    projected = (
        (magnitudes[code & 7] * torch.where(code & 8 != 0, -1, 1) * scale.float())
        .half()
        .T
    )
    gate, up = projected.chunk(2, dim=1)
    silu = (gate.float() / (1 + torch.exp(-gate.float()))).half()
    torch.testing.assert_close(outputs[1], silu * up, rtol=0, atol=0)


@pytest.mark.parametrize(
    "global_scale", [0.000156947542564, 0.000502813432831, 2**-20, 0.5]
)
@pytest.mark.parametrize("rows", [8, 16, 32])
@pytest.mark.parametrize("shared", [False, True])
def test_qpn2_basis_vectors_preserve_dequantized_weights(global_scale, rows, shared):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    n, k = 32, 128
    codes = torch.arange(n * k, device="cuda").reshape(n, k).remainder(16)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    raw_scales = (
        torch.tensor([0, 2**-9, 0.5, 1.5, 7, 24, 192, 448], device="cuda")
        .to(torch.float8_e4m3fn)
        .repeat(n, 1)
    )
    weight, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw_scales)
    magnitudes = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda", dtype=torch.float64
    )
    # Independent codebook and the existing W4A16 contract: FP32 global/group
    # scale product rounded once to FP16, followed by FP16 weight rounding.
    effective_scales = (raw_scales.float() * global_scale).half().double()
    expected = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    expected = (expected * effective_scales.repeat_interleave(16, 1)).half()
    if shared:
        tm_weight, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
            codes.T.to(torch.uint8).contiguous(),
            effective_scales.T.half().contiguous(),
            16,
            False,
        )
        k_ld, q_ld = int(meta[0]), int(meta[1])
    x = torch.zeros(rows, k, dtype=torch.float16, device="cuda")
    out = torch.empty(rows, n, dtype=torch.float16, device="cuda")

    def run():
        if shared:
            torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
                out,
                x,
                tm_weight,
                scales,
                global_scale,
                8,
                2,
                tm_scales,
                16,
                k_ld,
                q_ld,
                False,
                0,
            )
        else:
            torch.ops._C.nvfp4_qpn2_gemm_sm70_out(
                out, x, weight, scales, global_scale, 8, 2
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for begin in range(0, k, rows):
        columns = torch.arange(begin, begin + rows, device="cuda")
        x.zero_()
        x[torch.arange(rows, device="cuda"), columns] = 1
        graph.replay()
        torch.testing.assert_close(out, expected[:, columns].T, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [64, 256])
@torch.inference_mode()
def test_compact_scales_reuse_graph_scratch_without_changing_outputs(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(71)
    n = k = 2048
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    raw = torch.randint(1, 8, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    compact = torch.ops._C.nvfp4_qpn2_prepare_scales_sm70(raw)
    tm_weight, scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        codes.T.contiguous(), (raw.float() * 0.125).T.half().contiguous(), 16, False
    )
    _, scales2, _ = torch.ops._C.nvfp4_sm70_prepare(
        codes.T.contiguous(), (raw.float() * 0.25).T.half().contiguous(), 16, False
    )
    k_ld, q_ld = int(meta[0]), int(meta[1])
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    outputs = [
        torch.empty(rows, n, device="cuda", dtype=torch.float16) for _ in range(24)
    ]
    references = [torch.empty_like(outputs[0]) for _ in range(2)]

    def run():
        for i, out in enumerate(outputs):
            torch.ops._C.nvfp4_qpn2_compact_tm_gemm_sm70_out(
                out, x, tm_weight, compact, 0.125 * (1 + i % 2), k_ld, q_ld, False
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    # Warm up TurboMind on the capture stream so its own scratch is excluded.
    # One scale matrix per stream/shape, not one per layer in the captured graph.
    assert torch.cuda.memory_allocated() - before < 4 * scales.nbytes + 2 * 2**20
    for _ in range(3):
        x.normal_()
        graph.replay()
        for reference, scale in zip(references, [scales, scales2]):
            torch.ops._C.nvfp4_gemm_sm70_out(
                reference, x, tm_weight, scale, 16, k_ld, q_ld, False
            )
        for i, out in enumerate(outputs):
            torch.testing.assert_close(out, references[i % 2], rtol=0, atol=0)
