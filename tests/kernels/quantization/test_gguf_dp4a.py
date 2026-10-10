# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import dequantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def quantize_reference(x):
    groups = x.float().reshape(x.shape[0], -1, 32)
    maximum = groups.abs().amax(-1, keepdim=True)
    # CUDA uses FP32 division; Torch division by a scalar can multiply by
    # its rounded reciprocal instead. Explicit rounding avoids a different
    # side of a half-integer boundary in this independent oracle.
    d = (maximum.double() / 127).float()
    ratio = torch.where(maximum == 0, 0, (groups.double() / d.double()).float())
    q = (ratio.sign() * (ratio.abs() + 0.5).floor()).to(torch.int8)
    scales = d.squeeze(-1).half()
    sums = groups.sum(-1).half()
    return q, scales, sums


@pytest.mark.parametrize("m", [1, 5, 20])
def test_q8_1_matches_round_away_oracle_and_graph(m):
    torch.manual_seed(970 + m)
    x = torch.randn((m, 768), device="cuda", dtype=torch.float16)
    x[:, :32] = 0
    x[0, 32:64] = torch.tensor([127, 0.5, -0.5, 63.5, -63.5] + [0] * 27, device="cuda")
    out = torch.empty((m, 24, 36), device="cuda", dtype=torch.uint8)

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(out, x)

    def check():
        q, d, s = quantize_reference(x)
        torch.testing.assert_close(out[:, :, 4:].view(torch.int8), q, rtol=0, atol=0)
        ds = out[:, :, :4].contiguous().view(torch.float16)
        torch.testing.assert_close(ds[:, :, 0], d, rtol=0, atol=0)
        torch.testing.assert_close(ds[:, :, 1], s, rtol=0, atol=0)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x))
    graph.replay()
    check()


@pytest.mark.parametrize("m", [1, 5, 20])
@pytest.mark.parametrize("activated", [False, True])
@pytest.mark.parametrize(
    "source_type,block_bytes", [(18, 98), (20, 18), (21, 110), (22, 82), (23, 136)]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_lattice_dot_matches_official_weight_and_q8_oracle(
    m, activated, source_type, block_bytes, index_dtype
):
    experts, n, k, top_k = 4, 7, 768, 2
    block = 32 if source_type == 20 else 256
    rng = np.random.default_rng(21)
    weights, reference = [], []
    for _ in range(2):
        data = rng.integers(
            0, 256, (experts * n, k // block, block_bytes), dtype=np.uint8
        )
        d = rng.uniform(0.001, 0.005, data.shape[:2]).astype("<f2")
        data[:, :, :2] = d[..., None].view(np.uint8)
        data = data.reshape(experts * n, -1)
        packed = (
            data
            if source_type in (20, 23)
            else RawGGUFProjection.from_rows(data, source_type).data
        )
        weights.append(torch.from_numpy(packed.reshape(experts, n, -1)).cuda())
        if source_type == 23:
            from vllm.model_executor.layers.quantization.gguf_lut_transcode import (
                transcode_lut4,
            )

            decoded = transcode_lut4(data, source_type).dequantize()
        else:
            decoded = dequantize(data, source_type)
        reference.append(torch.from_numpy(decoded).reshape(experts, n, k).cuda())
    torch.manual_seed(970 + m)
    x = (torch.randn((m, k), device="cuda") * 0.125).half()
    ids = torch.stack(
        (torch.zeros(m, dtype=torch.int64), torch.arange(m) % 3 + 1), 1
    ).to(device="cuda", dtype=index_dtype)
    q8 = torch.empty((m, k // 32, 36), dtype=torch.uint8, device="cuda")
    out = torch.empty(
        (m, top_k, n) if activated else (m, top_k, 2, n),
        dtype=torch.float16,
        device="cuda",
    )

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            out, q8, ids, *weights, source_type, activated
        )

    def check():
        q, d, _ = quantize_reference(x)
        quantized = (q.float() * d[..., None].float()).reshape(m, k)
        gate, up = [
            torch.einsum("mk,mtnk->mtn", quantized, w[ids]).half() for w in reference
        ]
        expected = (
            (torch.nn.functional.silu(gate) * up)
            if activated
            else torch.stack((gate, up), 2)
        )
        torch.testing.assert_close(
            out.float(), expected.float(), rtol=0.002, atol=0.001
        )
        assert torch.isfinite(out).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x) * 0.125)
    graph.replay()
    check()


@pytest.mark.parametrize("quantized_input", [False, True])
@pytest.mark.parametrize("m", [5, 20])
@pytest.mark.parametrize("source_type", [20, 42])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_down_unroute_matches_tp4_official_weights(
    m, source_type, index_dtype, quantized_input
):
    from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
        GGUFExpertBank,
    )
    from vllm.transformers_utils.gguf_tensor_reader import quant_size

    experts, n, full_k, k, top_k = 3, 64, 640, 160, 2
    block, size = quant_size(source_type)
    rng = np.random.default_rng(source_type)
    bank = GGUFExpertBank(source_type, experts, torch.device("cuda"), torch.float16)
    reference = []
    for expert in range(experts):
        data = rng.integers(0, 256, (n, full_k // block, size), dtype=np.uint8)
        d = rng.uniform(0.001, 0.005, data.shape[:2]).astype("<f2")
        data[:, :, :2] = d[..., None].view(np.uint8)
        data = data.reshape(n, -1)
        bank.add(expert, torch.from_numpy(data), rank=1, size=4, axis=1)
        reference.append(torch.from_numpy(dequantize(data, source_type)[:, k : 2 * k]))
    bank.finalize()
    weights = torch.stack(reference).cuda()
    torch.manual_seed(970 + m)
    hidden = torch.randn((m, top_k, k), device="cuda", dtype=torch.float16)
    ids = torch.stack((torch.arange(m) % experts, torch.zeros(m)), 1).to(
        device="cuda", dtype=index_dtype
    )
    probabilities = torch.softmax(torch.randn((m, top_k), device="cuda"), 1)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)

    def run():
        input = hidden
        if quantized_input:
            q, d, s = quantize_reference(hidden.reshape(m * top_k, k))
            ds = torch.stack((d, s), dim=-1).contiguous().view(torch.uint8)
            input = torch.cat((ds, q.view(torch.uint8)), dim=-1).reshape(
                m, top_k, k // 32, 36
            )
        torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
            out,
            input,
            ids,
            probabilities,
            bank.weight_ptrs,
            bank.stat_ptrs,
            source_type,
            experts,
        )

    def check():
        q, d, _ = quantize_reference(hidden.reshape(m * top_k, k))
        quantized = (q.float() * d[..., None].float()).reshape(m, top_k, k)
        down = torch.einsum("mtk,mtnk->mtn", quantized, weights[ids]).half()
        expected = (down.float() * probabilities[..., None]).sum(1).half()
        torch.testing.assert_close(out, expected, rtol=0.002, atol=0.002)
        assert torch.isfinite(out).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    hidden.copy_(torch.randn_like(hidden))
    probabilities.copy_(torch.softmax(torch.randn_like(probabilities), 1))
    ids.copy_((ids + 1) % experts)
    graph.replay()
    check()


@pytest.mark.parametrize("m", [5, 20])
@pytest.mark.parametrize(
    "source_type,size", [(18, 98), (20, 18), (21, 110), (22, 82), (23, 136)]
)
@pytest.mark.parametrize("lanes", [4, 8, 16])
def test_fused_gated_q8_keeps_fp16_boundary_and_changed_graph(
    m, source_type, size, lanes
):
    experts, n, k, top_k = 3, 160, 768, 2
    torch.manual_seed(1167 + m + source_type)
    rng = np.random.default_rng(source_type)
    weights = []
    for _ in range(2):
        block = 32 if source_type == 20 else 256
        raw = rng.integers(0, 256, (experts * n, k // block, size), dtype=np.uint8)
        d = rng.uniform(0.0001, 0.001, raw.shape[:2]).astype("<f2")
        raw[:, :, :2] = d[..., None].view(np.uint8)
        data = raw.reshape(experts * n, -1)
        packed = (
            data
            if source_type in (20, 23)
            else RawGGUFProjection.from_rows(data, source_type).data
        )
        weights.append(torch.from_numpy(packed.reshape(experts, n, -1)).cuda())
    x = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.125
    ids = torch.randint(experts, (m, top_k), device="cuda", dtype=torch.int32)
    activation = torch.empty((m, k // 32, 36), device="cuda", dtype=torch.uint8)
    hidden = torch.empty((m, top_k, n), device="cuda", dtype=torch.float16)
    out = torch.empty((m, top_k, n // 32, 36), device="cuda", dtype=torch.uint8)

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(activation, x)
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            hidden, activation, ids, *weights, source_type, True
        )
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            out, activation, ids, *weights, source_type, True, lanes
        )

    def check():
        q, d, s = quantize_reference(hidden.reshape(m * top_k, n))
        if lanes == 16:
            torch.testing.assert_close(
                out[..., 4:].view(torch.int8).reshape_as(q), q, rtol=0, atol=0
            )
            ds = out[..., :4].contiguous().view(torch.float16)
            torch.testing.assert_close(ds[..., 0].reshape_as(d), d, rtol=0, atol=0)
            torch.testing.assert_close(ds[..., 1].reshape_as(s), s, rtol=0, atol=0)
        else:
            # Changing the lane reduction tree can move an FP16 rounding
            # boundary. Compare against independently quantized FP16 values;
            # comparing Q8 directly with FP16 confounds this with its expected
            # half-step error, especially for the wider IQ4_XS dynamic range.
            actual_q = out[..., 4:].view(torch.int8).reshape_as(q)
            assert (actual_q.int() - q.int()).abs().max() <= 1
            ds = out[..., :4].contiguous().view(torch.float16)
            torch.testing.assert_close(
                ds[..., 0].reshape_as(d), d, rtol=0.002, atol=1e-6
            )
            sum_error = (ds[..., 1].reshape_as(s).float() - s.float()).abs()
            magnitude = hidden.float().reshape(m * top_k, -1, 32).abs().sum(-1)
            assert (sum_error <= 0.002 * magnitude + 1e-4).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x) * 0.125)
    ids.copy_((ids + 1) % experts)
    graph.replay()
    check()


@pytest.mark.parametrize("m", [5, 20])
@pytest.mark.parametrize("source_type", [20, 42])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("quantized_input", [False, True])
def test_original_down_handles_all_tp4_block_boundaries(
    m, source_type, rank, quantized_input
):
    from vllm.model_executor.layers.quantization.gguf_native import packed_tp_span
    from vllm.transformers_utils.gguf_tensor_reader import quant_size

    experts, n, full_k, k, top_k = 3, 64, 640, 160, 2
    block, size = quant_size(source_type)
    rng = np.random.default_rng(1165 + source_type)
    raw = rng.integers(0, 256, (experts, n, full_k // block, size), dtype=np.uint8)
    raw[..., :2] = (
        rng.uniform(0.001, 0.005, raw.shape[:-1])
        .astype("<f2")[..., None]
        .view(np.uint8)
    )
    full = raw.reshape(experts, n, -1)
    first, last, left, _ = packed_tp_span(full_k, source_type, 4, rank)
    weight = torch.from_numpy(full[..., first:last].copy()).cuda()
    reference = torch.from_numpy(
        dequantize(full, source_type)[..., rank * k : (rank + 1) * k]
    ).cuda()
    hidden = torch.randn((m, top_k, k), device="cuda", dtype=torch.float16)
    ids = torch.randint(experts, (m, top_k), device="cuda", dtype=torch.int32)
    probabilities = torch.softmax(torch.randn((m, top_k), device="cuda"), 1)
    output = torch.empty((m, n), device="cuda", dtype=torch.float16)
    packet = torch.empty((m * top_k, k // 32, 36), device="cuda", dtype=torch.uint8)

    def run():
        x = hidden
        if quantized_input:
            torch.ops._C.gguf_quantize_q8_1_sm70_out(
                packet, hidden.reshape(m * top_k, k)
            )
            x = packet.reshape(m, top_k, k // 32, 36)
        torch.ops._C.gguf_dp4a_raw_down_unroute_sm70_out(
            output,
            x,
            ids,
            probabilities,
            weight,
            source_type,
            left,
        )

    def check():
        q, d, _ = quantize_reference(hidden.reshape(m * top_k, k))
        x = (q.float() * d[..., None].float()).reshape_as(hidden)
        down = torch.einsum("mtk,mtnk->mtn", x, reference[ids]).half()
        expected = (down.float() * probabilities[..., None]).sum(1).half()
        torch.testing.assert_close(output, expected, rtol=0.002, atol=0.002)
        assert torch.isfinite(output).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    hidden.copy_(torch.randn_like(hidden))
    ids.copy_((ids + 1) % experts)
    graph.replay()
    check()
