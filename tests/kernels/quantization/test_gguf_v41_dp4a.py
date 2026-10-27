# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 original integer dots against the exact Q8_1 arithmetic boundary."""

import gguf
import numpy as np
import pytest
import torch
from test_gguf_dp4a import quantize_reference
from test_gguf_lattice_transcode import source

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM70 required")


def integer_weights(data, kind, scale, n, k):
    dense = gguf.quants.dequantize(
        data.reshape(-1, data.shape[-1]), gguf.GGMLQuantizationType(kind)
    )
    d = np.float32(np.float16(scale))
    # All three formats have integer eighth-step values after multiplying
    # local odd scales into the grid. This derives the integers independently
    # from gguf-py's official dequantizer, not from the new CUDA reader.
    return np.rint(dense / d * np.float32(8)).astype(np.int16).reshape(-1, n, k)


def group_values(integer, q, d, scale):
    m, top_k, groups, _ = q.shape
    n = integer.shape[2]
    result = np.empty((m, top_k, n, groups), dtype=np.float32)
    for token in range(m):
        dot = np.einsum(
            "tngv,tgv->tng",
            integer[token].reshape(top_k, n, groups, 32),
            q[token],
            dtype=np.int32,
            optimize=False,
        )
        coefficient = np.float32(np.float16(scale)) * d[token].astype(np.float32)
        result[token] = dot.astype(np.float32) * coefficient[:, None, :]
    return result * np.float32(0.125)


def gate_reference(integer, q, d, scale):
    values = group_values(integer, q, d, scale)
    partial = np.zeros((*values.shape[:-1], 16), dtype=np.float32)
    for start in range(0, values.shape[-1], 16):
        partial += values[..., start : start + 16]
    for offset in (8, 4, 2, 1):
        partial[..., :offset] += partial[..., offset : 2 * offset]
    return partial[..., 0].astype(np.float16)


def down_reference(integer, q, d, scale):
    values = group_values(integer, q, d, scale)
    sums = np.zeros(values.shape[:-1], dtype=np.float32)
    for group in range(values.shape[-1]):
        sums += values[..., group]
    rounded = sums.astype(np.float16).astype(np.float32)
    partials = (
        rounded[:, 0] + rounded[:, 4],
        rounded[:, 1] + rounded[:, 5],
        rounded[:, 2],
        rounded[:, 3],
    )
    expected = np.zeros((values.shape[0], values.shape[2]), dtype=np.float32)
    for part in partials:
        expected += part
    return expected.astype(np.float16)


@pytest.mark.parametrize("kind", [16, 17, 29])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("scale", [0.0009765625, 0.00137])
def test_v41_gate_up_exact_integer_groups_fp16_boundary_and_graph(kind, m, scale):
    experts, n, k, top_k = 6, 576, 5120, 6
    data = [
        source(kind, scale, n=experts * n, k=k, seed=41 + i).reshape(experts, n, -1)
        for i in range(2)
    ]
    weights = [torch.from_numpy(value).cuda() for value in data]
    integer = [integer_weights(value, kind, scale, n, k) for value in data]
    ids = (
        torch.arange(m, device="cuda")[:, None] + torch.arange(top_k, device="cuda")
    ) % experts
    ids = ids.int().contiguous()
    torch.manual_seed(41)
    x = (torch.randn(m, k, device="cuda") * 0.25).half()
    q8 = torch.empty((m, k // 32, 36), dtype=torch.uint8, device="cuda")
    out = torch.empty((m, top_k, 2, n), dtype=torch.half, device="cuda")

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            out, q8, ids, *weights, kind, False, 16, False
        )

    def check():
        q, d, _ = quantize_reference(x)
        np.testing.assert_array_equal(
            q8[:, :, 4:].cpu().numpy().view(np.int8), q.cpu().numpy()
        )
        selected = ids.cpu().numpy()
        repeated_q = np.repeat(q.cpu().numpy()[:, None], top_k, axis=1)
        repeated_d = np.repeat(d.cpu().numpy()[:, None], top_k, axis=1)
        for part in range(2):
            values = group_values(
                integer[part][selected], repeated_q, repeated_d, scale
            )
            partial = np.zeros((*values.shape[:-1], 16), dtype=np.float32)
            for start in range(0, k // 32, 16):
                partial += values[..., start : start + 16]
            for offset in (8, 4, 2, 1):
                partial[..., :offset] += partial[..., offset : 2 * offset]
            expected = partial[..., 0].astype(np.float16)
            assert np.isfinite(expected).all()
            np.testing.assert_array_equal(out[:, :, part].cpu().numpy(), expected)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.mul_(-0.75)
    ids.copy_((ids + 1) % experts)
    graph.replay()
    check()


@pytest.mark.parametrize("kind", [16, 17, 29])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("left", [0, 64, 128, 192])
@pytest.mark.parametrize("tiles", [None, 1, 2, 4, 8])
def test_v41_down_exact_integer_boundary_all_tp_offsets(kind, m, left, tiles):
    experts, n, physical_k, k, top_k, scale = 6, 5120, 768, 576, 6, 0.00137
    data = source(kind, scale, n=experts * n, k=physical_k).reshape(experts, n, -1)
    raw = torch.from_numpy(data).cuda()
    integer = integer_weights(data, kind, scale, n, physical_k)[:, :, left : left + k]
    ids = (
        torch.arange(m, device="cuda")[:, None] + torch.arange(top_k, device="cuda")
    ) % experts
    ids = ids.int().contiguous()
    torch.manual_seed(41)
    hidden = (torch.randn(m, top_k, k, device="cuda") * 0.25).half()
    probabilities = torch.ones((m, top_k), dtype=torch.float32, device="cuda")
    out = torch.empty((m, n), dtype=torch.half, device="cuda")
    q8 = torch.empty((m, top_k, k // 32, 36), dtype=torch.uint8, device="cuda")

    def run():
        if tiles is not None:
            torch.ops._C.gguf_quantize_q8_1_sm70_out(
                q8.view(-1, k // 32, 36), hidden.view(-1, k)
            )
            torch.ops._C.gguf_v41_dp4a_q8_down_tiled_sm70_out(
                out, q8, ids, probabilities, raw, kind, left, tiles
            )
        else:
            torch.ops._C.gguf_dp4a_raw_down_unroute_sm70_out(
                out, hidden, ids, probabilities, raw, kind, left
            )

    def check():
        q, d, _ = quantize_reference(hidden.reshape(m * top_k, k))
        q = q.cpu().numpy().reshape(m, top_k, k // 32, 32)
        d = d.cpu().numpy().reshape(m, top_k, k // 32)
        values = group_values(integer[ids.cpu().numpy()], q, d, scale)
        sums = np.zeros(values.shape[:-1], dtype=np.float32)
        for group in range(k // 32):
            sums += values[..., group]
        rounded = sums.astype(np.float16).astype(np.float32)
        partials = (
            rounded[:, 0] + rounded[:, 4],
            rounded[:, 1] + rounded[:, 5],
            rounded[:, 2],
            rounded[:, 3],
        )
        expected = np.zeros((m, n), dtype=np.float32)
        for part in partials:
            expected += part
        expected = expected.astype(np.float16)
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(out.cpu().numpy(), expected)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    hidden.mul_(-0.75)
    ids.copy_((ids + 1) % experts)
    graph.replay()
    check()


@pytest.mark.parametrize("gate_kind", [16, 17, 29])
@pytest.mark.parametrize("down_kind", [16, 17, 29])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_v41_expert_clamps_and_weights_the_down_input(gate_kind, down_kind, m):
    from vllm.model_executor.layers.quantization import gguf_v41_moe

    assert gguf_v41_moe.v41_dp4a_available()
    n, k, left, physical_k, top_k = 576, 5120, 128, 768, 6
    gate_scale, down_scale = 0.0625, 0.00137
    gate_data = [
        source(gate_kind, gate_scale, n=n, k=k, seed=41 + i).reshape(1, n, -1)
        for i in range(2)
    ]
    down_data = source(down_kind, down_scale, n=k, k=physical_k).reshape(1, k, -1)
    weights = [torch.from_numpy(value).cuda() for value in (*gate_data, down_data)]
    integer_gate = [
        integer_weights(value, gate_kind, gate_scale, n, k) for value in gate_data
    ]
    integer_down = integer_weights(down_data, down_kind, down_scale, k, physical_k)
    integer_down = integer_down[:, :, left : left + n]
    ids = torch.zeros((m, top_k), dtype=torch.int32, device="cuda")
    probabilities = torch.tensor([0.0001, 0.01, 0.0499, 0.14, 0.3, 0.5], device="cuda")
    probabilities = probabilities[None].expand(m, -1).contiguous()
    torch.manual_seed(41)
    x = (torch.randn(m, k, device="cuda") * 0.5).half()

    def run():
        return torch.ops.vllm.gguf_v41_original_expert_dp4a(
            x,
            ids,
            probabilities,
            *weights,
            gate_kind,
            down_kind,
            left,
            10.0,
        )

    def check(actual):
        selected = ids.cpu().numpy()
        q, d, _ = quantize_reference(x)
        q = np.repeat(q.cpu().numpy()[:, None], top_k, axis=1)
        d = np.repeat(d.cpu().numpy()[:, None], top_k, axis=1)
        projected = [
            torch.from_numpy(gate_reference(value[selected], q, d, gate_scale))
            .cuda()
            .float()
            for value in integer_gate
        ]
        # The official Expert.forward keeps these operations in FP32 and
        # casts only after routing weights multiply the down input.
        expected_hidden = (
            torch.nn.functional.silu(projected[0].clamp(max=10))
            * projected[1].clamp(-10, 10)
            * probabilities[..., None]
        ).half()
        qh, dh, _ = quantize_reference(expected_hidden.reshape(m * top_k, n))
        qh = qh.cpu().numpy().reshape(m, top_k, n // 32, 32)
        dh = dh.cpu().numpy().reshape(m, top_k, n // 32)
        expected = down_reference(integer_down[selected], qh, dh, down_scale)
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)

    check(run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    x.mul_(-0.75)
    probabilities.copy_(probabilities.flip(1))
    graph.replay()
    check(out)
