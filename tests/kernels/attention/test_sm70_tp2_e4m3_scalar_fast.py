# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise gates for the opt-in TP2 E4M3 scalar decode implementation."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

FLAG = "VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST"


def test_requested_fast_path_rejects_stale_library(monkeypatch):
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(
        interface,
        "flash_attn_v100_cuda",
        SimpleNamespace(grouped_e4m3_fp32_precision_version=lambda: 4),
    )
    monkeypatch.setattr(
        interface, "flash_attn_grouped_e4m3_fp32_available", lambda: True
    )
    monkeypatch.setattr(
        interface,
        "_get_decode_plan",
        lambda *args, **kwargs: SimpleNamespace(partition_size=1024),
    )
    monkeypatch.setattr(
        interface, "_assert_decode_launch_covers_seq_lens", lambda *args, **kwargs: None
    )
    q = torch.empty((8, 12, 256), dtype=torch.float16)
    kv = torch.empty((1, 3296, 2, 256), dtype=torch.uint8)
    table = torch.zeros((8, 1), dtype=torch.int32)
    seq = torch.zeros(8, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="TP2 E4M3 scalar fast revision 1"):
        interface.flash_attn_decode_paged(
            q, kv, kv, table, seq, kv_cache_dtype="fp8_e4m3"
        )


@pytest.fixture
def native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    extension = interface.flash_attn_v100_cuda
    version = getattr(extension, "tp2_e4m3_scalar_fast_version", lambda: 0)
    if version() < 1:
        pytest.skip("rebuild Flash-V100 with TP2 scalar fast revision 1")
    return extension


def test_all_256_e4m3_encodings(native, tmp_path):
    from torch.utils.cpp_extension import load_inline

    extension = load_inline(
        name="tp2_e4m3_codec_test",
        cpp_sources="void decode_lut(torch::Tensor output);",
        cuda_sources=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "fp8_kv_utils.cuh"
__global__ void tp2_e4m3_codec_test_kernel(float* output) {
  const uint8_t raw = threadIdx.x;
  output[raw] = flash_v100::fp8_e4m3fn_to_float(raw);
  output[256 + raw] = flash_v100::fp8_e4m3fn_to_float_bits(raw);
}
void decode_lut(torch::Tensor output) {
  tp2_e4m3_codec_test_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<float>());
}
""",
        functions=["decode_lut"],
        extra_include_paths=[
            str(Path(__file__).resolve().parents[3] / "flash-attention-v100/kernel")
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(tmp_path),
    )
    output = torch.empty((2, 256), device="cuda", dtype=torch.float32)
    extension.decode_lut(output)
    assert torch.equal(output[0].view(torch.int32), output[1].view(torch.int32))
    expected = torch.arange(256, device="cuda").byte().view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(expected)
    assert torch.equal(
        output[1, finite].view(torch.int32), expected[finite].view(torch.int32)
    )
    assert bool(torch.isnan(output[1, ~finite]).all())


def workspace(q, parts):
    rows, heads, dim = q.shape
    # Check that a noncontiguous output does not overwrite adjacent storage.
    storage = torch.full((rows, heads, dim + 16), 123, device=q.device, dtype=q.dtype)
    return (
        storage[..., :dim],
        torch.empty((rows, heads, parts, dim), device=q.device),
        torch.empty((rows, heads, parts), device=q.device),
        torch.empty((rows, heads, parts), device=q.device),
        storage,
    )


@pytest.mark.parametrize("length", [1025, 3297, 262144])
def test_live_graph_bitwise_state_and_fp64_reference(native, monkeypatch, length):
    torch.manual_seed(20260908)
    page, parts = 3296, 256
    pages = (length + page - 1) // page
    # Interleaved, strided K/V with a nonidentity page table, as in the service.
    kv = torch.randn((pages, 2, page, 2, 256), device="cuda", dtype=torch.float16)
    kv = kv.to(torch.float8_e4m3fn).view(torch.uint8)
    k, v = kv.unbind(1)
    order = torch.randperm(pages, device="cuda").int()
    table = order[None].repeat(8, 1)
    q = torch.randn((8, 12, 256), device="cuda", dtype=torch.float16)
    q_initial = q.clone()
    seq = torch.zeros(8, device="cuda", dtype=torch.int32)
    active = torch.full((1,), parts, device="cuda", dtype=torch.int32)
    states = [workspace(q, parts), workspace(q, parts)]
    graphs = []
    for enabled, state in enumerate(states):
        monkeypatch.setenv(FLAG, str(enabled))

        def call(state=state):
            native.decode_paged_fwd(
                q,
                k,
                v,
                state[0],
                table,
                seq,
                *state[1:4],
                active,
                0.0625,
                1024,
                parts,
                "fp8_e4m3",
                0.5,
                1.25,
                -1,
                -1,
                None,
                0,
            )

        before = native.tp2_e4m3_scalar_fast_launch_count()
        call()
        assert native.tp2_e4m3_scalar_fast_launch_count() - before == enabled
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs.append(graph)

    initial_seq = torch.arange(length - 7, length + 1, device="cuda").int()
    for replay in range(4):
        seq.copy_(initial_seq)
        if replay == 1:
            seq[0] = 0
        elif replay == 2:
            seq.zero_()
        q.copy_(q_initial * (0.5 if replay % 2 else 1.0))
        for state in states:
            for tensor in state[:4]:
                tensor.fill_(float("nan"))
        for graph in graphs:
            graph.replay()
        assert torch.equal(
            states[0][0].view(torch.int16), states[1][0].view(torch.int16)
        )
        assert bool(torch.isfinite(states[1][0]).all())
        valid = (
            torch.arange(parts, device="cuda")[None, None]
            < ((seq + 1023) // 1024)[:, None, None]
        )
        valid = valid.expand(8, 12, parts)
        for control, candidate in zip(states[0][1:4], states[1][1:4]):
            assert torch.equal(
                control[valid].view(torch.int32), candidate[valid].view(torch.int32)
            )
        for state in states:
            assert bool((state[4][..., 256:] == 123).all())

        if length == 3297 and replay == 0:
            rk = k[order.long()].reshape(-1, 2, 256)[:length]
            rv = v[order.long()].reshape(-1, 2, 256)[:length]
            rk = rk.view(torch.float8_e4m3fn).double() * 0.5
            rv = rv.view(torch.float8_e4m3fn).double() * 1.25
            expected = torch.empty_like(q, dtype=torch.float64)
            for head in range(2):
                score = (
                    q[:, head * 6 : (head + 1) * 6].transpose(0, 1).double()
                    @ rk[:, head].T
                    * 0.0625
                )
                score.masked_fill_(
                    torch.arange(length, device="cuda")[None, None]
                    >= seq[None, :, None],
                    -torch.inf,
                )
                expected[:, head * 6 : (head + 1) * 6] = (
                    score.softmax(-1) @ rv[:, head]
                ).transpose(0, 1)
            error = (states[1][0].double() - expected).norm() / expected.norm()
            floor = (expected.half().double() - expected).norm() / expected.norm()
            assert float(error) <= 1.04 * float(floor) + 2e-6


@pytest.mark.parametrize(
    "rows,heads,kv_heads,partition,window",
    [
        (4, 12, 2, 1024, -1),
        (8, 6, 1, 1024, -1),
        (8, 12, 2, 256, -1),
        (8, 12, 2, 1024, 127),
    ],
)
def test_unverified_shapes_use_original_route(
    native, monkeypatch, rows, heads, kv_heads, partition, window
):
    q = torch.randn((rows, heads, 256), device="cuda", dtype=torch.float16)
    kv = torch.randn((2, 1, 1024, kv_heads, 256), device="cuda", dtype=torch.float16)
    k, v = kv.to(torch.float8_e4m3fn).view(torch.uint8).unbind(0)
    seq = torch.full((rows,), 65, device="cuda", dtype=torch.int32)
    table = torch.zeros((rows, 1), device="cuda", dtype=torch.int32)
    parts = 1024 // partition
    active = torch.full((1,), parts, device="cuda", dtype=torch.int32)
    results = []
    before = native.tp2_e4m3_scalar_fast_launch_count()
    for enabled in ("0", "1"):
        monkeypatch.setenv(FLAG, enabled)
        state = workspace(q, parts)
        native.decode_paged_fwd(
            q,
            k,
            v,
            state[0],
            table,
            seq,
            *state[1:4],
            active,
            0.0625,
            partition,
            parts,
            "fp8_e4m3",
            0.5,
            1.25,
            window,
            -1,
            None,
            0,
        )
        results.append(state[0])
    assert native.tp2_e4m3_scalar_fast_launch_count() == before
    assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))
