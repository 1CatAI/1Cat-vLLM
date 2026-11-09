# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exhaustive FP8 conversion and live-length graph checks for the KV bridge."""

import pytest
import torch


def test_e4m3_bridge_missing_native_entry_fails_closed(monkeypatch):
    native = pytest.importorskip("flash_attn_v100_cuda")
    from flash_attn_v100 import fp8_e4m3_paged_kv_to_fp16

    monkeypatch.delattr(native, "fp8_e4m3_paged_kv_to_fp16", raising=False)
    dummy = torch.empty(0)
    with pytest.raises(RuntimeError, match="Rebuild Flash-V100"):
        fp8_e4m3_paged_kv_to_fp16(dummy, dummy, dummy, dummy, dummy, dummy)


@pytest.mark.parametrize("dtype_name", ["e4m3", "e5m2"])
@pytest.mark.parametrize("page", [800, 848, 1616, 1648, 3296])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 1.25), (0.003, 3.14159)])
def test_fp8_bridge_all_codes_and_graph_lengths(dtype_name, page, scales):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    native = pytest.importorskip("flash_attn_v100_cuda")
    op = getattr(native, f"fp8_{dtype_name}_paged_kv_to_fp16", None)
    if op is None:
        pytest.skip("rebuild the explicit FP8 bridge entries")
    if dtype_name == "e4m3":
        from flash_attn_v100 import fp8_e4m3_paged_kv_to_fp16

        op = fp8_e4m3_paged_kv_to_fp16
    dtype = torch.float8_e4m3fn if dtype_name == "e4m3" else torch.float8_e5m2
    torch.manual_seed(20260906)
    batch, blocks, heads, dim, output_page = 2, 3, 1, 256, 32
    backing = torch.empty(
        (batch * blocks, 2, page, heads, dim), dtype=torch.uint8, device="cuda"
    )
    k, v = backing.unbind(1)
    codes = torch.arange(256, device="cuda", dtype=torch.int32).byte()
    k.copy_(codes)
    v.copy_(codes.flip(0))
    table = torch.randperm(batch * blocks, device="cuda").int().reshape(batch, blocks)
    capacity = blocks * page
    output_blocks = (capacity + output_page - 1) // output_page
    output = torch.empty(
        (batch * output_blocks, 2, output_page, heads, dim),
        dtype=torch.float16,
        device="cuda",
    )
    ko, vo = output.unbind(1)
    lengths = torch.tensor([capacity - 5, page + 3], device="cuda", dtype=torch.int32)
    initial = lengths.clone()

    def call():
        assert op is not None
        op(k, v, table, lengths, ko, vo, *scales)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for state in ("live", "zero", "short", "restore"):
        lengths.copy_(initial)
        if state == "zero":
            lengths.zero_()
        elif state == "short":
            lengths.copy_(torch.tensor([1, 17], device="cuda", dtype=torch.int32))
        output.fill_(float("nan"))
        graph.replay()
        for row, length in enumerate(lengths.tolist()):
            for src, dst, scale in ((k, ko, scales[0]), (v, vo, scales[1])):
                actual = dst[row * output_blocks : (row + 1) * output_blocks].flatten(
                    0, 1
                )
                expected = (
                    src[table[row].long()].flatten(0, 1).view(dtype).float() * scale
                ).half()
                torch.testing.assert_close(
                    actual[:length], expected[:length], rtol=0, atol=0, equal_nan=True
                )
                zero = expected[:length] == 0
                assert torch.equal(
                    torch.signbit(actual[:length])[zero],
                    torch.signbit(expected[:length])[zero],
                )
                padded = (length + 15) // 16 * 16
                assert bool((actual[length:padded] == 0).all())
                assert bool(torch.isnan(actual[padded:]).all())
