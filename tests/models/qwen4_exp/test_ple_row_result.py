# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import ple_layer as ple


@pytest.mark.parametrize("active", [0, 3, 5])
def test_owned_conv_result_preserves_input_and_initialized_padding(monkeypatch, active):
    source = torch.full((5, 4), 7, dtype=torch.half)
    before = source.clone()
    result = torch.arange(active * 4).reshape(active, 4).half()
    owners = {
        "first": NS(_short_conv=lambda x: result),
        "second": NS(_short_conv=lambda x: result + 10),
    }
    monkeypatch.setattr(
        ple, "get_forward_context", lambda: NS(no_compile_layers=owners)
    )
    for name, offset in (("first", 0), ("second", 10)):
        output = ple.qwen4_exp_ple_short_conv_result(source, name)
        assert output.shape == source.shape and output.dtype == source.dtype
        assert torch.equal(output[:active], result + offset)
        assert bool((output[active:] == 0).all())
        assert torch.equal(source, before)
        assert output.data_ptr() != source.data_ptr()
    if active == 5:
        assert (
            ple.qwen4_exp_ple_short_conv_result(source, "first").data_ptr()
            == result.data_ptr()
        )


@pytest.mark.parametrize("active", [0, 3, 5])
@pytest.mark.parametrize("rank", range(4))
def test_owned_conv_rows_keep_profile_padding_and_input(monkeypatch, active, rank):
    source = torch.full((5, 4), 7, dtype=torch.half)
    before = source.clone()
    result = torch.arange(active * 4).reshape(active, 4).half()
    layer = ple.Qwen4ExpPLELayer.__new__(ple.Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._short_conv_fallback = lambda x: result
    monkeypatch.setattr(
        ple,
        "get_forward_context",
        lambda: NS(attn_metadata=None, no_compile_layers={"owner": layer}),
    )
    output = ple.qwen4_exp_ple_short_conv_rows(source, "owner", rank, 4)
    expected = source.new_zeros(source.shape)
    expected[:active] = result
    expected = torch.nn.functional.pad(expected, (0, 0, 0, 3))
    assert torch.equal(output, expected[rank * 2 : rank * 2 + 2])
    assert output.shape == (2, 4)
    assert torch.equal(source, before)
    assert output.data_ptr() != source.data_ptr()
    for bad_rank, parts in [(-1, 4), (4, 4), (0, 0)]:
        with pytest.raises(ValueError):
            ple.qwen4_exp_ple_short_conv_rows(source, "owner", bad_rank, parts)


@pytest.mark.parametrize("rows", [0, 1, 7, 31, 32, 33, 65])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_conv_rows_preserve_complete_state_and_graph(rows):
    from vllm.model_executor.kernels.ple.short_conv import dilated_prefill_short_conv
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    torch.manual_seed(6015 + rows)
    x = torch.randn(rows, 32, device="cuda", dtype=torch.half)
    before = x.clone()
    state = torch.randn(6, 32, 11, device="cuda", dtype=torch.half)
    initial = state.clone()
    weights = torch.randn(32, 4, device="cuda", dtype=torch.half)
    starts = torch.tensor([0, 0, rows // 3, 2 * rows // 3, rows], device="cuda")
    slots = torch.tensor([1, 2, NULL_BLOCK_ID, 4], device="cuda", dtype=torch.int32)
    has_initial = torch.tensor([True, True, True, False], device="cuda")
    max_length = rows - 2 * rows // 3

    def run(**owned):
        return dilated_prefill_short_conv(
            x,
            state,
            weights,
            starts,
            slots,
            has_initial,
            max_length,
            0,
            3,
            NULL_BLOCK_ID,
            **owned,
        )

    expected = run()
    expected_state = state.clone()
    width = (rows + 3) // 4
    padded = torch.nn.functional.pad(expected, (0, 0, 0, width * 4 - rows))
    for rank in range(4):
        state.copy_(initial)
        actual = run(output_start=rank * width, output_rows=width)
        torch.testing.assert_close(
            actual, padded[rank * width : (rank + 1) * width], rtol=0, atol=0
        )
        assert torch.equal(state, expected_state)
        assert torch.equal(x, before)
    # An owner with no output still commits all requests' causal histories.
    state.copy_(initial)
    assert run(output_start=0, output_rows=0).shape == (0, 32)
    assert torch.equal(state, expected_state)
    if rows == 33:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(output_start=3 * width, output_rows=width)
        for _ in range(3):
            x.mul_(0.75)
            state.copy_(initial)
            expected = run()
            expected_state = state.clone()
            padded = torch.nn.functional.pad(expected, (0, 0, 0, width * 4 - rows))
            state.copy_(initial)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(captured, padded[3 * width :])
            assert torch.equal(state, expected_state)
