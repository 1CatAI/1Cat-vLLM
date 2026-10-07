# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.models.qwen4_exp.test_ple_short_conv_prefill import (
    _case,
    _reference_prefill_batched,
)
from vllm.models.qwen4_exp.nvidia.ops.ple_prefill_conv import prefill_conv_reason
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires SM70",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
@torch.inference_mode()
def test_ragged_prefill_null_empty_initial_and_tail(dilation, state_dtype):
    module, x, args = _case(
        "cuda",
        torch.float16,
        lengths=[511, 0, 257, 1],
        decode_tokens=2,
        state_indices=[2, 1, 0, 3],
        has_initial=[True, False, True, False],
        hidden_size=129,
        dilation=dilation,
    )
    metadata, state, *rest = args
    state = state.to(state_dtype).uniform_(-0.25, 0.25)
    reference_state = state.clone()
    expected = _reference_prefill_batched(module, x, metadata, reference_state, *rest)
    actual = module._short_conv_dilated_prefill_batched(x, metadata, state, *rest)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
    assert actual.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_graph_replay_changed_inputs_and_initial_states():
    module, x, args = _case(
        "cuda",
        torch.float16,
        lengths=[513, 37],
        decode_tokens=0,
        state_indices=[2, 3],
        has_initial=[True, True],
        hidden_size=257,
        dilation=3,
    )
    metadata, state, *rest = args

    def run():
        return module._short_conv_dilated_prefill_batched(x, metadata, state, *rest)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for initial in (True, False, True):
        metadata.has_initial_states_p.fill_(initial)
        x.normal_(std=0.25)
        state.normal_(std=0.25)
        reference_state = state.clone()
        expected = _reference_prefill_batched(
            module, x, metadata, reference_state, *rest
        )
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_small_m_and_unqualified_formats_retain_fallback():
    x = torch.empty(5, 128, device="cuda", dtype=torch.float16)
    state = x.new_empty(4, 128, 11)
    weight = x.new_empty(128, 4)
    assert (
        prefill_conv_reason(x, state, weight, 1, 3) == "query_rows_below_prefill_band"
    )
    assert (
        prefill_conv_reason(x.float(), state, weight, 1, 3) == "requires_fp16_operands"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_chunk_carry_with_strided_channels():
    from types import SimpleNamespace

    module, first, args = _case(
        "cuda",
        torch.float16,
        lengths=[513, 257],
        decode_tokens=0,
        state_indices=[2, 3],
        has_initial=[True, False],
        hidden_size=129,
        dilation=3,
    )
    metadata, state, *rest = args
    storage = torch.zeros(
        first.shape[0], first.shape[1] * 2, device=first.device, dtype=first.dtype
    )
    storage[:, ::2].copy_(first)
    first = storage[:, ::2]
    reference_state = state.clone()
    expected = _reference_prefill_batched(
        module, first, metadata, reference_state, *rest
    )
    actual = module._short_conv_dilated_prefill_batched(first, metadata, state, *rest)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    second = torch.randn(127 + 389, 129, device=first.device, dtype=first.dtype)
    metadata = SimpleNamespace(
        non_spec_query_start_loc=torch.tensor(
            [0, 127, 516], device=first.device, dtype=torch.int32
        ),
        has_initial_states_p=torch.ones(2, device=first.device, dtype=torch.bool),
        max_prefill_query_len=389,
    )
    rest[-1] = second.shape[0]
    expected = _reference_prefill_batched(
        module, second, metadata, reference_state, *rest
    )
    actual = module._short_conv_dilated_prefill_batched(second, metadata, state, *rest)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(state, reference_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_opaque_prefill_writes_output_without_batch_temporary(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import ple_layer

    module, x, args = _case(
        "cuda",
        torch.float16,
        lengths=[513, 257],
        decode_tokens=0,
        state_indices=[2, 3],
        has_initial=[True, False],
        hidden_size=129,
        dilation=3,
    )
    metadata, state, weight, state_indices, *counts = args
    prefill_metadata = ple_layer.PleShortConvAttentionMetadata.__new__(
        ple_layer.PleShortConvAttentionMetadata
    )
    prefill_metadata.__dict__.update(vars(metadata))
    metadata = prefill_metadata
    metadata.num_prefills = 2
    metadata.num_decodes = metadata.num_decode_tokens = 0
    metadata.num_prefill_tokens = metadata.num_actual_tokens = x.shape[0]
    metadata.spec_sequence_masks = None
    metadata.state_indices_tensor = state_indices
    module.prefix = "ple"
    module.num_spec_tokens = 2
    module.kv_cache = [state]
    module.conv1d = SimpleNamespace(weight=weight.unsqueeze(1))
    context = SimpleNamespace(
        attn_metadata={"ple": metadata}, no_compile_layers={"ple": module}
    )
    monkeypatch.setattr(ple_layer, "get_forward_context", lambda: context)
    monkeypatch.setattr(ple_layer, "is_conv_state_dim_first", lambda: True)
    # The opaque entry may receive a padded buffer; its unused rows remain zero.
    padded = torch.cat((x, torch.randn(7, 129, device=x.device, dtype=x.dtype)))
    output = torch.zeros_like(padded)
    reference_state = state.clone()
    expected = _reference_prefill_batched(
        module, x, metadata, reference_state, weight, state_indices, *counts
    )
    result = module._short_conv(padded, output=output)
    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output[: x.shape[0]], expected, rtol=1e-3, atol=1e-3)
    assert torch.count_nonzero(output[x.shape[0] :]) == 0
    state.copy_(reference_state)
    expected = _reference_prefill_batched(
        module, x, metadata, reference_state, weight, state_indices, *counts
    )
    ple_layer.qwen4_exp_ple_short_conv(padded, output, "ple")
    torch.testing.assert_close(output[: x.shape[0]], expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
