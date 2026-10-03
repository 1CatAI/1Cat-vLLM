# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _module(conv_state_len: int, dilation: int) -> Qwen4ExpPLELayer:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = conv_state_len
    module.short_conv_dilation = dilation
    return module


def _reference_prefill_batched(
    module,
    x_p,
    metadata,
    conv_state,
    conv_weights,
    state_indices_tensor_p,
    num_prefills,
    num_decode_tokens,
    num_prefill_tokens,
):
    """The padded-copy implementation this path replaced, kept as the oracle."""
    query_start_loc_p = (
        metadata.non_spec_query_start_loc[-num_prefills - 1 :] - num_decode_tokens
    )
    has_initial_states_p = metadata.has_initial_states_p
    output = torch.empty_like(x_p)
    q_starts = query_start_loc_p.to(torch.int64)
    lengths = q_starts[1:] - q_starts[:-1]
    max_len = metadata.max_prefill_query_len
    hidden_size = x_p.shape[1]
    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
    packed_tokens[req_indices, col_indices] = x_p
    packed_tokens = packed_tokens.transpose(1, 2).contiguous()

    state_indices = state_indices_tensor_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    has_initial = has_initial_states_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.bool
    )
    if module.conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros(
                (num_prefills, hidden_size, module.conv_state_len), dtype=x_p.dtype
            )
        else:
            state = conv_state.index_select(0, state_indices)[
                ..., : module.conv_state_len
            ].to(x_p.dtype)
        use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
        initial_state = torch.where(use_initial_mask, state, torch.zeros_like(state))
        history = torch.cat((initial_state, packed_tokens), dim=-1)
    else:
        history = packed_tokens

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=module.short_conv_dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()
    token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
        num_prefills, 1
    )
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
    output.copy_(conv_output[req_indices, col_indices])

    if module.conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
            num_prefills, 1, 1
        )
        state_offsets = torch.arange(
            module.conv_state_len, device=history.device, dtype=torch.int64
        ).view(1, 1, module.conv_state_len)
        next_state = history.gather(
            dim=2,
            index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
        )
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., : module.conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(
            update_mask.view(num_prefills, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        existing_state[..., : module.conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)
    return output


def _case(
    device,
    dtype,
    *,
    lengths,
    decode_tokens,
    state_indices,
    has_initial,
    kernel_size=4,
    dilation=3,
    hidden_size=16,
    state_slots=6,
    empty_cache=False,
    seed=0,
):
    generator = torch.Generator().manual_seed(seed)
    conv_state_len = (kernel_size - 1) * dilation
    num_prefills = len(lengths)
    num_prefill_tokens = sum(lengths)
    decode_starts = list(range(decode_tokens + 1))
    prefill_starts = [decode_tokens]
    for length in lengths:
        prefill_starts.append(prefill_starts[-1] + length)
    metadata = SimpleNamespace(
        non_spec_query_start_loc=torch.tensor(
            decode_starts[:-1] + prefill_starts, dtype=torch.int32, device=device
        ),
        has_initial_states_p=torch.tensor(has_initial, device=device),
        max_prefill_query_len=max(lengths),
    )
    x_p = torch.randn(num_prefill_tokens, hidden_size, generator=generator).to(
        device=device, dtype=dtype
    )
    conv_weights = torch.randn(hidden_size, kernel_size, generator=generator).to(
        device=device, dtype=dtype
    )
    rows = 0 if empty_cache else state_slots
    conv_state = torch.randn(
        rows, hidden_size, conv_state_len + 2, generator=generator
    ).to(device=device, dtype=dtype)
    args = (
        metadata,
        conv_state,
        conv_weights,
        torch.tensor(state_indices, dtype=torch.int32, device=device),
        num_prefills,
        decode_tokens,
        num_prefill_tokens,
    )
    return _module(conv_state_len, dilation), x_p, args


CASES = {
    "mixed_lengths": dict(
        lengths=[5, 1, 7, 3],
        decode_tokens=2,
        state_indices=[3, 0, 5, 1],
        has_initial=[True, False, True, True],
    ),
    "null_state_and_empty_row": dict(
        lengths=[4, 0, 6],
        decode_tokens=0,
        state_indices=[2, 4, NULL_BLOCK_ID],
        has_initial=[True, True, True],
    ),
    "no_cache_rows": dict(
        lengths=[3, 8],
        decode_tokens=1,
        state_indices=[0, 1],
        has_initial=[False, False],
        empty_cache=True,
    ),
    "no_state_window": dict(
        lengths=[2, 5],
        decode_tokens=0,
        state_indices=[1, 0],
        has_initial=[True, True],
        kernel_size=1,
    ),
}


def _devices():
    devices = [("cpu", torch.float32)]
    if torch.cuda.is_available():
        devices.append(("cuda", torch.float16))
    return devices


@pytest.mark.parametrize("device,dtype", _devices())
@pytest.mark.parametrize("name", sorted(CASES))
def test_prefill_short_conv_matches_padded_copy_bitwise(name, device, dtype) -> None:
    module, x_p, args = _case(device, dtype, **CASES[name])
    metadata, conv_state, *rest = args
    reference_state = conv_state.clone()

    expected = _reference_prefill_batched(module, x_p, metadata, reference_state, *rest)
    actual = module._short_conv_dilated_prefill_batched(
        x_p, metadata, conv_state, *rest
    )

    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert torch.equal(conv_state, reference_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefill_short_conv_holds_two_batch_sized_buffers() -> None:
    lengths = [512] * 4
    module, x_p, args = _case(
        "cuda",
        torch.float16,
        lengths=lengths,
        decode_tokens=0,
        state_indices=[0, 1, 2, 3],
        has_initial=[True] * 4,
        hidden_size=1024,
    )
    metadata, conv_state, *rest = args
    batch_bytes = x_p.numel() * x_p.element_size()

    def peak(fn, state):
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        base = torch.accelerator.memory_allocated()
        result = fn(x_p, metadata, state, *rest)
        torch.accelerator.synchronize()
        grown = torch.accelerator.max_memory_allocated() - base
        del result
        return grown

    reference = peak(
        lambda *a: _reference_prefill_batched(module, *a), conv_state.clone()
    )
    optimized = peak(module._short_conv_dilated_prefill_batched, conv_state.clone())
    # History, the convolution output and the gathered result: never more than
    # two of them alive at once, plus index scratch.
    assert optimized <= 2.25 * batch_bytes, (optimized, batch_bytes)
    assert reference - optimized >= 2 * batch_bytes, (reference, optimized)
