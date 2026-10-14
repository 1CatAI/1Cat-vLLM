# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
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


@pytest.mark.parametrize(
    ("lengths", "hidden_size", "expected"),
    [
        ([100_000], 10240, 2560),  # long single prefills need transient headroom
        ([4095], 10240, None),
        ([4096], 10240, 2560),
        ([6944], 10240, 2560),
        ([5, 1, 7, 3], 16, None),  # narrower than one chunk
        ([512] * 4, 1024, 448),
        ([1024] + [16] * 7, 10240, 704),
    ],
)
def test_prefill_short_conv_channel_chunk_width(lengths, hidden_size, expected) -> None:
    module = _module(conv_state_len=9, dilation=3)
    num_prefills, max_len, tokens = len(lengths), max(lengths), sum(lengths)

    chunk = module._short_conv_prefill_channel_chunk(
        num_prefills, max_len, tokens, hidden_size
    )

    assert chunk == expected
    if chunk is not None:
        # Each padded chunk buffer is at most half of one unpadded step.
        padded = num_prefills * chunk * (module.conv_state_len + max_len)
        assert padded <= (tokens + module.conv_state_len) * hidden_size // 2


# One long prefill next to short ones pads every request to the long one; the
# channel-chunked path must give the same bits and keep its buffers bounded by
# the unpadded step.
SKEWED_CASES = {
    "skewed_with_state": dict(
        lengths=[200, 3, 2, 1],
        decode_tokens=2,
        state_indices=[3, 0, 5, 1],
        has_initial=[True, False, True, True],
        hidden_size=256,
    ),
    "skewed_null_and_empty": dict(
        lengths=[1, 150, 0, 4],
        decode_tokens=0,
        state_indices=[2, 4, 1, NULL_BLOCK_ID],
        has_initial=[True, True, False, True],
        hidden_size=192,
    ),
    "skewed_no_cache_rows": dict(
        lengths=[90, 2],
        decode_tokens=1,
        state_indices=[0, 1],
        has_initial=[False, False],
        hidden_size=320,
        empty_cache=True,
    ),
}


@contextmanager
def _single_threaded_on_cpu(device):
    # ATen's CPU kernels can round an element one ulp differently at the edges
    # of their thread partitions, which move with the tensor shape. Compare the
    # layouts single-threaded; CUDA computes every element the same way.
    if device != "cpu":
        yield
        return
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def _single_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        Qwen4ExpPLELayer,
        "_short_conv_prefill_channel_chunk",
        lambda self, *args: None,
    )


@pytest.mark.parametrize("device,dtype", _devices())
@pytest.mark.parametrize("name", sorted(SKEWED_CASES))
def test_prefill_short_conv_channel_chunks_bitwise(
    name, device, dtype, monkeypatch
) -> None:
    module, x_p, args = _case(device, dtype, **SKEWED_CASES[name])
    metadata, conv_state, *rest = args
    reference_state = conv_state.clone()
    single_pass_state = conv_state.clone()

    chunks = []
    original = Qwen4ExpPLELayer._short_conv_prefill_channel_chunk

    def spy(self, *args):
        chunks.append(original(self, *args))
        return chunks[-1]

    monkeypatch.setattr(Qwen4ExpPLELayer, "_short_conv_prefill_channel_chunk", spy)
    with _single_threaded_on_cpu(device):
        expected = _reference_prefill_batched(
            module, x_p, metadata, reference_state, *rest
        )
        actual = module._short_conv_dilated_prefill_batched(
            x_p, metadata, conv_state, *rest
        )
        _single_pass(monkeypatch)
        single_pass = module._short_conv_dilated_prefill_batched(
            x_p, metadata, single_pass_state, *rest
        )

    assert chunks and chunks[0] is not None and chunks[0] < x_p.shape[1], chunks
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert torch.equal(actual, single_pass)
    assert torch.equal(conv_state, reference_state)
    assert torch.equal(conv_state, single_pass_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefill_short_conv_channel_chunks_bound_memory(monkeypatch) -> None:
    lengths = [1024] + [16] * 7
    module, x_p, args = _case(
        "cuda",
        torch.float16,
        lengths=lengths,
        decode_tokens=0,
        state_indices=list(range(8)),
        has_initial=[True] * 8,
        hidden_size=1024,
        state_slots=8,
    )
    metadata, conv_state, *rest = args
    batch_bytes = x_p.numel() * x_p.element_size()

    def peak(state):
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        base = torch.accelerator.memory_allocated()
        result = module._short_conv_dilated_prefill_batched(x_p, metadata, state, *rest)
        torch.accelerator.synchronize()
        grown = torch.accelerator.max_memory_allocated() - base
        return grown, result

    chunked, actual = peak(conv_state.clone())
    _single_pass(monkeypatch)
    padded, expected = peak(conv_state.clone())
    assert torch.equal(actual, expected)
    # Output plus at most half a step per chunk buffer, plus index scratch.
    assert chunked <= 2.5 * batch_bytes, (chunked, batch_bytes)
    assert padded >= 4 * batch_bytes, (padded, batch_bytes)


@pytest.mark.parametrize("device,dtype", _devices())
@pytest.mark.parametrize("state_mode", ["initial", "fresh", "null", "empty"])
def test_long_single_prefill_channel_chunks_preserve_output_and_state(
    monkeypatch, device, dtype, state_mode
) -> None:
    module, x_p, args = _case(
        device,
        dtype,
        lengths=[4096],
        decode_tokens=2,
        state_indices=[NULL_BLOCK_ID if state_mode == "null" else 3],
        has_initial=[state_mode == "initial"],
        hidden_size=320,
        empty_cache=state_mode == "empty",
    )
    metadata, conv_state, *rest = args
    actual_state = conv_state.clone()
    expected_state = conv_state.clone()
    with _single_threaded_on_cpu(device):
        actual = module._short_conv_dilated_prefill_batched(
            x_p, metadata, actual_state, *rest
        )
        _single_pass(monkeypatch)
        expected = module._short_conv_dilated_prefill_batched(
            x_p, metadata, expected_state, *rest
        )
    assert torch.equal(actual, expected)
    assert torch.equal(actual_state, expected_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_long_single_prefill_channel_chunks_release_device_headroom(monkeypatch):
    module, x_p, args = _case(
        "cuda",
        torch.float16,
        lengths=[6944],
        decode_tokens=0,
        state_indices=[3],
        has_initial=[True],
        hidden_size=10240,
    )
    metadata, conv_state, *rest = args
    batch_bytes = x_p.numel() * x_p.element_size()

    def peak(state):
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        base = torch.accelerator.memory_allocated()
        result = module._short_conv_dilated_prefill_batched(x_p, metadata, state, *rest)
        torch.accelerator.synchronize()
        return torch.accelerator.max_memory_allocated() - base, result

    actual_state, expected_state = conv_state.clone(), conv_state.clone()
    chunked, actual = peak(actual_state)
    _single_pass(monkeypatch)
    single, expected = peak(expected_state)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_state, expected_state)
    assert chunked <= 1.9 * batch_bytes, (chunked, batch_bytes)
    assert single - chunked >= 0.4 * batch_bytes, (single, chunked, batch_bytes)
    print(f"single_prefill_peak_bytes: single={single} chunked={chunked}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", [0, 1, 8, 9, 10, 16, 17, 2049])
@pytest.mark.parametrize("initial", [False, True])
def test_sm70_prefill_convolution_output_and_state(length, initial):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 admission")
    module, x, args = _case(
        "cuda",
        torch.float16,
        lengths=[length, 0, 3],
        decode_tokens=2,
        state_indices=[3, 4, NULL_BLOCK_ID],
        has_initial=[initial, True, True],
        hidden_size=10240,
    )
    metadata, state, *rest = args
    expected_state = state.clone()
    expected = module._short_conv_dilated_prefill_batched(
        x, metadata, expected_state, *rest
    )
    module._prefill_short_conv = True
    actual = module._short_conv_dilated_prefill_batched(x, metadata, state, *rest)
    assert torch.equal(state, expected_state)
    assert bool(torch.isfinite(actual).all())
    # Preserve the operation-aware predeclared FP16 bound. These constructed
    # cases also record the stronger exact equality through assert_close.
    error = (actual.float() - expected.float()).norm(dim=1)
    scale = expected.float().norm(dim=1).clamp_min(1e-6)
    assert (error / scale).max() <= 0.001
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("transpose_state", [False, True])
def test_sm70_prefill_convolution_changed_input_graph_and_continuation(transpose_state):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 admission")
    module, x, args = _case(
        "cuda",
        torch.float16,
        lengths=[7, 18],
        decode_tokens=1,
        state_indices=[4, 2],
        has_initial=[True, False],
        hidden_size=10240,
    )
    metadata, state, *rest = args
    if transpose_state:
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    seed = state.clone()
    module._prefill_short_conv = True
    module._short_conv_dilated_prefill_batched(x, metadata, state, *rest)
    state.copy_(seed)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = module._short_conv_dilated_prefill_batched(x, metadata, state, *rest)
    torch.cuda.current_stream().wait_stream(stream)
    state.copy_(seed)
    x.mul_(-0.75)
    metadata.has_initial_states_p.fill_(True)
    rest[1].copy_(rest[1].flip(0))
    reference_state = state.clone()
    module._prefill_short_conv = False
    for _ in range(3):
        expected = module._short_conv_dilated_prefill_batched(
            x, metadata, reference_state, *rest
        )
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(state, reference_state)
