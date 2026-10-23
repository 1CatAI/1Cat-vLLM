# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.forward_context import ForwardContext
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.prefill_overlap import (
    PrefillOverlap,
    token_range_batch,
    token_range_model_inputs,
)
from vllm.v1.worker.ubatching import (
    dbo_overlaps_collectives,
    dbo_run_collective_overlapped,
)


def _chunk_batch() -> InputBatch:
    """Two requests of six new tokens each after 4090 computed tokens."""
    buffers = InputBuffers(max_num_reqs=4, max_num_tokens=16, device="cpu")
    batch = InputBatch.make_dummy(2, 12, buffers)
    batch.seq_lens.fill_(4096)
    batch.seq_lens_cpu_upper_bound = torch.full((2,), 4096, dtype=torch.int32)
    batch.num_computed_tokens_np = np.full(2, 4090, dtype=np.int32)
    batch.input_ids.copy_(torch.arange(12))
    batch.positions.copy_(torch.arange(12) + 4090)
    return batch


def test_token_range_cuts_the_request_that_spans_the_split():
    batch = _chunk_batch()

    first = token_range_batch(batch, 0, 9)
    assert first.req_ids == batch.req_ids
    assert first.num_tokens == first.num_tokens_after_padding == 9
    assert first.num_scheduled_tokens.tolist() == [6, 3]
    assert first.query_start_loc.tolist() == first.query_start_loc_np.tolist()
    assert first.query_start_loc_np.tolist() == [0, 6, 9]
    # The cut request ends three tokens early and continues in the next range.
    assert first.seq_lens.tolist() == [4096, 4093]
    assert first.seq_lens_cpu_upper_bound.tolist() == [4096, 4093]
    assert first.num_computed_tokens_np.tolist() == [4090, 4090]
    assert first.is_incomplete_prefilling_np.tolist() == [False, True]
    assert first.input_ids.tolist() == list(range(9))

    second = token_range_batch(batch, 9, 12)
    assert second.req_ids == batch.req_ids[1:]
    assert second.idx_mapping.tolist() == [1]
    assert second.num_scheduled_tokens.tolist() == [3]
    assert second.query_start_loc_np.tolist() == [0, 3]
    assert second.seq_lens.tolist() == [4096]
    assert second.num_computed_tokens_np.tolist() == [4093]
    assert second.positions.tolist() == [4099, 4100, 4101]
    # Neither range samples; the whole batch does.
    assert first.logits_indices.numel() == second.logits_indices.numel() == 0
    assert second.cu_num_logits.tolist() == [0, 0]


def test_token_range_on_a_request_boundary_keeps_requests_whole():
    batch = _chunk_batch()
    first = token_range_batch(batch, 0, 6)
    second = token_range_batch(batch, 6, 12)
    assert first.req_ids == batch.req_ids[:1]
    assert second.req_ids == batch.req_ids[1:]
    assert first.seq_lens.tolist() == second.seq_lens.tolist() == [4096]
    assert first.num_computed_tokens_np.tolist() == [4090]
    assert second.num_computed_tokens_np.tolist() == [4090]
    assert not first.is_incomplete_prefilling_np.any()


def test_model_inputs_split_along_tokens_and_mrope_positions():
    inputs = {
        "input_ids": torch.arange(12),
        "positions": torch.arange(36).view(3, 12),
        "inputs_embeds": torch.arange(24).view(12, 2),
    }
    half = token_range_model_inputs(inputs, 9, 12)
    assert half["input_ids"].tolist() == [9, 10, 11]
    assert half["positions"].tolist() == [[9, 10, 11], [21, 22, 23], [33, 34, 35]]
    assert half["inputs_embeds"].tolist() == [[18, 19], [20, 21], [22, 23]]
    with pytest.raises(NotImplementedError):
        token_range_model_inputs({"encoder_outputs": []}, 0, 1)


def _align_split(query_start_loc, num_computed, split, block_size=800):
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_spec = SimpleNamespace(block_size=block_size)
    query_start_loc = np.array(query_start_loc, dtype=np.int32)
    batch = SimpleNamespace(
        query_start_loc_np=query_start_loc,
        num_scheduled_tokens=np.diff(query_start_loc),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
    )
    return state.align_ubatch_split(batch, split)


@pytest.mark.parametrize(
    ("num_computed", "num_new", "expected"),
    [
        (0, 1600, 801),  # first chunk of two state blocks
        (1600, 1600, 801),  # aligned middle chunk
        (148800, 1233, 801),  # unaligned last chunk of a prompt
        (0, 600, 300),  # the whole chunk lies in one state block
    ],
)
def test_align_mode_split_keeps_the_last_state_block(num_computed, num_new, expected):
    split = _align_split([0, num_new], [num_computed], num_new // 2)
    assert split == expected
    seq_end = num_computed + num_new
    assert (num_computed + split - 1) // 800 == (seq_end - 1) // 800


def test_align_mode_split_on_a_request_boundary_stays():
    assert _align_split([0, 800, 1600], [0, 0], 800) == 800


def _overlap_config(dcp: int = 1, encoder_decoder: bool = False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=dcp,
        ),
        model_config=SimpleNamespace(is_encoder_decoder=encoder_decoder),
        lora_config=None,
        kv_transfer_config=None,
        kernel_config=SimpleNamespace(tp_prefill_overlap_min_tokens=8),
    )


@pytest.mark.parametrize(
    "config",
    [_overlap_config(dcp=2), _overlap_config(encoder_decoder=True)],
    ids=["decode_context_parallel", "encoder_decoder"],
)
def test_overlap_rejects_configurations_it_cannot_split(config):
    with pytest.raises(ValueError, match="tp_prefill_overlap_min_tokens excludes"):
        PrefillOverlap(config, torch.device("cpu"))


def test_overlap_rejects_moe_and_deepstack_models():
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    def model_of(*modules):
        return SimpleNamespace(modules=lambda: iter(modules))

    with pytest.raises(ValueError, match="dense models only"):
        PrefillOverlap.check_model(model_of(object.__new__(FusedMoE)))
    with pytest.raises(ValueError, match="deepstack"):
        PrefillOverlap.check_model(model_of(SimpleNamespace(deepstack_num_level=3)))
    with pytest.raises(ValueError, match="deepstack"):
        PrefillOverlap.check_model(model_of(SimpleNamespace(_ds_layer_indices=[0, 1])))
    # A vision tower without deepstack levels (Qwen3.5) splits like text.
    PrefillOverlap.check_model(model_of(SimpleNamespace(deepstack_num_level=0)))


class _ToyModel(torch.nn.Module):
    """Elementwise layers with a slow stand-in all-reduce between them."""

    def __init__(self, num_layers: int, fail_tokens: int | None = None):
        super().__init__()
        self.num_layers = num_layers
        self.fail_tokens = fail_tokens

    @staticmethod
    def _all_reduce(x: torch.Tensor) -> torch.Tensor:
        torch.cuda._sleep(2_000_000)
        return x * 2

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = input_ids.float()
        for layer in range(self.num_layers):
            x = x * 1.5 + layer
            if layer == 2 and input_ids.shape[0] == self.fail_tokens:
                raise ValueError("half failed")
            if dbo_overlaps_collectives():
                x = dbo_run_collective_overlapped(self._all_reduce, x)
            else:
                x = self._all_reduce(x)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_halves_overlapping_collectives_match_the_whole_batch():
    device = torch.device("cuda", torch.accelerator.current_device_index())
    overlap = PrefillOverlap(_overlap_config(), device)
    model = _ToyModel(num_layers=6)
    input_ids = torch.arange(4096, device=device) % 97
    with torch.inference_mode():
        expected = model(input_ids=input_ids)
    contexts = [
        ForwardContext(
            no_compile_layers={}, attn_metadata={}, slot_mapping={}, ubatch_slices=None
        )
        for _ in range(2)
    ]
    for _ in range(3):
        halves = [{"input_ids": input_ids[:2049]}, {"input_ids": input_ids[2049:]}]
        output = overlap.run(model, halves, contexts)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert not dbo_overlaps_collectives()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fail_tokens", [2049, 2047], ids=["first", "second"])
def test_a_failing_half_stops_both_and_reports_its_error(fail_tokens):
    device = torch.device("cuda", torch.accelerator.current_device_index())
    overlap = PrefillOverlap(_overlap_config(), device)
    model = _ToyModel(num_layers=6, fail_tokens=fail_tokens)
    input_ids = torch.arange(4096, device=device) % 97
    contexts = [
        ForwardContext(
            no_compile_layers={}, attn_metadata={}, slot_mapping={}, ubatch_slices=None
        )
        for _ in range(2)
    ]
    halves = [{"input_ids": input_ids[:2049]}, {"input_ids": input_ids[2049:]}]
    # Without the shared abort the other half waited forever for the failed one.
    with pytest.raises(ValueError, match="half failed"):
        overlap.run(model, halves, contexts)
    assert not dbo_overlaps_collectives()
    # The overlap keeps working after a failed batch.
    healthy = _ToyModel(num_layers=6)
    with torch.inference_mode():
        expected = healthy(input_ids=input_ids)
    torch.testing.assert_close(overlap.run(healthy, halves, contexts), expected)
