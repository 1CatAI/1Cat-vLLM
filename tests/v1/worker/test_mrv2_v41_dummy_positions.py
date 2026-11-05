# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2 capture inputs vs the V4.1 Engram hash.

On 8x V100 PP2 (2026-11-03) with MRV2 on, every first-stage worker
died in the first capture warmup (cudagraph_utils.capture ->
forward_fn(CUDAGraphMode.NONE) -> model.py forward -> hash_state) with
"Triton Error [CUDA]: unspecified launch failure" at _hash_rows' module load.
The fault was the previous launch, _hash_request, whose ``_fail_if`` (a PTX
``trap``) requires one contiguous token chain: positions[t] == positions[0]
+ t. MRV2's ``InputBatch.make_dummy`` zeroes the positions, and capture
passes that buffer to the model; the V1 runner's ``_dummy_run`` writes
``arange(n) % max_model_len`` first (gpu_model_runner.py). The capture sizes
were 1..6, so every size from 2 up traps.

These tests run the real ``InputBatch.make_dummy`` on real ``InputBuffers``
and the real ``V41EngramHashState`` Triton kernels (the Triton interpreter on
CPU, where ``_fail_if``'s inline asm raises instead of trapping), with the
Serve shapes: max_num_batched_tokens 8192, max_model_len 524288, vocabulary
129280, a PAD_SLOT_ID slot mapping (the dummy ``valid`` mask is all False).
"""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.deepseek_v41_engram import V41EngramHashState
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import DummyPositions, InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

KERNELS = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"
needs_kernels = pytest.mark.skipif(
    not KERNELS, reason="Triton kernels need CUDA or TRITON_INTERPRET=1"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The PP2 serve run: non-default args and capture sizes.
MAX_NUM_TOKENS = 8192
MAX_MODEL_LEN = 524288
VOCAB = 129280
CAPTURE_SIZES = (6, 5, 4, 3, 2, 1)
V41 = "deepseek_v41_text"


def _runner(model_type=V41, *, max_num_tokens=MAX_NUM_TOKENS):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        ),
        max_num_tokens=max_num_tokens,
        max_model_len=MAX_MODEL_LEN,
    )


def _runner_buffers(model_type=V41):
    # The runner's own construction (model_runner.py __init__).
    runner = _runner(model_type)
    return InputBuffers(
        max_num_reqs=1,
        max_num_tokens=runner.max_num_tokens,
        device=DEVICE,
        dummy_positions=GPUModelRunner._dummy_positions(runner),
    )


def _hash_state(monkeypatch):
    # model.py prepare_postload's arguments. The constructor insists on CUDA
    # tables; on CPU the Triton interpreter runs the same kernels.
    if DEVICE.type == "cpu":
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    generator = torch.Generator().manual_seed(1103)
    token_map = torch.randint(0, 2**20, (VOCAB,), generator=generator)
    primes = torch.randint(1000, 100000, (2, 24), generator=generator)
    offsets = torch.randint(0, 2**20, (2, 24), generator=generator)
    multipliers = torch.randint(1, 2**30, (2, 4), generator=generator)
    state = V41EngramHashState(
        *(t.to(DEVICE) for t in (token_map, primes, offsets, multipliers)),
        pad_token_id=2,
        max_tokens=MAX_NUM_TOKENS,
        max_context=max(16384, MAX_MODEL_LEN),
    )
    monkeypatch.undo()
    return state


def _capture_inputs(buffers, num_tokens):
    # cudagraph_utils.ModelCudaGraphManager.capture: model_inputs holds views
    # of the persistent buffers, then prepare_inputs_to_capture builds the
    # dummy batch (make_dummy) before the warmup forward.
    model_inputs = {
        "input_ids": buffers.input_ids[:num_tokens],
        "positions": buffers.positions[:num_tokens],
    }
    batch = InputBatch.make_dummy(1, num_tokens, buffers)
    # model.py forward: valid = slot_mapping[:tokens] >= 0 (PAD_SLOT_ID).
    slots = torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.int64, device=DEVICE)
    return model_inputs, batch, slots >= 0


def test_runner_chooses_a_token_chain_for_v41_only():
    assert GPUModelRunner._dummy_positions(_runner()) is DummyPositions.CHAIN
    assert GPUModelRunner._dummy_positions(_runner("qwen3")) is DummyPositions.ZEROS


def test_chain_refuses_a_batch_longer_than_the_model():
    # V1 wraps positions at max_model_len; a wrapped chain is not one chain
    # and would trap the hash, so the runner refuses it at construction.
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        GPUModelRunner._dummy_positions(_runner(max_num_tokens=MAX_MODEL_LEN + 1))


@pytest.mark.parametrize("num_tokens", CAPTURE_SIZES)
def test_v41_capture_and_dummy_positions_are_one_chain(num_tokens):
    buffers = _runner_buffers()
    buffers.positions.fill_(77)  # stale values from an earlier step
    model_inputs, batch, _ = _capture_inputs(buffers, num_tokens)
    want = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    # V1 _dummy_run: arange(n) % max_model_len (n <= max_model_len here).
    assert torch.equal(model_inputs["positions"], want)
    assert torch.equal(batch.positions, want)
    # The graph must read the persistent buffer that real steps write.
    assert batch.positions.data_ptr() == buffers.positions.data_ptr()
    assert model_inputs["positions"].data_ptr() == buffers.positions.data_ptr()
    assert torch.equal(batch.input_ids, torch.zeros_like(batch.input_ids))


def test_other_models_keep_upstream_zero_positions():
    buffers = _runner_buffers("qwen3")
    buffers.positions.fill_(77)
    batch = InputBatch.make_dummy(1, 6, buffers)
    assert torch.equal(
        batch.positions, torch.zeros(6, dtype=torch.int64, device=DEVICE)
    )


@needs_kernels
@pytest.mark.parametrize("num_tokens", CAPTURE_SIZES)
def test_v41_capture_warmup_passes_the_engram_hash(monkeypatch, num_tokens):
    # The boot failure, through the real capture inputs and hash kernels.
    hash_state = _hash_state(monkeypatch)
    buffers = _runner_buffers()
    model_inputs, _, valid = _capture_inputs(buffers, num_tokens)
    hashes = hash_state(model_inputs["input_ids"], model_inputs["positions"], valid)
    assert hashes.shape == (num_tokens, 2, 24)
    # The next request starts at position 0, as after a V1 capture.
    hash_state(
        torch.tensor([5, 6, 7], dtype=torch.int32, device=DEVICE),
        torch.arange(3, device=DEVICE),
    )


@needs_kernels
@pytest.mark.parametrize("num_tokens", [n for n in CAPTURE_SIZES if n > 1])
def test_upstream_zero_positions_trap_the_engram_hash(monkeypatch, num_tokens):
    # The failing input itself: zeroed dummy positions break the token chain and
    # _hash_request's _fail_if fires (a trap on CUDA; the interpreter raises
    # on the inline asm instead).
    hash_state = _hash_state(monkeypatch)
    buffers = InputBuffers(max_num_reqs=1, max_num_tokens=MAX_NUM_TOKENS, device=DEVICE)
    model_inputs, _, valid = _capture_inputs(buffers, num_tokens)
    assert torch.equal(
        model_inputs["positions"], torch.zeros_like(model_inputs["positions"])
    )
    if DEVICE.type == "cuda":
        pytest.skip("a device trap poisons the CUDA context; CPU interpreter only")
    with pytest.raises(Exception, match="inline_asm"):
        hash_state(model_inputs["input_ids"], model_inputs["positions"], valid)


def test_prefill_skip_without_an_mrv2_hook_fails(monkeypatch):
    # An enabled deployed model must expose its binding; rows=0 needs none.
    from tests.models.test_deepseek_v41_pp_commit import _stage_model
    from tests.v1.worker.test_mrv2_v41 import _LanguageModel, _Log, _model_state
    from vllm.v1.worker.gpu.model_states import deepseek_v41 as v41_state

    log = _Log()
    monkeypatch.setattr(v41_state, "logger", log)
    stage = _stage_model(20, 40)
    _model_state(monkeypatch, _LanguageModel(stage))
    assert log.warnings() == []
    stage._v41_skip_eligible, stage._v41_skip_rows = True, 3072
    with pytest.raises(ValueError, match="needs prefill_skip_bind"):
        _model_state(monkeypatch, _LanguageModel(stage))
