# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only orchestration tests; no vLLM binary, Triton or CUDA initialization.

Load the actual Python definitions from source, replacing only GPU arithmetic
and imports. The CPU top-k/log-softmax oracle is shared with the legacy loop.
This proves orchestration equality for fixed logits, not GPU GEMM/kernel parity.

Run with --confcutdir=tests/v1/worker to avoid GPU-oriented root conftest imports.
"""

import ast
import copy
import time
import weakref
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / "vllm/v1/worker/gpu/sample/prompt_logprob.py"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module([future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def cpu_topk(logits, k, targets):
    ids = targets.long().unsqueeze(1)
    if k:
        ids = torch.cat((ids, torch.topk(logits, k, dim=-1).indices), dim=1)
    values = torch.log_softmax(logits.float(), dim=-1).gather(1, ids)
    ranks = (logits >= logits.gather(1, targets.long()[:, None])).sum(1)
    return SimpleNamespace(
        logprob_token_ids=ids, logprobs=values, selected_token_ranks=ranks
    )


@pytest.fixture
def runtime():
    ns = dict(np=np, torch=torch, NamedTuple=NamedTuple, Callable=Callable)
    load_definitions(ROOT / "vllm/v1/outputs.py", {"LogprobsTensors"}, ns)
    ns.update(
        envs=SimpleNamespace(VLLM_PROMPT_LOGPROBS_CHUNK_SIZE=32),
        compute_topk_logprobs=cpu_topk,
    )
    load_definitions(
        PROMPT, {"compute_prompt_logprobs_with_chunking", "PromptLogprobsWorker"}, ns
    )
    return ns


def legacy_loop(targets, hidden, logits_fn, k):
    # The main@4ca7c9d9a1 loop, with the same CPU arithmetic oracle as the SUT.
    outputs = []
    for start in range(0, len(targets), 1024):
        logits = logits_fn(hidden[start : start + 1024])
        result = cpu_topk(
            logits, logits.shape[1] if k == -1 else k, targets[start : start + 1024]
        )
        outputs.append(
            (result.logprob_token_ids, result.logprobs, result.selected_token_ranks)
        )
    return tuple(torch.cat(parts) for parts in zip(*outputs))


def assert_bits(actual, expected):
    for a, b in zip(actual, expected):
        assert a.device.type == "cpu"
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


@pytest.mark.parametrize("chunk_size", [1, 7, 32, 64, 1024])
@pytest.mark.parametrize("k", [0, 1, 20, 128, 256, 257, -1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunked_results_match_legacy_bits(runtime, chunk_size, k, dtype):
    # Non-contiguous hidden states, uneven tails, repeated and -inf logits.
    logits = torch.randn(67, 514, generator=torch.Generator().manual_seed(7))[
        :, ::2
    ].to(dtype)
    logits[:, -1] = -torch.inf
    logits[:, 1] = logits[:, 0]
    targets = torch.arange(67, dtype=torch.int32) % 257
    expected = legacy_loop(targets, logits, lambda x: x, k)
    actual = runtime["compute_prompt_logprobs_with_chunking"](
        targets, logits, lambda x: x, k, chunk_size=chunk_size
    )
    assert_bits(actual, expected)


@pytest.mark.parametrize("k", [0, 1, 20, -1])
def test_1024_boundary_and_lifetimes(runtime, k):
    targets = torch.arange(2053) % 31
    hidden = torch.arange(2053)[:, None]
    table = torch.randn(2053, 31, generator=torch.Generator().manual_seed(3))
    previous: weakref.ref[torch.Tensor] | None = None
    sizes = []

    def logits_fn(rows):
        nonlocal previous
        assert previous is None or previous() is None, "previous logits still live"
        result = table[rows[:, 0]].clone()
        previous = weakref.ref(result)
        sizes.append(len(rows))
        return result

    actual = runtime["compute_prompt_logprobs_with_chunking"](
        targets, hidden, logits_fn, k
    )
    assert max(sizes) == 32 and sizes[-1] == 5
    assert previous is not None and previous() is None
    assert_bits(actual, legacy_loop(targets, table, lambda x: x, k))


@pytest.mark.parametrize("k", [0, 20, -1])
def test_production_vocab(runtime, k):
    logits = torch.arange(248320, dtype=torch.float32)[None, :].repeat(2, 1) / 256
    targets = torch.tensor([0, 248319], dtype=torch.int32)
    assert_bits(
        runtime["compute_prompt_logprobs_with_chunking"](
            targets, logits, lambda x: x, k, chunk_size=1
        ),
        legacy_loop(targets, logits, lambda x: x, k),
    )


@pytest.mark.parametrize("chunk_size", [0, -1, 1025])
def test_invalid_chunk_size(runtime, chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        runtime["compute_prompt_logprobs_with_chunking"](
            torch.tensor([1]),
            torch.ones(1, 3),
            lambda _: pytest.fail(),
            0,
            chunk_size=chunk_size,
        )


def make_batch(req_ids, lengths):
    cu = np.array([0, *np.cumsum(lengths)], dtype=np.int32)
    return SimpleNamespace(
        req_ids=req_ids,
        idx_mapping_np=np.arange(len(lengths)),
        num_tokens=sum(lengths),
        num_scheduled_tokens=np.array(lengths),
        query_start_loc_np=cu,
        query_start_loc=None,
        idx_mapping=None,
    )


def worker_step(
    runtime, worker, batch, logits, targets, prompt_lens, computed=None, prefill=None
):
    runtime["get_prompt_logprobs_token_ids"] = lambda *args: targets
    return worker.compute_prompt_logprobs(
        lambda x: x,
        logits,
        batch,
        None,
        None,
        np.array(prompt_lens),
        np.array(prefill if prefill is not None else prompt_lens),
        np.array(computed if computed is not None else [0] * len(prompt_lens)),
        None,
    )


def test_mixed_full_small_decode_requests_use_own_width(runtime):
    worker = runtime["PromptLogprobsWorker"](3)
    for i, k in enumerate([-1, 1, None]):
        worker.add_request(str(i), i, SimpleNamespace(prompt_logprobs=k))
    batch = make_batch(["0", "1", "2"], [3, 35, 1])
    logits = torch.arange(257, dtype=torch.float32)[None, :].repeat(39, 1)
    targets = torch.arange(39) % 257
    sizes = []

    def tracked(x, k, ids):
        sizes.append((len(x), k))
        return cpu_topk(x, k, ids)

    runtime["compute_topk_logprobs"] = tracked
    result = worker_step(runtime, worker, batch, logits, targets, [3, 35, 1], [0, 0, 1])
    assert set(result) == {"0", "1"}
    assert sizes == [(2, 257), (32, 1), (2, 1)]
    assert_bits(result["0"][:3], legacy_loop(targets[:2], logits[:2], lambda x: x, -1))
    # No ties here: per-request widths also match the legacy mixed-batch slice.
    reference = legacy_loop(targets, logits, lambda x: x, -1)
    assert_bits(
        result["1"][:3],
        (reference[0][3:37, :2], reference[1][3:37, :2], reference[2][3:37]),
    )
    assert result["0"].to_cpu_nonblocking() is result["0"]


def test_prefill_chunks_merge_on_cpu_and_skip_preempted_requests(runtime):
    worker = runtime["PromptLogprobsWorker"](1)
    worker.add_request("r", 0, SimpleNamespace(prompt_logprobs=2))
    logits = torch.randn(7, 31, generator=torch.Generator().manual_seed(6))
    targets = torch.arange(7)
    first = worker_step(
        runtime, worker, make_batch(["r"], [3]), logits[:3], targets[:3], [7]
    )
    assert first == {}
    saved = worker.in_progress_prompt_logprobs["r"][0]
    assert saved.logprobs.device.type == "cpu"
    result = worker_step(
        runtime, worker, make_batch(["r"], [4]), logits[3:], targets[3:], [7], [3]
    )
    assert_bits(result["r"][:3], legacy_loop(targets[:6], logits[:6], lambda x: x, 2))
    assert worker.in_progress_prompt_logprobs["r"] == []
    resumed = worker_step(
        runtime, worker, make_batch(["r"], [1]), logits[:1], targets[:1], [7], [0], [8]
    )
    assert resumed == {}
    worker.remove_request("r")
    assert "r" not in worker.in_progress_prompt_logprobs


def test_no_prompt_requests_return_before_any_work(runtime):
    worker = runtime["PromptLogprobsWorker"](1)
    worker.add_request("r", 0, SimpleNamespace(prompt_logprobs=None))
    result = worker.compute_prompt_logprobs(
        lambda _: pytest.fail("unexpected logits computation"),
        None,
        make_batch(["r"], [1]),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert result == {}


def test_last_single_prompt_row_produces_no_logits(runtime):
    worker = runtime["PromptLogprobsWorker"](1)
    worker.add_request("r", 0, SimpleNamespace(prompt_logprobs=0))
    runtime["compute_topk_logprobs"] = lambda *args: pytest.fail()
    assert (
        worker_step(
            runtime,
            worker,
            make_batch(["r"], [1]),
            torch.ones(1, 3),
            torch.tensor([0]),
            [1],
        )
        == {}
    )


@pytest.fixture
def limits():
    ns: dict[str, Any] = {}
    load_definitions(ROOT / "vllm/exceptions.py", {"VLLMValidationError"}, ns)
    load_definitions(
        ROOT / "vllm/v1/prompt_logprobs_limits.py",
        {"validate_prompt_logprobs_output"},
        ns,
    )
    return ns


@pytest.mark.parametrize("k", [-1, 248319, 248320])
def test_full_vocab_admission_boundary(limits, k):
    validate = limits["validate_prompt_logprobs_output"]
    width = (248320 if k == -1 else k) + 1
    limit = 64 * 1024 * 1024
    max_rows = limit // (12 * width + 8)
    assert max_rows == 22
    validate(max_rows + 1, k, 248320, limit)
    for length in [max_rows + 2, 8192, 32768, 131072]:
        with pytest.raises(limits["VLLMValidationError"], match="per-request limit"):
            validate(length, k, 248320, limit)


@pytest.mark.parametrize("k", [0, 1, 20])
def test_long_small_topk_admitted(limits, k):
    limits["validate_prompt_logprobs_output"](131072, k, 248320, 64 * 1024 * 1024)


def test_output_cap_and_partial_topk_guard(limits):
    validate = limits["validate_prompt_logprobs_output"]
    validate(2, 0, 257, 20)
    with pytest.raises(limits["VLLMValidationError"]):
        validate(3, 0, 257, 20)
    with pytest.raises(limits["VLLMValidationError"], match="up to 128"):
        validate(2, 129, 248320, 64 * 1024 * 1024)
    with pytest.raises(ValueError, match="positive"):
        validate(2, 0, 257, 0)


@pytest.mark.parametrize(
    "count,length,should_reject",
    [(None, 8192, False), (20, 8192, False), (-1, 8192, True), (-1, 23, False)],
)
def test_input_processor_rejects_before_engine_request(
    limits, count, length, should_reject
):
    class Params:
        prompt_logprobs = count
        max_tokens = 1

        def clone(self):
            return copy.copy(self)

        def update_from_generation_config(self, *args):
            pass

    ns = dict(
        limits,
        time=time,
        SamplingParams=Params,
        current_platform=SimpleNamespace(validate_request=lambda *a: None),
        split_enc_dec_input=lambda x: (None, x),
        length_from_prompt_token_ids_or_embeds=lambda ids, embeds: len(ids),
        envs=SimpleNamespace(VLLM_PROMPT_LOGPROBS_MAX_OUTPUT_BYTES=64 * 1024 * 1024),
    )
    tree = ast.parse((ROOT / "vllm/v1/engine/input_processor.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "InputProcessor"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_inputs"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([future, method], type_ignores=[])),
            "input_processor.py",
            "exec",
        ),
        ns,
    )
    created: list[dict[str, Any]] = []

    def fake_engine_core_request(**kwargs: Any) -> dict[str, Any]:
        created.append(kwargs)
        return kwargs

    ns["EngineCoreRequest"] = fake_engine_core_request
    processor = SimpleNamespace(
        _validate_params=lambda *a: None,
        _validate_lora=lambda *a: None,
        _validate_model_inputs=lambda *a: None,
        use_v2_model_runner=True,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_size_local=1,
                local_engines_only=True,
            )
        ),
        model_config=SimpleNamespace(get_vocab_size=lambda: 248320),
        generation_config_fields={},
        tokenizer=None,
        renderer=SimpleNamespace(get_eos_token_id=lambda: 0),
    )
    if count is None:
        ns["validate_prompt_logprobs_output"] = lambda *a: pytest.fail(
            "ordinary path changed"
        )
    prompt = {"type": "token", "prompt_token_ids": [0] * length}
    if should_reject:
        with pytest.raises(limits["VLLMValidationError"]):
            ns["process_inputs"](processor, "r", prompt, Params(), ("generate",))
        assert created == []
    else:
        ns["process_inputs"](processor, "r", prompt, Params(), ("generate",))
        assert len(created) == 1
