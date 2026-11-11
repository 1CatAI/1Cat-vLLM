# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check TP-local top-1 merging, greedy rejection and changing inputs."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from vllm.models.deepseek_v41 import native
from vllm.models.deepseek_v41.greedy_head import (
    _global_top1,
    _local_pair,
    greedy_top_tokens,
    verify_top_tokens,
)
from vllm.triton_utils import triton
from vllm.v1.sample.ops.fast_argmax import _argmax_parts
from vllm.v1.worker import gpu_model_runner as runner_module

torch.manual_seed(618)
cases = 0
for rows in range(1, 7):
    for columns in (7, 4097, 32320):
        for trial in range(3):
            full = torch.randn(rows, 4 * columns, device="cuda").half()
            if trial == 1:
                full[:, 1] = 20
                full[:, columns + 1] = 20
            elif trial == 2:
                full.fill_(-float("inf"))
            rank_pairs = []
            for rank in range(4):
                logits = full[:, rank * columns : (rank + 1) * columns].contiguous()
                valid = columns - (3 if rank == 3 else 0)
                parts = triton.cdiv(valid, 4096)
                width = triton.next_power_of_2(parts)
                values = torch.empty(rows, width, device="cuda")
                indices = torch.empty(rows, width, dtype=torch.int32, device="cuda")
                _argmax_parts[(rows, parts)](
                    logits,
                    columns,
                    valid,
                    values,
                    indices,
                    PARTS=width,
                    CHUNK=4096,
                    num_warps=8,
                )
                pairs = torch.empty(rows, 4, device="cuda")
                _local_pair[(rows,)](
                    values,
                    indices,
                    pairs,
                    parts,
                    START=rank * columns,
                    PARTS=width,
                    num_warps=1,
                )
                rank_pairs.append(pairs)
            gathered = torch.cat(rank_pairs, dim=1)
            actual = torch.empty(rows, dtype=torch.int64, device="cuda")
            _global_top1[(rows,)](gathered, actual, RANKS=4, num_warps=1)
            expected = full[:, : 4 * columns - 3].argmax(dim=1)
            assert torch.equal(actual, expected), (rows, columns, trial)
            cases += 1

for drafts in range(6):
    for rejected in range(drafts + 1):
        for index_dtype in (torch.int32, torch.int64):
            order = torch.randperm(drafts + 1, device="cuda").to(index_dtype)
            top = torch.randint(0, 129280, (drafts + 1,), device="cuda")
            expected = torch.full((1, drafts + 1), -1, dtype=torch.int32, device="cuda")
            proposed = top[order[:drafts].long()].int().clone()
            if rejected < drafts:
                proposed[rejected] = (proposed[rejected] + 1) % 129280
                expected[0, : rejected + 1] = top[order[: rejected + 1].long()].int()
            else:
                expected[0] = top[order.long()].int()
            metadata = SimpleNamespace(
                num_draft_tokens=[drafts],
                target_logits_indices=order[:drafts],
                bonus_logits_indices=order[drafts:],
                draft_token_ids=proposed,
            )
            actual = verify_top_tokens(top, metadata)
            assert torch.equal(actual, expected), (drafts, rejected, index_dtype)
            cases += 1


# The head check uses a local emulation of TP exchange. The real
# TP4 collective is covered by the four-node serving benchmark.
original_gather = native.tp_all_gather
native.tp_all_gather = lambda pairs: pairs.repeat(1, 4)
try:
    weight = torch.randn(64, 128, dtype=torch.float16, device="cuda") * 0.1
    head = SimpleNamespace(
        quant_method=SimpleNamespace(
            apply=lambda layer, hidden, bias: hidden @ weight.T
        ),
        shard_indices=SimpleNamespace(num_org_vocab_padding=0, org_vocab_start_index=0),
    )
    for rows in range(1, 7):
        for _ in range(4):
            hidden = torch.randn(rows * 2, 128, device="cuda").half()[::2]
            actual = greedy_top_tokens(head, hidden)
            expected = (hidden @ weight.T).argmax(dim=1)
            assert torch.equal(actual, expected), rows
            cases += 1
finally:
    native.tp_all_gather = original_gather

sampling = SimpleNamespace(
    all_greedy=True,
    max_num_logprobs=None,
    logprob_token_ids=None,
    no_penalties=True,
    allowed_token_ids_mask=None,
    bad_words_token_ids={},
    thinking_budget_state_holder=None,
    logitsprocs=SimpleNamespace(non_argmax_invariant=[], argmax_invariant=[]),
)
runner = SimpleNamespace(
    speculative_config=SimpleNamespace(
        method="dspark", use_dflash_ddtree=lambda: False
    ),
    input_batch=SimpleNamespace(num_reqs=1, sampling_metadata=sampling),
    parallel_config=SimpleNamespace(tensor_parallel_size=4),
    broadcast_pp_output=False,
    is_pooling_model=False,
    rejection_sampler=SimpleNamespace(
        synthetic_mode=False,
        _sampling_policy=SimpleNamespace(dynamic_tail_size=0),
        _trace=SimpleNamespace(value=lambda field: False),
    ),
    model=SimpleNamespace(get_v41_top_tokens=lambda: None),
    _ddtree_greedy_sampling_supported=(
        runner_module.GPUModelRunner._ddtree_greedy_sampling_supported
    ),
)
schedule = SimpleNamespace(has_structured_output_requests=False)
metadata = SimpleNamespace(num_draft_tokens=[5], max_spec_len=5)
runner_module._V41_GREEDY_VERIFY = True
runner_module._V41_TRUNC = False
eligible = runner_module.GPUModelRunner._can_use_v41_greedy_top_tokens
assert eligible(runner, schedule, metadata)
changes = (
    ("all_greedy", False),
    ("max_num_logprobs", 0),
    ("logprob_token_ids", {0: [1]}),
    ("no_penalties", False),
    ("allowed_token_ids_mask", object()),
    ("bad_words_token_ids", {0: [[1]]}),
    (
        "thinking_budget_state_holder",
        SimpleNamespace(has_tracked_requests=lambda: True),
    ),
    (
        "logitsprocs",
        SimpleNamespace(non_argmax_invariant=[object()], argmax_invariant=[]),
    ),
)
for attribute, value in changes:
    modified = copy.copy(runner)
    modified.input_batch = copy.copy(runner.input_batch)
    modified.input_batch.sampling_metadata = copy.copy(sampling)
    setattr(modified.input_batch.sampling_metadata, attribute, value)
    assert not eligible(modified, schedule, metadata), attribute
    cases += 1
assert not eligible(
    runner, SimpleNamespace(has_structured_output_requests=True), metadata
)
for flag in ("alignment", "corruption", "draft_logits", "rejection_profile"):
    runner.rejection_sampler._trace = SimpleNamespace(
        value=lambda field, selected=flag: field == selected
    )
    assert not eligible(runner, schedule, metadata), flag
    cases += 1
runner.rejection_sampler._trace = SimpleNamespace(value=lambda field: False)
runner.rejection_sampler.synthetic_mode = True
assert not eligible(runner, schedule, metadata)

# Extended or truncated vocabularies retain the established logits path.
from vllm.models.deepseek_v41 import model as model_module

original_tp_size = model_module.get_tensor_model_parallel_world_size
model_module.get_tensor_model_parallel_world_size = lambda: 4
try:
    marker = object()
    model = SimpleNamespace(
        lm_head=SimpleNamespace(
            quant_method=object.__new__(native.H16Head),
            num_embeddings=129280,
            org_vocab_size=129280,
        ),
        logits_processor=SimpleNamespace(
            soft_cap=None,
            scale=1.0,
            vocab_size=129280,
            get_top_tokens=lambda head, hidden: marker,
        ),
    )
    hidden = torch.empty(1, 128, device="cuda", dtype=torch.float16)
    model.lm_head.num_embeddings += 1
    assert (
        model_module.DeepseekV41ForCausalLM.get_v41_top_tokens(model, hidden) is marker
    )
    model.lm_head.num_embeddings -= 1
    model.logits_processor.vocab_size -= 1
    assert (
        model_module.DeepseekV41ForCausalLM.get_v41_top_tokens(model, hidden) is marker
    )
    cases += 2
finally:
    model_module.get_tensor_model_parallel_world_size = original_tp_size

print(f"PASS {cases + 3} checks; includes emulated TP4 and changing input tensors")
