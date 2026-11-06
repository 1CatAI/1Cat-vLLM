# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for two-request DFlash2 verify-batch cudagraph dispatch.

A single resident decoder produces a clean uniform verify batch every step
(1 req x (1 + num_draft) tokens, all draft tokens present) and replays the
captured FULL decode cudagraph -> GPU-bound. With two resident decoders the
verify batch is only dispatched to a FULL graph when it is *also* a clean
uniform verify shape (e.g. 2 reqs x 8 tokens, both carrying 7 drafts). When the
two decoders fall out of lock-step the batch is non-uniform (uniform_tok_count
becomes None) and dispatch must fall back to the PIECEWISE/eager forward. That
eager forward, launch-bound over a long context, is the reqs>1 host-time cost
measured with py-spy; these checks pin the dispatch decision so a future fix
(capturing FULL graphs for more verify shapes, or keeping the decoders aligned)
can be validated against the intended behavior."""

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
    get_uniform_token_count,
    is_speculative_uniform_batch,
)

Q = 8  # DFlash2: num_speculative_tokens(7) + 1 query tokens per verify request


def _manager_with_captured_verify_graphs():
    """A manager with FULL (8,1,8)/(16,2,8) decode graphs and their PIECEWISE
    fallbacks captured, matching the SM70 FULL_AND_PIECEWISE decode policy with
    max_num_seqs=2 and capture_sizes=(8, 16)."""
    m = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    m._graphs_captured = True
    full8 = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 1, Q)
    pw8 = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 8, None, None)
    full16 = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 2, Q)
    pw16 = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 16, None, None)
    cands: list[list[BatchExecutionDescriptor]] = [[] for _ in range(17)]
    cands[8] = [full8, pw8]
    cands[16] = [full16, pw16]
    m._candidates = cands
    return m, full8, pw8, full16, pw16


def _select_uniform_tok_count(num_scheduled, spec_tokens):
    """Mirror model_runner: shape-uniform AND a real speculative verify."""
    num_reqs = len(num_scheduled)
    num_toks = sum(num_scheduled.values())
    max_q = max(num_scheduled.values())
    utc = get_uniform_token_count(num_reqs, num_toks, max_q)
    if utc is not None and not is_speculative_uniform_batch(
        utc, num_scheduled, spec_tokens
    ):
        utc = None
    return num_reqs, num_toks, utc


def test_two_request_aligned_verify_dispatches_full():
    m, _, _, full16, _ = _manager_with_captured_verify_graphs()
    drafts = {"a": list(range(7)), "b": list(range(7))}
    num_reqs, num_toks, utc = _select_uniform_tok_count({"a": 8, "b": 8}, drafts)
    assert (num_reqs, num_toks, utc) == (2, 16, 8)
    assert m.dispatch(num_reqs, num_toks, utc) is full16


def test_single_request_verify_dispatches_full():
    m, full8, *_ = _manager_with_captured_verify_graphs()
    num_reqs, num_toks, utc = _select_uniform_tok_count({"a": 8}, {"a": list(range(7))})
    assert (num_reqs, num_toks, utc) == (1, 8, 8)
    assert m.dispatch(num_reqs, num_toks, utc) is full8


def test_two_request_misaligned_token_counts_fall_back_to_piecewise():
    # One decoder verifies 8 tokens, the other emits a single (bonus) token:
    # shape is non-uniform, so no FULL verify graph matches.
    m, _, _, _, pw16 = _manager_with_captured_verify_graphs()
    num_reqs = 2
    num_toks = 9
    utc = get_uniform_token_count(num_reqs, num_toks, 8)
    assert utc is None
    # padded/rounded to the nearest captured token bucket (16); FULL needs an
    # exact uniform match, so the PIECEWISE descriptor is selected instead.
    assert m.dispatch(num_reqs, 16, utc) is pw16


def test_two_request_missing_drafts_fall_back_to_piecewise():
    # Shape-uniform 2x8, but one decoder is not carrying a full 7-draft tree
    # this step (e.g. just after a rejection); the verify graph would read
    # stale spec-state metadata, so it must not be replayed.
    m, _, _, _, pw16 = _manager_with_captured_verify_graphs()
    drafts = {"a": list(range(7)), "b": list(range(3))}
    num_reqs, num_toks, utc = _select_uniform_tok_count({"a": 8, "b": 8}, drafts)
    assert (num_reqs, num_toks, utc) == (2, 16, None)
    assert m.dispatch(num_reqs, num_toks, utc) is pw16


def test_uncaptured_shape_falls_back_to_eager_none():
    m, *_ = _manager_with_captured_verify_graphs()
    desc = m.dispatch(3, 24, 8)  # 3 reqs never captured (max_num_seqs=2)
    assert desc.cg_mode is CUDAGraphMode.NONE
