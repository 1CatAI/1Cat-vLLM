# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for conservative MRV2 attention graph selection."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.v1.attention.ops.sm70_e4m3_long import MAX_CONTEXT
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)


@pytest.fixture
def graph_pair():
    manager = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    ordinary = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 1, 8)
    bounded = replace(ordinary, attention_context_bucket=MAX_CONTEXT)
    manager._long_attention_graphs = {ordinary: bounded}
    manager.graphs = {ordinary: object(), bounded: object()}
    return manager, ordinary, bounded


def test_context_boundary_and_switch_back(graph_pair):
    manager, ordinary, bounded = graph_pair
    for upper, expected in (
        (1024, bounded),
        (MAX_CONTEXT, bounded),
        (MAX_CONTEXT + 1, ordinary),
        (262144, ordinary),
        (32768, bounded),
        (0, ordinary),
    ):
        assert (
            manager.select_attention_graph(ordinary, torch.tensor([upper])) == expected
        )


def test_device_hint_never_copied_to_host(graph_pair):
    manager, ordinary, _ = graph_pair
    # Accessing a device hint's values would fail: selection must inspect the
    # device first and use the full-context graph without requesting a copy.
    device_hint = SimpleNamespace(device=torch.device("cuda"))
    assert manager.select_attention_graph(ordinary, device_hint) == ordinary


def test_scalar_graph_retains_short_and_oversized_fallback():
    manager = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    ordinary = BatchExecutionDescriptor(CUDAGraphMode.FULL, 1, 1, 1)
    bounded = replace(ordinary, attention_context_bucket=262144)
    manager._long_attention_graphs = {ordinary: bounded}
    manager.graphs = {ordinary: object(), bounded: object()}
    for upper in (1, 1024, 131071, 262145):
        assert (
            manager.select_attention_graph(ordinary, torch.tensor([upper])) == ordinary
        )
    for upper in (131072, 261888, 262144):
        assert (
            manager.select_attention_graph(ordinary, torch.tensor([upper])) == bounded
        )


def test_other_batch_shapes_and_missing_capture_fall_back(graph_pair):
    manager, ordinary, bounded = graph_pair
    other = replace(ordinary, num_tokens=16, num_reqs=2)
    assert manager.select_attention_graph(other, torch.tensor([1024, 1024])) == other
    assert manager.select_attention_graph(ordinary, torch.tensor([1024, 0])) == ordinary
    del manager.graphs[bounded]
    assert manager.select_attention_graph(ordinary, torch.tensor([1024])) == ordinary


def test_disabled_operator_preserves_original_binding(monkeypatch):
    from vllm.v1.attention.ops.sm70_e4m3_long import MANIFEST_ENV, wrap_long_attention

    monkeypatch.delenv(MANIFEST_ENV, raising=False)

    def original(*args, **kwargs):
        raise AssertionError("Binding inspection must not launch an operator")

    assert wrap_long_attention(original) is original


@pytest.mark.parametrize(
    "enabled,method,sm70,target,sequence_parallel,expect_tail",
    [
        (True, "dflash", True, True, False, True),
        (False, "dflash", True, True, False, False),
        (True, "mtp", True, True, False, False),
        (True, "dflash", False, True, False, False),
        (True, "dflash", True, False, False, False),
        (True, "dflash", True, True, True, False),
    ],
)
def test_tail_capture_and_dispatch_from_real_initialization(
    monkeypatch, enabled, method, sm70, target, sequence_parallel, expect_tail
):
    monkeypatch.setenv("VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS", str(int(enabled)))
    monkeypatch.setenv("VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS", "0")
    monkeypatch.delenv("VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST", raising=False)
    monkeypatch.setattr(cg.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(cg.current_platform, "is_device_capability", lambda cap: sm70)
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(
        cg,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    compilation = CompilationConfig(
        cudagraph_capture_sizes=[8, 16, 24, 32], max_cudagraph_capture_size=32
    )
    compilation.pass_config.enable_sp = sequence_parallel
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=compilation,
        parallel_config=SimpleNamespace(
            data_parallel_size=1, tensor_parallel_size=4, pipeline_parallel_size=1
        ),
        speculative_config=SimpleNamespace(
            method=method, num_speculative_tokens=7, ngram_assist=False
        ),
    )
    cls = ModelCudaGraphManager if target else cg.CudaGraphManager
    manager = cls(config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 8)
    manager._graphs_captured = True
    for q in range(1, 8):
        desc = manager.dispatch(1, q, q)
        assert desc.cg_mode == (
            CUDAGraphMode.FULL if expect_tail else CUDAGraphMode.NONE
        )
        if expect_tail:
            assert desc in manager._capture_descs[CUDAGraphMode.FULL]
            assert desc.num_tokens == desc.uniform_token_count == q
        # Admission is B1 only.
        assert manager.dispatch(2, 2 * q, q).cg_mode == CUDAGraphMode.NONE
    assert manager.dispatch(1, 8, 8).cg_mode == CUDAGraphMode.FULL


@pytest.mark.parametrize(
    "manifest",
    [
        {"max_context": 262145},
        {"max_context": 0},
        {"query_rows": [1, 8]},
        {"query_rows": [9]},
        {"query_rows": []},
    ],
)
def test_reject_unadmitted_attention_manifest(manifest):
    from vllm.v1.attention.ops.sm70_e4m3_long import long_attention_contract

    with pytest.raises(ValueError):
        long_attention_contract(manifest)


def test_explicit_attention_manifest_preserves_default_and_extends_tail_domain():
    from vllm.v1.attention.ops.sm70_e4m3_long import long_attention_contract

    assert long_attention_contract({}) == (132096, (8,))
    assert long_attention_contract(
        {"max_context": 262144, "query_rows": list(range(2, 9))}
    ) == (262144, tuple(range(2, 9)))


@pytest.mark.parametrize("query_len", [1, 6, 7])
@pytest.mark.parametrize("prefilling", [True, False])
def test_runner_does_not_dispatch_short_prefill_as_tail(
    monkeypatch, query_len, prefilling
):
    from vllm.v1.worker.gpu import model_runner as mrv2

    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.update_pp_decode_requests = lambda: None
    for method in ("finish_requests", "free_states", "add_requests", "update_requests"):
        setattr(runner, method, lambda _: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.cudagraph_manager = SimpleNamespace(_sm70_dflash2_tail_graphs=True)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"r": 0},
        num_computed_prefill_tokens=[0 if prefilling else 16],
        prefill_len=SimpleNamespace(np=[16]),
    )
    runner.is_encoder_decoder = False
    runner.dp_size = 1
    runner.dp_rank = 0
    scheduler = SimpleNamespace(
        new_block_ids_to_zero=[],
        total_num_scheduled_tokens=query_len,
        num_scheduled_tokens={"r": query_len},
    )

    class DispatchObserved(Exception):
        pass

    def dispatch(manager, num_reqs, num_tokens, uniform, *args, **kwargs):
        assert uniform == (None if prefilling else query_len)
        raise DispatchObserved

    monkeypatch.setattr(mrv2, "dispatch_cg_and_sync_dp", dispatch)
    with pytest.raises(DispatchObserved):
        runner.execute_model(scheduler)
