# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


def _make_config(max_num_seqs: int, verifier_sizes: list[int]):
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_state_tokens=lambda: 4,
    )
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            data_parallel_size=1,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=verifier_sizes.copy(),
            max_cudagraph_capture_size=max(verifier_sizes),
        ),
        speculative_config=speculative_config,
    )


@pytest.mark.parametrize(
    ("max_num_seqs", "request_sizes", "verifier_sizes"),
    [
        (16, [1, 2, 4, 6, 8, 12, 16], [5, 10, 20, 30, 40, 60, 80]),
        (32, [1, 2, 4, 6, 8, 12, 16, 32], [5, 10, 20, 30, 40, 60, 80, 160]),
    ],
)
def test_split_managers_keep_exact_full_graph_shapes(
    monkeypatch,
    max_num_seqs: int,
    request_sizes: list[int],
    verifier_sizes: list[int],
):
    monkeypatch.setenv("VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS", "1")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.is_device_capability",
        lambda capability: capability == (7, 0),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_global_graph_pool",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_pp_group",
        lambda: MagicMock(is_first_rank=True, is_last_rank=True),
    )

    config = _make_config(max_num_seqs, verifier_sizes)
    shared_sizes = config.compilation_config.cudagraph_capture_sizes.copy()
    shared_max_size = config.compilation_config.max_cudagraph_capture_size
    target = CudaGraphManager(
        config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=5,
    )
    draft = CudaGraphManager(
        config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=1,
    )

    assert target._capture_sizes == verifier_sizes
    assert draft._capture_sizes == request_sizes
    assert config.compilation_config.cudagraph_capture_sizes == shared_sizes
    assert config.compilation_config.max_cudagraph_capture_size == shared_max_size

    target._graphs_captured = True
    draft._graphs_captured = True
    for batch_size in request_sizes:
        target_desc = target.dispatch(batch_size, batch_size * 5, 5)
        draft_desc = draft.dispatch(batch_size, batch_size, 1)
        assert target_desc.cg_mode == CUDAGraphMode.FULL
        assert target_desc.num_tokens == batch_size * 5
        assert target_desc.num_reqs == batch_size
        assert draft_desc.cg_mode == CUDAGraphMode.FULL
        assert draft_desc.num_tokens == batch_size
        assert draft_desc.num_reqs == batch_size
