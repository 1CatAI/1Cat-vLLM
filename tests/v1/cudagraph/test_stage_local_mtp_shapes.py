# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A native draft on the last Ada PP rank must get its own decode shapes."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.platforms.interface import DeviceCapability
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import EagleCudaGraphManagerBase

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("capability", [(7, 0), (8, 6), (8, 9)])
@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_stage_local_mtp_preserves_target_and_captures_one_request(
    monkeypatch, capability, tokens
):
    monkeypatch.setattr(cg.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        cg.current_platform,
        "get_device_capability",
        lambda *a, **k: DeviceCapability(*capability),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(
        cg, "get_pp_group", lambda: NS(is_first_rank=False, is_last_rank=True)
    )
    compilation = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[tokens + 1],
        max_cudagraph_capture_size=tokens + 1,
    )
    compilation.runtime.split_draft_graphs = False
    config = NS(
        scheduler_config=NS(max_num_seqs=1),
        parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1),
        compilation_config=compilation,
        attention_config=NS(flash_v100=NS(options=NS(value=lambda *a: False))),
        speculative_config=NS(
            method="mtp",
            num_speculative_state_tokens=lambda: tokens,
            draft_model_config=NS(hf_config=NS(mtp_draft_pipeline_parallel_size=1)),
        ),
    )
    target = cg.CudaGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, tokens + 1
    )
    draft = EagleCudaGraphManagerBase(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 1
    )
    assert compilation.cudagraph_capture_sizes == [tokens + 1]
    assert target._capture_sizes == [tokens + 1]
    assert draft._capture_sizes == [1]
    assert len(draft._capture_descs[CUDAGraphMode.FULL]) == 1
    draft._graphs_captured = target._graphs_captured = True
    assert draft.dispatch(1, 1, 1).cg_mode == CUDAGraphMode.FULL
    assert target.dispatch(1, tokens + 1, tokens + 1).cg_mode == CUDAGraphMode.FULL
