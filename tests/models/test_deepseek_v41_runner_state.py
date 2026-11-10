# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.models.deepseek_v41.model import DeepseekV41Model
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41TextConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    _clip_stateful_sample_limits,
)


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("deepseek_v41_text", (37, 38, 39)),
        ("deepseek_v4", (38, 39, 40)),
    ],
)
def test_dspark_capture_boundary_preserves_v41_pre_block_and_legacy_outputs(
    model_type, expected
):
    config_class = (
        DeepseekV41TextConfig if model_type == "deepseek_v41_text" else SimpleNamespace
    )
    config = config_class(dspark_target_layer_ids=[37, 38, 39])
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        )
    )
    assert (
        GPUModelRunner._get_eagle3_aux_layers_from_config(cast(GPUModelRunner, runner))
        == expected
    )


@pytest.mark.parametrize(
    "model_type,mode,expected",
    [
        ("deepseek_v41_text", CUDAGraphMode.FULL, 1),
        ("deepseek_v41_text", CUDAGraphMode.FULL_DECODE_ONLY, 6),
        ("deepseek_v4", CUDAGraphMode.FULL, 6),
    ],
)
def test_single_v41_full_graphs_preserve_exact_target_and_draft_sizes(
    model_type, mode, expected
):
    compilation = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=mode,
        cudagraph_capture_sizes=[1, 2, 3, 4, 5, 6],
        max_cudagraph_capture_size=6,
    )
    dispatch = Mock()
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        ),
        max_num_reqs=1,
        uniform_decode_query_len=6,
        compilation_config=compilation,
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        kv_cache_config=None,
        speculative_config=None,
        cudagraph_dispatcher=SimpleNamespace(initialize_cudagraph_keys=dispatch),
    )
    GPUModelRunner._check_and_update_cudagraph_mode(
        cast(GPUModelRunner, runner), [], []
    )
    dispatch.assert_called_once_with(mode, expected)
    assert compilation.cudagraph_capture_sizes == (
        [1, 2, 3, 4, 5, 6] if expected == 1 else [6]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("accepted", range(1, 7))
def test_nonhybrid_v41_commits_the_contiguous_accepted_input_prefix(accepted):
    selected: list[torch.Tensor] = []
    runner = SimpleNamespace(
        model=SimpleNamespace(commit_speculative_state=selected.append),
        speculative_config=SimpleNamespace(method="dspark"),
        model_config=SimpleNamespace(is_hybrid=False),
    )
    # Stale output values after rejection must not extend the accepted prefix.
    tokens = torch.full((1, 6), 17, device="cuda", dtype=torch.int64)
    if accepted < 6:
        tokens[0, accepted] = -1
    GPUModelRunner._update_states_after_model_execute(
        cast(GPUModelRunner, runner), tokens, cast(SchedulerOutput, None)
    )
    assert len(selected) == 1
    assert selected[0].item() == accepted


@pytest.mark.parametrize("stop_at", range(6))
def test_stateful_output_stops_at_first_eos_including_stale_tail(stop_at):
    params = SamplingParams(max_tokens=100)
    params.update_from_generation_config({}, eos_token_id=1)
    request = SimpleNamespace(sampling_params=params, output_token_ids=[])
    tokens = torch.full((1, 6), 17)
    tokens[0, stop_at] = 1
    _clip_stateful_sample_limits(tokens, ["req"], {"req": request})
    assert tokens[0, stop_at] == 1
    assert (tokens[0, :stop_at] == 17).all()
    assert (tokens[0, stop_at + 1 :] == -1).all()


def test_stateful_ignore_eos_and_output_budget():
    params = SamplingParams(max_tokens=5, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=1)
    request = SimpleNamespace(sampling_params=params, output_token_ids=[7, 8])
    tokens = torch.tensor([[17, 1, 17, 1, 17, 17]])
    _clip_stateful_sample_limits(tokens, ["req"], {"req": request})
    torch.testing.assert_close(tokens, torch.tensor([[17, 1, 17, -1, -1, -1]]))


@pytest.mark.parametrize(
    "prefill,start,length,prompt,expected",
    [
        (True, 1024, 992, 2016, 32),
        (False, 2016, 6, 2016, 31),
        (False, 16380, 3, 16352, 3),
        (True, 2048, 32, 2016, 0),
    ],
)
def test_device_output_budget_uses_corrected_input_position(
    prefill, start, length, prompt, expected
):
    model = SimpleNamespace(
        pending_prefill=torch.tensor([prefill]),
        pending_position=torch.tensor([start]),
        pending_length=torch.tensor([length]),
    )
    budget = DeepseekV41Model.output_budget(cast(DeepseekV41Model, model), 32, prompt)
    assert budget.item() == expected
    request = SimpleNamespace(
        sampling_params=SamplingParams(max_tokens=32, ignore_eos=True),
        output_token_ids=[-1] * 99,  # Deliberately stale optimistic CPU length.
    )
    tokens = torch.full((1, 6), 17)
    _clip_stateful_sample_limits(tokens, ["req"], {"req": request}, budget)
    assert (tokens[0, : min(expected, 6)] == 17).all()
    assert (tokens[0, expected:] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_plain_decode_commits_and_legacy_model_without_hook_is_untouched():
    selected: list[torch.Tensor] = []
    runner = SimpleNamespace(
        model=SimpleNamespace(commit_speculative_state=selected.append),
        speculative_config=None,
        model_config=SimpleNamespace(is_hybrid=False),
    )
    tokens = torch.tensor([[1]], device="cuda")  # EOS still commits its input.
    GPUModelRunner._update_states_after_model_execute(
        cast(GPUModelRunner, runner), tokens, cast(SchedulerOutput, None)
    )
    assert selected[0].item() == 1
    runner.model = SimpleNamespace()
    GPUModelRunner._update_states_after_model_execute(
        cast(GPUModelRunner, runner), tokens, cast(SchedulerOutput, None)
    )
    assert len(selected) == 1
