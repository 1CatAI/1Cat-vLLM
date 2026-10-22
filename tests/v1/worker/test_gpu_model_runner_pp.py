# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu_model_runner import _select_dummy_sample_hidden_states


def test_pp_dummy_intermediates_are_not_sampled() -> None:
    intermediate = IntermediateTensors(
        {"hidden_states": torch.empty((8, 16), dtype=torch.float16)}
    )

    result = _select_dummy_sample_hidden_states(
        intermediate, np.array([3, 5]), torch.device("cpu")
    )

    assert result is None


def test_last_pp_rank_selects_final_scheduled_tokens() -> None:
    hidden_states = torch.arange(8 * 2).view(8, 2)

    result = _select_dummy_sample_hidden_states(
        hidden_states, np.array([3, 5]), torch.device("cpu")
    )

    assert result is not None
    torch.testing.assert_close(result, hidden_states[[2, 7]])


def test_pp_packet_packs_sampled_and_next_drafts() -> None:
    packet = pp_utils._pack_token_packet(
        torch.tensor([[42, 43]], dtype=torch.int64),
        torch.tensor([[101, 102, 103]], dtype=torch.int64),
        max_sample_len=4,
        max_draft_len=3,
    )

    torch.testing.assert_close(
        packet,
        torch.tensor([[42, 43, -1, -1, 101, 102, 103]], dtype=torch.int64),
    )


def test_pp_packet_rejects_tokens_wider_than_receive_buffer() -> None:
    with pytest.raises(ValueError, match="exceeds the PP receive contract"):
        pp_utils._pack_token_packet(
            torch.zeros((1, 3), dtype=torch.int64),
            None,
            max_sample_len=2,
            max_draft_len=0,
        )


@pytest.mark.parametrize(
    ("old_computed", "scheduled", "prefill_len", "max_seq_len", "expected"),
    [
        (0, 4, 8, 32, None),
        (0, 8, 8, 32, [True]),
        (31, 1, 8, 32, None),
    ],
)
def test_compute_need_sampled_mask(
    old_computed, scheduled, prefill_len, max_seq_len, expected
) -> None:
    batch = SimpleNamespace(
        num_computed_tokens_np=np.array([old_computed], dtype=np.int32),
        num_scheduled_tokens=np.array([scheduled], dtype=np.int32),
        prefill_len_np=np.array([prefill_len], dtype=np.int32),
        max_seq_len_np=np.array([max_seq_len], dtype=np.int32),
    )

    result = pp_utils.compute_need_sampled_mask(batch)

    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result.tolist() == expected


def test_pp_slot_ring_filters_reused_request_indices() -> None:
    waited: list[object] = []
    slot = pp_utils.PendingRecv(
        event=object(),  # type: ignore[arg-type]
        sampled_tokens=torch.tensor([[42], [43]], dtype=torch.int64),
        draft_tokens=torch.tensor([[101, 102], [103, 104]], dtype=torch.int64),
        num_sampled=torch.tensor([1, 1], dtype=torch.int32),
        num_rejected=torch.tensor([0, 0], dtype=torch.int32),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        need_sampled_mask=np.array([True, True]),
        gen_at_receive_np=np.array([0, 0], dtype=np.int32),
    )
    handler = object.__new__(pp_utils.PPHandler)
    handler.queue = deque([slot])
    handler.req_idx_gen_np = np.array([1, 0], dtype=np.int32)
    handler.device = torch.device("cpu")
    handler.main_stream = SimpleNamespace(wait_event=waited.append)

    outputs = handler.get_prev_sampled_outputs()

    assert outputs is not None
    torch.testing.assert_close(
        outputs["idx_mapping"], torch.tensor([-1, 1], dtype=torch.int32)
    )
    assert outputs["draft_tokens"] is slot.draft_tokens
    assert waited == [slot.event]
    assert list(handler.queue) == [None]


def test_mamba_neutral_pp_update_accepts_int32_slot_indices(monkeypatch) -> None:
    launches: list[tuple[torch.Tensor, int]] = []

    class FakeKernel:
        def __getitem__(self, _grid):
            def launch(idx_mapping, _output, *, VALUE):
                launches.append((idx_mapping, VALUE))

            return launch

    monkeypatch.setattr(mamba_hybrid, "_fill_num_accepted_kernel", FakeKernel())
    state = object.__new__(mamba_hybrid.MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.zeros(2, dtype=torch.int32)
    state._align_mode = False
    state._spec_replay = None
    idx_mapping = torch.tensor([0, -1], dtype=torch.int32)

    state.postprocess_state(idx_mapping, 0)

    assert launches == [(idx_mapping, 1)]


# ---------------------------------------------------------------------------
# PLE under PP: the runner gate follows the partition, not the PP size (#479)
# ---------------------------------------------------------------------------


def _pp2_vllm_config_with_ple(ple_layer_ids: list[int]):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_config = ModelConfig(model="facebook/opt-125m", dtype="float16", seed=42)
    # Give the text config the PLE shape of a Qwen4Exp checkpoint.
    model_config.hf_text_config.ple_layer_ids = ple_layer_ids
    model_config.hf_text_config.ngram_size = 3
    # A world of two ranks in a single-GPU test process: set after
    # construction, the validator would otherwise count the local GPUs.
    parallel_config = ParallelConfig()
    parallel_config.pipeline_parallel_size = 2
    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto"
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=10,
            max_num_batched_tokens=512,
            max_model_len=512,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        parallel_config=parallel_config,
    )


def _patch_pp_group(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(
            world_size=2,
            rank=0,
            rank_in_group=0,
            ranks=[0, 1],
            is_first_rank=True,
            is_last_rank=False,
        ),
    )


def test_runner_accepts_pp2_with_ple_layers_on_first_rank(monkeypatch) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _patch_pp_group(monkeypatch)
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    vllm_config = _pp2_vllm_config_with_ple([2])  # decoder layer 1 of 12
    with set_current_vllm_config(vllm_config):
        runner = GPUModelRunner(vllm_config, current_platform.device_type)
    assert runner.uses_ngram_embedding


def test_runner_rejects_pp2_with_ple_layer_on_second_rank(monkeypatch) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _patch_pp_group(monkeypatch)
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    vllm_config = _pp2_vllm_config_with_ple([10])  # decoder layer 9 of 12
    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="first pipeline rank"),
    ):
        GPUModelRunner(vllm_config, current_platform.device_type)
