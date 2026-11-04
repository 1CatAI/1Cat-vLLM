# SPDX-License-Identifier: Apache-2.0
"""CPU unit test of non-last-stage accepted-token reconstruction."""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_each_request_gets_its_own_accepted_prefix_and_bonus(accepted):
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(spec_token_ids=[[10, 20, 30], [40, 50, 60]])
    )
    for row, req_id in enumerate(["a", "b"]):
        result = GPUModelRunner._pp_prepend_accepted_drafts(
            runner,
            req_id,
            SimpleNamespace(num_tokens=100),
            row,
            100 + accepted,
            [90 + row],
            {},
        )
        expected = [[10, 20, 30], [40, 50, 60]][row][:accepted] + [90 + row]
        assert result == expected


def test_supplied_short_history_refuses_even_with_held_drafts():
    runner = SimpleNamespace(input_batch=SimpleNamespace(spec_token_ids=[[10, 20, 30]]))
    with pytest.raises(RuntimeError, match="full token history is shorter"):
        GPUModelRunner._pp_prepend_accepted_drafts(
            runner, "a", SimpleNamespace(num_tokens=100), 0, 102, [90], {"a": [1] * 101}
        )


def test_full_history_owns_tokens_for_resumed_request():
    runner = SimpleNamespace(input_batch=SimpleNamespace(spec_token_ids=[[10, 20, 30]]))
    assert GPUModelRunner._pp_prepend_accepted_drafts(
        runner,
        "a",
        SimpleNamespace(num_tokens=100),
        None,
        102,
        [90],
        {"a": list(range(103))},
    ) == [100, 101, 90]


def test_missing_history_and_missing_held_prefix_refuse():
    runner = SimpleNamespace(input_batch=SimpleNamespace(spec_token_ids=[[]]))
    with pytest.raises(RuntimeError, match="accepted draft token"):
        GPUModelRunner._pp_prepend_accepted_drafts(
            runner,
            "a",
            SimpleNamespace(num_tokens=100),
            None,
            102,
            [90],
            {},
        )


@pytest.mark.parametrize("num_requests", [2, 3])
def test_real_runner_keeps_each_requests_accepted_history(monkeypatch, num_requests):
    import torch

    import vllm.v1.worker.gpu_model_runner as runner_module
    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.config import DeviceConfig
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 2)
    monkeypatch.setattr(
        runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    scheduler = create_scheduler(
        pipeline_parallel_size=2,
        num_speculative_tokens=3,
        device_config=DeviceConfig(device="cpu"),
        max_model_len=512,
        block_size=16,
    )
    runner = object.__new__(GPUModelRunner)
    runner.pp_keep_batch = False
    runner._pp_keep_batch_kept_logged = False
    runner.device = torch.device("cpu")
    runner.input_batch = InputBatch(
        max_num_reqs=4,
        max_model_len=512,
        max_num_batched_tokens=512,
        device=runner.device,
        pin_memory=False,
        vocab_size=50272,
        block_sizes=[16],
        kernel_block_sizes=[16],
        num_spec_tokens=3,
    )
    runner.requests = {}
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.late_interaction_runner = runner_module.LateInteractionRunner()
    runner._dynamic_draft_vocab_prefill_bootstrap = (
        runner_module.DynamicDraftVocabPrefillBootstrapState()
    )
    runner.speculative_config = None
    runner.use_async_spec_decode = False
    runner.sync_spec_decode_accept_counts = False
    runner.use_async_scheduling = False
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    requests = create_requests(
        num_requests=num_requests, num_tokens=37, max_tokens=200, block_size=16
    )
    for request in requests:
        scheduler.add_request(request)
    for round_id in range(6):
        scheduled = scheduler.schedule()
        runner._update_states(scheduled)
        ids = list(scheduled.num_scheduled_tokens)
        for request in requests:
            row = runner.input_batch.req_id_to_index[request.request_id]
            count = int(runner.input_batch.num_tokens_no_spec[row])
            assert runner.input_batch.token_ids_cpu[row, :count].tolist() == list(
                request.all_token_ids
            )
            assert runner.requests[request.request_id].output_token_ids == list(
                request.output_token_ids
            )
        empty = scheduler.schedule()
        assert empty.total_num_scheduled_tokens == 0
        runner._update_states(empty)
        sampled = [
            list(scheduled.scheduled_spec_decode_tokens.get(req, []))[: round_id % 4]
            + [1000 + round_id * 10 + i]
            for i, req in enumerate(ids)
        ]
        scheduler.update_from_output(
            scheduled,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={req: i for i, req in enumerate(ids)},
                sampled_token_ids=sampled,
            ),
        )
        scheduler.update_draft_token_ids(
            DraftTokenIds(
                ids, [[200 + i * 10 + j for j in range(3)] for i in range(len(ids))]
            )
        )
