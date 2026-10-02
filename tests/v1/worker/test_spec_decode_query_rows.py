# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _buffer(size, dtype=torch.int32):
    tensor = torch.zeros(size, dtype=dtype)
    return NS(cpu=tensor, gpu=tensor, np=tensor.numpy())


def _runner(sizes):
    """Run real input preparation and metadata math with CPU-backed buffers."""
    runner = object.__new__(GPUModelRunner)
    n = len(sizes)
    req_ids = [f"req-{i}" for i in range(n)]
    tokens = torch.arange(n * 32, dtype=torch.int32).reshape(n, 32)
    runner.device = torch.device("cpu")
    runner.input_batch = NS(
        num_reqs=n,
        req_ids=req_ids,
        req_id_to_index=dict(zip(req_ids, range(n))),
        num_computed_tokens_cpu=np.zeros(n, dtype=np.int32),
        num_computed_tokens_cpu_tensor=torch.zeros(n, dtype=torch.int32),
        num_prompt_tokens=np.zeros(n, dtype=np.int32),
        token_ids_cpu=tokens.numpy(),
        token_ids_cpu_tensor=tokens,
        req_prompt_embeds={},
        prev_req_id_to_index={},
        prev_sampled_token_ids=None,
    )
    runner.requests = {req_id: NS(num_tokens=0) for req_id in req_ids}
    runner.arange_np = np.arange(32, dtype=np.int32)
    runner._arange_scratch = np.zeros(32, dtype=np.int32)
    for name, size in (
        ("input_ids", 32),
        ("query_pos", 32),
        ("query_start_loc", n + 1),
        ("prev_positions", n),
        ("num_accepted_tokens", n),
        ("spec_state_slot_selectors", n),
        ("req_indices", 32),
        ("num_scheduled_tokens", n),
        ("num_decode_draft_tokens", n),
    ):
        setattr(runner, name, _buffer(size))
    runner.discard_request_mask = _buffer(n, torch.bool)
    runner.optimistic_seq_lens_cpu = torch.zeros(n, dtype=torch.int32)
    runner.num_computed_tokens = torch.zeros(n, dtype=torch.int32)
    runner.positions = torch.zeros(32, dtype=torch.int64)
    runner.seq_lens = torch.zeros(n, dtype=torch.int32)
    runner.speculative_config = runner.lora_config = None
    runner.num_accepted_tokens_event = runner.mamba_prev_last_scheduled_idx = None
    runner.uses_mrope = runner.enable_prompt_embeds = False
    runner.use_async_spec_decode = False
    runner.uses_xdrope_dim = 0
    runner.kv_cache_config = NS(kv_cache_groups=[])
    runner._commit_block_table_to_gpu = Mock()
    runner._apply_ddtree_position_overrides = Mock()
    # CPU/GPU views alias in this fixture; keep all indexing and metadata math.
    runner._copy_buffer_to_gpu = lambda buffer, n=None: buffer.gpu
    runner._calc_spec_decode_metadata = Mock(wraps=runner._calc_spec_decode_metadata)
    return runner


def _prepare(sizes, drafts):
    runner = _runner(sizes)
    output = NS(
        total_num_scheduled_tokens=sum(sizes),
        scheduled_spec_decode_tokens={f"req-{i}": ids for i, ids in drafts.items()},
    )
    return runner, output, np.array(sizes, dtype=np.int32)


@pytest.mark.parametrize(
    "sizes,drafts",
    [([2], {0: [1, 2]}), ([5, 2], {1: [1, 2]}), ([3, 2, 4], {1: [1, 2], 2: [3]})],
)
def test_insufficient_rows_fail_before_selecting_previous_request(sizes, drafts):
    runner, output, scheduled = _prepare(sizes, drafts)
    with pytest.raises(AssertionError):
        runner._prepare_inputs(output, scheduled)
    runner._calc_spec_decode_metadata.assert_not_called()


@pytest.mark.parametrize(
    "sizes,drafts,expected",
    [
        ([3], {0: [1, 2]}, [0, 1, 2]),
        ([5, 3], {1: [1, 2]}, [4, 5, 6, 7]),
        ([3, 5, 4], {2: [1, 2]}, [2, 7, 9, 10, 11]),
        ([3, 2, 4], {2: [1, 2], 0: [3]}, [1, 2, 4, 6, 7, 8]),
        ([2, 1], {1: []}, [1, 2]),
    ],
)
def test_enough_rows_keep_sampling_within_request(sizes, drafts, expected):
    runner, output, scheduled = _prepare(sizes, drafts)
    logits, metadata = runner._prepare_inputs(output, scheduled)
    assert logits.tolist() == expected
    assert metadata.num_draft_tokens == [
        len(drafts.get(i, [])) for i in range(len(sizes))
    ]
    runner._calc_spec_decode_metadata.assert_called_once()
