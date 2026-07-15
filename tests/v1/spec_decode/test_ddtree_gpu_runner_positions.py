# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.ddtree_payload import DDTreeDraftPayload
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _branched_payload() -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=(11, 21, 12),
        parent_indices=(-1, 0, -1),
        node_depths=(1, 2, 1),
        node_scores=(0.0, 0.0, -0.1),
        top1_chain_token_ids=(11, 21),
        flat_draft_token_ids=(11, 21),
        budget=3,
        top_k=2,
        chain_seed=True,
    )


def _flat_payload() -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=(11, 21, 31),
        parent_indices=(-1, 0, 1),
        node_depths=(1, 2, 3),
        node_scores=(0.0, 0.0, 0.0),
        top1_chain_token_ids=(11, 21, 31),
        flat_draft_token_ids=(11, 21, 31),
        budget=3,
        top_k=1,
        chain_seed=True,
    )


def _scheduler_output(payload: DDTreeDraftPayload) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"r0": 5, "r1": 2},
        total_num_scheduled_tokens=7,
        scheduled_spec_decode_tokens={"r0": list(payload.tree_token_ids)},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        scheduled_ddtree_payloads={"r0": payload},
    )


def test_apply_ddtree_position_overrides_uses_node_depths_on_spec_tail() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.input_batch = SimpleNamespace(req_ids=["r0", "r1"])
    runner.positions = torch.tensor(
        [100, 101, 102, 103, 104, 200, 201],
        dtype=torch.int64,
    )
    payload = _branched_payload()

    runner._apply_ddtree_position_overrides(
        _scheduler_output(payload),
        num_reqs=2,
        num_scheduled_tokens=np.array([5, 2], dtype=np.int32),
        cu_num_tokens=np.array([5, 7], dtype=np.int32),
    )

    assert runner.positions.tolist() == [100, 101, 102, 103, 102, 200, 201]


def test_clamp_ddtree_sampler_output_to_request_max_tokens() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["r0", "r1"])
    runner.requests = {
        "r0": SimpleNamespace(
            sampling_params=SimpleNamespace(max_tokens=3),
            output_token_ids=[101],
        ),
        "r1": SimpleNamespace(
            sampling_params=SimpleNamespace(max_tokens=10),
            output_token_ids=[201, 202],
        ),
    }
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor(
            [
                [11, 12, 13, 14, -1],
                [21, 22, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor(
            [
                [0, 1, 2, 3, -1],
                [0, 4, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )

    clamped = runner._clamp_ddtree_sampler_output_to_request_limits(sampler_output)

    assert clamped.sampled_token_ids.tolist() == [
        [11, 12, -1, -1, -1],
        [21, 22, -1, -1, -1],
    ]
    assert clamped.ddtree_accepted_node_indices is not None
    assert clamped.ddtree_accepted_node_indices.tolist() == [
        [0, 1, -1, -1, -1],
        [0, 4, -1, -1, -1],
    ]


def test_apply_ddtree_position_overrides_requires_scheduled_tree_tokens() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.input_batch = SimpleNamespace(req_ids=["r0"])
    runner.positions = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    payload = _branched_payload()
    scheduler_output = _scheduler_output(payload)
    scheduler_output.scheduled_spec_decode_tokens = {"r0": [11, 21]}

    runner._apply_ddtree_position_overrides(
        scheduler_output,
        num_reqs=1,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        cu_num_tokens=np.array([4], dtype=np.int32),
    )

    assert runner.positions.tolist() == [10, 11, 12, 13]


def test_ddtree_accepted_kv_local_copies_move_branch_to_prefix() -> None:
    copies = GPUModelRunner._ddtree_accepted_kv_local_copies(
        req_ids=["r0"],
        num_scheduled_tokens={"r0": 6},
        scheduled_spec_decode_tokens={"r0": [11, 21, 31, 12, 22]},
        accepted_node_indices=torch.tensor([[0, 4, 5, -1]], dtype=torch.int32),
    )

    assert copies == [(4, 1), (5, 2)]


def test_ddtree_accepted_nodes_are_flat_prefix() -> None:
    assert GPUModelRunner._ddtree_accepted_nodes_are_flat_prefix(
        torch.tensor([[0, 1, 2, 3], [0, -1, -1, -1]], dtype=torch.int32)
    )
    assert not GPUModelRunner._ddtree_accepted_nodes_are_flat_prefix(
        torch.tensor([[0, 1, 3, -1]], dtype=torch.int32)
    )
    assert not GPUModelRunner._ddtree_accepted_nodes_are_flat_prefix(
        torch.tensor([[0, 4, 5, -1]], dtype=torch.int32)
    )


def test_ddtree_state_slot_selectors_use_last_accepted_node() -> None:
    selectors = GPUModelRunner._ddtree_state_slot_selectors_from_accepted_nodes(
        torch.tensor(
            [
                [0, 1, 2, -1],
                [0, -1, -1, -1],
                [0, 4, 5, -1],
            ],
            dtype=torch.int32,
        )
    )

    assert selectors.tolist() == [3, 1, 6]


def test_ddtree_state_slot_selectors_cover_o128_branch_index() -> None:
    selectors = GPUModelRunner._ddtree_state_slot_selectors_from_accepted_nodes(
        torch.tensor([[0, 1, 27, -1]], dtype=torch.int32)
    )

    assert selectors.tolist() == [28]


def test_ddtree_state_slot_selectors_fallback_to_flat_for_mixed_rows() -> None:
    selectors = GPUModelRunner._ddtree_state_slot_selectors_from_accepted_nodes(
        torch.tensor(
            [
                [0, 2, 5, -1],
                [0, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
        flat_selectors=torch.tensor([3, 4], dtype=torch.int32),
    )

    assert selectors.tolist() == [6, 4]


def test_stage_ddtree_parent_metadata_is_idempotent() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.max_num_reqs = 2
    runner.max_spec_state_slots = 4
    runner.ddtree_parent_ids = SimpleNamespace(
        cpu=torch.zeros((2, 4), dtype=torch.int32),
        gpu=torch.zeros((2, 4), dtype=torch.int32),
    )
    runner.ddtree_num_tree_tokens_cpu = torch.zeros(2, dtype=torch.int32)
    copy_calls = 0

    def copy_to_gpu(_buffer: object, _rows: int) -> None:
        nonlocal copy_calls
        copy_calls += 1

    runner._copy_buffer_to_gpu = copy_to_gpu
    metadata = SimpleNamespace(
        parent_ids=torch.tensor(
            [[-1, -1, 1, 0], [0, 0, 0, 0]],
            dtype=torch.int32,
        ),
        num_tree_tokens_cpu=torch.tensor([3, 0], dtype=torch.int32),
        request_ids=("r0", "r1"),
    )

    staged = runner._stage_ddtree_parent_metadata(
        metadata,
        num_reqs=2,
    )
    restaged = runner._stage_ddtree_parent_metadata(
        staged,
        num_reqs=2,
    )

    assert copy_calls == 1
    assert staged is not None
    assert restaged is not None
    assert restaged.parent_ids.data_ptr() == runner.ddtree_parent_ids.gpu.data_ptr()
    assert restaged.num_tree_tokens_cpu.data_ptr() == (
        runner.ddtree_num_tree_tokens_cpu.data_ptr()
    )


def test_mamba_postprocess_ddtree_state_slot_bias_uses_compact_node_index(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int]] = []
    state = torch.arange(32, dtype=torch.float32).reshape(16, 2)

    def copy_func(
        state_tensor: torch.Tensor,
        block_ids: list[int],
        copy_src_block_idx: int,
        copy_num_accepted_tokens: int,
    ) -> SimpleNamespace:
        calls.append((copy_src_block_idx, copy_num_accepted_tokens))
        block_id = block_ids[copy_src_block_idx]
        return SimpleNamespace(
            start_addr=state_tensor[block_id].data_ptr(),
            num_elements=state_tensor[block_id].numel(),
        )

    monkeypatch.setattr(mamba_utils, "do_mamba_copy_block", lambda copy_bufs: None)

    copy_bufs = SimpleNamespace(
        mamba_group_ids=[0],
        mamba_spec=SimpleNamespace(block_size=4),
        offset=0,
        src_ptrs=SimpleNamespace(np=np.zeros(4, dtype=np.int64)),
        dst_ptrs=SimpleNamespace(np=np.zeros(4, dtype=np.int64)),
        sizes=SimpleNamespace(np=np.zeros(4, dtype=np.int32)),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r0": 5},
        scheduled_spec_decode_tokens={"r0": [11, 21, 31, 12]},
    )
    input_batch = SimpleNamespace(
        req_ids=["r0"],
        num_accepted_tokens_cpu=np.array([4], dtype=np.int32),
        spec_num_accepted_tokens_cpu=np.array([10], dtype=np.int32),
    )
    requests = {
        "r0": SimpleNamespace(
            num_computed_tokens=4,
            block_ids={0: list(range(16))},
        )
    }
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["mamba"])]
    )
    forward_context = {"mamba": SimpleNamespace(kv_cache=[state])}

    mamba_utils.postprocess_mamba(
        scheduler_output=scheduler_output,
        kv_cache_config=kv_cache_config,
        input_batch=input_batch,
        requests=requests,
        mamba_state_idx={"r0": 0},
        forward_context=forward_context,
        mamba_state_copy_funcs=(copy_func,),
        copy_bufs=copy_bufs,
        ddtree_accepted_node_indices=torch.tensor(
            [[0, 2, 5, 9]], dtype=torch.int32
        ),
    )

    assert calls == [(9, 1)]


def test_update_states_after_model_execute_keeps_compact_state_selector() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.speculative_config = object()
    runner.model_config = SimpleNamespace(is_hybrid=True)
    runner.cache_config = SimpleNamespace(mamba_cache_mode="none")
    runner.num_accepted_tokens = SimpleNamespace(
        gpu=torch.ones(1, dtype=torch.int32)
    )
    runner.spec_state_slot_selectors = SimpleNamespace(
        gpu=torch.ones(1, dtype=torch.int32)
    )
    runner.input_batch = SimpleNamespace(
        num_accepted_tokens_cpu_tensor=torch.ones(1, dtype=torch.int32),
        spec_num_accepted_tokens_cpu_tensor=torch.ones(1, dtype=torch.int32),
    )
    runner.num_accepted_tokens_event = SimpleNamespace(record=lambda: None)

    runner._update_states_after_model_execute(
        torch.tensor([[12, 99]], dtype=torch.int32),
        _scheduler_output(_branched_payload()),
        torch.tensor([[0, 3]], dtype=torch.int32),
    )

    assert runner.num_accepted_tokens.gpu.tolist() == [2]
    assert runner.spec_state_slot_selectors.gpu.tolist() == [4]
    assert runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [2]
    assert runner.input_batch.spec_num_accepted_tokens_cpu_tensor.tolist() == [4]


def test_compact_ddtree_mamba_state_defers_to_fused_align_postprocess() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
    runner._ddtree_scheduled_payloads_require_hybrid_tree_state = lambda _: True

    def fail_slot_copy(*args, **kwargs):
        raise AssertionError("align mode should not run legacy mamba compact")

    runner._ddtree_accepted_state_slot_copies = fail_slot_copy
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[31, 99]], dtype=torch.int32),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor([[0, 3]], dtype=torch.int32),
    )

    assert not runner._compact_ddtree_accepted_mamba_state(
        sampler_output,
        _scheduler_output(_branched_payload()),
    )


def test_compact_ddtree_drafter_context_moves_branch_to_prefix() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["r0"], num_reqs=1)
    runner.input_ids = SimpleNamespace(
        gpu=torch.tensor([10, 11, 21, 31, 12, 22], dtype=torch.int32)
    )
    hidden_states = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    aux_hidden = [torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)]
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[12, 22, 99]], dtype=torch.int32),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor([[0, 4, 5]], dtype=torch.int32),
    )
    payload = DDTreeDraftPayload(
        tree_token_ids=(11, 21, 31, 12, 22),
        parent_indices=(-1, 0, 1, -1, 3),
        node_depths=(1, 2, 3, 1, 2),
        node_scores=(0.0, 0.0, 0.0, -0.1, -0.2),
        top1_chain_token_ids=(11, 21, 31),
        flat_draft_token_ids=(11, 21, 31),
        budget=5,
        top_k=2,
        chain_seed=True,
    )
    scheduler_output = _scheduler_output(payload)
    scheduler_output.num_scheduled_tokens = {"r0": 6}
    scheduler_output.total_num_scheduled_tokens = 6
    scheduler_output.scheduled_spec_decode_tokens = {
        "r0": list(payload.tree_token_ids)
    }

    runner._compact_ddtree_drafter_context(
        hidden_states,
        aux_hidden,
        sampler_output,
        scheduler_output,
    )

    assert hidden_states[1].tolist() == [8.0, 9.0]
    assert hidden_states[2].tolist() == [10.0, 11.0]
    assert aux_hidden[0][1].tolist() == [12.0, 13.0, 14.0]
    assert aux_hidden[0][2].tolist() == [15.0, 16.0, 17.0]
    assert runner.input_ids.gpu.tolist() == [10, 12, 22, 31, 12, 22]


def test_validate_ddtree_hybrid_state_path_allows_flat_chain() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(is_hybrid=True)
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[11, 21, 99]], dtype=torch.int32),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor([[0, 1, 2]], dtype=torch.int32),
    )

    runner._validate_ddtree_hybrid_state_path(
        sampler_output,
        _scheduler_output(_flat_payload()),
    )


def test_validate_ddtree_hybrid_state_path_rejects_branch_payload() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(is_hybrid=True)
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[12, 21, 99]], dtype=torch.int32),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor([[0, 4, 5]], dtype=torch.int32),
    )

    try:
        runner._validate_ddtree_hybrid_state_path(
            sampler_output,
            _scheduler_output(_branched_payload()),
        )
    except RuntimeError as exc:
        assert "tree-aware GDN/Mamba" in str(exc)
        assert "branched" in str(exc)
    else:
        raise AssertionError("expected hybrid DDTree branch guard to raise")


def test_copy_attention_kv_slot_copies_both_kv_planes() -> None:
    kv_cache = torch.arange(2 * 2 * 4 * 1 * 1, dtype=torch.float32).reshape(
        2,
        2,
        4,
        1,
        1,
    )
    src = kv_cache[1, :, 2].clone()

    GPUModelRunner._copy_attention_kv_slot(
        kv_cache,
        src_slot=6,
        dst_slot=1,
        block_size=4,
    )

    assert torch.equal(kv_cache[0, :, 1], src)


def test_copy_attention_kv_slot_copies_kv_first_layout() -> None:
    kv_cache = torch.arange(2 * 3 * 4 * 1 * 1, dtype=torch.float32).reshape(
        2,
        3,
        4,
        1,
        1,
    )
    src = kv_cache[:, 1, 2].clone()

    GPUModelRunner._copy_attention_kv_slot(
        kv_cache,
        src_slot=6,
        dst_slot=1,
        block_size=4,
    )

    assert torch.equal(kv_cache[:, 0, 1], src)


def test_compact_ddtree_attention_kv_visits_shared_storage_groups() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["r0"], num_reqs=1)
    kv_cache = torch.arange(2 * 2 * 4 * 1 * 1, dtype=torch.float32).reshape(
        2,
        2,
        4,
        1,
        1,
    )
    group0_dst = kv_cache[0, :, 1].clone()
    group1_dst = kv_cache[1, :, 1].clone()
    group0_src = kv_cache[0, :, 3].clone()
    group1_src = kv_cache[1, :, 3].clone()

    spec = AttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=spec, layer_names=["attn0"]),
            SimpleNamespace(kv_cache_spec=spec, layer_names=["attn1"]),
        ]
    )
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "attn0": SimpleNamespace(kv_cache=kv_cache),
            "attn1": SimpleNamespace(kv_cache=kv_cache),
        }
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[31, 99]], dtype=torch.int32),
        logprobs_tensors=None,
        ddtree_accepted_node_indices=torch.tensor([[0, 3]], dtype=torch.int32),
    )
    scheduler_output = _scheduler_output(_flat_payload())
    scheduler_output.num_scheduled_tokens = {"r0": 4}
    scheduler_output.total_num_scheduled_tokens = 4
    scheduler_output.scheduled_spec_decode_tokens = {"r0": [11, 21, 31]}
    slot_mappings_by_group = {
        0: torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        1: torch.tensor([4, 5, 6, 7], dtype=torch.int64),
    }

    runner._compact_ddtree_accepted_attention_kv(
        sampler_output,
        scheduler_output,
        slot_mappings_by_group,
    )

    assert not torch.equal(group0_dst, group0_src)
    assert not torch.equal(group1_dst, group1_src)
    assert torch.equal(kv_cache[0, :, 1], group0_src)
    assert torch.equal(kv_cache[1, :, 1], group1_src)
