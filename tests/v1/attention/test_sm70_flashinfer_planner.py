# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm.v1.attention.backends.flashinfer_sm70_planner import (
    SM70DeviceLimits,
    SM70KernelResources,
    make_delegate_decision,
    plan_decode,
    plan_prefill,
    plan_prefill_requests,
)

LIMITS = SM70DeviceLimits()


@pytest.mark.parametrize(
    ("resources", "expected_ctas"),
    [
        (SM70KernelResources(512, 64, 41_936), 2),
        (SM70KernelResources(256, 128, 43_008), 2),
        (SM70KernelResources(256, 129, 43_008), 1),
        (SM70KernelResources(256, 186, 41_220), 1),
        (SM70KernelResources(128, 255, 49_152), 2),
    ],
)
def test_sm70_resident_cta_envelope(
    resources: SM70KernelResources,
    expected_ctas: int,
) -> None:
    assert resources.resident_ctas(LIMITS) == expected_ctas


def test_long_prefill_does_not_split_an_already_wide_grid() -> None:
    plan = plan_prefill(
        packed_qo_len=8096 * 6,
        kv_len=121_440,
        num_kv_heads=1,
        cta_tile_q=64,
        kv_tile_size=16,
        resources=SM70KernelResources(256, 128, 43_008),
    )

    assert not plan.split_kv
    assert plan.logical_ctas == 759
    assert plan.grid_capacity == 144
    assert plan.waves == pytest.approx(759 / 144)


def test_small_prefill_uses_resource_driven_split_kv() -> None:
    plan = plan_prefill(
        packed_qo_len=64,
        kv_len=65_536,
        num_kv_heads=1,
        cta_tile_q=64,
        kv_tile_size=128,
        resources=SM70KernelResources(256, 128, 43_008),
        enable_cuda_graph=True,
    )

    assert plan.split_kv
    assert plan.kv_chunk_size == 512
    assert plan.logical_ctas == 128
    assert plan.padded_ctas == 144


def test_prefill_work_rounds_q_tiles_per_request_without_split() -> None:
    work = plan_prefill_requests(
        qo_lens=[1, 33],
        kv_lens=[32, 64],
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_vo=8,
        cta_tile_q=32,
        kv_tile_size=128,
        resources=SM70KernelResources(512, 64, 41_936),
        fixed_split_size=1024,
    )

    assert work.is_exact
    assert work.packed_qo_lens == (1, 33)
    assert work.request_indices == (0, 1, 1)
    assert work.qo_tile_indices == (0, 0, 1)
    assert work.kv_tile_indices == (0, 0, 0)
    assert work.o_indptr == (0, 1, 34)
    assert work.valid_work_items == 3
    assert work.valid_ctas == 3
    assert work.padded_ctas == 3
    assert not work.split_kv
    assert work.merge_indptr == ()
    assert work.block_valid_mask == ()
    assert work.int_workspace_bytes == 64
    assert work.float_workspace_bytes == 0
    assert work.tmp_v_bytes == 0
    assert work.tmp_s_bytes == 0
    with pytest.raises(FrozenInstanceError):
        work.valid_work_items = 4  # type: ignore[misc]


def test_prefill_work_describes_split_kv_merge_and_workspace() -> None:
    work = plan_prefill_requests(
        qo_lens=[2, 1],
        kv_lens=[9, 4],
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_vo=8,
        cta_tile_q=4,
        kv_tile_size=4,
        resources=SM70KernelResources(512, 64, 41_936),
        fixed_split_size=4,
    )

    assert work.packed_qo_lens == (4, 2)
    assert work.request_indices == (0, 0, 0, 1)
    assert work.qo_tile_indices == (0, 0, 0, 0)
    assert work.kv_tile_indices == (0, 1, 2, 0)
    assert work.merge_indptr == (0, 3, 6, 7)
    assert work.o_indptr == (0, 6, 7)
    assert work.block_valid_mask == (True, True, True, True)
    assert work.split_kv
    assert work.valid_ctas == 4
    assert work.padded_ctas == 4
    assert work.merge_indptr_bytes == 16
    assert work.block_valid_mask_bytes == 4
    assert work.int_workspace_bytes == 96
    assert work.tmp_v_bytes == 1024
    assert work.tmp_s_bytes == 128
    assert work.float_workspace_bytes == 1152


def test_prefill_work_cuda_graph_pads_ctas_and_workspace() -> None:
    kwargs = {
        "qo_lens": [1, 1],
        "kv_lens": [8, 4],
        "num_qo_heads": 1,
        "num_kv_heads": 1,
        "head_dim_vo": 8,
        "cta_tile_q": 4,
        "kv_tile_size": 4,
        "resources": SM70KernelResources(512, 64, 41_936),
        "limits": SM70DeviceLimits(sm_count=2),
        "fixed_split_size": 4,
    }
    eager_work = plan_prefill_requests(**kwargs)
    graph_work = plan_prefill_requests(**kwargs, enable_cuda_graph=True)

    assert eager_work.valid_ctas == 3
    assert eager_work.padded_ctas == 3
    assert eager_work.int_workspace_bytes == 96
    assert eager_work.float_workspace_bytes == 432
    assert graph_work.valid_work_items == 3
    assert graph_work.padded_work_items == 4
    assert graph_work.valid_ctas == 3
    assert graph_work.padded_ctas == 4
    assert graph_work.block_valid_mask == (True, True, True, False)
    assert graph_work.int_workspace_bytes == 112
    assert graph_work.tmp_v_bytes == 512
    assert graph_work.tmp_s_bytes == 64
    assert graph_work.float_workspace_bytes == 576
    assert graph_work.int_workspace_bytes % 16 == 0
    assert graph_work.float_workspace_bytes % 16 == 0


def test_prefill_work_wide_graph_grid_without_kv_split_has_no_merge_workspace() -> None:
    work = plan_prefill_requests(
        qo_lens=[4, 4, 4, 4, 4],
        kv_lens=[4, 4, 4, 4, 4],
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_vo=8,
        cta_tile_q=4,
        kv_tile_size=4,
        resources=SM70KernelResources(512, 64, 41_936),
        limits=SM70DeviceLimits(sm_count=2),
        enable_cuda_graph=True,
    )

    assert work.valid_work_items == 5
    assert work.padded_work_items == 9
    assert work.valid_ctas == 5
    assert work.padded_ctas == 9
    assert not work.split_kv
    assert work.float_workspace_bytes == 0
    assert work.tmp_v_bytes == 0
    assert work.tmp_s_bytes == 0
    assert work.merge_indptr == ()
    assert work.merge_indptr_bytes == 0
    assert work.block_valid_mask == ()
    assert work.block_valid_mask_bytes == 0


def test_decode_preserves_quality_locked_p256_partitions() -> None:
    plan = plan_decode(
        seq_lens=[131_072],
        num_kv_heads=1,
        kv_tile_size=128,
        resources=SM70KernelResources(256, 186, 41_220),
        fixed_split_size=256,
    )

    assert plan.split_kv
    assert plan.kv_chunk_size == 256
    assert plan.logical_ctas == 512
    assert plan.grid_capacity == 72


def test_adaptive_decode_fills_one_resident_grid_without_oversplitting() -> None:
    plan = plan_decode(
        seq_lens=[131_072],
        num_kv_heads=1,
        kv_tile_size=128,
        resources=SM70KernelResources(256, 186, 41_220),
        enable_cuda_graph=True,
    )

    assert plan.split_kv
    assert plan.logical_ctas <= plan.grid_capacity
    assert plan.logical_ctas == 69
    assert plan.kv_chunk_size == 1_920
    assert plan.padded_ctas == 72


def test_invalid_kernel_resource_claim_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[1, 255\]"):
        SM70KernelResources(256, 256, 0).resident_ctas(LIMITS)


def test_unpromoted_prefill_decision_separates_candidate_from_delegate() -> None:
    plan = plan_prefill(
        packed_qo_len=384,
        kv_len=1024,
        num_kv_heads=1,
        cta_tile_q=32,
        kv_tile_size=128,
        resources=SM70KernelResources(512, 64, 41_936),
    )

    decision = make_delegate_decision(
        stage="build",
        workload="prefill",
        plan=plan,
    )

    assert decision.dispatch == "delegate"
    assert decision.delegate_backend == "FLASH_ATTN_V100"
    assert decision.planner_candidate == "accepted_bm32_resource_envelope"
    assert decision.delegate_dispatch == "flash_v100_runtime_dispatch"
    assert not decision.kernel_promoted
    assert decision.route_proof == {
        "selected_backend": "FLASHINFER_SM70",
        "planner_stage": "build",
        "workload": "prefill",
        "planner_candidate": "accepted_bm32_resource_envelope",
        "dispatch": "delegate",
        "delegate_backend": "FLASH_ATTN_V100",
        "delegate_dispatch": "flash_v100_runtime_dispatch",
        "kernel_promoted": False,
        "plan": {
            "split_kv": plan.split_kv,
            "kv_chunk_size": plan.kv_chunk_size,
            "logical_ctas": plan.logical_ctas,
            "padded_ctas": plan.padded_ctas,
            "resident_ctas_per_sm": plan.resident_ctas_per_sm,
            "grid_capacity": plan.grid_capacity,
        },
    }
