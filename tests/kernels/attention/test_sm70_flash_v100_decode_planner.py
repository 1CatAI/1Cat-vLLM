# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

flash_attn_v100 = pytest.importorskip("flash_attn_v100.flash_attn_interface")


def _clear_decode_caches() -> None:
    flash_attn_v100._decode_plan_cache.clear()
    flash_attn_v100._decode_workspace_cache.clear()


def test_short_hd256_decode_default_partition_stays_256(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)

    partition = flash_attn_v100._get_decode_partition_size(
        max_seq_capacity=4096,
        head_dim=256,
        num_q_heads=4,
        num_kv_heads=1,
        max_seq_len_hint=1,
        batch_size_hint=1,
    )

    assert partition == 256


def test_long_hd256_gqa_decode_default_partition_preserves_256(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)

    partition = flash_attn_v100._get_decode_partition_size(
        max_seq_capacity=8192,
        head_dim=256,
        num_q_heads=8,
        num_kv_heads=1,
        max_seq_len_hint=4097,
        batch_size_hint=1,
    )

    assert partition == 256


def test_static_decode_plan_uses_fixed_launch_and_active_runtime(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    _clear_decode_caches()

    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((512, 16, 1, 256), dtype=torch.float16)
    block_table = torch.zeros((1, 512), dtype=torch.int32)

    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
    )

    assert plan.partition_size == 256
    assert plan.actual_num_partitions == 17
    assert plan.launch_num_partitions == 32
    assert plan.workspace_num_partitions == 32

    tmp_out, max_logits, exp_sums, active_num_partitions = (
        flash_attn_v100._get_decode_workspace_for_plan(
            q,
            batch_capacity=1,
            num_heads=8,
            head_dim=256,
            plan=plan,
        )
    )

    assert tmp_out.shape[2] >= plan.launch_num_partitions
    assert max_logits.shape[:3] == tmp_out.shape[:3]
    assert exp_sums.shape[:3] == tmp_out.shape[:3]
    assert active_num_partitions.dtype == torch.int32
    assert active_num_partitions.item() == plan.actual_num_partitions


def test_static_decode_cuda_graph_capture_uses_workspace_active(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setattr(flash_attn_v100, "_cuda_graph_capture_active", lambda: True)
    _clear_decode_caches()

    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((512, 16, 1, 256), dtype=torch.float16)
    block_table = torch.zeros((1, 512), dtype=torch.int32)

    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
    )

    assert plan.partition_size == 256
    assert plan.actual_num_partitions == 32
    assert plan.launch_num_partitions == 32
    assert plan.workspace_num_partitions == 32


def test_static_decode_cuda_graph_capture_runtime_active_keeps_short_plan(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setattr(flash_attn_v100, "_cuda_graph_capture_active", lambda: True)
    _clear_decode_caches()

    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((512, 16, 1, 256), dtype=torch.float16)
    block_table = torch.zeros((1, 512), dtype=torch.int32)
    active_num_partitions = torch.empty((1,), dtype=torch.int32)

    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
        active_num_partitions=active_num_partitions,
    )

    assert plan.partition_size == 256
    assert plan.actual_num_partitions == 17
    assert plan.launch_num_partitions == 32
    assert plan.workspace_num_partitions == 32
