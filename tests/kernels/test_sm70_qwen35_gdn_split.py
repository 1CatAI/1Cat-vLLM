# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _sm70_pack_qwen_gdn_qkv,
)
from vllm.model_executor.models.qwen3_5 import (
    _sm70_materialize_qwen35_gdn_splits,
)


@pytest.mark.parametrize("num_rows", [1, 8, 32])
def test_qwen35_gdn_split_materialization_is_bitwise_exact(num_rows: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Qwen3.5 GDN split kernel")

    torch.manual_seed(7)
    qkv_size = 2560
    z_size = 1536
    ba_size = 12

    # Slice wider allocations so the source row strides are deliberately
    # larger than their logical widths, matching the projection-view contract.
    qkvz_storage = torch.randn(
        (num_rows, qkv_size + z_size + 7),
        dtype=torch.float16,
        device="cuda",
    )
    ba_storage = torch.randn(
        (num_rows, 2 * ba_size + 5),
        dtype=torch.float16,
        device="cuda",
    )
    mixed_qkvz = qkvz_storage[:, : qkv_size + z_size]
    mixed_ba = ba_storage[:, : 2 * ba_size]

    actual_z, actual_b, actual_a = _sm70_materialize_qwen35_gdn_splits(
        mixed_qkvz,
        mixed_ba,
        qkv_size,
        z_size,
        ba_size,
    )

    expected_z = mixed_qkvz[:, qkv_size : qkv_size + z_size].contiguous()
    expected_b = mixed_ba[:, :ba_size].contiguous()
    expected_a = mixed_ba[:, ba_size : 2 * ba_size].contiguous()
    assert torch.equal(actual_z, expected_z)
    assert torch.equal(actual_b, expected_b)
    assert torch.equal(actual_a, expected_a)


def test_qwen35_gdn_split_graph_replay_reads_current_projection_values():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Qwen3.5 GDN split kernel")

    num_rows, qkv_size, z_size, ba_size = 8, 2560, 1536, 12
    mixed_qkvz = torch.randn(
        (num_rows, qkv_size + z_size), dtype=torch.float16, device="cuda"
    )
    mixed_ba = torch.randn((num_rows, 2 * ba_size), dtype=torch.float16, device="cuda")
    _sm70_materialize_qwen35_gdn_splits(mixed_qkvz, mixed_ba, qkv_size, z_size, ba_size)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_z, actual_b, actual_a = _sm70_materialize_qwen35_gdn_splits(
            mixed_qkvz, mixed_ba, qkv_size, z_size, ba_size
        )

    mixed_qkvz.copy_(torch.randn_like(mixed_qkvz))
    mixed_ba.copy_(torch.randn_like(mixed_ba))
    graph.replay()
    torch.accelerator.synchronize()

    assert torch.equal(actual_z, mixed_qkvz[:, qkv_size:].contiguous())
    assert torch.equal(actual_b, mixed_ba[:, :ba_size].contiguous())
    assert torch.equal(actual_a, mixed_ba[:, ba_size:].contiguous())


@pytest.mark.parametrize("compiled", [False, True])
def test_qwen35_combined_split_preserves_tails_across_graph_replay(compiled):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Qwen3.5 GDN split kernel")

    # Real TP4 NVFP4 geometry: 4120 logical columns in a 4128-column row.
    # Both input arguments alias this allocation, and b/a start at an offset.
    storage = torch.empty((8, 4128), dtype=torch.float16, device="cuda")
    projection = storage[:, :4120]

    def split(values):
        return _sm70_materialize_qwen35_gdn_splits(
            values, values[:, 4096:4120], 2560, 1536, 12
        )

    if compiled:
        split = torch.compile(split, fullgraph=True)
    storage.normal_()
    split(projection)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = split(projection)
        # The consumer may overwrite QKV in the same graph. Tail outputs must
        # be independent snapshots, including the two small gating vectors.
        projection[:, :2560].zero_()

    for seed in (41, 42, 43):
        torch.manual_seed(seed)
        storage.normal_()
        expected = [
            projection[:, start:end].clone()
            for start, end in ((2560, 4096), (4096, 4108), (4108, 4120))
        ]
        padding = storage[:, 4120:].clone()
        graph.replay()
        torch.cuda.synchronize()
        for result, reference in zip(actual, expected):
            assert result.is_contiguous()
            assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
        assert torch.count_nonzero(projection[:, :2560]) == 0
        assert torch.equal(storage[:, 4120:], padding)


@pytest.mark.parametrize("num_rows", [1, 8])
def test_qwen35_gdn_qkv_pack_is_bitwise_exact(num_rows: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Qwen3.5 GDN Q/K/V pack kernel")

    torch.manual_seed(11)
    q_dim, k_dim, v_dim = 512, 512, 1536
    storage = torch.randn(
        (num_rows, q_dim + k_dim + v_dim + 13),
        dtype=torch.float16,
        device="cuda",
    )
    mixed_qkv = storage[:, : q_dim + k_dim + v_dim]
    actual = _sm70_pack_qwen_gdn_qkv(mixed_qkv, q_dim, k_dim, v_dim)

    q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    expected = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)])
    assert torch.equal(actual, expected)


def test_qwen35_gdn_qkv_pack_graph_replay_reads_current_values():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Qwen3.5 GDN Q/K/V pack kernel")

    num_rows, q_dim, k_dim, v_dim = 8, 512, 512, 1536
    mixed_qkv = torch.randn(
        (num_rows, q_dim + k_dim + v_dim),
        dtype=torch.float16,
        device="cuda",
    )
    _sm70_pack_qwen_gdn_qkv(mixed_qkv, q_dim, k_dim, v_dim)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _sm70_pack_qwen_gdn_qkv(mixed_qkv, q_dim, k_dim, v_dim)

    mixed_qkv.copy_(torch.randn_like(mixed_qkv))
    graph.replay()
    torch.accelerator.synchronize()

    q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    expected = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)])
    assert torch.equal(actual, expected)
