# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.models.qwen4_exp.nvidia.sm70_mtp_head import shard_hot_ids


def test_shard_hot_ids_partitions_the_ranked_prefix():
    rows, world = 100, 4
    ranking = torch.randperm(rows * world, generator=torch.Generator().manual_seed(0))
    count = 70
    shards = [shard_hot_ids(ranking, count, world, r, rows) for r in range(world)]
    widths = {s.numel() for s in shards}
    assert len(widths) == 1 and widths.pop() % 32 == 0
    chosen = set()
    for rank, ids in enumerate(shards):
        assert ((ids // rows) == rank).all()
        assert (ids[1:] >= ids[:-1]).all()
        chosen |= set(ids.tolist())
    assert chosen == set(ranking[:count].tolist())


def test_shard_hot_ids_handles_an_empty_shard():
    rows, world = 64, 4
    ranking = torch.arange(10)  # every hot id lives on rank 0
    ids = shard_hot_ids(ranking, 10, world, 3, rows)
    assert ids.numel() == 32 and (ids == 3 * rows).all()
    assert shard_hot_ids(ranking, 10, world, 0, rows)[:10].tolist() == list(range(10))
