# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tracked writes, collision overflow and rebinding preserve live indexer keys."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_mqa_paged
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer_staging import (
    QSAIndexerPersistentLayer,
    QSAIndexerStagingWorkspace,
)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("c", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("capacity", [2, 4, 16])
@pytest.mark.parametrize("page", [32, 204])
def test_versioned_keys_track_writes_rebind_and_changed_graph(c, capacity, page):
    owner = QSAIndexerStagingWorkspace(tokens=32)
    torch.manual_seed(735 + c + capacity)
    data = (torch.randn(32, page, 1, 128, dtype=torch.float16) * 0.125).pin_memory()
    cache = get_accelerator_view_from_cpu_tensor(data)
    state = QSAIndexerPersistentLayer(owner, tokens=capacity * page)
    state.bind(cache)
    table = torch.tensor(
        [[0, 1, 2, 3 + i] for i in range(c)], device="cuda", dtype=torch.int32
    )
    q = torch.randn(c * 5, 4, 128, device="cuda", dtype=torch.float16) * 0.125
    req = torch.arange(c, device="cuda", dtype=torch.int32).repeat_interleave(5)
    pos = torch.full((c * 5,), page * 16 - 1, device="cuda", dtype=torch.int64)
    lens = torch.full((c,), page * 16, device="cuda", dtype=torch.int32)
    starts = torch.arange(0, c * 5 + 1, 5, dtype=torch.int32)
    writes = torch.full((c * 5,), -1, device="cuda", dtype=torch.int64)

    def run(owner):
        if owner is None:
            return qsa_mqa_paged(
                q,
                cache,
                table,
                req,
                pos,
                lens,
                4,
                shared_key_scoring=True,
                indexer_staging_workspace=None,
                shared_key_query_start_loc_cpu=starts,
            )
        return qsa_mqa_paged(
            q,
            cache,
            table,
            req,
            pos,
            lens,
            4,
            shared_key_scoring=True,
            indexer_staging_workspace=owner,
            shared_key_query_start_loc_cpu=starts,
        )

    def check(output):
        gold = run(None)
        torch.accelerator.synchronize()
        assert torch.equal(output[1], gold[1])
        for row, n in enumerate(gold[1].tolist()):
            if n > 0:
                assert torch.equal(output[0][row, :n], gold[0][row, :n]), (
                    c,
                    capacity,
                    row,
                )

    output = run(state)
    check(output)
    output = run(state)
    check(output)
    # Collision winners may change between launches; stable, distinct leases
    # must reuse all unchanged pages without copying.
    if capacity >= c + 3:
        assert int(state.copied.sum().item()) == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.touch(writes)
        output = run(state)
    for turn in range(6):
        torch.accelerator.synchronize()
        q.mul_(0.875)
        writes.fill_(-1)
        if turn == 0:
            data[1].mul_(0.5)
            writes[0] = page
        elif turn == 1:
            # Tentative overwrite then a later rewrite under the same physical ID.
            data[1].add_(0.03125)
            writes[0] = page
        elif turn == 2:
            data[1].sub_(0.03125)
            writes[0] = page
        elif turn == 3:
            table[:, -1].copy_(
                torch.arange(12, 12 + c, device="cuda", dtype=torch.int32)
            )
            lens[0] = page * 4
            pos.sub_(8)
        elif turn == 4:
            table.copy_(table.roll(1, 0))
            req[-1] = -1
        else:
            data[1].mul_(1.125)
            writes[0] = page
        graph.replay()
        check(output)
    # Write while no selection occurs; the next selection must observe it.
    torch.accelerator.synchronize()
    data[1].add_(0.0625)
    writes.fill_(-1)
    writes[0] = page
    state.touch(writes)
    check(run(state))
    # Binding a replacement storage is an initialization/recapture boundary.
    del graph
    data2 = (data.clone() + 0.015625).pin_memory()
    newcache = get_accelerator_view_from_cpu_tensor(data2)
    state.bind(newcache)
    # A caller still holding another generation uses protected full refresh.
    data[1].add_(0.0625)
    assert len(state.prepare(cache, table, req, pos, lens, 4, page * 4)) == 2
    check(run(state))
    cache = newcache
    check(run(state))
    # Reusing an address is still a new cache generation.
    data2.add_(0.03125)
    state.bind(cache)
    check(run(state))
    rebound_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(rebound_graph):
        state.touch(writes)
        rebound_output = run(state)
    data2[1].mul_(0.75)
    writes.fill_(-1)
    writes[0] = page
    rebound_graph.replay()
    check(rebound_output)
    del rebound_graph
    other = QSAIndexerPersistentLayer(
        QSAIndexerStagingWorkspace(tokens=32), tokens=capacity * page
    )
    other.bind(cache)
    assert other.bank.data_ptr() != state.bank.data_ptr()
    check(run(other))
    other.close()
    state.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.bind(cache)
    owner.close()
