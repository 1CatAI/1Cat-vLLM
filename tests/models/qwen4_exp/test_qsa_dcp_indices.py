# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

pytestmark = pytest.mark.skip_global_cleanup
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(indices, world, rank, interleave, page):
    result = torch.full_like(indices, -1)
    for row, selection in enumerate(indices.tolist()):
        ids = []
        for token in selection:
            if token < 0:
                continue
            # Independently reproduce the slot mapper's virtual-page formula.
            block, offset = divmod(token, page * world)
            stripe, within = divmod(offset, interleave)
            if stripe % world == rank:
                ids.append(block * page + stripe // world * interleave + within)
        result[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return result


@pytest.mark.parametrize("world,rank", [(0, 0), (2, -1), (2, 2)])
def test_invalid_rank(world, rank):
    indices = torch.zeros((2, 3), dtype=torch.int32)
    with pytest.raises(ValueError, match="rank"):
        qsa_localize_dcp_indices(
            indices,
            torch.empty_like(indices),
            dcp_world_size=world,
            dcp_rank=rank,
            interleave_size=1,
            local_block_size=16,
        )


def test_selection_alias_is_rejected_even_for_dcp1():
    indices = torch.zeros((2, 3), dtype=torch.int32)
    for world in (1, 2):
        with pytest.raises(ValueError, match="alias"):
            qsa_localize_dcp_indices(
                indices,
                indices.view_as(indices),
                dcp_world_size=world,
                dcp_rank=0,
                interleave_size=1,
                local_block_size=16,
            )


@pytest.mark.parametrize("interleave,page", [(0, 16), (3, 16), (1, 0)])
def test_nondivisible_page_is_rejected(interleave, page):
    indices = torch.zeros((2, 3), dtype=torch.int32)
    with pytest.raises(ValueError, match="interleave"):
        qsa_localize_dcp_indices(
            indices,
            torch.empty_like(indices),
            dcp_world_size=2,
            dcp_rank=0,
            interleave_size=interleave,
            local_block_size=page,
        )


@gpu
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4, 16])
@pytest.mark.parametrize("width", [3, 2051])
def test_localization_matches_slot_mapper_and_preserves_mtp(world, interleave, width):
    page = 32
    generator = torch.Generator().manual_seed(42)
    indices = torch.randint(
        -1, 8 * page * world, (5, width), dtype=torch.int32, generator=generator
    )
    indices[0].fill_(-1)
    # Include an empty owner, cross-page IDs, and a valid final column.
    indices[1].fill_(0)
    indices[2, :3] = torch.tensor([page * world - 1, page * world, page * world + 1])
    source = indices.cuda()
    # Rows have padding: the kernel must respect both input and output strides.
    backing = torch.full((5, width + 7), 123, device="cuda", dtype=torch.int32)
    output = backing[:, :width]
    for rank in range(world):
        expected = _reference(indices, world, rank, interleave, page)
        for _ in range(3):
            qsa_localize_dcp_indices(
                source,
                output,
                dcp_world_size=world,
                dcp_rank=rank,
                interleave_size=interleave,
                local_block_size=page,
            )
            assert torch.equal(output.cpu(), expected)
            assert torch.equal(source.cpu(), indices)
        assert torch.all(backing[:, width:] == 123)


@gpu
def test_graph_replay_overwrites_old_indices_and_keeps_original_selection():
    host = torch.tensor([[0, 1, 63, 64, 65], [2, 4, 6, -1, -1]], dtype=torch.int32)
    source = host.cuda()
    output = torch.empty_like(source)

    def run():
        return qsa_localize_dcp_indices(
            source,
            output,
            dcp_world_size=2,
            dcp_rank=1,
            interleave_size=1,
            local_block_size=32,
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    pointer = output.data_ptr()
    for updated in (host, torch.full_like(host, -1), host + 100):
        source.copy_(updated)
        graph.replay()
        assert output.data_ptr() == pointer
        assert torch.equal(output.cpu(), _reference(updated, 2, 1, 1, 32))
        assert torch.equal(source.cpu(), updated)
