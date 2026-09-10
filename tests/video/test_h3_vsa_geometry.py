# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H3 geometry, selection and learned-gate checks without model weights."""

import pytest
import torch

from vllm.model_executor.models.minimax_h3 import vsa


@pytest.mark.parametrize("prefix", [(1, 65, 2), (97, 414), ()])
@pytest.mark.parametrize("shape", [(1, 1, 1), (5, 6, 7), (4, 4, 8)])
def test_geometry_round_trip_and_segment_boundaries(prefix, shape):
    partition, sizes, non_pad, untile, prefix_blocks, video_blocks = (
        vsa._get_h3_tile_metadata(prefix, shape, torch.device("cpu"))
    )
    source = torch.arange(sum(prefix) + shape[0] * shape[1] * shape[2])
    tiled = torch.full((sizes.numel() * 64,), -1)
    tiled[non_pad] = source[partition]
    torch.testing.assert_close(tiled[untile], source)
    assert (sizes > 0).all() and (sizes <= 64).all()
    assert sizes.sum() == source.numel()
    assert prefix_blocks + video_blocks == sizes.numel()
    boundaries = torch.tensor(prefix).cumsum(0).tolist()
    for block in range(prefix_blocks):
        rows = tiled[block * 64 : block * 64 + sizes[block]].tolist()
        assert all(not rows[0] < end <= rows[-1] for end in boundaries)
        assert max(rows) < sum(prefix)
    for block in range(prefix_blocks, sizes.numel()):
        rows = tiled[block * 64 : block * 64 + sizes[block]] - sum(prefix)
        coords = torch.stack(
            (
                rows // (shape[1] * shape[2]),
                rows // shape[2] % shape[1],
                rows % shape[2],
            ),
            dim=1,
        )
        assert torch.equal(coords[0] // 4, (coords // 4).amin(0))
        assert torch.equal(coords[0] // 4, (coords // 4).amax(0))


def test_prefix_scores_do_not_consume_video_topk_budget():
    scores = torch.zeros(1, 2, 5, 5)
    scores[..., :2] = 100000
    scores[..., 4] = 1
    mask = vsa._build_h3_block_map(scores, 2, 3, 1)
    assert mask[:, :, :2].all()
    assert mask[..., :2].all()
    assert mask[:, :, 2:, 4].all()
    assert not mask[:, :, 2:, 2:4].any()
    assert vsa._build_h3_block_map(scores, 2, 3, 99).all()


@pytest.mark.parametrize("gate_value", [0.0, 2.0])
def test_learned_compression_and_exact_sparse_work(gate_value, monkeypatch):
    def selected_blocks(q, k, v, block_map, block_sizes, *, scale):
        # Check the sparse operator receives holes as zeros, not repeated tokens.
        for i, size in enumerate(block_sizes):
            for operand in (q, k, v):
                assert not operand[:, i * 64 + size : (i + 1) * 64].count_nonzero()
        return torch.ones_like(q)

    monkeypatch.setattr(vsa, "block_sparse_attention", selected_blocks)
    x = torch.zeros(1, 8, 2, 128, dtype=torch.float16)
    output, work = vsa.h3_vsa_attention(
        x,
        x,
        torch.full_like(x, 3),
        prefix_segments=(1, 2),
        video_shape=(1, 1, 5),
        gate_compress=torch.full_like(x, gate_value),
        topk=1,
        scale=128**-0.5,
    )
    torch.testing.assert_close(output, torch.full_like(x, 1 + 3 * gate_value))
    assert work["prefix_blocks"] == 2 and work["video_blocks"] == 2
    assert work["compression_flops"] == 4 * 2 * 4**2 * 128
    # Two prefix queries per head see all four blocks; two video queries see
    # both prefix blocks and exactly one video block. Token counts use real
    # lengths even if tied top-k scores choose either video tile.
    assert work["selected_blocks"] == 2 * (2 * 4 + 2 * 3)
    assert work["selected_token_pairs"] in (2 * (3 * 8 + 5 * 4), 2 * (3 * 8 + 5 * 7))


def test_missing_gate_is_rejected_before_sparse_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid geometry must not launch a sparse kernel")

    monkeypatch.setattr(vsa, "block_sparse_attention", forbidden)
    x = torch.zeros(1, 1, 2, 128, dtype=torch.float16)
    with pytest.raises(ValueError, match="learned compression gate"):
        vsa.h3_vsa_attention(
            x,
            x,
            x,
            prefix_segments=(),
            video_shape=(1, 1, 1),
            gate_compress=None,
            topk=1,
            scale=128**-0.5,
        )
