# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 VSA geometry and compression from the pinned official Omni implementation.

Source: fastvideo_vsa.py at 7be014bce6374f06c95b703763bdbac4c6198f31.
The selected blocks execute through a native SM70 sparse kernel. Geometry
caches retain indices only; pooled activations, scores and gates are per-call.
"""

import functools
import math

import torch

from vllm.model_executor.layers.sm70_sparse_attention import (
    _h3_block_sparse_attention as block_sparse_attention,
)


@functools.lru_cache(maxsize=32)
def _get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    t_size, h_size, w_size = dit_seq_shape
    tile_t, tile_h, tile_w = tile_size
    indices = torch.arange(
        t_size * h_size * w_size, device=device, dtype=torch.long
    ).reshape(t_size, h_size, w_size)
    tiles = []
    for tile_t_idx in range(math.ceil(t_size / tile_t)):
        for tile_h_idx in range(math.ceil(h_size / tile_h)):
            for tile_w_idx in range(math.ceil(w_size / tile_w)):
                tiles.append(
                    indices[
                        tile_t_idx * tile_t : min((tile_t_idx + 1) * tile_t, t_size),
                        tile_h_idx * tile_h : min((tile_h_idx + 1) * tile_h, h_size),
                        tile_w_idx * tile_w : min((tile_w_idx + 1) * tile_w, w_size),
                    ].flatten()
                )
    return torch.cat(tiles, dim=0)


@functools.lru_cache(maxsize=32)
def _construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    num_tiles = tuple(
        math.ceil(seq_dim / tile_dim)
        for seq_dim, tile_dim in zip(dit_seq_shape, tile_size)
    )

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.Tensor:
        sizes = torch.full((n_tiles,), tile, dtype=torch.int32, device=device)
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(dit_seq_shape[0], tile_size[0], num_tiles[0])
    h_sizes = _sizes(dit_seq_shape[1], tile_size[1], num_tiles[1])
    w_sizes = _sizes(dit_seq_shape[2], tile_size[2], num_tiles[2])
    return (
        t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]
    ).reshape(-1)


@functools.lru_cache(maxsize=32)
def _get_non_pad_index(
    variable_block_sizes: torch.Tensor, max_block_size: int
) -> torch.Tensor:
    num_blocks = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts = torch.arange(num_blocks, device=device) * max_block_size
    padded_index = (
        starts[:, None] + torch.arange(max_block_size, device=device)[None, :]
    )
    valid = (
        torch.arange(max_block_size, device=device)[None, :]
        < variable_block_sizes[:, None]
    )
    return padded_index[valid]


@functools.lru_cache(maxsize=32)
def _get_h3_tile_metadata(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Official FastVideo H3 geometry: pure prefix chunks + 3-D video tiles."""
    block_size = (4, 4, 4)
    block_elements = 64
    prefix_len = sum(prefix_segments)
    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, block_elements)
        prefix_sizes.extend([block_elements] * full)
        if remainder:
            prefix_sizes.append(remainder)

    video_indices = (
        _get_tile_partition_indices(video_shape, block_size, device) + prefix_len
    )
    video_sizes = _construct_variable_block_sizes(video_shape, block_size, device)
    partition = torch.cat(
        [torch.arange(prefix_len, device=device, dtype=torch.long), video_indices]
    )
    sizes = torch.cat(
        [
            torch.tensor(prefix_sizes, device=device, dtype=torch.int32),
            video_sizes.to(torch.int32),
        ]
    )
    non_pad = _get_non_pad_index(sizes, block_elements)
    untile = non_pad[torch.argsort(partition)]
    total = prefix_len + math.prod(video_shape)
    if int(sizes.sum()) != total or untile.numel() != total:
        raise ValueError(
            f"invalid H3 VSA geometry: prefix={prefix_segments}, video={video_shape}, "
            f"sizes_sum={int(sizes.sum())}, total={total}"
        )
    return (
        partition,
        sizes,
        non_pad,
        untile,
        len(prefix_sizes),
        int(video_sizes.numel()),
    )


def _pool_h3_tiles(x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    batch, seq_len, heads, dim = x.shape
    blocks = seq_len // 64
    pooled = x.view(batch, blocks, 64, heads, dim).sum(dim=2, dtype=torch.float32)
    pooled = pooled / sizes.view(1, -1, 1, 1).clamp_min(1)
    return pooled.permute(0, 2, 1, 3)


def _build_h3_block_map(
    scores: torch.Tensor,
    num_prefix_blocks: int,
    num_video_blocks: int,
    topk: int,
) -> torch.Tensor:
    """Prefix K/V are exempt and prefix queries stay dense, as in FastVideo."""
    keep_video = min(topk, num_video_blocks)
    if keep_video == num_video_blocks:
        return torch.ones_like(scores, dtype=torch.bool)
    block_map = torch.zeros_like(scores, dtype=torch.bool)
    indices = (
        scores[..., num_prefix_blocks:].topk(keep_video, dim=-1).indices
        + num_prefix_blocks
    )
    block_map.scatter_(-1, indices, True)
    block_map[..., :num_prefix_blocks] = True
    block_map[:, :, :num_prefix_blocks, :] = True
    return block_map


def h3_vsa_attention(
    q, k, v, *, prefix_segments, video_shape, gate_compress, topk, scale
):
    """Return sparse output and exact useful pair/block counts.

    Prefix queries stay dense. Video queries select all prefix blocks plus
    top-k video blocks, then add the official learned compressed contribution.
    """
    if (
        len(video_shape) != 3
        or any(n <= 0 for n in video_shape)
        or any(n <= 0 for n in prefix_segments)
        or isinstance(topk, bool)
        or not isinstance(topk, int)
        or topk <= 0
    ):
        raise ValueError("VSA needs positive segment/grid dimensions and topk")
    expected = sum(prefix_segments) + math.prod(video_shape)
    if (
        q.ndim != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or q.shape[1] != expected
        or q.shape[3] != 128
        or q.dtype != torch.float16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or k.device != q.device
        or v.device != q.device
    ):
        raise ValueError("VSA operands must match FP16 [B,valid_rows,H,128] geometry")
    if (
        gate_compress is None
        or gate_compress.shape != q.shape
        or gate_compress.dtype != q.dtype
        or gate_compress.device != q.device
    ):
        raise ValueError("FastH3 VSA requires its matching learned compression gate")
    if not math.isclose(scale, 128**-0.5, rel_tol=0, abs_tol=1e-6):
        raise ValueError("FastH3 VSA requires the official head-dimension scale")
    partition, sizes, non_pad, untile, prefix_blocks, video_blocks = (
        _get_h3_tile_metadata(tuple(prefix_segments), tuple(video_shape), q.device)
    )
    blocks = sizes.numel()
    shape = (q.shape[0], blocks * 64, q.shape[2], q.shape[3])
    tiled = []
    for operand in (q, k, v):
        target = torch.zeros(shape, device=q.device, dtype=q.dtype)
        target[:, non_pad] = operand[:, partition]
        tiled.append(target)
    q_tiled, k_tiled, v_tiled = tiled
    q_pool, k_pool = (_pool_h3_tiles(x, sizes) for x in (q_tiled, k_tiled))
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * scale
    block_map = _build_h3_block_map(scores, prefix_blocks, video_blocks, topk)
    output = block_sparse_attention(
        q_tiled, k_tiled, v_tiled, block_map, sizes, scale=scale
    )
    gate_tiled = torch.zeros_like(q_tiled)
    gate_tiled[:, non_pad] = gate_compress[:, partition]
    v_pool = _pool_h3_tiles(v_tiled, sizes)
    compressed = torch.matmul(torch.softmax(scores, dim=-1), v_pool)
    compressed = compressed.permute(0, 2, 1, 3).to(output.dtype)
    output = (
        output.view(q.shape[0], blocks, 64, q.shape[2], q.shape[3])
        + compressed.unsqueeze(2)
        * gate_tiled.view(q.shape[0], blocks, 64, q.shape[2], q.shape[3])
    ).view_as(output)
    # Keep dynamic counts on the device until complete denoise accounting.
    # Padding and unselected blocks never contribute useful model FLOPs.
    pair_sizes = sizes.to(torch.int64)[:, None] * sizes.to(torch.int64)[None, :]
    work = {
        "dense_token_pairs": q.shape[0] * q.shape[2] * q.shape[1] ** 2,
        "selected_token_pairs": (block_map * pair_sizes).sum(),
        "selected_blocks": block_map.sum(),
        "compression_flops": 4 * q.shape[0] * q.shape[2] * blocks**2 * q.shape[3],
        "prefix_blocks": prefix_blocks,
        "video_blocks": video_blocks,
        "heads": q.shape[2],
        "head_size": q.shape[3],
    }
    return output[:, untile].contiguous(), work
