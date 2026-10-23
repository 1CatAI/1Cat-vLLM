# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check byte order and shard/page boundaries in the shipped CPU gather."""

import mmap

import pytest
import torch

import vllm._custom_ops  # noqa: F401 - load the shipped CPU registration

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops._C, "ple_disk_gather_u8"),
    reason="requires the native CPU row gather",
)


@pytest.mark.parametrize("m", [1, 5, 17, 33])
@pytest.mark.parametrize("row_bytes", [17, 160, 8193])
def test_mapped_shards_preserve_bytes_and_repeated_ids(tmp_path, m, row_bytes):
    shard_size, total = 97, 271
    references, mapped, tensors = [], [], []
    try:
        for shard, count in enumerate((97, 97, 77)):
            values = (
                (torch.arange(count * row_bytes, dtype=torch.int64) * 11 + shard) % 256
            ).to(torch.uint8)
            path = tmp_path / f"shard-{shard}"
            path.write_bytes(values.numpy().tobytes())
            with path.open("rb") as source:
                region = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY)
            mapped.append(region)
            tensor = torch.frombuffer(region, dtype=torch.uint8).reshape(
                count, row_bytes
            )
            tensors.append(tensor)
            references.append(values.reshape(count, row_bytes))
        pointers = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64)
        ids = ((torch.arange(m * 16) * 41) % total).reshape(m, 16)
        ids.reshape(-1)[:4] = torch.tensor([0, 96, 97, total - 1])
        output = torch.empty((m, 16, row_bytes), dtype=torch.uint8)
        torch.ops._C.ple_disk_gather_u8(
            ids, pointers, shard_size, total, row_bytes, output
        )
        expected = torch.cat(references)[ids.reshape(-1)].reshape_as(output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        # Reuse the same buffers with different IDs; no stale cache/generation.
        ids.fill_(total - 1)
        torch.ops._C.ple_disk_gather_u8(
            ids, pointers, shard_size, total, row_bytes, output
        )
        torch.testing.assert_close(
            output, references[-1][-1].expand_as(output), rtol=0, atol=0
        )
    finally:
        tensors.clear()
        # Torch may retain the last local view until the function returns.
        if "tensor" in locals():
            del tensor
        for region in mapped:
            region.close()


def test_invalid_rows_are_rejected_before_output_writes():
    table = torch.arange(48, dtype=torch.uint8).reshape(3, 16)
    pointers = torch.tensor([table.data_ptr()], dtype=torch.int64)
    output = torch.full((2, 16), 123, dtype=torch.uint8)
    for invalid in (-1, 3):
        with pytest.raises(IndexError, match="row id out of range"):
            torch.ops._C.ple_disk_gather_u8(
                torch.tensor([0, invalid]), pointers, 3, 3, 16, output
            )
        assert torch.all(output == 123)


def test_strided_ids_do_not_silently_read_adjacent_storage():
    table = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
    pointers = torch.tensor([table.data_ptr()], dtype=torch.int64)
    ids = torch.arange(4)[::2]
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops._C.ple_disk_gather_u8(
            ids, pointers, 4, 4, 16, torch.empty((2, 16), dtype=torch.uint8)
        )


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "ple_disk_cached_gather_u8"),
    reason="requires bounded native cache",
)
def test_row_cache_preserves_order_under_collisions_and_reuse():
    table = (torch.arange(9 * 17) % 256).to(torch.uint8).reshape(9, 17)
    pointers = torch.tensor([table.data_ptr()], dtype=torch.int64)
    keys = torch.full((2,), -1, dtype=torch.int64)
    rows = torch.empty((2, 17), dtype=torch.uint8)
    for selected in ([0, 2, 0, 1, 3], [2, 3, 3, 2], [8, 2, 0, 8, 1]):
        ids = torch.tensor(selected)
        output = torch.empty((len(selected), 17), dtype=torch.uint8)
        torch.ops._C.ple_disk_cached_gather_u8(
            ids,
            pointers,
            9,
            9,
            17,
            output,
            keys,
            rows,
        )
        torch.testing.assert_close(output, table[ids], rtol=0, atol=0)
    output = torch.full((2, 17), 123, dtype=torch.uint8)
    with pytest.raises(IndexError, match="row id out of range"):
        torch.ops._C.ple_disk_cached_gather_u8(
            torch.tensor([1, 9]),
            pointers,
            9,
            9,
            17,
            output,
            keys,
            rows,
        )
    assert torch.all(output == 123)
