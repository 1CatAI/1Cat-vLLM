# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import struct
from pathlib import Path

import gguf
import pytest

from benchmarks.deepseek_v41.checkpoint_audit import (
    Tensor,
    packed_bank_bytes,
    safetensors_names,
    shard_header,
    verified_paths,
)


def expert(role: str, kind: int, shape: tuple[int, int, int]) -> Tensor:
    block, unit = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)]
    return Tensor(
        f"blk.7.ffn_{role}_exps.weight",
        shape,
        kind,
        math.prod(shape) // block * unit,
    )


@pytest.mark.parametrize(
    ("kind", "expected"), [(29, 330301496), (16, 389283906), (17, 436469834)]
)
def test_tp4_down_keeps_three_blocks_and_final_row_guard(kind, expected):
    # A 576-column logical slice needs three 256-value blocks, not 2.25.
    tensor = expert("down", kind, (2304, 5120, 384))
    result = packed_bank_bytes(tensor, 4)
    assert result == [expected] * 4
    assert sum(result) > tensor.nbytes


def test_gate_tp_splits_rows_without_expanding_source_blocks():
    tensor = expert("gate", 16, (5120, 2304, 384))
    assert packed_bank_bytes(tensor, 4) == [tensor.nbytes // 4] * 4


@pytest.mark.parametrize(
    ("role", "shape"),
    [("down", (2305, 5120, 384)), ("gate", (5120, 2303, 384))],
)
def test_incomplete_blocks_or_unequal_tp_slices_are_rejected(role, shape):
    with pytest.raises(ValueError):
        packed_bank_bytes(expert(role, 16, shape), 4)


def test_declared_byte_count_cannot_silently_change_the_memory_budget():
    tensor = Tensor("blk.7.ffn_down_exps.weight", (2304, 5120, 384), 16, 1)
    with pytest.raises(ValueError, match="byte count"):
        packed_bank_bytes(tensor, 4)


@pytest.mark.parametrize("success", ["OK", "成功"])
def test_existing_checksum_results_accept_both_locales(success):
    manifest = (
        f"{'a' * 64}  gguf/model.gguf\n{'b' * 64}  base/dspark/draft.safetensors\n"
    )
    log = (
        f"gguf/model.gguf: {success}\n"
        f"base/dspark/draft.safetensors: {success}\nEXIT=0\nEXIT=0\n"
    )
    assert verified_paths(manifest, log) == {
        "gguf/model.gguf",
        "base/dspark/draft.safetensors",
    }


@pytest.mark.parametrize(
    "log",
    ["EXIT=0\n", "gguf/model.gguf: OK\nEXIT=1\n", "gguf/model.gguf: FAILED\nEXIT=0\n"],
)
def test_exit_zero_alone_is_not_checksum_success(log):
    with pytest.raises(ValueError, match="incomplete or failed"):
        verified_paths(f"{'a' * 64}  gguf/model.gguf\n", log)


def write_safetensors(path: Path, offsets=(8, 12)):
    header = json.dumps(
        {
            "mtp.0.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "mtp.1.weight": {
                "dtype": "I8",
                "shape": [4],
                "data_offsets": list(offsets),
            },
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 12)


def test_safetensors_audit_does_not_load_tensor_payloads(tmp_path):
    path = tmp_path / "draft.safetensors"
    write_safetensors(path)
    assert set(safetensors_names(path)) == {"mtp.0.weight", "mtp.1.weight"}


@pytest.mark.parametrize("offsets", [(4, 8), (8, 40)])
def test_safetensors_overlapping_or_out_of_bounds_ranges_are_rejected(
    tmp_path, offsets
):
    path = tmp_path / "draft.safetensors"
    write_safetensors(path, offsets)
    with pytest.raises(ValueError):
        safetensors_names(path)


def write_split_header(path: Path, index: int, architecture=None):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    entries = [
        ("split.no", 2, struct.pack("<H", index)),
        ("split.count", 2, struct.pack("<H", 4)),
    ]
    if architecture is not None:
        entries.append(("general.architecture", 8, string(architecture)))
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries))
    for key, kind, value in entries:
        data += string(key) + struct.pack("<I", kind) + value
    path.write_bytes(data + b"\0" * (-len(data) % 32))


def test_nonfirst_split_can_omit_repeated_architecture_metadata(tmp_path):
    path = tmp_path / "part2.gguf"
    write_split_header(path, 1)
    header = shard_header(gguf.GGUFReader(path))
    assert header["architecture"] is None
    assert header["split_index"] == 1
    assert header["split_count"] == 4


@pytest.mark.parametrize("index,architecture", [(0, None), (4, None), (1, "qwen4")])
def test_split_metadata_cannot_hide_a_wrong_or_missing_backbone(
    tmp_path, index, architecture
):
    path = tmp_path / "invalid.gguf"
    write_split_header(path, index, architecture)
    with pytest.raises(ValueError):
        shard_header(gguf.GGUFReader(path))
