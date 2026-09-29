# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap

import pytest
import torch

import vllm.models.qwen4_exp.nvidia.ple_layer as ple_module


def test_ple_disk_shard_guard_rejects_shared_anonymous_memory(tmp_path):
    # mmap.mmap(-1, n) is shared anonymous memory; /proc/self/maps lists it as
    # "/dev/zero (deleted)", which must not pass for a file on disk.
    anonymous = mmap.mmap(-1, mmap.PAGESIZE)
    with pytest.raises(RuntimeError, match="file-backed"):
        ple_module._advise_random_file_access(
            torch.frombuffer(anonymous, dtype=torch.uint8)
        )

    path = tmp_path / "shard.bin"
    path.write_bytes(bytes(mmap.PAGESIZE))
    with open(path, "rb") as file:
        mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
    assert ple_module._advise_random_file_access(
        torch.frombuffer(mapped, dtype=torch.uint8)
    ) == str(path)
