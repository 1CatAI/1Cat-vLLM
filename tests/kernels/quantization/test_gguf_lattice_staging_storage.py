# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.gguf_lattice_staging import (
    GGUFLatticeStaging,
)


def test_staging_storage_is_shared_across_formats_but_not_projections():
    pool = GGUFLatticeStaging(512, 160, 2560, "meta")
    storages = {id(buffer.untyped_storage()): buffer for buffer in pool.buffers()}
    assert sum(buffer.untyped_storage().nbytes() for buffer in storages.values()) == (
        200 * 1024**2
    )
    for kind in (18, 21, 22):
        gate, gate_stats = pool.slot("w1", kind)
        up, up_stats = pool.slot("w3", kind)
        assert gate.shape == up.shape == (512, 2560, 10)
        assert (
            gate_stats.shape == up_stats.shape == (512, 160 if kind == 22 else 80, 160)
        )
        assert gate_stats.dtype == (torch.int32 if kind == 22 else torch.int64)
        assert gate.untyped_storage() is up.untyped_storage()
        assert gate.storage_offset() != up.storage_offset()
        assert gate_stats.untyped_storage() is up_stats.untyped_storage()
        assert gate_stats.storage_offset() != up_stats.storage_offset()
