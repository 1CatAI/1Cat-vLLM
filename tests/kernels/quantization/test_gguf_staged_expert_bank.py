# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_gguf_lattice_transcode import source

from vllm.model_executor.layers.quantization.gguf_lattice_staging import (
    GGUFLatticeStaging,
)


@pytest.mark.parametrize("kind", [18, 21, 22])
def test_staged_bank_keeps_original_rows_without_per_expert_converter(
    monkeypatch, kind
):
    from vllm.model_executor.layers.quantization import gguf_turbomind_moe as module

    # Isolate storage from device admission, which the CUDA suite covers at
    # the actual 512-expert shape. No device Converter may run during loading.
    admitted = (SimpleNamespace(reason=None),)
    monkeypatch.setattr(
        module, "raw_grouped_gate_up_capabilities", lambda *a, **k: admitted
    )
    monkeypatch.setattr(
        module, "lattice_grouped_capabilities", lambda *a, **k: admitted
    )
    monkeypatch.setattr(
        module, "current_platform", SimpleNamespace(is_device_capability=lambda _: True)
    )

    def no_converter(*args):
        pytest.fail("staged loading must not construct persistent canonical experts")

    monkeypatch.setattr(
        torch.ops._C, "gguf_lattice_sm70_prepare", no_converter, raising=False
    )

    def pointers(weights, stats, k_ld, q_ld, experts):
        assert (k_ld, q_ld, experts) == (256 * 32, 64, 2)
        return torch.arange(experts, dtype=torch.int64), torch.arange(
            experts, dtype=torch.int64
        )

    monkeypatch.setattr(
        torch.ops._C, "awq_moe_build_strided_ptrs", pointers, raising=False
    )
    pool = GGUFLatticeStaging(2, 64, 256, "cpu")
    banks = []
    for shard in ("w1", "w3"):
        bank = module.GGUFExpertBank(
            kind,
            2,
            torch.device("cpu"),
            torch.float16,
            retain_raw=True,
            staging_pool=pool,
            shard=shard,
        )
        original = []
        for expert in range(2):
            rows = source(kind, 0.00137, n=64, k=256, seed=47 + expert)
            original.append(rows)
            bank.add(expert, torch.from_numpy(rows), 0, 1, 0)
            assert bank.pending[expert] is None
        bank.finalize()
        assert not bank.pending and not bank.raw_pending
        assert bank.staging_pool is pool
        codes, stats = pool.slot(shard, kind)
        assert bank.weights.data_ptr() == codes.data_ptr()
        assert bank.stats.data_ptr() == stats.data_ptr()
        payload = original[0].shape[1]
        np.testing.assert_array_equal(
            bank.raw_weights[:, :, :payload].numpy(), original
        )
        assert bank.raw_weights[:, :, payload:].count_nonzero() == 0
        banks.append(bank)
    assert banks[0].weights.untyped_storage() is banks[1].weights.untyped_storage()
    assert banks[0].weights.data_ptr() != banks[1].weights.data_ptr()
