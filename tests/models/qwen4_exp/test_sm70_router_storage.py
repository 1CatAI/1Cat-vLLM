# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import KernelConfig
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import _router_batch_storage


@pytest.mark.parametrize("storage", ["row_major", "dual"])
def test_router_storage_preserves_checkpoint_bits(storage):
    raw = torch.randint(-(2**15), 2**15, (512, 2560), dtype=torch.int16)
    weight = raw.view(torch.float16)
    selected = _router_batch_storage(weight, storage)
    if storage == "row_major":
        assert selected.data_ptr() == weight.data_ptr()
        assert selected.untyped_storage().nbytes() == weight.untyped_storage().nbytes()
        restored = selected
    else:
        assert selected.data_ptr() != weight.data_ptr()
        restored = selected.permute(0, 4, 3, 1, 2, 5).contiguous().view(512, 2560)
    torch.testing.assert_close(restored.view(torch.int16), raw, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(511, 2560), (512, 2559)])
def test_router_original_storage_rejects_unsupported_shape(shape):
    with pytest.raises(ValueError, match="contiguous"):
        _router_batch_storage(torch.empty(shape), "row_major")


def test_router_original_storage_rejects_noncontiguous():
    with pytest.raises(ValueError, match="contiguous"):
        _router_batch_storage(torch.empty(2560, 512).T, "row_major")


def test_router_storage_policy_defaults_to_existing_route():
    assert KernelConfig().sm70_router_weight_storage == "dual"
    assert (
        KernelConfig(sm70_router_weight_storage="row_major").sm70_router_weight_storage
        == "row_major"
    )
    with pytest.raises(ValueError):
        KernelConfig(sm70_router_weight_storage="invalid")


def test_router_storage_rejects_unknown_layout():
    with pytest.raises(ValueError, match="Unknown"):
        _router_batch_storage(torch.empty(512, 2560), "invalid")
