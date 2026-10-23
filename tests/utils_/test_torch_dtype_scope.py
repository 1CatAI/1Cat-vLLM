# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.utils.torch_utils import set_default_torch_dtype


@pytest.mark.parametrize("fail", [False, True])
def test_nested_dtype_scope_restores_after_model_load_failure(fail):
    original = torch.get_default_dtype()
    try:
        with set_default_torch_dtype(torch.float64):
            try:
                with set_default_torch_dtype(torch.float16):
                    assert torch.empty(1).dtype == torch.float16
                    if fail:
                        raise RuntimeError("model construction failed")
            except RuntimeError:
                assert fail
            assert torch.empty(1).dtype == torch.float64
        assert torch.get_default_dtype() == original
    finally:
        torch.set_default_dtype(original)
