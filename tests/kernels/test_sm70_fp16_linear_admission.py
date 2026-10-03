# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for admission and unmeasured-width fallback."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear.mixed_precision import sm70_fp16 as impl
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.scalar_type import scalar_types


def config():
    return impl.Sm70Fp16LinearLayerConfig(
        full_weight_shape=(73, 17),
        partition_weight_shape=(73, 17),
        weight_type=scalar_types.float16,
        act_type=torch.float16,
        group_size=-1,
        zero_points=False,
        has_g_idx=False,
        out_type=torch.float16,
    )


def test_admission_requires_precision_policy_and_unpacked_weights():
    c = config()
    with (
        patch.object(impl.current_platform, "is_device_capability", return_value=True),
        patch.object(
            torch.backends.cuda,
            "matmul",
            SimpleNamespace(
                allow_fp16_reduced_precision_reduction=False,
                allow_fp16_accumulation=False,
            ),
        ),
    ):
        assert impl.Sm70Fp16LinearKernel.can_implement(c) == (True, None)
        for changes in (
            {"zero_points": True},
            {"has_g_idx": True},
            {"group_size": 32},
            {"act_type": torch.float32},
            {"out_type": torch.float32},
            {"partition_weight_shape": (0, 17)},
        ):
            assert not impl.Sm70Fp16LinearKernel.can_implement(
                dataclasses.replace(c, **changes)
            )[0]
        assert not impl.Sm70Fp16LinearKernel.can_implement(
            MPLinearLayerConfig(**dataclasses.asdict(c))
        )[0]
        with patch.object(
            torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True
        ):
            assert not impl.Sm70Fp16LinearKernel.can_implement(c)[0]


@pytest.mark.parametrize("shape", [(1, 73), (5, 73), (17, 73), (2, 3, 73)])
def test_unmeasured_width_uses_vendor_and_preserves_leading_dimensions(shape):
    x = torch.randn(shape, dtype=torch.float16)
    weight = torch.randn(17, 73, dtype=torch.float16)
    with patch.object(
        impl, "_candidate", side_effect=AssertionError("unmeasured route")
    ):
        actual = impl._dispatch(x, weight, False, [])
    torch.testing.assert_close(
        actual, torch.nn.functional.linear(x, weight), rtol=0, atol=0
    )
    assert impl._fake(x, weight, False, []).shape == actual.shape
