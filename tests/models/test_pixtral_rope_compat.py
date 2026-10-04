# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.pixtral import (
    PixtralRotaryEmbedding,
    position_ids_in_meshgrid,
)


def test_flat_positions_keep_rectangular_image_boundaries():
    patches = [torch.empty(1, 8, 2, 3), torch.empty(1, 8, 3, 1)]
    ids = position_ids_in_meshgrid(patches, max_width=16)
    assert ids.tolist() == [0, 1, 2, 16, 17, 18, 0, 16, 32]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("new_config", [False, True])
def test_rotary_matches_independent_axial_formula(dtype, new_config):
    config = SimpleNamespace(head_dim=32, image_size=64, patch_size=4)
    if new_config:
        config.rope_parameters = {"rope_type": "axial", "rope_theta": 10000.0}
    else:
        config.rope_theta = 10000.0
    rope = PixtralRotaryEmbedding(config, device="cpu")
    ids = torch.tensor([0, 1, 15, 16, 18, 239, 255])
    x = torch.empty(1, len(ids), 64, dtype=dtype)
    cos, sin = rope(x, ids)
    # Independently assign every H-W-H-W channel its spatial coordinate.
    channels = torch.arange(32)
    spatial_channel = channels % 16
    is_height = spatial_channel < 8
    frequency_index = 2 * (spatial_channel % 8) + (~is_height).long()
    coordinate = torch.where(is_height[None], ids[:, None] // 16, ids[:, None] % 16)
    angle = coordinate.double() / (10000.0 ** (frequency_index.double() / 16))
    expected_cos, expected_sin = angle.cos().to(dtype), angle.sin().to(dtype)
    # The double oracle avoids the two FP32 power/trigonometric roundings.
    tolerance = {
        torch.float32: 4 * torch.finfo(dtype).eps,
        torch.float16: 5e-4,
        torch.bfloat16: 4e-3,
    }[dtype]
    torch.testing.assert_close(cos, expected_cos, atol=tolerance, rtol=0)
    torch.testing.assert_close(sin, expected_sin, atol=tolerance, rtol=0)
    assert cos.dtype == sin.dtype == dtype
    assert rope.state_dict() == {}


def test_llava_and_pixtral_import_with_supported_transformers():
    from vllm.model_executor.models.llava import LlavaForConditionalGeneration
    from vllm.model_executor.models.pixtral import PixtralHFVisionModel

    assert LlavaForConditionalGeneration is not None
    assert PixtralHFVisionModel is not None
