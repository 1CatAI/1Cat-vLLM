# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""YaRN must be off on DeepSeek-V4's sliding-window-only layers.

For compress_ratio == 0 layers the reference passes original_seq_len=0 to
precompute_freqs_cis (inference/model.py:481-486), which skips the correction
range entirely and leaves plain rope at the base rope_theta. Swapping only
rope_theta and leaving rope_type="deepseek_yarn" with factor=16 keeps the
interpolation on for those layers.
"""

import math

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform

# DeepSeek-V4-Flash config.json.
_HEAD_SIZE = 512
_ROPE_DIM = 64
_MAX_POSITION = 262144  # served context; the model passes max_position_embeddings
_ORIGINAL_MAX_POSITION = 65536
_YARN_FACTOR = 16
_ROPE_THETA = 10000  # SWA-only layers; compressing layers use 160000
_SWA_WINDOW = 128


@pytest.fixture(autouse=True)
def _default_device():
    """DeepseekV4ScalingRotaryEmbedding builds its position grid on
    current_platform.device_type but its inv_freq on the default device, so it
    only assembles under the device context the model loader already sets."""
    with torch.device(current_platform.device_type):
        yield


def _rope_parameters(factor: int, original_max_position: int) -> dict:
    """What DeepseekV4Attention hands get_rope for a compress_ratio<=1 layer."""
    return {
        "rope_type": "deepseek_yarn",
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": factor,
        "original_max_position_embeddings": original_max_position,
        "rope_theta": _ROPE_THETA,
        "mscale": 0,
        "mscale_all_dim": 0,
        "is_deepseek_v4": True,
        "rope_dim": _ROPE_DIM,
    }


def _get_rope(factor: int, original_max_position: int):
    return get_rope(
        _HEAD_SIZE,
        max_position=_MAX_POSITION,
        rope_parameters=_rope_parameters(factor, original_max_position),
        is_neox_style=False,
    )


def _plain_inv_freq() -> torch.Tensor:
    exponents = torch.arange(0, _ROPE_DIM, 2, dtype=torch.float32) / _ROPE_DIM
    return 1.0 / (_ROPE_THETA**exponents)


def test_factor_one_is_exactly_plain_rope(default_vllm_config) -> None:
    """factor=1 collapses YaRN to an identity, so no class swap is needed.

    DeepseekV4ScalingRotaryEmbedding is what the fused kernels expect: it
    rotates the *last* rotary_dim and keeps cos/sin in fp32. Neither is true of
    the base RotaryEmbedding, so "turn YaRN off" has to mean factor=1, not
    rope_type="default".
    """
    swa = _get_rope(1, _MAX_POSITION)

    assert swa.mscale == 1.0
    # inv_freq_interpolation == inv_freq_extrapolation at factor=1, so the ramp
    # blend `a * (1 - m) + a * m` is algebraically `a`; in fp32 it lands within
    # a couple of ulp on the two pairs where the mask is neither 0 nor 1.
    torch.testing.assert_close(
        swa._compute_inv_freq(swa.scaling_factor),
        _plain_inv_freq(),
        rtol=1e-6,
        atol=0,
    )
    # Cache length is original_max_position * factor, so original_max_position
    # has to carry the full range once factor is 1.
    assert swa.cos_sin_cache.shape[0] == _MAX_POSITION


def test_yarn_config_still_interpolates_the_low_frequency_pairs(
    default_vllm_config,
) -> None:
    """The unfixed config is a real divergence, not a no-op."""
    yarn = _get_rope(_YARN_FACTOR, _ORIGINAL_MAX_POSITION)

    yarn_inv_freq = yarn._compute_inv_freq(yarn.scaling_factor)
    plain = _plain_inv_freq()
    assert not torch.allclose(yarn_inv_freq, plain)

    # Only the low-frequency pairs move: high-frequency pairs sit above the
    # correction range and extrapolate unchanged.
    moved = (yarn_inv_freq != plain).nonzero().flatten().tolist()
    assert moved and min(moved) >= _ROPE_DIM // 4

    # Accumulated rotation error across one 128-token SWA window.
    worst_radians = ((plain - yarn_inv_freq).abs() * (_SWA_WINDOW - 1)).max().item()
    assert 0.0 < math.degrees(worst_radians) < 5.0


def test_shared_rope_parameters_dict_must_not_be_mutated(default_vllm_config) -> None:
    """A shared rope_parameters dict must not be mutated in place.

    DeepseekV4Attention reads config.rope_parameters itself, and every later
    layer reads it back, so setting factor=1 in place would disable YaRN
    model-wide.
    """
    shared = _rope_parameters(_YARN_FACTOR, _ORIGINAL_MAX_POSITION)
    before = dict(shared)

    swa_parameters = dict(shared)
    swa_parameters["factor"] = 1
    swa_parameters["original_max_position_embeddings"] = _MAX_POSITION
    get_rope(
        _HEAD_SIZE,
        max_position=_MAX_POSITION,
        rope_parameters=swa_parameters,
        is_neox_style=False,
    )

    assert shared == before
    compressing = _get_rope(_YARN_FACTOR, _ORIGINAL_MAX_POSITION)
    assert compressing.scaling_factor == _YARN_FACTOR
