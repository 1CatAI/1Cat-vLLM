# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.engine import utils


@pytest.mark.parametrize(
    "rank,local_size,expected",
    [
        (0, None, "GPU-a,GPU-b"),
        (1, None, "GPU-c,GPU-d"),
        (1, 1, "GPU-c"),
    ],
)
def test_uuid_masks_do_not_require_nvml_index_translation(
    monkeypatch, rank, local_size, expected
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " GPU-a, GPU-b, GPU-c, GPU-d ")

    def unexpected_translation(index):
        pytest.fail(f"UUID unexpectedly translated through index {index}")

    monkeypatch.setattr(
        utils,
        "current_platform",
        SimpleNamespace(device_id_to_physical_device_id=unexpected_translation),
    )
    assert (
        utils.get_device_indices("CUDA_VISIBLE_DEVICES", rank, 2, local_size)
        == expected
    )


def test_integer_mask_uses_existing_platform_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1,2,0")
    monkeypatch.setattr(
        utils,
        "current_platform",
        SimpleNamespace(
            device_id_to_physical_device_id=lambda index: [3, 1, 2, 0][index]
        ),
    )
    assert utils.get_device_indices("CUDA_VISIBLE_DEVICES", 1, 2) == "2,0"
