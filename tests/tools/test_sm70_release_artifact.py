# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tools.check_sm70_release_artifact import ArtifactError, check_wheel

REQUIRED_MEMBERS = [
    "vllm/_C.abi3.so",
    "vllm/_C_stable_libtorch.abi3.so",
    "vllm/_sm70_sampler_C.abi3.so",
    "vllm/_sm70_exact_reduce_C.cpython-312-x86_64-linux-gnu.so",
    "vllm/_sm70_sparse_attention_C.cpython-312-x86_64-linux-gnu.so",
    "vllm/_h3_w8a16_C.cpython-312-x86_64-linux-gnu.so",
    "vllm/_h3_flashinfer_C.cpython-312-x86_64-linux-gnu.so",
    "vllm/_h3_flashattn_C.cpython-312-x86_64-linux-gnu.so",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "flash_attn_v100/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so",
    "flash_attn_v100/paged_kv_utils.cpython-312-x86_64-linux-gnu.so",
    "flash_qla/ops/gated_delta_rule/chunk/sm70/flash_qla_sm70_gdn_strided.so",
    "1cat_vllm-0.0.0.data/scripts/serve_qwen38_27b_nvfp4_v100.sh",
]


def write_wheel(path: Path, members: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        for member in members:
            wheel.writestr(member, b"fixture; dynamic inspection is disabled")


def test_complete_sm70_wheel_manifest_passes(tmp_path: Path) -> None:
    wheel = tmp_path / "complete.whl"
    write_wheel(wheel, REQUIRED_MEMBERS)

    check_wheel(wheel, inspect_dynamic=False)


@pytest.mark.parametrize(
    "missing",
    [
        "vllm/_sm70_sampler_C.abi3.so",
        "flash_qla/ops/gated_delta_rule/chunk/sm70/flash_qla_sm70_gdn_strided.so",
    ],
)
def test_missing_companion_extension_fails_closed(tmp_path: Path, missing: str) -> None:
    wheel = tmp_path / "incomplete.whl"
    write_wheel(wheel, [member for member in REQUIRED_MEMBERS if member != missing])

    with pytest.raises(ArtifactError, match="missing"):
        check_wheel(wheel, inspect_dynamic=False)


def test_launcher_is_required(tmp_path: Path) -> None:
    wheel = tmp_path / "without-launcher.whl"
    write_wheel(
        wheel,
        [member for member in REQUIRED_MEMBERS if "serve_qwen38" not in member],
    )

    with pytest.raises(ArtifactError, match="launcher"):
        check_wheel(wheel, inspect_dynamic=False)


def test_release_launcher_has_no_developer_runtime_overlays() -> None:
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "serve_qwen38_27b_nvfp4_v100.sh"
    ).read_text()

    assert "/home/" not in launcher
    assert "/data/" not in launcher
    assert "PREBUILT_EXTENSION_PATH" not in launcher
    assert "export VLLM_" not in launcher
    assert "export FLASH_" not in launcher
