# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import shutil
import subprocess
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
        wheel.writestr(
            "1cat_vllm-0.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: test\n"
            "Tag: cp312-cp312-manylinux_2_28_x86_64\n",
        )


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


def test_release_wheel_requires_cp312_tag(tmp_path: Path) -> None:
    wheel = tmp_path / "wrong-tag.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member in REQUIRED_MEMBERS:
            archive.writestr(member, b"fixture")
        archive.writestr(
            "1cat_vllm-0.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: test\n"
            "Tag: cp311-cp311-manylinux_2_28_x86_64\n",
        )

    with pytest.raises(ArtifactError, match="Python 3.12"):
        check_wheel(wheel, inspect_dynamic=False)


def test_unsafe_zip_member_is_rejected_before_dynamic_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "unsafe.whl"
    write_wheel(wheel, [*REQUIRED_MEMBERS, "../outside.so"])

    def fail_if_called(*args, **kwargs):
        pytest.fail("dynamic inspection must not extract unsafe members")

    monkeypatch.setattr(
        "tools.check_sm70_release_artifact._check_dynamic_dependencies",
        fail_if_called,
    )
    with pytest.raises(ArtifactError, match="unsafe wheel member"):
        check_wheel(wheel)


def test_dynamic_inspection_rejects_private_rpath(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("cc is required for the native artifact fixture")
    assert compiler is not None

    source = tmp_path / "fixture.c"
    source.write_text("int fixture(void) { return 1; }\n", encoding="utf-8")
    native = tmp_path / "fixture.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(source),
            "-Wl,-rpath,/home/private-build/lib",
            "-o",
            str(native),
        ],
        check=True,
    )

    wheel = tmp_path / "private-rpath.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        payload = native.read_bytes()
        for member in REQUIRED_MEMBERS:
            archive.writestr(member, payload)
        archive.writestr(
            "1cat_vllm-0.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: test\n"
            "Tag: cp312-cp312-manylinux_2_28_x86_64\n",
        )

    with pytest.raises(ArtifactError, match="private build path|RPATH"):
        check_wheel(wheel)


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
