# SPDX-License-Identifier: Apache-2.0
"""Source-complete build without CUDA, Torch or optional EP build side effects."""

import json
import shutil
from pathlib import Path

from setuptools import build_meta as backend

ROOT = Path(__file__).resolve().parent


def _metadata():
    lock = json.loads((ROOT / "UPSTREAM.lock.json").read_text())
    package = ROOT / "flashinfer"
    version = (ROOT / "version.txt").read_text().strip()
    (package / "_build_meta.py").write_text(
        f"__version__ = {version!r}\n"
        f"__git_commit__ = {lock['commit']!r}\n"
        f"__upstream_git_commit__ = {lock['commit']!r}\n"
    )
    shutil.copy2(ROOT / "UPSTREAM.lock.json", package / "UPSTREAM.lock.json")


def _prepare():
    _metadata()
    package = ROOT / "flashinfer"
    # The upstream package-data mapping installs these directories under
    # flashinfer.data. Real directories also make an sdist self-contained.
    data = package / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir()
    for source, name in (
        ("include", "include"),
        ("csrc", "csrc"),
        ("3rdparty/cutlass/include", "cutlass/include"),
        ("3rdparty/cutlass/tools/util/include", "cutlass/tools/util/include"),
        ("3rdparty/cccl/cub/cub", "cccl/cub/cub"),
        ("3rdparty/cccl/libcudacxx/include", "cccl/libcudacxx/include"),
        ("3rdparty/cccl/thrust/thrust", "cccl/thrust/thrust"),
        ("3rdparty/spdlog/include", "spdlog/include"),
    ):
        shutil.copytree(ROOT / source, data / name)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare()
    return backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _prepare()
    return backend.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare()
    return backend.build_editable(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _prepare()
    return backend.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


def get_requires_for_build_wheel(config_settings=None):
    _metadata()
    return backend.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    _metadata()
    return backend.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    _metadata()
    return backend.get_requires_for_build_editable(config_settings)
