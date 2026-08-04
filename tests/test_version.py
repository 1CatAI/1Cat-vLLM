# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm import version


def test_version_is_defined():
    assert version.__version__ is not None


def test_version_tuple():
    assert len(version.__version_tuple__) in (3, 4, 5)


def test_distribution_version_prefers_1cat_package(monkeypatch):
    def distribution_version(distribution_name):
        assert distribution_name == "1cat-vllm"
        return "1.2.2"

    monkeypatch.setattr(version.importlib.metadata, "version", distribution_version)

    assert version.get_vllm_distribution_version() == "1.2.2"


def test_distribution_version_falls_back_to_upstream_package(monkeypatch):
    def distribution_version(distribution_name):
        if distribution_name == "1cat-vllm":
            raise version.importlib.metadata.PackageNotFoundError
        assert distribution_name == "vllm"
        return "0.13.0"

    monkeypatch.setattr(version.importlib.metadata, "version", distribution_version)

    assert version.get_vllm_distribution_version() == "0.13.0"


def test_distribution_version_falls_back_to_source_version(monkeypatch):
    def distribution_version(distribution_name):
        raise version.importlib.metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(version.importlib.metadata, "version", distribution_version)
    monkeypatch.setattr(version, "__version__", "source")

    assert version.get_vllm_distribution_version() == "source"


@pytest.mark.parametrize(
    "version_tuple, version_str, expected",
    [
        ((0, 0, "dev"), "0.0", True),
        ((0, 0, "dev"), "foobar", True),
        ((0, 7, 4), "0.6", True),
        ((0, 7, 4), "0.5", False),
        ((0, 7, 4), "0.7", False),
        ((1, 2, 3), "1.1", True),
        ((1, 2, 3), "1.0", False),
        ((1, 2, 3), "1.2", False),
        # This won't work as expected
        ((1, 0, 0), "1.-1", True),
        ((1, 0, 0), "0.9", False),
        ((1, 0, 0), "0.17", False),
    ],
)
def test_prev_minor_version_was(version_tuple, version_str, expected):
    with patch("vllm.version.__version_tuple__", version_tuple):
        assert version._prev_minor_version_was(version_str) == expected
