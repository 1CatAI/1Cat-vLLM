# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.platforms import cuda


@pytest.mark.parametrize("operation", ["handle", "name"])
def test_nvml_warning_query_failure_returns_placeholder(monkeypatch, operation):
    def handle(index):
        if operation == "handle":
            raise cuda.pynvml.NVMLError(cuda.pynvml.NVML_ERROR_GPU_IS_LOST)
        return index

    def name(handle):
        raise cuda.pynvml.NVMLError(cuda.pynvml.NVML_ERROR_GPU_IS_LOST)

    monkeypatch.setattr(cuda.pynvml, "nvmlDeviceGetHandleByIndex", handle)
    monkeypatch.setattr(cuda.pynvml, "nvmlDeviceGetName", name)
    assert cuda.NvmlCudaPlatform._get_physical_device_name(2) == "<unavailable:2>"


def test_healthy_device_name_is_preserved(monkeypatch):
    monkeypatch.setattr(cuda.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(cuda.pynvml, "nvmlDeviceGetName", lambda handle: "Test GPU")
    assert cuda.NvmlCudaPlatform._get_physical_device_name(2) == "Test GPU"


def test_unexpected_programming_error_is_not_hidden(monkeypatch):
    def handle(index):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(cuda.pynvml, "nvmlDeviceGetHandleByIndex", handle)
    with pytest.raises(RuntimeError, match="unexpected failure"):
        cuda.NvmlCudaPlatform._get_physical_device_name(2)
