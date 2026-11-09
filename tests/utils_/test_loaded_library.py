# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import ctypes
import io
import shutil
import subprocess

import pytest

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.utils.system_utils import find_loaded_library

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def runtime_symbols(tmp_path):
    """A real ELF with the required symbols, without CUDA dependencies.

    The tests only bind symbols; they never invoke these placeholder functions.
    """
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the runtime fixture")
    assert compiler is not None
    source = tmp_path / "runtime.c"
    source.write_text(
        "\n".join(
            f"int {func.name}(void) {{ return 0; }}"
            for func in CudaRTLibrary.exported_functions
        )
    )
    runtime = str(tmp_path / "libcudart.so.12")
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-o", runtime], check=True
    )
    return runtime


def mock_maps(monkeypatch, paths):
    maps = "0000-1000 r-xp 0000 00:00 0 [heap]\n" + "".join(
        f"1000-2000 r-xp 0000 00:00 1 {path}\n" for path in paths
    )
    real_open = builtins.open

    def open_file(path, *args, **kwargs):
        if path == "/proc/self/maps":
            return io.StringIO(maps)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_file)


@pytest.mark.parametrize(
    "lib_name,filename",
    [
        ("libcudart", "libcudart.so.12"),
        ("libcudart", "libcudart-abc123.so.11.0"),
        ("libamdhip64", "libamdhip64.so.6"),
        ("cumem_allocator", "cumem_allocator.abi3.so"),
    ],
)
def test_loaded_library_matches_filename(monkeypatch, lib_name, filename):
    expected = f"/runtime/{filename}"
    mock_maps(
        monkeypatch,
        [f"/stubs/{lib_name}_stub.so", f"/contains/{lib_name}/other.so", expected],
    )
    assert find_loaded_library(lib_name) == expected


@pytest.mark.parametrize(
    "paths",
    [[], ["/stubs/libcudart_stub.so"], ["/contains/libcudart/other.so"]],
)
def test_no_real_loaded_library(monkeypatch, paths):
    mock_maps(monkeypatch, paths)
    assert find_loaded_library("libcudart") is None


@pytest.mark.parametrize("env_fallback", [False, True])
def test_cudart_ignores_loaded_partial_stub(
    tmp_path, monkeypatch, env_fallback, runtime_symbols
):
    """Load a real partial ELF, then put it first in deterministic maps order.

    Mapping addresses depend on ASLR, so loading the stub first alone does not
    reliably reproduce the bug. No CUDA function is called in this test.
    """
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the partial runtime fixture")
    assert compiler is not None
    runtime = runtime_symbols
    source = tmp_path / "stub.c"
    source.write_text(
        "int cudaSetDevice(int device) { return 0; }\n"
        "int cudaDeviceSynchronize(void) { return 0; }\n"
    )
    stub_path = str(tmp_path / "libcudart_stub.so")
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-o", stub_path], check=True
    )
    stub = ctypes.CDLL(stub_path)
    assert hasattr(stub, "cudaSetDevice")
    assert not hasattr(stub, "cudaDeviceReset")
    import vllm.distributed.device_communicators.cuda_wrapper as wrapper

    mock_maps(monkeypatch, [stub_path] if env_fallback else [stub_path, runtime])
    monkeypatch.setattr(wrapper.envs, "VLLM_CUDART_SO_PATH", runtime)
    # Isolate the class caches without changing any other process's state.
    monkeypatch.setattr(CudaRTLibrary, "path_to_library_cache", {})
    monkeypatch.setattr(CudaRTLibrary, "path_to_dict_mapping", {})
    lib = CudaRTLibrary()
    assert lib.lib._name == runtime
    assert set(lib.funcs) == {func.name for func in lib.exported_functions}


def test_loaded_runtime_keeps_function_bindings(monkeypatch, runtime_symbols):
    """Ordinary discovery still chooses and caches the identical library."""
    import vllm.distributed.device_communicators.cuda_wrapper as wrapper

    path = runtime_symbols
    mock_maps(monkeypatch, [path])
    monkeypatch.setattr(CudaRTLibrary, "path_to_library_cache", {})
    monkeypatch.setattr(CudaRTLibrary, "path_to_dict_mapping", {})
    explicit = CudaRTLibrary(path)
    automatic = CudaRTLibrary()
    assert automatic.lib is explicit.lib
    assert automatic.funcs is explicit.funcs
    for function in wrapper.CudaRTLibrary.exported_functions:
        assert automatic.funcs[function.name].argtypes == function.argtypes
        assert automatic.funcs[function.name].restype == function.restype
