# SPDX-License-Identifier: Apache-2.0
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
import pytest

import flashinfer
from flashinfer.sm70 import _jit


def test_clean_process_import_is_independent_and_lazy():
    code = """
import importlib.util, sys
# cuda-bindings may create its namespace from a Python startup .pth file.
# Check what importing this package adds, including any new backend submodules.
before = set(sys.modules)
import flashinfer
assert importlib.util.find_spec('vllm') is None
assert '1cat.sm70' in flashinfer.__version__
backends = {'torch','tvm_ffi','cutlass','triton','cuda','vllm'}
assert not any(m.split('.')[0] in backends for m in set(sys.modules) - before)
from flashinfer import rmsnorm, gemma_rmsnorm, silu_and_mul, apply_rope_pos_ids
assert all(callable(x) for x in (rmsnorm, gemma_rmsnorm, silu_and_mul, apply_rope_pos_ids))
assert 'vllm' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_minimal_install_has_no_modern_backend_dependency():
    requirements = [
        Requirement(r) for r in importlib.metadata.requires("flashinfer-python")
    ]
    base = {
        r.name
        for r in requirements
        if r.marker is None or r.marker.evaluate({"extra": ""})
    }
    assert {"torch", "apache-tvm-ffi", "ninja"} <= base
    assert (
        not {"nvidia-cutlass-dsl", "cuda-tile", "nccl-extensions", "nccl4py", "vllm"}
        & base
    )


def test_source_and_provenance_are_installed():
    package = Path(flashinfer.__file__).parent
    lock = json.loads((package / "UPSTREAM.lock.json").read_text())
    assert lock["commit"] == "34bb66e4d0b420b5a03c99b2b6a1c0831dfaa738"
    assert len(lock["dependencies"]) == 4
    for path in (
        "csrc/sm70/basics.cu",
        "csrc/sm70/common.h",
        "csrc/sm70/qsa.cu",
        "include/flashinfer/sm70/rope.cuh",
        "include/flashinfer/sm70/qsa.cuh",
        "include/flashinfer/norm.cuh",
    ):
        assert (package / "data" / path).is_file(), path
    assert "vllm" not in sys.modules


def test_public_signature_compatible():
    assert list(inspect.signature(flashinfer.rmsnorm).parameters) == [
        "input",
        "weight",
        "eps",
        "out",
        "enable_pdl",
    ]
    assert list(inspect.signature(flashinfer.apply_rope_pos_ids).parameters) == [
        "q",
        "k",
        "pos_ids",
        "rotary_dim",
        "interleave",
        "rope_scale",
        "rope_theta",
    ]
    assert callable(flashinfer.silu_and_mul)


def test_no_global_architecture_gate_relaxation():
    from flashinfer.jit.core import check_cuda_arch, current_compilation_context

    saved = current_compilation_context.TARGET_CUDA_ARCHS.copy()
    try:
        current_compilation_context.TARGET_CUDA_ARCHS = {(7, "0")}
        with pytest.raises(RuntimeError):
            check_cuda_arch()
    finally:
        current_compilation_context.TARGET_CUDA_ARCHS = saved


def test_unregistered_kernel_cannot_bypass_arch_guard():
    with pytest.raises(ValueError, match="Unknown SM70 component"):
        _jit.build("decode")


def test_prepare_cannot_compile_during_capture(monkeypatch):
    import torch

    monkeypatch.setattr(_jit, "_MODULES", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        _jit.prepare()


def test_sources_and_compiler_change_cache_identity(tmp_path, monkeypatch):
    (tmp_path / "include/flashinfer").mkdir(parents=True)
    (tmp_path / "csrc/sm70").mkdir(parents=True)
    (tmp_path / "csrc/sm70/basics.cu").write_text("binding")
    header = tmp_path / "include/flashinfer/primitive.cuh"
    header.write_text("first")
    monkeypatch.setattr(_jit, "source_root", lambda: tmp_path)
    monkeypatch.setattr(
        _jit, "_compiler", lambda: ("nvcc", {"nvcc": "12.8", "cxx": "gcc11"})
    )
    initial = _jit._identity("basics")[2]
    header.write_text("second")
    assert _jit._identity("basics")[2] != initial
    before_toolchain = _jit._identity("basics")[2]
    monkeypatch.setattr(
        _jit, "_compiler", lambda: ("nvcc", {"nvcc": "12.9", "cxx": "gcc11"})
    )
    assert _jit._identity("basics")[2] != before_toolchain


def test_loaded_identity_survives_source_changes(tmp_path, monkeypatch):
    import tvm_ffi

    first = tmp_path / "original"
    first.mkdir()
    library = first / "kernels.so"
    record = {"source_sha256": "original", "compiler": {"nvcc": "12.8"}}
    manifest = first / "manifest.json"
    manifest.write_text(json.dumps(record))
    monkeypatch.setattr(_jit, "_MODULES", {})
    monkeypatch.setattr(_jit, "_LOADED_PROVENANCE", {})
    monkeypatch.setattr(_jit, "build", lambda component: library)
    module = object()
    monkeypatch.setattr(tvm_ffi, "load_module", lambda path: module)
    assert _jit.prepare("qsa") is module
    # Rebuilding/editing files must not change the description of the loaded DSO.
    manifest.write_text('{"source_sha256": "replaced"}')

    def unexpected_build(*args):
        raise AssertionError("provenance must not compile again after load")

    monkeypatch.setattr(_jit, "build", unexpected_build)
    loaded = _jit.provenance("qsa")
    assert loaded["source_sha256"] == "original" and loaded["loaded"] is True
    loaded["compiler"]["nvcc"] = "changed by caller"
    assert _jit.provenance("qsa")["compiler"]["nvcc"] == "12.8"
