# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("arch", ["70", "75", "80", "90"])
def test_device_compiler_preserves_source_flags_and_other_architectures(
    monkeypatch, arch
):
    import tvm_ffi
    from tilelang import PassConfigKey
    from tilelang.contrib import nvcc

    from flash_qla import compiler

    captured = {}
    calls: list[tuple[Any, ...]] = []

    def register(name, override):
        def save(function):
            captured[name] = function
            return function

        return save

    def original(*args):
        calls.append(("original", args))
        return b"original"

    def nvrtc(source, **kwargs):
        calls.append(("nvrtc", source, kwargs))
        return b"runtime"

    monkeypatch.setattr(tvm_ffi, "register_global_func", register)
    monkeypatch.setattr(tvm_ffi, "get_global_func", lambda name: original)
    monkeypatch.setattr(nvcc, "get_target_compute_version", lambda target: target)
    monkeypatch.setattr(nvcc, "get_target_arch", lambda version: version)
    monkeypatch.setattr(compiler, "_registered", False)
    monkeypatch.setattr(compiler, "_cuda_headers", lambda: ["/pip/include"])
    monkeypatch.setitem(
        sys.modules, "tilelang.contrib.nvrtc", SimpleNamespace(compile_cuda=nvrtc)
    )
    compiler.register_sm70_runtime_compiler()
    options = {
        PassConfigKey.TL_ENABLE_FAST_MATH: True,
        PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--define-macro="VALUE=2"'],
        PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 2,
    }
    result = captured["tilelang_callback_cuda_compile"](
        "same device source", arch, options
    )
    if arch not in ("70", "75"):
        assert result == b"original"
        assert calls == [("original", ("same device source", arch, options))]
    else:
        assert result == b"runtime"
        assert calls[0][:2] == ("nvrtc", "same device source")
        kwargs = calls[0][2]
        assert kwargs["target_format"] == "cubin"
        assert kwargs["arch"] == int(arch)
        assert "-I/pip/include" in kwargs["options"]
        assert "--use_fast_math" in kwargs["options"]
        assert "--define-macro=VALUE=2" in kwargs["options"]
        assert "--ptxas-options=--register-usage-level=2" in kwargs["options"]
