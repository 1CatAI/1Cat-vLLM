# SPDX-License-Identifier: Apache-2.0
"""Explicit, source-addressed SM70 compilation, independent of the engine JIT."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from filelock import FileLock

from ..version import __version__

_SOURCES = {"basics": "sm70/basics.cu", "qsa": "sm70/qsa.cu"}
_MODULES = {}
_LOADED_PROVENANCE = {}
_FLAGS = ["-O3", "-lineinfo", "-DFLASHINFER_ENABLE_F16", "-DFLASHINFER_SM70_RMS_ONLY"]


def source_root():
    package = Path(__file__).resolve().parents[1]
    installed = package / "data"
    source = package.parent
    return (
        source
        if (source / "UPSTREAM.lock.json").is_file()
        and (source / "csrc" / "sm70").is_dir()
        else installed
    )


def _compiler():
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda) / "bin/nvcc") if cuda else shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            "SM70 compilation requires a CUDA 12 toolkit (12.8 for release)"
        )
    report = subprocess.check_output([nvcc, "--version"], text=True)
    release = re.search(r"release (\d+)\.(\d+)", report)
    if release is None or int(release[1]) != 12:
        raise RuntimeError(
            "SM70 compilation requires CUDA 12; CUDA 13 cannot emit Volta"
        )
    cxx = os.environ.get("CXX", "c++")
    return nvcc, {
        "nvcc": report,
        "cxx": subprocess.check_output([cxx, "--version"], text=True),
    }


def _identity(component):
    if component not in _SOURCES:
        raise ValueError(f"Unknown SM70 component: {component!r}")
    root = source_root()
    nvcc, compiler = _compiler()
    # Hash transitive framework-independent headers as well as bindings. This is
    # deliberately preparation-time work, never part of an execution or replay.
    paths = [root / "csrc" / _SOURCES[component], Path(__file__)]
    paths += sorted((root / "include" / "flashinfer").rglob("*.cuh"))
    paths += sorted((root / "include" / "flashinfer").rglob("*.h"))
    paths += sorted((root / "csrc" / "sm70").glob("*.h"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    lock = Path(__file__).resolve().parents[1] / "UPSTREAM.lock.json"
    if not lock.exists():
        lock = root / "UPSTREAM.lock.json"
    record = {
        "version": __version__,
        "component": component,
        "arch": "sm_70",
        "source_sha256": digest.hexdigest(),
        "upstream": json.loads(lock.read_text()),
        "compiler": compiler,
        "compiler_environment": {
            key: os.environ.get(key)
            for key in (
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
                "CUDAHOSTCXX",
                "CXX",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
            )
        },
        "flags": _FLAGS,
        "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
    }
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return root, nvcc, key, record


def build(component="basics"):
    """Compile without initializing CUDA or importing Torch; return the DSO path."""
    root, nvcc, key, record = _identity(component)
    cache = Path(
        os.environ.get(
            "FLASHINFER_SM70_CACHE_DIR", Path.home() / ".cache/flashinfer/sm70"
        )
    )
    directory = cache / key
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "kernels.so"
    manifest = directory / "manifest.json"
    with FileLock(str(directory / "build.lock")):
        if library.exists() and manifest.exists():
            saved = json.loads(manifest.read_text())
            if (
                saved.get("library_sha256")
                == hashlib.sha256(library.read_bytes()).hexdigest()
            ):
                return library
        if os.environ.get("FLASHINFER_DISABLE_JIT"):
            raise RuntimeError(
                f"SM70 {component} is not prebuilt for source/toolchain {key}"
            )
        config = {
            "name": f"flashinfer_sm70_{component}_{key[:16]}",
            "cuda_files": [str(root / "csrc" / _SOURCES[component])],
            "extra_cuda_cflags": _FLAGS,
            "extra_include_paths": [
                str(root / "include"),
                str(root / "csrc" / "sm70"),
                str(
                    root / "cutlass" / "include"
                    if (root / "cutlass").exists()
                    else root / "3rdparty/cutlass/include"
                ),
            ],
            "build_directory": str(directory),
            "output": str(library),
        }
        # A child compiler process avoids changing process-wide arch variables
        # when one engine has instances on different device architectures.
        environment = dict(
            os.environ, TVM_FFI_CUDA_ARCH_LIST="7.0", TVM_FFI_GPU_BACKEND="cuda"
        )
        environment["CUDA_HOME"] = str(Path(nvcc).parent.parent)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys; from tvm_ffi.cpp import build; build(**json.load(sys.stdin))",
            ],
            input=json.dumps(config),
            text=True,
            env=environment,
            check=True,
        )
        record["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(manifest)
    return library


def prepare(component="basics"):
    """Build/load before capture. Each plan still owns its own mutable workspace."""
    if component not in _MODULES:
        torch = sys.modules.get("torch")
        if (
            torch is not None
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Call flashinfer.sm70.prepare() before CUDA graph capture"
            )
        import tvm_ffi

        library = build(component)
        record = json.loads(library.with_name("manifest.json").read_text())
        _MODULES[component] = tvm_ffi.load_module(str(library))
        # Snapshot the artifact we actually loaded. A subsequent source change
        # or explicit offline build must not relabel a running kernel.
        _LOADED_PROVENANCE[component] = {
            **record,
            "library": str(library),
            "loaded": True,
        }
    return _MODULES[component]


def provenance(component="basics"):
    if component in _MODULES:
        # Return a copy so callers cannot mutate the retained loaded identity.
        return json.loads(json.dumps(_LOADED_PROVENANCE[component]))
    library = build(component)
    return {
        **json.loads(library.with_name("manifest.json").read_text()),
        "library": str(library),
        "loaded": False,
    }
