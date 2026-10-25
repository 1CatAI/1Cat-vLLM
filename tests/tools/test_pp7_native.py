# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native provenance checks with tiny fake ELF files, no Torch/CUDA imports."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "tools/pp7_native.py"
SPEC = importlib.util.spec_from_file_location("pp7_native", SOURCE)
assert SPEC and SPEC.loader
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "source.py").write_text("# model implementation\n")
    for name in native.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELFfake native payload")
    runtime = {"python": "3.12", "torch": "2.10", "cxx11_abi": True}
    tilelang = {"version": "test", "files": {"target.py": "patched"}}
    monkeypatch.setattr(native, "runtime_identity", lambda _: dict(runtime))
    monkeypatch.setattr(native, "tilelang_identity", lambda *_: dict(tilelang))
    monkeypatch.setattr(native, "toolchain", lambda _: {"nvcc": "test"})
    monkeypatch.setattr(
        native,
        "architecture_list",
        lambda binary, _: ["80"] if "_sm80" in binary.name else ["70", "86", "89"],
    )
    native.write_manifest(root, Path("venv-python"), Path("cuda"), {"kind": "observed"})
    return root


def verify(root):
    return native.verify(root, Path("venv-python"), Path("cuda"))


def test_unchanged_inventory_verifies_without_claiming_rebuild(bundle):
    assert verify(bundle)["provenance"]["kind"] == "observed"


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_native_inventory_detects_mutations(bundle, mutation):
    path = bundle / next(iter(native.REQUIRED))
    if mutation == "changed":
        path.write_bytes(b"\x7fELFchanged")
    elif mutation == "missing":
        path.unlink()
    else:
        (path.parent / "unexpected.so").write_bytes(b"\x7fELFextra")
    with pytest.raises(ValueError, match="Native libraries differ|Missing required"):
        verify(bundle)


def test_runtime_abi_mismatch_is_rejected(bundle, monkeypatch):
    monkeypatch.setattr(native, "runtime_identity", lambda _: {"cxx11_abi": False})
    with pytest.raises(ValueError, match="ABI differs"):
        verify(bundle)


def test_dependency_patch_drift_is_rejected(bundle, monkeypatch):
    monkeypatch.setattr(native, "tilelang_identity", lambda *_: {"files": {}})
    with pytest.raises(ValueError, match="TileLang version/patch"):
        verify(bundle)


def test_source_drift_is_rejected(bundle):
    (bundle / "source.py").write_text("# unreviewed implementation\n")
    with pytest.raises(ValueError, match="source files differ"):
        verify(bundle)


def test_test_and_lint_caches_do_not_change_source_identity(bundle):
    for name in (".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__"):
        cache = bundle / name
        cache.mkdir()
        (cache / "mutable-cache").write_text("tool-generated")
    verify(bundle)


def test_build_outputs_are_not_installed_or_inventoried(bundle, tmp_path):
    build = bundle / "vllm/build"
    build.mkdir()
    (build / "temporary.so").write_bytes(b"\x7fELFtemporary")
    verify(bundle)
    destination = tmp_path / "installed"
    native.install(bundle, destination, Path("venv-python"), Path("cuda"))
    assert not (destination / "vllm/build").exists()
    verify(destination)


def test_architecture_coverage_is_checked(bundle, monkeypatch):
    monkeypatch.setattr(native, "architecture_list", lambda *_: ["70"])
    with pytest.raises(ValueError, match="Missing required library/architectures"):
        verify(bundle)


@pytest.mark.parametrize("path", ["../outside.so", "/absolute.so"])
def test_manifest_path_traversal_is_rejected(bundle, path):
    manifest_path = bundle / "pp7-native-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["libraries"][0]["path"] = path
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        verify(bundle)


def test_install_never_overwrites_existing_destination(bundle, tmp_path):
    destination = tmp_path / "active"
    destination.mkdir()
    marker = destination / "running.so"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError, match="new destination"):
        native.install(bundle, destination, Path("venv-python"), Path("cuda"))
    assert marker.read_text() == "preserve"


def test_install_new_directory_and_verify(bundle, tmp_path):
    destination = tmp_path / "candidate"
    native.install(bundle, destination, Path("venv-python"), Path("cuda"))
    assert verify(destination) == verify(bundle)


def test_install_rejects_recursive_destination(bundle):
    with pytest.raises(ValueError, match="outside the source"):
        native.install(bundle, bundle / "nested", Path("venv-python"), Path("cuda"))


def test_source_snapshot_excludes_environment_and_binary_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("# model\n")
    (source / ".env").write_text("SECRET=must-not-copy")
    (source / "native.so").write_bytes(b"native")
    (source / ".venv").mkdir()
    (source / ".venv/private").write_text("private")
    destination = tmp_path / "snapshot"
    identity = native.snapshot(source, destination)
    assert {p.name for p in destination.iterdir()} == {"model.py"}
    assert identity["tree_sha256"] == native.source_identity(destination)["tree_sha256"]


def test_elf_architecture_parser_preserves_actual_coverage(tmp_path, monkeypatch):
    binary = tmp_path / "native.so"
    binary.write_bytes(b"\x7fELFstub")
    monkeypatch.setattr(
        native,
        "output",
        lambda *_: (
            "ELF file 1: a.sm_70.cubin\n"
            "ELF file 2: b.sm_89.cubin\nELF file 3: c.sm_89.cubin"
        ),
    )
    assert native.architecture_list(binary, "cuobjdump") == ["70", "89"]
    binary.write_bytes(b"not an ELF")
    with pytest.raises(ValueError, match="Not an ELF"):
        native.architecture_list(binary, "cuobjdump")


def test_path_safety_rejects_source_escape(tmp_path):
    with pytest.raises(ValueError, match="Unsafe relative path"):
        native.contained(tmp_path, "../../outside")


def test_host_only_elf_is_inventoried_without_hiding_other_errors(
    tmp_path, monkeypatch
):
    binary = tmp_path / "host.so"
    binary.write_bytes(b"\x7fELFstub")

    def no_device_code(*args):
        raise subprocess.CalledProcessError(
            255, "cuobjdump", output="File host.so does not contain device code"
        )

    monkeypatch.setattr(native, "output", no_device_code)
    assert native.architecture_list(binary, "cuobjdump") == []

    def invalid(*args):
        raise subprocess.CalledProcessError(255, "cuobjdump", output="Invalid ELF")

    monkeypatch.setattr(native, "output", invalid)
    with pytest.raises(subprocess.CalledProcessError):
        native.architecture_list(binary, "cuobjdump")


def test_external_native_dependency_is_verified_by_content(bundle, tmp_path):
    external = tmp_path / "jit-cache.so"
    external.write_bytes(b"\x7fELFcompiled-cache")
    manifest_path = bundle / "pp7-native-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["external_dependencies"] = native.external_identity(
        [{"label": "test JIT", "files": [str(external)]}],
        Path("venv-python"),
        Path("cuobjdump"),
    )
    manifest_path.write_text(json.dumps(manifest))
    verify(bundle)
    external.write_bytes(b"\x7fELFrecompiled-cache")
    with pytest.raises(ValueError, match="External dependency"):
        verify(bundle)


def test_incomplete_dependency_fails_before_build_or_stage_creation(bundle, tmp_path):
    stage = tmp_path / "new-build"
    args = SimpleNamespace(source=bundle, stage=stage, cutlass=tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="Incomplete prepared dependency"):
        native.build(args)
    assert not stage.exists()


@pytest.fixture
def built_bundle(bundle):
    # If the probe imports any package instead of resolving its spec, fail.
    for name in native.BUNDLED_PACKAGES:
        directory = bundle / name
        directory.mkdir(exist_ok=True)
        (directory / "__init__.py").write_text(
            "raise RuntimeError('must not import')\n"
        )
    for module in native.BUNDLED_EXTENSIONS:
        path = bundle / (module.replace(".", "/") + ".so")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELFfake bundled extension")
    (bundle / "pp7-native-manifest.json").unlink()
    native.write_manifest(bundle, Path(sys.executable), Path("cuda"), {"kind": "built"})
    return bundle


def test_materialized_v100_package_is_complete_and_importable_from_root(tmp_path):
    root = tmp_path / "bundle"
    flash = root / "flash-attention-v100"
    source = flash / "flash_attn_v100"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("raise RuntimeError('must not import')\n")
    (source / "kernel-resource.bin").write_bytes(b"package resource")
    for name in ("flash_attn_v100_cuda", "paged_kv_utils"):
        (flash / (name + ".so")).write_bytes(b"\x7fELFstub")
    package = native.materialize_flash_package(flash, root)
    assert package == root / "flash_attn_v100"
    assert (package / "kernel-resource.bin").read_bytes() == b"package resource"
    layout = {
        "packages": {"flash_attn_v100": "flash_attn_v100/__init__.py"},
        "extensions": {
            "flash_attn_v100.flash_attn_v100_cuda": (
                "flash_attn_v100/flash_attn_v100_cuda.so"
            ),
            "flash_attn_v100.paged_kv_utils": "flash_attn_v100/paged_kv_utils.so",
        },
    }
    native.verify_bundled_imports(root, Path(sys.executable), layout)
    with pytest.raises(FileExistsError):
        native.materialize_flash_package(flash, root)


def test_built_bundle_relocation_verifies_actual_candidate_origins(
    built_bundle, tmp_path, monkeypatch
):
    # An inherited path to an old checkout must not supply the candidate's packages.
    old = tmp_path / "old"
    shutil.copytree(built_bundle, old)
    monkeypatch.setenv("PYTHONPATH", str(old))
    installed = tmp_path / "installed"
    native.install(built_bundle, installed, Path(sys.executable), Path("cuda"))
    manifest = native.verify(installed, Path(sys.executable), Path("cuda"))
    assert manifest["bundled_imports"]["packages"]["flash_attn_v100"] == (
        "flash_attn_v100/__init__.py"
    )


def test_built_bundle_rejects_nested_only_v100_package(built_bundle):
    nested = built_bundle / "flash-attention-v100"
    nested.mkdir()
    shutil.move(str(built_bundle / "flash_attn_v100"), nested)
    with pytest.raises(ValueError, match="Missing bundled package"):
        native.bundled_import_layout(built_bundle, [])


def test_built_bundle_rejects_missing_import_contract(built_bundle):
    path = built_bundle / "pp7-native-manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["bundled_imports"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bundled import layout"):
        native.verify(built_bundle, Path(sys.executable), Path("cuda"))


@pytest.mark.parametrize("name", ["flash_attn_v100_cuda", "paged_kv_utils"])
def test_top_level_v100_extension_tampering_is_detected(built_bundle, name):
    (built_bundle / "flash_attn_v100" / (name + ".so")).write_bytes(b"\x7fELFchanged")
    with pytest.raises(ValueError, match="Native libraries differ"):
        native.verify(built_bundle, Path(sys.executable), Path("cuda"))


def test_extension_origin_probe_rejects_incompatible_python_suffix(built_bundle):
    manifest = json.loads((built_bundle / "pp7-native-manifest.json").read_text())
    module = "flash_attn_v100.flash_attn_v100_cuda"
    original = built_bundle / manifest["bundled_imports"]["extensions"][module]
    invalid = original.with_name("flash_attn_v100_cuda.incompatible-python.so")
    original.rename(invalid)
    layout = native.bundled_import_layout(
        built_bundle, native.library_rows(built_bundle, Path("cuobjdump"))
    )
    with pytest.raises(ValueError, match="Bundled import origin mismatch"):
        native.verify_bundled_imports(built_bundle, Path(sys.executable), layout)


def test_build_does_not_inherit_triton_vendoring_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_KERNELS_SRC_DIR", "/old/python-package-only")
    env = native.build_environment(tmp_path, Path(sys.executable), Path("cuda"), 3)
    assert "TRITON_KERNELS_SRC_DIR" not in env
    assert env["MAX_JOBS"] == "3"
    assert env["PYTHONPATH"] == str(tmp_path)
