# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build, inventory and verify isolated Linux SM70/SM86/SM89 native bundles.

No command installs into an existing checkout or patches a running environment.
An observed inventory establishes file identity, not reproducible-build provenance.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import regex as re

SCHEMA = 1
TARGETS = ("_C", "_C_gguf", "_C_stable_libtorch", "_moe_C", "triton_kernels")
COMPANION_TARGETS = (
    "cumem_allocator",
    "spinloop",
    "_sm70_sampler_C",
    "_h3_w8a16_C",
    "_h3_flashinfer_C",
    "_h3_flashattn_C",
    "_sm70_qsa_indexer_C",
    "_sm70_router_C",
    "_sm70_qsa_device_C",
    "_sm70_exact_reduce_C",
    "_sm70_sparse_attention_C",
)
CORE = {f"vllm/{name}.abi3.so": {"70", "86", "89"} for name in TARGETS[:4]}
REQUIRED = CORE | {
    "vllm/vllm_flash_attn/_vllm_fa2_C_sm70.abi3.so": {"70"},
    # FA2 deliberately emits SM80 cubins for the validated SM80/86/89 group.
    # Cubins are compatible with higher minor revisions of the same major;
    # this does not authorize selecting the library for SM90 or newer.
    "vllm/vllm_flash_attn/_vllm_fa2_C_sm80.abi3.so": {"80"},
}
BUNDLED_PACKAGES = ("vllm", "flash_attn_v100", "flash_qla")
BUNDLED_EXTENSIONS = (
    "flash_attn_v100.flash_attn_v100_cuda",
    "flash_attn_v100.paged_kv_utils",
    "flash_qla.ops.gated_delta_rule.chunk.sm70.flash_qla_sm70_gdn_strided",
)
IGNORED = {
    ".git",
    ".venv",
    ".deps",
    "__pycache__",
    "build",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def output(argv, *, cwd=None, env=None):
    return subprocess.check_output(
        list(map(str, argv)), cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT
    ).strip()


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    return target


def source_files(root):
    """Snapshot source, excluding native/build/env outputs, without following links."""
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in IGNORED)
        for name in sorted(files):
            path = Path(current) / name
            if name in (".git", ".env", "pp7-native-manifest.json") or name.endswith(
                (".so", ".pyc", ".pyo")
            ):
                continue
            relative = path.relative_to(root)
            if path.is_symlink():
                raise ValueError(
                    f"Source symlink requires explicit materialization: {path}"
                )
            yield relative, path
        for name in directories:
            if (Path(current) / name).is_symlink():
                raise ValueError(f"Source directory symlink: {Path(current) / name}")


def source_identity(root):
    rows = [
        (relative.as_posix(), digest(path)) for relative, path in source_files(root)
    ]
    result = {
        "path": str(root),
        "files": len(rows),
        "tree_sha256": hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    try:
        top = Path(output(["git", "rev-parse", "--show-toplevel"], cwd=root))
        if top.resolve() != root.resolve():
            raise ValueError("Dependency is not its own Git checkout")
        result.update(
            commit=output(["git", "rev-parse", "HEAD"], cwd=root),
            submodules=output(["git", "submodule", "status", "--recursive"], cwd=root),
            dirty_diff_sha256=hashlib.sha256(
                subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
            ).hexdigest(),
            status=output(["git", "status", "--porcelain"], cwd=root),
        )
    except (subprocess.CalledProcessError, ValueError):
        result["commit"] = None
    return result


def snapshot(source, destination):
    identity = source_identity(source)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, path in source_files(source):
        target = contained(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if source_identity(destination)["tree_sha256"] != identity["tree_sha256"]:
        raise RuntimeError(f"Source changed during snapshot: {source}")
    return identity


def runtime_identity(python):
    code = """
import json, sys, sysconfig, torch
if sys.prefix == sys.base_prefix:
    raise RuntimeError('Use the deployment virtual environment Python')
print(json.dumps({'python': sys.version.split()[0],
 'soabi': sysconfig.get_config_var('SOABI'), 'torch': str(torch.__version__),
 'torch_cuda': torch.version.cuda,
 'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI)}))
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return json.loads(output([python, "-c", code], env=env))


def tilelang_identity(python, patch):
    code = """
import importlib.util, importlib.metadata, json
from pathlib import Path
spec = importlib.util.find_spec('tilelang')
if spec is None or spec.origin is None:
    raise RuntimeError('tilelang is not installed')
print(json.dumps({'root': str(Path(spec.origin).parent),
                  'version': importlib.metadata.version('tilelang')}))
"""
    info = json.loads(output([python, "-c", code]))
    root = Path(info["root"])
    # A reverse dry run proves the documented patch is present. This command
    # never modifies site-packages; a different upstream fix needs review.
    output(
        [
            "patch",
            "--dry-run",
            "--reverse",
            "--force",
            "--batch",
            "-p0",
            "-d",
            root.parent,
            "-i",
            patch.resolve(),
        ]
    )
    return {
        "version": info["version"],
        "patch_sha256": digest(patch),
        "files": {
            name: digest(root / name) for name in ("utils/target.py", "contrib/nvcc.py")
        },
    }


def architecture_list(binary, cuobjdump):
    with binary.open("rb") as stream:
        magic = stream.read(4)
    if magic != b"\x7fELF":
        raise ValueError(f"Not an ELF library: {binary}")
    try:
        listing = output([cuobjdump, "--list-elf", binary])
    except subprocess.CalledProcessError as error:
        # Host-only extensions (for example cumem_allocator) are still part
        # of the inventory. Required CUDA libraries are checked separately.
        if error.returncode == 255 and "does not contain device code" in error.output:
            return []
        raise
    return sorted(set(re.findall(r"\bsm_(\d+[a-z]?)\b", listing)))


def library_rows(root, cuobjdump):
    rows = []
    for directory in ("vllm", "flash-attention-v100", "flash_attn_v100", "flash_qla"):
        for path in sorted((root / directory).rglob("*.so")):
            if any(part in IGNORED for part in path.relative_to(root).parts):
                continue
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                    "architectures": architecture_list(path, cuobjdump),
                }
            )
    by_path = {row["path"]: row for row in rows}
    for name, arches in REQUIRED.items():
        if name not in by_path or not arches <= set(by_path[name]["architectures"]):
            raise ValueError(
                f"Missing required library/architectures: {name}: {arches}"
            )
    return rows


def materialize_flash_package(flash, bundle):
    """Make the complete V100 package importable from the bundle root alone."""
    package = bundle / "flash_attn_v100"
    shutil.copytree(
        flash / "flash_attn_v100",
        package,
        ignore=shutil.ignore_patterns(*IGNORED, "*.pyc", "*.pyo"),
    )
    for name in ("flash_attn_v100_cuda", "paged_kv_utils"):
        libraries = list(flash.glob(name + "*.so"))
        if len(libraries) != 1:
            raise RuntimeError(f"Expected one {name} library, got {libraries}")
        shutil.copy2(libraries[0], package / libraries[0].name)
    return package


def bundled_import_layout(root, libraries):
    """Require local critical extensions for builds; observed inventories may differ."""
    packages = {name: f"{name}/__init__.py" for name in BUNDLED_PACKAGES}
    for relative in packages.values():
        if not contained(root, relative).is_file():
            raise ValueError(f"Missing bundled package: {relative}")
    by_path = {row["path"]: row for row in libraries}
    extensions = {}
    for module in BUNDLED_EXTENSIONS:
        parent, name = module.rsplit(".", 1)
        directory = contained(root, parent.replace(".", "/"))
        matches = list(directory.glob(name + "*.so"))
        if len(matches) != 1:
            raise ValueError(f"Expected one bundled extension: {module}")
        relative = matches[0].relative_to(root).as_posix()
        contained(root, relative)
        row = by_path.get(relative)
        if row is None or "70" not in row["architectures"]:
            raise ValueError(f"Missing bundled SM70 extension: {relative}")
        extensions[module] = relative
    return {"packages": packages, "extensions": extensions}


def verify_bundled_imports(root, python, layout):
    # Resolve the candidate in its own activation context. Do not import the
    # package __init__ files or dlopen extensions: verification needs no GPU.
    code = """
import importlib.machinery, importlib.util, json, sys
from pathlib import Path
root = Path.cwd().resolve()
layout = json.loads(sys.argv[1])
origins = {}
for module, relative in layout['packages'].items():
    spec = importlib.util.find_spec(module)
    origins[module] = None if spec is None else spec.origin
for module, relative in layout['extensions'].items():
    spec = importlib.machinery.PathFinder.find_spec(
        module, [str((root / relative).parent)])
    origins[module] = None if spec is None else spec.origin
print(json.dumps(origins))
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root.resolve()))
    origins = json.loads(
        output([python, "-c", code, json.dumps(layout)], cwd=root, env=env)
    )
    for group in layout.values():
        for module, relative in group.items():
            expected = contained(root, relative).resolve()
            actual = origins.get(module)
            if actual is None or Path(actual).resolve() != expected:
                raise ValueError(
                    f"Bundled import origin mismatch: {module}: {actual} != {expected}"
                )


def build_environment(bundle, python, cuda_home, jobs):
    env = dict(
        os.environ,
        CUDA_HOME=str(cuda_home),
        CUDA_VISIBLE_DEVICES="",
        MAX_JOBS=str(jobs),
        NVCC_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=str(bundle),
    )
    # The CMake vendoring layout changes merely when this variable exists.
    # --triton always supplies the complete repository through FetchContent.
    env.pop("TRITON_KERNELS_SRC_DIR", None)
    env["PATH"] = (
        str(python.parent)
        + os.pathsep
        + str(cuda_home / "bin")
        + os.pathsep
        + env.get("PATH", "")
    )
    return env


def toolchain(cuda_home):
    tools = {"nvcc": cuda_home / "bin/nvcc", "cuobjdump": cuda_home / "bin/cuobjdump"}
    tools.update(
        {
            name: Path(shutil.which(name) or name)
            for name in ("gcc", "g++", "cmake", "ninja", "patch")
        }
    )
    return {
        name: {"path": str(path.resolve()), "version": output([path, "--version"])}
        for name, path in tools.items()
    }


def external_identity(entries, python, cuobjdump):
    result = []
    for entry in entries:
        row = {"label": entry["label"], "files": []}
        for file in entry["files"]:
            path = Path(file["path"] if isinstance(file, dict) else file).resolve()
            row["files"].append(
                {
                    "path": str(path),
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                    "architectures": architecture_list(path, cuobjdump)
                    if path.suffix == ".so"
                    else None,
                }
            )
        if module := entry.get("module"):
            if not re.fullmatch(r"[a-zA-Z_]\w*", module):
                raise ValueError("External package checks require a top-level module")
            code = """
import importlib.util, importlib.metadata, json, sys
spec = importlib.util.find_spec(sys.argv[1])
if spec is None:
    raise RuntimeError('External module is not importable: ' + sys.argv[1])
try:
    version = importlib.metadata.version(sys.argv[1])
except importlib.metadata.PackageNotFoundError:
    version = None
print(json.dumps({'origin': spec.origin, 'distribution_version': version}))
"""
            row["module"] = module
            row["package"] = json.loads(output([python, "-c", code, module]))
        result.append(row)
    return result


def write_manifest(root, python, cuda_home, provenance, *, source=None, external=()):
    patch = root / "docs/deployment/patches/tilelang-current-device.patch"
    manifest = {
        "schema": SCHEMA,
        "provenance": provenance,
        "source": source or source_identity(root),
        "bundle_source_sha256": source_identity(root)["tree_sha256"],
        "runtime": runtime_identity(python),
        "toolchain": toolchain(cuda_home),
        "tilelang": tilelang_identity(python, patch),
        "libraries": library_rows(root, cuda_home / "bin/cuobjdump"),
        "external_dependencies": external_identity(
            external, python, cuda_home / "bin/cuobjdump"
        ),
    }
    if provenance["kind"] == "built":
        layout = bundled_import_layout(root, manifest["libraries"])
        verify_bundled_imports(root, python, layout)
        manifest["bundled_imports"] = layout
    path = root / "pp7-native-manifest.json"
    # Exclude the manifest itself from source identity on subsequent records.
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(root, python, cuda_home):
    manifest = json.loads((root / "pp7-native-manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported native manifest schema")
    if manifest["runtime"] != runtime_identity(python):
        raise ValueError("Runtime Python/Torch/CUDA/C++ ABI differs from manifest")
    patch = root / "docs/deployment/patches/tilelang-current-device.patch"
    if manifest["tilelang"] != tilelang_identity(python, patch):
        raise ValueError("TileLang version/patch differs from manifest")
    if source_identity(root)["tree_sha256"] != manifest["bundle_source_sha256"]:
        raise ValueError("Bundle source files differ from manifest")
    # Reject lexical traversal. Native symlinks in observed checkouts are read
    # for identity checks, never written through; install materializes them.
    for row in manifest["libraries"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact path: {relative}")
    actual = library_rows(root, cuda_home / "bin/cuobjdump")
    if manifest["libraries"] != actual:
        raise ValueError("Native libraries differ: content, size or ELF architectures")
    if manifest["provenance"]["kind"] == "built":
        layout = bundled_import_layout(root, actual)
        if manifest.get("bundled_imports") != layout:
            raise ValueError("Built manifest lacks matching bundled import layout")
        verify_bundled_imports(root, python, layout)
    if manifest["external_dependencies"] != external_identity(
        manifest["external_dependencies"], python, cuda_home / "bin/cuobjdump"
    ):
        raise ValueError(
            "External dependency contents or package origin/version changed"
        )
    return manifest


def install(source, destination, python, cuda_home):
    verify(source, python, cuda_home)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "Install requires a new destination, never active libraries"
        )
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Install destination must be outside the source tree")
    # Materialize files, including native symlinks in an observed checkout.
    # No overwrite/merge operation is available.
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        ignore=shutil.ignore_patterns(*IGNORED, ".env"),
    )
    verify(destination, python, cuda_home)


def build(args):
    if args.stage.exists() or args.stage.is_symlink():
        raise FileExistsError("Build staging directory must be new")
    source = args.source.resolve()
    if args.stage.resolve().is_relative_to(source):
        raise ValueError("Build staging directory must be outside source")
    dependency_files = {
        "cutlass": ("CMakeLists.txt", "include/cutlass/cutlass.h"),
        "triton": ("CMakeLists.txt",),
        "fa2_sm70": (
            "CMakeLists.txt",
            "csrc/cutlass/include/cutlass/cutlass.h",
            "csrc/cutlass/examples/35_gemm_softmax/gemm_with_softmax.h",
            "csrc/flash_attn/src/flash_fwd_d256_splitd_sm70.cu",
            "csrc/flash_attn/src/flash_fwd_d256_gqa_arch_sm70.cu",
        ),
        "fa2_rtx": ("CMakeLists.txt", "csrc/cutlass/include/cutlass/cutlass.h"),
    }
    for name, files in dependency_files.items():
        for relative in files:
            path = getattr(args, name) / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"Incomplete prepared dependency: {path}. "
                    "Initialize pinned submodules and apply the documented patches."
                )
    # Refuse an unpatched runtime before spending time compiling.
    runtime_identity(args.python)
    tilelang_identity(
        args.python, source / "docs/deployment/patches/tilelang-current-device.patch"
    )
    args.stage.mkdir(parents=True)
    bundle = args.stage / "bundle"
    identity = snapshot(source, bundle)
    dependencies = {}
    for name in ("cutlass", "triton", "fa2_sm70", "fa2_rtx"):
        original = getattr(args, name).resolve()
        destination = args.stage / "dependencies" / name
        dependencies[name] = snapshot(original, destination)
    commands = []
    env = build_environment(bundle, args.python, args.cuda_home, args.jobs)
    if shutil.which("patchelf", path=env["PATH"]) is None:
        raise RuntimeError("patchelf is required before starting the native build")
    patchelf_version = output(["patchelf", "--version"], env=env)

    def run(argv, *, cwd=None, arches="7.0;8.6;8.9"):
        command = list(map(str, argv))
        commands.append({"argv": command, "cwd": str(cwd or bundle), "arches": arches})
        (args.stage / "build-commands.json").write_text(json.dumps(commands, indent=2))
        subprocess.run(
            command,
            cwd=cwd or bundle,
            env=dict(env, TORCH_CUDA_ARCH_LIST=arches),
            check=True,
        )

    def configure(directory, arches, fa2):
        dep = args.stage / "dependencies"
        run(
            [
                "cmake",
                "-S",
                bundle,
                "-B",
                directory,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DVLLM_TARGET_DEVICE=cuda",
                f"-DVLLM_PYTHON_EXECUTABLE={args.python}",
                f"-DCMAKE_INSTALL_PREFIX={bundle}",
                f"-DCMAKE_CUDA_COMPILER={args.cuda_home}/bin/nvcc",
                f"-DCMAKE_CUDA_ARCHITECTURES={arches}",
                "-DNVCC_THREADS=1",
                f"-DFETCHCONTENT_SOURCE_DIR_CUTLASS={dep}/cutlass",
                f"-DFETCHCONTENT_SOURCE_DIR_TRITON_KERNELS={dep}/triton",
                f"-DFETCHCONTENT_SOURCE_DIR_VLLM-FLASH-ATTN={dep}/{fa2}",
                "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
                "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            ],
            arches=";".join(f"{a[0]}.{a[1:]}" for a in arches.split(";")),
        )

    native = args.stage / "native-build"
    configure(native, "70;86;89", "fa2_rtx")
    run(
        [
            "cmake",
            "--build",
            native,
            "-j",
            args.jobs,
            "--target",
            *TARGETS,
            *COMPANION_TARGETS,
        ]
    )
    for target in (*TARGETS, *COMPANION_TARGETS):
        run(["cmake", "--install", native, "--component", target])
    volta = args.stage / "volta-build"
    configure(volta, "70", "fa2_sm70")
    run(
        ["cmake", "--build", volta, "-j", args.jobs, "--target", "_vllm_fa2_C"],
        arches="7.0",
    )
    run(["cmake", "--install", volta, "--component", "_vllm_fa2_C"], arches="7.0")
    fa_directory = bundle / "vllm/vllm_flash_attn"
    shutil.copy2(
        fa_directory / "_vllm_fa2_C.abi3.so", fa_directory / "_vllm_fa2_C_sm70.abi3.so"
    )
    rtx = args.stage / "rtx-build"
    run(
        [
            "cmake",
            "-S",
            args.stage / "dependencies/fa2_rtx",
            "-B",
            rtx,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DVLLM_PYTHON_EXECUTABLE={args.python}",
            f"-DPython_EXECUTABLE={args.python}",
            f"-DCMAKE_CUDA_COMPILER={args.cuda_home}/bin/nvcc",
            "-DCMAKE_CUDA_ARCHITECTURES=86;89",
            "-DVLLM_GPU_ARCHES=86-real;89-real",
            "-DNVCC_THREADS=1",
        ],
        arches="8.6;8.9",
    )
    run(
        ["cmake", "--build", rtx, "-j", args.jobs, "--target", "_vllm_fa2_C"],
        arches="8.6;8.9",
    )
    shutil.copy2(rtx / "_vllm_fa2_C.abi3.so", fa_directory / "_vllm_fa2_C_sm80.abi3.so")
    flash = bundle / "flash-attention-v100"
    run([args.python, "setup.py", "build_ext", "--inplace"], cwd=flash, arches="7.0")
    flash_package = materialize_flash_package(flash, bundle)
    qla = bundle / "flash_qla/ops/gated_delta_rule/chunk/sm70"
    qla_build = args.stage / "qla-build"
    qla_build.mkdir()
    code = """
import shutil, sys
from pathlib import Path
from torch.utils.cpp_extension import load
source, work = map(Path, sys.argv[1:])
module = load(name='flash_qla_sm70_gdn_strided',
 sources=[str(source/'csrc/gdn_forward.cu')], build_directory=str(work),
 extra_cuda_cflags=['-O3','-gencode=arch=compute_70,code=sm_70'],
 extra_cflags=['-O3'], with_cuda=True, verbose=True)
shutil.copy2(module.__file__, source/Path(module.__file__).name)
"""
    run([args.python, "-c", code, qla, qla_build], arches="7.0")
    # Match the deployed recipe: these Python extensions must resolve Torch
    # from the runtime environment, not from an absolute staging RPATH.
    for directory in (flash, flash_package, qla):
        for library in directory.rglob("*.so"):
            run(["patchelf", "--remove-rpath", library], arches="7.0")
    provenance = {
        "kind": "built",
        "dependencies": dependencies,
        "commands": commands,
        "jobs": args.jobs,
        "nvcc_threads": 1,
        "patchelf": patchelf_version,
        "compiler_environment": {
            key: env.get(key)
            for key in (
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
            )
        },
        "cmake_cache": {
            name: (args.stage / name / "CMakeCache.txt").read_text()
            for name in ("native-build", "volta-build", "rtx-build")
        },
        "note": "Build success does not establish GPU numerical correctness",
    }
    write_manifest(bundle, args.python, args.cuda_home, provenance, source=identity)
    verify(bundle, args.python, args.cuda_home)
    print(f"Built and verified bundle: {bundle}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cuda-home", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    builder.add_argument("--stage", type=Path, required=True)
    builder.add_argument("--jobs", type=int, default=4)
    for name in ("cutlass", "triton", "fa2-sm70", "fa2-rtx"):
        builder.add_argument("--" + name, type=Path, required=True)
    for command in ("record", "verify", "install"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--root", type=Path, required=True)
        if command == "record":
            subparser.add_argument(
                "--external-json",
                type=Path,
                help="Explicit external native/package file inventory",
            )
        if command == "install":
            subparser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.python = (
        args.python.absolute()
    )  # Preserve the venv executable, not its symlink target.
    args.cuda_home = args.cuda_home.resolve()
    if args.command == "build":
        if args.jobs < 1:
            parser.error("--jobs must be positive")
        args.stage = args.stage.absolute()
        build(args)
    elif args.command == "record":
        write_manifest(
            args.root,
            args.python,
            args.cuda_home,
            {"kind": "observed", "note": "Existing files; build provenance unknown"},
            external=json.loads(args.external_json.read_text())
            if args.external_json
            else (),
        )
        print("Observed inventory recorded; no rebuild was performed")
    elif args.command == "verify":
        verify(args.root, args.python, args.cuda_home)
        print("Verified native files, architectures, runtime ABI and TileLang patch")
    else:
        install(args.root, args.destination, args.python, args.cuda_home)
        print(f"Installed into new directory: {args.destination}")


if __name__ == "__main__":
    main()
