#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate that a 1Cat SM70 wheel is self-contained before publication.

This is an artifact check, not a CUDA correctness or throughput test.  It
rejects the two delivery failures that are easy to miss in a successful wheel
build: a missing SM70 companion extension and a build-host RPATH that makes a
wheel depend on a private cache or worktree.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import regex as re

REQUIRED_MEMBERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vllm main extension", re.compile(r"^vllm/_C(?:\.abi3|\.cpython-[^/]+)\.so$")),
    (
        "vllm stable libtorch extension",
        re.compile(r"^vllm/_C_stable_libtorch(?:\.abi3|\.cpython-[^/]+)\.so$"),
    ),
    (
        "SM70 sampler extension",
        re.compile(r"^vllm/_sm70_sampler_C(?:\.abi3|\.cpython-[^/]+)\.so$"),
    ),
    (
        "SM70 exact-reduce extension",
        re.compile(r"^vllm/_sm70_exact_reduce_C(?:\.abi3|\.cpython-[^/]+)\.so$"),
    ),
    (
        "SM70 sparse-attention extension",
        re.compile(r"^vllm/_sm70_sparse_attention_C(?:\.abi3|\.cpython-[^/]+)\.so$"),
    ),
    (
        "SM70 H3 W8A16 extension",
        re.compile(r"^vllm/_h3_w8a16_C(?:\.abi3|\.cpython-[^/]+)\.so$"),
    ),
    (
        "SM70 H3 FlashInfer extension",
        re.compile(r"^vllm/_h3_flashinfer_C(?:\.abi3|\.cpython-[^/]+)\.so$"),
    ),
    (
        "SM70 H3 FlashAttention extension",
        re.compile(r"^vllm/_h3_flashattn_C(?:\.abi3|\.cpython-[^/]+)\.so$"),
    ),
    (
        "SM70 FlashAttention extension",
        re.compile(r"^vllm/vllm_flash_attn/_vllm_fa2_C(?:\.abi3|\.cpython-[^/]+)\.so$"),
    ),
    (
        "Flash-V100 attention extension",
        re.compile(r"^flash_attn_v100/flash_attn_v100_cuda[^/]*\.so$"),
    ),
    (
        "Flash-V100 paged-KV extension",
        re.compile(r"^flash_attn_v100/paged_kv_utils[^/]*\.so$"),
    ),
    (
        "bundled FlashQLA extension",
        re.compile(r"^flash_qla/ops/gated_delta_rule/chunk/sm70/[^/]+\.so$"),
    ),
)

LAUNCHER_SUFFIX = ".data/scripts/serve_qwen38_27b_nvfp4_v100.sh"
WHEEL_METADATA_PATTERN = re.compile(r"^[^/]+\.dist-info/WHEEL$")
RELEASE_PYTHON_TAG = "cp312"
PRIVATE_PATH_MARKERS = (
    "/home/",
    "/data/",
    "worktree",
    ".cache/",
    "/tmp/",
)


class ArtifactError(ValueError):
    """Raised when a release wheel cannot run from a clean installation."""


def _dynamic_section_uses_private_path(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in PRIVATE_PATH_MARKERS)


def _dynamic_section_errors(member: str, output: str) -> list[str]:
    """Reject loader paths that escape the wheel or name an absolute DSO."""

    errors: list[str] = []
    if _dynamic_section_uses_private_path(output):
        errors.append(f"{member}: private build path in dynamic section")

    for line in output.splitlines():
        if "(RPATH)" not in line and "(RUNPATH)" not in line:
            continue
        match = re.search(r"\[(.*?)\]", line)
        if match is None:
            continue
        for path in match.group(1).split(":"):
            path = path.strip()
            if not path:
                continue
            if path.startswith("$ORIGIN") or path.startswith("${ORIGIN}"):
                continue
            errors.append(f"{member}: non-portable RPATH/RUNPATH entry {path!r}")

    # A DT_NEEDED entry with a slash is an absolute or caller-relative loader
    # path.  A wheel may rely on system CUDA/driver libraries by soname, but it
    # must not reach back into the build host for a private .so.
    for match in re.finditer(r"Shared library: \[(.*?)\]", output):
        needed = match.group(1)
        if "/" in needed or "\\" in needed:
            errors.append(f"{member}: non-portable DT_NEEDED entry {needed!r}")
    return errors


def _check_dynamic_dependencies(
    wheel: zipfile.ZipFile,
    native_members: list[str],
) -> list[str]:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise ArtifactError("readelf is required for the SM70 wheel RPATH check")

    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="1cat-sm70-wheel-") as temp_dir:
        root = Path(temp_dir)
        for member in native_members:
            target = root / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wheel.read(member))
            result = subprocess.run(
                [readelf, "-d", str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append(f"{member}: readelf failed: {result.stderr.strip()}")
                continue
            errors.extend(_dynamic_section_errors(member, result.stdout))
    return errors


def check_wheel(path: Path, *, inspect_dynamic: bool = True) -> None:
    """Validate one wheel and raise ``ArtifactError`` on a delivery defect."""

    if not path.is_file():
        raise ArtifactError(f"wheel does not exist: {path}")

    with zipfile.ZipFile(path) as wheel:
        member_list = wheel.namelist()
        members = set(member_list)
        errors: list[str] = []

        if len(members) != len(member_list):
            errors.append("wheel contains duplicate member names")

        for member in member_list:
            if "\\" in member or "\x00" in member:
                errors.append(f"unsafe wheel member: {member}")
                continue
            pure = PurePosixPath(member)
            if pure.is_absolute() or ".." in pure.parts:
                errors.append(f"unsafe wheel member: {member}")

        for description, pattern in REQUIRED_MEMBERS:
            if not any(pattern.fullmatch(member) for member in members):
                errors.append(f"missing {description}")

        if not any(member.endswith(LAUNCHER_SUFFIX) for member in members):
            errors.append("missing packaged V100 release launcher")

        wheel_metadata = [
            member for member in members if WHEEL_METADATA_PATTERN.fullmatch(member)
        ]
        if not wheel_metadata:
            errors.append("missing wheel WHEEL metadata")
        else:
            tags = []
            for member in wheel_metadata:
                metadata = wheel.read(member).decode("utf-8", errors="replace")
                tags.extend(
                    line.partition(":")[2].strip()
                    for line in metadata.splitlines()
                    if line.startswith("Tag:")
                )
            if not tags:
                errors.append("wheel WHEEL metadata has no Tag entry")
            elif not any(tag.startswith(f"{RELEASE_PYTHON_TAG}-") for tag in tags):
                errors.append(
                    "SM70 release wheel must publish a Python 3.12-compatible Tag"
                )

        # Do not inspect or extract native members until all member names have
        # passed validation.  This keeps a malformed ZIP from writing outside
        # the temporary directory used by the dynamic dependency check.
        if errors:
            raise ArtifactError("; ".join(errors))

        native_members = sorted(
            member for member in members if re.search(r"\.so(?:\..*)?$", member)
        )
        if inspect_dynamic:
            errors.extend(_check_dynamic_dependencies(wheel, native_members))

        if errors:
            raise ArtifactError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        check_wheel(args.wheel)
    except (ArtifactError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(f"SM70 release artifact OK: {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
