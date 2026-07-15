# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect and patch the isolated SM70 HMMA.884 WMMA scheduling cubin.

This is deliberately a native-cubin patcher rather than a source-level fence
experiment.  It swaps two complete 16-byte SASS bundles in a cubin generated
from ``sm70_hmma884_schedule_micro.cu`` and then verifies the changed order
with cuobjdump.  The default patch reuses the original ELF metadata and
control codes; an explicit wait-mask variant is available for one bounded
scoreboard experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KERNEL_NAME = "hmma884_patch_target_kernel"
TEXT_SECTION_PREFIX = ".text."
INSTRUCTION_BYTES = 16
CONTROL_BYTES = 8
WAIT_MASK_SHIFT = 52
WAIT_MASK_BITS = 0x3F

FUNCTION_RE = re.compile(r"^\s*Function\s*:\s*(?P<name>\S+)\s*$", re.M)
INSTRUCTION_RE = re.compile(
    r"^\s*/\*(?P<offset>[0-9a-fA-F]{4,})\*/\s*"
    r"(?P<asm>.*?)\s*/\*\s*0x(?P<encoding>[0-9a-fA-F]{16})\s*\*/\s*$"
)
CONTROL_RE = re.compile(r"^\s*/\*\s*0x(?P<control>[0-9a-fA-F]{16})\s*\*/\s*$")
HMMA_RE = re.compile(
    r"\bHMMA\.884\.[A-Z0-9.]+\.STEP(?P<step>[0-3])\s+"
    r"R(?P<destination>\d+),\s*R(?P<a>\d+)(?:\.[A-Za-z]+)*,\s*"
    r"R(?P<b>\d+)(?:\.[A-Za-z]+)*,\s*(?:R(?P<c>\d+)|RZ)"
)
LDG_RE = re.compile(
    r"\bLDG\.E\.64(?:\.[A-Z]+)*\s+R(?P<destination>\d+),\s*"
    r"\[R(?P<base>\d+)(?P<offset>\+0x[0-9a-fA-F]+)?\]"
)
REGISTER_DESTINATION_RE = re.compile(r"\bR(?P<destination>\d+)(?:,|\s)")


@dataclass(frozen=True)
class SassInstruction:
    offset: int
    assembly: str
    encoding: str
    control: str

    @property
    def bundle_hex(self) -> str:
        return f"{self.control}{self.encoding}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--candidate-cubin", type=Path, required=True)
    parser.add_argument("--kernel", default=KERNEL_NAME)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--turingas-dir", type=Path)
    parser.add_argument(
        "--moved-ldg-wait-mask",
        type=lambda value: int(value, 0),
        help="Replace only the moved B1 LDG wait mask (0x0 through 0x3f).",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("run:", " ".join(command), file=sys.stderr)
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is required for the SASS scheduling probe.")
    return path


def _function_section(sass: str, kernel: str) -> str:
    headers = list(FUNCTION_RE.finditer(sass))
    for index, header in enumerate(headers):
        if header.group("name") != kernel:
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(sass)
        return sass[header.start() : end]
    raise RuntimeError(f"cuobjdump did not find function {kernel!r}.")


def _dump_sass(
    cubin: Path, kernel: str, verbose: bool
) -> tuple[str, list[SassInstruction]]:
    cuobjdump = _require_tool("cuobjdump")
    result = _run(
        [cuobjdump, "--dump-sass", "-fun", kernel, str(cubin)], verbose=verbose
    )
    if result.returncode:
        raise RuntimeError(
            f"cuobjdump failed for {cubin}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    section = _function_section(result.stdout, kernel)
    instructions: list[SassInstruction] = []
    lines = section.splitlines()
    index = 0
    while index < len(lines):
        instruction_match = INSTRUCTION_RE.match(lines[index])
        if instruction_match is None:
            index += 1
            continue
        if index + 1 >= len(lines):
            raise RuntimeError(
                "cuobjdump ended after a SASS instruction without control code."
            )
        control_match = CONTROL_RE.match(lines[index + 1])
        if control_match is None:
            raise RuntimeError(
                "cuobjdump did not emit a control-code line after "
                f"instruction at {instruction_match.group('offset')}."
            )
        instructions.append(
            SassInstruction(
                offset=int(instruction_match.group("offset"), 16),
                assembly=instruction_match.group("asm").strip(),
                encoding=instruction_match.group("encoding"),
                control=control_match.group("control"),
            )
        )
        index += 2
    if not instructions:
        raise RuntimeError("No SASS instructions were parsed from cuobjdump output.")
    return section, instructions


def _hmma_groups(instructions: list[SassInstruction]) -> list[list[SassInstruction]]:
    hmma = [
        instruction
        for instruction in instructions
        if HMMA_RE.search(instruction.assembly)
    ]
    if len(hmma) != 32:
        raise RuntimeError(f"Expected 32 HMMA.884 instructions, found {len(hmma)}.")
    groups: list[list[SassInstruction]] = []
    for start in range(0, len(hmma), 4):
        group = hmma[start : start + 4]
        steps = [int(HMMA_RE.search(item.assembly).group("step")) for item in group]
        if steps != [0, 1, 2, 3]:
            raise RuntimeError(
                f"HMMA sequence at 0x{group[0].offset:x} is not STEP0..STEP3: {steps}."
            )
        groups.append(group)
    return groups


def _hmma_operands(group: list[SassInstruction]) -> dict[str, int]:
    matches = [HMMA_RE.search(item.assembly) for item in group]
    assert all(match is not None for match in matches)
    b_values = {int(match.group("b")) for match in matches if match is not None}
    a_values = {int(match.group("a")) for match in matches if match is not None}
    if len(a_values) != 1 or len(b_values) != 1:
        raise RuntimeError("Expected one A and one B base register per STEP0..3 group.")
    return {"a": next(iter(a_values)), "b": next(iter(b_values))}


def _control_summary(control: str) -> dict[str, int | bool | None]:
    word = int(control, 16)
    write_barrier = (word >> 46) & 0x7
    read_barrier = (word >> 49) & 0x7
    return {
        "raw": control,
        "stall": (word >> 41) & 0xF,
        "yield": not bool((word >> 45) & 0x1),
        "write_barrier": None if write_barrier == 7 else write_barrier,
        "read_barrier": None if read_barrier == 7 else read_barrier,
        "wait_mask": (word >> 52) & 0x3F,
    }


def _with_wait_mask(control: str, wait_mask: int | None) -> str:
    if wait_mask is None:
        return control
    if not 0 <= wait_mask <= WAIT_MASK_BITS:
        raise ValueError(
            f"wait mask must be between 0 and 0x{WAIT_MASK_BITS:x}, got {wait_mask}."
        )
    word = int(control, 16)
    word &= ~(WAIT_MASK_BITS << WAIT_MASK_SHIFT)
    word |= wait_mask << WAIT_MASK_SHIFT
    return f"{word:016x}"


def _instruction_destination(instruction: SassInstruction) -> int | None:
    match = REGISTER_DESTINATION_RE.search(instruction.assembly)
    return int(match.group("destination")) if match is not None else None


def _find_patch(
    instructions: list[SassInstruction], groups: list[list[SassInstruction]]
) -> dict[str, Any]:
    # One M16N16K16 WMMA is four STEP0..3 groups.  The second WMMA reuses the
    # same source BFragment, so its first B register is the one to overwrite.
    first_wmma = groups[:4]
    second_wmma = groups[4:]
    first_b = _hmma_operands(first_wmma[0])["b"]
    second_b = _hmma_operands(second_wmma[0])["b"]
    if first_b != second_b:
        raise RuntimeError(
            "The generated kernel did not reuse the first B sub-register "
            "across K16 tiles."
        )
    first_b_last_use = first_wmma[0][-1]

    # The probe's B tile is 16*16 fp16 = 0x200 bytes.  This must identify the
    # second K16 load that overwrites the first B sub-register, not an A load.
    second_b_loads = []
    for instruction in instructions:
        match = LDG_RE.search(instruction.assembly)
        if match is None or int(match.group("destination")) != second_b:
            continue
        if match.group("offset") is None or int(match.group("offset"), 16) != 0x200:
            continue
        second_b_loads.append(instruction)
    if len(second_b_loads) != 1:
        raise RuntimeError(
            "Expected one second-K16 LDG.E.64 for the reused B sub-register; "
            f"found {len(second_b_loads)}."
        )
    moved = second_b_loads[0]

    # Pick the existing A1 LDG that ptxas scheduled between STEP1 and STEP2 of
    # the third first-K16 group.  It is independent of B0 R10, whose last use
    # was the first group's STEP3.  Swapping whole bundles keeps instruction
    # count, code size, register allocation, and all encoded control fields.
    window_start = first_wmma[2][1].offset
    window_end = first_wmma[2][2].offset
    candidate_slots = [
        item
        for item in instructions
        if window_start < item.offset < window_end
        and item.assembly.startswith("LDG.E.128")
    ]
    if len(candidate_slots) != 1:
        raise RuntimeError(
            "Expected exactly one A1 LDG.E.128 in the selected STEP1/STEP2 window; "
            f"found {len(candidate_slots)}."
        )
    displaced = candidate_slots[0]
    displaced_destination = _instruction_destination(displaced)
    if displaced_destination is None:
        raise RuntimeError("Could not read the displaced LDG destination register.")
    if displaced.offset <= first_b_last_use.offset:
        raise RuntimeError("Candidate LDG window precedes B0's final use.")
    first_wmma_after_window = [
        item for group in first_wmma for item in group if item.offset > displaced.offset
    ]
    for item in first_wmma_after_window:
        match = HMMA_RE.search(item.assembly)
        assert match is not None
        if displaced_destination in (int(match.group("a")), int(match.group("b"))):
            raise RuntimeError(
                "The displaced A1 load would overwrite a live first-K16 HMMA operand."
            )
    if moved.offset <= displaced.offset:
        raise RuntimeError("Selected B1 LDG is not later than the candidate window.")

    return {
        "moved": moved,
        "displaced": displaced,
        "first_b_register": first_b,
        "first_b_last_use": first_b_last_use,
        "window_group": 2,
        "window_between_steps": [1, 2],
        "second_wmma_first_hmma": second_wmma[0][0],
    }


def _elf_section(data: bytes, section_name: str) -> tuple[int, int]:
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data, 0)
    section_header_offset = header[6]
    section_header_size = header[11]
    section_count = header[12]
    string_section_index = header[13]
    if section_header_size != 64:
        raise RuntimeError(
            f"Unexpected ELF64 section-header size {section_header_size}."
        )
    string_header_offset = (
        section_header_offset + string_section_index * section_header_size
    )
    string_header = struct.unpack_from("<IIQQQQIIQQ", data, string_header_offset)
    string_start, string_size = string_header[4], string_header[5]
    strings = data[string_start : string_start + string_size]
    for index in range(section_count):
        offset = section_header_offset + index * section_header_size
        values = struct.unpack_from("<IIQQQQIIQQ", data, offset)
        name_offset, section_offset, section_size = values[0], values[4], values[5]
        end = strings.find(b"\0", name_offset)
        if end < 0:
            raise RuntimeError("Malformed ELF section-name string table.")
        name = strings[name_offset:end].decode("ascii")
        if name == section_name:
            return section_offset, section_size
    raise RuntimeError(f"ELF section {section_name!r} was not found.")


def _patch_native_cubin(
    source: Path,
    candidate: Path,
    kernel: str,
    moved: SassInstruction,
    displaced: SassInstruction,
    moved_ldg_wait_mask: int | None,
) -> dict[str, Any]:
    data = bytearray(source.read_bytes())
    section_offset, section_size = _elf_section(data, f"{TEXT_SECTION_PREFIX}{kernel}")
    if section_size % INSTRUCTION_BYTES:
        raise RuntimeError(
            "The SASS text section is not a whole number of 16-byte bundles."
        )
    for instruction in (moved, displaced):
        if instruction.offset % INSTRUCTION_BYTES:
            raise RuntimeError(
                f"SASS offset 0x{instruction.offset:x} is not 16-byte aligned."
            )
        if instruction.offset + INSTRUCTION_BYTES > section_size:
            raise RuntimeError(
                f"SASS offset 0x{instruction.offset:x} is outside .text."
            )
    moved_start = section_offset + moved.offset
    displaced_start = section_offset + displaced.offset
    moved_bundle = bytes(data[moved_start : moved_start + INSTRUCTION_BYTES])
    displaced_bundle = bytes(
        data[displaced_start : displaced_start + INSTRUCTION_BYTES]
    )
    data[moved_start : moved_start + INSTRUCTION_BYTES] = displaced_bundle
    data[displaced_start : displaced_start + INSTRUCTION_BYTES] = moved_bundle
    original_moved_control = moved.control
    candidate_moved_control = _with_wait_mask(
        original_moved_control, moved_ldg_wait_mask
    )
    if candidate_moved_control != original_moved_control:
        control_offset = displaced_start + CONTROL_BYTES
        encoded_control = struct.unpack_from("<Q", data, control_offset)[0]
        if encoded_control != int(original_moved_control, 16):
            raise RuntimeError("Moved LDG control word did not match cuobjdump output.")
        struct.pack_into("<Q", data, control_offset, int(candidate_moved_control, 16))
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(data)
    return {
        "text_section_offset": section_offset,
        "text_section_size": section_size,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(data).hexdigest(),
        "bundle_bytes": INSTRUCTION_BYTES,
        "moved_ldg_control": {
            "source": original_moved_control,
            "candidate": candidate_moved_control,
        },
    }


def _sass_summary(
    instructions: list[SassInstruction], groups: list[list[SassInstruction]]
) -> dict[str, Any]:
    counters = {
        mnemonic: sum(
            re.search(rf"\b{mnemonic}\b", item.assembly) is not None
            for item in instructions
        )
        for mnemonic in ("HMMA", "LDG", "LDS", "BAR", "LDL", "STL")
    }
    group_summary = []
    for index, group in enumerate(groups):
        operands = _hmma_operands(group)
        group_summary.append(
            {
                "index": index,
                "offsets": [f"0x{item.offset:04x}" for item in group],
                "a_register": operands["a"],
                "b_register": operands["b"],
                "destination_registers": [
                    int(HMMA_RE.search(item.assembly).group("destination"))
                    for item in group
                ],
            }
        )
    return {"instruction_counts": counters, "hmma_step_groups": group_summary}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _turingas_probe(
    cubin: Path,
    kernel: str,
    native_b1_ldg: SassInstruction,
    work_dir: Path,
    turingas_dir: Path | None,
    verbose: bool,
) -> dict[str, Any]:
    if turingas_dir is None:
        default = Path("/tmp/1cat-turingas-hmma884")
        turingas_dir = default if default.is_dir() else None
    if turingas_dir is None or not (turingas_dir / "turingas" / "main.py").is_file():
        return {"available": False, "reason": "TuringAs checkout was not found."}

    python = sys.executable
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(turingas_dir)
        if not environment.get("PYTHONPATH")
        else f"{turingas_dir}:{environment['PYTHONPATH']}"
    )
    git = shutil.which("git")
    revision = None
    if git is not None:
        result = _run(
            [git, "-C", str(turingas_dir), "rev-parse", "HEAD"],
            verbose=verbose,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()

    extracted = work_dir / "turingas-native-extracted.sass"
    disassembler = turingas_dir / "tools" / "disasm.py"
    extract = _run(
        [python, str(disassembler), str(cubin), "-fun", kernel],
        env=environment,
        verbose=verbose,
    )
    _write_text(extracted, extract.stdout)
    native_roundtrip = work_dir / "turingas-native-roundtrip.cubin"
    roundtrip = _run(
        [
            python,
            "-m",
            "turingas.main",
            "-i",
            str(extracted),
            "-o",
            str(native_roundtrip),
            "-arch",
            "70",
        ],
        env=environment,
        verbose=verbose,
    )

    core_source = work_dir / "turingas-hmma884-core.sass"
    _write_text(
        core_source,
        (
            "<PARAMS>\nargument, 8\n</PARAMS>\n"
            "--:-:-:Y:4 LDG.E.64.SYS R10, [R30+0x200];\n"
            "--:-:-:Y:2 HMMA.884.F32.F32.STEP0 R16, R4.ROW, R10.ROW, R16;\n"
            "--:-:-:Y:2 HMMA.884.F32.F32.STEP1 R18, R4.ROW, R10.ROW, R18;\n"
            "--:-:-:Y:2 HMMA.884.F32.F32.STEP2 R8, R4.ROW, R10.ROW, R8;\n"
            "--:-:-:Y:2 HMMA.884.F32.F32.STEP3 R10, R4.ROW, R10.ROW, R10;\n"
            "--:-:-:Y:5 EXIT;\n"
        ),
    )
    core_cubin = work_dir / "turingas-hmma884-core.cubin"
    core = _run(
        [
            python,
            "-m",
            "turingas.main",
            "-i",
            str(core_source),
            "-o",
            str(core_cubin),
            "-arch",
            "70",
            "-name",
            "hmma884_core",
        ],
        env=environment,
        verbose=verbose,
    )
    core_sass = None
    encoding_checks: dict[str, bool] | None = None
    if core.returncode == 0 and core_cubin.is_file():
        try:
            core_sass, core_instructions = _dump_sass(
                core_cubin, "hmma884_core", verbose
            )
            core_ldg = next(
                item
                for item in core_instructions
                if item.assembly.startswith("LDG.E.64")
            )
            core_hmma = next(
                item
                for item in core_instructions
                if item.assembly.startswith("HMMA.884.F32.F32.STEP0")
            )
            _, native_instructions = _dump_sass(cubin, kernel, verbose)
            native_hmma = next(
                item
                for item in native_instructions
                if item.assembly == "HMMA.884.F32.F32.STEP0 R16, R4.reuse.ROW, "
                "R10.reuse.ROW, R16 ;"
            )
            encoding_checks = {
                "ldg_instruction_bits_match_native": (
                    core_ldg.encoding == native_b1_ldg.encoding
                ),
                "hmma_instruction_bits_match_native": (
                    core_hmma.encoding == native_hmma.encoding
                ),
            }
        except RuntimeError as error:
            core_sass = f"cuobjdump failed: {error}"

    return {
        "available": True,
        "path": str(turingas_dir),
        "revision": revision,
        "native_disassembly": {
            "returncode": extract.returncode,
            "path": str(extracted),
            "stderr": extract.stderr[-1000:],
        },
        "native_roundtrip": {
            "returncode": roundtrip.returncode,
            "output_exists": native_roundtrip.is_file(),
            "stdout_tail": roundtrip.stdout[-1200:],
            "stderr_tail": roundtrip.stderr[-1200:],
        },
        "core_hmma_ldg_assembler": {
            "returncode": core.returncode,
            "output_exists": core_cubin.is_file(),
            "stdout_tail": core.stdout[-1200:],
            "stderr_tail": core.stderr[-1200:],
            "cuobjdump_excerpt": (core_sass[-2400:] if core_sass is not None else None),
            "instruction_encoding_checks": encoding_checks,
        },
    }


def main() -> int:
    args = _parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not args.cubin.is_file():
        raise RuntimeError(f"Baseline cubin does not exist: {args.cubin}")
    baseline_sass, baseline_instructions = _dump_sass(
        args.cubin, args.kernel, args.verbose
    )
    _write_text(args.work_dir / "baseline.sass", baseline_sass)
    baseline_groups = _hmma_groups(baseline_instructions)
    patch = _find_patch(baseline_instructions, baseline_groups)
    patch_bytes = _patch_native_cubin(
        args.cubin,
        args.candidate_cubin,
        args.kernel,
        patch["moved"],
        patch["displaced"],
        args.moved_ldg_wait_mask,
    )
    candidate_sass, candidate_instructions = _dump_sass(
        args.candidate_cubin, args.kernel, args.verbose
    )
    _write_text(args.work_dir / "candidate.sass", candidate_sass)
    candidate_groups = _hmma_groups(candidate_instructions)
    candidate_by_offset = {item.offset: item for item in candidate_instructions}
    baseline_by_offset = {item.offset: item for item in baseline_instructions}
    moved = patch["moved"]
    displaced = patch["displaced"]
    expected_moved_control = _with_wait_mask(moved.control, args.moved_ldg_wait_mask)
    candidate_moved = candidate_by_offset[moved.offset]
    candidate_displaced = candidate_by_offset[displaced.offset]
    changed_order = (
        candidate_moved.bundle_hex == displaced.bundle_hex
        and candidate_displaced.control == expected_moved_control
        and candidate_displaced.encoding == moved.encoding
        and candidate_moved.assembly == displaced.assembly
        and candidate_displaced.assembly == moved.assembly
    )
    if not changed_order:
        raise RuntimeError("Patched cubin did not preserve the requested bundle swap.")

    payload = {
        "target": "sm_70",
        "kernel": args.kernel,
        "baseline_cubin": str(args.cubin),
        "candidate_cubin": str(args.candidate_cubin),
        "baseline_sass_path": str(args.work_dir / "baseline.sass"),
        "candidate_sass_path": str(args.work_dir / "candidate.sass"),
        "baseline": _sass_summary(baseline_instructions, baseline_groups),
        "candidate": _sass_summary(candidate_instructions, candidate_groups),
        "patch": {
            "method": "swap complete native 16-byte SASS bundles",
            "changed_real_sass_order": changed_order,
            "b_fragment_instances": 1,
            "moved_ldg": {
                "from": f"0x{moved.offset:04x}",
                "to": f"0x{displaced.offset:04x}",
                "assembly": moved.assembly,
                "source_control": _control_summary(moved.control),
                "candidate_control": _control_summary(candidate_displaced.control),
            },
            "displaced_instruction": {
                "from": f"0x{displaced.offset:04x}",
                "to": f"0x{moved.offset:04x}",
                "assembly": displaced.assembly,
                "control": _control_summary(displaced.control),
            },
            "b0_subregister_last_use": {
                "register": patch["first_b_register"],
                "offset": f"0x{patch['first_b_last_use'].offset:04x}",
                "assembly": patch["first_b_last_use"].assembly,
            },
            "candidate_window": {
                "between_steps": patch["window_between_steps"],
                "group_index": patch["window_group"],
                "offset": f"0x{displaced.offset:04x}",
                "next_second_k16_hmma": (
                    f"0x{patch['second_wmma_first_hmma'].offset:04x}"
                ),
            },
            "byte_patch": patch_bytes,
            "baseline_bundle_at_window": (
                baseline_by_offset[displaced.offset].bundle_hex
            ),
            "candidate_bundle_at_window": (
                candidate_by_offset[displaced.offset].bundle_hex
            ),
        },
        "turingas": _turingas_probe(
            args.cubin,
            args.kernel,
            moved,
            args.work_dir,
            args.turingas_dir,
            args.verbose,
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        _write_text(args.json_out, text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
