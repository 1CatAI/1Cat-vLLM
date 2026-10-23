# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline contracts; run with --confcutdir=tests/tools (no inference fixtures)."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from tools import generate_architecture_reference as reference


@pytest.fixture
def source_tree(tmp_path):
    for name in (reference.CODECS, reference.ROUTES, reference.BINDINGS):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference.ROOT / name, target)
    config = tmp_path / "vllm/config"
    config.mkdir(parents=True)
    (config / "vllm.py").write_text(
        "from vllm.config.kernel import KernelConfig\n"
        "class VllmConfig:\n    kernel_config: KernelConfig\n"
    )
    (config / "kernel.py").write_text(
        "class KernelConfig:\n    def compute_hash(self):\n        return 'value'\n"
    )
    for name in reference.MAINTAINED_DOCS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Document\n")
    return tmp_path


def test_reference_is_deterministic_and_links_actual_owners(source_tree):
    first = reference.collect(source_tree)
    assert first == reference.collect(source_tree)
    assert reference.render(first) == reference.render(reference.collect(source_tree))
    kernel = next(
        r for r in first["configurations"] if r["name"].endswith("KernelConfig")
    )
    assert kernel["owners"] == ["kernel_config"]
    assert kernel["methods"]["compute_hash"] == "vllm/config/kernel.py:2"
    assert {row["symbol"] for row in first["codecs"]} == {
        "FP16",
        "BF16",
        "FP8_E4M3",
        "FP8_E5M2",
    }
    assert first["routes"] == sorted(first["routes"], key=lambda r: r["name"])
    (source_tree / reference.OUTPUT).write_text(reference.render(first))
    assert reference.check_links(source_tree) == []


@pytest.mark.parametrize(
    "name,old,new",
    [
        (reference.CODECS, '("fp8", "fp8_e4m3")', '("fp8", "fp8_e4m3", "new_alias")'),
        (reference.ROUTES, '"gqa_ratios": (4, 6, 8)', '"gqa_ratios": (4, 6)'),
        (reference.BINDINGS, '"compact active"', '"changed layout"'),
    ],
)
def test_declaration_change_invalidates_rendered_reference(source_tree, name, old, new):
    before = reference.render(reference.collect(source_tree))
    path = source_tree / name
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, new, 1))
    assert reference.render(reference.collect(source_tree)) != before


@pytest.mark.parametrize(
    "name,old,new",
    [
        (reference.CODECS, "quantized=True", "unknown_field=True"),
        (reference.ROUTES, "frozenset((FP16,))", "unknown_getter()"),
        (
            reference.BINDINGS,
            "STAGE_BINDINGS = {",
            "STAGE_BINDINGS = unknown_getter()\nIGNORED = {",
        ),
        (
            reference.BINDINGS,
            "STAGE_BINDINGS = {",
            "STAGE_BINDINGS = []\nIGNORED = {",
        ),
    ],
)
def test_unknown_declarations_are_errors(source_tree, name, old, new):
    path = source_tree / name
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError):
        reference.collect(source_tree)


def test_missing_source_is_not_silently_omitted(source_tree):
    (source_tree / reference.BINDINGS).unlink()
    with pytest.raises(FileNotFoundError):
        reference.collect(source_tree)


def test_report_does_not_import_or_execute_inference_code(source_tree):
    # Even valid, executable module-level code must never run in this reader.
    path = source_tree / reference.CODECS
    path.write_text(path.read_text() + "\nraise AssertionError('executed source')\n")
    script = """
import importlib.abc
import os
import sys
from pathlib import Path
class NoInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('vllm', 'torch', 'triton'):
            raise AssertionError('inference import: ' + fullname)
sys.meta_path.insert(0, NoInference())
from tools.generate_architecture_reference import collect, render
before = dict(os.environ)
os.getenv = lambda *args: (_ for _ in ()).throw(AssertionError('getenv'))
render(collect(Path(sys.argv[1])))
assert dict(os.environ) == before
"""
    env = dict(os.environ, VLLM_SM70_EXAMPLE="must-not-be-evaluated")
    subprocess.run(
        [sys.executable, "-c", script, str(source_tree)],
        cwd=reference.ROOT,
        env=env,
        check=True,
        capture_output=True,
    )


def test_cli_default_is_read_only_and_json_matches_source(source_tree):
    command = [
        sys.executable,
        "-m",
        "tools.generate_architecture_reference",
        "--root",
        str(source_tree),
    ]
    output = source_tree / reference.OUTPUT
    output.write_text(reference.render(reference.collect(source_tree)))
    before = {p: p.read_bytes() for p in source_tree.rglob("*") if p.is_file()}
    assert (
        subprocess.run(command, cwd=reference.ROOT, capture_output=True).returncode == 0
    )
    assert before == {p: p.read_bytes() for p in before}
    output.write_text("stale\n")
    failed = subprocess.run(command, cwd=reference.ROOT, capture_output=True, text=True)
    assert failed.returncode == 1 and "is stale" in failed.stdout
    assert output.read_text() == "stale\n"
    assert (
        subprocess.run(
            command + ["--write"], cwd=reference.ROOT, capture_output=True
        ).returncode
        == 0
    )
    data = subprocess.check_output(command + ["--json"], cwd=reference.ROOT, text=True)
    assert json.loads(data) == reference.collect(source_tree)


def test_links_headings_fences_and_source_line_bounds(tmp_path):
    (tmp_path / "code.py").write_text("first\nsecond\n")
    (tmp_path / "target.md").write_text("# Repeated\n\n# Repeated\n")
    page = tmp_path / "page.md"
    page.write_text(
        "# Page\n[valid](target.md#repeated-1)\n[code](code.py#L1-L2)\n"
        "```text\n[example](missing.md)\n```\n"
        "[web](https://example.com/no-local-check)\n"
    )
    assert reference.check_links(tmp_path, ["page.md"]) == []
    page.write_text(page.read_text() + "[bad](target.md#absent)\n[bad](code.py#L3)\n")
    errors = reference.check_links(tmp_path, ["page.md", "missing.md"])
    assert len(errors) == 3
    assert any("missing anchor" in e for e in errors)
    assert any("invalid source lines" in e for e in errors)
    assert any("document is missing" in e for e in errors)


def test_existing_reference_matches_source_and_maintained_links():
    assert (reference.ROOT / reference.OUTPUT).read_text() == reference.render(
        reference.collect()
    )
    assert reference.check_links(reference.ROOT) == []
