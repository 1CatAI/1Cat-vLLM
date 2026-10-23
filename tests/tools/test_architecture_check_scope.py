# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep local/CI filters covering the offline reader without widening scope."""

from fnmatch import fnmatchcase

import regex as re
import yaml

from tools import generate_architecture_reference as reference

HOOKS = {"check-architecture-reference", "test-architecture-reference"}


def test_local_and_ci_filters_cover_inputs_and_skip_unrelated_files():
    root = reference.ROOT
    hooks = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
    selected = [
        hook for repo in hooks["repos"] for hook in repo["hooks"] if hook["id"] in HOOKS
    ]
    assert {hook["id"] for hook in selected} == HOOKS
    workflow = yaml.load(
        (root / ".github/workflows/architecture-docs.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    events = workflow["on"]
    assert events["push"]["paths"] == events["pull_request"]["paths"]
    paths = events["pull_request"]["paths"]
    related = [
        *reference.MAINTAINED_DOCS,
        reference.CODECS,
        reference.ROUTES,
        reference.BINDINGS,
        "vllm/config/kernel.py",
        "vllm/config/quantization/base.py",
        "tools/generate_architecture_reference.py",
        "tools/config_ownership.py",
        "tools/pre_commit/environment_readers.py",
        "tools/sm70/path_inventory.py",
        "tests/tools/test_architecture_reference.py",
        "tests/tools/test_architecture_doc_urls.py",
        "tests/tools/test_architecture_check_scope.py",
        "docs/mkdocs/hooks/url_schemes.py",
        "docs/mkdocs/hooks/generate_argparse.py",
        ".github/workflows/architecture-docs.yml",
        ".github/workflows/pre-commit.yml",
        ".pre-commit-config.yaml",
        "mkdocs.yaml",
    ]
    unrelated = [
        "vllm/v1/core/sched/scheduler.py",
        "csrc/attention/paged_attention_v1.cu",
        "docs/design/architecture/sm70_phase_d.md",
        "docs/design/sm70_v100_migration_control.md",
    ]
    for name, expected in [(p, True) for p in related] + [
        (p, False) for p in unrelated
    ]:
        assert any(fnmatchcase(name, pattern) for pattern in paths) == expected, name
        for hook in selected:
            assert not hook.get("always_run", False)
            assert bool(re.search(hook["files"], name)) == expected, name


def test_all_files_lint_delegates_only_the_two_new_hooks():
    workflow = yaml.safe_load(
        (reference.ROOT / ".github/workflows/pre-commit.yml").read_text()
    )
    steps = workflow["jobs"]["pre-commit"]["steps"]
    action = next(
        step for step in steps if "pre-commit/action@" in step.get("uses", "")
    )
    assert set(action["env"]["SKIP"].split(",")) == HOOKS
    assert action["with"]["extra_args"] == "--all-files --hook-stage manual"
