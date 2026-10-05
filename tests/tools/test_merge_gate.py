# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The merge gate rejects missing scopes and preserves check failures."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "tools/merge_gate.sh"


def test_requires_a_test_scope():
    result = subprocess.run([str(GATE)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "TEST_PATH" in result.stderr


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_check_failure_stops_the_gate(tmp_path, failure_call):
    calls = tmp_path / "calls.jsonl"
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"path = pathlib.Path({str(calls)!r})\n"
        "prior = path.read_text().splitlines() if path.exists() else []\n"
        "with path.open('a') as stream:\n"
        "    stream.write(json.dumps({'args': sys.argv[1:], "
        "'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'offline': os.environ.get('HF_HUB_OFFLINE')}) + '\\n')\n"
        f"sys.exit(19 if len(prior) + 1 == {failure_call} else 0)\n"
    )
    python.chmod(0o755)
    # Keep the scope diff independent of the source checkout's merge history.
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    git = ["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null"]
    subprocess.run(git + ["config", "user.name", "Merge gate fixture"], check=True)
    subprocess.run(
        git + ["config", "user.email", "merge-gate@example.invalid"], check=True
    )
    scope = repository / "tests/tools/test_merge_gate.py"
    scope.parent.mkdir(parents=True)
    for content in ["# Initial scope\n", "# Changed scope\n"]:
        scope.write_text(content)
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(
            git + ["-c", "commit.gpgsign=false", "commit", "-q", "-s", "-m", "Scope"],
            check=True,
        )
    result = subprocess.run(
        [
            str(GATE),
            "--base",
            "HEAD~1",
            "--python",
            str(python),
            "tests/tools/test_merge_gate.py",
        ],
        cwd=repository,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "4,5,6,7"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 19
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records) == failure_call
    assert all(r["cuda"] == "" and r["offline"] == "1" for r in records)
    assert records[0]["args"][0] == "tools/run_cpu_tests.py"
    if failure_call == 3:
        assert records[-1]["args"][1] == "pre_commit"


def test_scope_survives_base_ref_advancing_during_tests(tmp_path):
    source = Path(__file__).resolve().parents[2] / "tools/merge_gate.sh"
    (tmp_path / "tools").mkdir()
    gate = tmp_path / "tools/merge_gate.sh"
    gate.write_text(source.read_text())
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/scoped.py").write_text("")
    log = tmp_path / "calls.txt"
    runner = tmp_path / "test-runner.sh"
    runner.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f'printf "%s\\n" "$@" >> "{log}"\n'
        "if [[ $1 == tools/run_cpu_tests.py ]]; then\n"
        "    git update-ref refs/heads/gate-base HEAD\n"
        "fi\n"
    )
    runner.chmod(0o755)

    def git(*args):
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Gate Test",
                "-c",
                "user.email=gate@example.test",
                *args,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "--quiet")
    git("add", "tools/merge_gate.sh", "tests/scoped.py", "test-runner.sh")
    git("commit", "--quiet", "-m", "Base")
    git("branch", "gate-base")
    changed = tmp_path / "changed.py"
    changed.write_text("value = 1\n")
    git("add", "changed.py")
    git("commit", "--quiet", "-m", "Candidate")
    subprocess.run(
        [
            "bash",
            str(gate),
            "--base",
            "refs/heads/gate-base",
            "--python",
            str(runner),
            "tests/scoped.py",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    calls = log.read_text().splitlines()
    assert "pre_commit" in calls
    assert "changed.py" in calls
