# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
from pathlib import Path


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
