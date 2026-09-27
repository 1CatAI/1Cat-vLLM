# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess

from tools.pre_commit.check_env_registration import (
    BASELINE,
    registered_variables,
    scan_file,
)


def _scan(tmp_path, source: str) -> int:
    path = tmp_path / "module.py"
    path.write_text(source)
    return scan_file(str(path), registered_variables())


def test_unregistered_read_is_rejected(tmp_path, capsys):
    assert _scan(tmp_path, 'import os\nx = os.getenv("VLLM_NOT_A_SWITCH", "0")\n')
    assert "module.py:2" in capsys.readouterr().out


def test_every_read_form_is_caught(tmp_path):
    for read in (
        'os.environ.get("VLLM_NOT_A_SWITCH")',
        'os.environ["VLLM_NOT_A_SWITCH"]',
        'os.environ.setdefault("VLLM_NOT_A_SWITCH", "1")',
    ):
        assert _scan(tmp_path, f"import os\nx = {read}\n"), read


def test_registered_read_and_write_pass(tmp_path):
    source = (
        "import os\n"
        'port = os.environ.get("VLLM_PORT")\n'
        'os.environ["VLLM_NOT_A_SWITCH"] = "1"\n'
    )
    assert not _scan(tmp_path, source)


def test_baseline_names_are_still_unregistered():
    # A baseline entry that got registered in vllm/envs.py should be removed
    # from the baseline, which only ever shrinks.
    assert not BASELINE & registered_variables()


def test_current_tree_is_clean():
    files = subprocess.run(
        ["git", "ls-files", "vllm/*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    known = registered_variables()
    assert not any(scan_file(path, known) for path in files if path != "vllm/envs.py")
