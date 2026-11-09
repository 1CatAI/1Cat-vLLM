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


def test_comments_strings_and_deletions_are_not_environment_reads(tmp_path):
    source = (
        '# os.getenv("VLLM_NOT_A_SWITCH")\n'
        "example = 'os.environ[\"VLLM_NOT_A_SWITCH\"]'\n"
        'del os.environ["VLLM_NOT_A_SWITCH"]\n'
    )
    assert not _scan(tmp_path, source)


def test_import_aliases_cannot_bypass_registration(tmp_path):
    sources = (
        'import os as system\nx = system.getenv("VLLM_NOT_A_SWITCH")\n',
        'from os import environ as env\nx = env.get("VLLM_NOT_A_SWITCH")\n',
        'from os import getenv as lookup\nx = lookup("VLLM_NOT_A_SWITCH")\n',
    )
    for source in sources:
        assert _scan(tmp_path, source)


def test_keyword_environment_keys_are_checked(tmp_path):
    assert _scan(tmp_path, 'import os\nx = os.getenv(key="VLLM_NOT_A_SWITCH")\n')


def test_native_readers_and_constant_aliases(tmp_path):
    from tools.pre_commit.check_env_registration import native_reads, scan_file

    source = """
// std::getenv("VLLM_COMMENT");
/* env_flag_enabled("VLLM_COMMENT_TOO"); */
const char* key = "VLLM_NATIVE_ALIAS";
std::getenv(key);
env_flag_enabled("VLLM_NATIVE_FLAG");
dispatch_policy_override_from_env("VLLM_NATIVE_POLICY");
std::puts("VLLM_LOG_ONLY");
"""
    assert native_reads(source) == [
        ("VLLM_NATIVE_ALIAS", 5),
        ("VLLM_NATIVE_FLAG", 6),
        ("VLLM_NATIVE_POLICY", 7),
    ]
    path = tmp_path / "kernel.cu"
    path.write_text(source)
    assert scan_file(str(path), {"VLLM_NATIVE_ALIAS"}) == 1
    assert (
        scan_file(
            str(path), {"VLLM_NATIVE_ALIAS", "VLLM_NATIVE_FLAG", "VLLM_NATIVE_POLICY"}
        )
        == 0
    )


def test_native_strings_preserve_comment_markers_and_line_numbers():
    from tools.pre_commit.check_env_registration import native_reads

    source = """
const char* url = "https://example.invalid/*this is not a comment*/";
/* multiple
   comment lines */
auto flag = std::getenv(
    "VLLM_MULTILINE_NATIVE");
"""
    assert native_reads(source) == [("VLLM_MULTILINE_NATIVE", 5)]


def test_native_alias_uses_the_preceding_binding():
    from tools.pre_commit.check_env_registration import native_reads

    source = """
void first() { const char* key = "VLLM_FIRST"; getenv(key); }
void second() { const char* key = "VLLM_SECOND"; getenv(key); }
"""
    assert native_reads(source) == [("VLLM_FIRST", 2), ("VLLM_SECOND", 3)]


def test_stable_module_alias_is_checked(tmp_path):
    assert _scan(
        tmp_path,
        'import os\nKEY = "VLLM_NOT_A_SWITCH"\n'
        "def read():\n    return os.getenv(KEY)\n",
    )


def test_module_alias_does_not_override_function_parameter(tmp_path):
    assert not _scan(
        tmp_path,
        'import os\nKEY = "VLLM_NOT_A_SWITCH"\n'
        "def read(KEY):\n    return os.getenv(KEY)\n",
    )


def test_module_alias_does_not_override_function_local_binding(tmp_path):
    assert not _scan(
        tmp_path,
        'import os\nKEY = "VLLM_NOT_A_SWITCH"\n'
        'def read():\n    KEY = "VLLM_PORT"\n    return os.getenv(KEY)\n',
    )
