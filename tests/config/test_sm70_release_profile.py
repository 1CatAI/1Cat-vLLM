# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from vllm.sm70_profiles.profile import load_profile, profile_argv

ROOT = Path(__file__).resolve().parents[2]


def _python_launcher(path):
    # Moving a symlink to a venv interpreter loses its pyvenv.cfg discovery.
    path.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    path.chmod(0o755)


def test_profile_schema():
    profile = load_profile()
    jsonschema.validate(
        profile,
        {
            "type": "object",
            "required": [
                "name",
                "version",
                "args",
                "draft",
                "hardware",
                "expected_acceleration",
            ],
            "properties": {
                "name": {"const": "qwen38_27b_nvfp4_dflash2"},
                "version": {"const": "1.5.1"},
                "args": {
                    "type": "object",
                    "required": [
                        "kv_cache_dtype",
                        "max_num_batched_tokens",
                        "block_size",
                        "mamba_block_size",
                        "enable_prefix_caching",
                        "speculative_config",
                        "gpu_memory_utilization",
                    ],
                },
                "expected_acceleration": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
        },
    )
    assert profile["args"]["kv_cache_dtype"] == "fp8_e4m3"
    assert profile["args"]["max_num_batched_tokens"] == 8192
    assert profile["args"]["block_size"] == 2048
    assert profile["args"]["mamba_block_size"] == 8192
    spec = profile["args"]["speculative_config"]
    assert spec["revision"] == profile["draft"]["revision"]
    assert len(spec["revision"]) == 40
    with pytest.raises(ValueError, match="Unknown SM70 profile"):
        load_profile("unknown")


def test_launcher_golden_and_overrides(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "serve_qwen38_27b_nvfp4_v100.sh"
    shutil.copy(ROOT / "scripts" / script.name, script)
    _python_launcher(bindir / "python")
    cli = bindir / "vllm"
    cli.write_text(
        f"#!{sys.executable}\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n"
    )
    cli.chmod(0o755)
    overrides = ["--kv-cache-dtype", "fp8_e4m3", "--port", "8123"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [str(script), "/checkpoint with spaces", *overrides],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == [
        "serve",
        "/checkpoint with spaces",
        *profile_argv(),
        *overrides,
    ]
    subprocess.run([str(script), "--help"], capture_output=True, check=True)


def test_launcher_propagates_profile_failure(tmp_path):
    script = tmp_path / "serve_qwen38_27b_nvfp4_v100.sh"
    shutil.copy(ROOT / "scripts" / script.name, script)
    cli = tmp_path / "vllm"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 17\n")
    python.chmod(0o755)
    result = subprocess.run([str(script), "MODEL"], capture_output=True)
    assert result.returncode == 17


def test_launcher_local_draft_and_override(tmp_path):
    draft = tmp_path / "draft with spaces"
    draft.mkdir()
    script = tmp_path / "serve_qwen38_27b_nvfp4_v100.sh"
    shutil.copy(ROOT / "scripts" / script.name, script)
    _python_launcher(tmp_path / "python")
    cli = tmp_path / "vllm"
    cli.write_text(
        f"#!{sys.executable}\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n"
    )
    cli.chmod(0o755)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [str(script), "target", "--port", "8000", "--draft", str(draft)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    argv = json.loads(result.stdout)
    assert argv == [
        "serve",
        "target",
        *profile_argv(draft=str(draft)),
        "--port",
        "8000",
    ]
    spec = json.loads(argv[argv.index("--speculative-config") + 1])
    assert spec["model"] == str(draft)
    assert "revision" not in spec
    assert spec["num_speculative_tokens"] == 7


@pytest.mark.parametrize("draft_args", [["--draft"], ["--draft", "/missing/draft"]])
def test_launcher_rejects_invalid_local_draft(tmp_path, draft_args):
    script = tmp_path / "serve_qwen38_27b_nvfp4_v100.sh"
    shutil.copy(ROOT / "scripts" / script.name, script)
    _python_launcher(tmp_path / "python")
    cli = tmp_path / "vllm"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    result = subprocess.run(
        [str(script), "target", *draft_args],
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
    )
    assert result.returncode == 2
