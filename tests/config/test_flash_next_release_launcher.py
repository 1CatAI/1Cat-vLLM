# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_flash_launcher_uses_adjacent_cli_and_preserves_overrides(tmp_path):
    script = tmp_path / "serve_flash_next_nvfp4_v100.sh"
    shutil.copy(ROOT / "scripts" / script.name, script)
    script.chmod(0o755)
    cli = tmp_path / "vllm"
    cli.write_text(
        f"#!{sys.executable}\nimport sys,json,os\n"
        "print(json.dumps({'argv':sys.argv[1:],'vllm_env':"
        "{k:v for k,v in os.environ.items() if k.startswith('VLLM_')}}))\n"
    )
    cli.chmod(0o755)
    overrides = ["--max-model-len", "32768", "--no-enable-prefix-caching"]
    result = subprocess.run(
        [str(script), "/model with spaces", *overrides],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["vllm_env"] == {}
    args = data["argv"]
    assert args[:2] == ["serve", "/model with spaces"]
    assert args[-len(overrides) :] == overrides
    assert args[args.index("--tensor-parallel-size") + 1] == "4"
    assert args[args.index("--kv-cache-dtype") + 1] == "float16"
    assert args[args.index("--max-model-len") + 1] == "262144"
    assert args[args.index("--max-num-seqs") + 1] == "8"
    assert args[args.index("--prefix-cache-retention-interval") + 1] == "8192"
    assert json.loads(args[args.index("--kernel-config") + 1])["qsa_host_kv"] is True
    assert json.loads(args[args.index("--speculative-config") + 1]) == {
        "method": "mtp",
        "num_speculative_tokens": 4,
        "draft_sample_method": "greedy",
    }


def test_flash_launcher_separate_draft(tmp_path):
    script = tmp_path / "serve_flash_next_nvfp4_v100.sh"
    shutil.copy(ROOT / "scripts" / script.name, script)
    script.chmod(0o755)
    cli = tmp_path / "vllm"
    cli.write_text(
        f"#!{sys.executable}\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n"
    )
    cli.chmod(0o755)
    draft = tmp_path / 'mtp "draft" dir'
    result = subprocess.run(
        [str(script), "/model", "--draft", str(draft), "--port", "8001"],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    args = json.loads(result.stdout)
    spec = json.loads(args[args.index("--speculative-config") + 1])
    assert spec["model"] == str(draft)
    assert args[-2:] == ["--port", "8001"]


@pytest.mark.parametrize(
    "args,code", [([], 2), (["--unknown"], 2), (["--help"], 0), (["/m", "--draft"], 2)]
)
def test_flash_launcher_usage(tmp_path, args, code):
    script = tmp_path / "serve_flash_next_nvfp4_v100.sh"
    shutil.copy(ROOT / "scripts" / script.name, script)
    script.chmod(0o755)
    assert subprocess.run([str(script), *args], capture_output=True).returncode == code
