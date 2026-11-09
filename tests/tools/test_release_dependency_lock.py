# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]


def test_test_lock_matches_release_torch_and_cuda():
    lock = {
        requirement.name: requirement
        for line in (ROOT / "requirements/test/cuda.txt").read_text().splitlines()
        if line and not line.startswith((" ", "#"))
        for requirement in [Requirement(line)]
    }
    release = {
        requirement.name: requirement
        for line in (ROOT / "requirements/cuda.txt").read_text().splitlines()
        if line.startswith(("torch==", "torchaudio==", "torchvision=="))
        for requirement in [Requirement(line.split("#", 1)[0])]
    }
    for name, requirement in release.items():
        pinned = Version(next(iter(lock[name].specifier)).version)
        assert pinned in requirement.specifier
        assert pinned.local == "cu128"
    assert not any("cu13" in name for name in lock)
    for name in ("nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12"):
        assert Version(next(iter(lock[name].specifier)).version).release[:2] == (
            12,
            8,
        )


def test_regeneration_hook_preserves_release_cuda_backend():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    hook = next(hook for hook in hooks if hook["id"] == "pip-compile")
    args = hook["args"]
    assert args[args.index("--torch-backend") + 1] == "cu128"
    assert args[args.index("-c") + 1] == "requirements/cuda.txt"


def test_dataset_unicode_round_trip(tmp_path):
    from datasets import Dataset, load_from_disk

    texts = ["hello", "你好，世界", "日本語の文章です", "café élève"]
    path = tmp_path / "dataset"
    Dataset.from_dict({"text": texts}).save_to_disk(str(path))
    assert load_from_disk(str(path))["text"] == texts
