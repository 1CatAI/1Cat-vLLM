# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Revision-pinned H3 resolution and bounded-memory checkpoint iteration."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

from .config import H3Config


def resolve_model_root(
    config: H3Config, *, require_original_transformer: bool = False
) -> Path:
    path = Path(config.model).expanduser()
    if path.is_dir():
        return path.absolute()
    partition = "FL2VA" if config.partition == "fl2va" else "Ref2VA"
    patterns = [f"{partition}/model_index.json", f"{partition}/transformer/**"]
    patterns += [
        f"FL2VA/{name}/**"
        for name in ("text_encoder", "tokenizer", "processor", "video_vae", "audio_vae")
    ]
    ignored = (
        [f"{partition}/transformer/*.safetensors"]
        if config.transformer_path and not require_original_transformer
        else None
    )
    return Path(
        snapshot_download(
            config.model,
            revision=config.revision,
            allow_patterns=patterns,
            ignore_patterns=ignored,
        )
    )


def iter_checkpoint_weights(path: str | Path, *, include: set[str] | None = None):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint at {path}")
    seen = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():  # noqa: SIM118 (safe_open is not iterable)
                if include is not None and name not in include:
                    continue
                if name in seen:
                    raise ValueError(f"duplicate checkpoint tensor {name}")
                seen.add(name)
                yield name, checkpoint.get_tensor(name)


def write_checkpoint_manifest(root: str | Path, output: str | Path, *, revision: str):
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    Path(output).write_text(
        json.dumps({"revision": revision, "files": records}, indent=2)
    )
