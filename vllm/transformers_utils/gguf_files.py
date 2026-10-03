# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve and validate GGUF checkpoint files before model allocation."""

from pathlib import Path

import gguf
import regex as re
from huggingface_hub import hf_hub_download


def gguf_shard_paths(path: str | Path) -> list[Path]:
    path = Path(path)
    match = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.gguf", path.name)
    if match is None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    prefix, index, total = match.groups()
    if not 1 <= int(index) <= int(total):
        raise ValueError(f"Invalid GGUF shard index: {path.name}")
    paths = [
        path.with_name(f"{prefix}-{i:0{len(index)}d}-of-{total}.gguf")
        for i in range(1, int(total) + 1)
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing GGUF shards: {', '.join(missing)}")
    return paths


def resolve_gguf_file(
    model: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
) -> Path:
    """Resolve a local file, Hub filename, or Hub quantization selector."""
    path = Path(model)
    if path.is_file():
        return path
    model = str(model)
    if model.endswith(".gguf") and not path.is_absolute():
        parts = model.split("/")
        if len(parts) < 3:
            raise FileNotFoundError(f"Not a Hub GGUF filename reference: {model}")
        repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    elif ":" in model:
        from .gguf_utils import get_gguf_file_path_from_hf

        repo_id, quant = model.rsplit(":", 1)
        filename = get_gguf_file_path_from_hf(repo_id, quant, revision=revision)
    else:
        raise FileNotFoundError(f"Not a GGUF checkpoint reference: {model}")
    match = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.gguf", filename)
    filenames = [filename]
    if match is not None:
        prefix, index, total = match.groups()
        if not 1 <= int(index) <= int(total):
            raise ValueError(f"Invalid GGUF shard index: {filename}")
        filenames = [
            f"{prefix}-{i:0{len(index)}d}-of-{total}.gguf"
            for i in range(1, int(total) + 1)
        ]
    downloaded = {
        name: Path(
            hf_hub_download(
                repo_id, name, revision=revision, cache_dir=cache_dir, token=token
            )
        )
        for name in filenames
    }
    return downloaded[filename]


def gguf_tensor_index(paths: list[Path]) -> dict[str, gguf.ReaderTensor]:
    """Reject duplicate tensors and inconsistent split metadata.

    Reader tensors keep the mmap-backed data alive; their payload is not copied.
    """
    tensors = {}
    architecture = None
    declared_total = None
    for index, path in enumerate(paths):
        reader = gguf.GGUFReader(path)
        field = reader.get_field("general.architecture")
        arch = field.contents() if field is not None else None
        if index == 0:
            architecture = arch
        elif arch is not None and arch != architecture:
            raise ValueError(f"Conflicting GGUF architecture in shard {path.name}")
        for key, expected in (("split.count", len(paths)), ("split.no", index)):
            if (
                field := reader.get_field(key)
            ) is not None and field.contents() != expected:
                raise ValueError(f"Invalid GGUF {key} in {path.name}")
        if (field := reader.get_field("split.tensors.count")) is not None:
            count = field.contents()
            if declared_total is not None and declared_total != count:
                raise ValueError("Conflicting GGUF split.tensors.count")
            declared_total = count
        for tensor in reader.tensors:
            if tensor.name in tensors:
                raise ValueError(f"Duplicate GGUF tensor {tensor.name!r}")
            tensors[tensor.name] = tensor
    if declared_total is not None and declared_total != len(tensors):
        raise ValueError(
            f"GGUF declares {declared_total} tensors, found {len(tensors)}"
        )
    return tensors
