# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incomplete sharded safetensors checkpoints fail before any tensor is read."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    raise_on_missing_safetensors_shards,
)

pytestmark = pytest.mark.skip_global_cleanup


def _shard(path, name: str) -> None:
    save_file({name: torch.tensor([1.0])}, path)


def _write_shards(tmp_path, present, total=3, stem="model", width=5):
    files = []
    for i in present:
        name = f"{stem}-{i:0{width}d}-of-{total:0{width}d}.safetensors"
        _shard(tmp_path / name, f"w{i}")
        files.append(name)
    return files


def _write_index(tmp_path, weight_map: dict[str, str]) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _prepare(tmp_path, overrides=None, **load_config_kwargs):
    loader = DefaultModelLoader(LoadConfig(**load_config_kwargs))
    return loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=False,
        allow_patterns_overrides=overrides,
    )


@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_complete_indexless_checkpoint_loads(tmp_path, load_format) -> None:
    _write_shards(tmp_path, [1, 2, 3])
    _, files, use_safetensors, _ = _prepare(tmp_path, load_format=load_format)
    assert use_safetensors and len(files) == 3


@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_indexless_checkpoint_missing_a_shard_fails(tmp_path, load_format) -> None:
    _write_shards(tmp_path, [1, 3])
    with pytest.raises(FileNotFoundError, match="model-00002-of-00003"):
        _prepare(tmp_path, load_format=load_format)


def test_indexless_checkpoint_missing_the_last_shard_fails(tmp_path) -> None:
    _write_shards(tmp_path, [1, 2])
    with pytest.raises(FileNotFoundError, match="model-00003-of-00003"):
        _prepare(tmp_path, load_format="safetensors")


def test_complete_indexed_checkpoint_loads(tmp_path) -> None:
    files = _write_shards(tmp_path, [1, 2, 3])
    _write_index(tmp_path, {f"w{i + 1}": f for i, f in enumerate(files)})
    _, selected, _, _ = _prepare(tmp_path, load_format="safetensors")
    assert len(selected) == 3


def test_indexed_checkpoint_missing_a_referenced_shard_fails(tmp_path) -> None:
    files = _write_shards(tmp_path, [1, 3])
    weight_map = {"w1": files[0], "w2": "model-00002-of-00003.safetensors"}
    weight_map["w3"] = files[1]
    _write_index(tmp_path, weight_map)
    with pytest.raises(FileNotFoundError, match="model-00002-of-00003"):
        _prepare(tmp_path, load_format="safetensors")


def test_single_file_checkpoint_loads(tmp_path) -> None:
    _shard(tmp_path / "model.safetensors", "w")
    _, files, _, _ = _prepare(tmp_path, load_format="safetensors")
    assert [f.rsplit("/", 1)[-1] for f in files] == ["model.safetensors"]


def test_non_standard_names_are_not_checked(tmp_path) -> None:
    # No "-of-" numbering: nothing to infer, so nothing is required.
    _shard(tmp_path / "weights-a.safetensors", "a")
    _shard(tmp_path / "weights-b.safetensors", "b")
    _shard(tmp_path / "adapter.safetensors", "c")
    _, files, _, _ = _prepare(tmp_path, load_format="safetensors")
    assert len(files) == 3


def test_mixed_families_are_checked_independently(tmp_path) -> None:
    # A complete main family next to a complete second family with another
    # stem and count, plus a loose file.
    _write_shards(tmp_path, [1, 2], total=2)
    _write_shards(tmp_path, [1, 2, 3, 4], total=4, stem="mtp")
    _shard(tmp_path / "extra.safetensors", "e")
    _, files, _, _ = _prepare(tmp_path, load_format="safetensors")
    assert len(files) == 7


def test_mixed_families_report_the_incomplete_one(tmp_path) -> None:
    _write_shards(tmp_path, [1, 2], total=2)
    _write_shards(tmp_path, [1, 2, 4], total=4, stem="mtp")
    with pytest.raises(FileNotFoundError, match="mtp-00003-of-00004"):
        _prepare(tmp_path, load_format="safetensors")


def test_index_reusing_a_shard_for_many_tensors_loads(tmp_path) -> None:
    files = _write_shards(tmp_path, [1, 2], total=2)
    weight_map = {f"a{i}": files[0] for i in range(5)}
    weight_map.update({f"b{i}": files[1] for i in range(5)})
    _write_index(tmp_path, weight_map)
    raise_on_missing_safetensors_shards(
        str(tmp_path),
        "model.safetensors.index.json",
        [str(tmp_path / f) for f in files],
    )


def test_index_is_authoritative_over_numbering(tmp_path) -> None:
    # The index names only shard 1 of a nominal 2-shard family (e.g. a
    # checkpoint that dropped an unused shard and rewrote its index).
    files = _write_shards(tmp_path, [1], total=2)
    _write_index(tmp_path, {"w1": files[0]})
    _, selected, _, _ = _prepare(tmp_path, load_format="safetensors")
    assert len(selected) == 1


def test_ignore_patterns_allow_absent_shards(tmp_path) -> None:
    _write_shards(tmp_path, [1, 3])
    raise_on_missing_safetensors_shards(
        str(tmp_path),
        "model.safetensors.index.json",
        [str(tmp_path / f) for f in sorted(p.name for p in tmp_path.iterdir())],
        ignore_patterns=["model-00002-*"],
    )
    with pytest.raises(FileNotFoundError):
        raise_on_missing_safetensors_shards(
            str(tmp_path),
            "model.safetensors.index.json",
            [str(tmp_path / f) for f in sorted(p.name for p in tmp_path.iterdir())],
            ignore_patterns=["original/**/*"],
        )


def test_allow_pattern_overrides_skip_the_index_check(tmp_path) -> None:
    files = _write_shards(tmp_path, [1], total=1)
    _write_index(tmp_path, {"w1": files[0], "w2": "gone.safetensors"})
    _, selected, _, _ = _prepare(
        tmp_path, overrides=["*.safetensors"], load_format="safetensors"
    )
    assert len(selected) == 1


def test_check_reads_no_tensor_data(tmp_path) -> None:
    # Truncated shard files: only names are looked at.
    for i in (1, 2):
        (tmp_path / f"model-{i:05d}-of-00002.safetensors").write_bytes(b"x")
    _write_index(
        tmp_path,
        {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        },
    )
    raise_on_missing_safetensors_shards(
        str(tmp_path),
        "model.safetensors.index.json",
        [str(p) for p in tmp_path.glob("*.safetensors")],
    )
