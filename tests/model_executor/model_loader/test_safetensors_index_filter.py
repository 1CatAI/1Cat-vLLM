# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
@pytest.mark.parametrize("load_strategy", [None, "lazy", "eager"])
def test_default_loader_only_yields_tensors_assigned_by_index(
    tmp_path, load_format: str, load_strategy: str | None
) -> None:
    main_shard = tmp_path / "model-00001-of-00002.safetensors"
    reused_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {"main.weight": torch.tensor([1.0])},
        main_shard,
    )
    save_file(
        {
            "ple.weight": torch.tensor([2.0]),
            "main.weight": torch.tensor([99.0]),
            "unindexed.weight": torch.tensor([3.0]),
        },
        reused_shard,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "main.weight": main_shard.name,
                    "ple.weight": reused_shard.name,
                },
            }
        )
    )

    loader = DefaultModelLoader(
        LoadConfig(
            load_format=load_format,
            safetensors_load_strategy=load_strategy,
            use_tqdm_on_load=False,
        )
    )
    source = DefaultModelLoader.Source(
        model_or_path=str(tmp_path),
        revision=None,
        fall_back_to_pt=False,
    )

    weights = list(loader._get_weights_iterator(source))

    assert [name for name, _ in weights] == ["main.weight", "ple.weight"]
    torch.testing.assert_close(weights[0][1], torch.tensor([1.0]))
    torch.testing.assert_close(weights[1][1], torch.tensor([2.0]))


@pytest.mark.skip_global_cleanup
def test_multithread_loader_only_yields_tensors_assigned_by_index(tmp_path) -> None:
    main_shard = tmp_path / "model-00001-of-00002.safetensors"
    reused_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"main.weight": torch.tensor([1.0])}, main_shard)
    save_file(
        {
            "ple.weight": torch.tensor([2.0]),
            "main.weight": torch.tensor([99.0]),
            "unindexed.weight": torch.tensor([3.0]),
        },
        reused_shard,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "main.weight": main_shard.name,
                    "ple.weight": reused_shard.name,
                },
            }
        )
    )

    loader = DefaultModelLoader(
        LoadConfig(
            load_format="safetensors",
            use_tqdm_on_load=False,
            model_loader_extra_config={"enable_multithread_load": True},
        )
    )
    source = DefaultModelLoader.Source(
        model_or_path=str(tmp_path),
        revision=None,
        fall_back_to_pt=False,
    )

    weights = sorted(loader._get_weights_iterator(source), key=lambda kv: kv[0])

    assert [name for name, _ in weights] == ["main.weight", "ple.weight"]
    torch.testing.assert_close(weights[0][1], torch.tensor([1.0]))
    torch.testing.assert_close(weights[1][1], torch.tensor([2.0]))
