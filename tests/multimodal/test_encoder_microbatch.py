# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import Mock

import pytest
import torch

from vllm.config import MultiModalConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner


@pytest.mark.parametrize("limit", [0, 1, 2, 6, 7, 8])
def test_encoder_forward_retains_every_item(limit):
    model = Mock()
    model.embed_multimodal.side_effect = lambda x: [t.reshape(1, 1) for t in x]
    runner = EncoderRunner(
        model, 16, 1, EncoderCache(), torch.float32, torch.device("cpu"), limit
    )
    items = [
        (
            "image",
            MultiModalKwargsItem(
                {"x": MultiModalFieldElem(torch.tensor([i]), MultiModalBatchedField())}
            ),
        )
        for i in range(7)
    ]
    outputs = runner.execute_mm_encoder(items)
    assert [t.item() for t in outputs] == list(range(7))
    sizes = [
        call.kwargs["x"].shape[0] for call in model.embed_multimodal.call_args_list
    ]
    assert sum(sizes) == 7
    assert max(sizes) <= (limit or 7)


def test_encoder_batch_config_hash_and_count_are_independent():
    default = MultiModalConfig()
    assert (
        default.compute_hash()
        == MultiModalConfig(mm_encoder_batch_size=0).compute_hash()
    )
    bounded = MultiModalConfig(
        mm_encoder_batch_size=1, limit_per_prompt={"image": 2147483647}
    )
    assert bounded.get_limit_per_prompt("image") == 2147483647
    assert bounded.compute_hash() != default.compute_hash()
    assert (
        MultiModalConfig(language_model_only=True).compute_hash()
        == MultiModalConfig(
            language_model_only=True, mm_encoder_batch_size=1
        ).compute_hash()
    )
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_batch_size=-1)


def test_encoder_batch_cli_matches_engine_args():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(parser.parse_args(["--mm-encoder-batch-size", "1"]))
    assert (
        args.mm_encoder_batch_size
        == EngineArgs(mm_encoder_batch_size=1).mm_encoder_batch_size
    )
