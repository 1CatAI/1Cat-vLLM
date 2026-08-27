# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _Buffer:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.np = np.zeros(shape, dtype=np.int32)
        self.gpu = torch.zeros(shape, dtype=torch.int32)

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            self.gpu.copy_(torch.from_numpy(self.np))
        else:
            self.gpu[:n].copy_(torch.from_numpy(self.np[:n]))
        return self.gpu


def _runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.uses_ngram_embedding = True
    runner.ngram_context_len = 2
    runner.ngram_eos_token_id = 99
    runner.enable_prompt_embeds = False
    runner._sm70_async_staged_input_prep_active = False
    runner.ngram_context = _Buffer((4, 2))
    runner.query_start_loc = _Buffer((5,))
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([3, 1, 0, 0], dtype=np.int32),
        token_ids_cpu=np.array(
            [
                [1, 2, 3, 4],
                [20, 21, 22, 23],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        is_token_ids=np.ones((4, 4), dtype=bool),
    )
    return runner


def test_v1_runner_prepares_committed_ngram_context() -> None:
    context = _runner()._prepare_ngram_context(num_reqs=2, num_reqs_padded=3)

    torch.testing.assert_close(
        context,
        torch.tensor([[2, 3], [99, 20], [99, 99]], dtype=torch.int32),
    )


def test_v1_runner_uses_stable_dummy_ngram_buffers() -> None:
    runner = _runner()
    model_kwargs: dict[str, Any] = {}

    runner._maybe_add_ngram_kwargs(
        model_kwargs,
        num_reqs=2,
        num_reqs_padded=3,
        is_first_rank=True,
        is_encoder_decoder=False,
        use_dummy_context=True,
        num_scheduled_tokens=[2, 1],
    )

    torch.testing.assert_close(
        model_kwargs["query_start_loc"],
        torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model_kwargs["ngram_context"],
        torch.full((3, 2), 99, dtype=torch.int32),
    )
