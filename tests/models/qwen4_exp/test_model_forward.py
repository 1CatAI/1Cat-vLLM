# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpDecoderLayer


class _AttentionHyperConnection:
    def mix(self, hidden_states: torch.Tensor):
        return hidden_states, hidden_states, torch.zeros_like(hidden_states)


class _MlpHyperConnection:
    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ):
        return hidden_states, block_output, injection


class _OutputOnlyGdn:
    def __init__(self) -> None:
        self.output: torch.Tensor | None = None

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        assert output is not None
        self.output = output
        output.copy_(hidden_states + 1)
        return hidden_states + 100


def test_linear_attention_forwards_preallocated_output_buffer() -> None:
    layer = object.__new__(Qwen4ExpDecoderLayer)
    object.__setattr__(layer, "ple", None)
    object.__setattr__(layer, "layer_type", "linear_attention")
    object.__setattr__(layer, "attn_hyper_connection", _AttentionHyperConnection())
    object.__setattr__(layer, "mlp_hyper_connection", _MlpHyperConnection())
    gdn = _OutputOnlyGdn()
    object.__setattr__(layer, "linear_attn", gdn)
    object.__setattr__(layer, "mlp", lambda hidden_states: hidden_states)
    hidden_states = torch.arange(6, dtype=torch.float32).view(2, 3)

    _, mlp_out, _ = Qwen4ExpDecoderLayer.forward(
        layer,
        hidden_states,
        None,
        None,
        torch.arange(2),
        input_ids=None,
        query_start_loc=None,
        ngram_context=None,
    )

    assert gdn.output is not None
    torch.testing.assert_close(mlp_out, hidden_states + 1)
