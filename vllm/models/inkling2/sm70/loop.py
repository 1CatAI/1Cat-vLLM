# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Looped Inkling2 checkpoints (`loop_config`) on SM70.

Port of upstream `vllm/models/inkling2/nvidia/loop.py`. The core layers
`[core_start, core_end)` run `num_passes` times: a prelude, then each pass
from the AttnRes state after the prelude, then the coda from the last pass's
state. Each pass of a core layer has its own KV and conv-state caches. In
passes after the first, every core sublayer input (attention q/k/v and MLP)
is blended with a message: a normalized-value AttnRes read of the previous
pass's blocks, merged by a gated low-rank controller.

SM70: controller weights are fp16; each sublayer's short conv runs before the
next AttnRes mix (`Inkling2ShortConv.conv_mix`, no fused kernel).
"""

from collections.abc import Sequence

import torch
from torch import nn

from ..configs import Inkling2ModelConfig
from ..quant_config import Inkling2QuantConfig
from .model import Inkling2DecoderLayer, Inkling2ForConditionalGeneration, Inkling2Model
from .ops import blend_fused, loop_fused
from .ops.attn_res import attn_res
from .ops.loop import blend_rms_norm, pair_rms_norm

LOOP_STREAMS = ("attn", "k", "v", "mlp")


def _bmm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        return torch.bmm(a, b, out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return torch.bmm(a.float(), b.float())


class Inkling2MessageController(nn.Module):
    """Weights of one stream's message gate (views of the layer's stacked
    buffers): `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message`."""

    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    """A loop core layer: per-pass caches and a message controller per
    stream (`LOOP_STREAMS`), whose weights are stacked over the streams."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        streams = len(LOOP_STREAMS)
        dtype = torch.get_default_dtype()
        self.register_buffer(
            "controller_down",
            torch.empty(streams, rank, 2 * width, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "controller_gain_forget",
            torch.empty(streams, 2 * width, rank, dtype=dtype),
            persistent=False,
        )
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2MessageController(
                    self.controller_down[i],
                    self.controller_gain_forget[i, :width],
                    self.controller_gain_forget[i, width:],
                )
                for i, stream in enumerate(LOOP_STREAMS)
            }
        )

    def inkling_post_load(self) -> None:
        for i, controller in enumerate(self.controllers.values()):
            assert (
                controller.down_weight.data_ptr() == self.controller_down[i].data_ptr()
            )
            assert (
                controller.gain_weight.data_ptr()
                == self.controller_gain_forget[i].data_ptr()
            )
        # Fused decode-size blend (VLLM_INKLING2_SM70_FUSED_BLEND): ticket counters.
        self._fused_state = (
            loop_fused.make_state(self.controller_down.device)
            if loop_fused.enabled()
            else None
        )
        # One-launch blend (VLLM_INKLING2_SM70_BLEND_ONE): barrier counters + cross-CTA scratch.
        down = self.controller_down
        self._blend_one = blend_fused.make_state(
            down.device, down.shape[2] // 2, down.shape[1]
        )

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The `streams` controllers on fp32 `(S, T, D)` `dest` (may broadcast
        over streams) and `message`, each output RMS-normalized with its row of
        `norm_weight` `(S, D)`: model-dtype `(S, T, D)`."""
        one = getattr(self, "_blend_one", None)
        if one is not None and blend_fused.supports(
            *message.shape, self.controller_down.shape[1]
        ):
            return blend_fused.blend(
                dest,
                message,
                self.controller_down[streams],
                self.controller_gain_forget[streams],
                norm_weight,
                norm_eps,
                one,
            )
        state = getattr(self, "_fused_state", None)
        if state is not None and loop_fused.supports(
            message.shape[1], message.shape[2], self.controller_down.shape[1]
        ):
            return loop_fused.blend(
                dest,
                message,
                self.controller_down[streams],
                self.controller_gain_forget[streams],
                norm_weight,
                norm_eps,
                state,
            )
        pair = pair_rms_norm(dest, message)
        hidden = torch.bmm(pair, self.controller_down[streams].transpose(1, 2))
        gain_forget = _bmm_fp32(
            hidden, self.controller_gain_forget[streams].transpose(1, 2)
        )
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)


class Inkling2LoopModel(Inkling2Model):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self._fused_messages: torch.Tensor | None = None
        self.hidden_size = config.hidden_size
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer(
            "message_queries",
            torch.zeros(
                len(self.loop_core),
                len(LOOP_STREAMS),
                config.hidden_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def inkling_post_load(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)
        return super()._make_layer(config, layer_id, quant_config, prefix)

    def _max_loop_blocks(self) -> int:
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs: torch.Tensor | None,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: nn.Module | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor]:
        layer = self.layers[i]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        messages = None
        attn_inputs: Sequence[torch.Tensor]
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            if self._fused_messages is not None:
                messages = self._fused_messages[i - self.loop_core.start]
            else:
                messages = attn_res(
                    bank[:1],
                    bank[1],
                    None,
                    self.message_queries[i - self.loop_core.start],
                    None,
                    0.0,
                    True,
                )
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            attn_inputs = layer.blend(
                slice(0, 3), dests, messages[:3], norm_weight, norm.eps
            )
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

        first = i % self.block_size == 0
        attn_out = torch.empty_like(partial)
        mlp_in = layer.attn_sconv.conv_mix(
            attn_partial,
            positions,
            attn_out,
            stack[: num_blocks + first],
            None if first else partial,
            self.attn_res.mlp_queries[i],
            None if messages is not None else layer.mlp_norm,
            owner=owner,
            reduce=True,
        )
        if messages is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer)
            norm = layer.mlp_norm
            mlp_in = layer.blend(
                slice(3, 4), mlp_in, messages[3:], norm.weight, norm.eps
            )

        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        outputs = layer.mlp_sconv.conv_mix(
            layer.mlp(mlp_in[0], reduce=False, pass_idx=pass_idx),
            positions,
            stack[num_blocks],
            stack[:num_blocks],
            None,
            next_query,
            next_norm,
            addend=attn_out,
            residual=residual,
            owner=owner,
            reduce=True,
        )
        return num_blocks, outputs

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = inputs_embeds.new_empty(
            (self._max_loop_blocks() + 1, num_tokens, hidden)
        )
        stack[0].copy_(inputs_embeds)
        tau = self._log_scaling_tau(positions)
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        aux_hidden_states = []
        next_norm: nn.Module | None
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                # Messages depend only on the bank and queries: one launch per pass.
                self._fused_messages = (
                    loop_fused.all_messages(bank, self.message_queries)
                    if loop_fused.supports(num_tokens, hidden, 64)
                    and bank.shape[0] == 2
                    else None
                )
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        query,
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].clone())
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
            )
        assert inputs is not None
        out = inputs[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """The `inkling2_loop` architecture: `Inkling2ForConditionalGeneration`
    with an `Inkling2LoopModel`."""
