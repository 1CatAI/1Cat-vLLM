# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the drafter's MoE only for the rows that eager draft prefill samples.

An eager draft prefill runs the draft layer over every scheduled token because
the draft attention K/V (and any QSA indexer keys) must cover all positions.
Only the rows at ``last_token_indices`` reach draft sampling and the next draft
step. The MoE runs after the attention has written its cache, so computing it
for those rows alone leaves every cached K/V byte unchanged; the other rows of
the MoE output are zero and are never read.

The selector wraps each draft ``FusedMoE`` runner's ``_forward_impl``, which the
opaque ``moe_forward``/``moe_forward_shared`` custom ops call. No traced source
changes, so compiled graphs and their caches are unaffected.

With ONECAT_DRAFT47 ``d2a`` the selector also wraps each draft router's
``select_experts``: while :meth:`DraftMoERowSelector.pad_rows` is active (the
drafter decode graph capture), rows >= ``num_valid[0]`` get expert id -1, so
``fused_moe_kernel`` writes zeros for them and reads no expert weights. The
real rows keep the same kernels and the same M.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PadRowOps:
    """Optional padding operators supplied by the drafter's policy owner."""

    mask: Callable[[torch.Tensor, torch.Tensor], None]
    note: Callable[[str], None]


def _aliases(x: torch.Tensor, ref: torch.Tensor) -> bool:
    return x is ref or (
        x.data_ptr() == ref.data_ptr()
        and x.dtype == ref.dtype
        and x.shape == ref.shape
        and x.stride() == ref.stride()
    )


def _scatter_rows(
    out: torch.Tensor, rows: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    full = out.new_zeros((num_tokens, *out.shape[1:]))
    full.index_copy_(0, rows, out)
    return full


class DraftMoERowSelector:
    """Restrict the wrapped MoE layers to ``rows`` while :meth:`select` is active."""

    def __init__(
        self, layers: Iterable[nn.Module], pad_ops: PadRowOps | None = None
    ) -> None:
        self.layers = tuple(layers)
        self.pad_ops = pad_ops
        self._rows: torch.Tensor | None = None
        self._num_valid: torch.Tensor | None = None
        pad_rows = pad_ops is not None
        self.pad_rows_ready = pad_rows
        for layer in self.layers:
            runner = layer.runner
            runner._forward_impl = self._wrap(runner._forward_impl)
            if not pad_rows:
                continue
            router = getattr(runner, "router", None)
            quant_method = getattr(runner, "_quant_method", None)
            if (
                router is None
                or not hasattr(router, "select_experts")
                or getattr(quant_method, "is_monolithic", True)
            ):
                self.pad_rows_ready = False
                continue
            router.select_experts = self._wrap_router(router.select_experts)
        if pad_rows and not self.pad_rows_ready:
            assert pad_ops is not None
            pad_ops.note("fallback:router")

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        exclude: Iterable[nn.Module] = (),
        pad_ops: PadRowOps | None = None,
    ) -> "DraftMoERowSelector | None":
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        excluded = {id(module) for module in exclude}
        layers = [
            module
            for module in model.modules()
            if isinstance(module, FusedMoE) and id(module) not in excluded
        ]
        return cls(layers, pad_ops=pad_ops) if layers else None

    @contextmanager
    def select(self, rows: torch.Tensor | None) -> Iterator[None]:
        previous = self._rows
        self._rows = rows
        try:
            yield
        finally:
            self._rows = previous

    @contextmanager
    def pad_rows(self, num_valid: torch.Tensor | None) -> Iterator[None]:
        """Route rows >= ``num_valid[0]`` (a device scalar read at run and
        replay time) to expert -1 while active."""
        previous = self._num_valid
        self._num_valid = num_valid
        try:
            yield
        finally:
            self._num_valid = previous

    def _wrap_router(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def select_experts(
            *args: Any, **kwargs: Any
        ) -> tuple[torch.Tensor, torch.Tensor]:
            topk_weights, topk_ids = original(*args, **kwargs)
            num_valid = self._num_valid
            if num_valid is not None:
                assert self.pad_ops is not None
                self.pad_ops.mask(topk_ids, num_valid)
            return topk_weights, topk_ids

        return select_experts

    def _wrap(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def forward_impl(
            layer: nn.Module,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
            input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            rows = self._rows
            if rows is None:
                return original(
                    layer,
                    hidden_states,
                    router_logits,
                    shared_experts_input,
                    input_ids,
                )
            return self._forward_rows(
                original,
                rows,
                layer,
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
            )

        return forward_impl

    @staticmethod
    def _forward_rows(
        original: Callable[..., Any],
        rows: torch.Tensor,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        selected_hidden = hidden_states.index_select(0, rows)

        def take(x: torch.Tensor | None) -> torch.Tensor | None:
            # A runner that owns its gate receives hidden_states in place of
            # router logits; keep such aliases pointing at the selected rows.
            if x is None:
                return None
            if _aliases(x, hidden_states):
                return selected_hidden
            if x.dim() > 0 and x.shape[0] == num_tokens:
                return x.index_select(0, rows)
            return x

        result = original(
            layer,
            selected_hidden,
            take(router_logits),
            take(shared_experts_input),
            take(input_ids),
        )
        if isinstance(result, tuple):
            return tuple(_scatter_rows(out, rows, num_tokens) for out in result)
        return _scatter_rows(result, rows, num_tokens)
