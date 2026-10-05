# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-layer research HC with peer-pushed down and TP-sharded up/mix."""

from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from vllm.utils.torch_utils import direct_register_custom_op

_extension = None
_owners = {}


def build():
    global _extension
    if _extension is None:
        _extension = load(
            "sm70_hc_sharded_chain_screen",
            [str(Path(__file__).parents[1] / "csrc/sm70_hc_sharded_chain_screen.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
            verbose=True,
        )
    return _extension


def _chain(
    residual: torch.Tensor,
    core: torch.Tensor,
    injection: torch.Tensor,
    norm: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    down_epochs: torch.Tensor,
    norm_flags: torch.Tensor,
    partial: torch.Tensor,
    up_epochs: torch.Tensor,
    key: str,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = residual.shape[0]
    assert m in (1, 5) and residual.shape[1] == 10240
    owner = _owners[key]
    combined = torch.empty_like(residual)
    xn = torch.empty_like(residual)
    block = core.new_empty((m, 2560))
    next_injection = injection.new_empty((m, 4))
    # The first phase does not read or write these former gather destinations.
    unused = core.new_empty((0,))
    ext = build()
    ext.down(
        residual,
        core,
        injection,
        norm,
        down,
        combined,
        xn,
        unused,
        unused,
        next_injection,
        owner["peers"],
        down_epochs,
        norm_flags,
        partial,
        owner["rank"],
        epsilon,
        False,
    )
    ext.up(up, xn, block, next_injection, owner["peers"], up_epochs, owner["rank"])
    return combined, block, next_injection


def _fake(residual, core, injection, *args):
    return (
        torch.empty_like(residual),
        core.new_empty((residual.shape[0], 2560)),
        injection.new_empty((residual.shape[0], 4)),
    )


direct_register_custom_op(
    op_name="sm70_hc_sharded_chain_research",
    op_func=_chain,
    fake_impl=_fake,
    mutates_args=["down_epochs", "norm_flags", "partial", "up_epochs"],
)


def attach(layer, width):
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    assert group.world_size == 4 and width in (1, 5)
    comm = group.device_communicator.ca_comm
    assert not comm.disabled and comm.fully_connected
    assert layer.use_combine and layer.hc_count == 4
    assert layer.input_mix_weight_up.weight.shape == (10240, 320)
    rank = group.rank_in_group
    key = layer.input_mix_weight_up.prefix
    ext = build()
    pointer, handle = ext.allocate(width)
    handles = [None] * 4
    dist.all_gather_object(handles, handle, group=group.cpu_group)
    peers = [
        pointer if i == rank else ext.open_peer(value)
        for i, value in enumerate(handles)
    ]
    _owners[key] = {"rank": rank, "peers": peers, "width": width}
    down = layer.input_mix_weight_down_block_inject.weight
    local = down.new_zeros((88, 10240))
    local[:80].copy_(down[rank * 80 : (rank + 1) * 80])
    local[80].copy_(down[320 + rank])
    packed = local.reshape(11, 8, 640, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
    for name, value in (
        ("down", packed),
        ("down_epochs", torch.zeros(width, 27, dtype=torch.int32, device=down.device)),
        (
            "norm_flags",
            torch.zeros(width * 4 + 22, dtype=torch.int32, device=down.device),
        ),
        ("partial", torch.empty(22, width, 8, dtype=torch.float32, device=down.device)),
        ("up_epochs", torch.zeros(160, dtype=torch.int32, device=down.device)),
    ):
        layer.register_buffer(f"_hc_sharded_{name}", value, persistent=False)
    layer._hc_sharded_original = layer.combine_and_mix
    layer._hc_sharded_enabled = False

    def combine(self, hidden_states, prev_block_output, prev_injection):
        if not self._hc_sharded_enabled:
            return self._hc_sharded_original(
                hidden_states, prev_block_output, prev_injection
            )
        return torch.ops.vllm.sm70_hc_sharded_chain_research(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self._hc_sharded_down,
            self.input_mix_weight_up.weight,
            self._hc_sharded_down_epochs,
            self._hc_sharded_norm_flags,
            self._hc_sharded_partial,
            self._hc_sharded_up_epochs,
            key,
            self.config.rms_norm_eps,
        )

    layer.combine_and_mix = MethodType(combine, layer)
