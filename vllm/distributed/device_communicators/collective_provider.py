# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native collective binding and tensor contracts, independent of IPC ownership."""

from dataclasses import dataclass

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)


class NativeCollectiveBindings:
    """Bind one DSO before creating its opaque communicator.

    Registration, execution and destruction never fall through to a different
    extension. The ordinary packaged extension implements the configured ABI.
    """

    def __init__(self, policy):
        self.namespace = ops._custom_ar_owner_namespace()
        self.policy = policy

    @property
    def available(self):
        return self.has("init_custom_ar_configured")

    def has(self, *names):
        return all(hasattr(self.namespace, name) for name in names)

    def __getattr__(self, name):
        return getattr(self.namespace, name)

    def init_custom_ar(self, pointers, rank_data, rank, fully_connected):
        return self.namespace.init_custom_ar_configured(
            pointers, rank_data, rank, fully_connected, list(self.policy.values)
        )

    def call_optional_tail(self, op_name, argument_name, args):
        op = getattr(self.namespace, op_name)
        supports_split = any(
            argument_name in str(schema) for schema in op._schemas.values()
        )
        op(*(args if supports_split else args[:-1]))


@dataclass(frozen=True)
class CollectiveCapabilities:
    world_size: int
    fully_connected: bool
    hierarchical: bool
    dispatch_max_size: int
    allocation_size: int
    push_registered: bool
    long_norm_registered: bool

    def reduce(self, inp):
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False
        size = inp.numel() * inp.element_size()
        if size % 16 or not weak_contiguous(inp):
            return False
        if self.hierarchical:
            return inp.dtype == torch.float16 and inp.numel() in (4096, 8 * 4096)
        return (
            self.world_size == 2 or self.fully_connected
        ) and size < self.dispatch_max_size

    def model_collective(self, contract, tensor, native):
        return bool(
            self.world_size == contract.world_size
            and self.fully_connected
            and self.push_registered
            and contract.accepts_layout(tensor)
            and native.has(*contract.operators)
        )

    def gemma_norm(self, inp, residual, weight, *, world_size, long=False):
        if self.world_size != world_size or (
            world_size == 4 and not self.fully_connected
        ):
            return False
        if long:
            admitted = (
                self.long_norm_registered
                and inp.ndim == 2
                and inp.shape[0] % world_size == 0
                and 1 <= inp.shape[0] // world_size <= 2048
                and inp.numel() * inp.element_size() <= self.allocation_size
            )
        else:
            admitted = self.reduce(inp) and inp.ndim == 2 and 1 <= inp.shape[0] <= 64
        return bool(
            admitted
            and inp.is_cuda
            and inp.dtype == torch.float16
            and inp.shape[1] == 5120
            and residual.shape == inp.shape
            and residual.dtype
            in ((torch.float16, torch.float32) if world_size == 2 else (torch.float32,))
            and weight.ndim == 1
            and weight.numel() == 5120
            and weight.dtype in (torch.float16, torch.float32)
            and inp.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
            and inp.device == residual.device == weight.device
        )


def weak_contiguous(inp):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def gemma_fusion_modes(
    *,
    capabilities,
    hidden_size,
    dtype,
    sm70,
    long_requested,
    tp_size,
    pp_size,
    speculative,
):
    common = hidden_size == 5120 and dtype == torch.float16 and sm70
    if not common:
        return False, False, False
    tp = tp_size
    long = bool(long_requested and tp == 4 and pp_size == 1 and not speculative)
    push = (
        tp == 4
        and capabilities is not None
        and capabilities.fully_connected
        and capabilities.push_registered
        and not long
    )
    return tp == 2, long, push


class CollectiveTrace:
    """Diagnostic counters belong to one communicator, never the process."""

    def __init__(self, enabled, name, custom, symm_mem, flashinfer):
        self.enabled = enabled
        self.name = name
        self.providers = (custom, symm_mem, flashinfer)
        self.seen = set()

    def record(self, backend, input_):
        if not self.enabled:
            return
        size = input_.element_size() * input_.numel()
        key = (self.name, backend, tuple(input_.shape), input_.dtype, size)
        if key in self.seen:
            return
        self.seen.add(key)
        logger.warning(
            "TP all-reduce trace backend=%s group=%s shape=%s dtype=%s bytes=%d "
            "custom_enabled=%s torch_symm_mem=%s flashinfer=%s",
            backend,
            self.name,
            tuple(input_.shape),
            input_.dtype,
            size,
            *self.providers,
        )

    def snapshot(self):
        return [
            dict(
                group=group, backend=backend, shape=shape, dtype=str(dtype), bytes=size
            )
            for group, backend, shape, dtype, size in sorted(self.seen, key=str)
        ]
