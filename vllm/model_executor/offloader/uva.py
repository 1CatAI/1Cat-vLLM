# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

import mmap
import os
import socket
import weakref
from collections.abc import Callable, Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.shared_host import SharedHostArena, SharedHostIdentity
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

# cudaHostRegisterPortable | cudaHostRegisterMapped
_HOST_REGISTER_FLAGS = 3


def _parse_owner(owner: str) -> tuple[int, int, SharedHostIdentity]:
    """Split an owner entry ``pid:fd:size:device:inode`` into its integers."""
    pid, fd, size, device, inode = (int(field) for field in owner.split(":"))
    return pid, fd, SharedHostIdentity(size, device, inode)


def _open_owner_arena(owner: str) -> SharedHostArena:
    pid, fd, identity = _parse_owner(owner)
    # The creator keeps its fd open for as long as it lives (SharedHostArena has
    # no finalizer), so /proc finds it. The seals belong to the inode and survive
    # the reopen; the arena checks them.
    return SharedHostArena(
        os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR | os.O_CLOEXEC), identity
    )


def _host_view(buffer: mmap.mmap | memoryview, like: torch.Tensor) -> torch.Tensor:
    """The bytes of ``buffer`` as a tensor of ``like``'s dtype and shape. The
    tensor keeps ``buffer`` alive."""
    nbytes = like.numel() * like.element_size()
    flat = torch.frombuffer(buffer, dtype=torch.uint8, count=nbytes)
    return flat.view(like.dtype).view(like.shape)


def _shared_host_copy(
    source: torch.Tensor, share_key: str
) -> tuple[SharedHostArena, int]:
    """The host copy of ``source`` that all processes using ``share_key`` map.

    Returns the arena and the pid of the process that created it. The first
    process to claim the key in the default torch.distributed store creates the
    arena and publishes its handle there; every other process opens that arena
    instead.

    A creator copies ``source`` into its arena before it publishes the handle,
    so a process that attaches finds the creator's values and nothing of the
    creator overwrites what the attacher or a later weight load writes. A process
    that loses the claim drops its own arena. Processes that share a key must
    hold the same table (see ``_share_key``) and write the same bytes into it.
    """
    if not source.is_contiguous():
        raise ValueError(f"Cannot share a non-contiguous tensor ({share_key})")
    arena = SharedHostArena.create(source.numel() * source.element_size(), "vllm-uva")
    host = _host_view(arena.buffer, source)
    host.copy_(source)
    identity = arena.identity
    mine = f"{os.getpid()}:{arena.fileno()}"
    mine += f":{identity.size}:{identity.device}:{identity.inode}"
    store = dist.distributed_c10d._get_default_store()
    owner = store.compare_set(share_key, "", mine).decode()
    if owner == mine:
        return arena, os.getpid()
    del host
    arena.close()
    owner_pid, _, _ = _parse_owner(owner)
    if owner_pid == os.getpid():
        raise RuntimeError(
            f"{share_key} was already shared by this process; one process must "
            "not offload the same shared table twice"
        )
    return _open_owner_arena(owner), owner_pid


def _register_pinned(arena: SharedHostArena, exporter: memoryview) -> None:
    """Pin the arena's pages for CUDA, for as long as a tensor over ``exporter``
    (a memoryview of the arena's mapping) exists.

    The registration is released once the last tensor over ``exporter`` is
    gone, while the memory is still mapped. The device view that UVA offloading
    builds holds the host tensor, so it holds the registration too.
    """
    pointer = arena.address
    cudart = torch.cuda.cudart()
    # Concurrent long-term pins of one shared region can make
    # pin_user_pages fail with ENOMEM (see pin_mmap_region in
    # vllm/v1/kv_offload/cpu/gpu_worker.py); the stages that map one
    # embedding table register it one after another.
    with arena.registration_guard():
        error = cudart.cudaHostRegister(pointer, arena.size, _HOST_REGISTER_FLAGS)
    if error != cudart.cudaError.success:
        raise RuntimeError(f"cudaHostRegister of {arena.size} bytes failed: {error}")
    _release_with_mapping(exporter, lambda: cudart.cudaHostUnregister(pointer))


def _release_with_mapping(exporter: memoryview, release: Callable[[], object]) -> None:
    """Call ``release`` once ``exporter`` is dropped, while the memory it views
    is still mapped.

    A finalizer on the mmap object itself would run after its munmap. The
    finalizer here holds the mapping (``exporter.obj``) as an argument, so the
    memory is unmapped only after ``release`` has returned."""
    finalizer = weakref.finalize(exporter, _release_then_unmap, release, exporter.obj)
    # At exit the CUDA context goes first and takes the registration along.
    # ``atexit`` is a documented writable property; typeshed's __slots__
    # declaration does not list it.
    finalizer.atexit = False  # type: ignore[misc]


def _release_then_unmap(release: Callable[[], object], mapping: object) -> None:
    # ``mapping`` is held only so that it outlives ``release``.
    release()


def _share_key(module: nn.Module, qualified_name: str, p: torch.Tensor) -> str | None:
    """Key under which processes on this host share an offloaded tensor, or
    None for a private copy.

    Only non-empty embedding tables are shared: pipeline stages that hold the
    same table (the first stage and a drafter's replica on the last one) are
    built under the same qualified name and load it from the same checkpoint
    tensor, and no stage transforms it afterwards, so all of them write the
    same bytes. The checkpoint is part of the key, so another model's table of
    the same name and shape does not match.
    """
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.distributed.parallel_state import get_dp_group
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    if (
        not isinstance(module, VocabParallelEmbedding)
        or not dist.is_initialized()
        or p.numel() == 0
    ):
        return None
    model_config = get_current_vllm_config().model_config
    # The owner entry names a pid, so only processes that see the same pids
    # (one pid namespace on one host) may share.
    return (
        f"vllm.uva_offload/{socket.gethostname()}/{os.readlink('/proc/self/ns/pid')}"
        f"/{model_config.model}@{model_config.revision}"
        f"/dp{get_dp_group().rank_in_group}/tp{get_tensor_model_parallel_rank()}"
        f"/{qualified_name}/{tuple(p.shape)}/{p.dtype}"
    )


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.
    """

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        modules = [self._maybe_offload_to_cpu(module) for module in modules_generator]
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def wrap_module(
        self, module: nn.Module, *, parameter_prefix: str = ""
    ) -> nn.Module:
        """Apply UVA offloading to one component with its qualified prefix."""
        previous_offload_bytes = self.cpu_offload_bytes
        module = self._maybe_offload_to_cpu(module, parameter_prefix=parameter_prefix)
        if self.cpu_offload_bytes > previous_offload_bytes:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return module

    def _is_parameter_selected(self, name: str, parameter_prefix: str = "") -> bool:
        if not self.cpu_offload_params:
            return True
        qualified_name = f"{parameter_prefix}.{name}" if parameter_prefix else name
        return any(
            f".{param}." in f".{qualified_name}." for param in self.cpu_offload_params
        )

    def _host_copy(
        self,
        module: nn.Module,
        name: str,
        data: torch.Tensor,
        *,
        parameter_prefix: str,
    ) -> torch.Tensor:
        """Host tensor holding one parameter's current values.

        A table that processes on this host share is pinned at its exact size in
        a shared arena (``Tensor.pin_memory()`` rounds up to a power of two and
        locks all of it); every other parameter is a private copy."""
        share_key = None
        if self.pin_memory and current_platform.is_cuda_alike():
            qualified_name = f"{parameter_prefix}.{name}" if parameter_prefix else name
            owner_name, _, _ = name.rpartition(".")
            owner = module.get_submodule(owner_name) if owner_name else module
            share_key = _share_key(owner, qualified_name, data)
        if share_key is None:
            cpu_data = data.to(device="cpu")
            return cpu_data.pin_memory() if self.pin_memory else cpu_data
        arena, owner_pid = _shared_host_copy(data, share_key)
        exporter = memoryview(arena.buffer)
        host = _host_view(exporter, data)
        _register_pinned(arena, exporter)
        if owner_pid != os.getpid():
            logger.info(
                "UVA offload: %s (%s) maps the host copy of process %d.",
                qualified_name,
                format_gib(data.numel() * data.element_size()),
                owner_pid,
            )
        return host

    def _maybe_offload_to_cpu(
        self, module: nn.Module, *, parameter_prefix: str = ""
    ) -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # Add dots to require exact path-segment matches. A qualified
            # prefix lets selectors such as ``visual`` target an entire tower
            # even though ``module.named_parameters()`` returns relative names.
            if not self._is_parameter_selected(name, parameter_prefix):
                continue

            # A component may contain layers already visited by make_layers().
            # Do not copy or account for the same UVA parameter twice when the
            # enclosing component is registered afterwards.
            if getattr(p, "_vllm_is_uva_offloaded", False):
                continue

            cpu_data = self._host_copy(
                module, name, p.data, parameter_prefix=parameter_prefix
            )

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device,
                    # it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                module.forward = forward
                return output

            module.forward = forward

        return module
