# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether the installed Triton maps ``tl.dot`` onto tensor-core MMA."""

import functools

from vllm.triton_utils import tl, triton


@triton.jit
def _dot_probe_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 16)
    tile = offsets[:, None] * 16 + offsets[None, :]
    tl.store(c_ptr + tile, tl.dot(tl.load(a_ptr + tile), tl.load(b_ptr + tile)))


@functools.cache
def lowers_dot_to_mma(capability: int) -> bool:
    """Whether an fp16 16x16x16 ``tl.dot`` becomes MMA for CUDA SM ``capability``.

    Compiles only; nothing runs on a device.
    """
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(
        _dot_probe_kernel, {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"}
    )
    ptx = triton.compile(source, target=GPUTarget("cuda", capability, 32)).asm["ptx"]
    return "mma.sync" in ptx
