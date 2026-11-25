# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for platform-local HC custom-op registration and dispatch."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("order", [("amd", "nvidia"), ("nvidia", "amd")])
def test_hc_platform_registration(order: tuple[str, str]) -> None:
    # A fresh dispatcher is needed to exercise both first-import orders.
    result = subprocess.run(
        [sys.executable, "-m", "tests.models.qwen4_exp.test_hc_registration", *order],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_platform_registration(order: tuple[str, ...]) -> None:
    from contextlib import ExitStack
    from importlib import import_module
    from unittest.mock import MagicMock, patch

    import torch
    import triton
    import triton.language as tl

    import vllm.platforms as platforms
    import vllm.triton_utils as triton_utils
    from vllm.platforms.cpu import CpuPlatform

    # Use real torch.library registration on CPU, replacing only GPU launches.
    platforms._current_platform = CpuPlatform()
    triton_utils.triton = triton
    triton_utils.tl = tl
    modules = {
        backend: import_module(f"vllm.models.qwen4_exp.{backend}.ops.hc")
        for backend in order
    }
    x = torch.empty(2, 8)
    weight = torch.empty(4)
    block = torch.empty(2, 4)
    injection = torch.empty(2, 2)
    cases = [
        ("grouped_gemma_rmsnorm", (x, weight, 1e-6, 2), (2,), ((2, 8),)),
        ("hc_silu", (x, 2), (1,), ((2, 8),)),
        ("hc_gate_mix", (x, x, 2), (2,), ((2, 4),)),
        ("hc_combine", (x, block, injection, 2), (3,), ((2, 8),)),
        (
            "hc_combine_norm",
            (x, block, injection, weight, 1e-6, 2),
            (4, 5),
            ((2, 8), (2, 8)),
        ),
    ]

    for name, args, output_positions, shapes in cases:
        with ExitStack() as stack:
            kernels = {}
            for backend, module in modules.items():
                kernel = MagicMock()

                def launch(
                    *kernel_args,
                    value=1 if backend == "amd" else 2,
                    positions=output_positions,
                    **kwargs,
                ):
                    for position in positions:
                        kernel_args[position].fill_(value)

                kernel.__getitem__.return_value.side_effect = launch
                stack.enter_context(patch.object(module, f"_{name}_kernel", kernel))
                kernels[backend] = kernel

            for backend, module in modules.items():
                wrapper = getattr(module, name)
                output = wrapper(*args)
                outputs = output if isinstance(output, tuple) else (output,)
                assert len(outputs) == len(shapes)
                for tensor, shape in zip(outputs, shapes):
                    assert tensor.shape == shape
                    assert tensor.dtype == x.dtype
                    assert torch.all(tensor == (1 if backend == "amd" else 2))
                kernels[backend].__getitem__.return_value.assert_called_once()
                other = "nvidia" if backend == "amd" else "amd"
                kernels[other].__getitem__.return_value.assert_not_called()
                kernels[backend].reset_mock()

                # The fake implementation must remain available under each name.
                meta_args = tuple(
                    arg.to(device="meta") if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                )
                meta_output = wrapper(*meta_args)
                meta_outputs = (
                    meta_output if isinstance(meta_output, tuple) else (meta_output,)
                )
                assert len(meta_outputs) == len(shapes)
                for tensor, shape in zip(meta_outputs, shapes):
                    assert tensor.device.type == "meta"
                    assert tensor.shape == shape
                    assert tensor.dtype == x.dtype
                for kernel in kernels.values():
                    kernel.__getitem__.return_value.assert_not_called()


if __name__ == "__main__":
    _check_platform_registration(tuple(sys.argv[1:]))
