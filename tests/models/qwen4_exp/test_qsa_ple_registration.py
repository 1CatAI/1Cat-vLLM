# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for platform-local QSA and PLE custom-op registration."""

import os
import subprocess
import sys

import pytest

_OPS = {
    "qsa": ("qwen4_exp_qsa_with_output", "qwen4_exp_qsa_with_output_amd"),
    "short_conv": ("qwen4_exp_ple_short_conv", "qwen4_exp_ple_short_conv_amd"),
}


@pytest.mark.parametrize("order", [("amd", "nvidia"), ("nvidia", "amd")])
def test_platform_imports_register_distinct_ops(order: tuple[str, str]) -> None:
    # A fresh dispatcher is needed to exercise both first-import orders.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.models.qwen4_exp.test_qsa_ple_registration",
            *order,
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_platform_registration(order: tuple[str, ...]) -> None:
    from contextlib import ExitStack
    from importlib import import_module
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform

    # Use real torch.library registration on CPU.
    platforms._current_platform = CpuPlatform()
    modules = {}
    for backend in order:
        # Importing the platform package entry avoids its internal cycles.
        import_module(f"vllm.models.qwen4_exp.{backend}.model")
        modules[backend] = (
            import_module(f"vllm.models.qwen4_exp.{backend}.qsa"),
            import_module(f"vllm.models.qwen4_exp.{backend}.ple_layer"),
        )

    # Each platform keeps exactly one schema under its own operator name.
    for names in _OPS.values():
        for name in names:
            schemas = torch._C._jit_get_schemas_for_operator(f"vllm::{name}")
            assert len(schemas) == 1, (name, schemas)

    calls: list[tuple[str, str]] = []
    layers: dict[str, dict[str, object]] = {}
    for backend, (qsa_module, ple_module) in modules.items():
        qsa_layer = qsa_module.Qwen4ExpQSAAttention.__new__(
            qsa_module.Qwen4ExpQSAAttention
        )
        torch.nn.Module.__init__(qsa_layer)

        def run_qsa(*args, name=backend):
            calls.append((name, "qsa"))

        qsa_layer._run_qsa = run_qsa
        ple_layer = ple_module.Qwen4ExpPLELayer.__new__(ple_module.Qwen4ExpPLELayer)
        torch.nn.Module.__init__(ple_layer)

        def short_conv(inputs, name=backend):
            calls.append((name, "short_conv"))
            return inputs[:1].clone()

        ple_layer._short_conv = short_conv
        layers[backend] = {"qsa.layer": qsa_layer, "ple.layer": ple_layer}

    x = torch.zeros(2, 4)
    out = torch.zeros(2, 4)
    with ExitStack() as stack:
        for backend, (qsa_module, ple_module) in modules.items():
            for module in (qsa_module, ple_module):
                stack.enter_context(
                    patch.object(
                        module,
                        "get_forward_context",
                        lambda b=backend: SimpleNamespace(no_compile_layers=layers[b]),
                    )
                )
            index = 0 if backend == "nvidia" else 1
            # Each platform name must dispatch to its own implementation.
            qsa_args = [x, x, x, x, x, out]
            if index == 0:
                qsa_args.append(None)  # NVIDIA-only optional output gate.
            qsa_args.append("qsa.layer")
            getattr(torch.ops.vllm, _OPS["qsa"][index])(*qsa_args)
            getattr(torch.ops.vllm, _OPS["short_conv"][index])(x, out, "ple.layer")

    assert sorted(calls) == sorted(
        [
            ("nvidia", "qsa"),
            ("nvidia", "short_conv"),
            ("amd", "qsa"),
            ("amd", "short_conv"),
        ]
    ), calls

    # The Meta device reaches each name's registered fake implementation.
    meta = torch.empty(2, 4, device="meta")
    meta_out = torch.empty(2, 4, device="meta")
    nvidia_qsa, amd_qsa = _OPS["qsa"]
    getattr(torch.ops.vllm, nvidia_qsa)(
        meta, meta, meta, meta, meta, meta_out, None, "layer"
    )
    getattr(torch.ops.vllm, amd_qsa)(meta, meta, meta, meta, meta, meta_out, "layer")
    for name in _OPS["short_conv"]:
        getattr(torch.ops.vllm, name)(meta, meta_out, "layer")


if __name__ == "__main__":
    _check_platform_registration(tuple(sys.argv[1:]))
