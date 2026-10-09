# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in native dispatch observation, separate from selector predictions.

Use around eager execution or graph capture, never around the timed loop.
Capture records launches into a graph; replay correctness/timing is separate.
No tensors or process-local addresses are retained in the report.
"""

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class NativeDispatchTrace(TorchDispatchMode):
    def __init__(self, phase: str = "eager"):
        super().__init__()
        if phase not in ("eager", "capture"):
            raise ValueError("native trace phase must be eager or capture")
        self.phase = phase
        self.records: list[dict] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = func._schema.name
        if name.split("::")[0] in ("_C", "_moe_C", "_C_gguf", "_C_qwen38"):
            tensors = [
                value
                for value in tree_leaves((args, kwargs))
                if isinstance(value, torch.Tensor)
            ]
            fake = any(
                isinstance(tensor, FakeTensor) or tensor.is_meta for tensor in tensors
            )
            cuda = any(tensor.is_cuda for tensor in tensors)
            self.records.append(
                {
                    "operator": name,
                    "phase": self.phase,
                    "evidence": "fake_dispatch"
                    if fake
                    else "cuda_dispatch"
                    if cuda
                    else "host_dispatch",
                    "inputs": [
                        {
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "stride": list(tensor.stride()),
                        }
                        for tensor in tensors
                    ],
                }
            )
        return result

    def report(self) -> dict:
        return {
            "evidence": "observed operator dispatch; not a selector prediction",
            "completion": (
                "caller must check synchronized outputs; capture is not replay"
            ),
            "records": self.records,
        }
