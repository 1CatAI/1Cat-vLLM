# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and screen the native FlashInfer-style SM70 MQA work scheduler.

No runtime dispatch/default change. The production-source extension is built
before capture; all mutable state belongs to the caller, including the schedule.
"""

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def build():
    prebuilt = os.environ.get("SM70_FLASHINFER_MQA_TEST_LIBRARY")
    if prebuilt:
        path = str(Path(prebuilt).resolve(strict=True))
        if not hasattr(torch.ops._C_flashinfer_mqa_sm70, "run"):
            torch.ops.load_library(path)
        return path
    from torch.utils.cpp_extension import load

    if os.environ.get("TORCH_CUDA_ARCH_LIST") != "7.0":
        raise RuntimeError("Explicit TORCH_CUDA_ARCH_LIST=7.0 required")
    return load(
        name="flashinfer_mqa_sm70_v1",
        sources=[str(ROOT / "csrc/flashinfer_sm70/qsa_mqa.cu")],
        extra_include_paths=[str(ROOT / "flashinfer-sm70/include")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        is_python_module=False,
        verbose=True,
    )


class FlashInferMQA:
    def __init__(self, q, columns, workers):
        self.workers = workers
        self.logits = torch.empty(
            (q.shape[0], columns), device=q.device, dtype=torch.float32
        )
        self.visible = torch.empty(q.shape[0], device=q.device, dtype=torch.int32)
        self.schedule = torch.empty(
            (workers + 1, 2), device=q.device, dtype=torch.int32
        )

    def __call__(self, q, k, table, requests, positions, lengths, ratio=4):
        torch.ops._C_flashinfer_mqa_sm70.run(
            q,
            k,
            table,
            requests,
            positions,
            lengths,
            self.logits,
            self.visible,
            self.schedule,
            ratio,
            q.shape[2] ** 0.5,
            self.workers,
        )
        return self.logits, self.visible


if __name__ == "__main__":
    print(build())
