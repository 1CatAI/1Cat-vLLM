# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in NVTX windows that exclude long-prefill chunks from q8 traces."""

import functools
import importlib
import json
import os
from pathlib import Path


def is_q8_decode(scheduler_output) -> bool:
    scheduled = scheduler_output.num_scheduled_tokens
    if len(scheduled) != 1 or scheduler_output.total_num_scheduled_tokens != 8:
        return False
    request_id = next(iter(scheduled))
    proposals = scheduler_output.scheduled_spec_decode_tokens.get(request_id, ())
    cached = scheduler_output.scheduled_cached_reqs
    emitted = dict(zip(cached.req_ids, cached.num_output_tokens))
    return len(proposals) == 7 or emitted.get(request_id, 0) > 0


def install_context_trace(root: Path) -> None:
    import torch

    targets = {
        "vllm.v1.worker.gpu.model_runner": [
            "GPUModelRunner.execute_model",
            "GPUModelRunner.sample_tokens",
            "GPUModelRunner.sample",
            "GPUModelRunner.postprocess_sampled",
        ],
        "vllm.v1.worker.gpu.cudagraph_utils": [
            "ModelCudaGraphManager.run_fullgraph",
        ],
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator": [
            "DFlash2Speculator.propose",
        ],
    }
    step = -1
    armed = False
    finished = False
    observations = []

    def wrap(original, label):
        @functools.wraps(original)
        def call(*args, **kwargs):
            nonlocal step, armed, finished
            rank = torch.distributed.get_rank()
            if label == "GPUModelRunner.execute_model":
                scheduler = args[1]
                if not armed and (root / "arm").exists():
                    armed = True
                if armed and not finished and is_q8_decode(scheduler):
                    step += 1
                    cached = scheduler.scheduled_cached_reqs
                    observations.append(
                        {
                            "step": step,
                            "scheduled": scheduler.total_num_scheduled_tokens,
                            "request_ids": list(cached.req_ids),
                            "computed_tokens": list(cached.num_computed_tokens),
                            "output_tokens": list(cached.num_output_tokens),
                        }
                    )
                    if step == 8:
                        torch.accelerator.synchronize()
                        torch.distributed.barrier()
                        torch.cuda.cudart().cudaProfilerStart()
                        print(
                            f"CONTEXT_TRACE_START rank={rank} step={step}", flush=True
                        )
                    elif step == 20:
                        torch.accelerator.synchronize()
                        torch.distributed.barrier()
                        torch.cuda.cudart().cudaProfilerStop()
                        finished = True
                        (root / f"rank{rank}-rounds.json").write_text(
                            json.dumps(observations, indent=2) + "\n"
                        )
                        print(f"CONTEXT_TRACE_STOP rank={rank} step={step}", flush=True)
            if not armed or finished:
                return original(*args, **kwargs)
            torch.cuda.nvtx.range_push(f"quasar/rank{rank}/round{step}/{label}")
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        return call

    for module_name, labels in targets.items():
        module = importlib.import_module(module_name)
        for label in labels:
            owner_name, method = label.split(".")
            owner = getattr(module, owner_name)
            setattr(owner, method, wrap(getattr(owner, method), label))


class ContextCostTraceExtension:
    """No route changes; explicitly select the performance installer separately."""


if trace_root := os.getenv("VLLM_DFLASH2_CONTEXT_COST_TRACE_DIR"):
    install_context_trace(Path(trace_root))
