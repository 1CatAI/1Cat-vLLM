# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture a bounded window of late prefill chunks, excluding q8 decode."""

import functools
import importlib
import json
import os
from pathlib import Path


def prefill_position(scheduler):
    if len(scheduler.num_scheduled_tokens) != 1:
        return None
    count = scheduler.total_num_scheduled_tokens
    if count <= 32 or any(scheduler.scheduled_spec_decode_tokens.values()):
        return None
    cached = scheduler.scheduled_cached_reqs
    if len(cached.req_ids) != 1 or any(cached.num_output_tokens):
        return None
    return cached.num_computed_tokens[0], count


def install_prefill_trace(root: Path, start_position: int, chunks: int):
    import torch

    targets = {
        "vllm.v1.worker.gpu.model_runner": [
            "GPUModelRunner.execute_model",
            "GPUModelRunner.sample_tokens",
            "GPUModelRunner.sample",
            "GPUModelRunner.postprocess_sampled",
        ],
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator": [
            "DFlash2Speculator.propose",
        ],
    }
    active = False
    finished = False
    step = -1
    observations = []

    def wrap(original, label):
        @functools.wraps(original)
        def call(*args, **kwargs):
            nonlocal active, finished, step
            if finished:
                return original(*args, **kwargs)
            rank = torch.distributed.get_rank()
            if label == "GPUModelRunner.execute_model":
                position = prefill_position(args[1])
                if position is not None and position[0] >= start_position:
                    if not active:
                        torch.accelerator.synchronize()
                        torch.distributed.barrier()
                        torch.cuda.cudart().cudaProfilerStart()
                        active = True
                    step += 1
                    observations.append(
                        {
                            "step": step,
                            "computed_tokens": position[0],
                            "scheduled": position[1],
                        }
                    )
                elif active:
                    raise RuntimeError("Prefill capture reached a non-prefill step")
            if not active:
                return original(*args, **kwargs)
            torch.cuda.nvtx.range_push(f"prefill/rank{rank}/chunk{step}/{label}")
            try:
                result = original(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
            if label == "GPUModelRunner.sample_tokens" and step == chunks - 1:
                torch.accelerator.synchronize()
                torch.distributed.barrier()
                torch.cuda.cudart().cudaProfilerStop()
                active = False
                finished = True
                (root / f"rank{rank}-prefill.json").write_text(
                    json.dumps(observations, indent=2) + "\n"
                )
                print(f"PREFILL_TRACE_STOP rank={rank} chunks={chunks}", flush=True)
            return result

        return call

    for module_name, labels in targets.items():
        module = importlib.import_module(module_name)
        for label in labels:
            owner_name, method = label.split(".")
            owner = getattr(module, owner_name)
            setattr(owner, method, wrap(getattr(owner, method), label))


if trace_root := os.getenv("VLLM_DFLASH2_PREFILL_COST_TRACE_DIR"):
    install_prefill_trace(
        Path(trace_root),
        int(os.getenv("VLLM_DFLASH2_PREFILL_COST_TRACE_START", "115360")),
        int(os.getenv("VLLM_DFLASH2_PREFILL_COST_TRACE_CHUNKS", "4")),
    )
