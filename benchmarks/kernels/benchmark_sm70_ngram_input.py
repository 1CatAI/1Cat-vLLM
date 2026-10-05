# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure exact PLE input preparation for MTP4 C1/C4 batches."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import vllm
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.nvidia.sm70_ngram_input import prepare_ngram_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert "site-packages" in vllm.__file__
    report = {"version": vllm.__version__, "cases": []}
    from vllm import _C

    report["core_sha256"] = hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest()
    for requests in (1, 4):
        eos = 248046
        context = torch.empty((requests, 2), dtype=torch.int32, device="cuda")
        query = torch.arange(requests + 1, dtype=torch.int32, device="cuda") * 5
        destination = torch.empty_like(query)
        mapping = torch.arange(requests, dtype=torch.int32, device="cuda")
        computed = torch.arange(requests, dtype=torch.int32, device="cuda") + 8192
        tokens = torch.arange(requests * 9216, dtype=torch.int32, device="cuda")
        tokens = tokens.reshape(requests, 9216)
        batch = SimpleNamespace(
            num_reqs=requests, num_reqs_after_padding=requests, idx_mapping=mapping
        )
        states = SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu=computed),
            all_token_ids=SimpleNamespace(gpu=tokens),
        )
        owner = SimpleNamespace(
            ngram_context=context,
            ngram_eos_token_id=eos,
            ngram_context_offsets=torch.arange(-2, 0, dtype=torch.int64, device="cuda"),
        )

        def control(
            destination=destination,
            query=query,
            owner=owner,
            batch=batch,
            states=states,
        ):
            destination.copy_(query)
            Qwen4ExpModelState._prepare_ngram_context(owner, batch, states)

        def candidate(
            context=context,
            destination=destination,
            query=query,
            mapping=mapping,
            computed=computed,
            tokens=tokens,
            requests=requests,
            eos=eos,
        ):
            prepare_ngram_input(
                context, destination, query, mapping, computed, tokens, requests, eos
            )

        results = {}
        for name, operation in (("control", control), ("candidate", candidate)):
            operation()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                operation()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(200):
                graph.replay()
            end.record()
            end.synchronize()
            gpu_us = start.elapsed_time(end) * 1000 / 200
            host_us = []
            for _ in range(100):
                torch.cuda.synchronize()
                begin = time.perf_counter_ns()
                operation()
                host_us.append((time.perf_counter_ns() - begin) / 1000)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                operation()
                torch.cuda.synchronize()
            kernels = [
                e.name
                for e in p.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            results[name] = {
                "graph_gpu_us": gpu_us,
                "host_us_median": statistics.median(host_us),
                "cuda_events": kernels,
                "cuda_event_count": len(kernels),
            }
        report["cases"].append({"m": requests * 5, "requests": requests, **results})
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
