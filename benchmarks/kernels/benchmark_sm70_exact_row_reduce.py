# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 explicit row-plan correctness controls; launch under an owned GPU lease.

Use torchrun --standalone --nproc_per_node=4 with this script. Measurements
are isolated communication diagnostics, never full-model acceptance.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--full-shape", action="store_true")
    args = parser.parse_args()
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers import sm70_collectives as shared
    from vllm.video.benchmark import source_provenance

    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise ValueError("This control requires exactly four TP ranks")
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.manual_seed(5091 + rank)
    record = dict(
        rank=rank,
        state="running",
        cases=[],
        guards=[],
        source=source_provenance(),
        scope="isolated operator control; no model acceptance",
    )
    if args.extension:
        spec = importlib.util.spec_from_file_location(
            args.extension.stem, args.extension
        )
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        shared._extension = lambda: extension
        sys.modules["onecat_sm70_exact_reduce"] = extension
        record["extension"] = {
            "path": str(args.extension),
            "sha256": hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        }
    source = (
        Path(shared.__file__).resolve().parents[3]
        / "csrc/sm70_turbomind/ops/exact_row_reduce.cu"
    )
    record["cuda_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    record["benchmark_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    ):
        init_distributed_environment(4, rank, "env://", local, "nccl")
        initialize_model_parallel(4)
        try:
            group = get_tp_group()
            for label, shape, budget in (
                ("one-rank-invalid-shape", (3, 3) if rank == 0 else (4, 3), 2**30),
                ("one-rank-small-budget", (4, 3), 1 if rank == 0 else 2**30),
                ("different-valid-shapes", (4, 3) if rank == 0 else (8, 3), 2**30),
            ):
                try:
                    shared.SM70ExactRowReductionPlan(
                        group, shape, memory_budget_bytes=budget
                    )
                except ValueError:
                    record["guards"].append(label)
                else:
                    raise AssertionError(f"Expected collective rejection: {label}")
            shapes = [
                (4, 3),
                (12, 65),
                (68, 257),
                (128, 768),
                (260, 1024),
                (1028, 3072),
            ]
            if args.full_shape:
                shapes.append((34560, 5376))
            for shape in shapes:
                started = time.perf_counter()
                plan = shared.SM70ExactRowReductionPlan(
                    group, shape, memory_budget_bytes=4 * 2**30
                )
                calibration_seconds = time.perf_counter() - started
                try:
                    with torch.inference_mode():
                        for label, scale in (
                            ("ordinary", 1.0),
                            ("wide", 1e30),
                            ("subnormal", 1e-40),
                        ):
                            storage = (
                                torch.randn(shape[0] * shape[1] + 1, device="cuda")
                                * scale
                            )
                            value = storage[1:].view(shape)
                            ref = group.all_reduce(value).chunk(4)[rank]
                            actual = plan.reduce(value)
                            mismatch = int(
                                torch.count_nonzero(
                                    ref.view(torch.int32) != actual.view(torch.int32)
                                )
                            )
                            record["cases"].append(
                                dict(
                                    shape=shape,
                                    input=label,
                                    storage_offset=1,
                                    mismatch=mismatch,
                                    calibration_seconds=calibration_seconds,
                                    raw_ipc_bytes=plan.raw_ipc_bytes,
                                )
                            )
                        value = torch.full(
                            shape,
                            float("inf") if rank < 2 else -float("inf"),
                            device="cuda",
                        )
                        ref = group.all_reduce(value).chunk(4)[rank]
                        actual = plan.reduce(value)
                        record["cases"].append(
                            dict(
                                shape=shape,
                                input="opposing-inf",
                                mismatch=int(
                                    torch.count_nonzero(
                                        ref.view(torch.int32)
                                        != actual.view(torch.int32)
                                    )
                                ),
                            )
                        )
                        with torch.cuda.stream(torch.cuda.Stream()):
                            try:
                                plan.reduce(value)
                            except RuntimeError:
                                record["guards"].append("different-stream")
                            else:
                                raise AssertionError("Different stream was accepted")
                        vote = torch.tensor(
                            int(all(x["mismatch"] == 0 for x in record["cases"])),
                            device="cuda",
                        )
                        torch.distributed.all_reduce(
                            vote, op=torch.distributed.ReduceOp.MIN
                        )
                        if not vote.item():
                            raise AssertionError("Native FP32 bits changed")
                        if shape == (34560, 5376):
                            value.normal_()
                            for _ in range(3):
                                group.all_reduce(value)
                                plan.reduce(value)
                            torch.cuda.synchronize()
                            times = {"native": [], "peer_rows": []}
                            for repeat in range(7):
                                for name in (
                                    ("native", "peer_rows")
                                    if repeat % 2 == 0
                                    else ("peer_rows", "native")
                                ):
                                    torch.distributed.barrier(group=group.cpu_group)
                                    torch.cuda.synchronize()
                                    start, end = (
                                        torch.cuda.Event(enable_timing=True),
                                        torch.cuda.Event(enable_timing=True),
                                    )
                                    start.record()
                                    output = (
                                        group.all_reduce(value)
                                        if name == "native"
                                        else plan.reduce(value)
                                    )
                                    end.record()
                                    end.synchronize()
                                    times[name].append(start.elapsed_time(end))
                                    del output
                            record["times_ms"] = times
                            record["median_ms"] = {
                                key: statistics.median(values)
                                for key, values in times.items()
                            }
                finally:
                    plan.close()
                try:
                    plan.reduce(value)
                except RuntimeError:
                    record["guards"].append("closed-plan")
                else:
                    raise AssertionError("Closed plan was accepted")
                print(
                    json.dumps(dict(rank=rank, shape=shape, state="passed")), flush=True
                )
            record["state"] = "passed_operator_control"
        except BaseException as error:
            record.update(state="failed", error=repr(error))
            raise
        finally:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"rank-{rank}.json").write_text(json.dumps(record, indent=2))
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
