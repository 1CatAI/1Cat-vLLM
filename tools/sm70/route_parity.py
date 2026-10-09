# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record/compare fixed-prompt greedy token IDs and worker attention routes.

The engine JSON is the complete workload-specific override set. Both runs must
use the same model bytes, tokenizer, TP, cache format and graph configuration.
Record startup counters too: Python route counters observe graph construction,
not each replay. Host-KV epochs/statistics prove actual cache activity.
"""

import argparse
import os
import sys
from pathlib import Path

from tools.sm70.parity_common import (
    compare_routes,
    digest,
    provenance,
    read_json,
    require_routes,
    runtime,
    write_json,
)


class ParityWorkerExtension:
    """Named, read-only RPCs compatible with secure engine serialization."""

    def parity_snapshot(self):
        return snapshot(self)

    def parity_native_provenance(self):
        return native_provenance(self)


def snapshot(worker):
    import torch

    from vllm.v1.attention.backends import flash_attn_v100 as backend

    torch.accelerator.synchronize()
    context = worker.model_runner.compilation_config.static_forward_context
    hosts = {}
    for name, layer in context.items():
        state = getattr(layer, "host_kv", None)
        if state is not None:
            hosts[name] = dict(
                epoch=int(state.epoch.item()),
                stats=state.stats.cpu().tolist(),
                fp8=state.fp8,
                device_reference=state.device_reference,
                history_bytes=state.history.nbytes,
                hot_shape=list(state.hot_values.shape),
            )
    return dict(rank=worker.rank, routes=dict(backend._route_counts), host_kv=hosts)


def native_provenance(worker):
    from tools.sm70.parity_common import digest

    libraries = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if (
            path
            and path.endswith(".so")
            and (name.startswith("vllm.") or "flash_attn_v100_cuda" in name)
        ):
            libraries[name] = dict(path=path, sha256=digest(path))
    return dict(rank=worker.rank, libraries=libraries)


def record(args):
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_FLASH_V100_ROUTE_SUMMARY"] = "1"
    from vllm import LLM, SamplingParams

    options = read_json(args.engine_args)
    required = {
        "model",
        "tensor_parallel_size",
        "dtype",
        "kv_cache_dtype",
        "max_model_len",
        "max_num_batched_tokens",
        "enforce_eager",
    }
    if not required <= options.keys():
        raise ValueError(f"Engine contract lacks {required - options.keys()}")
    options.setdefault("attention_backend", "FLASH_ATTN_V100")
    options.setdefault("seed", 1701)
    options.setdefault("enable_prefix_caching", False)
    extension = "tools.sm70.route_parity.ParityWorkerExtension"
    if options.setdefault("worker_extension_cls", extension) != extension:
        raise ValueError("Route parity requires its named worker extension")
    identity = read_json(args.model_identity)
    if not identity.get("files"):
        raise ValueError("Model identity must contain actual file paths and SHA256s")
    for path, expected in identity["files"].items():
        if digest(path) != expected:
            raise AssertionError(f"Model/tokenizer identity changed: {path}")
    if not args.require_route and not args.require_host_fp8:
        raise ValueError("Require at least one actual route or host-FP8 activity")
    prompts = read_json(args.prompts)
    if not prompts or not all(isinstance(p, str) and p for p in prompts):
        raise ValueError("Prompts must be a nonempty JSON list of fixed strings")
    sampling = dict(
        temperature=0.0, seed=1701, max_tokens=args.max_tokens, ignore_eos=False
    )
    contract = dict(
        engine=options,
        prompts=prompts,
        sampling=sampling,
        runtime=runtime(),
        model_identity=identity,
        required_routes=args.require_route,
        host_fp8=args.require_host_fp8,
        prompt_file_sha256=digest(args.prompts),
    )
    engine = LLM(**options)
    startup = engine.collective_rpc("parity_snapshot")
    outputs = engine.generate(prompts, SamplingParams(**sampling), use_tqdm=False)
    after = engine.collective_rpc("parity_snapshot")
    workers = options["tensor_parallel_size"] * options.get("pipeline_parallel_size", 1)
    if len(startup) != workers or len(after) != workers:
        raise AssertionError("Incomplete worker route snapshots")
    require_routes(after, args.require_route)
    if args.require_host_fp8:
        for before, current in zip(startup, after, strict=True):
            active = [
                name
                for name, state in current["host_kv"].items()
                if state["fp8"]
                and not state["device_reference"]
                and state["epoch"] > before["host_kv"][name]["epoch"]
                and sum(state["stats"]) > 0
            ]
            if not active:
                raise AssertionError("No actual host-FP8 QSA cache activity on worker")
    native = engine.collective_rpc("parity_native_provenance")
    contract["native_sha256"] = [
        {name: item["sha256"] for name, item in worker["libraries"].items()}
        for worker in native
    ]
    result = dict(
        contract=contract,
        provenance=provenance(args.source_sha),
        native=native,
        startup=startup,
        after=after,
        requests=[
            dict(
                prompt_token_ids=o.prompt_token_ids,
                token_ids=list(o.outputs[0].token_ids),
                text=o.outputs[0].text,
                finish_reason=o.outputs[0].finish_reason,
            )
            for o in outputs
        ],
    )
    if any(not row["token_ids"] for row in result["requests"]):
        raise AssertionError("Empty generation cannot establish token parity")
    write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("record")
    for name in ("engine-args", "prompts", "model-identity", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--source-sha", required=True)
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--require-route", action="append", default=[])
    run.add_argument("--require-host-fp8", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        record(args)
    else:
        print(compare_routes(read_json(args.reference), read_json(args.candidate)))


if __name__ == "__main__":
    main()
