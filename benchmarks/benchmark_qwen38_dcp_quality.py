# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded source-installed Qwen3.8 DCP/KV qualification; no resident server.

Run separate DCP1 and DCP2 processes with the same calibrated checkpoint and
compare task scores, with token IDs retained as diagnostics. Exact token parity
is opt-in, not the default acceptance gate. No private runtime overlay is loaded.
An 8K smoke is not a 256K quality claim; use --long-context for that boundary.
The worker manifest uses a trusted local callable RPC. Explicitly opt in with
VLLM_ALLOW_INSECURE_SERIALIZATION=1 for this offline check only, not a service.
"""

import argparse
import hashlib
import importlib
import json
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def runtime_sources():
    """Record actual package/extension locations, not only the driver cwd."""
    names = (
        "vllm",
        "vllm._C",
        "vllm._C_stable_libtorch",
        "vllm._moe_C",
        "flash_attn_v100",
        "flash_qla",
    )
    result = {
        name: str(Path(importlib.import_module(name).__file__).resolve())
        for name in names
    }
    result["version"] = importlib.import_module("vllm").__version__
    return result


def validate_spawn_runtime():
    """Catch dataset-import sys.path poisoning before loading model weights."""
    expected = runtime_sources()
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        actual = pool.submit(runtime_sources).result(timeout=60)
    if actual != expected:
        raise RuntimeError(
            f"Spawned runtime differs from driver: {actual} != {expected}"
        )
    return expected


def worker_manifest(worker):
    import torch

    import vllm

    cfg = worker.vllm_config
    cache = worker.model_runner.kv_cache_config
    return {
        "rank": worker.rank,
        "source": vllm.__file__,
        "runtime": runtime_sources(),
        "torch": torch.__version__,
        "vllm_version": vllm.__version__,
        "cuda": torch.version.cuda,
        "kv_dtype": cfg.cache_config.cache_dtype,
        "ssm_dtype": cfg.cache_config.mamba_ssm_cache_dtype,
        "dcp": cfg.parallel_config.decode_context_parallel_size,
        "dcp_comm_backend": cfg.parallel_config.dcp_comm_backend,
        "mtp": cfg.speculative_config is not None,
        "prefix_cache": cfg.cache_config.enable_prefix_caching,
        "graph_mode": str(cfg.compilation_config.cudagraph_mode),
        "allocated_bytes": torch.accelerator.memory_allocated(),
        "reserved_bytes": torch.accelerator.memory_reserved(),
        "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
        "physical_kv_bytes": sum(t.size for t in cache.kv_cache_tensors),
        "num_blocks": cache.num_blocks,
        "cache_groups": [
            {
                "layers": group.layer_names,
                "type": type(group.kv_cache_spec).__name__,
                "block_size": group.kv_cache_spec.block_size,
                "page_bytes": group.kv_cache_spec.page_size_bytes,
                "dcp_sharded": group.kv_cache_spec.dcp_sharded,
            }
            for group in cache.kv_cache_groups
        ],
        "fp16_reduced_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "bf16_reduced_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "fp16_accumulation": getattr(
            torch.backends.cuda.matmul, "allow_fp16_accumulation", False
        ),
        "sampler_checks_nans": worker.model_runner.sampler.compute_nans,
        "ple_environment": {
            k: v for k, v in os.environ.items() if "PLE" in k and k.startswith("VLLM_")
        },
    }


def validate_manifest_transport():
    """Reject unsupported callable RPC before loading any model weights."""
    from vllm.v1.serial_utils import MsgpackEncoder

    MsgpackEncoder().encode((0, 0, "collective_rpc", (worker_manifest, 30, (), {})))


def build_prompts(tokenizer, long_context):
    """Materialize and validate every input before allocating the model."""
    cases = []
    for name, text, expected in (
        ("arithmetic", "请计算 19 × 23，在最后写出 RESULT=计算结果。", "437"),
        ("copy", "档案编号是 CEDAR-47|8261。请准确复述编号。", "CEDAR-47|8261"),
    ):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        cases.extend(
            [("official_" + name, ids, expected), ("greedy_" + name, ids, expected)]
        )

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ARCHIVE_BODY\n找到档案口令，只输出口令。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    lead, tail = [
        tokenizer.encode(s, add_special_tokens=False)
        for s in rendered.split("ARCHIVE_BODY")
    ]
    filler = tokenizer.encode(
        "这是一条普通档案记录，没有口令。\n", add_special_tokens=False
    )
    record = tokenizer.encode(
        "\n唯一档案口令：MAPLE-8261。\n", add_special_tokens=False
    )
    for length in [8192, 32768, 261632] if long_context else [8192]:
        count = length - len(lead) - len(tail) - len(record)
        body = (filler * ((count + len(filler) - 1) // len(filler)))[:count]
        ids = lead + body[: count // 2] + record + body[count // 2 :] + tail
        if len(ids) != length:
            raise ValueError(f"Retrieval prompt length mismatch: {length}")
        cases.append((f"greedy_retrieval_{length}", ids, "MAPLE-8261"))
    boundary_ids = (
        (filler * ((262143 + len(filler) - 1) // len(filler)))[:262143]
        if long_context
        else None
    )
    prompts = [(name, ids) for name, ids, _ in cases]
    if boundary_ids is not None:
        prompts.append(("exact_256k_boundary", boundary_ids))
    for name, ids in prompts:
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(token, int) and token >= 0 for token in ids)
        ):
            raise TypeError(f"{name}: expected a nonempty list of integer token IDs")
    return cases, boundary_ids


def token_difference(control, candidate):
    """Report the first divergence without equating text health with parity."""
    common = 0
    for expected, actual in zip(control, candidate):
        if expected != actual:
            break
        common += 1
    matches = common == len(control) == len(candidate)
    return {
        "matches": matches,
        "first_differing_token_0based": None if matches else common,
        "reference_tokens": len(control),
        "candidate_tokens": len(candidate),
        "reference_token": control[common] if common < len(control) else None,
        "candidate_token": candidate[common] if common < len(candidate) else None,
    }


def run(args):
    # The corruption metric stays false when this opt-in is disabled. Enable
    # the native sampler check before worker startup; no production default is
    # changed. This diagnostic disables the greedy-only argmax fast path, so
    # these health timings must not be presented as production speed results.
    os.environ["VLLM_COMPUTE_NANS_IN_LOGITS"] = "1"
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    validate_manifest_transport()

    report = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cases": [],
        "quality_failures": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation = json.loads((Path(args.model) / "generation_config.json").read_text())
    report["generation_sampling"] = {
        key: generation[key] for key in ("temperature", "top_p", "top_k")
    }
    official = SamplingParams(
        max_tokens=1024,
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        top_k=generation["top_k"],
        seed=0,
        ignore_eos=False,
    )
    # Greedy, natural EOS is used ONLY for deterministic DCP1/2 token comparison.
    deterministic = SamplingParams(max_tokens=512, temperature=0, seed=0)
    prompts, boundary_ids = build_prompts(tokenizer, args.long_context)
    report["preflight"] = {
        "prompt_lengths": {name: len(ids) for name, ids, _ in prompts},
        "boundary_tokens": len(boundary_ids) if boundary_ids is not None else 0,
    }
    dataset_bundle = None
    if args.dataset_spec:
        if __package__:
            from .qwen38_dcp_datasets import evaluate_datasets, prepare_datasets
        else:
            from qwen38_dcp_datasets import evaluate_datasets, prepare_datasets
        dataset_bundle = prepare_datasets(args.dataset_spec, tokenizer)
        report["preflight"]["dataset_manifest"] = dataset_bundle[2]
    report["preflight"]["runtime"] = validate_spawn_runtime()
    if args.preflight_only:
        save()
        print(json.dumps(report["preflight"]), flush=True)
        return
    reference = {}
    dataset_reference = None
    if args.reference:
        previous = json.loads(args.reference.read_text())
        previous_args = previous["args"]
        if not previous.get("complete"):
            raise ValueError("Reference run did not complete")
        for key in (
            "model",
            "kv_dtype",
            "kv_gib",
            "gpu_memory_utilization",
            "long_context",
        ):
            if previous_args.get(key) != report["args"][key]:
                raise ValueError(f"Reference contract mismatch: {key}")
        if (
            previous.get("preflight", {}).get("runtime")
            != report["preflight"]["runtime"]
        ):
            raise ValueError("Reference runtime contract mismatch")
        reference = {case["name"]: case for case in previous["cases"]}
        if dataset_bundle:
            dataset_reference = previous.get("dataset")
            if (
                not dataset_reference
                or not dataset_reference.get("complete")
                or dataset_reference["manifest"] != dataset_bundle[2]
                or previous["generation_sampling"] != report["generation_sampling"]
            ):
                raise ValueError("Dataset reference contract mismatch")

    llm = None
    try:
        save()
        llm = LLM(
            model=args.model,
            dtype="half",
            tensor_parallel_size=4,
            decode_context_parallel_size=args.dcp,
            kv_cache_dtype=args.kv_dtype,
            kv_cache_memory_bytes=int(args.kv_gib * (1 << 30)),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=262144,
            max_num_batched_tokens=8192,
            max_num_seqs=4,
            language_model_only=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            mamba_cache_mode="align",
            mamba_ssm_cache_dtype="float32",
            disable_log_stats=False,
            seed=0,
        )
        report["workers"] = llm.collective_rpc(worker_manifest, timeout=30)
        for worker in report["workers"]:
            if (
                worker["mtp"]
                or worker["prefix_cache"]
                or worker["ssm_dtype"] != "float32"
                or worker["fp16_reduced_reduction"]
                or worker["bf16_reduced_reduction"]
                or worker["fp16_accumulation"]
                or not worker["sampler_checks_nans"]
                or worker["runtime"] != report["preflight"]["runtime"]
                or worker["dcp"] != args.dcp
                or worker["kv_dtype"]
                != ("fp8_e4m3" if args.kv_dtype == "auto" else args.kv_dtype)
                or worker["ple_environment"].get("VLLM_PLE_DISK_OFFLOAD") != "1"
            ):
                raise RuntimeError("Worker precision/state contract mismatch")
        save()

        def check(name, ids, sampling, expected):
            start = time.perf_counter()
            result = llm.generate(
                [{"prompt_token_ids": ids}], sampling, use_tqdm=False
            )[0]
            out = result.outputs[0]
            metrics = result.metrics
            answer = out.text.rsplit("</think>", 1)[-1]
            prompt_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            corrupted = metrics is not None and metrics.is_corrupted
            passed = (
                out.finish_reason == "stop" and expected in answer and not corrupted
            )
            case = {
                "name": name,
                "input_tokens": len(ids),
                "prompt_hash": prompt_hash,
                "sampling": str(sampling),
                "elapsed_s": time.perf_counter() - start,
                "text": out.text,
                "token_ids": list(out.token_ids),
                "finish_reason": out.finish_reason,
                "health_passed": passed,
            }
            if metrics is not None and not metrics.is_corrupted:
                prefill = metrics.first_token_ts - metrics.scheduled_ts
                decode = metrics.last_token_ts - metrics.first_token_ts
                case.update(
                    prefill_s=prefill,
                    prefill_tps=len(ids) / prefill if prefill > 0 else None,
                    decode_s=decode,
                    decode_tps=(len(out.token_ids) - 1) / decode
                    if decode > 0
                    else None,
                )
            if reference and name.startswith("greedy_"):
                control = reference[name]
                case["token_difference"] = token_difference(
                    control["token_ids"], case["token_ids"]
                )
                case["matches_reference"] = (
                    control["prompt_hash"] == prompt_hash
                    and control["sampling"] == case["sampling"]
                    and case["token_difference"]["matches"]
                )
                case["final_answer_matches_reference"] = (
                    control["text"].rsplit("</think>", 1)[-1] == answer
                )
                if args.require_token_parity:
                    passed = passed and case["matches_reference"]
            report["cases"].append(case)
            if not passed:
                report["quality_failures"].append(name)
            save()
            print(
                json.dumps(
                    {k: v for k, v in case.items() if k not in ("text", "token_ids")}
                ),
                flush=True,
            )
            if corrupted:
                raise RuntimeError(f"Non-finite model output: {name}")

        for name, ids, expected in prompts:
            sampling = official if name.startswith("official_") else deterministic
            check(name, ids, sampling, expected)
        if boundary_ids is not None:
            # An allocation/finite-output gate, not complete-answer quality.
            boundary = llm.generate(
                [{"prompt_token_ids": boundary_ids}],
                SamplingParams(max_tokens=1, temperature=0, logprobs=1),
                use_tqdm=False,
            )[0].outputs[0]
            import math

            boundary_finite = (
                len(boundary.token_ids) == 1
                and bool(boundary.logprobs)
                and math.isfinite(boundary.logprobs[0][boundary.token_ids[0]].logprob)
            )
            report["exact_256k_boundary"] = {
                "finite": boundary_finite,
                "token_ids": list(boundary.token_ids),
            }
            if not boundary_finite:
                raise RuntimeError("Exact 256K boundary failed")
        report["workers_after"] = llm.collective_rpc(worker_manifest, timeout=30)
        if dataset_bundle:
            evaluate_datasets(
                llm, *dataset_bundle, generation, report, save, dataset_reference
            )
            report["workers_after"] = llm.collective_rpc(worker_manifest, timeout=30)
            if dataset_reference:
                for name, summary in report["dataset"]["summary"].items():
                    if not summary["observed_no_regression"]:
                        report["quality_failures"].append(f"dataset/{name}")
        report["checks_finished"] = True
        # A bounded text/parity failure must not discard the remaining quality
        # evidence and force another startup. Runtime/NaN failures still abort.
        if report["quality_failures"]:
            raise RuntimeError(f"Quality gate failed: {report['quality_failures']}")
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=30)
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dcp", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--kv-dtype", choices=("auto", "float16", "fp8_e4m3"), default="auto"
    )
    parser.add_argument("--kv-gib", type=float, default=4.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-token-parity", action="store_true")
    parser.add_argument("--dataset-spec", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
