# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded source-installed Qwen3.8 DCP/KV qualification; no resident server.

Run separate DCP1 and DCP2 processes with the same calibrated checkpoint and
compare the saved token IDs. No private extension or runtime overlay is loaded.
An 8K smoke is not a 256K quality claim; use --long-context for that boundary.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def worker_manifest(worker):
    import torch

    import vllm

    cfg = worker.vllm_config
    cache = worker.model_runner.kv_cache_config
    return {
        "rank": worker.rank,
        "source": vllm.__file__,
        "torch": torch.__version__,
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
        "ple_environment": {
            k: v for k, v in os.environ.items() if "PLE" in k and k.startswith("VLLM_")
        },
    }


def run(args):
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    report = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cases": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation = json.loads((Path(args.model) / "generation_config.json").read_text())
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
    reference = {}
    if args.reference:
        previous = json.loads(args.reference.read_text())
        previous_args = previous["args"]
        for key in ("model", "kv_dtype", "kv_gib", "long_context"):
            if previous_args[key] != report["args"][key]:
                raise ValueError(f"Reference contract mismatch: {key}")
        reference = {case["name"]: case for case in previous["cases"]}

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
            gpu_memory_utilization=0.90,
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
            passed = out.finish_reason == "stop" and expected in answer
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
                case["matches_reference"] = (
                    control["prompt_hash"] == prompt_hash
                    and control["sampling"] == case["sampling"]
                    and control["token_ids"] == case["token_ids"]
                )
                passed = passed and case["matches_reference"]
            report["cases"].append(case)
            save()
            print(
                json.dumps(
                    {k: v for k, v in case.items() if k not in ("text", "token_ids")}
                ),
                flush=True,
            )
            if not passed:
                raise RuntimeError(f"Quality gate failed: {name}")

        for name, text, expected in (
            ("arithmetic", "请计算 19 × 23，在最后写出 RESULT=计算结果。", "437"),
            ("copy", "档案编号是 CEDAR-47|8261。请准确复述编号。", "CEDAR-47|8261"),
        ):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            check("official_" + name, ids, official, expected)
            check("greedy_" + name, ids, deterministic, expected)

        # Crosses scheduler and compressed-index pages, including chunked prefill.
        lengths = [8192, 32768, 261632] if args.long_context else [8192]
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
        for length in lengths:
            count = length - len(lead) - len(tail) - len(record)
            body = (filler * ((count + len(filler) - 1) // len(filler)))[:count]
            ids = lead + body[: count // 2] + record + body[count // 2 :] + tail
            check(f"greedy_retrieval_{length}", ids, deterministic, "MAPLE-8261")
        if args.long_context:
            # An allocation/finite-output gate, not complete-answer quality.
            boundary_ids = (filler * ((262143 + len(filler) - 1) // len(filler)))[
                :262143
            ]
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
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
