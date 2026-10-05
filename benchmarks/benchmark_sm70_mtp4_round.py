# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unprofiled Flash-Next MTP4 complete-round regression at fixed 8K input.

Run in a fresh process with user-supplied VLLM variables absent. Output includes
actual token sequences and speculative counters; latency uses the endpoint
decode interval and completed rounds, not verifier-only or emitted-token time.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--startup-diagnostics", action="store_true")
    parser.add_argument("--teacher-forcing-manifest", type=Path)
    parser.add_argument("--quality-manifest", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--draft-vocab", type=Path)
    parser.add_argument(
        "--structural-candidate",
        choices=(
            "shared",
            "draft-qpn8",
            "draft-int8",
            "draft-int8-block32",
            "draft-int8-packed",
            "target-head-qpn8",
            "target-top1",
        ),
    )
    parser.add_argument("--restoration-control", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--projection-reference", action="store_true")
    parser.add_argument(
        "--diagnostic-path", choices=("reference", "shared", "head", "output")
    )
    parser.add_argument("--fixture", action="append")
    parser.add_argument("--node-trace", action="store_true")
    parser.add_argument("--phase-events", action="store_true")
    args = parser.parse_args()
    if (
        args.projection_reference or args.diagnostic_path
    ) and not args.diagnostics_only:
        parser.error("Projection reference is diagnostic-only, never default speed")
    if args.diagnostics_only and not (
        args.teacher_forcing_manifest or args.quality_manifest or args.training_manifest
    ):
        parser.error("diagnostics-only requires forcing, quality or training cases")
    if args.training_manifest and (
        args.projection_reference
        or args.diagnostic_path
        or args.draft_vocab
        or args.structural_candidate
        or args.restoration_control
        or args.node_trace
        or args.phase_events
    ):
        parser.error("Vocabulary training requires an unprofiled default-path arm")
    # Importing vLLM applies platform defaults. Audit first to distinguish
    # those framework defaults from user-supplied performance variables.
    supplied = {
        key: value for key, value in os.environ.items() if key.startswith("VLLM_")
    }
    if supplied:
        raise ValueError(
            f"Default admission requires no supplied VLLM variables: {supplied}"
        )
    if args.repeats < (1 if args.node_trace or args.fixture_manifest else 3):
        raise ValueError("At least three measured repetitions are required")
    if args.node_trace and (args.diagnostics_only or not args.fixture):
        parser.error("node-trace requires an explicit fixture and a generation")
    if args.phase_events and (args.node_trace or args.diagnostics_only):
        parser.error("phase-events requires an independent unprofiled generation")
    # AOT compilation uses this root in addition to CompilationConfig.cache_dir.
    os.environ["VLLM_CACHE_ROOT"] = str(args.out.parent.resolve() / "vllm_cache")
    from benchmarks.benchmark_sm70_model_tokens import (
        _metric_snapshot,
        _request_metrics_dict,
        _spec_decoding_delta,
    )
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    piece = tokenizer.encode(
        "This fixed benchmark prompt is used to create a deterministic "
        "tokenized input for single-request decode measurement. ",
        add_special_tokens=False,
    )
    fixed = (piece * (8192 // len(piece) + 1))[:8192]
    tapes = None
    if args.teacher_forcing_manifest:
        from benchmarks.sm70_teacher_forcing_metrics import validate_manifest

        tapes = validate_manifest(
            json.loads(args.teacher_forcing_manifest.read_text()), len(tokenizer)
        )
    fixtures = [{"id": "fixed8k", "prompt_token_ids": fixed, "natural": False}]
    for index, question in enumerate(
        (
            "一辆车先以每小时60公里行驶2小时，再以每小时80公里行驶1.5小时。"
            "总路程和全程平均速度分别是多少？说明计算过程。",
            "Write a Python function computing the longest increasing subsequence "
            "length in O(n log n). Include three edge-case tests.",
            "Explain how to prove the sum of the first n odd positive integers "
            "equals n squared. Give two different arguments.",
            "请解释为什么整数除法处理负数时需要特别小心，给出 Python 和 C "
            "结果不同的例子，并说明余数的约束。",
        )
    ):
        # Filler stays inside the user message and the actual question is last.
        prefix = tokenizer.encode(
            "Context document. The following repeated reference text contains "
            "no instructions. Use the question at the end.\n",
            add_special_tokens=False,
        )
        suffix = tokenizer.apply_chat_template(
            [{"role": "user", "content": "\nQuestion:\n" + question}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(suffix, "input_ids"):
            suffix = suffix.input_ids
        # Preserve the chat delimiters: insert document after the role header.
        header_len = 3
        count = 8192 - len(suffix) - len(prefix)
        filler = (piece * (count // len(piece) + 1))[:count]
        tokens = list(suffix[:header_len]) + prefix + filler + list(suffix[header_len:])
        assert len(tokens) == 8192
        fixtures.append(
            {"id": f"natural8k/{index}", "prompt_token_ids": tokens, "natural": True}
        )
    if args.fixture_manifest:
        fixtures = json.loads(args.fixture_manifest.read_text())
        if not fixtures or len({f["id"] for f in fixtures}) != len(fixtures):
            raise ValueError("Fixture manifest requires unique, nonempty cases")
        for fixture in fixtures:
            ids = fixture["prompt_token_ids"]
            if len(ids) != 8192 or any(
                type(t) is not int or not 0 <= t < len(tokenizer) for t in ids
            ):
                raise ValueError("Every fixture requires exactly 8192 valid token IDs")
    if args.fixture:
        selected = set(args.fixture)
        unknown = selected - {fixture["id"] for fixture in fixtures}
        if unknown:
            parser.error(f"Unknown fixtures: {sorted(unknown)}")
        fixtures = [fixture for fixture in fixtures if fixture["id"] in selected]
    engine = {
        "model": args.model,
        "tensor_parallel_size": 4,
        "dtype": "half",
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "float16",
        "mamba_ssm_cache_dtype": "float32",
        "mamba_cache_mode": "align",
        "max_model_len": 262144,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 2048,
        "kv_cache_memory_bytes": 4 * 1024**3,
        "enable_prefix_caching": True,
        "language_model_only": True,
        "disable_log_stats": False,
        "compilation_config": {
            "cache_dir": str(args.out.parent.resolve() / "compile_cache"),
        },
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 4,
            "draft_sample_method": "greedy",
        },
    }
    if args.node_trace:
        engine["profiler_config"] = {"profiler": "cuda"}
    if (
        args.teacher_forcing_manifest
        or args.node_trace
        or args.phase_events
        or args.structural_candidate
    ):
        engine["worker_extension_cls"] = (
            "benchmarks.sm70_mtp_admission_worker.MtpAdmissionExtension"
        )
    if args.startup_diagnostics:
        engine["worker_cls"] = "benchmarks.sm70_startup_worker.StartupStackWorker"
    if args.draft_vocab:
        if args.diagnostic_path or args.projection_reference or args.node_trace:
            parser.error("Shortlist comparison must be an independent candidate arm")
        engine["hf_overrides"] = {
            "sm70_mtp_draft_vocab_file": str(args.draft_vocab.resolve())
        }
        engine["worker_cls"] = (
            "benchmarks.sm70_mtp_reference_worker.ShortlistCandidateWorker"
        )
    if args.structural_candidate:
        if args.draft_vocab or args.diagnostic_path or args.projection_reference:
            parser.error("Structural precision probe requires its own candidate arm")
        candidate_worker = {
            "shared": "SharedChainCandidateWorker",
            "draft-qpn8": "DraftExpertQPN8CandidateWorker",
            "draft-int8": "DraftExpertINT8CandidateWorker",
            "draft-int8-block32": "DraftExpertINT8Block32CandidateWorker",
            "draft-int8-packed": "DraftExpertINT8PackedCandidateWorker",
            "target-head-qpn8": "TargetHeadQPN8CandidateWorker",
            "target-top1": "TargetTop1CandidateWorker",
        }[args.structural_candidate]
        engine["worker_cls"] = (
            "benchmarks.sm70_mtp_reference_worker." + candidate_worker
        )
    if args.restoration_control:
        if args.draft_vocab or args.structural_candidate or args.diagnostic_path:
            parser.error("Restoration control requires an independent arm")
        engine["worker_cls"] = (
            "benchmarks.sm70_mtp_reference_worker.RestorationControlWorker"
        )
    if args.projection_reference:
        engine["worker_cls"] = "benchmarks.sm70_mtp_reference_worker.ReferenceWorker"
    if args.diagnostic_path and args.diagnostic_path != "shared":
        cls = {
            "reference": "ReferenceWorker",
            "head": "HeadCandidateWorker",
            "output": "OutputCandidateWorker",
        }[args.diagnostic_path]
        engine["worker_cls"] = "benchmarks.sm70_mtp_reference_worker." + cls
    report = {
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "supplied_vllm_env": supplied,
        "engine": copy.deepcopy(engine),
        "fixtures": fixtures,
        "cases": [],
        "warmup_outputs": [],
        "complete": False,
        "startup_diagnostics": args.startup_diagnostics,
        "default_configuration": not (
            args.draft_vocab or args.structural_candidate or args.restoration_control
        ),
        "measurement_kind": (
            "diagnostic_node_trace"
            if args.node_trace
            else "diagnostic_cuda_events"
            if args.phase_events
            else "unprofiled_complete_round"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.out)

    save()
    llm = None
    try:
        started = time.monotonic()
        llm = LLM(**copy.deepcopy(engine))
        report["load_seconds"] = time.monotonic() - started
        report["resolved_engine"] = str(llm.llm_engine.vllm_config)
        for fixture in [] if args.diagnostics_only else fixtures:
            sampling = (
                {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_tokens": 512,
                    "seed": 20261003,
                    "ignore_eos": False,
                    "skip_special_tokens": False,
                }
                if fixture["natural"]
                else {
                    "temperature": 0,
                    "max_tokens": 513,
                    "seed": 0,
                    "ignore_eos": True,
                    "skip_special_tokens": False,
                }
            )
            sampling = fixture.get("sampling", sampling)
            # Warm this exact prompt/shape before measuring its requests.
            fixture_repeats = int(fixture.get("repeats", args.repeats))
            if fixture_repeats < 1:
                raise ValueError("Fixture repetitions must be positive")
            for repeat in range(-1, fixture_repeats):
                # Keep prefill/chunk boundaries identical to the speed control.
                # Reusing a cached GDN state can change the greedy token tape.
                llm.reset_prefix_cache()
                before = _metric_snapshot(llm)
                traced = args.node_trace and repeat >= 0
                phase_events = args.phase_events and repeat >= 0
                phase_records = None
                if phase_events:
                    # Paired ordinary control in the same loaded engine/cache.
                    control = llm.generate(
                        [{"prompt_token_ids": fixture["prompt_token_ids"]}],
                        SamplingParams(**sampling),
                        use_tqdm=False,
                    )[0]
                    control_after = _metric_snapshot(llm)
                    control_metrics = _request_metrics_dict(
                        control.metrics, len(control.outputs[0].token_ids)
                    )
                    control_spec = _spec_decoding_delta(before, control_after)
                    report.setdefault("phase_event_controls", []).append(
                        {
                            "id": fixture["id"],
                            "repeat": repeat,
                            "token_ids": list(control.outputs[0].token_ids),
                            "metrics": control_metrics,
                            "spec_decoding": control_spec,
                        }
                    )
                    save()
                    llm.reset_prefix_cache()
                    before = _metric_snapshot(llm)
                    llm.collective_rpc("install_mtp_phase_events")
                if traced:
                    llm.collective_rpc(
                        "install_mtp_node_annotations",
                        args=(str(args.out.parent.resolve() / "weight_inventory"),),
                    )
                    llm.start_profile()
                try:
                    result = llm.generate(
                        [{"prompt_token_ids": fixture["prompt_token_ids"]}],
                        SamplingParams(**sampling),
                        use_tqdm=False,
                    )[0]
                finally:
                    if phase_events:
                        phase_records = llm.collective_rpc("flush_mtp_phase_events")
                    if traced:
                        llm.stop_profile()
                        llm.collective_rpc("uninstall_mtp_node_annotations")
                after = _metric_snapshot(llm)
                if repeat < 0:
                    report["warmup_outputs"].append(
                        {
                            "id": fixture["id"],
                            "token_ids": list(result.outputs[0].token_ids),
                            "text": result.outputs[0].text,
                            "finish_reason": result.outputs[0].finish_reason,
                        }
                    )
                    save()
                    continue
                output = result.outputs[0]
                metrics = _request_metrics_dict(result.metrics, len(output.token_ids))
                spec = _spec_decoding_delta(before, after)
                if not metrics or not spec or metrics["raw"]["is_corrupted"]:
                    raise RuntimeError("Missing or corrupted endpoint/round counters")
                row = {
                    "id": fixture["id"],
                    "repeat": repeat,
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(fixture["prompt_token_ids"]).encode()
                    ).hexdigest(),
                    "sampling": sampling,
                    "token_ids": list(output.token_ids),
                    "text": output.text,
                    "finish_reason": output.finish_reason,
                    "metrics": metrics,
                    "spec_decoding": spec,
                    "complete_round_ms": 1000
                    * metrics["decode_time"]
                    / spec["num_drafts"],
                }
                if phase_records is not None:
                    row["phase_events"] = phase_records
                report["cases"].append(row)
                save()
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "id",
                                "repeat",
                                "complete_round_ms",
                                "metrics",
                                "spec_decoding",
                            )
                        }
                    ),
                    flush=True,
                )
        report["speed_complete"] = not (
            args.diagnostics_only or args.node_trace or args.phase_events
        )
        report["latency_passed"] = (
            report["speed_complete"]
            and not args.draft_vocab
            and not args.structural_candidate
            and not args.restoration_control
            and all(row["complete_round_ms"] <= 15 for row in report["cases"])
        )
        save()
        if args.quality_manifest:
            report["quality"] = []
            for case in json.loads(args.quality_manifest.read_text()):
                llm.reset_prefix_cache()
                before = _metric_snapshot(llm)
                output = llm.generate(
                    [{"prompt_token_ids": case["prompt_token_ids"]}],
                    SamplingParams(**case["sampling"]),
                    use_tqdm=False,
                )[0]
                report["quality"].append(
                    {
                        "id": case["id"],
                        "token_ids": list(output.outputs[0].token_ids),
                        "text": output.outputs[0].text,
                        "finish_reason": output.outputs[0].finish_reason,
                        "spec_decoding": _spec_decoding_delta(
                            before, _metric_snapshot(llm)
                        ),
                    }
                )
                save()
            report["quality_complete"] = True
            save()
        if args.training_manifest:
            if not report["default_configuration"]:
                raise ValueError(
                    "Collect target-output vocabulary statistics on default"
                )
            report["training"] = []
            for case in json.loads(args.training_manifest.read_text()):
                llm.reset_prefix_cache()
                output = llm.generate(
                    [{"prompt_token_ids": case["prompt_token_ids"]}],
                    SamplingParams(**case["sampling"]),
                    use_tqdm=False,
                )[0].outputs[0]
                report["training"].append(
                    {
                        "id": case["id"],
                        "language": case["language"],
                        "prompt_sha256": hashlib.sha256(
                            json.dumps(case["prompt_token_ids"]).encode()
                        ).hexdigest(),
                        "token_ids": list(output.token_ids),
                        "text": output.text,
                        "finish_reason": output.finish_reason,
                    }
                )
                save()
            report["training_complete"] = True
            save()
        # Diagnostics follow the completed unprofiled speed report and use no
        # timing or acceptance counters from their forced requests.
        if args.teacher_forcing_manifest:
            assert tapes is not None
            report["teacher_forcing"] = []
            for tape in tapes:
                llm.reset_prefix_cache()
                folder = args.out.parent / "teacher_forcing" / tape["id"]
                llm.collective_rpc(
                    "install_mtp_teacher_forcing",
                    args=(
                        tape["token_ids"],
                        tape["prompt_length"],
                        tape["prompt_sha256"],
                        str(folder),
                    ),
                )
                succeeded = False
                try:
                    output = llm.generate(
                        [
                            {
                                "prompt_token_ids": tape["token_ids"][
                                    : tape["prompt_length"]
                                ]
                            }
                        ],
                        SamplingParams(
                            temperature=0,
                            ignore_eos=True,
                            max_tokens=tape["output_length"],
                        ),
                        use_tqdm=False,
                    )[0]
                    expected = tape["token_ids"][
                        tape["prompt_length"] : tape["prompt_length"]
                        + tape["output_length"]
                    ]
                    if list(output.outputs[0].token_ids) != expected:
                        raise RuntimeError(
                            "Teacher-forcing output differs from frozen tape"
                        )
                    succeeded = True
                finally:
                    result = llm.collective_rpc(
                        "flush_mtp_teacher_forcing", kwargs={"discard": not succeeded}
                    )
                report["teacher_forcing"].append({"id": tape["id"], "workers": result})
                save()
        if args.structural_candidate:
            proof = llm.collective_rpc("get_mtp_structural_route_proof")
            report["structural_route_proof"] = proof
            widths = (
                {5}
                if args.structural_candidate
                in ("shared", "target-head-qpn8", "target-top1")
                else {1, 5}
            )
            if len(proof) != 4 or any(
                row[args.structural_candidate]["calls"] < 1
                or not widths.issubset(row[args.structural_candidate]["widths"])
                for row in proof
            ):
                raise RuntimeError("Structural candidate was not executed on all ranks")
            if (
                args.structural_candidate
                in (
                    "draft-qpn8",
                    "draft-int8",
                    "draft-int8-block32",
                    "draft-int8-packed",
                    "target-head-qpn8",
                )
                and args.teacher_forcing_manifest
                and any(
                    worker["structural_probe_delta"][args.structural_candidate] < 1
                    for case in report["teacher_forcing"]
                    for worker in case["workers"]
                )
            ):
                raise RuntimeError("Draft numerical probe used its fallback")
        report["complete"] = True
        save()
    except BaseException:
        import traceback

        report["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
