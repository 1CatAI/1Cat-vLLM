# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired, fixed-subset dataset gate for the bounded Qwen DCP driver."""

import hashlib
import importlib.util
import json
import random
from decimal import Decimal, InvalidOperation
from pathlib import Path

import regex as re

# Mirror the existing GSM8K scoring convention without importing benchmark
# entrypoints: benchmark_sm70_decode changes sys.path at import time, which
# would make spawned workers import the source checkout instead of the wheel.
INVALID_ANSWER = -9_999_999
GSM8K_PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
_BOXED_RE = re.compile(r"\\boxed\{(?P<value>(?:[^{}]+|\{(?&value)\})*)\}")


def _answer_value(text: str) -> int:
    boxed = list(_BOXED_RE.finditer(text))
    answer_text = boxed[-1].group("value") if boxed else text
    numbers = _NUMBER_RE.findall(answer_text)
    if not numbers:
        return INVALID_ANSWER
    try:
        value = Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return INVALID_ANSWER
    if not value.is_finite() or value != value.to_integral_value():
        return INVALID_ANSWER
    return int(value)


def prepare_datasets(spec_path, tokenizer):
    spec = json.loads(spec_path.read_text())
    cases, sources = [], {}
    rng = random.Random(spec["selection_seed"])
    metrics = {}
    if "longbench" in spec:
        root = Path(spec["longbench"]["metrics_root"])
        metric_path = root / "metrics.py"
        module_spec = importlib.util.spec_from_file_location(
            "qwen38_longbench_metrics", metric_path
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        metrics = {
            "multifieldqa_en": module.qa_f1_score,
            "multifieldqa_zh": module.qa_f1_zh_score,
        }
        templates = json.loads((root / "config/dataset2prompt.json").read_text())
        limits = json.loads((root / "config/dataset2maxlen.json").read_text())
        for path in (
            metric_path,
            root / "config/dataset2prompt.json",
            root / "config/dataset2maxlen.json",
        ):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for metric in metrics.values():
            if metric("answer", "answer") != 1.0:
                raise ValueError("LongBench scorer identity check failed")

    def add_dataset(name, path, count, max_tokens):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not 0 < count <= len(rows):
            raise ValueError(f"Invalid sample count for {name}: {count}/{len(rows)}")
        for index in rng.sample(range(len(rows)), count):
            row = rows[index]
            thinking = name == "gsm8k"
            text = (
                row["question"] + GSM8K_PROMPT_SUFFIX
                if thinking
                else templates[name].format(**row)
            )
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            if (
                not isinstance(ids, list)
                or not ids
                or not all(isinstance(token, int) and token >= 0 for token in ids)
                or len(ids) + max_tokens > 262144
            ):
                raise ValueError(f"Invalid dataset prompt: {name}/{index}")
            cases.append(
                {
                    "id": f"{name}/{index}",
                    "dataset": name,
                    "index": index,
                    "prompt_token_ids": ids,
                    "prompt_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    "answers": [row["answer"]] if thinking else row["answers"],
                    "max_tokens": max_tokens,
                    "seed": spec["generation_seed"] + len(cases),
                }
            )

    if "gsm8k" in spec:
        item = spec["gsm8k"]
        add_dataset("gsm8k", Path(item["path"]), item["count"], item["max_tokens"])
    if "longbench" in spec:
        item = spec["longbench"]
        for name, count in item["datasets"].items():
            add_dataset(
                name, Path(item["data_dir"]) / f"{name}.jsonl", count, limits[name]
            )
    if not cases or not 1 <= spec["batch_size"] <= 4:
        raise ValueError("Dataset plan must be nonempty with batch_size in [1,4]")
    manifest = {
        "sources": sources,
        "selection_seed": spec["selection_seed"],
        "generation_seed": spec["generation_seed"],
        "batch_size": spec["batch_size"],
        "cases": [
            {k: v for k, v in case.items() if k not in ("prompt_token_ids", "answers")}
            | {"input_tokens": len(case["prompt_token_ids"])}
            for case in cases
        ],
    }
    return cases, metrics, manifest


def summarize_dataset(cases, reference=None):
    result = {}
    controls = {case["id"]: case for case in reference or []}
    for name in dict.fromkeys(case["dataset"] for case in cases):
        subset = [case for case in cases if case["dataset"] == name]
        summary = {
            "samples": len(subset),
            "score": 100 * sum(case["score"] for case in subset) / len(subset),
            "truncated": sum(case["finish_reason"] == "length" for case in subset),
            "empty": sum(not case["answer"].strip() for case in subset),
            "non_finite": sum(case["corrupted"] for case in subset),
        }
        if controls:
            baseline = [controls[case["id"]] for case in subset]
            before = 100 * sum(case["score"] for case in baseline) / len(subset)
            summary.update(
                reference_score=before,
                delta_pp=summary["score"] - before,
                wins=sum(c["score"] > b["score"] for c, b in zip(subset, baseline)),
                losses=sum(c["score"] < b["score"] for c, b in zip(subset, baseline)),
                observed_no_regression=(
                    summary["score"] >= before
                    and summary["truncated"]
                    <= sum(b["finish_reason"] == "length" for b in baseline)
                    and summary["empty"]
                    <= sum(not b["answer"].strip() for b in baseline)
                    and summary["non_finite"] == 0
                ),
            )
        result[name] = summary
    return result


def evaluate_datasets(
    llm, cases, metrics, manifest, generation, report, save, reference=None
):
    from vllm import SamplingParams

    report["dataset"] = {"manifest": manifest, "cases": [], "summary": {}}
    state = report["dataset"]
    batch_size = manifest["batch_size"]
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        sampling = [
            SamplingParams(
                max_tokens=c["max_tokens"],
                temperature=generation["temperature"],
                top_p=generation["top_p"],
                top_k=generation["top_k"],
                seed=c["seed"],
                ignore_eos=False,
            )
            for c in batch
        ]
        outputs = llm.generate(
            [{"prompt_token_ids": c["prompt_token_ids"]} for c in batch],
            sampling,
            use_tqdm=False,
        )
        for case, output in zip(batch, outputs, strict=True):
            candidate = output.outputs[0]
            answer = candidate.text.rsplit("</think>", 1)[-1].strip()
            if case["dataset"] == "gsm8k" and "</think>" not in candidate.text:
                answer = ""  # Do not count an unfinished reasoning number as an answer.
            corrupted = bool(output.metrics and output.metrics.is_corrupted)
            if case["dataset"] == "gsm8k":
                prediction = _answer_value(answer)
                score = float(
                    prediction != INVALID_ANSWER
                    and prediction == _answer_value(case["answers"][0])
                )
            else:
                score = max(
                    metrics[case["dataset"]](answer, gold) for gold in case["answers"]
                )
            state["cases"].append(
                {
                    "id": case["id"],
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "prompt_hash": case["prompt_hash"],
                    "input_tokens": len(case["prompt_token_ids"]),
                    "text": candidate.text,
                    "answer": answer,
                    "score": 0.0 if corrupted else score,
                    "token_ids": list(candidate.token_ids),
                    "finish_reason": candidate.finish_reason,
                    "corrupted": corrupted,
                }
            )
        state["summary"] = summarize_dataset(
            state["cases"], reference["cases"] if reference else None
        )
        save()
        print(
            json.dumps(
                {
                    "dataset_progress": len(state["cases"]),
                    "dataset_total": len(cases),
                    "summary": state["summary"],
                }
            ),
            flush=True,
        )
        if any(case["corrupted"] for case in state["cases"][-len(batch) :]):
            raise RuntimeError("Non-finite dataset output")
    state["complete"] = True
