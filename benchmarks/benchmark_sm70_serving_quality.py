# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving-path quality dump/compare harness for SM70 migration.

This script intentionally talks to the OpenAI-compatible HTTP server instead
of constructing an offline LLM. It is used to validate the exact production
path under CUDA graph capture, including backend selection, scheduler, sampler,
and response serialization.
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark_sm70_model_tokens import (
    _parse_extra_engine_args,
    _sm70_attention_policy,
    _sm70_comm_policy,
    _sm70_gdn_fla_policy,
    _sm70_graph_policy,
    _sm70_moe_policy,
    _sm70_sampling_policy,
    _sm70_tune_policy,
    _sm70_turbomind_policy,
    _tracked_env,
)

DEFAULT_PROMPTS = [
    "Write one sentence about deterministic validation.",
]


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt)
    if args.prompts_json is not None:
        loaded = json.loads(args.prompts_json.read_text())
        if not isinstance(loaded, list) or not all(isinstance(p, str) for p in loaded):
            raise TypeError("--prompts-json must contain a JSON list of strings")
        prompts.extend(loaded)
    return prompts or DEFAULT_PROMPTS


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _extract_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"Expected exactly one choice, got {len(choices)}")
    choice = choices[0]
    return {
        "text": choice.get("text"),
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "token_ids": choice.get("token_ids"),
        "prompt_token_ids": choice.get("prompt_token_ids"),
        "logprobs": choice.get("logprobs"),
        "prompt_logprobs": choice.get("prompt_logprobs"),
    }


def _dump(args: argparse.Namespace) -> int:
    if args.out is None:
        raise ValueError("--out is required in dump mode")

    prompts = _load_prompts(args)
    endpoint = args.base_url.rstrip("/") + "/v1/completions"
    server_kwargs = _parse_extra_engine_args(args.server_arg)
    records = []
    start = time.perf_counter()
    for index, prompt in enumerate(prompts):
        payload: dict[str, Any] = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stream": False,
            "logprobs": args.logprobs,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "prompt_logprobs": args.prompt_logprobs,
        }
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        if args.ignore_eos:
            payload["ignore_eos"] = True
        response = _post_json(endpoint, payload, args.timeout)
        records.append({
            "index": index,
            "prompt": prompt,
            "request": payload,
            "response": response,
            "choice": _extract_choice(response),
        })
    elapsed = time.perf_counter() - start

    payload = {
        "base_url": args.base_url,
        "model": args.model,
        "elapsed_seconds": elapsed,
        "request_count": len(records),
        "env": _tracked_env(),
        "sm70_tune_policy": _sm70_tune_policy(),
        "sm70_turbomind_policy": _sm70_turbomind_policy(),
        "sm70_attention_policy": _sm70_attention_policy(
            server_kwargs.get("kv_cache_dtype", "auto")
        ),
        "sm70_graph_policy": _sm70_graph_policy(),
        "sm70_comm_policy": _sm70_comm_policy(server_kwargs),
        "sm70_gdn_fla_policy": _sm70_gdn_fla_policy(),
        "sm70_moe_policy": _sm70_moe_policy(),
        "sm70_sampling_policy": _sm70_sampling_policy(),
        "server_kwargs": server_kwargs,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "ignore_eos": args.ignore_eos,
            "logprobs": args.logprobs,
            "prompt_logprobs": args.prompt_logprobs,
        },
        "records": records,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"},
                     indent=2,
                     sort_keys=True))
    return 0


def _logprob_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        raw = value.get("logprob")
        if raw is not None:
            return float(raw)
    return None


def _step_logprob_for_token(step: Any, token_id: int) -> float | None:
    if not isinstance(step, dict):
        return None
    for key in (str(token_id), f"token_id:{token_id}"):
        if key in step:
            return _logprob_value(step[key])
    return None


def _prompt_logprob_values(choice: dict[str, Any]) -> list[float]:
    prompt_token_ids = choice.get("prompt_token_ids") or []
    prompt_logprobs = choice.get("prompt_logprobs") or []
    values: list[float] = []
    for index, step in enumerate(prompt_logprobs):
        if index == 0 or index >= len(prompt_token_ids):
            continue
        value = _step_logprob_for_token(step, int(prompt_token_ids[index]))
        if value is not None:
            values.append(value)
    return values


def _prompt_perplexity(choice: dict[str, Any]) -> dict[str, Any]:
    values = _prompt_logprob_values(choice)
    if not values:
        return {
            "token_count": 0,
            "avg_nll": None,
            "perplexity": None,
        }
    avg_nll = -sum(values) / len(values)
    return {
        "token_count": len(values),
        "avg_nll": avg_nll,
        "perplexity": math.exp(avg_nll),
    }


def _token_logprobs(choice: dict[str, Any]) -> list[float | None]:
    logprobs = choice.get("logprobs") or {}
    values = logprobs.get("token_logprobs") or []
    return [None if value is None else float(value) for value in values]


def _normal_logprob_key(key: Any) -> str:
    text = str(key)
    if text.startswith("token_id:"):
        return text[len("token_id:"):]
    return text


def _top_logprob_maps(steps: Any) -> list[dict[str, float]]:
    if not isinstance(steps, list):
        return []
    maps: list[dict[str, float]] = []
    for step in steps:
        if not isinstance(step, dict):
            maps.append({})
            continue
        values: dict[str, float] = {}
        for key, value in step.items():
            logprob = _logprob_value(value)
            if logprob is not None:
                values[_normal_logprob_key(key)] = logprob
        maps.append(values)
    return maps


def _output_top_logprob_maps(choice: dict[str, Any]) -> list[dict[str, float]]:
    logprobs = choice.get("logprobs") or {}
    return _top_logprob_maps(logprobs.get("top_logprobs") or [])


def _diff_top_logprob_maps(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
) -> dict[str, Any]:
    diffs: list[float] = []
    missing_left = 0
    missing_right = 0
    same_step_count = len(left) == len(right)
    for left_step, right_step in zip(left, right, strict=False):
        left_keys = set(left_step)
        right_keys = set(right_step)
        missing_left += len(right_keys - left_keys)
        missing_right += len(left_keys - right_keys)
        for key in left_keys & right_keys:
            diffs.append(abs(left_step[key] - right_step[key]))
    return {
        "same_step_count": same_step_count,
        "step_count": min(len(left), len(right)),
        "common_value_count": len(diffs),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
    }


def _diff_values(left: list[float | None],
                 right: list[float | None]) -> dict[str, Any]:
    diffs: list[float] = []
    same_count = len(left) == len(right)
    for left_value, right_value in zip(left, right, strict=False):
        if left_value is None or right_value is None:
            same_count = same_count and left_value is right_value
            continue
        diffs.append(abs(left_value - right_value))
    return {
        "same_count": same_count,
        "count": len(diffs),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
    }


def _first_mismatch(left: list[int] | None,
                    right: list[int] | None) -> dict[str, Any] | None:
    if left is None or right is None:
        return None if left == right else {"index": 0, "left": left, "right": right}
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if left_id != right_id:
            return {"index": index, "left": left_id, "right": right_id}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "left": None if index >= len(left) else left[index],
            "right": None if index >= len(right) else right[index],
        }
    return None


def _load_logits_dumps(path: Path) -> list[dict[str, Any]]:
    import torch

    entries: list[dict[str, Any]] = []
    for file_path in sorted(path.glob("sampler_logits_*.pt")):
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        logits = payload["logits"].float()
        entries.append({
            "path": str(file_path),
            "step": int(payload["step"]),
            "pid": payload.get("pid"),
            "stage": payload.get("stage"),
            "shape": list(payload.get("shape", tuple(logits.shape))),
            "dtype": payload.get("dtype"),
            "all_greedy": bool(payload.get("all_greedy")),
            "output_token_ids": payload.get("output_token_ids"),
            "logits": logits,
        })
    entries.sort(key=lambda item: (
        int(item["step"]),
        str(item.get("stage")),
        str(item["path"]),
    ))
    return entries


def _compare_logits_dirs(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    import torch

    left_entries = _load_logits_dumps(left_dir)
    right_entries = _load_logits_dumps(right_dir)
    comparisons: list[dict[str, Any]] = []
    global_max = 0.0
    total_sum = 0.0
    total_count = 0
    same_dump_count = len(left_entries) == len(right_entries)
    all_shape_equal = same_dump_count
    all_argmax_equal = same_dump_count
    num_argmax_mismatch = 0
    first_argmax_mismatch = None
    for index, (left, right) in enumerate(
            zip(left_entries, right_entries, strict=False)):
        shape_equal = tuple(left["logits"].shape) == tuple(right["logits"].shape)
        all_shape_equal = all_shape_equal and shape_equal
        payload = {
            "index": index,
            "left_path": left["path"],
            "right_path": right["path"],
            "left_step": left["step"],
            "right_step": right["step"],
            "left_stage": left["stage"],
            "right_stage": right["stage"],
            "left_pid": left["pid"],
            "right_pid": right["pid"],
            "shape_equal": shape_equal,
            "left_shape": left["shape"],
            "right_shape": right["shape"],
            "left_dtype": left["dtype"],
            "right_dtype": right["dtype"],
        }
        if shape_equal:
            diff = (left["logits"] - right["logits"]).abs()
            max_diff = float(diff.max().item()) if diff.numel() else 0.0
            mean_diff = float(diff.mean().item()) if diff.numel() else 0.0
            left_argmax = left["logits"].argmax(dim=-1)
            right_argmax = right["logits"].argmax(dim=-1)
            argmax_equal = bool(torch.equal(left_argmax, right_argmax))
            all_argmax_equal = all_argmax_equal and argmax_equal
            if not argmax_equal:
                num_argmax_mismatch += 1
                if first_argmax_mismatch is None:
                    first_argmax_mismatch = index
            global_max = max(global_max, max_diff)
            total_sum += float(diff.sum().item())
            total_count += int(diff.numel())
            payload.update({
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "argmax_equal": argmax_equal,
                "argmax_first_mismatch": _first_mismatch(
                    left_argmax.reshape(-1).tolist(),
                    right_argmax.reshape(-1).tolist(),
                ),
            })
        else:
            all_argmax_equal = False
        comparisons.append(payload)

    return {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "same_dump_count": same_dump_count,
        "left_dump_count": len(left_entries),
        "right_dump_count": len(right_entries),
        "all_shape_equal": all_shape_equal,
        "all_argmax_equal": all_argmax_equal,
        "num_argmax_mismatch": num_argmax_mismatch,
        "first_argmax_mismatch": first_argmax_mismatch,
        "max_abs_diff": global_max,
        "mean_abs_diff": total_sum / total_count if total_count else None,
        "comparisons": comparisons,
    }


def _compare(args: argparse.Namespace) -> int:
    if args.json_out is None:
        raise ValueError("--json-out is required in compare mode")
    left = json.loads(args.compare[0].read_text())
    right = json.loads(args.compare[1].read_text())

    same_request_count = len(left["records"]) == len(right["records"])
    token_equal = same_request_count
    pairs = []
    prompt_logprob_diffs: list[float] = []
    output_logprob_diffs: list[float] = []
    prompt_top_logprob_diffs: list[float] = []
    output_top_logprob_diffs: list[float] = []
    prompt_perplexity_diffs: list[float] = []
    for index, (left_record, right_record) in enumerate(
            zip(left["records"], right["records"], strict=False)):
        left_choice = left_record["choice"]
        right_choice = right_record["choice"]
        prompt_equal = (
            left_choice.get("prompt_token_ids") == right_choice.get("prompt_token_ids")
        )
        output_equal = left_choice.get("token_ids") == right_choice.get("token_ids")
        text_equal = left_choice.get("text") == right_choice.get("text")
        token_equal = token_equal and prompt_equal and output_equal and text_equal

        prompt_diff = _diff_values(
            _prompt_logprob_values(left_choice),
            _prompt_logprob_values(right_choice),
        )
        output_diff = _diff_values(
            _token_logprobs(left_choice),
            _token_logprobs(right_choice),
        )
        prompt_top_diff = _diff_top_logprob_maps(
            _top_logprob_maps(left_choice.get("prompt_logprobs") or []),
            _top_logprob_maps(right_choice.get("prompt_logprobs") or []),
        )
        output_top_diff = _diff_top_logprob_maps(
            _output_top_logprob_maps(left_choice),
            _output_top_logprob_maps(right_choice),
        )
        if prompt_diff["max_abs_diff"] is not None:
            prompt_logprob_diffs.append(float(prompt_diff["max_abs_diff"]))
        if output_diff["max_abs_diff"] is not None:
            output_logprob_diffs.append(float(output_diff["max_abs_diff"]))
        if prompt_top_diff["max_abs_diff"] is not None:
            prompt_top_logprob_diffs.append(float(prompt_top_diff["max_abs_diff"]))
        if output_top_diff["max_abs_diff"] is not None:
            output_top_logprob_diffs.append(float(output_top_diff["max_abs_diff"]))
        left_ppl = _prompt_perplexity(left_choice)
        right_ppl = _prompt_perplexity(right_choice)
        left_ppl_value = left_ppl["perplexity"]
        right_ppl_value = right_ppl["perplexity"]
        prompt_perplexity_abs_diff = (
            None if left_ppl_value is None or right_ppl_value is None else
            abs(float(left_ppl_value) - float(right_ppl_value))
        )
        if prompt_perplexity_abs_diff is not None:
            prompt_perplexity_diffs.append(prompt_perplexity_abs_diff)
        pairs.append({
            "index": index,
            "prompt_equal": prompt_equal,
            "output_equal": output_equal,
            "text_equal": text_equal,
            "prompt_first_mismatch": _first_mismatch(
                left_choice.get("prompt_token_ids"),
                right_choice.get("prompt_token_ids"),
            ),
            "output_first_mismatch": _first_mismatch(
                left_choice.get("token_ids"),
                right_choice.get("token_ids"),
            ),
            "prompt_logprob_diff": prompt_diff,
            "output_logprob_diff": output_diff,
            "prompt_top_logprob_diff": prompt_top_diff,
            "output_top_logprob_diff": output_top_diff,
            "left_prompt_perplexity": left_ppl,
            "right_prompt_perplexity": right_ppl,
            "prompt_perplexity_abs_diff": prompt_perplexity_abs_diff,
        })

    sampler_logits_diff = (
        _compare_logits_dirs(args.left_logits_dir, args.right_logits_dir)
        if args.left_logits_dir is not None and args.right_logits_dir is not None
        else None
    )
    result = {
        "left": str(args.compare[0]),
        "right": str(args.compare[1]),
        "same_request_count": same_request_count,
        "token_equal": token_equal,
        "max_prompt_logprob_diff": (
            max(prompt_logprob_diffs) if prompt_logprob_diffs else None
        ),
        "max_output_logprob_diff": (
            max(output_logprob_diffs) if output_logprob_diffs else None
        ),
        "max_prompt_top_logprob_diff": (
            max(prompt_top_logprob_diffs) if prompt_top_logprob_diffs else None
        ),
        "max_output_top_logprob_diff": (
            max(output_top_logprob_diffs) if output_top_logprob_diffs else None
        ),
        "max_prompt_perplexity_abs_diff": (
            max(prompt_perplexity_diffs) if prompt_perplexity_diffs else None
        ),
        "sampler_logits_diff": sampler_logits_diff,
        "pairs": pairs,
        "left_meta": {k: v for k, v in left.items() if k != "records"},
        "right_meta": {k: v for k, v in right.items() if k != "records"},
    }
    result["model_quality_gate"] = _model_quality_gate(
        token_equal=token_equal,
        result=result,
        args=args,
    )
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_model_quality_gate:
        return 0 if result["model_quality_gate"]["label"] == "model-pass" else 1
    if not token_equal:
        return 1
    if args.fail_on_logprob_diff:
        prompt_bad = (result["max_prompt_logprob_diff"] or 0.0) != 0.0
        output_bad = (result["max_output_logprob_diff"] or 0.0) != 0.0
        return 1 if prompt_bad or output_bad else 0
    return 0


def _within_bound(value: float | None, bound: float | None) -> bool | None:
    if value is None or bound is None:
        return None
    return value <= bound


def _model_quality_gate(
    *,
    token_equal: bool,
    result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sampler_logits_diff = result.get("sampler_logits_diff")
    checks = {
        "token_equal": token_equal,
        "prompt_logprob_max_abs_diff": result.get("max_prompt_logprob_diff"),
        "output_logprob_max_abs_diff": result.get("max_output_logprob_diff"),
        "prompt_top_logprob_max_abs_diff": result.get(
            "max_prompt_top_logprob_diff"
        ),
        "output_top_logprob_max_abs_diff": result.get(
            "max_output_top_logprob_diff"
        ),
        "prompt_perplexity_max_abs_diff": result.get(
            "max_prompt_perplexity_abs_diff"
        ),
        "sampler_logits_max_abs_diff": None
        if sampler_logits_diff is None
        else sampler_logits_diff.get("max_abs_diff"),
        "sampler_logits_all_argmax_equal": None
        if sampler_logits_diff is None
        else sampler_logits_diff.get("all_argmax_equal"),
    }
    bounds = {
        "prompt_logprob_max_abs_diff": args.max_prompt_logprob_diff_for_accept,
        "output_logprob_max_abs_diff": args.max_output_logprob_diff_for_accept,
        "prompt_top_logprob_max_abs_diff": (
            args.max_prompt_top_logprob_diff_for_accept
        ),
        "output_top_logprob_max_abs_diff": (
            args.max_output_top_logprob_diff_for_accept
        ),
        "prompt_perplexity_max_abs_diff": (
            args.max_prompt_perplexity_abs_diff_for_accept
        ),
        "sampler_logits_max_abs_diff": args.max_sampler_logits_diff_for_accept,
    }
    failed = []
    pending = []
    warning = []
    if not token_equal:
        message = "deterministic token IDs or output text differ"
        if args.allow_token_diff_for_model_quality_gate:
            warning.append(message)
        else:
            failed.append(message)
    sampler_argmax_equal = checks["sampler_logits_all_argmax_equal"]
    if sampler_argmax_equal is False:
        message = "sampler logits argmax differs"
        if args.allow_sampler_argmax_diff_for_model_quality_gate:
            warning.append(message)
        else:
            failed.append(message)
    elif sampler_argmax_equal is None:
        message = "sampler logits argmax evidence missing"
        if args.allow_missing_sampler_logits_for_model_quality_gate:
            warning.append(message)
        else:
            pending.append(message)
    for name, bound in bounds.items():
        value = checks[name]
        if (
            name == "sampler_logits_max_abs_diff"
            and value is None
            and args.allow_missing_sampler_logits_for_model_quality_gate
        ):
            warning.append(f"{name} evidence missing")
            continue
        within = _within_bound(value, bound)
        if within is False:
            failed.append(f"{name}={value} exceeds bound {bound}")
        elif value is not None and bound is None:
            pending.append(f"{name}={value} has no configured acceptance bound")
        elif value is None:
            pending.append(f"{name} evidence missing")

    if failed:
        label = "model-fail"
        default_acceptance = "not default-accepted"
    elif pending:
        label = "B-pending"
        default_acceptance = "not default-accepted"
    else:
        label = "model-pass"
        default_acceptance = "model-level gate passed"
    return {
        "label": label,
        "default_acceptance": default_acceptance,
        "checks": checks,
        "bounds": bounds,
        "failed_evidence": failed,
        "pending_evidence": pending,
        "warning_evidence": warning,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--left-logits-dir", type=Path)
    parser.add_argument("--right-logits-dir", type=Path)
    parser.add_argument(
        "--require-model-quality-gate",
        action="store_true",
        help=(
            "Fail compare mode unless model_quality_gate.label is model-pass. "
            "Use for default-enable acceptance, not exploratory token checks."
        ),
    )
    parser.add_argument(
        "--allow-token-diff-for-model-quality-gate",
        action="store_true",
        help=(
            "Record deterministic token/text differences as warning evidence "
            "instead of failing the model quality gate. Use only with explicit "
            "numeric or dataset-level acceptance bounds."
        ),
    )
    parser.add_argument(
        "--allow-sampler-argmax-diff-for-model-quality-gate",
        action="store_true",
        help=(
            "Record sampler-logits argmax differences as warning evidence "
            "instead of failing the model quality gate. Use only when low-margin "
            "token flips are covered by separate numeric/dataset gates."
        ),
    )
    parser.add_argument(
        "--allow-missing-sampler-logits-for-model-quality-gate",
        action="store_true",
        help=(
            "Record missing sampler-logits dumps as warning evidence instead "
            "of keeping the model quality gate pending. Use for dataset or "
            "serving quality gates where logits dumps are intentionally absent."
        ),
    )
    parser.add_argument("--max-prompt-logprob-diff-for-accept", type=float)
    parser.add_argument("--max-output-logprob-diff-for-accept", type=float)
    parser.add_argument("--max-prompt-top-logprob-diff-for-accept", type=float)
    parser.add_argument("--max-output-top-logprob-diff-for-accept", type=float)
    parser.add_argument("--max-prompt-perplexity-abs-diff-for-accept", type=float)
    parser.add_argument("--max-sampler-logits-diff-for-accept", type=float)
    parser.add_argument("--fail-on-logprob-diff", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--logprobs", type=int, default=1)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help=(
            "Record a server-side config value as KEY=VALUE for policy "
            "metadata, e.g. disable_custom_all_reduce=true. This does not "
            "modify the HTTP request."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.compare is not None:
        return _compare(args)
    if args.base_url is None or args.model is None:
        raise ValueError("--base-url and --model are required in dump mode")
    return _dump(args)


if __name__ == "__main__":
    raise SystemExit(main())
