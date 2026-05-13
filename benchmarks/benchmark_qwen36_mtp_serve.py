import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_COMPILATION_CONFIG = {
    "cudagraph_mode": "full_and_piecewise",
    "cudagraph_capture_sizes": [1, 2, 4, 8, 9, 18],
}


@dataclass(frozen=True)
class Scenario:
    name: str
    speculative_config: dict[str, Any] | None


def _default_python_executable() -> str:
    preferred = Path("/home/ymzx/miniconda3/envs/1cat-vllm-0.0.3/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("all speculative token counts must be >= 1")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.6-27B-AWQ")
    parser.add_argument("--python-executable", default=_default_python_executable())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8460)
    parser.add_argument("--gpu-ids", default="1,2,3,4")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--served-model-name", default="qwen36-27b-awq-mtp-bench")
    parser.add_argument("--attention-backend", default="FLASH_ATTN_V100")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--kv-cache-auto-trim-ratio", type=float, default=0.0)
    parser.add_argument("--swap-space", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4608)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--prompt-text-file", default=None)
    parser.add_argument("--num-speculative-tokens-list", type=_parse_int_list,
                        default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", default="bench_results/qwen36_mtp_serve")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument(
        "--compilation-config-json",
        default=json.dumps(DEFAULT_COMPILATION_CONFIG),
    )
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    return parser.parse_args()


def _scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios: list[Scenario] = []
    if not args.skip_baseline:
        scenarios.append(Scenario(name="baseline", speculative_config=None))
    scenarios.extend(
        Scenario(
            name=f"mtp_n{num_speculative_tokens}",
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": num_speculative_tokens,
            },
        )
        for num_speculative_tokens in args.num_speculative_tokens_list
    )
    return scenarios


def _base_env(args: argparse.Namespace, scenario_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    env["PATH"] = f"/usr/local/cuda-12.8/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-12.8/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    env["VLLM_SM70_ENABLE_LM_HEAD_FASTPATH"] = "1"
    env["VLLM_CACHE_ROOT"] = str(scenario_dir / "cache")
    return env


def _server_command(args: argparse.Namespace, scenario: Scenario,
                    port: int) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        f"{args.served_model_name}-{scenario.name}",
        "--trust-remote-code",
        "--quantization",
        args.quantization,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-auto-trim-ratio",
        str(args.kv_cache_auto_trim_ratio),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--skip-mm-profiling",
        "--mm-processor-cache-gb",
        "0",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--generation-config",
        "vllm",
        "--attention-backend",
        args.attention_backend,
        "--swap-space",
        str(args.swap_space),
        "--host",
        args.host,
        "--port",
        str(port),
        "--compilation-config",
        args.compilation_config_json,
    ]
    if args.disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    if scenario.speculative_config is not None:
        command.extend(
            ["--speculative-config", json.dumps(scenario.speculative_config)])
    return command


def _wait_for_server(host: str, port: int, timeout_s: int,
                     proc: subprocess.Popen[Any]) -> None:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with return code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for server at {url}")


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=30)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _fetch_spec_decode_metrics(base_url: str) -> tuple[int, int, int, dict[int, int]]:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return 0, 0, 0, {}

    drafts = 0
    draft_tokens = 0
    accepted_tokens = 0
    accepted_per_pos: dict[int, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split()
        if not parts:
            continue
        value = int(float(parts[-1]))
        if "num_drafts" in line:
            drafts += value
        elif "num_draft_tokens" in line:
            draft_tokens += value
        elif "num_accepted_tokens_per_pos" in line:
            pos_label = 'position="'
            if pos_label in line:
                start = line.index(pos_label) + len(pos_label)
                end = line.index('"', start)
                pos = int(line[start:end])
                accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + value
        elif "num_accepted_tokens" in line:
            accepted_tokens += value
    return drafts, draft_tokens, accepted_tokens, accepted_per_pos


def _format_per_position_rates(
    before: dict[int, int],
    after: dict[int, int],
    delta_drafts: int,
) -> list[float]:
    if delta_drafts <= 0:
        return []
    return [
        (after.get(pos, 0) - before.get(pos, 0)) / delta_drafts
        for pos in sorted(set(before) | set(after))
    ]


def _build_prompt_text(model: str, input_tokens: int, seed: int) -> tuple[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    all_special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    if vocab_size <= 0:
        raise RuntimeError("tokenizer.vocab_size is unavailable")
    allowed = [idx for idx in range(vocab_size) if idx not in all_special_ids]
    if not allowed:
        raise RuntimeError("no non-special tokenizer ids found")
    offset = seed % len(allowed)
    token_ids = [
        allowed[(offset + idx) % len(allowed)] for idx in range(input_tokens)
    ]
    text = ""
    encoded: list[int] = []
    for _ in range(20):
        text = tokenizer.decode(token_ids)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == input_tokens:
            return text, len(encoded)
        if len(encoded) > input_tokens:
            token_ids = encoded[:input_tokens]
            continue
        deficit = input_tokens - len(encoded)
        start = (offset + len(encoded)) % len(allowed)
        token_ids = encoded + [
            allowed[(start + idx) % len(allowed)] for idx in range(deficit)
        ]
    return text, len(encoded)


def _stream_completion(
    *,
    base_url: str,
    model_name: str,
    prompt_text: str,
    prompt_token_len: int,
    output_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "temperature": 0.0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "add_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_s = time.perf_counter()
    first_token_s: float | None = None
    last_token_s: float | None = None
    chunks = 0
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            now_s = time.perf_counter()
            if obj.get("usage") is not None:
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                piece = choice.get("text") or ""
                if piece:
                    chunks += 1
                    if first_token_s is None:
                        first_token_s = now_s
                    last_token_s = now_s
                    text_parts.append(piece)
    end_s = time.perf_counter()

    prompt_tokens = (
        prompt_token_len
        if usage is None
        else int(usage.get("prompt_tokens") or prompt_token_len)
    )
    completion_tokens = (
        output_tokens
        if usage is None
        else int(usage.get("completion_tokens") or output_tokens)
    )
    wall_s = end_s - start_s
    if first_token_s is None:
        ttft_ms = None
        decode_throughput = None
    else:
        ttft_ms = (first_token_s - start_s) * 1000.0
        if last_token_s is not None and completion_tokens > 1 and last_token_s > first_token_s:
            decode_throughput = (completion_tokens - 1) / (last_token_s - first_token_s)
        else:
            decode_throughput = None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_s": wall_s,
        "mean_ttft_ms": ttft_ms,
        "prefill_tps": (
            prompt_tokens / (ttft_ms / 1000.0)
            if ttft_ms is not None and ttft_ms > 0
            else None
        ),
        "decode_throughput": decode_throughput,
        "output_throughput": (
            completion_tokens / wall_s if wall_s > 0 else None
        ),
        "chunks": chunks,
        "text_head": "".join(text_parts)[:200],
    }


def _run_direct_benchmark(
    *,
    args: argparse.Namespace,
    scenario: Scenario,
    port: int,
    repeat_idx: int,
    prompt_text: str,
    prompt_token_len: int,
) -> dict[str, Any]:
    base_url = f"http://{args.host}:{port}"
    served_model_name = f"{args.served_model_name}-{scenario.name}"
    for _ in range(args.num_warmups):
        _stream_completion(
            base_url=base_url,
            model_name=served_model_name,
            prompt_text=prompt_text,
            prompt_token_len=prompt_token_len,
            output_tokens=args.output_tokens,
        )

    before = _fetch_spec_decode_metrics(base_url)
    result = _stream_completion(
        base_url=base_url,
        model_name=served_model_name,
        prompt_text=prompt_text,
        prompt_token_len=prompt_token_len,
        output_tokens=args.output_tokens,
    )
    after = _fetch_spec_decode_metrics(base_url)

    delta_drafts = after[0] - before[0]
    delta_draft_tokens = after[1] - before[1]
    delta_accepted_tokens = after[2] - before[2]
    if delta_draft_tokens > 0:
        result["spec_decode_acceptance_rate"] = (
            delta_accepted_tokens / delta_draft_tokens * 100.0
        )
        result["spec_decode_acceptance_length"] = (
            1.0
            + (delta_accepted_tokens / delta_drafts if delta_drafts > 0 else 0.0)
        )
        result["spec_decode_num_drafts"] = delta_drafts
        result["spec_decode_draft_tokens"] = delta_draft_tokens
        result["spec_decode_accepted_tokens"] = delta_accepted_tokens
        result["spec_decode_per_position_acceptance_rates"] = (
            _format_per_position_rates(before[3], after[3], delta_drafts)
        )
    result["scenario"] = scenario.name
    result["repeat_idx"] = repeat_idx
    return result


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        return []
    return [sum(vector[idx] for vector in vectors) / len(vectors)
            for idx in range(width)]


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output_throughputs = [
        float(run["output_throughput"])
        for run in runs
        if run.get("output_throughput") is not None
    ]
    ttfts = [
        float(run["mean_ttft_ms"])
        for run in runs
        if run.get("mean_ttft_ms") is not None
    ]
    decode_throughputs = [
        float(run["decode_throughput"])
        for run in runs
        if run.get("decode_throughput") is not None
    ]
    acceptance_rates = [
        float(run["spec_decode_acceptance_rate"])
        for run in runs
        if run.get("spec_decode_acceptance_rate") is not None
    ]
    acceptance_lengths = [
        float(run["spec_decode_acceptance_length"])
        for run in runs
        if run.get("spec_decode_acceptance_length") is not None
    ]
    per_position_vectors = [
        [float(value) for value in run["spec_decode_per_position_acceptance_rates"]]
        for run in runs
        if run.get("spec_decode_per_position_acceptance_rates")
    ]
    return {
        "runs": runs,
        "median_output_throughput": _median(output_throughputs),
        "median_decode_throughput": _median(decode_throughputs),
        "median_mean_ttft_ms": _median(ttfts),
        "median_spec_decode_acceptance_rate": _median(acceptance_rates),
        "median_spec_decode_acceptance_length": _median(acceptance_lengths),
        "mean_spec_decode_per_position_acceptance_rates": _mean_vector(
            per_position_vectors),
    }


def main() -> None:
    args = _parse_args()
    root_dir = Path(args.result_dir) / time.strftime("%Y%m%d-%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)
    if args.prompt_text_file is not None:
        prompt_path = Path(args.prompt_text_file)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
        )
        prompt_token_len = len(tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
        ))
    else:
        prompt_text, prompt_token_len = _build_prompt_text(
            args.model, args.input_tokens, args.seed)

    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario_index, scenario in enumerate(_scenarios(args)):
        port = args.base_port + scenario_index
        scenario_dir = root_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        env = _base_env(args, scenario_dir)
        server_log_path = scenario_dir / "server.log"
        server_command = _server_command(args, scenario, port)

        with server_log_path.open("w") as server_log:
            server_log.write(" ".join(server_command) + "\n")
            server_log.flush()
            proc = subprocess.Popen(
                server_command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                _wait_for_server(args.host, port, args.startup_timeout, proc)
                runs: list[dict[str, Any]] = []
                for repeat_idx in range(args.repeats):
                    run_output_path = scenario_dir / f"run_{repeat_idx}.json"
                    run_result = _run_direct_benchmark(
                        args=args,
                        scenario=scenario,
                        port=port,
                        repeat_idx=repeat_idx,
                        prompt_text=prompt_text,
                        prompt_token_len=prompt_token_len,
                    )
                    run_result["server_command"] = server_command
                    run_result["server_log_path"] = str(server_log_path)
                    with run_output_path.open("w") as f:
                        json.dump(run_result, f, indent=2)
                    runs.append(run_result)
            finally:
                _terminate_process(proc)

        scenario_summaries[scenario.name] = _summarize_runs(runs)
        with (scenario_dir / "summary.json").open("w") as f:
            json.dump(scenario_summaries[scenario.name], f, indent=2)

    summary = {
        "args": vars(args),
        "scenarios": scenario_summaries,
    }
    with (root_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
