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


DEFAULT_COMPILATION_CONFIG = {
    "cudagraph_mode": "full_and_piecewise",
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
}


@dataclass(frozen=True)
class Scenario:
    name: str
    disable_custom_all_reduce: bool
    speculative_config: dict[str, Any] | None


def _default_python_executable() -> str:
    preferred = Path("/opt/venv/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/test/下载/model/Qwen3.5-35B-A3B-AWQ",
    )
    parser.add_argument(
        "--draft-model",
        default="/home/test/下载/model/Qwen3.5-35B-A3B-DFlash",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "baseline",
            "baseline_no_custom_all_reduce",
            "dflash",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--python-executable", default=_default_python_executable())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--served-model-name", default="qwen35-dflash-bench")
    parser.add_argument("--attention-backend", default="FLASH_ATTN_V100")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4416)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4416)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", default="bench_results/dflash_serve")
    parser.add_argument(
        "--compilation-config-json",
        default=json.dumps(DEFAULT_COMPILATION_CONFIG),
    )
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--dflash-profile", action="store_true")
    parser.add_argument("--dflash-profile-log-interval", type=int, default=32)
    return parser.parse_args()


def _scenarios(args: argparse.Namespace) -> list[Scenario]:
    base = Scenario(
        name="baseline",
        disable_custom_all_reduce=False,
        speculative_config=None,
    )
    base_no_car = Scenario(
        name="baseline_no_custom_all_reduce",
        disable_custom_all_reduce=True,
        speculative_config=None,
    )
    dflash = Scenario(
        name="dflash",
        disable_custom_all_reduce=False,
        speculative_config={
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
    )
    mapping = {
        base.name: base,
        base_no_car.name: base_no_car,
        dflash.name: dflash,
    }
    if args.scenario == "all":
        return [base, base_no_car, dflash]
    return [mapping[args.scenario]]


def _base_env(
    args: argparse.Namespace,
    scenario_dir: Path,
    scenario: Scenario,
) -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    env["PATH"] = f"/usr/local/cuda-12.8/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-12.8/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    env["VLLM_CACHE_ROOT"] = str(scenario_dir / "cache")
    if scenario.name == "dflash" and args.dflash_profile:
        env["VLLM_DFLASH_PROFILE"] = "1"
        env["VLLM_DFLASH_PROFILE_LOG_INTERVAL"] = str(
            args.dflash_profile_log_interval
        )
    return env


def _server_command(
    args: argparse.Namespace,
    scenario: Scenario,
    port: int,
) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        f"{args.served_model_name}-{scenario.name}",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--quantization",
        args.quantization,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--skip-mm-profiling",
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
        "--trust-remote-code",
    ]
    if scenario.disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    if scenario.speculative_config is not None:
        command.extend(
            ["--speculative-config", json.dumps(scenario.speculative_config)]
        )
    return command


def _bench_command(
    args: argparse.Namespace,
    scenario: Scenario,
    port: int,
    output_path: Path,
) -> list[str]:
    served_model_name = f"{args.served_model_name}-{scenario.name}"
    return [
        args.python_executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--label",
        scenario.name,
        "--backend",
        "openai",
        "--base-url",
        f"http://{args.host}:{port}",
        "--model",
        served_model_name,
        "--served-model-name",
        served_model_name,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_tokens),
        "--random-output-len",
        str(args.output_tokens),
        "--num-prompts",
        "1",
        "--num-warmups",
        str(args.num_warmups),
        "--max-concurrency",
        "1",
        "--request-rate",
        str(args.request_rate),
        "--temperature",
        "0",
        "--ignore-eos",
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        str(output_path.parent),
        "--result-filename",
        output_path.name,
        "--seed",
        str(args.seed),
    ]


def _wait_for_server(
    host: str,
    port: int,
    timeout_s: int,
    proc: subprocess.Popen[Any],
) -> None:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with return code {proc.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(2)
            continue
    raise TimeoutError(f"Timed out waiting for server at {url}.")


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
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
    return [
        sum(vector[idx] for vector in vectors) / len(vectors)
        for idx in range(width)
    ]


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output_throughputs = [
        float(run["output_throughput"])
        for run in runs
        if run.get("output_throughput") is not None
    ]
    total_token_throughputs = [
        float(run["total_token_throughput"])
        for run in runs
        if run.get("total_token_throughput") is not None
    ]
    ttfts = [
        float(run["mean_ttft_ms"])
        for run in runs
        if run.get("mean_ttft_ms") is not None
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
        "median_total_token_throughput": _median(total_token_throughputs),
        "median_mean_ttft_ms": _median(ttfts),
        "median_spec_decode_acceptance_rate": _median(acceptance_rates),
        "median_spec_decode_acceptance_length": _median(acceptance_lengths),
        "mean_spec_decode_per_position_acceptance_rates": _mean_vector(
            per_position_vectors
        ),
    }


def _delta_summary(
    higher: dict[str, Any],
    lower: dict[str, Any],
    *,
    higher_name: str,
    lower_name: str,
) -> dict[str, Any]:
    higher_value = higher.get("median_output_throughput")
    lower_value = lower.get("median_output_throughput")
    if higher_value is None or lower_value is None:
        return {"higher": higher_name, "lower": lower_name}
    delta = higher_value - lower_value
    pct = (delta / lower_value * 100.0) if lower_value else None
    return {
        "higher": higher_name,
        "lower": lower_name,
        "output_throughput_delta": delta,
        "output_throughput_delta_pct": pct,
    }


def main() -> None:
    args = _parse_args()
    root_dir = Path(args.result_dir) / time.strftime("%Y%m%d-%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario_index, scenario in enumerate(_scenarios(args)):
        port = args.base_port + scenario_index
        scenario_dir = root_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        env = _base_env(args, scenario_dir, scenario)
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
                    bench_command = _bench_command(
                        args,
                        scenario,
                        port,
                        run_output_path,
                    )
                    subprocess.run(
                        bench_command,
                        cwd=Path(__file__).resolve().parents[1],
                        env=env,
                        check=True,
                    )
                    run_result = _load_json(run_output_path)
                    run_result["benchmark_command"] = bench_command
                    run_result["server_command"] = server_command
                    run_result["server_log_path"] = str(server_log_path)
                    runs.append(run_result)
            finally:
                _terminate_process(proc)

        summary = _summarize_runs(runs)
        summary["server_command"] = server_command
        summary["server_log_path"] = str(server_log_path)
        scenario_summaries[scenario.name] = summary
        with (scenario_dir / "summary.json").open("w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    comparison: dict[str, Any] = {"scenarios": scenario_summaries}
    if (
        "baseline" in scenario_summaries
        and "baseline_no_custom_all_reduce" in scenario_summaries
    ):
        comparison["custom_all_reduce_tax"] = _delta_summary(
            scenario_summaries["baseline"],
            scenario_summaries["baseline_no_custom_all_reduce"],
            higher_name="baseline",
            lower_name="baseline_no_custom_all_reduce",
        )
    if (
        "baseline_no_custom_all_reduce" in scenario_summaries
        and "dflash" in scenario_summaries
    ):
        comparison["dflash_delta_vs_comm_matched_baseline"] = _delta_summary(
            scenario_summaries["dflash"],
            scenario_summaries["baseline_no_custom_all_reduce"],
            higher_name="dflash",
            lower_name="baseline_no_custom_all_reduce",
        )

    summary_path = root_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {"summary_path": str(summary_path), **comparison},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
