"""Per-model configuration for the fleet regression runner.

Each entry tells the runner:
- which launch script to invoke (user-local at ~/launch_<name>.sh)
- which served_id to address over HTTP after the engine boots
- which measurement suites to run
- the baseline tok/s values for floor-style perf assertions

Baselines are recorded measurements from the last green run on this fork.
Update them when a legitimate perf change lands (with a one-line PR note);
treat unexpected movement as a regression to investigate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PERF_FLOOR = 0.85  # measured tok/s must be >= floor * baseline to pass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    launch_script: str
    served_id: str
    ready_timeout_s: int
    suites: tuple[str, ...]
    baselines_tokps: dict[str, float] = field(default_factory=dict)
    pi_prompt: str = (
        "Create hello.txt containing HELLO. Read it back and tell me its size in bytes."
    )
    pi_expected_file: str = "hello.txt"
    pi_expected_content: str = "HELLO"


REGISTRY: dict[str, ModelConfig] = {
    "granite": ModelConfig(
        name="granite",
        launch_script="~/launch_granite.sh",
        served_id="granite-4.1-8b-AWQ-INT4",
        # Bumped from 120s — the 120s budget covered weight load + Dynamo cache
        # hit but didn't leave room for cudagraph capture under sequential
        # disk pressure (full-fleet sweep 2026-05-18 hit it at 67s into boot
        # with cudagraph capture still running). 300s is the realistic worst-
        # case for a cold-page-cache boot following a 119B-GGUF launch.
        ready_timeout_s=300,
        suites=("smoke", "perf_t1", "pi_toolcall"),
        baselines_tokps={
            # 2026-05-17: HTTP-measured perf_t1 = 164.0 tok/s, TP=2, cudagraph.
            # Higher than the 127 from in-process LLM.generate() (prior memory)
            # because our prefill_subtract_s=0.5 is too generous for short
            # prompts — keep as the operational baseline for this measurement.
            "perf_t1": 164.0,
        },
    ),
    "qwen36": ModelConfig(
        name="qwen36",
        launch_script="~/launch_qwen36.sh",
        served_id="Qwen3.6-27B-AWQ-INT4",
        # Bumped from 180s — cold-cache torch.compile (Dynamo + inductor) can
        # take ~70s on top of ~100s weight load on this 27B-AWQ when running
        # right after another model evicted the page cache.
        ready_timeout_s=600,
        suites=("smoke", "perf_t1", "pi_toolcall"),
        baselines_tokps={
            # 2026-05-17: HTTP-measured perf_t1 = 65.3 tok/s, TP=4, cudagraph
            # + TRITON_ATTN, max_num_seqs=1. 8.8× over the prior --enforce-eager
            # config (which was 7.4 tok/s) — cudagraph engages cleanly on the
            # hybrid DeltaNet + dense-attention layers despite the SSM state.
            "perf_t1": 65.3,
        },
    ),
    "mistral4": ModelConfig(
        name="mistral4",
        launch_script="~/launch_mistral4.sh",
        served_id="Mistral-Small-4-119B-Q4_K_M",
        # Bumped from 1200s — full-fleet sweep 2026-05-18 timed out at 1200s
        # mid-tensor-materialization (cold disk after granite + qwen36 page-
        # cache eviction). 1800s covers worst-case cold-disk on this 119B
        # GGUF without changing warm-disk fast-path behavior.
        ready_timeout_s=1800,
        suites=("smoke", "perf_t1", "perf_t2", "perf_t3", "pi_toolcall"),
        baselines_tokps={
            "perf_t1": 83.5,  # short-prompt 256-tok decode (T1)
            "perf_t2": 23.6,  # ~6k prompt 512-tok decode (T2)
            "perf_t3": 26.3,  # prefix-cache 128-tok decode (T3)
        },
    ),
    "mimo_v25": ModelConfig(
        name="mimo_v25",
        launch_script="~/launch_mimo_v25.sh",
        served_id="MiMo-V2.5-Q3_K_M",
        ready_timeout_s=1200,
        suites=("smoke", "perf_t1", "pi_toolcall"),
        baselines_tokps={
            # 2026-05-17: HTTP-measured perf_t1 = 47.48 tok/s, TP=8, cudagraph.
            # Within +13% of the prior "42 tok/s single-stream" memory note;
            # locking the higher number here as our HTTP-measurement baseline.
            "perf_t1": 47.5,
        },
    ),
    "minimax_m27": ModelConfig(
        name="minimax_m27",
        launch_script="~/launch_minimax_m2.sh",
        served_id="MiniMax-M2.7-AWQ-4bit",
        # Cold-disk weight load measured 1645s standalone, but full-fleet
        # sweep 2026-05-18 ran past 1800s mid-load (26/27 shards at the
        # timeout, ~69s/shard sustained — sequential disk pressure from the
        # 4 prior launches in the same session). 2400s covers worst-case
        # cold-disk-after-fleet-pressure for this 122 GiB AWQ (27 shards
        # × ~70s + init).
        ready_timeout_s=2400,
        # nul_scan is the load-bearing regression test for the fp16 last-layer
        # AllReduce overflow fix in vllm/model_executor/models/minimax_m2.py.
        # If anyone edits that file (or env-var defaults) and breaks the
        # protection, this suite catches it via 4-turn polyfact decode + NUL
        # scan. See humanjesse/vllm-v100#11.
        suites=("smoke", "perf_t1", "pi_toolcall", "nul_scan"),
        baselines_tokps={
            # 2026-05-18: HTTP-measured perf_t1 = 73.35 tok/s median over
            # 5 warm runs (stdev ~1.0), TP=8, cudagraph + FLASH_ATTN_V100,
            # max_num_seqs=4. Higher than the project memory note of
            # "~54 tok/s decode at batch=1" — that earlier figure was a
            # single eager-mode measurement; this is the cudagraph regime
            # called out in bayley's reproducer config.
            #
            # NOTE: this baseline assumes the fp16 last-layer AllReduce
            # overflow fix landed in vllm/model_executor/models/minimax_m2.py
            # is active (default VLLM_ALLREDUCE_OVERFLOW_STRATEGY=fast).
            # Without it, the model produces NUL bytes in long-context
            # output (humanjesse/vllm-v100#11).
            "perf_t1": 73.35,
        },
    ),
}


def long_prompt_corpus() -> str:
    """Read the committed long-prompt corpus (used by perf_t2/t3 suites)."""
    return (Path(__file__).parent / "corpus" / "long_prompt.txt").read_text()
