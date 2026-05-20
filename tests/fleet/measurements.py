"""Per-suite measurement functions.

Each returns a result dict:
    {
        "name": str,             # suite name
        "passed": bool,
        "elapsed_s": float,      # wall-clock for the measurement itself
        "details": dict,         # suite-specific (tok/s, n_out, finish_reason, ...)
        "error": str | None,
    }

Floor-style perf assertion: measured >= 0.85 * baseline (see registry.PERF_FLOOR).
Smoke + Pi assertions are exact-match / boolean.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from tests.fleet.registry import PERF_FLOOR, ModelConfig, long_prompt_corpus

PI_CLI = (
    "/home/admin/.npm-global/lib/node_modules/"
    "@earendil-works/pi-coding-agent/dist/cli.js"
)
PI_NODE = "/home/admin/node26/bin/node"


def _post_json(url: str, payload: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


# ---------- smoke ----------

def smoke(model: ModelConfig, base_url: str = "http://127.0.0.1:8000") -> dict:
    """Trivial 10-token completion — would have caught the qwen36 auto-trim deadlock."""
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    try:
        body = _post_json(
            f"{base_url}/v1/completions",
            {
                "model": model.served_id,
                "prompt": "Hello, world",
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout_s=60,
        )
        n_out = body["usage"]["completion_tokens"]
        text = body["choices"][0]["text"]
        details = {"n_out": n_out, "text": text[:80]}
        passed = n_out >= 1
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": "smoke", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- perf ----------

def _decode_tokps(usage: dict, wall_s: float, prefill_subtract_s: float) -> float:
    """tok/s = decoded_tokens / (wall - prefill estimate). Floor wall at 1e-3."""
    n_out = usage.get("completion_tokens", 0)
    denom = max(wall_s - prefill_subtract_s, 1e-3)
    return n_out / denom


def _run_chat(base_url: str, model_id: str, prompt: str, max_tokens: int) -> dict:
    """Non-stream chat completion. Returns full JSON response."""
    return _post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout_s=600,
    )


def perf_t1(model: ModelConfig, base_url: str = "http://127.0.0.1:8000") -> dict:
    """Short prompt, 256-token decode. Mirrors /tmp/test_s13_stress.py Test 1."""
    return _perf_one_shot(
        model, base_url,
        suite="perf_t1",
        prompt="Explain photosynthesis in three short paragraphs.",
        max_tokens=256,
        prefill_subtract_s=0.5,
    )


def perf_t2(model: ModelConfig, base_url: str = "http://127.0.0.1:8000") -> dict:
    """~6k-token prompt, 512-token decode. Long-context decode regime."""
    base = long_prompt_corpus()
    body = (base + "\n\n") * 4
    prompt = (
        body
        + "\n\nGiven the above text, summarize the main themes in detail "
          "covering geography, biology, physics, history, and technology "
          "as separate sections with examples from the text."
    )
    return _perf_one_shot(
        model, base_url,
        suite="perf_t2",
        prompt=prompt,
        max_tokens=512,
        prefill_subtract_s=2.0,
    )


def perf_t3(model: ModelConfig, base_url: str = "http://127.0.0.1:8000") -> dict:
    """Re-send T2 prompt for 128-tok decode — exercises prefix cache."""
    base = long_prompt_corpus()
    body = (base + "\n\n") * 4
    prompt = (
        body
        + "\n\nGiven the above text, summarize the main themes in detail "
          "covering geography, biology, physics, history, and technology "
          "as separate sections with examples from the text."
    )
    return _perf_one_shot(
        model, base_url,
        suite="perf_t3",
        prompt=prompt,
        max_tokens=128,
        prefill_subtract_s=0.5,
    )


def _perf_one_shot(
    model: ModelConfig, base_url: str, *,
    suite: str, prompt: str, max_tokens: int, prefill_subtract_s: float,
) -> dict:
    baseline = model.baselines_tokps.get(suite, 0.0)
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    try:
        body = _run_chat(base_url, model.served_id, prompt, max_tokens)
        wall = time.perf_counter() - t0
        usage = body.get("usage", {})
        finish = body["choices"][0].get("finish_reason")
        tokps = _decode_tokps(usage, wall, prefill_subtract_s)
        details = {
            "n_prompt": usage.get("prompt_tokens", 0),
            "n_out": usage.get("completion_tokens", 0),
            "wall_s": round(wall, 2),
            "tokps": round(tokps, 2),
            "baseline_tokps": baseline,
            "floor_tokps": round(PERF_FLOOR * baseline, 2) if baseline > 0 else 0.0,
            "finish_reason": finish,
        }
        # Pass criteria: at least some tokens were generated AND
        # (if a real baseline is set) tokps clears the floor.
        produced_output = usage.get("completion_tokens", 0) > 0
        if baseline > 0:
            passed = produced_output and tokps >= PERF_FLOOR * baseline
        else:
            # record-only mode (baseline=0.0): just need any output
            passed = produced_output
            details["mode"] = "record_only"
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "name": suite, "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- pi tool-call ----------

def pi_toolcall(model: ModelConfig, base_url: str = "http://127.0.0.1:8000") -> dict:
    """Spawn pi-coding-agent against this model, assert it executed tools and
    produced the expected file.
    """
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    workdir = tempfile.mkdtemp(prefix=f"pi_{model.name}_")
    try:
        # Pi reads its provider config from ~/.pi/agent/models.json and the
        # model id must be registered there. base_url isn't passed to Pi —
        # it uses the URL from models.json (localhost:8000 for local-vllm).
        cmd = [
            PI_NODE, PI_CLI,
            "--provider", "local-vllm",
            "--model", model.served_id,
            "--no-context-files", "--no-extensions", "--no-skills",
            "--no-prompt-templates", "--no-themes",
            "-p", model.pi_prompt,
        ]
        proc = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=300
        )
        expected = Path(workdir) / model.pi_expected_file
        file_ok = expected.is_file()
        content_ok = file_ok and expected.read_text().strip() == model.pi_expected_content
        details = {
            "workdir": workdir,
            "exit_code": proc.returncode,
            "file_created": file_ok,
            "content_matches": content_ok,
            "stdout_tail": proc.stdout[-200:],
        }
        passed = proc.returncode == 0 and file_ok and content_ok
    except subprocess.TimeoutExpired:
        err = "Pi cli timed out (300s)"
    except FileNotFoundError as e:
        err = f"Pi binary missing: {e}"
    finally:
        # Only clean up if we passed — leave artifacts on failure for inspection.
        if passed:
            shutil.rmtree(workdir, ignore_errors=True)
    return {
        "name": "pi_toolcall", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


# ---------- nul-byte scan ----------

# Multi-turn polyfact-style prompts used by `nul_scan`. Chosen because each
# response decodes ~4-6K tokens of real code (the largest single tool the model
# has for producing varied logits), and back-to-back turns build context depth
# to mirror the regime where humanjesse/vllm-v100#11 originally fired. 4 turns
# x ~4K tokens = ~16K tokens of decoded content per run -- short enough for a
# fleet sweep, long enough to catch a regressed fp16 AR fix on V100 MoE models.
_NUL_SCAN_TURNS: tuple[str, ...] = (
    "Write a complete Python module implementing polynomial arithmetic over "
    "Z[x]: a Polynomial class with __init__, __add__, __sub__, __mul__, "
    "__divmod__, __mod__, __floordiv__, __eq__, __repr__, degree, "
    "leading_coeff, is_zero. 100+ lines, docstrings included, raw code -- no "
    "markdown wrapping, no bullet-point todo lists.",
    "Now write a complete gcd.py: Euclidean GCD over Z[x] using "
    "pseudo-remainder, extended Euclidean algorithm, content (GCD of "
    "coefficients), primitive_part. 80+ lines.",
    "Now write a complete finite_field.py: polynomial arithmetic mod p, "
    "multiplicative inverses mod p, polynomial GCD mod p, division mod p. "
    "100+ lines.",
    "Now write a complete factor.py: top-level factor() combining content "
    "extraction, square-free factorization, finite-field factorization, "
    "brute-force factor recombination. 100+ lines.",
)


def nul_scan(model: ModelConfig, base_url: str = "http://127.0.0.1:8000") -> dict:
    """Multi-turn polyfact-style decode + NUL-byte scan of all assistant
    output. Targets the fp16 last-layer AllReduce overflow class
    (humanjesse/vllm-v100#11) where NaN logits get sampled to token id 0 and
    surface as 0x00 bytes via byte-fallback tokenizers.

    Pass criteria:
      * All N turns return a response with at least 1 token.
      * Zero NUL bytes (0x00) across the entire concatenated assistant
        transcript.

    Fail modes worth investigating if this regresses:
      * NULs > 0 -> the per-model fp32-AR fix in the relevant model file is
        no longer doing its job (env var flipped? rebase dropped it?
        upstream merge clobbered it?).
      * Empty response -> tool-call/template/health issue, not this bug.
    """
    t0 = time.perf_counter()
    err: str | None = None
    details: dict = {}
    passed = False
    transcript_bytes = b""
    n_turns = 0
    n_total_tokens = 0
    try:
        messages: list[dict] = []
        for prompt in _NUL_SCAN_TURNS:
            messages.append({"role": "user", "content": prompt})
            body = _post_json(
                f"{base_url}/v1/chat/completions",
                {
                    "model": model.served_id,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    # Disable thinking blocks on models that support the
                    # chat_template_kwarg -- they bloat tokens without
                    # adding decode surface for this bug class.
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout_s=600,
            )
            msg = body["choices"][0]["message"]
            # Collect every generated-token surface vLLM exposes. Models with
            # a reasoning parser configured (e.g. MiniMax-M2.7 with
            # --reasoning-parser minimax_m2) route tokens into `reasoning` /
            # `reasoning_content`; tool-call args land in `tool_calls`. All of
            # these went through the same TP AllReduce path, so all of them
            # need to be NUL-scanned -- the bug we're catching surfaces in
            # whichever output field happened to be active when a NaN logit
            # picked token id 0.
            parts: list[str] = []
            for k in ("content", "reasoning", "reasoning_content"):
                v = msg.get(k)
                if v:
                    parts.append(v)
            for tc in msg.get("tool_calls") or []:
                args = ((tc or {}).get("function") or {}).get("arguments")
                if args:
                    parts.append(args)
            assistant_text = "".join(parts)
            # Re-thread the assistant turn for the next prompt. For most
            # models `content` carries the visible answer; for reasoning-
            # parser models we splice the reasoning back in so the next turn
            # has the same context the model thinks it has.
            messages.append({"role": "assistant", "content": assistant_text})
            transcript_bytes += assistant_text.encode("utf-8", errors="surrogateescape")
            n_turns += 1
            n_total_tokens += body.get("usage", {}).get("completion_tokens", 0)

        nul_count = transcript_bytes.count(b"\x00")
        # Other control chars (everything <0x20 except \t \n \r). Symptom of
        # the same class -- byte-fallback tokenizers can also emit other
        # low-bytes when logits go NaN at certain positions.
        ctrl_count = sum(
            1 for b in transcript_bytes
            if b < 0x20 and b not in (0x09, 0x0A, 0x0D)
        )
        details = {
            "n_turns": n_turns,
            "bytes": len(transcript_bytes),
            "tokens": n_total_tokens,
            "nul_bytes": nul_count,
            "other_ctrl_bytes": ctrl_count,
        }
        passed = n_turns == len(_NUL_SCAN_TURNS) and nul_count == 0
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
        details = {
            "n_turns": n_turns,
            "bytes": len(transcript_bytes),
            "nul_bytes": transcript_bytes.count(b"\x00"),
        }
    return {
        "name": "nul_scan", "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "details": details, "error": err,
    }


SUITE_FUNCS = {
    "smoke": smoke,
    "perf_t1": perf_t1,
    "perf_t2": perf_t2,
    "perf_t3": perf_t3,
    "pi_toolcall": pi_toolcall,
    "nul_scan": nul_scan,
}
