# Fleet regression runner

End-to-end "is this build still good?" check for the verified V100 model fleet.
For each model: launches the engine, runs a small suite (smoke + perf + Pi tool-call),
tears down, moves to next. Reports a markdown pass/fail summary with tok/s numbers
and deltas vs recorded baselines.

Designed to catch the class of regression that unit tests don't — triton-version
drift, KV-cache auto-trim misconfig, chat-template parser breakage, etc.

## Usage

```bash
# Full sweep (all models, each model's default suite list)
python -m tests.fleet.run

# A subset of models
python -m tests.fleet.run --models granite,qwen36

# A specific suite across all models
python -m tests.fleet.run --suites smoke,perf_t1

# Write summary to a file (also always printed to stdout)
python -m tests.fleet.run --models mistral4 --report /tmp/mistral4.md
```

Exit code: `0` if all suites passed, `1` if any failed, `2` for argument errors.

## What gets checked

| Suite | What | Pass criteria |
|---|---|---|
| smoke | `POST /v1/completions` with `max_tokens=10`, 60s timeout | engine produced ≥1 token |
| perf_t1 | short prompt, 256-tok decode | tok/s ≥ 0.85 × baseline |
| perf_t2 | ~6k-tok prompt, 512-tok decode (long-context regime) | tok/s ≥ 0.85 × baseline |
| perf_t3 | T2 prompt re-sent, 128-tok decode (prefix-cache replay) | tok/s ≥ 0.85 × baseline |
| pi_toolcall | spawn `pi-coding-agent` against the endpoint, run a file-create+read task | exit 0, file written, content matches |

Not every model runs every suite — see `registry.REGISTRY` for per-model
suite lists. Mistral4 is the only one currently running T2/T3 (Granite/Qwen3.6
launch configs use shorter max_model_len; MiMo's T2/T3 not yet baselined).

## Contract

The runner expects these user-local files to exist (they're not in this repo —
they're per-machine launch wrappers):

- `~/launch_granite.sh`
- `~/launch_qwen36.sh`
- `~/launch_mistral4.sh`
- `~/launch_mimo_v25.sh`

Each must:
- Bind to `0.0.0.0:8000` (the runner hits `localhost:8000`)
- Serve the `served_id` declared in `registry.py` for that model
- Pass `--enable-auto-tool-choice --tool-call-parser <name>` (required by the
  `pi_toolcall` suite — see `reference_tool_call_parsers_per_model.md` in memory)
- Print `Application startup complete` once the engine is ready (vLLM default)
- Be runnable as `bash <script>` without args

The runner also expects `~/.pi/agent/models.json` to list each `served_id`
under the `local-vllm` provider, otherwise `pi_toolcall` will 400.

## Updating baselines

Baselines live in `registry.py` as `baselines_tokps` per model. Floor is
`PERF_FLOOR = 0.85` — measured tok/s must clear `0.85 × baseline` to pass.

When a legitimate perf change lands (kernel rewrite, dependency upgrade, etc.):
1. Run the sweep — note the new tok/s numbers
2. Update `baselines_tokps` to the new measurements
3. Include the new numbers in the PR description that landed the change

A baseline of `0.0` is a sentinel — that suite runs in record-only mode (no
floor assertion, just verifies the engine produced output). Use it for new
models before they've been baselined.

## Measurement caveats

- `perf_*` uses a fixed `prefill_subtract_s` to back out prefill from a single
  non-stream round-trip. This is a rough estimate (cloned from
  `/tmp/test_s13_stress.py`) and is *too aggressive* for short prompts, biasing
  reported tok/s upward. A future iteration could switch to streaming + TTFT
  subtraction for a cleaner decode-only number — but the current methodology
  is consistent within itself, so floor=0.85 still catches regressions reliably.
- Each suite runs once (no median-of-N). Measurement noise is real; a single
  failing run on a borderline model is worth re-running before declaring a
  regression.
- Engine launch time isn't counted toward suite wall time — it's reported
  separately in the runner's stdout (`[fleet] ready` log line).

## Pre-flight

The runner refuses to launch if any GPU is holding >200 MiB before start —
that signals orphan workers from a crashed prior session. Clear them with:

```bash
pkill -9 -f 'vllm serve|EngineCore|Worker_TP'
```

then retry.
