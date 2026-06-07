# V100 field validation — 2026-06-06/07 (pve3, 2× V100-PCIE-32GB)

Companion to [PR #55](https://github.com/1CatAI/1Cat-vLLM/pull/55). Bench session comparing stock 1.1.0 wheels vs rollup overlay (`/opt/1cat-vllm/.venv-v110`) on **Qwen3.6-27B-FP8** (community recipe) and **Deckard-40B-FP8-MTP**.

**Bench script:** `bench_deckard_toks.py` (c1, 256 tokens, `ignore_eos`). Do **not** use `vllm bench serve` for cross-repo community comparisons — it reported ~27 tok/s on Qwen27B where real serving is ~51 tok/s.

## Headline: rollup is not a Qwen27B throughput tax

| Variant | Mean c1 | Notes |
|---------|---------|-------|
| Stock 1.1.0, community flags, auto fp16 KV | **51.6** | No rollup overlay |
| Rollup v110 overlay, same flags | **51.5** | `attention.py` + serving + parser |
| Rollup, auto fp16 KV @ 177k | **49.6** | Run-to-run variance |
| Rollup, `fp8_e5m2` KV @ 177k | **49.7** | MTP accept 70% vs 68% — noise |
| Stock + `fp8_e5m2` on FP8 weights | **BOOT_FAIL** | Needs **#54** |

The ~⅓ throughput concern was a **bench-tool artifact**, not a serving regression. For max Qwen27B speed, stock wheels + community flags suffice. Rollup remains required for Deckard prod (e5m2 @ 177k, reasoning split).

## Deckard @ 177k — MTP `n=2` unlocks ~40 tok/s (2026-06-07)

| Variant | Mean c1 | MTP accept | Notes |
|---------|---------|------------|-------|
| **MTP `n=2`, batched 8192 (prod)** | **~40.0** | 71.4% | ~25% over n=1 |
| MTP `n=2`, fullgraph | 40.0 | 71.4% | fullgraph adds nothing |
| MTP `n=2`, batched 4096 | 40.0 | 71.4% | batched size irrelevant at c1 |
| MTP `n=1` (old prod-ref) | **~32.3** | 82.9% | higher accept %, fewer tokens/step |

`n=1` has a higher acceptance *rate*, but `n=2` proposes more draft tokens per step → higher net throughput. Do not pick `n=1` based on accept % alone.

Other prod knobs: `fp8_e5m2` KV (required for 177k), `NCCL_P2P_DISABLE=1` (P2P on = ~4× slowdown on this topology), default cudagraph (fullgraph no win).

## KV cache matrix on SM70

| `--kv-cache-dtype` | Qwen27B @ 177k | Deckard40B @ 177k |
|--------------------|----------------|-------------------|
| *(omit — auto fp16)* | ✓ ~50 tok/s | **BOOT_FAIL** (KV OOM) |
| `fp8` (e4m3) | **BOOT_FAIL** | **BOOT_FAIL** |
| `fp8_e5m2` | ✓ (rollup only) | ✓ (rollup only) |

- **`--kv-cache-dtype fp8` (e4m3):** hard fails on SM70 (`fp8e4nv not supported`).
- **`fp8_e5m2`:** load-bearing for Deckard @ 177k; neutral on Qwen27B speed.
- **#54 (`attention.py`):** required for `fp8_e5m2` on dense FP8 checkpoints.

## Practical guide

| Goal | Config |
|------|--------|
| Max tok/s, 2×V100 | Qwen27B-FP8 stock, community flags, auto fp16 KV (~51 c1) |
| Deckard + 177k ctx | Rollup + `fp8_e5m2` + **MTP `n=2`** (~40 c1) |
| Deckard speed, no e5m2 | Deckard @ 99k auto fp16 KV (~35.6 c1) |
| Do not use on V100 | `--kv-cache-dtype fp8` (e4m3) |

## Trip-ups validated

- Clients need `--served-model-name deckard-fp8mtp <model>` when swapping models behind a fixed endpoint.
- One bench process at a time; fail fast on boot errors (check `journalctl`, don't blind-poll health).

## Open follow-ups

- n=2 concurrency sweep (c4/c8/c16) — only n=1 matrix run so far
- End-to-end client smoke from `10.4.20.110` with thinking enabled
