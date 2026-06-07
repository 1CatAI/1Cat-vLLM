# V100 field validation — 2026-06-06 (pve3, 2× V100-PCIE-32GB)

Companion to [PR #55](https://github.com/1CatAI/1Cat-vLLM/pull/55). Bench session comparing stock 1.1.0 wheels vs rollup overlay (`/opt/1cat-vllm/.venv-v110`) on **Qwen3.6-27B-FP8** (community recipe) and **Deckard-40B-FP8-MTP**.

**Bench script:** `bench_deckard_toks.py` (c1, 256 tokens, `ignore_eos`). Do **not** use `vllm bench serve` for community tok/s comparisons — it reported ~27 tok/s on Qwen27B where real serving is ~51 tok/s.

## Headline: rollup is not a Qwen27B throughput tax

| Variant | Mean c1 | Notes |
|---------|---------|-------|
| Stock 1.1.0, community flags, auto fp16 KV | **51.6** | No rollup overlay |
| Rollup v110 overlay, same flags | **51.5** | `attention.py` + serving + parser |
| Rollup, auto fp16 KV @ 177k | **49.6** | Run-to-run variance |
| Rollup, `fp8_e5m2` KV @ 177k | **49.7** | MTP accept 70% vs 68% — noise |
| Stock + `fp8_e5m2` on FP8 weights | **BOOT_FAIL** | Needs **#54** |

The ~⅓ throughput concern was a **bench-tool artifact**, not a serving regression. For max Qwen27B speed, stock wheels + community flags suffice. Rollup remains required for Deckard prod (e5m2 @ 177k, reasoning split).

## KV cache matrix on SM70

| `--kv-cache-dtype` | Qwen27B @ 177k | Deckard40B @ 177k |
|--------------------|----------------|-------------------|
| *(omit — auto fp16)* | ✓ ~50 tok/s | **BOOT_FAIL** (KV OOM) |
| `fp8` (e4m3) | **BOOT_FAIL** | **BOOT_FAIL** |
| `fp8_e5m2` | ✓ (rollup only) | ✓ (rollup only) |

- **`--kv-cache-dtype fp8` (e4m3):** hard fails on SM70 (`fp8e4nv not supported`) — stock, v110, and rollup all fail identically.
- **`fp8_e5m2`:** load-bearing for Deckard @ 177k; neutral on Qwen27B speed (optional memory saver).
- **#54 (`attention.py`):** required for `fp8_e5m2` on dense FP8 checkpoints.

## Deckard-40B-FP8-MTP reference

| Config | Mean c1 |
|--------|---------|
| Rollup + e5m2 @ 177k (prod, MTP n=1) | **32.5** |
| Auto fp16 KV @ 99k | **35.6** |
| Auto fp16 KV @ 177k | **BOOT_FAIL** |

## Practical guide

| Goal | Config |
|------|--------|
| Max tok/s, 2×V100 | Qwen27B-FP8 stock, community flags, auto fp16 KV |
| Deckard + 177k ctx | Rollup + `fp8_e5m2` |
| Deckard speed, no e5m2 | Deckard @ 99k auto fp16 KV |
| Do not use on V100 | `--kv-cache-dtype fp8` (e4m3) |

## Trip-ups validated

- Clients need `--served-model-name deckard-fp8mtp <model>` when swapping models behind a fixed endpoint.
- One bench process at a time; fail fast on boot errors (check `journalctl`, don't blind-poll health).

## Open follow-ups (not blocking merge)

- Deckard MTP n=1 (prod) vs n=2 (community) on rollup+e5m2 @ 177k
- CUDA graph capture (`cudagraph_mode=FULL_AND_PIECEWISE`, `cudagraph_num_of_warmups=3`) A/B on no-NVLink V100
- Concurrency sweep (c4/c8/c16) — all numbers here are c1
