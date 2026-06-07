# Field validation (2026-06-06, pve3, 2× V100-PCIE-32GB)

Follow-up bench session comparing stock 1.1.0 wheels vs rollup overlay
(`/opt/1cat-vllm/.venv-v110`) across **Qwen3.6-27B-FP8** (community recipe) and
**Deckard-40B-FP8-MTP**.

All throughput numbers are **c1 single-stream** from `bench_deckard_toks.py` (256 tokens,
`ignore_eos`). **Do not use `vllm bench serve`** for community comparisons — it
understated Qwen27B at ~27 tok/s vs the real ~51 tok/s.

## Qwen27B-FP8 — rollup is not a throughput tax

| Variant | Mean c1 | Notes |
|---------|---------|-------|
| Stock 1.1.0 wheels, community flags, auto fp16 KV | **51.6** | No rollup overlay |
| Rollup v110 overlay, same flags, auto fp16 KV | **51.5** | `attention.py` + serving + parser |
| Rollup v110, auto fp16 KV @ 177k | **49.6** | Run-to-run variance, not KV mode |
| Rollup v110, `fp8_e5m2` KV @ 177k | **49.7** | MTP accept 70% vs 68% — within noise |
| Stock + `fp8_e5m2` on FP8 weights | **BOOT_FAIL** | Needs **#54** (`attention.py`) |

**Verdict:** Rollup patches are **~0 tok/s regression** on Qwen27B-FP8 community recipe.
The ~⅓ throughput concern was a **bench-tool artifact** (`vllm bench serve`), not a real
serving penalty.

## KV cache modes on SM70

| `--kv-cache-dtype` | Qwen27B @ 177k | Deckard40B @ 177k | Notes |
|--------------------|----------------|-------------------|-------|
| *(omit — auto fp16)* | ✓ boots, ~50 tok/s | **BOOT_FAIL** (KV OOM) | Community recipe for 27B |
| `fp8` (e4m3) | **BOOT_FAIL** | **BOOT_FAIL** | `fp8e4nv not supported` on SM70 |
| `fp8_e5m2` | ✓ boots (rollup only) | ✓ boots (rollup only) | Memory lever for 40B @ 177k; neutral on Qwen speed |

- **#54 is load-bearing** for `fp8_e5m2` on dense FP8 checkpoints.
- **`--kv-cache-dtype fp8` (e4m3) is not a V100 knob today.**
- **`fp8_e5m2` on Qwen27B** is optional memory saver with no measured speed penalty.

## Deckard-40B-FP8-MTP reference numbers

| Config | Mean c1 | Notes |
|--------|---------|-------|
| Rollup + e5m2 @ 177k (prod recipe) | **32.5** | MTP n=1 |
| Auto fp16 KV @ 99k (no e5m2) | **35.6** | Max ctx without e5m2 |
| Auto fp16 KV @ 177k | **BOOT_FAIL** | KV OOM |

## Practical guide

| Goal | Config |
|------|--------|
| Max tok/s on 2×V100 | Qwen27B-FP8 stock wheels, community flags, auto fp16 KV (~51 c1) |
| Deckard tune + 177k ctx | Rollup overlay + `fp8_e5m2` KV (~32.5 c1) |
| Deckard speed without e5m2 | Deckard @ 99k auto fp16 KV (~35.6 c1) |
| Do not use on V100 | `--kv-cache-dtype fp8` (e4m3) |

## Open follow-ups (not blocking merge)

- Deckard prod MTP `n=1` vs community `n=2` — not A/B'd on rollup+e5m2 @ 177k
- CUDA graph capture mode not systematically A/B'd on no-NVLink V100 pair
- All numbers are c1; prod runs c16
- End-to-end client smoke from `10.4.20.110` lightly tested after alias fix
