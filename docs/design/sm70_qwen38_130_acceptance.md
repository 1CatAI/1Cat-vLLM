# SM70 Qwen3.8-27B 1.3.0 Acceptance

Date: 2026-08-15

## Contract

- Model: `Qwen3.8-27B-FP8`, TP4 on four V100-SXM2-16GB GPUs.
- Runtime: Python 3.12, Torch 2.10.0+cu128, CUDA 12.8, CUDA graphs enabled.
- Production attention: `FLASH_ATTN_V100`; no eager or Marlin performance
  evidence is accepted.
- Official sampling is `temperature=1.0`, `top_p=0.95`, `top_k=20`.
- Main long-context configuration uses 262144 tokens, chunked prefill, Mamba
  align mode, and prefix caching. Speed tables use no MTP unless stated.

## Accepted Changes

- The G6 XQA QK producer/consumer pipeline is default-on only for the proven
  111104-147840 token range. The final range keeps the prior kernel. At 128K,
  full-model TPOT improves from 21.292 to 20.293 ms (-4.69%) with the same
  token hash; the 256K route is neutral within 0.16%.
- No-MTP CUDA graph capture now covers request shapes 1, 2, 4, 8, and 16,
  bounded by `max_num_seqs`. This fixes the severe eager fallback above two
  concurrent requests for 0.07 GiB/rank additional graph memory.
- Qwen3.8 does not reuse the Qwen3.6 dynamic-draft-vocabulary asset. The
  implicit asset is now restricted to model paths containing `qwen3.6-27b`;
  Qwen3.8 MTP therefore fails closed to the full vocabulary.
- FlashQLA SM70 is compiled into the wheel. Runtime first tries the packaged
  extension and only source installs may fall back to JIT, with fixed SM70/75
  gencode and retained first-error diagnostics.
- The benchmark harness records official sampling, generated-token count,
  natural-stop state, TTFT/prefill, pure decode TPOT, acceptance length, route
  policy, and token hashes.
- Large-M TP4 FP8 gate/up/down/output projections now use a compile-safe
  runtime exact-dense dispatch with one shared 85 MiB workspace. The workspace
  address is passed as an integer so it is not copied into compiled graph
  inputs. Decode, tails, and numerically unsafe QKV shapes retain TurboMind.

## No-MTP Long Context

The FP16-KV prefill column is the final no-MTP chunk-15680 route. The 1K-64K
rows are two-repeat cold-cache means; 128K and 256K are one cold-cache request.
Decode columns retain the official-sampling release sweep. The FP8-KV sweep
uses a generation suffix and produces 256 tokens at every point.

| Context | FP16 KV prefill tok/s | FP16 KV decode tok/s | FP8 KV prefill tok/s | FP8 KV decode tok/s |
|---:|---:|---:|---:|---:|
| 1K | 2515.1 | 65.05 | 3430.8 | 66.36 |
| 4K | 3847.8 | 63.35 | 3591.3 | 65.61 |
| 8K | 3725.4 | 64.04 | 2995.9 | 65.50 |
| 16K | 3901.6 | 63.90 | 2797.0 | 65.36 |
| 32K | 3437.2 | 60.07* | 2516.2 | 59.97 |
| 64K | 2851.7 | 58.44* | 2063.4 | 52.30 |
| 128K | 2446.5 | 46.95 | 1519.6 | 42.35 |
| 256K | 1602.0 | 36.96 | 1007.1 | 31.49 |

`*` The FP16-KV 32K and 64K requests naturally stopped after 2 and 17 output
tokens. Keep those rows as route checks, not high-confidence decode means.

The 128K Nsight Systems graph-node trace attributes 21.30 ms TPOT to:

| Category | Time | Share |
|---|---:|---:|
| FP8 TurboMind dense GEMM | 8.946 ms | 42.1% |
| Flash-V100 q=1 XQA | 5.274 ms | 25.2% |
| TP all-reduce | 1.874 ms | 8.6% |
| LM head, sample, and gather | 1.301 ms | 6.2% |
| Other graph work and gaps | about 3.6 ms | 17.9% |

NCU counter collection is unavailable on this host (`ERR_NVGPUCTRPERM`), so
no counter-derived bottleneck claim is made for this release.

## Concurrency

All rows use 64 requests of exact 1K input and 256 output tokens with official
sampling. Prompt seeds differ between rows to prevent cross-row prefix-cache
reuse. Every request completed successfully.

| Concurrency | Prior accepted tok/s | Current tok/s | Change | Scale | Efficiency | Mean / P99 TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60.06 | 60.44 | +0.63% | 1.00x | 100.0% | 15.16 / 15.32 ms |
| 2 | 107.00 | 108.72 | +1.61% | 1.80x | 89.9% | 16.29 / 17.13 ms |
| 4 | 193.66 | 192.39 | -0.65% | 3.18x | 79.6% | 16.88 / 19.58 ms |
| 8 | 296.35 | 309.70 | +4.50% | 5.12x | 64.0% | 18.16 / 24.55 ms |
| 16 | 383.00 | 412.35 | +7.66% | 6.82x | 42.6% | 23.03 / 38.57 ms |

The first concurrency-4 run measured 188.68 tok/s; its independent repeat
measured 192.39 tok/s and is reported above. Across the timed intervals all
four V100s average 98.0-99.8% SM busy. Average memory busy falls from 50.7% at
concurrency 1 to 38.3% at concurrency 16 while power rises from 194.7 to
229.8 W/GPU. Remaining non-linear scaling is a high-M execution-efficiency
issue, not an idle-GPU or missing graph-shape issue.

## MTP And Quality

- Full-vocabulary MTP4, FP8 KV, 128K: 58.25 tok/s, 17.168 ms TPOT,
  acceptance length 3.866, 256 valid output tokens.
- Full-vocabulary MTP4, FP16 KV, 128K: 55.40 tok/s, 18.051 ms TPOT,
  acceptance length 4.031, 256 valid output tokens.
- Exact 256K boundary (`input_len=262120`, 24 outputs), MTP4 plus FP8 KV:
  all token IDs are legal, `is_corrupted=false`, and acceptance length is 4.0.
- Tool-call JSON and streaming tool calls parse correctly. A repeated 3136-token
  prefix records a cache hit and reduces request latency from 1.965 to 1.152 s.
- The original acceptance run contains one unmatched no-MTP/FP16-KV sample
  that reached the 8192-token request limit and began repeating after about
  6377 tokens. Its request seed was not retained, so that artifact is neither
  proof of a deterministic kernel defect nor sufficient evidence to dismiss
  the failure as sampling variance.
- A matched audit at source `7f409a7727` used the 74-token macOS prompt,
  seed `20260620`, `temperature=1.0`, `top_p=0.95`, `top_k=20`, and a
  32600-token output allowance. All four production combinations stopped
  naturally with complete HTML/CSS/JavaScript and closed code fences:

  | MTP | KV cache | Output tokens | Decode tok/s | Result |
  | --- | --- | ---: | ---: | --- |
  | off | FP16 | 24,441 | 61.63 | pass |
  | off | E5M2 | 25,055 | 61.71 | pass |
  | MTP4 | FP16 | 25,753 | 75.21 | pass |
  | MTP4 | E5M2 | 18,642 | 81.82 | pass |

  Repeated same-seed requests were byte-identical in every combination.
  Prefix-cache-hit probes were also byte-identical for no-MTP/FP16,
  MTP4/FP16, and MTP4/E5M2. The MTP4/E5M2 sample initially tripped the local
  repeated-window heuristic because fourteen decorative `=====` section
  separators contributed overlapping single-character windows; 200-character
  windows did not repeat. The gate now leaves homogeneous runs to its separate
  same-character-run check.
- The corrected FP8 workspace ABI passes a 6278-token natural prompt: it
  follows the final four-line instruction, emits no replacement characters,
  and stops naturally after 47 tokens.

## Wheel And Compatibility

- Wheel: `1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl`.
- SHA256: `e4ffba3bf61e16dcba0eeeafac4036b7cead5abe047e4474ecacb7e5abb35da3`
  (75,765,498 bytes).
- A cloned environment imports vLLM 1.3.0 from site-packages with Torch
  2.10.0+cu128. `_moe_C.moe_permute_sort_workspace_size` is present.
- The wheel contains `_C`, `_C_stable_libtorch`, `_moe_C`, Flash-V100,
  vLLM FA2, and `flash_qla_sm70_gdn_strided.so`.
- A clean-cache MTP4 plus FP8-KV boundary run does not create
  `TORCH_EXTENSIONS_DIR`, proving FlashQLA does not invoke NVCC at runtime.
- The 2026-08-17 release candidate was packaged from the accepted SM70 build
  at source `383bbdd166`, installed into a fresh Python 3.12 environment, and
  served Qwen3.8-27B-FP8 TP4 with the `fp8` shorthand resolving to E5M2. In
  both no-MTP and MTP4
  lanes, repeated fixed-seed greedy responses were byte-identical and official
  sampling stopped naturally without replacement characters. MTP's first
  request may still JIT Triton helper kernels; this is a cold-latency concern,
  not a numerical-output difference.
- Qwen3.6-35B-A3B-AWQ TP2 final-wheel checks pass. Official 1K/256 and
  4K/1024 pure decode are 95.63 and 95.29 tok/s (10.457 and 10.494 ms TPOT),
  with complete, non-corrupted outputs and the TurboMind AWQ MoE route.
- No Qwen3.6-35B-A3B-FP8 checkpoint is available on this host. FP8 35B speed
  remains unmeasured and must not be inferred from AWQ or 27B results.

## Evidence

Artifacts are rooted at
`/data/minimax-h3/task-cache/qwen38-130-acceptance-20260815`. Retained groups
are `context/`, `concurrency/`, `mtp/`, `correctness/`, `profiles/nsys/`,
`compat/`, `build/final-wheel/`, and `logs/`.

The corrected FP8 prefill implementation, A/B results, and retained profiles
are documented in `docs/design/sm70_fp8_long_prefill_exact_dense.md`.
The pointer-fix and concurrency artifacts are rooted at
`/data/minimax-h3/task-cache/qwen38-130-concurrency-latest-20260815`.
