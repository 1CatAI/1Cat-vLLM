# 1Cat-vLLM 1.2.2

1Cat-vLLM 1.2.2 is a V100/SM70 release focused on Qwen3.5/Qwen3.6
long-context serving, FP8 KV cache correctness, and MTP decode performance.
It supersedes 1.2.1 and includes the release work after baseline commit
`ce9e61bf5`.

## Highlights

### Flash-V100 long-context prefill and decode

- Reworked the exact SM70 D=256 paged-prefill path from the original BM16
  implementation to BM32 phase reuse, all-P scheduling, and conflict-reduced
  pair scratch. The production kernel reuses K/V operands across top and
  bottom M16 panels, retains FP32 accumulation order, and remains bitwise exact
  on normal, reversed, and tail page tables.
- At `M=1024,N=65536,Hq=6,Hkv=1,D=256,page=784`, the accepted production
  operator falls from `43.605 ms` to `31.717 ms` (`-27.26%`). Shared bank
  conflicts fall by 37.50% in the final pair-scratch step, with unchanged
  tensor-instruction count, 64 registers per thread, zero local spill, and two
  resident CTAs per SM.
- Qwen3.6-27B-AWQ TP4 64K full-model prefill falls from the locked
  `47.9785 s` reference to `38.7454 s` through the kernel changes
  (`-19.24%`). Raising the production chunk to 8,096 tokens reduces it further
  to `33.0984 s`, for a cumulative `-31.01%` prefill-latency reduction and
  approximately `+44.96%` prefill throughput.
- Added an FP8 E5M2 bridge that expands each paged K/V value once into a reused
  FP16 workspace instead of repeating page lookup and conversion in every
  query CTA. The same bridge supports complete and partial BM32 query tiles.
- Added graph-safe FP8 E5M2 XQA decode. It removes six-way KV rereads for the
  Qwen G6 shape by launching per KV head rather than per query head and keeps a
  scalar crossover fallback for short non-graph requests.
- On the matched Qwen3.6-27B-AWQ TP2 FP8-KV route, the 64K prefill changes
  `127.842 -> 62.964 s` (`-50.75%`) and TPOT changes
  `32.768 -> 26.910 ms`, increasing decode throughput by `21.77%`. At 16K,
  prefill improves `29.72%` and decode throughput improves `10.39%`.
- Flash-V100 keeps graph-safe dynamic active-partition metadata, so long
  decode scans the live KV length instead of replaying stale capture-time
  partition counts. FP16 paged decode, FP8 XQA, MTP small-query XQA, and scalar
  fallback remain separate, explicitly reported routes.

### Flash-V100 long-context correctness

- Fixed corrupt output at the exact 256K model-length boundary when using FP8
  E5M2 KV cache. The FP8-to-FP16 bridge now zeroes only the masked tail up to
  the next 16-token WMMA boundary, preventing stale NaN/Inf values from
  entering the PV HMMA path.
- The production-shaped `input_len=262120`, `max_model_len=262144` checks now
  return finite hidden states and coherent nonzero tokens with both no-MTP and
  MTP4.
- The repaired bridge preserves the fast route. In the recorded 27B AWQ TP4
  boundary check, bridged prefill completed in `252.49 s` versus `800.61 s`
  with the bridge disabled.

### Full-Flash MTP on V100

- The SM70 MTP profile now defaults the drafter attention backend to
  `FLASH_ATTN_V100` instead of `TRITON_ATTN`, unless explicitly overridden.
- Added graph-safe Flash-V100 drafter metadata and dedicated small-query paths
  for the first multi-token draft step and later single-token draft steps.
- The target M=5 verifier uses the exact Flash-V100 XQA route where the runtime
  shape is supported.
- Qwen3.6-27B-AWQ TP4 MTP4 with FP8 KV reaches:

  | Context | No-MTP | 1.2.2 MTP4 | MTP speedup |
  | --- | ---: | ---: | ---: |
  | 64K | `57.121 tok/s` | `100.564 tok/s` | `+76.05%` |
  | 128K | `45.438 tok/s` | `85.258 tok/s` | `+87.64%` |
  | 261888 | `32.858 tok/s` | `49.772 tok/s` | `+51.47%` |

- Acceptance remains `99.52%-100%` at these long-context points, with stable
  repeated token hashes and no corruption.
- The final 128K row includes the exact P1024 dual-CTA verifier below. The 64K
  and 261888 rows retain the accepted full-Flash endpoint measurements because
  those two endpoint lengths were not rerun after the dual-CTA promotion.

### Exact-M5 dual-CTA verifier

- Added a P1024, 192-thread, two-CTA specialization for the TP4 exact
  `M=5,G6,D256` FP8-KV verifier. It preserves the reduction order and is
  bitwise equal to the previous P1024 implementation.
- Registers fall from 186 to 168 per thread, occupancy rises from 12.49% to
  18.67%, and eligible warps per scheduler rise from 0.34 to 0.58.
- At 128K, Qwen3.6-27B-AWQ TP4 improves from `75.298` to
  `85.258 tok/s` (`+13.23%`) on top of the full-Flash MTP path.
- The same quantization-independent route improves Qwen3.6-27B-FP8 TP4 from
  `74.306` to `83.864 tok/s` (`+12.86%`).
- TP2 uses the same G6 gate. Qwen3.6-27B-FP8 TP2 improves from `43.925` to
  `53.693 tok/s` (`+22.24%`) at 128K.
- Qwen3.6-35B-A3B uses G8 on TP2 and replicated-KV G4 on TP4, so this exact G6
  specialization does not apply to 35B.

### MTP4 release baseline

The primary 1.2.2 MTP4 baseline is Qwen3.6-27B-AWQ, TP4 on four V100 GPUs,
TurboMind AWQ, FP8 E5M2 KV, 256K model length, 8,096-token chunking, prefix
caching, Mamba align, official sampling, target and drafter
`FLASH_ATTN_V100`, dynamic draft vocabulary, fused proposal, local argmax,
CUDA graphs, and no eager execution.

| Workload | TPOT | Decode | Acceptance length / rate | Quality gate |
| --- | ---: | ---: | ---: | --- |
| Natural macOS application, 26,708 output tokens | `8.426 ms` | `118.674 tok/s` | `3.92` | natural stop, complete output |
| 64K fixed long-context gate | `9.944 ms` | `100.564 tok/s` | `4.981 / 99.52%` | deterministic, no corruption |
| 128K fixed long-context gate | `11.729 ms` | `85.258 tok/s` | `4.981 / 99.52%` | deterministic, dual-CTA |
| 261888 long-context gate | `20.092 ms` | `49.772 tok/s` | `5.000 / 100%` | deterministic, no corruption |

The supplementary TP2 baseline uses Qwen3.6-27B-FP8 weights plus FP8 KV. At
128K, MTP4 records `18.624 ms`, `53.693 tok/s`, and acceptance
`4.942 / 98.56%`. The normal TP2 natural-output Flash-drafter baseline is
approximately `102.724 tok/s` with acceptance `3.910`; it is retained as the
TP2 natural workload reference rather than substituted for the TP4 release
baseline.

### MTP and hybrid-state safety

- Stops creating speculative placeholders when the drafter reaches its model
  length and skips stale async acceptance correction when the previous row had
  no drafts.
- Makes dynamic-vocabulary top-20 reduction robust to NaN/Inf candidates and
  guarantees a valid normalized proposal token.
- Tightens DDTree payload typing, state-slot selection, GDN metadata, mixed
  speculative batching, and explicit-root versus rootless tree replay.
- Keeps quality-sensitive verifier and proposal paths exact. Rejected variants
  that changed reduction order or caused long-output repetition remain
  default-off.

### FP8 KV and model coverage

- Extends FP8 E5M2 KV bridge coverage across TP4 Qwen3.6-27B, Qwen3.6-35B-A3B,
  and Qwen3.5-27B-NVFP4 runtime shapes.
- Allows compressed-tensors checkpoints without an explicit `kv_cache_scheme`
  to use the SM70 unit-scale E5M2 route. Explicit checkpoint KV scale schemes
  remain rejected.
- Production no-MTP decode checks include:

  | Model route | 16K TPOT | 64K TPOT | Additional coverage |
  | --- | ---: | ---: | --- |
  | Qwen3.6-27B AWQ, TP4 | `14.280 ms` | `17.507 ms` | 128K and 261888 |
  | Qwen3.6-27B FP8, TP4 | `16.018 ms` | `19.318 ms` | 128K and 261888 |
  | Qwen3.6-35B-A3B AWQ, TP4 | `9.290 ms` | `11.418 ms` | matched no-MTP |
  | Qwen3.6-35B-A3B FP8, TP4 | `9.204 ms` | `11.320 ms` | matched no-MTP |
  | Qwen3.5-27B NVFP4, TP4 | `14.466 ms` | `18.022 ms` | TurboMind NVFP4 |

- FP8 KV decode overhead at 64K remains within 0.49%-1.16% of matched FP16 KV
  across the accepted model matrix.

### TurboMind SM70 routes

- Keeps TurboMind as the production SM70 path for AWQ, FP8 weights, NVFP4,
  MXFP4, and applicable MoE layers.
- Includes exact small-M AWQ scheduling, row-projection selectors, and MTP
  verifier integration developed for TP2 and TP4.
- The release evidence does not credit Marlin for TurboMind benchmark results.
  Explicit backend selection remains available through
  `VLLM_SM70_QUANT_BACKEND`.

### Self-contained FlashQLA packaging

- Vendors the tested FlashQLA Python and SM70 CUDA sources in the top-level
  `flash_qla` package and retains the upstream MIT license.
- The wheel includes all FlashQLA modules plus the JIT-required
  `gdn_forward.cu`; no external FlashQLA checkout is required.
- First use of the SM70 FlashQLA JIT path still requires a compatible local
  CUDA toolkit and C++ compiler.

### Release validation and tooling

- Release quality checks use the model's official sampling settings and
  require natural `finish_reason=stop` for long-output quality evidence.
- The corrected macOS HTML/code quality run naturally stopped after 26,708
  tokens, remained structurally complete, and reached `118.674 tok/s` with
  aggregate acceptance length `3.92`.
- Added release-matrix, quality, route, prefix-cache, attention exactness, and
  MTP safety checks.
- Cleared all tracked Python Ruff diagnostics. The complete 1.2.2 change set
  passes Ruff check/format, typos, clang-format, markdownlint, mypy, SPDX,
  forbidden-import, configuration, and attention-backend documentation hooks.

## Experimental Paths

- `FLASHINFER_SM70` remains explicit-only. Its exact split-KV prefill route is
  promoted only for the measured 27B TP4 FP16-KV shapes and is not a default
  replacement for `FLASH_ATTN_V100`.
- BFLA sparse prefill remains approximate and default-off.
- Global P256 dual-CTA MTP verification remains rejected because its changed
  reduction order caused long-output repetition.

## Known Limitations

- The available Qwen3.5-27B-NVFP4 checkpoint still reports a fused Q/K/V
  global-scale warning. Production output-health checks pass, but the warning
  remains a checkpoint-quality risk rather than a resolved quantization issue.
- No valid MTP body exists in the tested 27B NVFP4 checkpoint, and no accepted
  35B NVFP4 MTP baseline is claimed.
- The exact-M5 dual-CTA specialization currently requires the G6 FP8-KV shape;
  unsupported verifier shapes use their existing exact fallback.

## Build Target

The 1.2.2 release wheel is built for Python 3.12, CUDA 12.8, Torch 2.10, and
SM70/V100. The wheel bundles Flash-V100, paged-KV utilities, vLLM CUDA
extensions, SM70 TurboMind kernels, and the FlashQLA source package.
