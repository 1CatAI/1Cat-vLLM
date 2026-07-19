# 1Cat-vLLM 1.2.2

1Cat-vLLM 1.2.2 improves V100/SM70 long-context attention, FP8 KV cache,
and Qwen3.6 MTP4 performance. It supersedes 1.2.1.

## Highlights

### Flash-V100 long-context attention

- Replaced the D=256 paged-prefill BM16 path with exact BM32 phase reuse,
  all-P scheduling, conflict-reduced pair scratch, and 8,096-token chunking.
- Added one-pass FP8 E5M2-to-FP16 KV bridging and graph-safe FP8 XQA decode.
- Preserved softmax, FP16 probability rounding, FP32 accumulation order, and
  exact output on the accepted prefill paths.
- Fixed stale CUDA Graph partition metadata so decode scans the live KV length.

| Matched path | Before | 1.2.2 | Change |
| --- | ---: | ---: | ---: |
| FP16 paged-prefill operator, 64K | `43.605 ms` | `31.717 ms` | `-27.26%` |
| 27B-AWQ TP4 full-model prefill, 64K | `47.9785 s` | `33.0984 s` | `-31.01%` |
| 27B-AWQ TP2 FP8-KV prefill, 64K | `127.842 s` | `62.964 s` | `-50.75%` |
| 27B-AWQ TP2 FP8-KV TPOT, 64K | `32.768 ms` | `26.910 ms` | `+21.77%` decode throughput |

### MTP4 release baseline

Primary configuration: Qwen3.6-27B-AWQ, TP4 on four V100 GPUs, TurboMind,
FP8 E5M2 KV, 256K model length, 8,096-token chunks, prefix cache, Mamba align,
official sampling, Flash-V100 target and drafter, CUDA Graph, and no eager mode.

| Workload | No-MTP | MTP4 | Acceptance | MTP gain |
| --- | ---: | ---: | ---: | ---: |
| Natural 26,708-token output | - | `118.674 tok/s` | `3.92` | quality pass |
| 64K | `57.121 tok/s` | `100.564 tok/s` | `4.981 / 99.52%` | `+76.05%` |
| 128K | `45.438 tok/s` | `85.258 tok/s` | `4.981 / 99.52%` | `+87.64%` |
| 261888 | `32.858 tok/s` | `49.772 tok/s` | `5.000 / 100%` | `+51.47%` |

The exact P1024 dual-CTA verifier contributes a further `+13.23%` at 128K
on TP4 AWQ and `+12.86%` on TP4 FP8 weights. The same G6 route improves the
TP2 FP8-weight baseline by `+22.24%` to `53.693 tok/s` at 128K.

### Correctness and stability

- Fixed FP8 KV output corruption at `input_len=262120` near the 256K limit by
  zeroing masked bridge rows up to the next 16-token WMMA boundary.
- Stops drafting at the drafter model limit and skips stale async acceptance
  correction when no draft was produced.
- Makes dynamic-vocabulary top-20 proposal robust to NaN/Inf candidates.
- Tightens DDTree payload, GDN state-slot, mixed-batch, and tree-root handling.
- Long-output quality evidence now requires natural `finish_reason=stop`.

### Supported release matrix

| Model | Weight route | TP | FP8 KV | MTP4 |
| --- | --- | ---: | --- | --- |
| Qwen3.6-27B | AWQ | 2/4 | supported | supported |
| Qwen3.6-27B | FP8 | 2/4 | supported | supported |
| Qwen3.6-35B-A3B | AWQ | 2/4 | supported | supported fallback |
| Qwen3.6-35B-A3B | FP8 | 2/4 | supported | supported fallback |
| Qwen3.5-27B | NVFP4 | 4 | supported | no valid MTP weights |

FP8 KV decode overhead at 64K is within 0.49%-1.16% of matched FP16 KV across
the accepted matrix. TurboMind remains the production SM70 quantization path.

### Packaging and validation

- The wheel now includes Flash-V100, paged-KV utilities, vLLM CUDA extensions,
  TurboMind SM70 kernels, and the complete FlashQLA Python/CUDA source package.
- FlashQLA no longer requires an external source checkout. Its first JIT build
  still requires a compatible CUDA toolkit and C++ compiler.
- All tracked Python files pass Ruff. The release change set also passes
  formatting, typos, mypy, SPDX, configuration, and backend documentation
  checks.

## Known Limitations

- The tested 27B-NVFP4 checkpoint retains its fused Q/K/V global-scale warning.
- No 27B-NVFP4 or accepted 35B-NVFP4 MTP performance is claimed.
- The exact dual-CTA verifier currently targets the G6 FP8-KV shape; other
  shapes use exact fallbacks.
- `FLASHINFER_SM70` and BFLA sparse prefill remain explicit experimental paths.

## Build Target

Python 3.12, CUDA 12.8, Torch 2.10, and SM70/V100.
