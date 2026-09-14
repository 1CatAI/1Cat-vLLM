# SM70 Q8000/Q8192 integration validation (2026-09-14)

The first integration measurements below are retained as historical evidence
and explicitly identify their eager configuration. Follow-up validation from
the E4M3 route-parity work uses normal CUDA graphs. The current
`benchmark_sm70_79t_cold.py` fixes `enforce_eager=False`; future results from
this benchmark must not be compared with an eager run without labeling the
mode difference.

## Contract

Base `7217bb5d4f3866f87bf6a961204c894af3b03261`; optimized kernel
`359ae7c30a`. The implementation is built into the normal
`vllm.vllm_flash_attn._vllm_fa2_C` extension. It has no private DSO or
preload dependency. SM70 FA2 builds and the admitted Q8000 runtime route
enable it by default; explicit build and runtime switches retain rollback.

The test host uses Python 3.12, Torch 2.10.0+cu128, CUDA 12.8, and four
V100-SXM2-32GB GPUs (physical 4-7 in PCI order) at TP4. The model is
Qwen3.8-27B-FP8 (`Qwen3_5ForConditionalGeneration`, 64 layers, 16
full-attention layers, 24 query heads / 4 KV heads / D256). Weights are FP8,
compute is FP16, KV storage is E4M3, the quantization backend is TurboMind,
and the attention backend is FLASH_ATTN_V100.

End-to-end runs use max length 262144, chunk size 8000, one sequence, memory
utilization 0.85, FP16 Mamba state/cache, eager execution, CUDA graphs off,
MTP off, and prefix caching off. Sampling is greedy with a maximum of 32
output tokens and EOS respected. Engine initialization and a short warmup
are outside TTFT. A deterministic natural-language prompt places a marker
at its beginning, then asks for that marker and the largest Solar System
planet.

Run `benchmarks/benchmark_sm70_79t_cold.py --model "$MODEL" --lengths 16000
128000 256000 --output-len 32 --out "$RESULT"` with the runtime flags in
README, plus `VLLM_SM70_QUANT_BACKEND=turbomind`,
`VLLM_FLASH_V100_PREFILL_USE_TRITON=0`,
`VLLM_FLASH_V100_FP8_PREFILL_BRIDGE=1`,
`VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1`, and `VLLM_USE_AOT_COMPILE=0`.
Set V37=0 for the architecture route and V37=1 for the matched control.

## Failure mechanism and final guard

The raw historical recipe assumes that unshifted exponentials and the FP16
PV numerator fit in FP16. They do on zero-mean random tensors, but they do
not on the model tensors. All captured Q/K/V inputs were finite while the
raw output contained infinities and the model emitted invalid token ID -1.

The first short-request failure occurred on rank 3, attention call 4. A
later candidate with 16x value headroom remained finite through KV144K, then
overflowed independently on ranks 2 and 3 at KV152K. Sparse score maxima
missed by the sampled shift were 15.31-16.00 above the sample. The largest
observed FP16 PV partial was 75310, beyond the FP16 finite limit 65504.

The qualified recipe uses FP16 Tensor Core operands with FP32 MMA
accumulation and applies four guards:

- sample one score in eight, add a 4.0 shift margin, and cap positive
  exponent input at 10.0;
- subtract a per-dimension V center when the first-4096-token mean magnitude
  is at least 0.05;
- scan all residual V values and scale them by an exact power of two with
  64x additional headroom;
- write each 24K prefix numerator block in FP32 and keep block masses and the
  online prefix/tail merge in FP32, then restore the V center after
  normalization.

The 64x headroom makes all three captured failure tensors finite. It is a
model-qualified bound, not a proof over arbitrary FP16 inputs. The
128x256/64x64 PV topology has no local-memory spills in the admitted prefix
kernel. Expanding the prefix block from 8K to 24K reduces the number of FP32
online merges while keeping the score workspace within the 32-GiB V100
end-to-end budget.

## Operator result

The following medians use the final source-built artifact, 30 warmups and
100 CUDA-event samples on one V100. Useful causal FLOPs are
`4*Hq*D*(Q*(KV-Q)+Q*(Q+1)/2)`.

| KV tokens | Median (ms) | p10 (ms) | p90 (ms) | TFLOP/s | Relative L2 | Worst-row relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128000 | 79.9700 | 79.6289 | 80.4487 | **76.2145** | 0.002364 | 0.003096 |
| 256000 | 164.3465 | 163.5427 | 165.1093 | **75.3672** | 0.002317 | 0.003070 |

Both medians clear the 75-TFLOP/s target. Relative L2 improves by about 3x
from the guarded FP16-accumulator result while median throughput remains in
the same 75-TFLOP/s band.

## Cold end-to-end result

| Prompt tokens | TTFT (s) | Prompt tokens / TTFT | Full wall (s) | Subsequent decode (tok/s) |
| ---: | ---: | ---: | ---: | ---: |
| 256000 | 94.40737 | **2711.65** | 95.76525 | 11.0468 |

This is a single unprofiled cold request, not a confidence interval. TTFT
includes prefill and first-token overhead. Subsequent decode uses the 15
intervals after the first token. `cached_tokens` is zero. All four TP ranks
record 496 architecture calls and 496 E4M3 bridge calls, and the loaded FA2
module is the source-built worktree artifact. An independent immediately
preceding run measured 94.51140-second TTFT and 2708.67 prompt tok/s.

The request emits the same 16 tokens as the guarded FP16 build, stable FP32
control, and matched v37 control, including EOS:

```text
119920 96919 95761 12512 96143 97460 115783 10119
3709 145551 99960 114931 95761 147482 1710 248046
```

The decoded output is `校验词是「海蓝石榴」，太阳系最大的行星是木星。`.
Both retrieval and knowledge checks pass. The token sequence also matches
the stable FP32 implementation and matched v37 control. This is a scoped
long-context text-health gate, not broad model equivalence.

## Matched comparison

| Route | 16K TTFT / tok/s | 128K TTFT / tok/s | 256K TTFT / tok/s |
| --- | ---: | ---: | ---: |
| Qualified FP32 MMA + FP32 block output | - | - | **94.4074 / 2711.65** |
| Guarded FP16 MMA accumulator | 3.3187 / 4821.17 | 36.0745 / 3548.21 | 94.5881 / 2706.47 |
| Earlier stable FP32 diagnostic | 3.3608 / 4760.75 | 40.0996 / 3192.05 | 111.3729 / 2298.58 |
| v37 control | 3.3294 / 4805.64 | 38.7531 / 3302.96 | 105.6480 / 2423.14 |

At 256K, the qualified FP32 route has 10.64% lower TTFT and 11.91% higher
prompt throughput than the matched v37 control. It has 15.23% lower TTFT and
17.97% higher prompt throughput than the earlier spill-heavy stable FP32
implementation. It is 0.19% faster than the guarded FP16 route in this
single-run comparison while cutting the operator's relative L2 by about
threefold.

The earlier 1032.23-second report did not select the intended architecture
and bridge. The optimized route reduces that wall-clock anomaly by about
10.8x. The reported 62.73-second second request hit prefix cache and is not
a cold-prefill result.

## Broader NVFP4 plus DFlash2 serving sample

A separate serving-quality run checked whether the guarded attention build
remains healthy in the Qwen3.8-27B-NVFP4 plus DFlash2 stack. It predates the
FP32-accumulation change. The selected prompts are too short to enter the
Q8000 architecture route, and the change does not affect decode, so rerunning
them would not exercise the new code. This is a compatibility and
output-health result, not a matched comparison with the FP8 target-only
cold-prefill contract above. It used TP4, E4M3 target KV,
Flash-V100 for target and draft attention, seven probabilistic draft tokens,
four concurrent requests, max length 262144, and a 65536-token output cap.
Sampling used temperature 0.6, top-p 0.95, top-k 20, seed 0, and xhigh
reasoning. The selected 96-case corpus SHA256 is
`46fcb5e990bfeb01069b9d676f87285e5672edcb8557eeada98d0a35d8b9af1e`.

The run was stopped after 78 complete cases at the operator's request. The 18
unstarted or interrupted AIME cases are excluded from every score:

| Suite | Complete / selected | Raw pass | Output tokens | Suite wall | Aggregate output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro, category-balanced | 32 / 32 | **29 / 32** | 134370 | 1044.38 s | **128.66** |
| MBPP sanitized | 32 / 32 | **31 / 32** | 101776 | 522.75 s | **194.69** |
| AIME 2024/2025 | 14 / 32 | **13 / 14** | 183238 | interrupted | not reported |
| Total complete | 78 / 96 | **73 / 78** | 419384 | mixed | not reported |

All 78 complete requests reached natural EOS, returned a nonempty final
answer, and contained no replacement characters. None hit the output cap.
The service log contains no NaN, Inf, overflow, traceback, CUDA error, or dead
engine/worker report. The three MMLU-Pro misses and one completed AIME miss
are coherent wrong answers rather than malformed output. The only raw MBPP
failure is task 229 (`mbpp:102`): its prose requires stable order within both
sign groups, while its first public assertion moves the value 2 behind values
4, 5, and 6. The model follows the prose, so this is retained as a raw failure
but classified as a dirty evaluation case.

Across all active ten-second service windows, including smoke cases and the
four interrupted requests, median aggregate generation throughput was 141.4
tok/s and p90 was 218.72 tok/s. Median DFlash2 acceptance length was 4.08 and
p90 was 4.62; median draft-token acceptance was 44.0%. The completed MMLU-Pro,
MBPP, and AIME output lengths reached 46458, 20699, and 30371 tokens. These
long natural-EOS traces show that the stack remains coherent deep into decode,
but they also expose costly xhigh overthinking tails.

The selected dataset prompts contain only 125-713 tokens. Setting max length
to 262144 verifies service capacity and long-decode compatibility; it does not
exercise the Q8000 architecture prefill route or constitute another near-256K
cold-prefill measurement. The 256K speed claim remains the matched target-only
result in the preceding section. A combined NVFP4 plus DFlash2 near-256K cold
request would be a separate acceptance item.

## Numerical gates and rejected variants

- Four focused architecture route-policy tests pass after the FP32 change.
  The broader preceding integration run had 18 route/bridge policy tests
  pass.
- Three SM70 CUDA regressions pass. Two cover KV16K/KV128K large random
  scores with values biased by +8. The third places correlated score/value
  spikes at a fixed nonzero residue to reproduce numerator growth missed by
  sparse max sampling. All compare sampled rows with full-KV FP32 attention.
- Seven real failure captures are finite after the final guard. The four
  early captures have sampled relative L2 0.002513, 0.001059, 0.001085, and
  0.001019. The first model-invalid capture is 0.001415. The two KV152K
  overflow captures are 0.003546 and 0.001385, with worst-row relative L2
  0.020091 and 0.007039. The guarded FP16 build measured 0.014670 and
  0.006221 on those last two captures.
- The raw recipe reached about 81.5/80.5 TFLOP/s at KV128K/256K on random
  inputs but emitted invalid model output. It is rejected.
- Exact row maxima plus the earlier 32x64-warp FP32 PV reached only about
  54.1/54.0 TFLOP/s. A source-rebuilt stable 32x64-warp route reached about
  59 TFLOP/s because its 128-register cap caused heavy spills. Both are
  diagnostic baselines.
- The corrected 128x256/64x64 FP32 topology with exact scalar exponential and
  8K blocks reached 73.39 TFLOP/s at KV128K. FP32 block output with 16K blocks
  reached 75.19 TFLOP/s; 24K blocks provide the final margin.
- A stable half2 degree-5 exponential with 16x range reduction fell to 67.31
  TFLOP/s and raised relative L2 to 0.004941. Its repeated-square dependency
  chain is slower than the SM70 scalar exponential, so it is rejected.
- 4x value headroom failed the 16K model request. 16x passed 128K but failed
  at KV152K during the 256K request. Both are rejected.
- Increasing the score margin from 4 to 6 kept output finite but raised the
  three captured relative-L2 errors from 1.08%/1.47%/0.62% to
  3.91%/2.93%/1.25% because more weights lost FP16 dynamic range. It is
  rejected.

## Q8192 and concurrent-request expansion

The follow-up build keeps the same FP16 Tensor Core inputs and FP32 MMA
accumulation, and adds a native Q8192 specialization beside Q8000. Its 256-token
tail tiles cover all 8192 causal query rows without a residual. Q8001 through
Q8191 are leading-padded to Q8192 and the matching leading output is discarded;
this preserves bottom-right causal positions and adds at most 2.4% query work.

The default architecture admission is now FP16 Q/K/V/output,
Q8000 through Q8192, Hq6/Hkv1/D256, causal, scale 1/16, KV at least Q through
262144, and KV length aligned to 32 tokens. The paged-to-dense integration
has no single-request gate. It gathers and dispatches every eligible request in
the scheduler batch. The reusable gather and architecture workspaces remain
stream ordered, so requests share them without allocating one multi-gigabyte
workspace per sequence.

The 32-token KV alignment is also a correctness boundary for the prefix PV
Tensor Core K tile. Direct boundary probes at KV=Q+8/Q+16/Q+24 stayed finite
but had approximately 72%/48%/24% sampled relative L2 error; KV=Q and Q+32
returned to roughly 0.001% and 0.034%. The earlier alignment-one performance
probe was therefore insufficient to admit these shapes. Lengths not divisible
by 32 retain the general attention route.

Single-V100 operator medians below use 30 warmups and 100 CUDA-event samples
from the final Q8192 build:

| Q | KV | Median (ms) | TFLOP/s | Relative L2 | Max abs | Worst-row relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 128000 | 81.3069 | **76.7010** | 0.002387 | 0.0000606 | 0.003918 |
| 8192 | 128032 | 82.4936 | **75.6171** | 0.002350 | 0.0000625 | 0.004003 |
| 8192 | 256000 | 167.2011 | **75.8294** | 0.002607 | 0.0000794 | 0.003297 |
| 8192 | 262144 | 171.9081 | **75.5520** | 0.002290 | 0.0000390 | 0.003106 |

The Q8000 regression matrix remains in the same band: 76.27/75.54/75.77/75.83
TFLOP/s at KV128000/128032/256000/262144. Padded Q8064 and Q8191 remain above
76 TFLOP/s at KV128K and KV256K. The worst padding ratio, Q8001, reports
74.31/74.56 useful TFLOP/s because useful FLOPs exclude the 191 padded rows;
the native Q8192 work itself remains in the qualified band. All outputs are
finite, with sampled relative L2 around 0.23%-0.26%.

Calling one, two, and four independent Q8192/KV128000 operations from one
batch-like caller stream took 80.8366, 161.8506, and 325.2469 ms. Aggregate
throughput was **77.15, 77.06, and 76.70 TFLOP/s**, and every output was finite.
A single long-attention operation already saturates V100, so safe stream-ordered
sharing preserves aggregate throughput rather than trying to overlap several
workspace-heavy kernels on the same GPU.

A TP4 Qwen3.8-27B-FP8 cold request with Q8192 chunks, E4M3 KV, prefix caching
off, and a 256000-token prompt measured 96.9743-second TTFT, 2639.88 prompt
tok/s, and 98.3537-second wall time. This is 2.65% lower prompt throughput than
the earlier Q8000 single run; it is a small single-run difference while the
matched operator medians remain above 75 TFLOP/s. Every rank recorded 480
native Q8192 architecture calls and 496 E4M3 bridge calls. It returned exactly
the same 16 token IDs as Q8000 and the stable control, including EOS:

```text
119920 96919 95761 12512 96143 97460 115783 10119
3709 145551 99960 114931 95761 147482 1710 248046
```

The decoded answer is `校验词是「海蓝石榴」，太阳系最大的行星是木星。`;
both retrieval and knowledge checks pass, `cached_tokens` is zero, and the log
has no NaN, Inf, overflow, OOM, CUDA error, or worker failure.

A follow-up TP4 full-model gate verifies that the same architecture is reached
from an NVFP4/compressed-tensors checkpoint with E4M3 KV and normal
`FULL_AND_PIECEWISE` CUDA graphs. With Q8192 chunks, maximum length 262144,
one live request, prefix caching off, and no speculative decoding, the 256000-
token cold request measures 102.7519-second TTFT and **2491.44 prompt tok/s**.
It returns the same complete 16-token answer above; both quality checks pass
and `cached_tokens=0`. Its 15 token intervals are too short for a decode-speed
claim. A separate 256000-input/256-output fixed run measures 255 intervals in
5.3902 seconds, or **47.308 tok/s** and **21.138 ms TPOT**. Its TTFT is
102.9135 seconds and uncached prefill is 2487.53 tok/s. Every rank records 480
native Q8192 calls and 496 E4M3 bridge calls. The final route summary records
48 dynamic page-800 E4M3 XQA decode calls per rank. Neither run enables eager
mode.

For concurrent full chunks, 8192 is a per-request scheduling threshold rather
than the total batch limit. Two chunks use `max_num_batched_tokens=16384`,
`max_num_seqs=2`, and `long_prefill_token_threshold=8192`; larger total token
budgets can admit more requests. Two simultaneous cold 128000-token prompts
completed with a 74.2707-second batch TTFT and **3446.85 aggregate prompt
tok/s** over 256000 input tokens. Both requests had zero cached tokens and
returned the same 16-token answer shown above. Every TP rank recorded 448
native Q8192 calls and 480 E4M3 bridge calls.

That run also exposed and closed a pre-existing E4M3 batch-decode planner
mismatch: the Python wrapper could select a 1024-token partition for B2 at long
context while the native batch XQA kernel accepts 64, 128, or 256. The wrapper
now caps its automatic B2-B16 plan at 256 and rejects an incompatible explicit
override. The complete two-request run uses the repaired automatic plan rather
than a benchmark-only partition override.

The final FA2 artifact SHA256 is
`9f55da1ae54d87b008cb3452e23e1e4b272dd66115d37620d63674f3841c0055`.

## Original Q8000 artifact identity and promotion decision

The original Q8000 end-to-end-qualified formatted FA2 SHA256 was:
`2e88f8c0fa177ab64c19fe0419a311a847aa10c9825eb6c94bf150e5f9c7c049`.
ELF dependencies are standard Torch/CUDA/cuBLAS/system libraries. Build,
pre-commit, route-policy tests, CUDA numerical tests, operator benchmark,
and cold model runs all use the owned worktree. Raw logs and captured model
tensors remain in task-local `.artifacts` and are deliberately not committed.

Before default promotion, the branch was synchronized with `main` at
`6def188c21`. Reconfiguring the SM70 build after removing the cached
`VLLM_SM70_79T_PREFILL` value selected `ON` without an explicit option. The
rebuilt FA2 SHA256 is
`5a57d7349b65896d8d01f0f07a3f3c196ec3f06708be7bf718ea0f71ac774687`.
With 30 warmups and 100 measurements, this default-built artifact reaches
76.3245 TFLOP/s at KV128K and 75.6356 TFLOP/s at KV256K. Both outputs are
finite; sampled relative-L2 errors remain 0.23644% and 0.23166%. Seven focused
default/route tests and all three SM70 overflow regressions pass against this
promotion state.

The expanded Q8000/Q8192 route remains restricted to the tensor and causal
contract above, but no longer has a single-request gate or an 8000-token KV
step. It is the default long-prefill architecture on SM70 builds for admitted
shapes; all other shapes retain their existing attention fallback. Extending
the architecture to other head layouts or unaligned prefix PV tiles requires
separate numerical and end-to-end qualification because FP16 score/probability
storage remains approximate even though PV accumulation and prefix output are
FP32.

## B2-B32 CUDA-graph serving promotion gate

The final TP4 NVFP4/E4M3 server uses normal `FULL_AND_PIECEWISE` CUDA graphs,
prefix caching off, 262144 maximum context, 65536 batched tokens, and 32
maximum sequences. The no-MTP default full-graph capture set is expanded from
`[1, 2, 4, 8, 16]` to `[1, 2, 4, 8, 16, 32]`. This removes the pre-existing
B16 capture ceiling. It does not add an attention batch ceiling: larger
scheduler batches retain the existing piecewise graph and the same E4M3
attention routes.

Standard `vllm bench serve` exact-2048-input/exact-256-output rows complete
with zero failures:

| Concurrency | Median TTFT | Pure decode* | Median/P90 ITL | Output TPS | Median request wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| C2 | 0.8554 s | 115.725 tok/s | 17.282 / 17.406 ms | 92.288 tok/s | 5.539 s |
| C4 | 1.6962 s | 223.730 tok/s | 17.879 / 18.038 ms | 151.222 tok/s | 6.761 s |
| C8 | 5.2214 s | 418.723 tok/s | 19.106 / 19.248 ms | 203.168 tok/s | 10.076 s |
| C16 | 10.7109 s | 708.450 tok/s | 22.585 / 22.837 ms | 248.859 tok/s | 16.451 s |
| C32 | 21.7727 s | **983.986 tok/s** | 32.521 / 32.824 ms | **272.868 tok/s** | 30.008 s |

`Pure decode` is concurrency times 1000 divided by pooled median ITL. All 62
requests generate the full 256 tokens, and a separate 32-request
natural-language burst passes both retrieval and knowledge checks on every
request. The B2/B4/B8/B16/B32 page-800 256K CUDA-graph operator matrix is
finite throughout, differs from scalar E4M3 by at most `4.77e-7`, and reaches
3.64x/6.12x/6.35x/6.46x/6.59x speedup, respectively.
