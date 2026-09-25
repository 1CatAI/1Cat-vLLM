# SM70 warmup fragmentation and prefill memory budget

This candidate reduces real allocations before changing KV admission. It is
not a claim that a 27B NVFP4 model with DFlash2 fits a single 32GB card at
GUM 0.93. Integration base: `d49e32b3587d4d34ffccb0ffd376e63974b06c88`.

## Changes

- Use the existing scoped `max_split_size_mb:20` allocator setting for the first
  SM70 compilation/warmup forward. Restore the original allocator configuration
  before the measured forward, KV allocation, graph capture and serving.
- For Flash-V100 hybrid Mamba layouts, shrink oversized sliding-window pages
  when their token count can be divided without changing dtype, window or the
  backend's multiple-of-16 block contract. Keep other backend overrides and
  explicitly padded attention specs unchanged. This avoids padding all Mamba
  states to the draft model's larger page. No TP or model-quantization gate is
  introduced.
- For NVFP4 large-chunk prefill, dequantize matching gate/up output columns in
  tiles, keeping the whole K reduction in each GEMM. Write SiLU products into
  their original output columns. Also tile non-gated projections whose dense
  FP16 weight is at least 128 MiB, so they can reuse smaller free segments.
  Both QPN2 and shared TurboMind weight layouts are supported. Small chunks
  below 4096 rows and caller-owned dense workspaces keep the original path.

Attention's FP16 operands and FP32 accumulation, FP8 target KV, FP16 draft KV,
and the 8192-token attention chunk are unchanged. cuBLAS may select a different
GEMM algorithm for tiled projections, so output equivalence is not bitwise and
requires a model quality gate before promotion.

## Reproduction contract

V100-SXM2-32GB, 300W limit, driver 580.173.02, CUDA compiler 12.8,
Torch 2.10.0+cu128, Python 3.12. Own source-built CMake and Flash-V100
extensions; no wheel, preload, private DSO, or external kernel overlay.

`Qwen3.8-27B-QUASAR-NVFP4-d8e6fbfa`, FP16 activations; DFlash2 revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, seven draft tokens, probabilistic
sampling. TP1, maximum two sequences, max length 65536, chunk 8192, attention
block 2048, Mamba checkpoint block 8192, align mode, prefix caching, text only.
V2 runner and FULL_AND_PIECEWISE CUDA Graph remain enabled. The capacity probes
explicitly use `VLLM_V2_CUDAGRAPH_MEM_MIB=512`, calibrated against actual C2
capture. This is an operator override, not a new automatic reserve default.
The default graph estimator and other models' graph budgets are unchanged.

The control is the source tree of main at the integration base. Allocator
attribution, launches, native SHA256 manifests, retained failures and benchmark
scripts are in the local `memory-budget-20260925` artifact collection. Do not
compare these 300W operator timings with the earlier 185W serving runs.

## Measured reductions

Warmup-only allocator scoping reduces retained inactive split storage from
1.443 to approximately 0.136 GiB. It does not eliminate the roughly 1.18 GiB
of actual native prefill workspaces. The first compiling forward can reserve
more releasable storage; the serving allocator is restored before measuring.

At TP1 with 64K configured context, the old uniform 8 MiB pages require 6 GiB
for KV admission. Keeping target/Mamba pages at 4 MiB and draft pages at 1024
tokens reduces the requirement to 4.28125 GiB. The draft remains FP16. CPU
checks cover TP1/TP2/TP4 layout capacities; they are not multi-GPU serving tests.

CUDA Graph replay microbenchmarks, M=8192 and K=5120:

| Gate/up local N | Control ms | Tiled ms | Latency change | Scratch saved |
|---:|---:|---:|---:|---:|
| 34816 (TP1 shape) | 35.213 | 34.242 | -2.76% | 676 MiB |
| 17408 (TP2 shape) | 17.010 | 17.075 | +0.38% | 234 MiB |
| 8704 (TP4 shape) | 8.539 | 8.515 | -0.28% | 14 MiB |

The TP1 down projection, M=8192/K=17408/N=5120, changes 16.245 to 16.362 ms
(+0.72%) and saves 84 MiB of peak allocation. These are operator measurements,
not end-to-end prefill rates. At M=4096, TP1 gate/up changes 17.543 to 17.641 ms
(+0.56%). The rejected M=1024 tile changes 5.160 to 5.430 ms (+5.24%); the
final dispatch retains the original small-chunk implementation.

Graph replay with changed inputs passes independent effective-weight and
column-boundary checks for both packed layouts. Sampled M=8192 gate/up output
relative L2 differences from the control are 0.000420/0.000129/0.000132 for
TP1/TP2/TP4 shapes; maximum absolute difference is 0.0004883. This does not
establish dataset-level output quality.
The down-projection sampled relative L2 difference is 0.000270, with maximum
absolute difference 0.001953. Its separate one-hot Graph test checks every
output column against exact independently dequantized weights.

## Capacity failures retained

With scoped warmup and a calibrated 0.5 GiB graph reserve, the untiled TP1
request activation peak is 2.254 GiB and the KV budget at GUM 0.93 is 2.230 GiB.
Gate/up tiling lowers that peak to 1.594 GiB and raises KV budget to 2.886 GiB.
Even with the compact 4.28125 GiB KV layout, admission still lacks 1.395 GiB.

The gate/up-only GUM 0.98 boundary experiment passes startup, with actual Graph
capture taking 0.404 GiB. Its first cold 32768-token request fails at the down
projection's 170 MiB dense-weight allocation: 18 MiB device free and 563 MiB
allocator-reserved but unallocated. Startup success is therefore not serving
success. This failure motivated the additional down-projection tiling.

Down-projection tiling does not by itself fix serving: GUM 0.98 then fails a
128 MiB allocation with 126 MiB device free and 467 MiB reserved-unallocated.
Reducing GUM to 0.975 retains enough KV for admission but the first 32K request
still fails a 272 MiB allocation with 186 MiB free and 919 MiB
reserved-unallocated. Retaining `max_split_size_mb:20` during serving at
GUM 0.975 instead fails admission (4.25 GiB KV available versus 4.28 required),
because unsplit allocations change the measured activation peak. This setting
is a separate runtime experiment, not the warmup-only source default.

## Successful single-card boundary experiment

With the complete candidate, GUM 0.98, the calibrated 512 MiB Graph reserve,
and `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20` retained during serving,
the single-card C2/64K configuration starts and completes both cold retrieval
requests. They contain 32768 and 64512 input tokens, each returning all three
values and the sum correctly, with natural EOS after 16 output tokens.
Each request resets the prefix cache. TTFT is 36.278 and 72.989 seconds;
the first request includes additional kernel JIT compilation.

One subsequent `vllm bench serve` random 32768-input/256-output C1 request,
after request warmup and prefix-cache reset, completes with zero failures:

| Metric | Value |
|---|---:|
| TTFT | 35.049 s |
| Input tokens / TTFT (estimate including first-token work) | 934.9 tok/s |
| TPOT | 9.042 ms |
| Complete request duration | 37.355 s |
| Complete-request output throughput | 6.853 tok/s |

TPOT is the official benchmark average after first token, not a separate
steady pure-decode measurement; DFlash emits multiple accepted tokens per
step. This is single-GPU 32K evidence, not the earlier TP4 256K baseline.

With persistent allocator scoping the measured profile peak is 1.674 GiB
(versus 1.594 with the serving allocator restored), and available KV is
4.405 GiB. Actual Graph capture is 0.404 GiB. After the long requests, device
usage is 31.471 GiB; peak PyTorch active allocation is 30.759 GiB. Memory
headroom remains small. Both long retrievals pass, but this does not replace
a paired dataset-quality or end-to-end speed-regression test. No automatic
GUM, graph-reserve, or persistent allocator default is changed by this PR.

## Validation and promotion

Focused CUDA tests cover tiled/fallback gate-up, exact down-projection basis
vectors, changed-input graph replay, warmup peak accounting and allocator
restoration (29 passed). Run the operator benchmark with
`python benchmarks/kernels/benchmark_sm70_nvfp4_prefill_memory.py --tp-shape 1`
and `--projection down` for the corresponding projection. The KV utility suite
has 69 passes and one pre-existing DeepSeek
fixture failure (`max_in_flight_tokens` missing); the identical failure is
reproduced on the unmodified base. Do not report the whole suite as green.

Keep this change in Draft until matched serving speed and output quality are
validated against the updated integration source. In particular, a 64K probe is
not a 256K acceptance result, and local TP-shaped kernels do not replace TP2
and TP4 serving checks.

The successful serving probe used implementation `f39f7099fb` (native code
before formatting-only commit-hook edits). Main advanced to
`fcf59f8e9ae50c186333e98e5cf6aae705f320de` during this investigation, including
DFlash batched decode defaults. It was merged into this owned branch after the
serving probes. The measurements above remain tied to the stated integration
base; they are not serving validation of the newer main combined with this
candidate. The full runtime commands and native hashes are retained in the
local handoff, and all task-owned serving processes were stopped.
