# SM70 DFlash2 Prefill Closure

## Scope and integration base

This private campaign is stacked on the quality-audited DFlash2 branch at
`ee4ac48a479c3dbd458d5f7c09a59f39fd271d82`. It keeps the accepted NVFP4
target, official BF16 DFlash2 draft, FP8 E5M2 target KV, FP16 draft KV,
prefix caching, Mamba alignment, and CUDA Graph decode contract. Draft-MLP
QPN8 remains disabled.

The first objective is to restore already accepted SM70 prefill operators in
source-overlay deployments. Kernel arithmetic is not changed by that repair.
Any later shape expansion is a separate gate.

## Project PR audit

The historical short- and long-prefill numbers use different contracts:

| Evidence | Contract | Accepted result | Boundary |
|---|---|---:|---|
| Public PR #271 | Qwen3.8-27B-FP8, TP4, exact input 8000, target-only | 5121.44 request-wall and 5170.96 pure-prefill tok/s | Exact-8K only |
| Public PR #324 | Same exact-8K FP8 contract | 5500 tok/s is a campaign target, not a measured implementation result | Documentation-only PR |
| Public PR #224 | Qwen3.8-27B-FP8, TP4, exact input 65536, target-only | 2798.6 to 3496.4 prompt tok/s after the exact D256 operator | Closest retained 64K target-only reference |
| Private PR #8/#13 | Qwen3.8-27B-FP8, TP4, input 261888, chunk 8192 with FP16 Mamba/SSM cache and Q8000 aligned chunks, target-only | 2438.89 prompt tok/s | Stable max-aware D256 architecture |
| Rejected public PR #315 lane | Same 256K FP8 contract | 2971.51 prompt tok/s | Rejected: 32 output token IDs were zero |

The 2438.89 tok/s route uses max-shifted exponentiation and max-aware online
softmax merging. Its output hash exactly matched the exact control. It also
explicitly overrides the checkpoint's FP32 SSM-cache contract to FP16; that
override is a separate quality variable and is not inherited automatically by
DFlash2. The removed raw-logit half2 polynomial must not be restored.

## Current DFlash2 baseline

The same-card cold benchmark resets prefix cache before every warmup and
measurement. It uses the practical 256K API contract, including chunk 4096.

| Input | Mean pure prefill | Prompt throughput |
|---:|---:|---:|
| 32768 | 10.485347 s | 3125.12 tok/s |
| 65536 | 25.317796 s | 2588.54 tok/s |

All three repeats at each length emitted the same first-token hash. Artifact:
`/data/minimax-h3/task-cache/v100-dflash2-prefill-32k64k-20260827/current-dflash2-cold-prefill-v1/`.

## Confirmed root cause

Every retained practical DFlash2 long-prefill log reports that
`_vllm_fa2_C` cannot be imported. The source checkout contains the D256
dispatch and quality-safe architecture, but the source overlay shadows the
installed package containing the native extension. Therefore the merged D256
path is not merely underperforming; it has never executed in these runs.

The accepted stable binary is retained at
`/data/minimax-h3/task-cache/qwen38-d256-attn-80tflops-20260825/build/exact-stat-256k-py312-v2/_vllm_fa2_C.abi3.so`
with SHA256 `f9f9acbc610c87fce9984e8fbd93fe0c8fa59887542123a74b3eaef6d3b8abf9`.
It loads against the active Torch 2.10/CUDA 12.8 environment and registers the
required dense, paged, split-KV3, and stable GQA architecture operators.

This branch adds an explicit `VLLM_SM70_FA2_D256_LIBRARY` source-overlay
sidecar. It is opt-in and follows the existing SM70 native-sidecar convention.
Bundled-wheel behavior remains unchanged, and missing or incompatible
operators still fail closed to the existing fallback with a warning.
Both the benchmark preflight and runtime loader validate registered operators,
not merely a successful Python-interface import. This covers partially cached
interfaces that otherwise appear importable while exposing no native kernels.

## Shape boundary and next measurements

The practical chunk-4096 contract can use the exact D256 Split-D operators,
but it cannot enter the stable long GQA architecture, whose validated kernel
contract is Q8000 with KV16K..256K in 8K steps. With seven speculative slots,
the scheduler first reduces the configured 4096-token budget to 4089. The
checkpoint's FP32 SSM state makes one aligned attention/Mamba block 1648
tokens, so the observed steady prefill query is Q3296. A configured 8192-token
budget retains that FP32 state and yields Q6592, not Q8000. The historical
target-only FP16 Mamba/SSM contract used an 800-token block, but DFlash2 grows
the convolution state by seven speculative slots. Its FP16 block is therefore
880 tokens and the same budget yields Q7920, so it cannot enter the existing
Q8000 architecture either. The next paired measurements are therefore
deliberately separated:

1. chunk 4096 plus the stable sidecar, to measure the dependency-closure gain;
2. chunk 8192 plus the same sidecar and FP32 SSM state, to measure Q6592;
3. chunk 8192 plus FP16 Mamba/SSM cache, to measure the actual Q7920 DFlash2
   geometry rather than crediting the target-only Q8000 route;
4. profile the remaining NVFP4 projection cost before considering a new
   Q6592/Q7920 attention architecture.

Each candidate must prove operator-route hits, preserve output validity, fit
the 256K DFlash2 memory contract, and retain the quality-audit PPL and scored
coding gates. Prefix-hit time is reported separately and never counted as cold
prefill throughput.

## Dependency-closure A/B

The first candidate changes only native-extension resolution and keeps chunk
4096. All six measured requests are cold. The stable sidecar loads on all four
ranks and the benchmark reports both required D256 operators as available.

| Input | Missing-extension control | Stable-sidecar candidate | Throughput gain |
|---:|---:|---:|---:|
| 32768 | 3125.12 tok/s | 3476.53 tok/s | +11.24% |
| 65536 | 2588.54 tok/s | 3103.02 tok/s | +19.87% |

Candidate pure-prefill means are 9.425490 s and 21.120102 s. The three repeats
at each length retain the control first-token hash
`54363ddee68f4a5db81c9d37e5fb738d28f5b67dc7f725ad7333172b1ea157da`.
Artifact:
`/data/minimax-h3/task-cache/v100-dflash2-prefill-32k64k-20260827/candidate-stable-fa2-q4096-v1/`.

## Chunk and Mamba dtype closure

Increasing the configured budget to 8192 while retaining the checkpoint's
FP32 SSM state produces Q6592. It is neutral at 32K and slightly slower at 64K:

| Input | Q3296 sidecar | Q6592 sidecar | Change |
|---:|---:|---:|---:|
| 32768 | 3476.53 tok/s | 3481.83 tok/s | +0.15% |
| 65536 | 3103.02 tok/s | 3077.03 tok/s | -0.84% |

The FP16 Mamba/SSM arm produces Q7920, not Q8000, and loses at both lengths:

| Input | Q3296 sidecar | Q7920 FP16 state | Change |
|---:|---:|---:|---:|
| 32768 | 3476.53 tok/s | 3271.49 tok/s | -5.90% |
| 65536 | 3103.02 tok/s | 2827.03 tok/s | -8.90% |

All twelve measured requests retain the same first-token hash. Q6592 is not a
promotion, and Q7920 is rejected on speed before spending a dataset-quality
run. Artifacts are `candidate-stable-fa2-q8192-fp32-v4` and
`candidate-stable-fa2-q8000-mamba-fp16-v1` under the task root above.

## NVFP4 projection candidate

The accepted DFlash2 target keeps M<=8 verification on the existing QPN2
route, while large-M prefill currently falls through to TurboMind W4A16. A
single-V100 M3296 microbenchmark shows that the already present bounded FP16
QPN4 prefill operator is materially faster even though its timing includes
weight dequantization on every call:

| Projection | TurboMind | bounded QPN4 | Latency reduction |
|---|---:|---:|---:|
| fused gate/up, 5120x8704 | 5.351 ms | 3.896 ms | 27.19% |
| down, 4352x5120 | 2.646 ms | 1.866 ms | 29.47% |

The candidate reuses the QPN2 code and E4M3 scale-code buffers that are already
resident for verification, plus the existing shared 85-MiB FP16 workspace. It
does not retain a third packed weight layout: QPN2 and QPN4 expose different
2-D shapes but use the same flattened physical tile order, so the bridge uses
zero-copy views. Admission is default-off and requires
`VLLM_SM70_NVFP4_QPN2_PREFILL=1`; M below the separately recorded crossover
threshold keeps TurboMind, and M<=8 verification remains QPN2. The current
campaign stops at 64K; no 128K/256K throughput run is required for promotion.
The single-V100 bridge probe confirms byte-identical flattened code and scale
storage despite shapes `[N,K/2]` and `[K,N/2]`; both ordinary and fused
gate-SiLU QPN4-prefill calls are bitwise equal with maximum absolute error
zero.

## Rejected SWA tail-only shortcut

The DFlash2 draft uses a 2048-token sliding window, but projecting and writing
only the currently readable tail is not valid with automatic prefix caching.
Draft-group blocks are registered by content for later requests; a block whose
target tokens were computed without the matching draft-context KV write can be
reused later at a sequence length where those tokens are inside the window.
That produces a cache hit backed by uninitialized draft KV and collapses
acceptance. Therefore this campaign keeps full coverage of every newly
computed target token. A future tail-only design would first need to decouple
draft-group cache registration from target-prefix hits and prove rebuild
semantics; that is outside this prefill dependency closure.
