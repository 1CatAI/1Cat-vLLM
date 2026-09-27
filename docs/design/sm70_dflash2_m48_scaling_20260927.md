# SM70 DFlash2 M48 scaling audit, 2026-09-27

Main baseline: `ef6909830cbb7b40a24413bfe74ab49a4f7e1b90` (PR #706).
The qualified pre-merge worktree has exactly the same source tree. No native kernel, quantization, sampling, or model change is used for this audit.

## Contract

V100-SXM2-32GB TP4 on GPUs 4–7, CUDA 12.8, Torch 2.10.0+cu128, Qwen3.8-27B-NVFP4 with its existing mixed FP4/FP8 weights, FP16 compute, E4M3 KV, Flash-V100, DFlash2 q7 probabilistic. Maximum context 262144, max sequences 16, memory fraction 0.8, 8192 batched tokens, default CUDA Graph and prefix caching enabled.

32768 input / 256 fixed output tokens, seed 20260923+i, temperature 0.7, top-p 0.8, top-k 20. Same prefix-primed fixture and ordered admission as the PR #706 baseline. Ignore-EOS applies only to this synthetic performance fixture; separate long retrieval uses natural EOS. Fixture SHA256: `348dd1b93ba266c1a9a18949b6c41205cb130d1c95ead57597e3d5d643b18ac9`.

## Ordinary service, no worker instrumentation

One warm wave excluded, median of three subsequent independent waves. TPS counts exact returned token IDs during the common window when all C requests are decoding; excludes first-token/prompt time and new prefill. Dividing aggregate TPS by C is the average per-request rate in that same window.

| C | Pure decode tok/s | Average per request tok/s | ms/token per request | Accepted/drafted | TTFT p50 s |
|---:|---:|---:|---:|---:|---:|
| 1 | 162.364 | 162.364 | 6.159 | 33.395% | 1.240 |
| 4 | 436.677 | 109.169 | 9.160 | 37.093% | 3.364 |
| 6 | 578.872 | 96.479 | 10.365 | 48.874% | 5.182 |
| 8 | 704.171 | 88.021 | 11.361 | 51.029% | 5.314 |

All repeated token ID arrays are identical within each concurrency. C1/C4/C8 also match the prior qualified baseline. Long retrieval: 8/8 correct, 8/8 natural stops. C6 is now an explicit benchmark point.

## Low-overhead full-q8 GPU rounds

For each step, select the TP rank with the longest whole-round CUDA-event interval, then take every phase from that same rank. Exclude prefill, incomplete batches, and the first/last full-q8 step. This avoids summing unrelated per-category rank maxima. Means are additive; p50/P90/P99 remain in the raw JSON.

| GPU phase, ms | C1/M8 | C4/M32 | C6/M48 | C8/M64 |
|---|---:|---:|---:|---:|
| Target forward | 14.592 | 26.489 | 39.846 | 42.557 |
| LM-head / sampling | 1.417 | 1.298 | 1.590 | 1.527 |
| State update | 0.157 | 0.169 | 0.180 | 0.178 |
| Draft | 4.241 | 4.834 | 5.739 | 5.854 |
| Other / unattributed | 0.418 | 0.484 | 0.559 | 0.579 |
| Whole round | 20.825 | 33.274 | 47.913 | 50.695 |
| Whole round / C (amortized cost) | 20.825 | 8.318 | 7.985 | 6.337 |

Whole round / C is amortized GPU cost, not the latency of a request: each request still waits for the complete round. Accepted output width determines its eventual ms/emitted token.

## Nsight Systems node attribution

Nsight Systems 2025.1.1, CUDA Graph node tracing, per-cohort/step/phase NVTX labels, launch correlation joined by process. Kernel service sums are diagnostic and must not replace ordinary endpoint speed. Profiled service is slower; do not mix its totals with the low-overhead table. The following means select the same critical rank within each profiled step.

| Target-forward GPU kernel service, ms | C1 | C4 | C6 | C8 |
|---|---:|---:|---:|---:|
| FP4_GEMM | 4.474 | 6.543 | 9.684 | 9.786 |
| FP8_GEMM | 3.900 | 4.859 | 8.314 | 8.282 |
| attention | 1.989 | 7.266 | 10.678 | 14.042 |
| GDN | 1.483 | 2.657 | 4.639 | 4.470 |
| TP_reduce | 1.465 | 3.335 | 5.347 | 4.534 |
| norm | 0.160 | 0.325 | 0.328 | 0.339 |
| other_GEMM | 0.630 | 0.634 | 0.661 | 0.682 |
| other | 1.406 | 2.198 | 2.074 | 2.218 |
| service_ms | 15.509 | 27.817 | 41.724 | 44.352 |
| wall_ms | 16.478 | 29.096 | 42.837 | 45.528 |

C1→C8 growth in target kernel service: attention +12.053 ms, compressed GEMM +9.694 ms, TP reduction +3.068 ms, GDN +2.987 ms. At 32K, attention is the largest growth category.

| Target kernel service growth, ms | C1→C4 | C4→C6 |
|---|---:|---:|
| Compressed GEMM | +3.027 | +6.596 |
| Attention | +5.277 | +3.411 |
| GDN | +1.173 | +1.983 |
| TP reduction | +1.869 | +2.012 |

Other categories partly offset these increments. The two intervals need
different priorities: long-context attention dominates C1→C4, while GEMM
dominates C4→C6. Amortized forward cost is 6.622 ms/request at C4 and
6.641 ms/request at C6: this interval adds requests without reducing their
average forward cost. C6→C8 then improves amortization substantially.

## Actual M48 route and missing coverage

- Graph capture and actual replay use 48 real / 48 padded tokens for C6. C5 pads 40→48; C7 pads 56→64. There is no whole-request C6→C8 padding.
- Runtime warmup explicitly includes `[1,2,4,8,16,24,32,48,64]`. FP4/FP8 batch dispatch admits M33–M64. Four TP ranks export identical 34-record GEMM plans.
- M48 FP8: 144 batch-supply GEMMs using CTA 32×256×32, 178 registers/thread, 128 threads. M64 uses 128 full-tile 64×128×64 GEMMs plus 16 of the smaller tile.
- M48 FP4: 112 batch-supply GEMMs using CTA 32×128×32; M64 uses 56 of that tile plus 56 full-tile 64×128×64 GEMMs. Two M32 row tiles have 16 unused rows for M48. This is internal GEMM tiling, not eight actual requests.
- Thus M48 is optimized, but its FP4+FP8 cost is 17.998 ms, essentially M64’s 18.068 ms. Full-M64 iterator admission intentionally rejects M48; simply deleting that guard would permit out-of-bounds accesses.
- GDN has a genuine coverage hole: the former `N in (4,8)` gate leaves C6 on BV32 (225 registers/thread, grid 1×4×72). C8 uses BV8 (80 registers/thread, grid 1×16×96).
- C6 480-KiB TP payload uses ordinary one-stage reduction (512 threads, 36 CTAs), while C8 640-KiB uses the tuned two-stage launch (256 threads, 20 CTAs). The 512-KiB threshold explains this discontinuity; trace communication includes waiting and is not a standalone collective benchmark.

## Minimal default-on GDN correction

Replace the service-count whitelist with the measured operator range `4 <= N <= 32`, retaining SM70, q8, head geometry, FP32 state/gating, FP16 I/O, recurrence/norm, and override guards. No model name, weight quantization, or new environment flag is added. Larger batches retain BV32: the B64 microbenchmark made BV8 1.06% slower.

Eager initial comparisons are bitwise exact for output and state. C6 BV32→BV8: 71.200→58.771 us/layer (17.46% lower), approximately 0.597 ms for 48 layers; this projection is not an endpoint claim. B5/B7/B12/B16/B32 were also screened. The full existing GPU regression suite, extended to non-power-of-two batches, B32 and the B64 fallback, passes **80 tests**: strided QKV, gapped state storage, all accepted-state selectors 1–8, changed inputs, graph replay and untouched sentinel slots.

The change is Python-only and reuses the already qualified main native artifacts; native source trees are unchanged. Normal package extensions only, no LD_PRELOAD, private sidecar kernels or runtime overrides. Native core SHA256: `49b92da93e596ef8e3c2ec4b07907c5e2663c131413ce7f3dafdc8e4f68bb7b4`. Flash-V100 FA2 SHA256: `7962e11a7af0c0f88107459df7292c9f8c4a552c7be58efc7b1a0e32ae3ff2f8`.

## Candidate ordinary endpoint validation

The identical fixture, ordered admission, natural-output gate and three-repeat
rule are used on a fresh ordinary service with the default route change.

| C | Main median tok/s | Candidate median tok/s | Change | Accepted/drafted |
|---:|---:|---:|---:|---:|
| 1 | 162.364 | 162.857 | +0.30% | 33.395% |
| 4 | 436.677 | 435.723 | -0.22% | 37.093% |
| 6 | 578.872 | 588.630 | +1.69% | 48.874% |
| 8 | 704.171 | 718.268 | +2.00% | 51.029% |

C6 repeats are 579.883/578.872/574.902 before and
588.630/589.644/583.211 after. Its full-request mean-TPOT median changes from
12.768 to 12.618 ms; that metric includes interference from the initial prompt
admissions and is separate from the all-live pure-decode window.

Every token-ID array and every speculative counter matches main for all 76
performance requests, including warmup waves. Acceptance is unchanged at every
concurrency. The candidate retrieval gate remains 8/8 correct with 8/8 natural
stops. C1/C4/C8 already used the admitted route during complete batches;
in particular, do not attribute the C8 +2.00% median fluctuation to a new C8
full-batch optimization. Its repeat range is 703.593–728.633 tok/s and its
full-request mean-TPOT median is nearly unchanged (20.879→20.845 ms).

Both launches report 9.57 GiB model loading and 0.99 GiB actual graph pool per
rank. The first candidate startup reports 11.00 GiB available KV versus main's
11.45 GiB (1,009,312 versus 1,050,885 tokens). The patch allocates no new
persistent tensors; the startup budget difference needs separate attribution
and is not evidence of unchanged total memory capacity.

## Retained evidence

`profiles/c1-c4-c6-c8.nsys-rep`, its SQLite export, `results/steps/{ledger,nodes}-c{1,4,6,8}/rank{0,1,2,3}.jsonl`, `results/{ledger,nodes}-step-summary.json`, `results/nsys-attribution.json`, `results/representative-kernels.json`, `results/baseline-summary.json`, `results/candidate-summary.json`, `results/gdn-warp-supply.json`, `results/gdn-large-coverage.json`, and `results/gdn-exactness-tests.log`. The private campaign worklog records absolute locations, exact launch scripts and active process ownership.

The GDN admission fix does not claim to solve the remaining GEMM, long-attention or collective bottlenecks. Next candidates must preserve output and every state snapshot, prove a same-shape operator win, then confirm ordinary serving speed.

Priorities from this trace are tail-safe M33–M64 GEMM tiles with more weight
reuse across M, screening the collective strategy for the 480-KiB region, and
reducing long-attention resource pressure. The grouped long-attention kernel
uses 124 registers/thread and 72704 bytes of shared memory per CTA. These are
launch-resource observations, not fresh Nsight Compute measurements of tensor
core utilization or DRAM bandwidth. Attention KV work grows with requests even
when prompt storage is prefix-shared; not all of that growth is avoidable.

For planning only, a further 15% pure-decode speedup with unchanged accepted
tokens and all else equal would require about 4.34 ms off C4's 33.27-ms round,
6.25 ms off C6's 47.91-ms round, or 6.61 ms off C8's 50.70-ms round. Those are
whole-round savings targets, not predictions or measured candidate results.
