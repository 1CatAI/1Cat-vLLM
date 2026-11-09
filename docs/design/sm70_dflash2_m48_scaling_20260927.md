# SM70 DFlash2 M48 scaling audit, 2026-09-27

Main baseline: `ef6909830cbb7b40a24413bfe74ab49a4f7e1b90` (PR #706).
The initial audit uses the qualified pre-merge source tree without native changes. The default native candidates and their separate endpoint validation are recorded below; model, quantization and sampling remain fixed.

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

## Earlier GDN-only endpoint validation

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

## Default native optimization and acceptance targets

The next campaign freezes the preceding GDN-only service as its comparison.
C4/C6/C8 must improve pure decode by 10%/15%/20%, respectively. Conservative
thresholds use the higher median of main and GDN-only: **480.345 / 676.925 /
861.921 tok/s**. C1 must not materially regress. Kernel timings and projections
do not satisfy this endpoint requirement.

The retained implementation adds:

- Full M48 and safely masked tail tiles to the shared FP4/FP8 TurboMind
  registry. The policy is based on SM70, dense shapes, alignment and M/K/N,
  with the established split-K numerical partition. C1 stays on its existing
  fast route. Both formats share activation loading and dispatch; no model
  name, concurrency whitelist, second weight copy or environment switch is
  introduced.
- A registry-order correction: exclude batch-only candidates before
  Context::Filter chooses the covering CTA. Otherwise an excluded M48 tile
  can still hide the ordinary M64 reference and change its split-K family.
- A partial-warp activation-load guard. M48/K32 needs six loading warps in an
  eight-warp CTA. The base thread map marked the six-warp tile aligned and
  allowed two surplus warps to overwrite shared memory. Expanded generic
  FP4/FP8 tests exposed this despite actual model cases passing. The local
  map now keeps those row predicates active; all 127 quantized GEMM tests
  pass after the fix.
- Canonical-rank-order two-stage FP16 TP4 all-reduce for payloads from
  384 KiB inclusive to 512 KiB exclusive on fully connected SM70 groups.
  The existing small push and larger two-stage routes remain selected by
  their existing guards. The new kernel keeps the ordinary rank sum order,
  including adversarial cancellation cases.
- Eight-lane subgroup softmax in the full-q8 long-attention kernel. Four rows
  share a warp and preserve the original 32-column reduction tree,
  probability residual compensation, N32 panels and 80-way split contract.
  This changes execution layout, not attention coverage or KV precision.

The all-reduce 480-KiB screening result is 34.771 -> 26.058 us, with graph
replay, boundary sizes, stragglers, canaries and cancellation cases validated.
The 16-layer, 32K attention microbenchmark changes C4/C6/C8 from
7.129/10.460/13.878 ms to 6.316/9.256/12.264 ms. Output, partial numerator
and max/sum workspace bits match across 64 random layers at C1/C4/C6/C8,
mixed and empty rows, changed graph inputs, B16, unpaired KV strides, and
262K context. These are operator measurements, not endpoint claims.

### Ordinary service results for the combined default candidate

Same GPUs 4–7, fixture, ordered admission, natural-output gate and three
retained repeats. The custom streaming harness counts exact emitted tokens
inside the common all-requests-alive window; this is not vLLM bench output
throughput and does not include TTFT or new prefill.

| C | Prior GDN-only tok/s | Combined candidate tok/s | Change |
|---:|---:|---:|---:|
| 1 | 162.857 | 165.175 | +1.42% |
| 4 | 435.723 | 447.421 | +2.68% |
| 6 | 588.630 | 666.955 | +13.31% |
| 8 | 718.268 | 730.373 | +1.69% |

**The new C4/C6/C8 targets are not met.** C6 is close; C4 and C8 require
further work. Against the higher C4 main median, its gain is 2.46%.
All 76 performance-request token arrays and speculative counters remain
identical to the preceding baseline. Acceptance remains 33.395%, 37.093%,
48.874%, and 51.029% at C1/C4/C6/C8. Long retrieval remains 8/8 correct with
8/8 natural stops.

The service uses normal source-built package extensions, with no task-local
DSO dependency or new tuning switches. It reports 9.57 GiB model memory,
0.99 GiB actual graph pool, 11.45 GiB available KV memory and 1,050,885 KV
tokens. The 262144 maximum context remains configured. This is a startup
capacity observation, not an end-to-end 262K performance claim.

Test results: 127 quantized GEMM GPU tests, nine long-attention GPU cases
(eight existing cases plus C6), one four-GPU all-reduce test, and the prior
80 GDN GPU tests. Compute Sanitizer failed to attach before its first
instrumented CUDA call; no sanitizer pass is claimed. The shared 35B-A3B
AWQ/FP8 service performance gates have not been rerun for this candidate.

Service extension SHA256:

- core: `9e59de1e756537e5b6704054c179c3cf4114e429e075b6c92a0be254c179a5e5`
- FA2: `ac99fbe8e879df746478d2bb872e910c94b776532c4a71c6f3f7db8a0a7b6859`

A later ordinary rebuild after rejected experiments produces core hash
`095fef73349812415840f53166aafb714328d43ef273c852a29a82f0ba4f18ce`.
Its source diff is byte-identical to the recorded endpoint source patch;
the different binary hash is recorded separately and not substituted for
the hash used by the measured service.

### Retained and rejected experiments

Artifacts are under the campaign's `batch-targets-20260927/` directory:
`results/service-m48-softmax-summary.json`, ordinary service/evaluation logs,
`results/native-m48-softmax.{sha256,patch}`,
`results/normal-m48-gemm-tests-fixed.log`,
`results/normal-attention-softmax-tests.log`,
`results/softmax-validation.json`, and the earlier medium-all-reduce logs.
Original endpoint JSON, token IDs and retrieval results use
`long-service-batch-targets-m48-softmaxfix/`.

Request-interleaved attention CTAs, sequential PV panels, extra M64 warp
arrangements, deeper QPN2 prefetch, FP8 scale folding, TurboMind lookahead
tactics, smaller softmax subgroups and N64 attention panels did not provide
a sufficient consistent win. They are absent from runtime source. Reduced
attention split counts had a local win but changed arithmetic; the exact
subgroup optimization was preferred. Raw failed experiments are retained
in the campaign worklog to avoid repeating them.

Remaining measured constraints are M64 GEMM cost and long-attention work
that still scales with each request. A separate FP4 scale-supply probe was
bit-exact but slower (down 45.712 -> 46.152/46.824 us, gate/up
74.608 -> 78.432/77.160 us). It was rejected. Combining two QK MMA groups
before compensated accumulation reduced the attention operator cost by
about 4%, with similar error against an FP64 reference, but changed output
bits. It has no acceptance or endpoint qualification and is not enabled or
counted as an improvement. Draft review remains open; this report does not
claim completion of the new targets or a renewed PRO 6000 comparison.

### M64 hardware counters and occupancy screening

Nsight Compute 2022.4.1 profiled four real TP-local weight shapes after
ordinary tuning on V100 GPU 7, using the committed candidate's normal core.
The retained report is `results/ncu-current-m64-direct.ncu-rep`; text and raw
CSV exports are alongside it. These isolated kernel replays are diagnostic,
not service timing or useful model FLOPs as a fraction of hardware peak.

| M64 projection | Registers/thread | Achieved occupancy | DRAM throughput | Tensor pipeline active |
|---|---:|---:|---:|---:|
| FP4 gate/up | 121 | 23.12% | 27.65% | 44.88% |
| FP4 down | 131 | 12.47% | 27.49% | 47.00% |
| FP8 input | 178 | 11.67% | 34.50% | 36.53% |
| FP8 output | 146 | 12.36% | 25.74% | 27.25% |

The DRAM and tensor figures use NCU's elapsed-cycle denominator. The FP8
input sample selected the 128-thread M32xN256 kernel; FP4 down and FP8
output selected 256-thread M64xN128 kernels. The latter two permit only one
resident CTA with their present register requirements. Global memory
bandwidth is not saturated. L1/TEX, instruction dependencies, barriers and
insufficient ready warps must be considered along with register occupancy.

A separate launch-bounds screen preserved the numerical partition and
passed real-weight output oracles, but failed the performance screen.
Reducing the ordinary FP4 down candidate from 133 to 128 registers allowed
two resident CTAs but increased cold tuning time from 60.93 to 66.25 us.
FP8 input's M64 candidate at 128 registers was also slightly slower than its
146-register control; its local allocation increased from 64 to 96 bytes.
Aggressive M32 caps were substantially slower. The whole-model M64 estimate
was 16.634 -> 16.745 ms, so none of the launch-bounds candidates are kept.
Raw candidate costs, source patch and matched graph samples are retained as
`regcap-tactic-costs.log`, `rejected-regcap.patch`, and `gemm-*regcap-gpu3.*`.

The analogous M32/QPN2-QPN8 capture on GPU 3 shows tensor-pipeline activity
of 33.59/30.21/28.93/23.45% and DRAM throughput of
35.02/31.61/53.46/42.88% for FP4 gate, FP4 down, FP8 input, and FP8 output.
FP8 input/output spend their largest measured stall contribution on the
long scoreboard. This motivates supply scheduling experiments rather than
treating C4 as compute-saturated. The raw M32 report is
`results/ncu-current-m32-direct.ncu-rep`.

Explicit paired-half scale multiplication was exact but yielded no
whole-model gain (M64 16.634 -> 16.683 ms), so it was reverted. A separate
research build removed the batch split-K partition restriction: M48
14.304 -> 13.776 ms and M64 16.634 -> 15.953 ms. This changed output bits
(maximum tested difference 0.00048828125), despite passing the microbenchmark
tolerance checks. Its approximately 4.1% M64 GEMM gain is not an endpoint or
acceptance result. The change was reverted; the source-built runtime retains
the protected numerical partition. Raw data and the rejected patch are in
`gemm-partition-ceiling-gpu3.*` and `partition-ceiling.patch`.

The M32 FP8 L2-prefetch probe also failed screening: lookahead distances
four/eight slowed the three completed projection cases by approximately
2–9%. These cases were bit-exact, but the gate/up case subsequently failed
the exactness check (maximum difference 3.814697265625e-6). No further
performance claim is made for that incomplete probe. It exists only as
task-local research code, never as a service dependency. See
`results/fp8-m32-l2-micro.{json,log}`.

After these experiments, all production kernel sources are restored to
commit `a42e887cf65b537ede6f0b3db380706cbd9c5145`. The final ordinary core
rebuild has SHA256
`e06b8d46eda509c2cdd9bece49427c9ea0d8a8c908c8a8fe6600b0de22c45e1c`;
the FA2 hash is unchanged. This restoration is not an additional endpoint
run. The last qualified endpoint remains the combined-candidate table
above, and none of the new performance targets is claimed complete.

### Main integration and incremental merge

On 2026-09-27, the project owner requested merging this verified increment
after disclosure that C4 +10%, C6 +15% and C8 +20% remain unmet. Merge
authorization does not turn those targets, a renewed PRO comparison or the
35B-A3B AWQ/FP8 endpoint speed gates into passes.

Integration commit `bec7cb784c7fec909f37a344c98133461ac46888` incorporates
main `db292f9a49318459f064075bfdbbed438b4e77a3` (PR #703). The only conflict
was the appended migration histories; both are retained. The shared
communicator keeps main's new HC buffer offsets and this PR's canonical
medium-payload reduction. Main's optional mixed-QKV MTP verifier does not
replace the DFlash2 packed verifier.

Normal `_C` and `_C_stable_libtorch` were rebuilt from the integrated
source; the same CMake build verified `_vllm_fa2_C`. Fresh-process imports,
ELF dependencies and resolved mappings use only the shipped package and
standard Torch/CUDA libraries, with no private DSO or preload. SHA256:

| Extension | Integrated artifact SHA256 |
|---|---|
| `_C` | `921394e201f82b2ee07f051fe2d6b29a14a700d39456ae2179f35635d8f657a2` |
| `_C_stable_libtorch` | `54632ce86c8077a6d6e30273b00234b46e3e62f05811067f5ddb7df3fdd653fb` |
| `_vllm_fa2_C` | `ac99fbe8e879df746478d2bb872e910c94b776532c4a71c6f3f7db8a0a7b6859` |

The merged regression suite passed **269 tests in 148.74 s** on an idle,
exclusively leased V100 group 0–3: FP4/FP8 GEMM tails and prescaling,
TP4 all-reduce boundaries, long attention through 262K, exact DFlash2
GDN states, the newly integrated MTP GDN loader, and graph selection.
The 44 graph-selection tests also passed with CUDA devices hidden.
GitHub pre-commit CI passed. Build, imports, dependency inspection and
test logs are retained under `batch-targets-20260927/results/merge-*`.

The ordinary integrated service then passed one ordered 32K/256 wave at
C1/C4/C6/C8 on GPUs 0–3. All **19 complete token arrays and speculative
counters match** the previously qualified candidate's corresponding wave;
acceptance is unchanged. Long retrieval remains **8/8 correct and 8/8
natural stops**. Worker logs confirm the default GEMM/GDN routes and long
q8 graph replay for C4/C6/C8. These one-wave checks on a different GPU group
do not replace the frozen three-wave performance medians above.

This startup reports 9.57 GiB model, 0.99 GiB graph pool, **11.00 GiB KV
and 1,009,312 KV tokens**, versus the earlier measured service's 11.45 GiB
and 1,050,885 tokens. The 262144 context limit remains configured; the
capacity difference is not attributed to a specific cause or claimed
resolved. No persistent weight copy or workspace was added by this PR.
See `results/merge-service-summary.json`, `merge-service.log` and
`merge-eval.log`; raw request results are `long-service-merge708/`.
The owned private service is stopped, GPU leases released, and public
serving remains off.
