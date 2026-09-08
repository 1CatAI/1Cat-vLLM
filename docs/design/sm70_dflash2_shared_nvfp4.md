# DFlash2 QPN2 and TurboMind shared NVFP4 weights

## Scope and activation

`VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT=1` removes the separate QPN2 code
buffer from compatible SM70 TP4 NVFP4 projections. It requires a rebuild
providing `nvfp4_qpn2_prepare_scales_sm70` and
`nvfp4_qpn2_tm_dispatch_sm70_out`. Older binaries retain the existing
separate layouts and log the missing capability. The switch defaults to
zero pending broader model quality and concurrency performance acceptance.

The existing QPN2 model and shape gates still apply. The target is
Qwen3.8-27B-QUASAR-NVFP4 with DFlash2 q7. This does not enable QPN2 on
additional models or quantization schemes. Set the switch before loading;
changing it on a loaded model does not reclaim or restore weight buffers.

## Storage and dispatch

TurboMind keeps its non-interleaved SM70 HMMA884 B/Pack1 codes and FP16
scales. QPN2 stores only its packed E4M3 scales, including zero padding to
N=32 alignment. It reads the TurboMind code tensor directly. No QPN2 code
buffer, code alias buffer, or transient full QPN2 repack is registered.

Both formats preserve each E2M1 nibble. Within K=8 their physical order is
`[0, 2, 4, 6, 1, 3, 5, 7]`. For QPN lane `l`, let
`c = ((l >> 2) & 3) * 8 + (l & 3) + ((l & 16) ? 4 : 0)`.
The two words for K/16 group `g` in N/32 tile `t` are TurboMind words
`(t * (K / 8) + 2 * g) * 32 + c` and that index plus 32.
The common reader implements this address change for ordinary/gated QPN2
and the prefill dequantizer. Arithmetic, accumulation and activation order
are preserved.

Scales remain separate because TurboMind merges the global scale into FP16
while QPN2 retains E4M3 plus a separate global scale. Recovering the original
E4M3 data from rounded FP16 would change the numerical contract.

The new opaque C++ dispatcher keeps dynamic M selection outside Dynamo:

| Live M, with default prefill threshold | Route |
| --- | --- |
| 1–32 | QPN2 reading TurboMind codes |
| 33–1023 | Existing TurboMind GEMM |
| 1024 and above | Existing bounded FP16 prefill, reading TurboMind codes |

With QPN2 prefill disabled, the shared operator receives a zero threshold
and all M above 32 use TurboMind. Python still handles empty inputs and
crops padded outputs before bias. TurboMind warmup and state ownership are
unchanged.

For all 256 supported QUASAR TP4 target projections, the removed codes
total 2.835693 GiB per rank (11.342773 GiB across four ranks). The remaining
QPN2 scale allocation is approximately 0.354 GiB per rank. This is a tensor
storage calculation. Validate with the production KV allocation policy:
automatic sizing at the existing memory utilization, context and concurrency
settings. Record weight storage, KV budget/capacity and total NVML usage
separately, because freed weight memory can become additional KV capacity.

## Validation recorded on 2026-09-08

Integration base: `e5d63c51f0fcc1ddf75d229e3df06bf52df206f5`.
Torch 2.10.0+cu128, CUDA 12.8, V100-SXM2-32GB, FP16 activations;
`VLLM_SM70_NVFP4_QPN2_M16_NATIVE=1`. Actual checkpoint shards for TP ranks
0–3 were tested sequentially on one V100. These tests do not measure TP
communication or whole-model throughput.

- Nine CPU tests pass, covering shared loading without code preparation,
  old-binary fallback, original routes, output cropping, and prefill off.
- All 24 real projection shards match the old CUDA converter's code bits
  and packed scale bits, including GDN N=4120 padded to 4128.
- All 280 cases at M=1/8/16/32/33/64/135/1019/1024/4096 match FP16 output
  bits in eager execution and CUDA Graph replay after changing inputs.
  Gate/up is tested both as a linear projection and with fused SiLU.

Rank-0 graph timing brackets the shared candidate with controls compiled
in the same invocation. M8 shared/control ratios range from 0.982 to 1.044;
M16 ordinary projections are 1.091–1.123; gated M32 is 1.150. Large-prefill
ratios are 0.989–1.020. These are operator measurements under unlocked
clocks, not evidence of a model speedup. Two separated 32-bit loads cost
more than the previous 64-bit load for some shapes. These measurements
describe the initial shared reader, before the scheduling recovery below.

A bounded experiment replaced the shared reader's two streaming loads with
read-only cached loads. All 28 rank-0 cases remained bitwise equal. Although
the K=1536 output projections improved, QKV and MLP regressed (gated M8
ratio 1.131, MLP down M16 ratio 1.167). The global replacement was rejected;
the final recovery caches only the K=1536/N=5120 output projections.

### Scheduling recovery

The shared decode grid now places row blocks for the same weight tile next
to one another, improving reuse before traversing N. Shared M=9–16 uses
separate 8-row CTAs instead of retaining two row tiles in one CTA. The legacy
layout still honors `VLLM_SM70_NVFP4_QPN2_M16_NATIVE`. Split-K, accumulator
chains and each row's arithmetic order are unchanged. No additional weight
or temporary tensor is allocated.

Read-only cached loads are selected only for the 3.75 MiB GDN/attention
output weights at K=1536/N=5120. Larger QKV and MLP weights retain streaming
loads. This changes cache policy without allocating a tensor or changing
the accumulation order.

The final candidate passes all 336 ordinary/gated cases on 24 real TP4
shards at M=1/8/9/15/16/17/18/24/31/32/33/1024, in eager execution and
changed-input CUDA Graph replay. Timing captures 16 nodes per graph for
small M and takes eight ABBA rounds, retaining all raw samples.
Weighted by the model's projection counts, counting fused gate/up once:

| M | Repeated-projection shared/dual | Working-set shared/dual |
| --- | --- | --- |
| 8 | 1.0013 | 1.0259 |
| 16 | 0.9869 | 0.9915 |
| 32 | 0.9591 | 0.9019 |

Each ratio uses that run's control. Absolute times across the two timing
methods or unlocked-clock runs are not comparable. The working-set check
cycles six distinct real projection tensors in the model's 3-GDN/1-attention
pattern, exceeding L2 capacity, with 256 calls per graph. It preserves the
M16/M32 gain, while M8 retains a 2.59% cost. Without selective caching,
the working-set M8 ratio was 1.0639, so caching remains beneficial here.
These are isolated projections with independent inputs, without attention
or communication, and do not establish production throughput.

Rejected experiments retain their evidence: adjacent-column vector loads
plus a lane shuffle pass bitwise but regress speed, including the variant
using two row tiles at M17–32. Shared unroll factors one and two also regress
the weighted costs. Unroll eight brings gated M32 close to the control but
worsens gated M8, so it is not applied. Combining it with selective output
caching does not establish an advantage over caching alone; retain unroll
four. Hardware counter
profiling failed with `ERR_NVGPUCTRPERM`; no counter-based claim is made.

### Production service contract

Use the existing serving configuration: TP4, max length 262144,
`gpu_memory_utilization=0.8`, automatic E4M3 KV allocation, chunk4096,
maxseq4, FP16 activations, DFlash2 q7 with FP16 draft KV, FP32 logits,
context pipeline/KV graph and CUDA Graph enabled. Preserve production
sampling and xhigh thinking. Compare weight loading, KV capacity and NVML
usage separately, and report pure decode separately from TTFT.

The corrected production control completes with 11.08 GiB model loading
per rank, a 12.68 GiB KV budget, 1,190,275 logical KV tokens, 0.33 GiB graph
capture increment and 26,114 MiB NVML worker usage per rank. The existing
MBPP28 speed item returns the same 260 tokens with natural EOS in warmup
and three measured requests. Median pure decode is 277.52 tokens/s,
verification-round time 19.046 ms and TTFT 114.93 ms. Actual concurrency is
one; maxseq4 is capacity. A separate MBPP0 request completes with 2105
tokens and natural EOS. This is a focused check, not full quality admission.

The shared production retry completes after the user authorizes using idle
GPUs 0–3. The former reservation scheduler has no running or queued jobs
when it is gracefully released; active work is not preempted.

| Recorded allocation | Dual layout | Shared layout |
| --- | --- | --- |
| Model loading, GiB/rank | 11.08 | 8.20 |
| Automatic KV budget, GiB/rank | 12.68 | 15.76 |
| Logical KV tokens | 1,190,275 | 1,479,578 |
| Graph capture increment, GiB/rank | 0.33 | 0.26 |
| Idle worker NVML, MiB/rank | 26,114 | 26,030 |

Both idle snapshots have zero running/waiting requests and zero KV usage.
Automatic sizing turns the released memory into 289,303 additional logical
KV tokens, a 24.31% increase. NVML usage stays near 25.5 GiB/rank. Loading,
available KV budget and capture increments come from different profiling
stages; do not sum them as an exact allocation ledger or attribute every
budget difference to the exact 2.835693 GiB removed code storage.

**Output parity fails; production speed is not accepted.** The same MBPP28
prompt (135 input tokens), seed and sampling produce 260 control tokens and
754 shared tokens, each stable within its own warmup/three-repeat cohort.
The first difference is token 16 (one-based). MBPP0 produces 2105 versus 1093
tokens; both finish naturally with nonempty final answers. These differences
do not establish semantic degradation, but fail the deterministic gate.

Shared median pure decode is 221.40 tokens/s, round time 19.435 ms and
TTFT 112.37 ms. The different output sequences and acceptance lengths prevent
a matched-output throughput claim. Input text, launch script, seeds and the
six recorded native binary hashes are unchanged between arms. The existing
336 operator checks use same-build source controls. A further oracle against
the actual installed `_C` preparation/dispatch completes 36 projection/M
combinations: all six projections differ at M16/M32, while M1/M8/135/1024
match, including the gated cases reached. This exposes a native-version
validation gap that the same-build comparison could not detect. All 18 direct
comparisons at M9/16/32 match the installed dispatch to TurboMind bitwise,
while differing from current-source shared QPN2. Current Python's M<=32 log
does not prove that an older native binary selects QPN2 for those rows.

### Latest production allocation and source-aligned comparison

The 17:55/17:57 CST pair keeps the production contract above and loads both
legacy and shared QPN2 implementations from the same source-built sidecar.
Its optional `VLLM_QPN2_SHARED_ALIGN_NATIVE_CONTROL` build definition
overrides the legacy CUDA registrations only in this benchmark overlay.
Load the existing core DSO first, then the sidecar in every worker; loading
the sidecar alone leaves the old control dispatch active. Dispatch-table
inspection before and after importing `vllm._C` confirms that the owned
registration remains selected. A complete native rebuild naturally contains
both implementations and does not need this overlay.

| Latest recorded allocation | Dual layout | Shared layout |
| --- | --- | --- |
| Model loading, GiB/rank | 11.08 | 8.20 |
| Automatic KV budget, GiB/rank | 12.68 | 15.76 |
| Logical KV tokens | 1,190,275 | 1,479,578 |
| Graph capture increment, GiB/rank | 0.26 | 0.26 |
| Idle worker NVML, MiB/rank | 26,040 | 26,030 |

All four workers agree on the recorded per-rank values. Both snapshots show
zero running/waiting requests and zero KV usage. Idle worker usage is
25.430 versus 25.420 GiB/rank; under automatic sizing, saved weight memory
increases KV capacity rather than substantially decreasing total residency.
The 8.20 GiB loading measurement includes the target, draft and runtime
weight layouts; it is not the target checkpoint size divided by four.
The phase measurements are not a complete live tensor/allocator ledger.

Aligning native dispatch does **not** resolve model parity. MBPP28 returns
998 versus 297 tokens, with its first difference at token 155 (one-based).
Both arms repeat their own sequence across warmup and three measured
requests. MBPP0 returns 887 versus 760 tokens and first differs at token 287.
All outputs finish naturally with nonempty final answers. Thus the verified
old-binary route mismatch is not a complete explanation of the model result.
Keep the earlier mixed-native cohort separate; neither cohort passes the
deterministic gate.

For reproducibility, median pure decode in this latest pair is 222.82 versus
236.75 tokens/s; round time is 19.286 versus 19.234 ms and TTFT is 120.01
versus 121.19 ms. Different emitted sequences and acceptance behavior prevent
a matched-output speed claim. The next localization must compare actual
activations at the first-divergence prefix, preserving production KV,
context and sampling. Do not repeat unchanged full-model timing or adjust
those settings to conceal the discrepancy. Retain the default-off switch
and Draft PR pending model parity and broader acceptance.

The retained task artifacts, whose directory is recorded in the local handoff,
include
`source-aligned-summary.json`, `source-aligned-server-{0,1}/`,
`core-binary-oracle.json` and `core-tm-oracle.json`. The source-aligned DSO
SHA256 is `45a0dd65ec0c8d99adbe26bd1267479cbb64ba17013b2b026a5a6e34100306d9`.
It was built from source `608718f7a9` plus the optional sidecar registration
patch, recorded by diff hash in both manifests. Both services exit after
recording results and release their GPU locks.

An earlier fixed 2 GiB KV/8K/E5M2 diagnostic is excluded from production
conclusions. The first corrected control attempt exposes an old Flash-V100
extension without E4M3 precision revision four. Both production arms now
use a frozen copy of the production revision-four DSO, SHA256
`a751fed902279b0de23537c4aad2dc4fee360146d7fce7ef0c4f255a77f48b02`.
The failed and externally interrupted logs remain separate from the
completed control. Preserve the production runtime arguments when retrying.

The first sidecar build omitted `ENABLE_SM70_TURBOMIND`, hiding declarations
in `ops.h`; defining it fixes the build. The initial benchmark omitted the
stable activation library, so gated TurboMind dispatch failed to resolve
`silu_and_mul`; loading the matching stable library fixes the harness.
Neither failure was a shared-reader numerical discrepancy.

## Reproduce the focused operator check

Use an isolated worktree and owned GPU locks. The sidecar adds the two new
operators to an existing build and registers controls privately. Do not
load it into a build that already registers these operators.

```bash
export CUDA_HOME=/path/to/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions"
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
from torch.utils.cpp_extension import load
load(
    name="nvfp4_qpn2_shared",
    sources=[
        "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu",
        "csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu",
        "benchmarks/kernels/sm70_nvfp4_shared_sidecar.cpp",
    ],
    extra_cflags=["-O3", "-fvisibility=hidden", "-DENABLE_SM70_TURBOMIND"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math", "-lineinfo", "-Xcompiler=-fvisibility=hidden"
    ],
    extra_ldflags=["-Wl,-Bsymbolic"],
    is_python_module=False,
)
PY
CUDA_VISIBLE_DEVICES=0 VLLM_SM70_NVFP4_QPN2_M16_NATIVE=1 \
  .venv/bin/python benchmarks/kernels/benchmark_sm70_nvfp4_shared_weight.py \
  --model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --core-library /path/to/lib/_C.abi3.so \
  --shared-library "$TORCH_EXTENSIONS_DIR/nvfp4_qpn2_shared/nvfp4_qpn2_shared.so" \
  --json-out /path/to/task-artifacts/operator-results.json
```

The matching `_C_stable_libtorch.abi3.so` must accompany the core library.
The JSON records library hashes, the source base and diff hash, shard
shapes, bitwise comparisons, and both control timings.
