# Original-bit IQ3_S gated pairs on SM70

The small-M experiment combines QPN's gated-pair ownership and fused
SiLU/multiply epilogue with the shared `LatticeCompactDecoder` from #897.
The checkpoint is a mixed-format GGUF: a pair is admitted only when both
segments have a supported decoder. Unsupported pairs retain their existing
projection dispatch. The initial real pair is layer 6, whose gate and up
are both IQ3_S. Layer 5, for example, is IQ3_S plus IQ2_S.

## Storage and arithmetic

The experiment consumes an equal-byte permutation of original IQ3_S bits:
110 bytes per 256 weights. It stores no expanded coefficients or indices.
Original FP16 block scales and small scales are multiplied in FP32, using
the shared decoder. Reconstructed MMA operands are rounded to FP16 and
accumulated in FP32. Split partials are FP32. The fused epilogue preserves
the existing FP16 projection, SiLU, and multiply rounding.

Gate and up share a CTA. The initial implementation divides whole K blocks
among warps. A second candidate partitions K across CTAs; completion tickets
let the last CTA perform FP32 reduction and activation without a separate
reduce launch. Each graph/stream needs independent scratch and counters;
counters start at zero and the final CTA resets them before completion.

The staged candidate instead shares each complete K256 block among four
warps per projection. It prefetches the next original-bit block into
registers while decoding the current block in shared memory. This follows
PR #897's software pipeline; SM70 has no asynchronous copy instruction.

## Initial measurements

These are research extension measurements, not packaged runtime results.
No model dispatch has changed and no end-to-end run was performed.

Contract: one V100-SXM2-32GB from a TP4 system, rank-0 shards, FP16
activations, CUDA 12.8, Torch 2.10.0+cu128, graph replay, N=4352, K=5120,
M=8. Gate/up source bytes total 19,148,800. External CUDA events exclude a
16 MiB L2 eviction before each of 12 calls; results pool seven replays.
The NVFP4 comparison uses actual native checkpoint weights at the same
shape, totaling 25,067,520 code/scale bytes.

| Candidate | Median us | Weight footprint / time, GB/s |
| --- | ---: | ---: |
| Original-bit pair, dynamically indexed accumulator chains | 198.656 | 96.4 |
| Fixed accumulator chains, within-CTA split-K | 109.568 | 174.8 |
| Fixed chains, four CTA partitions, two K warps per projection | 97.280 | 196.8 |
| Whole-block shared staging, four CTA partitions | 101.376 | 188.9 |
| Canonical projections + cat + unfused PyTorch activation (diagnostic) | 96.256 | — |
| Existing native NVFP4 gated pair, split 16 | 48.128 | 520.9 |

All pair candidates pass the official FP32 GGUF dequantization plus FP32
GEMM oracle, followed by the existing FP16 activation contract. Maximum
absolute output difference is 0.00390625 and relative L2 is approximately
0.00050. The bitstream permutation independently reconstructs all original
index/sign packets. This does not establish a model quality result.

The initial canonical control uses a Python activation expression containing
multiple launches. It is a diagnostic comparator, not the model's one-kernel
`silu_and_mul` baseline. The next matched control uses that production
activation operator. A separate NVFP4 comparison consumes the existing
packed-input layout; its input preparation is explicitly outside timing.

NCU on the fixed-chain, split-4, prefetch candidate reports 136 CTAs,
256 threads/CTA, 64 registers/thread, 21.5% active warps, and approximately
42.6% long-scoreboard stalls. Its profiled duration is 140.93 us, DRAM
throughput 16.94%, and memory throughput 147.57 GB/s. Counter duration is
reported separately from the unprofiled graph timing table.

None of these candidates meets the 30 us gate. They therefore have no
accepted per-layer or per-round saving. Further decoder measurements and a
packaged runtime gate are pending; down, qkvz/a/b, and model integration
must follow a successful pair result.

Whole-block staging passes the same GEMM oracle and repeated graph counter
reset checks, but fails the speed gate. Its build uses 79 registers/thread,
17,284 bytes of shared memory, and no stack or spills.

An additional candidate moves each K32 subgroup's scale to the FP32 dot
product. The shared decoder reconstructs signed codebook integers exactly as
FP16 MMA operands; original scale products and the outer accumulation stay
FP32. A CPU algebra check on every real weight in both projections has zero
absolute error against official dequantization. The GPU pointwise decoder
also has zero error. At M8, subgroup postscale measures 94.208 us (203.3
weight GB/s), relative L2 0.00001797, and maximum difference 0.0009765625
against the FP32 GEMM/FP16 activation oracle. This is a separate arithmetic
candidate, not a claim of bitwise equivalence between differently ordered
FP32 reductions. It also fails the 30 us speed gate.

The matched production activation control measures 75.776 us. Native NVFP4
QPN2 measures 48.128 us; its shared TurboMind weight layout measures 49.152
us, and its prepacked-input diagnostic measures 44.032 us, excluding input
layout conversion. Prepacked IQ3 input does not close this gap.

NCU on subgroup postscale, P=4: 544 CTAs, 79 registers, 37.5% theoretical
and 32.92% achieved occupancy, 97.28 us profiled duration, 223.20 GB/s memory
throughput, and 62.32% compute throughput. The 842,649 excessive shared
wavefronts are concentrated in the two codebook reads: 404,121 and 403,712
at `gguf_lattice_compact.cuh` lines 131 and 132. These account for 95.9% of
excessive shared wavefronts. Read-only-cache codebooks and a byte-plane
packet permutation are the next isolated candidates. Their decoder formulas
continue to use the shared #897 functions.

The follow-up rejects both candidates: the byte-plane packet permutation
ties subgroup postscale at 91.136 us with P=5, and the global table regresses
to 104.448 us. Both pass numerical checks. A CPU model of the actual grid
indices reproduces 807,833 excessive wavefronts exactly. Three simple
bijective XOR bank permutations reduce that count by less than 1%, so no
GPU time is spent on them.

The next candidate interleaves 32 lane-specific copies of the 512-word
codebook, giving each lane its own shared-memory bank. It retains the shared
`LatticeRawDecoder::table_values` device function and the FP32 subgroup
scale product. The table uses 64 KiB of dynamic shared memory, alongside
23,440 bytes of static shared storage. Eight K warps per projection give
512 threads/CTA; the build uses 79 registers with no stack or spills.
Its occupancy and initialization cost must be measured before admission.

The replicated table passes the oracle but regresses to 143.360 us and is
rejected. Reusing #897 commit `c64a8d7c2f9fde305c96ae222442a36734b432fa`'s
exact FP16 operand interface also fails the gate: 100.352 us for weighted
operands and 106.496 us for subgroup postscale.

A literal QPN K16 loop retains the two interleaved FP32 accumulation chains
and replaces only weight reads/decoding. It uses 56 registers and measures
87.040 us. Prefetching the following two original-bit packet windows uses
58 registers and improves this to 84.992 us, or 225.30 weight GB/s. Maximum
output difference is 0.00390625 and relative L2 is 0.00049622. No source
scale is rounded into an expanded coefficient; the shared decoder's final
FP16 operands equal FP32 official dequantization rounded to FP16.

NCU on this last candidate: 136 CTAs of 512 threads, 58 registers, 50%
theoretical / 43.34% achieved occupancy, 92.70 us profiled duration, 226.66
GB/s memory throughput, and 33.17% long-scoreboard stalls. Excessive shared
wavefronts remain 842,649 (69% of 1,227,801 total). Register reduction alone
does not remove the lookup bottleneck. This candidate still loses to the
existing 75.776 us canonical control and is not admitted to model dispatch.

The checkpoint has eight pure IQ3_S pairs: layers 6, 23, 24, 25, 46, 51,
55, and 56. Forty pairs mix formats. Any eventual round-saving calculation
must count the admitted pairs rather than multiply a single type's saving
by all 64 layers.

The follow-up with 4, 8, and 16 interleaved codebook copies retains 58–59
registers and 24/32/48 KiB shared storage. It passes the same oracle but
measures 97.280 / 98.304 / 102.400 us, so partial replication is also rejected.

Instruction sampling on the literal prefetch kernel additionally attributes
733 long-scoreboard samples to the first block-scale conversion and 498 to
the first MMA consuming a newly read activation. These are sampled stalls,
not durations. Follow-ups independently prefetch block metadata/activations
and replace the warp packet reader with coalesced direct packet loads.
Neither changes the original weight footprint or shared decoding formula.
Each new cohort includes the production canonical activation and native
NVFP4 controls. Results are pending; no candidate is admitted.

Direct 32-bit packet loads pass an independent CPU check of all 5,570,560
original index/sign packets, without changing storage size. They measure
80.896 us, tied with the same-cohort canonical control; native NVFP4 measures
49.152 us. NCU reports 77.056 us separately, 43.60% achieved occupancy,
58 registers, and 39.8% long-scoreboard stalls. Sampling now concentrates on
the packet shift and first activation-consuming MMA; metadata conversion
has only 17 not-issued samples.

Full metadata/activation prefetch measures 95–96 us with 74 registers. Its
1024-thread variant exceeds the SM register budget and cannot launch. Full
shared metadata caching also regresses to 95.232 us. These paths are rejected.

Separating packet loads from extraction measures 77.824 us. Activation
prefetch with two accumulator chains raises register use to 69 and regresses
to 91.136 us; one FP32 accumulator chain stays at 63 registers and measures
75.776 us, equal to the same-cohort canonical control. Native NVFP4 is
48.128 us. No variant has passed the 30 us gate or delivered an accepted
model improvement.

The next controlled comparison copies the QPN weight-load cache policy:
streaming (`__ldcs`) versus L1 bypass (`__ldcg`). The original decoder uses
ordinary cached reads. These cache policies apply only to packet loads;
shared decoder arithmetic, source scales, and output rounding are identical.

Streaming packet loads measure 76.800 us with activation prefetch and one
FP32 accumulator chain; L1 bypass measures 79.872 us. The same-cohort
canonical control is 79.872 us and native NVFP4 49.152 us. This small local
difference does not pass the gate. Loop unroll hints of 2/4/8 produce the
same instruction count (504 or 520 instructions by accumulator policy)
and the same 76.800 us measurement, so they do not implement deeper K stages.
An explicit K64 chunk instead hoists the common block metadata read and
emits four K16 bodies; machine-code verification precedes any speed claim.

Explicit K64 chunks improve the cold-L2 pair to 70.656 us (271.01 weight
GB/s), with one FP32 accumulator chain and activation prefetch; the
matched canonical control is 80.896 us and native NVFP4 is 49.152 us.
Maximum difference is 0.00390625 and relative L2 is 0.00049515. This is a
10.240 us local difference, still below the required performance goal and
not admitted to model dispatch. The four K16 bodies are actually present
in generated code, using 64 registers and no spills.

A separate FP32 shared-reduction candidate pairs K warps before the final
activation reduction. It reduces static shared storage from 18,432 to
10,240 bytes, retaining 56/64 registers and no spills. A 32 KiB preferred
shared-memory carveout should retain two CTAs while increasing L1 capacity;
its extra CTA barrier and changed FP32 reduction order require measurement.
It introduces no separate reduction launch or lower-precision partials.

Paired shared reduction passes the oracle but measures 75.776/76.800 us
without/with activation prefetch. The matched canonical control is 79.872
us and native NVFP4 48.128 us. It does not establish a meaningful improvement
and is not promoted. All partials remain FP32.

The retained CUDA source is the explicit K64 research prototype. It is
excluded from the normal CMake build and registration is guarded by
`GGUF_IQ3_PAIR_RESEARCH`. It depends on the shared decoder interface from
PR #897; there is no model route or packaged-runtime performance claim. A
separate codebook-cache experiment factors the existing half decoder's
sign/scale finish into one common function rather than duplicating quantizer
math. Both decoder forms call that same function, and only the 512 codebook
entries are cached as exact half integers; source weights remain unchanged.
The API proposal is recorded on #897 pending numerical and speed evidence.

Exact half-integer codebook caching passes the GEMM oracle but measures
78.848 us versus the same-cohort canonical 79.872 us and NVFP4 49.152 us.
It is slower than the K64 prototype, so its proposed API refactor is not
required for the retained path and will not be promoted on this evidence.

The two-CTA K64 partition experiment also fails to improve the retained
kernel: best 74.752 us, versus same-cohort canonical 82.944 us and NVFP4
48.128 us. Replayed graphs reset the completion counters correctly.

NCU on retained K64 reports 66.656 us separately, 64 registers and 41.41%
achieved occupancy. Excessive shared wavefronts remain 842,649. Not-issued
sampling totals include long scoreboard 214, math-pipe throttle 134,
no-instruction 109, and short scoreboard 30. The largest remaining sampled
waits are at packet extraction and the scale-field shift/read. These counts
are not duration estimates. A follow-up stores the four-byte IQ3_S scale
field in `uint32_t` (IQ2_S retains `uint64_t`) and compares K32/K64 chunks;
it changes neither the source bits nor the numerical formula.

The four-byte scale-field specialization passes bitwise comparison with
both original K64 accumulator policies. At M8, K32 and K64 both measure
61.440 us (311.67 source-weight GB/s), versus same-cohort canonical 75.776
us and native NVFP4 48.128 us. Official GEMM-oracle relative L2 remains
0.00049515; maximum output difference remains 0.00390625. The original
source bitstream and arithmetic are unchanged. The common decoder change
has been reported on #897 and remains a research dependency until packaged
validation; it still does not pass the 30 us gate.

The next isolated pipeline test compares one versus four K16 packet windows
in registers, with one or two FP32 accumulation chains. Its four-window,
single-chain build uses 59 registers and no spills; the two-chain build
uses 67 registers. Both retain the original byte-codebook interface.

Follow-up measurement caveat: the identical retained scale32 K64 binary
measures 70.656 us in the deeper-prefetch cohort, versus 61.440 us in the
preceding cohort. Its canonical control also changes from 75.776 to 82.944
us while NVFP4 stays near 48.128 us. The scale-storage change is numerically
bitwise equivalent, but its isolated speed effect is not established by
these separate cohorts. A same-process old/new/new/old comparison with GPU
clock and temperature records is pending before attributing the difference.

Four-window packet lookahead is bitwise equivalent to one-window lookahead
but measures the same 73.728 us with one accumulator chain. Two chains
raise register use above 64 and regress to 86.016 us. Deeper lookahead is
therefore rejected in this form; no model route has changed.

The matched old/new/new/old scale-field test resolves the attribution:
62.464 / 61.440 / 61.440 / 61.440 us at recorded SM 1530 MHz, memory
877 MHz and 41 C. Outputs are bitwise equal. Narrower scale storage has no
established independent speed benefit, so the earlier separate-cohort
approximately 9 us difference must not be attributed to that change. The
retained original-interface pair is about 61–62 us at this workload, versus
canonical 75.776 us / native NVFP4 48.128 us. It still misses the 30 us goal;
no scale-width or model-route change is promoted on these data.

Byte-plane reads on the weighted K64 skeleton pass official elementwise
FP32 dequantization and bitwise comparison with tightly packed packets.
At recorded SM 1290 MHz / memory 877 MHz, single-chain byte planes measure
96.256 us versus tightly packed 73.728 us; two-chain byte planes measure
113.664 us versus tightly packed 71.680 us. Native NVFP4 is 49.152 us and
canonical 79.872 us in that cohort. The layout is rejected on this skeleton
as well. Future speed comparisons must include the observed clock state.

Using the same even-octet scale position for both K8 operands of a K16
segment is bitwise equivalent for IQ3_S. At recorded SM 1290 MHz, matched
single-chain K64 improves from 70.656 to 67.584 us; two-chain K64 improves
from 73.728 to 68.608 us. Native NVFP4 is 49.152 us and canonical 79.872
us. This saves redundant coefficient work while reusing the shared decoder,
but still misses the performance gate.

Joint K16 packets preserve all 110 bytes per K256 block: three adjacent
32-bit reads replace four reads. The all-weight official FP32 oracle passes,
and GEMM outputs are bitwise equal. At SM 1530 MHz, however, the joint
reader measures 62.464 us versus separate K8 packets 61.440 us. Two-chain
versions both measure 65.536 us. Canonical is 76.800 us and native NVFP4
48.128 us. This permutation is rejected.

Explicitly interleaving each K8 decode with its two MMA instructions does
not improve the best full-activation single-chain path: both measure
67.584 us at SM 1290 MHz, with bitwise equal outputs. The compiler already
interleaves parts of the original source. Both builds retain 64 registers
and zero spills; the source scheduling change is rejected.

The shared decoder's newer 32-bit warp-window reader is also tested on this
K64 pair skeleton, using unchanged original-bit storage and dequantization.
Its all-weight official FP32 oracle passes and GEMM outputs are bitwise
equal. At SM 1290 MHz, full-activation window reads measure 73.216 us
versus direct reads 67.584 us; two-chain window reads measure 72.704 us
versus direct reads 69.632 us. Canonical is 79.872 us and native NVFP4
49.152 us. Retain direct reads for this skeleton.

NCU on the retained shared-coefficient full-activation candidate reports
65.664 us profiled duration, 64 registers, 18,432 shared bytes and 40.54%
achieved occupancy. Long-scoreboard samples cluster at scale extraction
(101 samples) and original-packet extraction (97 samples); the total is
264 not-issued samples. Excessive shared wavefronts remain 842,649.
These sample counts are instruction attribution, not elapsed-time shares.
The next isolated step removes unnecessary row masks in an M=8-only
candidate before considering metadata prefetch within the register budget.
