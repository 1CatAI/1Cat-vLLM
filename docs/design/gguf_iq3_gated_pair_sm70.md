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

An M=8-only row specialization removes masked activation loads and output
checks, narrowing admission to exactly eight rows. It rejects M=7 before
launch. Full-activation registers fall from 64 to 56 without spills; outputs
remain bitwise equal. At SM 1290 MHz, matched timing improves from 67.584
to 65.536 us. This does not satisfy the 30 us gate.

One-block-ahead original metadata uses 60 registers on that single-chain
specialization, without spills. It is bitwise equivalent but measures
67.584 us versus its matched current-metadata control 66.560 us. Reject
metadata lookahead in this form. Its two-chain version uses 67 registers
and is skipped because it crosses the established occupancy boundary.

Byte-domain sign restoration is exhaustively checked on all IQ3_S and
IQ2_S codebook entries and all 16 four-byte sign patterns. Real weights of
both types match official dequantization bitwise in FP32 and FP16. The
best M8 single-chain pair still measures 65.536 us, equal to its control;
registers fall to 48 without spills. No shared decoder API is promoted on
this standalone timing result.

A pipeline with raw packets two segments ahead and decoded operands one
segment ahead initially uses 78 registers with activation prefetch, so it
is not measured. Without activation prefetch, a single-chain variant uses
66 registers. A two-CTA launch bound lowers that variant to 64 registers
without spills; its activation-prefetch counterpart spills and is skipped.
The bounded variant is bitwise equivalent but measures the same 66.560 us
as its matched raw-lookahead control. Reject this pipeline.

The retained owned source now dispatches an unmasked row specialization
only for actual M=8, preserving masked loads for M1–7. A fresh build of the
owned source passes eager and three graph replays at M1/2/4/7/8, with
bitwise equality to the original shared-coefficient pair. Its M8 cold graph
median is 65.536 us. It remains research-only, without model dispatch.

The next optimization sequence targets instruction issue. Static inspection of
this M8 loop reports approximately 138 instructions per K16 segment, including
74 integer instructions, 32 FP16 instructions and 16 HMMA instructions.
Register lookahead, decoded-operand pipelines and instruction interleaving are
stopped: none removes this instruction work. The revised gate+up target is
40 us under a matched cold-L2 graph cohort, with native NVFP4 measured in the
same process. Earlier 30 us targets above describe the previous experiments.

First verify issue activity and executed/issued instructions with NCU on the
retained M8 source. Then test a 32 KiB shared signed IQ3_S codebook; hoisted
pointer increments with aligned, warp-interleaved 16-byte original-bit records;
block-level half2 scale reuse; and an 80/160-CTA persistent scheduler. Each
step requires static loop instruction counts, an official dequantization oracle
and a matched graph timing. Expanded FP16 scale rounding needs separate error
and quality evidence before use. Until then the exact two-stage operand
formation and FP32 accumulation remain in force. QKVZ with FP16 alpha/beta is
being developed separately and does not wait for the gated-pair target.

The retained M8 NCU run executes 13,168,064 warp instructions and issues
13,188,535. With 2,176 launched warps, this is 6,051.5 executed instructions
per launched warp, including initialization and the epilogue. Issue activity
is 59.02% per active cycle and 50.99% of peak over elapsed cycles. Profiled
duration is 63.744 us, registers 56 and shared memory 18,432 bytes. These
counters show substantial instruction work, without establishing saturated
issue slots over the whole kernel.

The signed book preserves every actual gate/up weight bitwise in both FP32
and FP16, and the gated output is bitwise equal. Its complete K64 backedge
contains 478 instructions (119.5 per K16), versus 589 (147.25 per K16) for
the retained exact-M8 specialization. Counts include control and padding
instructions. It uses 50 registers, no spills and 49,152 shared bytes;
preferred shared carveout is 96 KiB. Matched SM1290/MEM877 cold graph ABBA
is retained/signed/signed/retained = 66.560/73.728/73.728/65.536 us.
The signed book regresses despite lower loop instruction count.

An equal-byte aligned layout groups each lane's K128 low indices and signs
into three 16-byte records, interleaved by warp. High index bits and original
metadata remain in separate planes. An independent inverse recovers every
source byte; each projection remains 9,574,400 bytes. All actual weights are
bitwise official in FP32/FP16; the gated output is bitwise retained. Its K128
backedge has 746 instructions (93.25 per K16), 60 registers and no spills.
The matched signed/aligned/aligned/signed cohort measures
75.776/70.656/69.632/73.728 us; retained is 65.536 us and native NVFP4
51.200–52.224 us. It remains slower than retained (source bandwidth
271–275 versus 292 GB/s). A scoped three-variant NCU run is required to
localize the regression before accepting either decoder/layout.

The three-variant NCU comparison confirms reduced executed instructions:
retained 13.168 million, signed 11.291 million, aligned 9.025 million.
However elapsed issue activity falls from 50.40% to 39.02% and 33.97%.
Profiled durations are 62.528/70.528/67.040 us respectively. The signed
variants request and receive 96 KiB shared carveout with two CTA occupancy
limits; the retained variant uses 64 KiB. Long-scoreboard not-issued samples
rise from 331 to 1,033 for signed, including 229 in signed-table initialization.
Aligned has 529 long-scoreboard samples (202 at initialization) and 517
LG-throttle samples, predominantly at vector/activation loads. These are
sample counts and cannot be converted directly into elapsed-time shares.

An exact base-half2 cache, Type21 u32 scale field and direct high-byte PRMT
selectors lower the K128 loop from 746 to 711 instructions (88.875 per
K16), with 52 registers and no spills. Every actual FP32/FP16 weight and
GEMM output remains bitwise equivalent. Matched timing is unchanged at
70.656 us. A generated fixed signed table replaces per-CTA sign computation
with coalesced uint4 copies; its matched timing is 67.584 us, versus retained
65.536 us and NVFP4 51.200–52.224 us. Neither is accepted. There is no
FP16 combined-scale rounding in these candidates.

A 13-bit signed-index record permutation folds the original 9-bit index and
4-bit sign nibble into a direct signed-book index. Each lane reads three
aligned uint4 records plus a u32 tail per K128, still exactly 52 payload
bytes. Fixed PRMT windows extract crossing fields without dynamic
26-bit funnel shifts. The independent inverse recovers every source byte.
All actual FP32/FP16 weights are bitwise official, and gated outputs are
bitwise retained. Its K128 loop is 562 instructions, **70.25 per K16**, with
51 registers, 32 KiB shared union and no spills. Matched ABBA is
separate-index/sign 66.560, signed-index 62.464, signed-index 62.464,
separate-index/sign 66.560 us; retained is 65.536 us and native NVFP4
51.200–52.224 us. Source bandwidth is 306.56 GB/s. It reaches the
instruction-count objective but still fails the 40 us latency target.

The union allows codebook storage to become partial-sum storage after a
CTA barrier. A previous 67% carveout request actually selected 96 KiB on
Volta; it must not be described as measured 64 KiB. NCU verifies that 66%
selects 64 KiB. Matched graph timing changes only from 67.584–68.096 to
66.560 us, without beating the retained source on its own.

NCU on signed-index K128 executes 6.541 million warp instructions, versus
7.518/7.596 million for native NVFP4 with one/two accumulator chains. NVFP4
uses 1,024-thread CTAs and 32 KiB carveout, versus 512 threads and 64 KiB
for this IQ3 path. Profiled duration is 59.008 versus 47.808/48.288 us;
elapsed issue activity is 27.18% versus 39.23%/39.01%. Actual profiled DRAM
throughput is 356.02 versus 535.50/530.16 GB/s. These counters reject total
instruction count as the sole remaining explanation. The native single-chain
control also rules out accumulator-chain count as the primary remaining gap.

K64 direct signed-index records use one uint4, one uint2 and a uint16 per
lane, preserving 26 payload bytes. Official all-weight FP32/FP16 dequantization
passes; split8 gated outputs are bitwise retained. Its loop contains 301
instructions (75.25 per K16). Matching the native streaming weight-load hint
preserves numerical results but has no timing benefit for K128: both cached
and streaming measure 62.464 us. K64 split8 measures 63.488 us; split16
cached/streaming is 67.584/66.560 us. The isolated split16 carveout check requests 32 KiB instead of 64 KiB
without changing decoder, layout or precision. Matched ABBA is
66.560/67.584/67.584/67.072 us, so it is rejected as well.

## Signed-index source checkpoint

The research-only `gguf_iq3_signed_pair_sm70.cu` records the measured
70.25-instruction/K16 implementation separately from the retained packet
kernel. `gguf_iq3_signed_records.py` contains the same-byte host permutation
and independent full inverse check. The common raw/compact decoder headers
come from #897 at `c64a8d7c2f9fde305c96ae222442a36734b432fa`, with signed
lookup, exact base caching and fixed PRMT selectors; scale decoding stays in
that shared family. The fixed signed book retains the original MIT provenance.
No normal build registration or model dispatch is enabled by this checkpoint.

In the latest NCU source attribution, 916,060 excessive shared wavefronts
are concentrated in signed-book LDS instructions. Long-scoreboard not-issued
samples include 122 at signed-book initialization and 117 at the first
index-to-book address dependency; total samples are 577. LG-throttle has
435 samples, including both activation and payload vector loads. These
counts identify dependencies for isolation, not additive time fractions.
The next isolated test changes only signed-book placement to the read-only
cache, removing per-CTA initialization and shared bank serialization while
retaining the aligned payload, exact operands and FP32 accumulation. The
earlier unsigned-book global-table result cannot validate this signed-index
implementation, which has a different instruction count and lookup footprint.

The signed-book read-only-cache isolation passes bitwise retained outputs but
regresses to 99.328 us in both ABBA arms, versus shared-book 62.464 us and
native NVFP4 51.200–52.224 us. Its loop has 78.375 instructions/K16,
54 registers, 16 KiB partial storage and no spills. It is rejected.

A CPU bank model over all 348,160 actual gate/up warp lookups predicts
881,244 codebook excessive wavefronts. Adding the measured 34,816 reduction
wavefronts exactly reproduces NCU's total 916,060. Searching all 1,287 choices
of five bank bits on a sample and validating the six best over the full
data reduces conflicts by only 0.042%; no GPU experiment is justified.
The next CPU/build isolation encodes each signed odd grid value in a nibble,
reducing the signed book from 32 to 16 KiB without changing source records
or numerical precision. Exact integers remain in [-15, 15]. Reconstructing
(q - 7.5) times twice the original small scale preserves the exact
intermediate; original d remains separate and accumulation stays FP32.

The nibble-book arithmetic check exhausts all 63,488 finite FP16 d bit
patterns, all 16 signed odd grid integers and all 16 small scales. All
16,252,928 final FP16 results match the official FP32 formula rounded to
FP16 bitwise, including overflow and signed zero. This transformation
introduces no combined-scale rounding. The first compiled candidate has
74.25 instructions/K16, 52 registers, 16 KiB shared storage and zero spills.
Its actual-weight device oracle fails on exactly half the elements before
timing: the 1031.5 FP16 low-nibble bias rounds to 1032. The corrected path
uses only the exact 64/71.5 construction and shifts low nibbles by four.
It has 78.125 instructions/K16, 51 registers and zero spills. The actual
all-weight device FP16 oracle passes bitwise, and outputs are bitwise retained.
On a second V100 host at the same 1290/877 MHz, matched shared32/nibble16/
nibble16/shared32 graph timing is 62.464 us in every arm; native NVFP4 is
51.200–52.224 us. The smaller codebook has no independent timing gain.
Both shared and individual GPU locks are held for this microbenchmark;
the first host queue is canceled before the second-host launch.

A two-N32-tile, two-K-partition persistent candidate keeps the signed book
live across tasks instead of reinitializing it. Its 73-register build cannot
keep two 512-thread CTAs resident and is not GPU-tested. A bounded-32-bit
address/launch-bound build is the next compilation gate. This is a scheduling
experiment; the rejected register-prefetch/pipeline variants stay disabled.

Bounded 32-bit task addressing and a 512-thread/two-CTA launch bound reduce
the persistent candidate to 64 registers without spills. The inner K loop
is 78.375 instructions/K16; the outer task backedge includes epilogue and
atomic completion work and must not be mislabeled as the dequant/MMA loop.
Both fixed grids pass official GEMM, changed-input graph replay and counter
wrap/reset checks with FP32 accumulation. In the matched second-host
cohort, fixed160 measures 62.464 us twice; fixed80 measures
65.536/66.048 us. The shared32 control changes from 62.464 to 65.536 us,
so there is no accepted stable gain. Native NVFP4 remains 51.200–52.224 us.

The next CPU/build isolation places the same signed book behind a texture
object, testing the separate TEX lookup queue against shared LDS and the
already-rejected ordinary read-only LDG implementation. Host initialization
is outside capture/timing; records, scale arithmetic and accumulation are
unchanged. This remains research-only, with no normal dispatch change.

Texture lookup compiles to 74 instructions/K16, 59 registers, 16 KiB
partials and no spills. The actual texture object returns all 8,192 book
words bitwise, and gated outputs are bitwise retained. Matched ABBA is
shared 63.488, texture 106.496, texture 106.496, shared 62.464 us; native
NVFP4 is 51.200–52.224 us. TEX lookup is rejected.

K32 postscale with explicit output-column scales and exact integer MMA
operands spills at the 64-register launch bound. Caching original d as
half2 and removing redundant first-MMA zero initialization still spills;
neither candidate is GPU-timed.

The next decoder arithmetic isolation replaces `(biased - 1152) * small`
with `hfma2(biased, small, -1152 * small)`, retaining the separate original
d multiply. Every negative bias product is exactly representable in half
for both Type21 and Type22. Exhaustive 256 biased bytes ×16 small scales
match the retained half intermediate bitwise for both types. The compiled
IQ3_S loop has 62.625 instructions/K16, 50 registers, 32 KiB union and
no spills. The Float decoder formula stays unchanged. Every actual Float/Half
weight matches official dequantization bitwise; gated outputs also match
retained bitwise. Matched ABBA is retained 63.488, HFMA 62.464, HFMA
62.464, retained 63.488 us. This cohort has a 1.024 us difference, but
HFMA does not improve the earlier 62.464 us best. Native NVFP4 is
51.200–52.224 us; the 40 us target remains unmet.

A scoped Nsight Compute 2022.4.1 run on the second host confirms 5,877,376
executed warp instructions for HFMA, 28.48% active-cycle issue activity,
59.584 us profiled duration and 352.28 GB/s DRAM throughput. Read-only
lookup executes 7,161,216 instructions with 21.25% issue activity and
97.344 us profiled duration. Its L1 sector hit rate is already 86.72%,
versus 55.22% for shared HFMA. However global-load sectors increase from
2,092,883 to 12,940,961; cache misses increase from 931,524 to 1,757,023.
High hit rate alone cannot explain or fix the random-gather request load.
A streaming-weight/smaller-carveout control is still needed to bound the
cache-policy contribution. These counter durations remain separate from
the unprofiled cold-L2 graph timings.

The parallel qkvz/a/b N32, sixteen-K-warp candidate compiles to
68 instructions/K16, 52 registers and no spills. It uses 128 quantized
CTAs plus 96 existing FP16 ba CTAs. Actual Float/Half dequantization and
all qkv/z/b/a outputs match the earlier 224-CTA implementation bitwise.
Matched old/new/new/old is 41.984/37.888/37.888/43.008 us; native NVFP4
quantized qkvz is 27.648 us, or 34.816 us with separate ba. The current
37.888 us candidate still fails its 20 us target. Only eight GDN layers
have both quantized segments in the supported type, so the local saving
corresponds to 0.032768–0.040960 ms per round, not a full-model result.

The bounded cache-policy control retains the read-only signed table and
changes weight loads to streaming, with a 33% carveout request. It has
78.375 instructions/K16, 54 registers, 16 KiB partial storage and no
spills. Outputs remain bitwise retained. Matched read-only-cached/stream/
stream/read-only-cached is 100.352/92.160/92.160/100.352 us, versus
shared 62.464 us and native NVFP4 51.200–52.224 us. Cache policy explains
part of the read-only regression but does not make it competitive.
The shared decoder remains the research candidate. No optimized route is
enabled in the normal model runtime by these source checkpoints.
