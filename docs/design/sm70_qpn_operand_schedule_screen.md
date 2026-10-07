# SM70 QPN operand and collective scheduling screens

These are independent research extensions. No serving route or precision is
changed, and their timings are not end-to-end model results. The controls are
extracted from the owned source and checked against the installed operators.

## Measurement contract

V100-SXM2-32GB on a fully NV2-connected TP4 group, 300 W, 1290 MHz SM and
877 MHz memory; Torch 2.10/cu128, CUDA toolkit 12.8. Original unsloth target
weights, rank-local layer 0, M8 FP16 activations and FP32 residual/state.
CUDA Graph measurements evict 128 MiB before external timing events and retain
200 paired samples after warmup. All five GPU file locks are acquired before
the process check. Compiler caches and extensions are isolated from serving.

## Effective-scale rejection

The FP4 decoder retains its two FP16 multiplication boundaries. A 512-byte
lookup stores the exact effective scale for each E4M3 code; a second candidate
stores effective FP16 scales beside the unchanged codes. Both match the
installed MLP's output bits at amplitudes 0.01, 0.1, 1 and 4.

| Complete MLP | Control us | Candidate us | Saving us, 95% interval |
| --- | ---: | ---: | ---: |
| Effective-scale lookup | 68.500 | 72.986 | -4.485, -4.777 to -4.234 |
| Inline FP16 scales | 68.833 | 71.449 | -2.616, -2.939 to -2.278 |

Lookup adds a dependent load. Inline FP16 scales enlarge each bundled group
from 288 to 320 bytes. Neither clears the full-kernel speed gate. Keep the
existing effective-scale arithmetic and bundled layout.

## Paired gate/up warps

Each warp computes both projections, sharing the activation registers while
preserving the original eight K slices, per-projection FP32 accumulation order,
ordered partial reduction, FP16 gate/up rounding, SiLU and FP16 multiply.
Sequential and interleaved schedules use 256 threads rather than 512, 80
registers per thread and 16 KiB shared memory, without spills or extra weights.

| Complete MLP | Control us | Candidate us | Saving us, 95% interval |
| --- | ---: | ---: | ---: |
| Sequential projections | 68.337 | 65.971 | 2.365, 2.207 to 2.524 |
| Interleaved HMMA | 68.705 | 66.058 | 2.647, 2.309 to 2.970 |

The sequential version also passes a complete real-weight GDN layer graph.
Outputs, FP32 residual, rollback states and convolution history match bitwise
on all four ranks at amplitudes 0.01, 0.125, 1 and 4. Each graph retains 11
kernel nodes, including the eviction outside the timed range. The timed layer
contains ten compute nodes in both arms.

Taking the maximum rank event duration for each paired sample gives
159.134 to 157.696 us, saving 1.438 us with a 95% bootstrap interval of
0.997 to 1.884 us. This is smaller than the isolated MLP saving. Extrapolating
48 GDN layers gives about 0.069 ms; the other eight NVFP4 MLPs still require
attention-layer admission. Do not substitute the optimistic isolated estimate
for a measured full round or promote this alone through a serving rebuild.

## Collective rejection

A direct-pull candidate retains the existing forty-CTA geometry, five CUB128
partials per row and ordered norm reduction. System-visible start/end peer
handshakes protect input visibility and reuse. It reads independent CUDA IPC
inputs directly, without casting the installed custom-allreduce class across
an extension ABI boundary. A second candidate retains push but uses the local
input in registers, omitting its local payload store/load/reset.

Both match output and residual bits for four amplitudes. Single cold graph
calls include rank launch skew, so a second screen uses 64 consecutive
collectives per graph. Maximum-rank means per call are:

| Route | Time us |
| --- | ---: |
| Original push + norm | 8.110 |
| Direct pull + norm | 20.540 |
| Push with local operand retained | 8.598 |

Both candidates fail. The local-operand variant regresses by 0.488 us with
95% interval 0.448 to 0.529 us. The direct-pull protocol does not remove the
latency cost of its visibility and completion handshakes.

## GDN normalization-cache rejection

A separate candidate lets convolution own a complete K128 head and prepare
normalized Q/K in FP32 for the unchanged BV2 recurrence. It needs no producer
flags or grid synchronization. Complete-layer timing regresses by about
1.8 us; the extra cache and preprocessing offset the removed repeated work.
Convolution-history bits match, while state and output bits can differ because
the normalization reduction layout changes. This fails speed admission, so no
teacher-forcing or serving rollout is justified for this candidate.

## FP8 two-phase rejection

Another screen halves physical warp count while retaining the original logical
K slices, two FP32 accumulator chains and ordered partial reduction. Each warp
finishes two slices sequentially; there are no extra splits or weight padding.
Both projections match the installed operator bitwise at four amplitudes.
Cold-L2 qkvz regresses 35.635 to 44.165 us and GDN out regresses 17.224 to
20.849 us. The register/shared-resource reduction does not compensate for the
longer per-warp serial work. Reject before adding fused a/b or model tests.

## Projection epilogue and rollback-state screens

A qkvz+a/b epilogue absorbs convolution, gating and core-output zeroing. It
retains the original rounded FP16 projection values before convolution and
leaves the BV2 recurrence and gated norm separate. Compiler allocation is 56
registers and 16.5 KiB shared, without spills. Complete TP4 layer timing is
160.333 to 158.894 us: saving 1.439 us, 95% interval 0.788 to 2.084 us.
Timed graph compute nodes fall from ten to nine. Convolution history is
bitwise; layer FP16 output differences reach 0.001953 and state differences
reach 1.8e-7. This small gain is not numerically admitted or deployed. A
teacher-forcing gate would be required before any serving promotion.

A separate hybrid saves the first four full FP32 state snapshots and exact
tail rank-one factors. Tail recovery explicitly rounds decay multiplication
before the original FMA. Outputs and all eight recovered states are bitwise
for accepted counts one through eight. At accepted count four, complete TP4
layer timing is 160.251 to 159.903 us; the saving interval, -0.635 to 1.249 us,
crosses zero. Kernel count is unchanged. The research harness still allocates
all original slots; no actual allocator-memory saving or speed admission is
claimed.

## Resident GDN fusion rejection

Eighty cooperative CTAs retain 768 independent two-column warp tasks and
parallelize all output-head norm rows after one resident-grid barrier. Both
the Q/K-cache and no-cache variants launch inside CUDA Graphs and reduce the
timed compute-node count from ten to nine. They use 48/46 registers without
spills; caching reserves 24.2 KiB shared memory.

Per-sample maximum-rank layer time regresses 161.546 to 191.539 us with cache,
and 159.068 to 188.969 us without cache. Both regress by approximately 30 us.
These measurements reject the implemented cooperative schedule; they do not
isolate the grid barrier as the sole cause. State/output rounding differs,
while convolution history remains exact. No model quality or rollout test is
warranted after this speed failure.

## Further scale-conversion rejections

Caching effective scales once per CTA avoids the previous repeated global
lookup, but still fails. A 512-byte half table regresses the complete MLP
68.613 to 69.944 us; a 1-KiB half2 table regresses 68.659 to 70.067 us. Both
preserve bits at four amplitudes.

Directly constructing FP32 E4M3 scale bits removes the intermediate FP16
decode conversion, retaining the global FP32 multiply and FP16 rounding.
All 256 scale bit patterns match the original arithmetic at the real global
factors and four additional factors. Projection outputs match at four input
amplitudes. Gate/up regresses 47.252 to 48.056 us, down is indistinguishable
at 25.917 to 25.938 us, and the complete MLP regresses 68.424 to 69.007 us.
Reducing conversion instructions alone does not improve the complete kernel.

## Norm partial-packet scheduling

The original push protocol remains unchanged. Each norm CTA publishes its
variance and generation in one aligned 64-bit packet, waits for the row's
partials, and computes its own ordered sum. This removes the separate leader
inverse packet and partial resets. Metadata for the screen is separate from
the original norm metadata; the common payload generation is retained.

| Geometry | Max-rank us/call, 64-call graph | Saving us, 95% interval |
| --- | ---: | ---: |
| Five parts, 40 CTAs, original arithmetic | 8.036 to 7.753 | 0.283, 0.260 to 0.308 |
| Ten parts, 80 CTAs | 8.023 to 7.302 | 0.721, 0.696 to 0.747 |
| Twenty parts, 160 CTAs | 8.022 to 7.135 | 0.887, 0.849 to 0.925 |

Five parts pass bitwise output/residual checks at four amplitudes. Wider
geometries retain bitwise residuals but change the variance grouping; norm
differences reach 0.000977. Complete TP4 GDN-layer checks retain exact state
and convolution history. Ten parts improve 160.113 to 158.449 us, saving
1.664 us with interval 1.147 to 2.217. Twenty parts improve 160.558 to
159.002 us, saving 1.556 us with interval 0.840 to 2.268. Both retain ten
timed compute nodes. The twenty-part isolated win does not establish a
complete-layer advantage over ten parts. Neither is teacher-forcing or
end-to-end admitted yet.

## Warp-local publication rejection

A new down epilogue gathers final FP16 output packs with warp shuffles rather
than the previous shared staging and extra CTA barrier. It writes peer push
payloads directly and omits the raw global output store. Consume-only norm
uses twenty partial packets. The existing split reduction, FP4 decoder and
FP16 projection rounding are unchanged; wider norm grouping remains numeric.

Complete TP4 layer timing regresses 160.609 to 167.670 us, saving -7.060 us
with interval -7.639 to -6.497. State and convolution history remain bitwise.
Both arms retain ten timed compute nodes. Removing the staging barrier did
not make publication worthwhile; reject this implementation.

## Activation-cache and address-layout screens

Identical original projection kernels are compiled under distinct symbols to
compare CUDA's automatic shared/L1 carveout with 25% and 100% shared preference.
All outputs are bitwise. At 25%, the MLP change is 68.332 to 68.239 us, with
an interval crossing zero. At 100%, it regresses 68.454 to 78.075 us. Cache
capacity matters, but the default already avoids this explicit bad choice.
The preference is a requested carveout, not an actual L1-byte counter.

A second screen transposes bundled K slices before capture so warps read
adjacent physical slices while retaining their original logical K ranges and
sum order. Byte count, grids, splits and arithmetic are unchanged. Both
read/checksum results and projection outputs match bitwise. Read-only gate/up
regresses 42.276 to 42.860 us and down 24.294 to 24.740 us; complete MLP
regresses 68.470 to 72.904 us. These skeletons include observable checksums;
their logical payload rates are not NCU DRAM counters. Reject this layout.

## Short-context attention partition screen

The target E4M3 q8 attention route activates seventeen context partitions at
1K. A 32-token minimum activates thirty-three while retaining N32 tiles and
compensated FP32 numerators. At 8K both arms retain eighty partitions.

Sixteen attention calls, using all sixteen real attention-layer projection
weights and checkpoint K/V scales, improve 508.836 to 416.010 us at 1K.
The paired saving is 92.826 us, with interval 92.221 to 93.409. At 8K the
change is 883.425 to 883.164 us, with interval crossing zero. Inputs use
synthetic hidden states, so this is an operator collection, not a full layer
or teacher-forcing result. Against an independent FP64 decoded-cache oracle,
worst absolute output error is about 0.00185 at 1K and 0.00187 at 8K in both
arms. Relative L2 error is at most 0.000211. Model-level admission remains.

## Inline GDN convolution rejection

A separate Triton screen retains the original BV2 recurrence and inlines
convolution, gating, core zeroing and history updates. Four Q/K history owners
wait for their 192 readers before overwriting shared history; uniquely owned
V columns need no such handoff. Only these four CTAs poll, avoiding a blocking
whole-grid barrier. No normalization parallelism or state arithmetic changes.

All outputs, residuals, convolution history and FP32 snapshots are bitwise at
four amplitudes on all TP ranks. However, the complete layer regresses
160.261 to 170.005 us, saving -9.744 us with interval -10.322 to -9.124.
Timed compute nodes decrease from ten to nine. Reject the candidate: removing
a node does not compensate for duplicated convolution and history coordination.

## Fused-projection CTA order rejection

The existing qkvz/b/a operator already fuses 128 FP8 CTAs and 96 small b/a
CTAs into one launch. An initial harness version did not activate the candidate;
those control/control samples are retained separately and excluded. The fixed
harness records the selected route and asserts its activation before capture.

ba-order1: 160.573 to 161.705 us; saving -1.132 us,
95% interval -1.956 to -0.041.
ba-order2: 161.428 to 162.376 us; saving -0.948 us,
95% interval -1.946 to 0.312.

Both preserve output, residual, state and history bits. Neither improves the
complete layer. Reject both schedules.

## Compact short-attention geometry rejection

The earlier compact-CTA transform targeted a frozen older source. Applying
it to the current source initially produced invalid outputs: its paired K
loader assigns one vector to each of 512 threads, but compact CTAs use 256.
Half of the panel was not loaded. The apparent 154.7-us saving at 1K is not
valid performance evidence; FP64 comparison finds absolute errors near 7.

Using the complete thread-strided K panel loader repairs this error. The
current operator guard requires finite outputs and relative L2 error below
0.001 before timing. Sixteen calls improve 509.542 to 433.332 us at 1K,
saving 76.211 us with interval 75.663 to 76.819. At 8K they regress 883.333
to 1283.379 us, losing 400.046 us. Reject this global replacement: the simpler
32-token partition candidate is faster at 1K and does not regress 8K.

## Follow-up operand and publication screens

The figures below use the maximum TP rank for each paired sample, not the
maximum of rank means. Intervals bootstrap 200 paired differences, seed 123.
These are complete layer graphs unless explicitly identified as collectives.

| Candidate | Control us | Candidate us | Saving us, 95% interval | Disposition |
| --- | ---: | ---: | --- | --- |
| Paired two static accumulator chains | 160.251 | 160.169 | 0.082, -0.815 to 0.885 | No admission |
| Paired four static accumulator chains | 159.790 | 158.556 | 1.234, 0.753 to 1.808 | No advantage over simpler exact paired route |
| FP8 channel scale after reduction | 159.555 | 159.769 | -0.215, -0.625 to 0.195 | Reject |
| Adjacent gate/up groups, sequential | 159.457 | 158.464 | 0.993, 0.281 to 1.608 | Exact; layout-only benefit not isolated |
| Adjacent gate/up groups, interleaved | 159.785 | 157.502 | 2.284, 1.807 to 2.781 | Exact; needs one-copy prefill plan |
| Norm explicit local pointer, 64-call burst | 8.023 | 7.430 | 0.593, 0.575 to 0.611 | Exact; source promotion prepared separately |
| Down publication with explicit local pointer | 160.758 | 158.735 | 2.022, 1.459 to 2.606 | Numeric; publisher increment not isolated |
| Head-local delta/norm, scalar states | 160.021 | 190.684 | -30.664, -31.303 to -30.023 | Reject |
| Head-local delta/norm, vector states | 160.174 | 160.635 | -0.461, -1.122 to 0.210 | No speed admission |
| Head-local convolution/delta/norm | 160.328 | 169.231 | -8.904, -9.590 to -8.238 | Reject |

The first paired-accumulator implementation indexed local arrays with a
runtime remainder and regressed by 110/236 us. Static unrolled chains repair
that compiler issue; they still change summation order, unlike the simpler
one-chain candidate. Retain the failed samples separately.

The scale-after-reduction candidate removes per-group FP16 scale multiplies,
but changes dequantization rounding and does not improve the complete layer.
No teacher-forcing run is warranted after its failed speed screen.

SASS for the rejected down publisher copies the eight RankData pointers into
a 64-byte thread-local frame with four STL.128 instructions and an LDL.64.
Passing the local pointer independently removes that frame. The qualified
norm partial geometry and peer protocol remain unchanged. Its 0.593-us
collective benefit includes the five-part packet change; it is not a pointer
change measured against an otherwise identical packet control.

The corrected publisher also uses twenty-part norm. Its 2.022-us layer benefit
cannot be added to a separate twenty-part norm estimate. A comparison against
the same local-pointer norm without publication is required before claiming
a producer-specific gain. State/history bits match; wider norm output differs
by at most 0.000977. Model-level numerical and concurrency admission remain.

A separate 64-bit payload protocol packs three FP16 values and a 16-bit
generation, eliminating sentinel resets. It increases traffic by 50% and
regresses the 64-call collective screen from 8.030 to 22.903 us, despite exact
output/residual bits. Reject this protocol.

Head-local fusion retains the BV2 warp tasks, grouping eight warps per CTA.
Only the eight CTAs belonging to each head exchange norm packets; there is no
whole-grid barrier. SASS identified scalar FP32 state loads/stores, compared
with LDG.E.128/STG.E.128 in the original Triton kernel. Explicit float4 state
operations remove almost the entire 30-us regression, while using 48 registers
and 8512 shared bytes without spills. The remaining layer difference crosses
zero; fusion is not admitted. This rules out attributing the previous 30-us
loss solely to cooperative launch or a grid barrier.

Adding convolution/gating/history/zeroing removes two timed compute nodes,
ten to eight. Q/K readers copy history before publishing a release packet;
the unique history owner waits only for its group's 24 readers. The resulting
kernel uses 54 registers and 10144 shared bytes without spills but loses
8.904 us per layer. History remains bitwise; state differences are at most
6e-8 and FP16 outputs differ by at most 0.003906. A first harness run expected
one removed node; its node-count guard failed before timing. The corrected
guard requires two. Reject the implementation, not the numerical gate.

## Reproduction and evidence

Benchmarks are under `benchmarks/kernels/`:

- `benchmark_sm70_qpn2_effective_scale.py`
- `benchmark_sm70_qpn2_paired_gate.py`
- `benchmark_sm70_qpn2_paired_layer.py`
- `benchmark_sm70_tp4_pull_norm.py`
- `benchmark_sm70_qpn8_two_phase.py`
- `benchmark_sm70_gdn_projection_conv.py`
- `benchmark_sm70_gdn_hybrid_snapshot.py`
- `benchmark_sm70_gdn_cooperative_core.py`
- `benchmark_sm70_qpn2_shared_scale.py`
- `benchmark_sm70_qpn2_scale_bits.py`
- `sm70_tp4_norm_partial_packets.py`
- `build_sm70_qpn2_warp_publish.py`
- `benchmark_sm70_qpn2_l1_partition.py`
- `benchmark_sm70_qpn2_warp_slice_layout.py`
- `benchmark_sm70_e4m3_short_splits.py`
- `sm70_gdn_inline_conv_screen.py`
- `build_sm70_qpn8_ba_order.py`
- `build_sm70_qpn2_paired_accumulators.py`

The layer benchmark loads the extension generated by the paired-gate benchmark.
`--mode 1` selects sequential paired warps; `--mode 0 --prepared-qk` isolates
the rejected GDN normalization cache. The collective benchmark accepts
`--burst 64`. Extensions are built with `TORCH_CUDA_ARCH_LIST=7.0`, `-O3` and
`-lineinfo`; no task-cache DSO is installed into a serving runtime.

Retained JSON samples, compiler logs, source/extension hashes, commands and
GPU snapshots are in the campaign's `qwen38-qpn2-effective-scale-20261008`
artifact directory. Both positive microbenchmarks and rejected candidates
remain outside production dispatch pending complete-artifact admission.
