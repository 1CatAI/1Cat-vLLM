# Coalesced GGUF dense projections on SM70

Flash-Next uses mixed Q4_K, Q5_K, Q6_K and IQ4 dense projections.
The new route issues one projection kernel for same-input segments at M1..8/M20,
reconstructing FP16 weights with mma884 and FP32 accumulation. Shared experts
use two launches: gate/up with SiLU and a parallel shared-gate dot, then down
with the sigmoid epilogue. Codes and the canonical FP16 coefficient rounding
are preserved. The segment decoder is shared by both operations.

## Same-card acceptance

The supplied implementations are compared in one process on one V100 with
supplied rank-0 weight slices from all 48 Flash-Next IQ3_S layers, M5, CUDA graphs,
Torch 2.10/CUDA 12.8 and observed SM/memory clocks 1530/877 MHz. The full dense
sequence uses ABBA ordering. These initial measurements use research JIT
extensions; production performance must be revalidated after incorporation
into the normal extension and wheel.

|Projection|Canonical and glue, us/layer|Segment route, us/layer|
|---|---:|---:|
|GDN input, 36 layers|33.59|16.70|
|GDN output, 36 layers|15.49|8.28|
|Attention input, 12 layers|34.79|15.27|
|Attention output, 12 layers|15.46|8.55|
|Full shared expert, 48 layers|44.81|12.13|

Complete dense replay: 4.751 to 2.001 ms, a 2.750 ms difference. Projection
errors relative to official FP32 references match the canonical implementation
at the reported precision. Full shared expert relative maximum difference
against the segmented FP16 chain is 0.00174.

A second same-process comparison includes the existing admitted Q8 shared
expert gate/up where supported and the unchanged fallback for other format
pairs. Complete shared-expert means are 47.93 us for canonical, 43.44/43.55 us
for this mixed Q8/fallback chain, and 12.12 us for the new two-launch route.
The new route is selected for integration. Its difference from the Q8 chain
is 0.03075 under relative maximum error; teacher-forcing and model acceptance
remain required, because the Q8 activation contract differs from FP16 MMA.

## Storage and capture requirements

One resident packed bank replaces the old canonical bank. Unqualified M uses
TurboMind after restoring its integer layout into shared transient scratch;
there is no second retained quantized bank. Restore must be byte-identical to
the existing converter and must be benchmarked at M20 before model admission.
Shared-expert intermediates, gate storage, split partials and completion
counters are allocated before graph capture. All threads publish their own
split partials before updating the completion counter; reduction order is
fixed. Changed-input replay tests check counter reuse and stale results.

The supplied shexp3 grid-barrier variant is excluded. The provided dense_mv2,
shexp2 and pack implementations are the source of the initial layout and
kernel schedule, with provenance retained in the adapted source.

## Installed storage and batch qualification

The initial complete installed extension reproduces the full 48-layer M5
sequence: 4.567 to 2.007 ms, with matching projection errors and shared-expert
relative maximum difference 0.00174. Six restored formats match the existing
TurboMind converter byte-for-byte. The full focused suite, including changed
inputs and M20 token groups, passes 37 checks.

Restoring the old bank on every M20 call is rejected as the default: GDN input
rises from approximately 41.27 to 87.42 us. A coalesced register restoration
reduces this cost, but remains slower than direct token groups. Prefill and
other unqualified M values retain the exact restored TurboMind route.

Parallel eight-row groups keep one dense launch and two full shared-expert
launches. M5 retains its original specialization. Modifying the segment
structure in each CTA caused a 240-byte stack frame and regressed M5; row
address offsets replace that copy. The final dense batch variant has zero
stack allocation, while its unchanged M5 variant retains the original small
frame. No new arithmetic precision is introduced.

Representative same-process ABBA cases from actual rank-0 TP4 shards give:

|Projection|M5 canonical / segment, us|M20 canonical / segment, us|
|---|---:|---:|
|GDN input|36.05 / 16.67|41.30 / 35.00|
|GDN output|15.33 / 8.37|18.27 / 20.42|
|Attention input|35.62 / 14.67|40.87 / 30.64|
|Attention output|11.22 / 6.45|15.15 / 16.52|
|Full shared expert|45.76 / 10.54|46.41 / 20.51|

These representative cases do not replace the complete 48-layer replay.
M20 outputs differ from the existing route by relative L2 approximately
0.000012–0.000031 for dense and 0.000581 for the full shared expert.
The output projections retain small M20 regressions; complete-round C4 must
confirm the combined input/shared-expert gains outweigh them. Capabilities
admit M1..8 and M20 explicitly, with one resident segment bank.

## Replicated KV heads in TP4

The supplied extraction divides every column-parallel tensor by four. The
model has two KV heads of width 256, so TP4 replicates each complete KV head
on two ranks. Corrected rank-0 K/V slices use 256 rows each. The original
48-layer sequence above therefore measures narrower attention inputs and
must not be presented as the complete production TP4 sequence.

All twelve corrected attention inputs pass same-process ABBA in the complete
installed extension. Source-compatible K/V segments include N512; five layers
coalesce Q/K/V into N3584. Both widths are now declared capabilities.

|Attention input, twelve-layer mean|Canonical us|Segment us|Maximum relative L2|
|---|---:|---:|---:|
|M5|34.015|15.665|0.0000350|
|M20|39.380|32.380|0.0000234|

These inputs use official FP32 dequantization references for the corrected
KV slices. Repeated fixed input rows extend M8 fixtures to M20. The complete
model comparison uses the same installed wheel and changes only the segment
and HC LL switches.

The first disabled-route control failed during initial profiling before
timing: allocated bytes were 28.84GB and reserved bytes 33.01GB, with only
1.5MiB driver memory free. The retry uses the standard PyTorch
`expandable_segments:True` allocator setting on both arms, preserving model,
batch budget and arithmetic. This failed startup is retained separately from
performance evidence.

## Whole-model comparison

Both arms use the same complete `dev1000+ge1b71bb5a` wheel on four
V100-SXM2-32GB GPUs at 1290/877MHz and 300W, Torch 2.10/CUDA 12.8,
Flash-Next IQ3_S, TP4 and MTP4 with FP16 draft weights. Activation/KV are
FP16; recurrent state and MMA accumulation are FP32. Target decode uses FULL
graphs, max length 9216, batch budget 512, four sequences and greedy sampling.
Only `small_m_hmma` and `hc_ll_shard` change. The Q8 expert intermediate path
is enabled in both arms. Both use `expandable_segments:True`.

|Unprofiled result|Control|Segments and HC LL|
|---|---:|---:|
|C1, 8K input / 256 output, first median round ms|23.652|22.077|
|C1, second median round ms|22.526|22.085|
|C1, pooled mean round ms|23.301|22.107|
|C1, steady tokens per round|4.886|4.886|
|C4, 128 input / 600 output each, median round ms|54.581|50.830|
|C4, mean round ms|55.110|51.008|
|C4, aggregate decode tokens/s|173.283|186.113|
|C4, steady tokens per round|9.550|9.493|

Eight natural prompts show mean draft acceptance 44.320% to 44.802%.
The paired prompt-bootstrap difference is +0.482 percentage points with
95% interval [-0.724, +1.634]. Mean acceptance length is 2.773 to 2.792,
with difference interval [-0.029, +0.065]. Matched full-vocabulary
teacher-forcing gives 63/64 equal top-1 choices, mean KL 0.001010 and maximum
KL 0.009730. Both bounded short answers stop normally and match.

All four loaded reports admit 257 segment projections and the TP HC LL
transport. Model weight allocation falls from approximately 26.78 to 25.57
GiB per rank. These results are joint changes; neither optimization receives
the whole measured gain. Microbenchmark service savings are larger than the
critical-path gain, and the 18.5ms round target remains open. A node trace is
required to quantify overlap and synchronization before selecting the next
change.

The comparison precedes the inactive-row tag-reuse guard described in
[the HC design](sm70_hc_ll_shards.md). The guard is qualified separately in a
fresh normal extension; these round numbers are not relabeled as timings of
that newer extension.
