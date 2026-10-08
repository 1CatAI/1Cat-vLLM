# Flash-Next MTP4 structural cost measurements

The latency reference uses Flash-Next IQ3_S, FP16 MTP4, TP4 on four
V100-SXM2-32GB GPUs, and device-resident per-vector E4M3 target KV history.
Draft KV is FP16. The existing protected FP16 hot-page reader remains active.
Host-history placement has separate measurements and is excluded from this
reference.

## Measurement contract

CUDA 12.8, Torch 2.10.0+cu128, FP16 activations, FP32 recurrent state and
accumulation, greedy MTP4, maximum length 9216, prefill budget 512, maximum
four sequences, prefix caching disabled, and language-only inference.
The workload is C1 input 8192/output 256 and C4 input 128/output 600 per
request. The unchanged native core SHA256 is
`784d1447f4f5f5593fa77db525e6b40e841366d654202aab5a56beec67398a0c`.

The unprofiled reference is **17.8493 ms/round C1**, **42.4171 ms/round C4**,
and 46.6626% draft-token acceptance on the natural-prompt cohort. The
12 ms/round objective has not been reached. No optimization gain is claimed
by this diagnostic change.

`sm70_round_cost_diagnostics` installs a bounded, graph-visible top-10 route
recorder before graph capture. It is disabled by default. The benchmark
requires `--trace-only --round-cost-ledger` for this mode: diagnostic copies
must not contaminate acceptance latency. Records include replay ordinal,
actual row count, expert IDs and explicit overflow. Loaded target and draft
tensor inventories include physical shapes, strides and storage aliases.
CPU offload models do not allocate CUDA routing recorders.

The capture contains 54 target M5 calls and 54 complete draft rounds. There
are 48 bootstrap draft M1 calls before steady decoding; eight fall outside
the bounded ring. Steady MTP4 is **M5, M1, M1, M1**, rather than four M1
calls. The analyzer excludes bootstrap calls and rejects incomplete rounds.
All four ranks have identical routing records. The 256 output token IDs
match all three unprofiled device-history control segments.

## Initial byte and service ledger

These are per-card, per-round operand estimates selected from the actual
loaded storage and routes. They are not measured DRAM transfers. Independent
experts define compulsory reads; route-issued reads can reuse L2. Quantized
canonical banks and retained original banks are not both counted. KV,
workspaces, decoder instruction ports and communication need their own rows.

HBM floors use 900 GB/s. Service is rank0 median from the same 37 central
profiled common-TP windows. It includes profiling and the additional routing
copies. Shared-expert service overlaps routed experts and cannot be added to
target wall time.

| Target work | Operand MB | HBM floor ms | Profiled service ms |
| --- | ---: | ---: | ---: |
| Routed gate/up | 526.90 | 0.585 | 1.800 |
| Routed down | 374.29 | 0.416 | 0.921 |
| HCX packed down/up, 94 boundaries | 338.82 | 0.376 | 3.008 |
| Dense projections and shared down | 560.33 | 0.623 | 2.302 |
| Shared gate/up, overlapping | 27.85 | 0.031 | 0.792 |
| Router projection | 125.83 | 0.140 | 0.710 |
| Router top-k score reads | 0.25 | <0.001 | 0.380 |
| GDN state read plus five snapshots | 169.87 | 0.189 | 0.694 |
| GDN a/b weights | 4.42 | 0.005 | 0.120 |
| QSA index projection weights | 39.32 | 0.044 | 0.314 |
| Target canonical LM head | 158.92 | 0.177 | 0.207 |

The router score and GDN state rows omit small activation/metadata traffic.
The QSA index projection service includes one small boundary GEMM. The
selected sparse KV union and cache counters are measured below. Remaining
workspace traffic and instruction-port costs are unresolved; this is an initial
ledger, not a complete lower-bound proof.

| Four draft steps | Operand MB | HBM floor ms | Profiled service ms |
| --- | ---: | ---: | ---: |
| Routed FP16 experts | 162.78 | 0.181 | 0.288 |
| Four canonical head projections | 635.70 | 0.706 | 0.849 |
| FC, QKV, output and index projections | 144.18 | 0.160 | 0.398 |
| Eight HC down/up pairs | 28.84 | 0.032 | 0.247 |
| Router weights | 10.49 | 0.012 | 0.053 |
| Shared matrices | 9.85 | 0.011 | 0.158 |

Draft dense/shared service classifications include helper GEMVs and
reductions; they are not isolated operator timings. Across four draft calls,
the first call averages 36.24 independent experts, then each M1 call selects
10. The retained GGUF head is Q6_K with true K=2560: its packed byte width
2100 is not the matrix's K. The selected canonical u4+u2/group16 storage is
158,924,800 bytes/card, rather than the raw bank's 130,368,000 bytes. Head
service is close enough to its byte floor to deprioritize another head path.

### Repeated expert work

The 50 target routes average **36.17 independent experts** across all 48
layers. The native reader issues 731.01 MB of gate/up storage and 518.40 MB
of down storage before cache reuse, versus 526.90 and 374.29 MB compulsory
reads. Of independent experts, 74.48% receive one token, 16.78% two, 5.61%
three, 2.31% four and 0.82% five. A grouped design must amortize decoding for
repeated experts without padding single-token experts to a large M tile.

### Decoder instruction evidence

Installed gate/up kernels were profiled in isolation at M5, N160, K2560,
50 routes, using real checkpoint weights. The IQ3_XXS and IQ2_S captures use
recorded model routes; the retained IQ3_S capture uses 47 unique experts.
Instruction counts are not model latency measurements.

| Format | Predicated thread instructions M | Integer-related M | Warp instructions M |
| --- | ---: | ---: | ---: |
| IQ3_S | 204.79 | 154.44 | 6.573 |
| IQ3_XXS | 192.63 | 142.34 | 6.189 |
| IQ2_S | 173.56 | 128.46 | 5.597 |

The integer-related classification includes address, predicate, shuffle and
permutation instructions: its division by peak INT32 throughput is a cost
model, not a proven common-port bound. Using the
[Volta execution model](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html),
at 1290 MHz, 80 SMs and 64 integer
lanes/SM, it gives about 0.99 ms across the 10/17/20 layers of these formats.
The independent warp-issue estimate is about 0.69 ms. Use the actual clock
and instruction-port limits before treating either estimate as a strict
roof. A bandwidth-only gate/up floor is too optimistic.

### Dense decoder measurements

The selected `gguf_dense_segments_sm70_out` was profiled at M5, K2560,
with actual TP4 input groups. These are the selected segment banks, rather
than the independently shipped DMV13 side-projection control.

| Input group | N | Formats | Predicated thread instructions M | Integer-related M | Warp issue floor us |
| --- | --- | --- | ---: | ---: | ---: |
| GDN layer 0 | 2560 + 1536 | Q6_K + Q4_K | 55.132 | 28.076 | 4.289 |
| GDN layer 2 | 2560 + 1536 | Q6_K + IQ4_XS | 55.659 | 28.454 | 4.332 |
| Attention layer 3 | 3072 + 256 + 256 | IQ4_XS + Q6_K + Q6_K | 44.539 | 20.657 | 3.478 |

The issue floor uses four schedulers per SM at 1290 MHz. It excludes
multi-cycle port limits and dependencies. NCU's replayed durations are not
endpoint timings. The larger operand bandwidth floor and the profiled
instruction/stall evidence must be considered together.

### Actual sparse KV operands

A second untimed C1 capture records causal indices, request IDs and cumulative
cache counters. Its 256 output IDs match all three device-history controls.
All four ranks agree on selections. Target statistics use 38 central M5
calls across twelve QSA layers; the first cache delta includes unrecorded
prefill and is excluded. Cache hit-plus-miss deltas equal selected entries
for every retained call.

| Per-card target QSA work | Per round |
| --- | ---: |
| Query-list entries | 122,970 tokens |
| Sum of within-layer, cross-query unions | 43,160 tokens |
| FP16 reader union | 44.196 MB |
| FP16 query-list operands | 125.921 MB |
| Encoded E4M3 union including two FP32 scales/token | 22.443 MB |
| Encoded miss reads, rank0 | 0.567 MB |
| Cache hits, rank0 | 121,879 tokens |
| Cache misses, rank0 | 1,091 tokens |

The target hit rate is about 99.11%. Rank-specific miss counts differ slightly
because cache installation contends; selections and output IDs remain equal.
Operand unions are not measured DRAM transfers or a count of physical hot-cache
allocations. The FP16 union floor at 900 GB/s is 0.049 ms; a direct E4M3 union
floor is 0.025 ms. Reading a vector once per query issues 2.85 times the union,
before L2 reuse and per-head implementation effects.

The bounded draft ring retains sixteen complete M5/M1/M1/M1 rounds;
twelve central rounds are used. Four-step FP16 reader unions total 9.458 MB
and query-list operands 16.789 MB. Draft M5 averages 3,088 distinct tokens
and 10,247.5 query-list entries; later M1 calls average 2,049.3 each.
The measured cache preparation and attention service still need critical-path
A/B attribution before any byte reduction is called an endpoint gain.

## HC communication ledger

The actual topology has NV1 edges 0-1, 1-3 and 2-3, NV2 on 0-2, and SYS
on 0-3 and 1-2. The logical order `[0,1,2,3]` uses direct XOR-1/XOR-2
peers and forwards diagonal traffic. It is not a full NV2 mesh.

Each M5 HC input reduction transfers 12,800 FP32 values with a 32-bit epoch
tag per value. Two phases send **204,800 wire bytes per card per boundary**,
not 25,600 FP16 bytes. Critical ranks 1/3 use NV1 in both phases, but on
different links. Each CTA forwards its 32-column chunk independently; there
is no whole-grid barrier between phases. Consequently the phases can pipeline.
At 25 GB/s/direction the bandwidth floor is the larger phase,
**4.096 us/boundary, or 0.385 ms over 94 boundaries**, plus unmeasured
pipeline startup and final-chunk latency. Adding both complete phase transfer
times would incorrectly give 0.770 ms. LoRA and output forwarding add traffic
and dependent arrival latency. Every boundary also has two intra-GPU grid
barriers; polling instruction counts are arrival-dependent.

Packed HC down/up storage is 1,966,080 plus 1,638,400 bytes/boundary. Weight
prefetch overlaps communication, so adding its 0.376 ms byte floor to the
wire floor would overstate the combined bound. Conversely, treating polling
inside the kernel as GPU idle would understate waiting.

NCCL's [LL128 eligibility rules](https://github.com/NVIDIA/nccl/blob/v2.27.6-1/src/graph/tuning.cc)
include homogeneous SM70 with eligible NVLink paths; SM90-only restrictions
cannot be assumed. However, replacing tagged LL words with a more efficient
wire protocol alone only removes roughly 0.18 ms from these input reductions
at the bandwidth roof. It is insufficient as an independent structural
proposal with a 0.5 ms minimum projected gain.

### Isolated HC phase measurements

The unchanged packaged kernel was measured on eight real HC weight pairs at
M5, TP4, on this topology. Non-instrumented whole-chain median is 29.136 us.
Device-local stage medians are:

| Stage | Rank 0 us | Rank 1 us | Rank 2 us | Rank 3 us |
| --- | ---: | ---: | ---: | ---: |
| Input sum and weight prefetch | 5.120 | 6.144 | 5.120 | 6.144 |
| First grid barrier | 2.048 | 3.072 | 2.048 | 3.072 |
| Norm and partial prefetch | 3.072 | 3.072 | 3.072 | 3.072 |
| LoRA arrival, up prefetch and second barrier | 6.144 | 4.096 | 7.168 | 5.120 |
| Up and gate mix | 3.072 | 2.048 | 2.048 | 2.048 |
| Output exchange | 1.024 | 2.048 | 1.024 | 2.048 |

Globaltimer samples have 1.024 us granularity. Per-CTA medians are not
additive, and timestamps cannot establish cross-rank ordering. These numbers
identify dependencies for a prototype; they are not endpoint improvements.
The 27.998 us instrumented chain is not a faster implementation: profiling
and clock variation explain why it must not be compared as a speed gain.

## Structural directions to evaluate

1. **Direct device-history QSA.** Bypass protected hot-cache ownership and
   miss staging for the already device-resident E4M3 reference. Reconstruct
   precisely the same FP16 key/value operands, use FP16 QK with FP32 MMA
   accumulation, and retain FP32 probabilities and PV accumulation. The
   current target partial/merge plus gather service is about 1.08 ms; a
   projected 0.5-0.8 ms chain gain must pass an isolated numerical and chain
   gate. This is a hypothesis, not a measured speedup. Host history remains
   a separate placement path.
2. **HC-to-input-projection pipeline.** Group four HC CTAs, publish their
   output-column readiness, and accumulate the next dense projection over
   each available 128-column K slice. Twenty deterministic split-K partials
   fit in L2 at M5; the current canonical packed weights can be retained.
   The hypothesis is to overlap projection with HC completion and remove
   the full-HC dependency, with at least 0.5 ms chain savings required.
   This split-K version was measured on eight actual input groups: selected
   segment control 46.688 us, copied HC control 45.028 us, candidate 58.660 us.
   Numerical checks pass, but the chain gate fails. The version is rejected;
   no model restart or end-to-end claim is made. Extra registers, readiness
   polling and partial traffic are possible causes, not separately measured
   attribution. See the companion HC consumer screen.
3. **Distributed-K HC.** Reduce-scatter the block partial into 640-column
   ownership, retain sharded residuals between HC boundaries, apply global
   RMS coefficients before publishing collapsed local down partials, reduce
   FP32 LoRA contributions, then gather the next block input. This can shrink
   the input wire payload and four-stream split-K workspace together. A final
   materialization is required before a consumer needing full residuals.
   Extra normalization collectives must overlap down computation; merely
   moving all-reduce later is not a gain. FP32 partial/communication precision
   must be retained. Pipelined input-wire savings alone are capped at about
   0.193 ms across 94 boundaries. The earlier 1.0-1.5 ms proposal relied on
   adding serial phase transfer times and is withdrawn. Additional workspace
   and dependency savings must be demonstrated before admitting this path.
4. **HC-to-router producer pipeline.** Produce router K-slice partials while
   HC output columns are resident, then combine deterministic FP32 partials
   with top-k/input quantization. The current projection/top-k/input-quant
   chain has about 1.22 ms service; a 0.5-0.8 ms proposal is plausible only if
   the added HC tail does not consume that saving.
5. **Read/decode once per repeated expert.** Share decoded IQ words among
   tokens of the same expert and preserve the down FP16 boundary and ordered
   weighted reduction. The 28% compulsory/issued gap is an upper opportunity,
   not an expected endpoint gain. Register occupancy and single-token expert
   efficiency can reject this design.

These ideas follow tile readiness and dependency-graph scheduling, rather
than assuming a faster launch changes the round. [Mirage MPK](https://github.com/mirage-project/mirage/tree/mpk)
and [MonoMoE](https://arxiv.org/abs/2609.04244) describe persistent task graphs
and weight-major execution. Their modern GPU mechanisms are not SM70
implementations. [mKernel](https://github.com/uccl-project/mKernel) and
[FLUX](https://github.com/bytedance/flux/blob/main/docs/design.md) motivate
chunk-level communication overlap; the transferable part is readiness and
ownership, not Hopper-only instructions. [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend)
includes HC-related prenormalization work, but its Ascend950 geometry and
runtime differ from this 320-dimensional HC chain.

Every proposed change still requires numerical isolation, chain measurements,
then same-wheel C1/C4 and acceptance A/B. Update this ledger after a positive
end-to-end change; keep unsuccessful variants with their rejection reason.
