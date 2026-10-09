# Flash-Next expert weight reuse on SM70

The useful distinction is weight reuse across routes, rather than kernel count
alone. Existing grouped HMMA is the strongest measured IQ3 gate/up path. A
two-route integer reader helps against the retained route-major dp4a chain,
but does not beat grouped HMMA for IQ3. A compressed-plane IQ2_S reader has a
small additional M20 benefit and no stable M5 benefit. None of the new research
kernels qualifies for a default model route.

The scope is isolated expert microbenchmarks. The historical 17.4-ms complete
round is unchanged by this report. No model was started, and no target-output,
acceptance, C4, or end-to-end speed result is inferred from these timings.

## Workload

- Model: Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S, both GGUF shards.
- Real TP4 rank-0 weights, 512 experts, top-10; gate/up N=160, K=2560;
  down N=2560, K=160. All three representative gate/up formats are included.
- Layer 17: IQ3_S / IQ4_NL down; layer 0: IQ3_XXS / IQ4_NL down;
  layer 1: IQ2_S / Q2_0 down. Q2_0 uses the extended GGUF reader.
- 38 recorded M5 routing windows per layer. M20 concatenates four successive
  M5 windows, wrapping the recording. This is a routing stress fixture, not
  a recording of C4. The overlapping windows are not independent prompts.
- Seeded synthetic FP16 inputs, standard deviation 0.1–0.45, and synthetic
  normalized FP32 routing weights. These are not captured model activations.
- V100-SXM2-32GB, SM70, driver 580.173.02, CUDA 12.8, Torch 2.10.0+cu128,
  Python 3.12. Recorded active clocks: SM 1530 MHz, HBM 877 MHz. The host has
  four fully connected NVLink V100s; the benchmark uses one isolated GPU.
- Source-complete control source `86e9675964`; core SHA256
  `815a458485194820bc4671295c924f26001911f74620d131492387c315cd10f2`.
  The installed-operator benchmark uses its ordinary native operators in a
  fresh process, without private extensions or preload overrides.

The actual model has ten IQ3_S, seventeen IQ3_XXS, twenty IQ2_S and one IQ4_XS
gate/up layers. Down has 39 IQ4_NL and nine Q2_0 layers. The three captured
layers do not cover every layer or the single IQ4_XS gate/up layer.

## Measurement and numerical boundaries

Each operation is captured eight times per CUDA Graph. Five warm replays
precede timing; 15 graph replays produce one event sample. Operation order
reverses between adjacent routing windows. Report paired full-chain deltas
as well as separate gate/up and down service times. Standalone service sums
need not equal a captured full-chain duration.

The retained chain quantizes the input once, fuses gate/up with FP16
SiLU-product and Q8_1 intermediate output, then calls the retained fused
down/unroute. It has three kernels. Grouped HMMA consumes FP16 input directly,
decodes coalesced compressed planes, emits the same intermediate format and
uses the same down/unroute. It has two kernels. Its input precision and weight
reconstruction boundary differ from the retained integer reader; comparisons
to that reader are not byte-equality assertions.

The benchmark independently dequantizes official GGUF weights to FP32. It
checks pre-quantizer FP16 hidden values, restored Q8 hidden values, and down
output against the corresponding FP32 reference. The raw reference uses the
restored input Q8_1 values; the HMMA reference uses original FP16 input. Gate,
up, SiLU-product, and individual down results retain the FP16 boundaries;
weighted route reduction uses FP32. Eight changed-input/routing graph replays
also poison intermediate packets before execution.

Numerical summaries are snapshotted before changing the routing fixture for
timing. Bandwidth divides the sum of each window's payload by its matching
time, instead of dividing the last window's bytes by a cohort median.
Logical unique bytes and issued payload estimates are separate. Neither is
an Nsight measurement of DRAM traffic: L2 reuse, repeated table loads,
activation traffic and transaction amplification can change actual traffic.

## Installed-operator results

The reproducible measurement uses
`benchmarks/kernels/benchmark_sm70_gguf_expert_routes.py`; raw samples and numerical
checks are retained in the accompanying data file. It exercises existing
native operators; it does not enable grouped HMMA in model configuration.

Full chains, medians in microseconds across 38 routing windows:

| Format | M | Retained dp4a | Existing grouped HMMA | Mean paired saving from HMMA | Faster windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ3_S | 5 | 62.665 | 53.201 | 9.701 | 38/38 |
| IQ3_XXS | 5 | 60.177 | 54.345 | 5.288 | 38/38 |
| IQ2_S | 5 | 57.677 | 59.017 | 0.242 | 9/38 |
| IQ3_S | 20 | 200.422 | 141.649 | 57.916 | 38/38 |
| IQ3_XXS | 20 | 191.791 | 159.586 | 31.996 | 38/38 |
| IQ2_S | 20 | 175.164 | 163.550 | 13.925 | 34/38 |

The IQ3 M5 chains improve 15.1% and 9.7% by median. IQ2_S M5 retains dp4a:
HMMA loses most windows and its small mean advantage is driven by a few
low-density windows. A positive mean is insufficient for uniform admission.
All M20 comparisons use the synthesized routing fixture described above.

Representative M5 component medians and matching payload rates:

| Format | Raw gate/up, us | HMMA gate/up, us | Retained down, us | Raw gate/up logical GB/s | HMMA gate/up logical GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ3_S | 40.990 | 33.434 | 19.887 | 334.7 | 417.8 |
| IQ3_XXS | 37.180 | 34.863 | 21.082 | 368.7 | 417.5 |
| IQ2_S | 34.679 | 40.252 | 18.044 | 290.4 | 303.0 |

HMMA's IQ2_S representation grows gate/up payload from 263,680 to 307,200
bytes per expert. Its higher apparent GB/s accompanies worse common-case M5
time. Optimize time and traffic together rather than maximizing this rate.

All 48 changed-input/routing cases pass both paths' independent checks.
Maximum pre-quantizer hidden relative L2 errors are 3.52e-5 for the retained
reader and 6.38e-4 for HMMA. Restored Q8 intermediate errors versus their FP16
reference are at most 8.40e-3 and 8.43e-3, respectively, including the existing
intermediate quantizer. Maximum down-output relative L2 errors are 2.25e-5
for both paths against the matching restored-Q8/official-weight reference.
These checks do not compare whole-model logits or acceptance.

Raw data: [native operator results](data/flashnext_expert_native_routes_20261010.json).
Benchmark SHA256:
`16187588947fa84f0ae7d70fb539c6c4fcd23f6edec560f13a691bf4e2a36b67`.
The gate/up kernel, dp4a kernel/header, reconstruction header and plane packer
hashes match the integration source used for this review scope.

For scale, the 38-window fixtures imply these HBM-only floors at 898 GB/s.
They count each unique expert once and omit decoder instructions, input/output
traffic and synchronization. They are necessary byte budgets, not predictions
of achievable latency.

| Format / evaluated representation | Unique gate/up + down payload per window, MB | HBM-only floor, us |
| --- | ---: | ---: |
| IQ3_S / existing HMMA planes | 22.76 | 25.35 |
| IQ3_XXS / existing HMMA planes | 24.40 | 27.17 |
| IQ2_S / retained raw gate/up | 15.93 | 17.74 |

These three recorded layers cannot establish an all-layer or complete-round
lower bound. In particular, other IQ2_S layers can use IQ4_NL down weights
rather than this fixture's Q2_0 weights. Expanding a codebook into scalar codes
also changes this byte floor and must be charged to its own representation.

## Structural experiments

The following matched results use private, task-owned research extensions.
They are screening results, not package performance or model admission data.
Rejected sources, build commands, hashes and raw reports are retained in the
experiment artifacts; they are not installed as production kernels.

Raw data: [research comparisons](data/flashnext_expert_structural_research_20261010.json).
Its prototype numerical checks precede routing changes for timing. Stale
post-timing oracle fields from the original research recorder are excluded;
they were not used to qualify a production path. The installed-operator
benchmark above snapshots its independent numerical references correctly.

### Route density matters

Across the 38 M5 windows, unique expert count is:

| Gate/up format | Mean | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| IQ3_S | 38.66 | 31 | 45 |
| IQ3_XXS | 43.32 | 34 | 50 |
| IQ2_S | 38.18 | 26 | 50 |

The previous fixed IQ2_S timing fixture had 48 distinct experts in 50 routes.
That fixture underrepresented reuse. Conversely, a favorable low-density
fixture cannot establish a universally faster route.

### Weight-major two-route integer dots

An O(routes) shared expert table pairs routes selecting the same expert. Pair
preparation shares the input-quantization launch and rewrites every plan slot;
no clear kernel or host active-expert count is needed. One decoded signed
integer weight group serves both activations. Source coefficients, integer
dot/scaling arithmetic and the original FP32 summation tree are retained.
Gate/up, FP16 SiLU-product and Q8 intermediate emission share one launch.
The faster retained down/unroute is reused, so the complete chain has three
kernels and no new global reduction workspace.

Input and hidden Q8 packets match the retained reader byte-for-byte on eight
changed replays for each format and M. Final paired-down experiments use a
different reduction schedule and must not inherit that equality claim.

| Format | M | Retained chain median, us | Pair chain median, us | Mean paired saving, us | Faster windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ3_S | 5 | 63.497 | 60.403 | 3.104 | 38/38 |
| IQ3_XXS | 5 | 60.147 | 60.087 | -0.562 | 20/38 |
| IQ2_S | 5 | 56.418 | 54.588 | 0.785 | 29/38 |
| IQ3_S | 20 | 199.262 | 177.374 | 20.534 | 38/38 |
| IQ3_XXS | 20 | 190.328 | 179.921 | 10.477 | 37/38 |
| IQ2_S | 20 | 175.509 | 163.755 | 14.155 | 38/38 |

At M5 the prepare launch grows from about 1.6–1.7 us for quantization alone
to about 2.5–2.7 us with pairing. Where reuse is low, this cost and the larger
per-thread live state erase the gate/up saving. The two-token reader should
not replace all formats merely because it reads fewer logical weights.

### Compressed-plane IQ2_S integer reader

This reader reuses existing compressed IQ2_S planes and source coefficients;
it does not expand weights into scalar FP16 or add another weight layout.
Coalesced row-stripe loads feed signed integer dp4a directly, and two routes
share a decoded weight group. Activation preparation and retained down/unroute
remain separate, for three kernels total. A different K summation tree causes
rare FP16 ties to cross the existing Q8 intermediate boundary.

| M | Retained chain median, us | Existing grouped HMMA median, us | Plane chain median, us | Mean saving vs retained, us | Mean saving vs HMMA, us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 56.418 | 57.988 | 57.152 | -1.128 | -0.989 |
| 20 | 175.509 | 160.772 | 157.909 | 19.772 | 2.830 |

The M20 plane chain wins 37/38 windows against retained dp4a and 29/38 against
grouped HMMA. Its M5 chain loses on average. Maximum relative L2 errors across
eight changed replays are 3.44e-5 before hidden quantization, 1.23e-4 after
restoring Q8 hidden values and 3.18e-5 in final output. It has 80 registers per
thread and no compiler spills. These compiler facts do not establish achieved
occupancy or measured DRAM throughput.

The additional gain over the strongest M20 control is too small to justify a
new default dispatch and weight lifetime change on this evidence. Retain this
result as a building block; do not extrapolate it to C4 or acceptance.

### Rejected complete-chain designs

| Design | Representative matched result | Decision |
| --- | --- | --- |
| Resident gate/up + down with per-expert readiness | M5 74–81 us versus 53–62 us controls | Co-resident state, resources and handoff outweigh launch removal; reject. |
| Register-only activations in grouped HMMA | IQ3_S M5 63.4 us versus 53.4 us | Removing staging/barriers does not improve the chain; reject. |
| K-parallel down with the original deterministic sum | Isolated M5 down can improve below 1 us; chain regresses, M20 down worsens | Do not promote isolated service savings. |
| IQ3_S affine signed nibbles, integer dots | M5 54.3 us versus native HMMA 52.9 us; M20 paired 154.2 versus HMMA 137.5 us | Source integer values preserved, but storage grows about 45%; reject. |
| Expanded scalar-code HMMA | M5 IQ3_S 55.6–57.7 versus native 52.6; IQ3_XXS 60.7 versus 51.2 us | Registers, extra storage and staging dependencies erase decoder savings; reject. |
| Paired down with last-CTA deterministic unroute in one launch | M5 down 29.4 versus 19.7 us; M20 93.5 versus 58.1 us | Extra work and producer state outweigh shared decoding; reject. |
| Exact integer-valued FP16 Tensor Core dots, FP32 post-scaling | IQ2_S chain M5 94.4 versus 59.0 us; M20 215.8 versus 174.9 us | Dot is cheap; per-subgroup normalization/shuffles dominate. Reject. |

The integer Tensor Core experiment preserves input Q8 bytes and has maximum
pre-quantizer hidden relative L2 error 3.09e-5. Its failure is speed, not a
license to lower the precision boundary. Scalar-code HMMA initially had an
incorrect nibble order for IQ3_XXS; that packing bug was fixed before collecting
its reported passing comparisons. The rejected layouts grow coefficient/code
storage by 21–50%, depending on type and packing.

## Community ideas and the SM70 boundary

[MonoMoE's implementation design](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/docs/design_docs/monomoe_kernel.md)
uses weight-major batches, per-expert producer readiness and cross-expert
prefetch/deferred epilogues. Its current kernel uses Hopper TMA, WGMMA,
large shared-memory allocations and asynchronous copies. The transferable
ideas are sharing a decoded expert tile across tokens and consuming completed
experts without a grid-wide barrier. Those ideas were tested here; removing
launches alone is insufficient on SM70. The resident experiment does not
implement a full equivalent of MonoMoE's pipeline.

[llama.cpp MMVQ](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)
provides the integer-dot and small-route design reference. Its
[shared-expert fusion](https://github.com/ggml-org/llama.cpp/pull/29184) motivates
keeping activation and weighted writeback inside the projection launches.
[Strix's Flash-Next work](https://github.com/halo-box/strix-llama.cpp/pull/106)
provides IQ3_S sign-table and long-K pipeline ideas. Platform-specific wave and
prefetch tuning is not a substitute for matching this chain's precision,
register budget and route density.

[FLUTE](https://github.com/HanGuo97/flute) and
[LUT-GEMM](https://github.com/naver-aics/lut-gemm) motivate moving decoder work
into load-time representation and avoiding divergent table traffic. The scalar
experiments test that tradeoff directly: simpler decode can lose when it adds
weight traffic, register state or staging dependencies. No external project
kernel was copied in this review scope. Existing packaged decoder provenance
and licenses remain unchanged.

## Reproduction

Use the normal installed SM70 build. The benchmark has no private-extension
argument. On an idle card, with the shared GPU lock:

```bash
flock /tmp/gpu0-3.lock env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
  EXPERT_SOURCE_SHA="$(git rev-parse HEAD)" \
  python benchmarks/kernels/benchmark_sm70_gguf_expert_routes.py \
  --model "$GGUF_SHARD1" --routes "$EXPERT_ROUTES_JSON" \
  --output "$EXPERT_RESULT_JSON" --layers 17 0 1 --m 5 20 \
  --rank 0 --tp 4 --windows 38 --verify-routes 8 --iterations 15
```

The route fixture is the existing per-layer record format, with
`{"17": {"records": [{"ids": [[...], ...]}]}, ...}`. Each window has five
rows, ten distinct expert IDs per row, and IDs in [0, 512). Private routing
recordings and GGUF weights are not committed. The result fingerprints the
recording, benchmark and native core. GPU legacy locks and process ownership
must also be respected where applicable.

Next admission requires a route that beats the strongest existing path across
the affected fixture distribution, plus the relevant numerical checks. If a
model dispatch is subsequently changed, acceptance and C4 remain separate
gates. The microbenchmark establishes neither of them.
