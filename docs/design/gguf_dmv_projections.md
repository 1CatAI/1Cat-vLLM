# Shared-activation GGUF projection planes on SM70

The M8 projection route packs Q4_K, IQ4_XS, IQ3_S and IQ3_XXS into
coalesced planes consumed by a shared-activation Volta MMA kernel. It supports
ordinary projections, fused GDN input projections with floating a/b weights,
and gate/up projections with a SiLU-multiply epilogue.

Weights are reconstructed with FP16 group coefficients and accumulated in
FP32. Admission requires numerical checks, same-wheel model distribution
comparisons, and matched graph benchmarks. Other batch sizes retain the
existing projection route. Split-K workspaces and counters belong to each
layer and must remain stable across graph replay.

The implementation derives from the dmv11 projection and pack/pack3 layout.
Measured operator results, memory accounting and model-level results will be
recorded here before admission.

IQ3_XXS retains the original FP16 block coefficient. Multiplying it by 0.25
before packing introduces a second rounding for subnormal coefficients; the
reader applies that factor in FP32 together with the local scale and rounds
only the expanded group coefficient. Canonical restoration retains every
index and sign and reconstructs the same group coefficient.

Initial format checks used representative quarter-width 27B weight slices
on V100-SXM2-32GB with post-run clock readings of 1290 MHz, CUDA 12.8
and Torch 2.10.0. Those GDN slices do not establish loader-equivalent head
sharding; the corrected-shard ABBA and full model checks below cover it.
Application clocks were not locked for these operator measurements. With M8, graph replay and rotating
more than 48 MB of weight planes, down projections measured 20.3–22.3 µs
and GDN output projections 10.6–11.1 µs. Relative L2 error against official
GGUF dequantization was 3.4e-4–7.5e-4 across 24 role/type cases. These are
operator measurements, not model-level latency results.

All 16 ordered format combinations passed fused gate/up checks against the
official reference and 50 unchanged-input graph replays. Mixed IQ3_S/IQ3_XXS
GDN inputs with a/b passed 20 changed-input split-K graph checks; counters
returned to zero after every replay. IQ3_S, IQ3_XXS and compact IQ4_XS planes
restored their canonical packed codes and coefficient metadata bitwise.

The temporary original-record DMVQ reader is admitted only for TP4 IQ2_XS
and IQ2_S down matrices (N5120,K4352,M8,KW8,split1). Same-process cold-L2
ABBA measured 23.81 versus 32.32 µs for IQ2_XS and 25.12 versus 33.91 µs
for IQ2_S against canonical GEMM. Relative L2 errors were 2.9e-4.
Q2_K down (33.70 versus 28.68 µs) and IQ1_M gate (43.46 versus 32.72 µs)
retain their prior routes. IQ2_XXS gate measured 23.76 versus 31.40 µs as a
single matrix, but that result does not qualify an entire fused gate/up pair.

GDN lattice and LUT4 planes retain their original GGUF head order. M8 loads
activations in that order inside the shared-activation kernel; other M uses
the existing input transpose and canonical arithmetic. This avoids changing
prefill accumulation order while removing the M8 activation-copy launch.
Affine GDN projections keep the existing restored-weight layout.

`projection_plane_scope` selects `all`, `gated_pair`, or `iq3_xxs` for
same-wheel numerical comparisons. The latter selects whole projections that
contain IQ3_XXS, including their fused companion shards; it does not isolate
an individual reader inside a fused launch. Only `all` admits temporary IQ2
down readers. The default production scope is `all`.

IQ3 admission also checks the cancellation term used by the byte-to-half
conversion. Expanded scales above 63.96875 would overflow `1024 * scale`,
even if the actual weight remained finite; these layers retain canonical
storage. Replacement is atomic across a fused projection, so a rejected shard
never leaves a mixture of plane and canonical buffers for a legacy reader.

## Installed-wheel model comparison

The ordinary SM70 wheel passed 41 GPU operator tests. The installed native
libraries matched the packaged libraries by SHA-256; no private extension
or library override was used. Native source was `2165020b2861` and Python
integration source was `d6268878bb10`.

The comparison uses Qwen3.8-27B GSQ-RCO IQ3_S with the DFlash2 Q8_0 draft,
TP4 on four V100-SXM2-32GB cards, CUDA 12.8 and Torch 2.10.0+cu128.
KV is FP16 and recurrent state is FP32. Maximum length is 262144,
batched-token budget is 1024, maximum sequences is four, prefix caching is
disabled and CUDA graphs are enabled. Sampling uses temperature 0.7,
top-p 0.9, top-k 20, seed 123 and seven probabilistic draft tokens.
Each of sixteen fixed prompts generates 600 tokens for timing. The first
20 output rounds per prompt are omitted. Separate natural requests retain
normal EOS handling.

Both arms use the same wheel, with only the projection-plane policy changed.
Within stable generation windows, every recorded per-card clock sample was
1530 MHz in both arms. The topology includes cross-NUMA connections;
results from other machines are not subtracted from this comparison.

| Input | Policy off: full round | Policy on: full round | Off: ms/output token | On: ms/output token | Off: tokens/round | On: tokens/round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 23.66 ms | 21.99 ms | 8.04 | 7.43 | 2.962 | 2.977 |
| 8K | 24.28 ms | 23.26 ms | 8.19 | 7.95 | 2.978 | 2.951 |

Full-round values are equal-weighted prompt means. Output-token latency is
pooled across steady rounds. Draft acceptance fractions are 27.94% versus
28.04% at 1K and 27.84% versus 27.36% at 8K. Mean TTFT is 354 versus
376 ms at 1K and 2659 versus 2816 ms at 8K; canonical restoration adds
work outside the M8 route.

The first verifier states from all sixteen prompts yielded 128 comparable
logit rows with identical input tokens and positions. Full-vocabulary KL
at temperature one averages 5.59e-6, with maximum 3.85e-5; top-1 agreement
is 100%. Both natural checks ended normally. C4 is a four-request smoke,
not a steady-throughput measurement.

Every rank admits 50 gate/up pairs, 56 down, 39 GDN input, 48 GDN output,
six full-attention QKV and sixteen attention output projections to the
plane route, plus seven IQ2 down readers. Model loading reports 5.57 GiB
per card versus 7.44 GiB for the control. Whole-model latency improves
less than the isolated projection estimate; the graph attribution below
records the remaining gap.

Separate C1 distribution checks use the same sixteen prompts and the same
ordinary wheel with `gated_pair` or `iq3_xxs` scope. Each compares 128 rows,
with no different input-token or position rows discarded. Both natural
checks also end normally. These diagnostics use one sequence; the matched
latency comparison above uses four sequence slots for the C4 smoke.

| Scope | Mean KL | Maximum KL | Top-1 agreement |
| --- | ---: | ---: | ---: |
| All planes | 5.59e-6 | 3.85e-5 | 100% |
| Gate/up only | 6.94e-6 | 7.67e-5 | 100% |
| XXS-containing projections | 5.67e-6 | 3.18e-5 | 100% |

The XXS probe records eight real tokens, eight padded tokens and hidden
states of shape `[8, 5120]`. Logits are recomputed from the verifier states
through the model's ordinary head; probes are disabled for timing requests.
Nsight Systems 2024.6.2 is used for graph-node attribution. A small spawned
CUDA-profiler API test recorded twenty graph nodes with that version;
2026.2.1 did not produce a report on this host.

## TP4 graph attribution

The graph trace uses one 1K-input, 64-output request, maximum length 32768,
seven speculative tokens, FP16 KV, TP4 and the same installed wheel. Its
historical trace sampling contract is temperature 0.7, top-p 0.95 and top-k
20; this differs from the top-p 0.9 timing comparison above. Seventeen M8
replays were recorded on each rank; the fourteen interior round intervals
are retained. A round begins at the first target GPU node and ends at the
next target GPU node. Recorded generation SM clocks were 1530 MHz.
Graph-node tracing perturbs wall time, so these intervals describe
attribution rather than the unprofiled latency result.

Rank-0 target projections are listed below. Bytes are the loaded packed
weight-stream footprint, including floating a/b where fused; the bandwidth
column divides that footprint by service time and is not an NCU measurement
of DRAM traffic. Each role may contain multiple formats and admitted paths.

| Projection | Calls/round | µs/call | MB/card/call | Effective GB/s | Service ms/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| qkvz+a_b | 48 | 30.73 | 9.92 | 322.9 | 1.475 |
| gdn_out | 48 | 18.04 | 3.93 | 217.9 | 0.866 |
| gate_up.native | 14 | 55.61 | 14.77 | 265.6 | 0.778 |
| down | 64 | 26.67 | 10.68 | 400.5 | 1.707 |
| gate_up.planes | 50 | 42.50 | 20.83 | 490.2 | 2.125 |
| attention.q+k+v | 16 | 32.77 | 8.78 | 267.9 | 0.524 |
| attention.o | 16 | 17.49 | 3.82 | 218.6 | 0.280 |

Total target projection service is **7.756 ms** on rank 0 and
7.860/7.764/7.853 ms on ranks 1–3. The requested 6 ms threshold has not been
met. Fused a/b has no separate launch on admitted GDN input projections;
fused gate/up has no separate SiLU-multiply launch. Unsupported combinations
retain the previous route.

| Rank | Full traced round | Target envelope | Target inter-kernel gaps | After target |
| --- | ---: | ---: | ---: | ---: |
| 0 | 27.321 ms | 20.840 ms | 5.190 ms | 6.481 ms |
| 1 | 27.426 ms | 21.952 ms | 2.841 ms | 5.474 ms |
| 2 | 27.298 ms | 22.605 ms | 2.996 ms | 4.693 ms |
| 3 | 27.296 ms | 22.591 ms | 2.915 ms | 4.705 ms |

The rank-0 tail contains draft attention and GEMM, both vocabulary-head
calls, sampling/sorting, communication, and residual work:

| Tail category | Calls/round | µs/call | Service ms/round |
| --- | ---: | ---: | ---: |
| other_tail | 141.21 | 7.44 | 1.051 |
| draft_GEMM | 23.00 | 41.09 | 0.945 |
| communication_or_reduction | 18.07 | 23.40 | 0.423 |
| target_head | 1.00 | 269.94 | 0.270 |
| sampling_and_sorting | 23.93 | 10.72 | 0.257 |
| draft_attention | 5.00 | 92.13 | 0.461 |
| draft_shared_head | 1.00 | 268.19 | 0.268 |

The two approximately 269 µs head calls are the target head and the draft's
shared head. They occur after the target body. Service sums may overlap and
must not be added to graph gaps as a closed wall-time decomposition.
Target communication/reduction service is 4.681 ms and RMSNorm service is
0.975 ms on rank 0; these remain outside projection-reader changes.

## Model-context gap and rejected changes

### Reproduction and complete weighted operator audit

The original dmv11 source and packing scripts were reproduced separately from
the ordinary installed wheel. Original-source research extensions are used
only for this operator comparison; model results above use the ordinary
wheel. The audit covers 70 role/type classes representing all 256 target
projection calls at M8, including unchanged fallback combinations. It uses
real rank-0 TP4 fixtures, cold resident-weight rotation, graph replay and
four balanced ABBA measurements. Production shape configurations are used
for the complete weighted totals, rather than selecting the minimum of a
configuration sweep. SM clocks are recorded but not locked; active samples
range from 1290 to 1530 MHz, with paired comparisons within each process.

| Role | Calls | Existing main operator ms | Candidate operator ms | Saving ms |
| --- | ---: | ---: | ---: | ---: |
| gate/up with SiLU | 64 | 2.982 | 2.361 | 0.621 |
| down | 64 | 1.859 | 1.294 | 0.565 |
| qkvz with a/b | 48 | 1.512 | 1.159 | 0.352 |
| GDN out | 48 | 0.674 | 0.473 | 0.201 |
| attention q/k/v | 16 | 0.475 | 0.408 | 0.067 |
| attention o | 16 | 0.221 | 0.156 | 0.065 |
| Total | 256 | 7.724 | 5.852 | 1.872 |

These are sums of isolated operator measurements weighted by layer count,
not measured model rounds. The control preserves existing main's native
pairs, native down, joint qkvz/qkv and small-output routes. Comparing only
against canonical GEMM would overstate the incremental benefit. Unchanged
fallback rows share the same measured latency in both arms; they do not
claim a new speedup. Fourteen gate/up layers and nine qkvz layers retain
their existing readers. The temporary IQ2 reader covers seven down layers.

The 193 calls covered by the four plane formats total 4.137 ms with the
original source and 4.143 ms with the ordinary wheel. The difference is
0.006 ms, or 0.14%; the port does not explain a multi-millisecond loss.
Original best-of-five runs reproduce roughly 31–35 us fused gate/up and
22–24 us qkvz with a/b. Corrected IQ3_XXS coefficient handling in the wheel
also reduces its numerical error; precision is not reduced.

A graph-context control interleaves the same compiled Torch strided-read
kernel over either reused addresses or a 10 GiB range. Nsight attributes
only projection kernel service, excluding the preceding read kernel:

| Reader | Isolated us | Reused-address interleaving us | Wide-address interleaving us |
| --- | ---: | ---: | ---: |
| Original dmv11 | 11.118 | 11.379 | 19.094 |
| Ordinary wheel | 11.416 | 11.723 | 19.367 |

Both readers reproduce the context cost. This establishes a limitation of
substituting isolated measurements for model-graph service; it does not
identify the hardware mechanism. Prototype fixtures use contiguous first
quarters of GGUF tensors. Actual GDN head-selected shards and layout are
validated separately by the loader-equivalent tests above; the prototype
fixtures alone are not a model-level correctness check.

The historical baseline also needs the matching source revision. The
projection batch at `49ac1c58ff` measured 20.639/21.734 ms for 1K/8K,
with 11.372 ms of projection trace service. The later output batch at
`6d11576ee9` measured **19.512/20.557 ms**, with **10.542 ms** of
projection trace service. Its 1290 MHz fully connected NVLink machine
differs from the current same-wheel comparison's topology and clocks.
Neither historical record supports treating 12.8 ms as current main's
projection baseline or subtracting 7.3 ms from a later model result.

The complete isolated audit's 1.872 ms increment is consistent with the
already measured 1K full-round improvement of 1.668 ms. The 8K improvement
is 1.018 ms; isolated sums do not explain that cohort's smaller benefit.
The measured model projection service remains 7.76–7.86 ms, above the
6 ms goal. No 7.3 ms model speedup or 13.34 ms model round is claimed.

Per-class measurements, operand footprints, clock histograms and evidence
hashes are in
[the audit data](data/gguf_dmv_operator_audit_20261006.json).

A new ordinary-wheel cold-rotation ABBA uses the actual loader-equivalent
TP4 shards, including `GGUFHeadTilingLayout.shard_weight` for GDN output.
All eight down/GDN-output type cases pass official-dequantization checks.
KW4/TN2/split1 remains faster than KW4/TN4/split2, KW8/TN2/split1 and
KW6/TN2/split1 in every tested case; none of those alternative configurations
is admitted.

The IQ3_S GDN output operator measures approximately 9.6–10.1 µs in
isolation, compared with approximately 18 µs in the target graph. A
same-operator Nsight calibration adds less than 1 µs, which does not explain
that difference. A sparse access spanning a 10 GiB allocation before each
operator reproduces 18.4 µs; returning to the original weight rotation
restores 10.1 µs. Interleaving a small contiguous-access kernel instead costs
only about 0.1–0.2 µs. This separates large-address/cache pressure from mere
kernel switching, but does not identify address translation as the cause.

Packing the unchanged planes into one device arena with 256-byte or 2 MiB
alignment does not improve the large-working-set case (18.5–18.7 µs).
The arena change is rejected. No precision, reader or shape admission was
changed for these diagnostics. Isolated operator latency is not substituted
for the measured model projection service.

An unprofiled CUDA-event subtraction confirms that the large-working-set
cost is not wholly a Nsight artifact: combined sparse-access/projection
service minus sparse access alone is approximately 16.7 µs, compared with
9.7 µs for the isolated projection. Repeated graph-node measurements of
KW4/TN2 with split 1/2/4 and KW2/TN2/split2 give 18.48/18.60/23.91/20.81 µs
under that pressure. The additional split configurations are rejected.

A paired NCU application-replay capture does not preserve that latency gap:
the direct normal/wide launches measure 14.98/12.32 µs. Consequently its
counter differences cannot identify the cause of the graph-context cost.
They are diagnostic evidence only, not support for a translation bottleneck
or a new production configuration.

A symmetric-u4 expansion of IQ3_S is being evaluated as a research operator.
Its codes represent exact odd signed values `v = 2u - 15`; group
coefficients remain FP16 and MMA accumulation remains FP32. Replacing all
144 IQ3_S tensors would add at most one bit per weight, or 0.247 GiB per
TP4 card. This is a storage bound, not a measured model allocation or speed
claim; the prototype has not been admitted or packaged in the model route.

The first symmetric-u4 research comparison uses a real IQ3_S down shard
(N5120,K4352,M8). Its output is bitwise equal to the admitted reader; relative
L2 against official dequantization is 3.54e-4. The expanded plane is
12.534 MB versus 9.748 MB. Same-process ABBA graph service improves only
from 22.28 to 21.22 µs in isolation and from 30.47 to 29.01 µs under
large-address interleaving. Clocks are not locked in this prototype
(1290–1485 MHz samples); the comparison is exploratory and uses a private
research extension. It does not qualify a production route or a model-level
speed claim. The representation is not admitted on this evidence.

A same-preceding-kernel control reads either a repeatedly reused strided
address range or a 10 GiB range. Both use the identical compiled Torch
elementwise kernel. Projection service is 11.78 versus 19.47 µs, with
11.50 µs without the preceding kernel. This strengthens the address/cache
pressure finding without attributing it to a particular translation cache.

An IQ3_S reader prototype adds code/scale endpoint L2 prefetch hints outside
the main loop. Outputs remain bitwise equal, but it does not improve the
large-working-set case. The hint is rejected rather than added to the
production kernel.

Explicit weight-page reads do not recover isolated latency: the wide case
measures 18.15 µs and warming code/scale page endpoints measures 18.63 µs.
Replicating page reads across 80 CTAs still gives 18.37 versus 18.10 µs,
plus approximately 6.11 µs of warming service. No warming kernel is admitted.

Wider N tiles without split-K also lose in the wide case: KW4/TN2,
KW4/TN4, KW2/TN4 and KW4/TN1 measure 18.47/20.51/21.35/19.90 µs. A separate
10/100 µs delay control, with and without a final dummy MMA, leaves
projection service at approximately 10.1 µs. Neither longer idle time nor
a tensor-core startup penalty explains the reproduced wide-access cost.
