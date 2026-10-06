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

Initial operator checks used real TP4 27B rank-0 shards on V100-SXM2-32GB
with post-run clock readings of 1290 MHz, CUDA 12.8 and Torch 2.10.0.
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
