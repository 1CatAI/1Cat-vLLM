# TP4 GGUF DFlash2 verification cost

Measured on 2026-10-08 with main `20ddd18ad6`. The installed complete wheel
is source `99de68fe3c`; its runtime files match that main revision. No runtime
optimization is introduced by this audit. The complete-round target below
12 ms remains unmet.

## Workload and unprofiled results

Four V100-SXM2-16GB GPUs have pairwise NV2 links. The runtime is CUDA 12.8,
Torch 2.10.0+cu128 and Python 3.12.3. Active samples record 1530 MHz SM and
877 MHz memory clocks; clocks are sampled rather than fixed.

The target is Qwen3.8-27B-GSQ-RCO-IQ3_S, with a Q8_0 DFlash2 draft and seven
probabilistic draft tokens. TP4, FP16 activations, E4M3 target KV, FP16 draft
KV and FP32 GDN state remain unchanged. CUDA Graph, projection planes, signed
IQ2 nibbles, draft-window split and allreduce/RMS fusion are enabled.
Maximum length is 32768, prefill budget 1024, maximum sequences one and GPU
memory utilization 0.88. Prefix caching is disabled.

One exact-length engineering-note fixture runs at 1K and 8K with 256 output
tokens, temperature 0.7, top-p 0.9, top-k 20 and seed 123. Timing ignores EOS
and excludes the first twenty output intervals. Separate natural requests
answer `391` and explain unit testing, both with normal EOS. This short
diagnostic does not replace the sixteen-prompt performance cohort or the
separately recorded 256K capacity test.

| Input | Steady rounds | Mean ms/round | p50 | p90 | p99 | Tokens/round | ms/output token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 53 | 14.307 | 14.307 | 14.344 | 14.374 | 3.283 | 4.358 |
| 8192 | 47 | 14.679 | 14.683 | 14.714 | 14.761 | 3.511 | 4.181 |

CUDA events around actual M8/C1 target graph replay provide an independent
check without Nsight. After twenty warm intervals, 55 target graphs average
11.360/11.358/11.355/11.354 ms on ranks 0/1/2/3. The instrumented request
averages 14.415 ms per complete round and 3.370 emitted tokens per round.
The approximately 3.06 ms difference also contains draft, sampling,
communication and host work; it is not draft forward alone. These requests
have different sampled outputs, so their difference from 14.307 ms is not
an exact event-instrumentation overhead measurement.

## Graph-node attribution

The qualified capture uses 1024 input, 64 output, maximum length 32768 and
the same sampling and route configuration. Capture stops before natural
health requests. Boundaries follow the first target GPU graph node, linked
through graph-launch correlation IDs, to the next such node. Four ranks
each have fourteen steady intervals after edge exclusion.

Rank 0 averages 15.971 ms under profiling: target graph 12.687 ms and
post-target 3.283 ms. Target inter-kernel gaps are 1.230 ms and post-target
gaps are 0.593 ms. These are diagnostic envelopes, not the unprofiled latency.

| Target projection role | Service ms/round |
| --- | ---: |
| gate/up planes and native fallback | 2.894 |
| down | 1.619 |
| qkvz with a/b | 1.425 |
| GDN out | 0.855 |
| Attention q/k/v | 0.478 |
| Attention o | 0.262 |
| Total projection service | 7.532 |

Other target service is communication/reduction 1.569 ms, GDN state/gating
1.048 ms, attention 0.567 ms, remaining norms 0.036 ms, layout/copies
0.086 ms and other kernels 0.637 ms. There are 122 fused push-allreduce/norm
calls within 130 communication/reduction calls. All 48 qkvz calls include
a/b; the earlier separate tiny-matrix a/b path is absent.

Post-target service is draft GEMM 0.903 ms, draft attention 0.249 ms, target
head 0.253 ms, draft shared head 0.256 ms, communication/reduction 0.191 ms,
sampling/sorting 0.148 ms and other kernels 0.691 ms. The five `part`/`merge`
pairs are draft attention. The two vocabulary projections are separated by
rejection, context preparation and draft dependencies; they are not two
equivalent calls that can simply be merged.

Service sums and gaps are not interchangeable with unprofiled wall time.
In particular, a standalone GDN output takes approximately 9 us while its
profiled model node takes approximately 18 us. This difference is not
demonstrated recoverable latency.

An earlier 96-output diagnostic also captured subsequent health requests.
Its unsliced ledger contains two request-boundary intervals near 331/334 ms.
The retained diagnostic is restricted by the first request's 25 draft-graph
count; it is superseded by the capture above, not used as latency evidence.

## Exact-shape operator screens

Real rank-0 TP4 weights run at M8 in CUDA Graph with rotating banks exceeding
48 MB and ABBA order. Outputs use FP16 scale operands and FP32 accumulation;
the oracle is official GGUF dequantization and FP32 matrix multiplication.
The GDN activation-head permutation is included in the reference.

| GDN output type, N5120/K1536 | Current KW4/TN2/split1 us | Best alternative result |
| --- | ---: | --- |
| IQ4_XS | 9.714 | KW4/TN1/split1: 9.864 us |
| Q4_K | 9.866 | KW4/TN1/split1: 9.892 us |
| IQ3_S | 9.400 | KW6/TN2/split1: 10.257 us |
| IQ3_XXS | 9.038 | KW6/TN2/split1: 9.470 us |

The screen includes KW/TN combinations 4/1, 2/2, 2/4, 4/2, 6/2 and 4/4,
with split1 and split2; a separate screen also checks KW8 and split4.
No alternative beats the admitted configuration. Relative L2 is around
4e-4. Additional split CTAs do not establish a bandwidth improvement.

The existing Q8 segment operator also beats canonical GEMM for several
actual draft shapes, without writing another decoder:

| Draft role | K/N | Canonical us | Segment us |
| --- | --- | ---: | ---: |
| Context FC | 25600/1280 | 67.693 | 49.645 |
| q/k/v together | 5120/1536 | 26.181 | 15.904 |
| Attention o | 1024/5120 | 14.118 | 11.049 |
| down | 4352/5120 | 39.757 | 34.138 |

One FC plus five calls of each other role predicts approximately 0.113 ms
of isolated savings. This is not an end-to-end result. Gate/up changes only
68.563 to 65.552 us and is not admitted. The Q8 convolution-projection probe
is not a runtime comparison: those ten matrices are already dequantized and
use the existing FP16 M8 kernel. None of these probes changes dispatch.

## Hardware counters and next optimization boundary

NCU profiles the shipped IQ3_S GDN-output operator at M8/N5120/K1536 with
the activation-head mapping, grid80 and 256 threads/CTA. The full counter
collection reads 3.489 MB from DRAM and 5.652 MB through global-load L1/L2
sectors. The latter includes approximately 1.966 MB of activation requests
across output CTAs and 0.246 MB of repeated table initialization. Activation
reuse inside each CTA is already present.

Active warps are 12.47%, Tensor Core activity 10.12%, SM throughput 31.34%
and DRAM throughput 32.76%. Executed warp instructions total 1513280.
Warp-active stall metrics include long scoreboard 15.48%, short scoreboard
11.85%, wait 11.27%, barrier 10.35% and math-pipe throttle 5.16%. A separate
counter pass records 143616 shared-load bank conflicts. NCU duration is
19.392 us; it must not replace the 9.400 us unprofiled graph result. The
three-pass and ten-pass issue metrics differ substantially, so these
percentages are not a quantitative end-to-end decomposition.

This is a short, underfilled operator with decode and synchronization
dependencies, not demonstrated saturation of HBM alone. Increasing CTA count
adds work and reduction overhead in the measured configurations. The next
projection change should shorten the decoder/synchronization path or reuse
work across a larger execution boundary, rather than repeat rejected tile
settings. Lossless smaller-codebook layouts require a memory budget and
exact restoration outside M8 before admission.

The full-round gap to 12 ms is approximately 2.3--2.7 ms. Q8 shape tuning
alone is too small. Prioritize the target projection chain and the shared
communication/GDN-state boundaries, with the draft and full-head tail as
secondary work. Existing GDN factor-publication and collective work should
be reused rather than duplicated. Any new default requires same-wheel
quality, acceptance and complete-round measurements.

[Portable measurements and evidence hashes](data/gguf_tp4_verify_cost_20261008.json)
