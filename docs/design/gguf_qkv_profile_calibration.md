# M8 three-format QKV and graph profiling calibration

The latest qualified ordinary wheel is source-equivalent to main
`d2a0cc20d0`. With TP4 on four V100-SXM2-32GB cards connected by NV2, CUDA
12.8, Torch 2.10.0+cu128, FP16 activations and KV, FP32 SSM state, and all
admitted paths enabled, the matched 16-prompt comparison measures complete
rounds of **16.097 ms at 1K** and **16.875 ms at 8K**. Active clocks are
1290 MHz SM and 877 MHz memory. These results remain above the 12 ms target.
The 600-output-token qualification uses max length 262144; it is separate
from the shorter profiling workload below.

## Latest trace

The single latest-main capture uses 1K input, 64 output, max length 32768,
TP4, temperature 0.7, top-p 0.95, top-k 20, and seed 123. Rank-0 rounds
start at the first target GPU graph node and end at the next such node.
Thirteen steady rounds average 17.823 ms: 14.121 ms in the target graph and
3.702 ms afterward. This profiled workload differs from the unprofiled
qualification; their difference is not an exact profiler-overhead estimate.

| Target role | Calls/round | us/call | Service ms/round | Weight MB/call | Weight-stream GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up planes | 62 | 48.639 | 3.016 | 21.238 | 436.6 |
| gate/up native fallback | 2 | 72.928 | 0.146 | 14.449 | 198.1 |
| down | 64 | 27.513 | 1.761 | 11.304 | 410.9 |
| qkvz and a/b | 48 | 34.300 | 1.646 | 9.924 | 289.3 |
| GDN out | 48 | 19.897 | 0.955 | 3.932 | 197.6 |
| attention q/k/v | 16 | 36.751 | 0.588 | 8.781 | 238.9 |
| attention o | 16 | 19.155 | 0.306 | 3.825 | 199.7 |
| Total projections | 256 | — | 8.418 | — | — |

Weight-stream throughput divides loaded operand footprints by profiled
service. It is **not measured HBM bandwidth**: activation, metadata/cache
transactions, and profiler perturbation require separate counters. IQ2
inverse tables are omitted because the M8 kernel does not read them.

Other target service comprises communication/reduction 1.742 ms, GDN
state/gating 1.222 ms, attention 1.033 ms, standalone norms 0.042 ms,
layout/copies 0.081 ms, and other target kernels 0.487 ms. The post-target
portion contains draft GEMM 0.994 ms, draft attention 0.294 ms, target head
0.272 ms, draft shared head 0.275 ms, communication/reduction 0.214 ms,
sampling/sorting 0.243 ms, and other kernels 0.829 ms. Kernel service sums
are non-additive; graph envelopes and gaps are retained in the data.

The draft split attention consists of five `part`/`merge` pairs; classifying
these names as generic kernels hid its improvement in the previous parser.
One outlying round adds roughly 2.3 ms of graph work dominated by full-vocab
sampling fallback: scan, radix sort, softmax, and resampling. It is not an
extra GGUF weight-read pass. Most remaining rounds are approximately
17.6–17.7 ms under profiling.

## Calibrate a serving-operator chain

An isolated projection calibration alone does not bound perturbation in an
interleaved graph. A new control chains the serving QKV, GDN split,
convolution/gating, packed recurrent update, output norm, and output
projection. It uses real layer-4 TP0 projection planes, synthetic small
weights/recurrent state, and sixteen rotating banks. Thus it localizes
measurement behavior, but does not establish a model-level quality result.

| Graph size | Event mean before capture ms | Profiled graph envelope ms | Unprofiled output-projection increment us | Profiled output-node us |
| --- | ---: | ---: | ---: | ---: |
| 96 nodes | 0.997 | 1.295 | 12.494 | 17.679 |
| 768 nodes | 8.200 | 10.511 | 12.127 | 18.151 |

The capture replays each graph twenty consecutive times. First and later
replays agree, and capture-window samples remain 1290/877 MHz. This rejects
first-replay cold start and clock ramp as explanations of this particular
difference. Enlarging the graph adds only about 0.47 us to the output node.

The graph-envelope comparison demonstrates substantial perturbation in
this operator chain. Event subtraction is an incremental whole-graph cost,
not a direct unprofiled kernel-service measurement. Earlier wide-address
controls still show a real incremental cost, but they do not justify
assigning the full model-minus-isolated trace difference to that mechanism.
In particular, the previously tabulated 2.66 ms projection gap must not be
subtracted from model latency as promised savings.

## Three-format attention QKV

Three attention layers retain the older shared-input operator because Q,
K, and V use three decoder formats. The DMV operator now compiles the
Q4_K/IQ4_XS/IQ3_S and Q4_K/IQ4_XS/IQ3_XXS family sets in one launch.
Admission is restricted to measured M8 FP16 TP4 shapes K5120 and widths
3072/256/256, with source order part of the configuration key. Other M values
restore the existing canonical representation. The native support query
rejects a newer Python package paired with an older extension.

Research-only cold-bank graph ABBA on real weights at 1290/877 MHz gives:

| Layer | Q/K/V types | Existing us | New us | KW/TN/split |
| --- | --- | ---: | ---: | --- |
| 35 | IQ3_S/IQ4_XS/Q4_K | 33.268 | 24.707 | 4/2/1 |
| 47 | IQ3_XXS/IQ4_XS/Q4_K | 35.196 | 23.407 | 4/2/1 |
| 63 | Q4_K/IQ4_XS/IQ3_S | 34.343 | 22.602 | 4/2/2 |

The weighted operator saving is 0.032 ms/round, not an end-to-end claim.
Relative L2 against official FP16-dequantized weights and FP64 reference
products is below 0.000704 across three input amplitudes. Scale precision
and FP32 accumulation are unchanged. Ordinary-artifact GPU and model checks
must complete before promotion.

[Portable measurements and evidence hashes](data/gguf_qkv_profile_calibration_20261007.json)
