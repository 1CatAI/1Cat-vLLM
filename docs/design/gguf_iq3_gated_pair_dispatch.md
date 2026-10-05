# IQ3_S gated-pair dispatch on SM70

Merged GGUF gate/up projections previously launched two canonical GEMMs,
concatenated their outputs, and launched SiLU/multiply. The calibrated IQ3_S
pair reads the source-sized lossless records and performs FP32 MMA,
FP32 split-K reduction, and the existing FP16 projection/activation rounding
in one launch.

Only `(types=(IQ3_S, IQ3_S), M=8, N=4352, K=5120)` is admitted.
Other row counts retain the canonical projection operators. The row-count
choice stays inside an opaque operator, including prefill workspace lookup,
so range compilation cannot freeze the prefill branch. Layout-transformed
layers, mixed types, unknown dimensions, missing operators and disabled
kernel policy retain their existing routes. Preparation reports explain
these conditions. No qkvz route is enabled by this change.

The model uses its existing fused-SiLU linear interface. Loading prepares
source-byte records before releasing checkpoint storage and retains
canonical storage for the other M ranges. Records do not expand indices,
signs or the original two scale levels. This adds 19,148,800 bytes per
admitted layer per TP4 rank; eight layers total 153,190,400 bytes. The
codebook and raw decoder retain their llama.cpp/TurboMind provenance and
licenses.

## Numerical and graph checks

The CPU permutation independently reconstructs every original source byte.
Random blocks, disabled/missing capability conditions, mixed types,
uncalibrated dimensions, FP16 policy and actual-M fallback are tested.
A single dynamic compilation covers M=512/8/16/512 and keeps exactly one
opaque gated-pair node. The existing runtime projection tests are rerun.
Six targeted CPU tests passed with Torch 2.10 / CUDA 12.8.

The public nibble-book decoder uses exact signed lookup and half unpacking;
the existing raw lattice device API remains intact. The original d and small scale
remain separate. The reference forms GGUF weights using the official reader,
and forms projection products with FP32 GEMM. Projection results and SiLU
round to FP16 as in the existing path; the dot products and reduction remain
FP32. A new packaged-operator benchmark checks native M8 and canonical
M1/M16/M512 before comparing cold-L2 graph medians in ABBA order.

## Research measurements

Frozen micro shape: TP4 rank0 layer6, M8/N4352/K5120, actual model weights,
FP16 activations, FP32 accumulation, 16MiB L2 eviction before each event,
84 timed samples, V100 SXM2 32GB, SM/memory 1290/877MHz, 300W,
Torch 2.10 / CUDA 12.8. Source-byte bandwidth is a workload ratio, not the
NCU DRAM counter. The following operators were research builds; packaged
operator measurements are required before promotion.

| Variant | Graph median | Source bandwidth | Resources |
| --- | ---: | ---: | --- |
| Original signed HFMA pair | 62.464us, earlier ABBA | 306.56GB/s | 62.625 instructions/K16, 50 registers, 32KiB shared, no spills |
| One 16KiB signed book | 62.464us, both ABBA controls | 306.56GB/s | 78.125 instructions/K16, 51 registers, no spills |
| Two 16KiB books, thread parity selects copy | 62.464us, both ABBA candidates | 306.56GB/s | 82.125 instructions/K16, 52 registers, 32KiB shared, no spills |
| Same-round signed HFMA control | 63.488us | 301.61GB/s | Unchanged decoder |
| Same-round NVFP4 pair, split8/16 | 51.200us each | 489.60GB/s | 25,067,520 source bytes |

Duplicating the book was numerically bitwise equal to the retained control
but gave no timing improvement. It is not selected. Read-only-cache,
texture and bank-permutation variants are already rejected and are not
repeated. Register prefetch/interleaving experiments are also closed.

A subsequent matched shared-activation prototype loads A once per CTA into
padded shared rows, reused by gate and up. Its ABBA medians are
63.488 / 50.176 / 50.176 / 63.488us (non-staged / staged / staged / non-staged),
with the original single-book control at 62.464us and NVFP4 at
51.200/52.224us for split8/16. SM/memory remain 1290/877MHz. Output is
bitwise equal to the retained IQ3 control; official-reference relative L2
is 0.000495 and maximum absolute error 0.00390625, unchanged. The staged
kernel uses 48 registers, 33,792 shared bytes, and no spills. Static loop
instructions increase to 82.375/K16 while global-load sites decrease from
22 to 7. This validates activation sharing rather than instruction-count
reduction. The packaged candidate uses this measured N32 shared-A version.

Wider N64 tasks with partition2 last-CTA reduction were also tested in one
matched sweep. Grid80 gave 58.368us and grid160 gave 53.248us, repeated
with unchanged results against N32 controls of 50.176us. The candidates
used 62 registers, no spills and 41,472 shared bytes. The changed FP32
reduction tree produced relative L2 7.153e-6 versus the retained output;
all counters reset correctly across the 84-sample graph replay. These
variants are slower and are not admitted. No separate reduction launch was
introduced.

Only eight layers have two IQ3_S FFN projections. Their source byte share is
12.765% of all gate/up pairs. A 62.5us pair versus the recorded 76–81us
canonical pair saves about 13.5–18.5us per eligible layer, approximately
0.108–0.148ms across those eight layers. This is a projection estimate,
not a measured full-round saving. The 40 mixed-type pairs account for
62.286% of pair bytes; see the complete
[source inventory](gguf_qwen38_iq3s_source_inventory.md).

## Activation traffic

A separate SourceCounters capture compared the signed HFMA and NVFP4
kernels, using the same M8 real-weight shape. SASS load addresses identify
activation and weight/codebook operations. Counters and profiler service
latencies below must not be substituted for unprofiled graph medians.

| Counter | IQ3_S | NVFP4 |
| --- | ---: | ---: |
| Actual total L1 global read bytes | 68,497,152 | 69,629,024 |
| Actual total L2 read sectors x32 | 31,000,096 | 31,630,048 |
| Actual DRAM read bytes | 19,466,304 | 25,149,664 |
| Activation source-correlated L1 tag requests | 1,392,640 | 1,392,640 |
| Weight/codebook source-correlated L1 tag requests | 192,576 | 261,120 |
| Activation theoretical L2 sector bytes | 44,564,480 | 44,564,480 |
| Weight/codebook theoretical L2 sector bytes | 23,814,144 | 25,067,520 |
| Profiled kernel service | 59.648us | 47.520us |

Activation tag requests are 7.23 times the IQ3_S weight/codebook requests.
The 44.56MB source-correlated value is theoretical sector demand before
cache hits; it does not mean measured activation-only L2 bytes. The
hardware counter provides total L2 bytes, without separating A and B.
The repeated activation demand justifies CTA-local staging and gate/up
reuse. The matched shared-A result is recorded above. Wider-N changes require
their own correctness, resource and matched graph comparison before replacing
that candidate.

## Reproduction

`benchmark_gguf_iq3_gated.py --model MODEL.gguf --output RESULT.json` uses
only the installed operator, records clocks and numerical errors, and
checks runtime canonical fallbacks. Acquire the shared GPU lock and the
selected device lock before running it. The complete source inventory is
reproducible with `benchmark_gguf_quantization_inventory.py`.

Latest-main end-to-end baseline, graph-boundary ledger and clean packaged
operator measurements are pending the normal CUDA build. No new
end-to-end claim is made from these research timings.
