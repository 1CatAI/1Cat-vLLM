# Shared packed INT8 reader qualification

The common `KVReader<3>` now converts each signed byte pair with exact packed
FP16 subtraction. Both `1024 + u8` and `1024 + 256 * sign` are exactly
representable; their difference is the signed byte, including positive zero.
Four pairs replace eight scalar integer/float conversions. The codec's layout,
scale application, rounding, algorithm selection and workspace are unchanged.
FP16, E4M3 and E5M2 readers retain their existing code.

This change benefits every native consumer of the shared reader. It introduces
no separate INT8 algorithm, model qualification or tuning policy.

## Correctness and native execution

`benchmark_sm70_kv_reader.py` compiles a benchmark-only harness against the
actual public `KVCodecTraits` header. All 65536 byte pairs at each of four pair
positions match independent Torch FP16 bits and the frozen original reader.
Changing inputs and retained checksums prevent loop hoisting or dead-code
elimination in the conversion experiment. Its conversion-only median changes
from 0.217600 ms to 0.141312 ms; this is not an attention or engine result.

The affected installed XQA, grouped, long-context and engine-binding checks
pass **86 distinct GPU cases**. The combined run passed 85 and failed one
profiler assertion because no partition-kernel event was returned. That
unchanged q16 fixed-stride case passes in a fresh isolated profiler process,
which records actual native execution. The initial failure remains recorded.
Four reader PTX/source-distribution checks pass, including unchanged legacy
format code and shared-header inclusion in both source packages.

## Frozen operator comparison

V100 SXM2, CUDA 12.8, Torch 2.10.0+cu128; one GPU, FP16 queries, six query
heads, one KV head, D256, page2048, non-contiguous pages. XQA uses q1; grouped
and long-context grouped use q8. Both FP8 and INT8 consume fixed encoded
inputs; this FP8 operator control does not establish model calibration.

The first short timing attempt was inconclusive: 40 graph replays exposed
baseline variation and apparent short-context regressions. Before the next
candidate run, the protocol was fixed to 20 ordinary warmups, 2000 warmup graph
replays and nine samples of 1000 graph replays. Two independent control
processes establish repeatability, followed by separate-process A/B/B/A
runs. Input and output hashes match across all six runs in every case.

| INT8 algorithm | Context (tokens) | Control (ms) | Candidate (ms) | Candidate/control |
| --- | ---: | ---: | ---: | ---: |
| XQA | 4096 | 0.056531 | 0.052458 | 0.9280 |
| grouped | 4096 | 0.045424 | 0.044319 | 0.9757 |
| grouped long | 4096 | 0.035303 | 0.035138 | 0.9953 |
| XQA | 32768 | 0.136089 | 0.128389 | 0.9434 |
| grouped | 32768 | 0.210060 | 0.206768 | 0.9843 |
| grouped long | 32768 | 0.156132 | 0.153613 | 0.9839 |

The 4K long-context difference is inside control variation and is not claimed
as a speedup. FP8 ratios for the same six cases are 1.0002, 0.9991, 1.0031,
0.9996, 0.9998 and 0.9995. The change is retained on these bounded operator
results; they do not replace end-to-end latency or quality qualification.

The standalone benchmark accepts replay counts and refuses to overwrite
evidence. Run it in each normally installed control/candidate environment in
the stated order, preserving input/output hashes and binary fingerprints.
Retained evidence basenames are `production-reader-cost-r2.json`,
`native-reader-r2.log`, `native-reader-profiler-r3.log`,
`reader-parity-r2.log`, `attention-cost-summary-r3.json` and the six
`*-cost-r3.json` files. The normal Flash-V100 component wheel SHA256 is
`0c638c12910dd8ed059546280f6556799f89eaf6fe4da3980e8e6ea4cb5d8306`;
the rebuilt normal FA2 component SHA256 is
`0dc31ceb26c3b9d08d6558b0337c14ea89273d8c70e9347964ce1b26ed0f590a`.

## Remaining acceptance

These are normal component builds with an explicitly recorded source overlay,
not complete clean-wheel acceptance. No engine loads the conversion benchmark
extension. The 43-item coverage, prefill cost regression, QSA/MLA engine
qualification, state/transfer lifecycle, paired task quality, 27B end-to-end
and matching 35B pure-decode gates remain open.
