# DFlash2 long-context verification curve

## Frozen scope

Integration base: `80545c010bbf6f5ed06458d992c189d75d0eff8f`. The task includes
the diagnostic-only changes from PR #586 at
`720457ff4a8f6361e8160bd1cc9210d327d9b8de`. Runtime numerical/performance
comparisons use a frozen source and native-library manifest, physical GPUs
4–7, TP4/B1/q8, QUASAR target revision
`d8e6fbfa3e3a78899b440222b827430045a05b44`, DFlash2 draft revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, CUDA 12.8, Torch 2.10.0+cu128,
E4M3 target KV, FP32 logits/state and the existing compensated attention.
Original FlashQLA GDN prefill and the verified FA2 sidecar stay enabled.
Capacity remains 262144; measured inputs stop at 131072 tokens.

For adjacent lengths 32K/64K/128K, candidate complete-round growth must not
exceed frozen baseline cold-prefill unit-token cost growth. Every long-context
absolute round cost must improve, and 1K must not regress. The initial
diagnostic reference ratios are 1.115, 1.211 and 1.350 for 32→64, 64→128 and
32→128 respectively. Freeze the final ratios after three independent baseline
startups; never relax them by slowing prefill or the shorter decode point.

## Implementation order

1. Repeat the restored-path baseline, one verified cold request and five warm
   requests per length/startup. Retain source/library hashes and all-rank route
   evidence. Supplement the existing 64K q8 trace with a 128K trace.
2. Independently screen aligned E4M3 vector loads and QK live-range/unrolling
   changes, preserving the 80-split, N32, K16 compensation and FP32 partial
   contract. Require byte-exact output and full partial/max/sum buffers.
3. Develop overlapping loads only after these measurements. If necessary,
   evaluate 80/160/320 splits, grouped-head KV reuse and versioned workspaces.
   Any graph specialization belongs in the actual MRV2 graph manager.

Model gates retain fixed-prefix distributions, acceptance and natural-output
checks. Arithmetic variants require independent FP64 error measurements before
the final FP16 cast. Never remove precision compensation as an optimization.
No experimental route is enabled by importing its builder or benchmark.

## Progress and artifacts

The first repeated baseline is queued with native dependencies frozen. Private
launch/build manifests and raw results are retained in the task artifact
archive; generated libraries and private cache paths are excluded from Git.
There is no new performance or numerical-admission result yet.

Prior rejected experiments remain recorded in the context-cost and long-verify
worklogs. Historical E5M2 and FP16-partial Pack-GQA timings are design references,
not quality/performance evidence for this E4M3 FP32 path.
