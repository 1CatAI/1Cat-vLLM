# DFlash2 attention resource experiments, September 10

## Scope and acceptance

This follow-up is based on the long-verification PR's
`1994baff8f188a2c915e883395ae96e682db0577`, with integration target `onecat/main`
(observed `c2ba9e2929cffe44e34cef8bb22dde99e2dc7b16` at branch creation).
It adds private operator experiments; it changes no default serving route.
The complete-round target remains **22 ms at 261888 input tokens and a
256-token output budget**, within 262144 service capacity. Decode time divided
by speculative rounds includes terminal q1 and partial-verifier steps.
An attention operator time is not this complete-round metric.

The target is QUASAR-QAT Qwen3.8-27B NVFP4 through TurboMind W4A16/QPN2,
TP4 on V100-SXM2-32GB, FP16 activations, E4M3 KV, FP32 state/logits,
CUDA 12.8, Torch 2.10.0+cu128 and Python 3.12.13. Its 16 full-attention
layers retain the complete context. The DFlash2 draft's 2048-token window
does not restrict target attention. Keep temperature 1, top-p 0.95, top-k 20,
the existing seeds, native FA2 prefill and original GDN prefill.

The existing visible-q8 parent preserves 80 logical splits, N32 updates,
K16 compensated QK products and probability-residual PV products. The
following experiments preserve that arithmetic and FP32 workspace contract.
All timings below use sixteen distinct layer KV allocations, page size 3296,
query shape `(8,6,256)`, alternating CUDA-graph A/B order and five event samples.

## Exact E4M3 lookup

`build_sm70_grouped_attention_candidate.py --e4m3-shared-lut` initializes
a 256-entry, 512-byte shared table with the existing E4M3 bit decoder.
The initial CTA barrier publishes the table before aligned 16-byte loads
or the existing stride fallback consume it. The option requires an isolated
six-head fixed-q8 build, vector loads and 80 splits. It is off by default.
Without the option, the generated parent source retains SHA256
`3b0c9688ce17e1870408ef81fb5cd9b63a677b7cfd7d4777b8df77dd0fc24132`.

| Sixteen-layer operator | Paired parent, ms | Lookup, ms |
| --- | ---: | ---: |
| 131072 tokens | 7.727 | 7.599 |
| 261888 tokens | 14.874 | 14.624 |

All 65 output/full-FP32-workspace/canary checks pass, including page crossings,
stride padding, zero/rejected rows, restored lengths and 262144 visibility.
The candidate's memcheck, racecheck and synccheck each pass 25 such checks
with zero errors or race hazards. Sanitizer runs contain no performance work.
The paired 3296-page q8 specialization uses 126 registers and 512 static
shared bytes, without reported stack or spills; dynamic shared memory is
73728 bytes. These are compiler resources, not achieved occupancy.

Source SHA256:
`e78e922b17ff44262c49e6664871d0692e0ecd048b676061516d0419d3819f92`.
DSO SHA256:
`3b5a53e413ce064cccac4c735f090fbd7f9e4beffb097120d15e1861f36a2ed2`.
Raw reports: `q8-kv-lut-screen.json`, `q8-kv-lut-{memcheck,racecheck,synccheck}.json`.
The first same-startup service A/B keeps compact scalar q1 in both arms.
Both endpoint lengths run one cold request followed by five repeats per arm.
All twelve request pairs retain identical tokens, finish reasons and acceptance;
each rank records 384 actual compact-scalar q1 calls per arm.

| Unprofiled endpoint | Parent | Lookup |
| --- | ---: | ---: |
| 1K complete round, ms | 15.870 | 15.691 |
| 261888 complete round, ms | 35.034 | 34.646 |
| 261888 pure decode, tokens/s | 145.575 | 147.202 |
| 261888 accepted drafts/round | 4.020 | 4.020 |
| 261888 emitted tokens/round | 5.120 | 5.120 |

The five-repeat 261888 prefill medians are 1.196/1.199 seconds and TTFT
1.429/1.430 seconds. These repeats retain the benchmark's prefix-cache policy;
they are not cold-prefill throughput. Cold requests remain in the raw reports.
The source is `1994baff`, whose runtime Python and Flash-V100 source match the
frozen `239d71c` service revision. Report: `q8-kv-lut-model-1-paired.json`.
This is one paired startup. The **22-ms target remains unmet**; repeated-startup
and new-candidate natural-output admission are incomplete.

## Rejected parallel QK products

`build_sm70_grouped_attention_qk_parallel.py` binds the hashed visible-q8
parent. Twelve warps compute independent K16 products into four shared slots;
six consume them with the original compensated addition order, and four
load V. The extra product storage raises dynamic shared memory to 98304 bytes.
The 3296-page q8 specialization uses 128 registers without reported spills.

All 65 byte/canary checks pass, but the added storage and barriers fail the
performance screen. At 131072 tokens the paired parent/candidate costs are
7.705/14.075 ms; at 261888 they are 14.877/27.464 ms. Reject before sanitizer
or model trials. More participating warps alone is not evidence of better
resource utilization. Preserve this negative result to avoid repeating it.

Source SHA256:
`71fa3ec257b358b6ef40956df2d4fe56c2777c6f0f9edd6e12a4b0a5a3d62134`.
DSO SHA256:
`50833f8b2fb025274635d719e0fe264cef41784dbe4acb0bdd758e4f522b5ca9`.
Raw report: `qk-parallel-screen.json`.

The existing two-head-group screen had not combined vector loads and QK
unroll2. Screening that combination with 3296-page specialization retains
all 65 byte/canary checks, but costs 11.635/22.766 ms at 128K/261888 against
its paired parent's 7.698/14.871 ms. Reject it before sanitizer or model runs.
Raw report: `heads2-vector-qk2-screen.json`. Reducing the per-CTA footprint
does not by itself overcome duplicated KV reads and padded three-head rows.

## Reproduction and promotion gates

Build the lookup using the measured parent's options:

```bash
.venv/bin/python benchmarks/kernels/build_sm70_grouped_attention_candidate.py \
  --output-dir "$CANDIDATE" --head-groups 1 --qk-unroll 2 --vector-load \
  --page-specialize --prefetch-v --reuse-pv-values --specialize-full-q8 \
  --all-visible-tiles --e4m3-shared-lut --grouped-only --build
.venv/bin/python benchmarks/kernels/benchmark_sm70_grouped_attention_long.py \
  --baseline "$PARENT/manifest.json" --candidate "$CANDIDATE/manifest.json" \
  --output "$REPORT" --performance-page 3296 \
  --performance-contexts 131072 261888 --extended-boundary 262144
```

Set the CUDA 12.8 toolkit, `TORCH_CUDA_ARCH_LIST=7.0`, owned compiler caches
and the GPU lease explicitly. For each sanitizer use the complete CUDA 12.8
bundle and replace timing options with `--sanitizer --correctness-only
--extended-boundary 262144`. Check source/DSO hashes in every report.

The owned worklog retains absolute artifact paths, environment, commands,
process ownership and service results. Native correctness and compiler-resource
counts do not establish full-model quality, repeatability, or the 22-ms target.
Promotion requires actual route hits, matching output tokens and acceptance,
natural-output gates, repeated unprofiled startups and the context sweep.
