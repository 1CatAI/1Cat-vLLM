# DFlash2 split windows on 832-token KV pages

The hybrid GGUF cache uses interleaved `[pages,2,832,2,128]` FP16 storage.
The draft context query is B1/Q8/H8/KV2/D128 with window `(2047,2047)`.
The public adapter unnecessarily required the native `dflash2_paged_bmhd_fwd`
symbol before trying the independent packaged Triton split implementation.
That symbol is absent from the qualified wheel, so the general paged kernel
ran five times per round at about105us/call in the diagnostic1K trace.

Admit the packaged FP32 split path independently for the qualified input
layout, dtype and page sizes. Only the native concurrent1024/2048 branch
requires its BMHD symbol. Missing split imports, unsupported layouts and
page832 concurrent requests retain general paged attention. The per-engine
`speculative_config.sm70_dflash2.draft_window_split` setting defaults on and
participates in the graph hash. There is no new environment variable or
kernel implementation, and probability/PV arithmetic remains FP32.

## Same-wheel model comparison

The source-complete wheel from `6549744cd41e9e926c26c43b2d833d09cbb51cc8`
contains the adapter/backend/configuration change and all integration fixes.
All installed Python and sixteen native hashes match its audit. Only the
page832 policy changes between arms; projection planes and collective/norm
fusion are enabled in both. Four V100-SXM2-32GB cards use full NVLink, TP4,
1290/877MHz, CUDA12.8 and Torch2.10.0+cu128. FP16 KV, FP32 SSM,
FULL_AND_PIECEWISE, context262144 and seven draft tokens are fixed.

Sixteen matched prompts generate600 tokens each at temperature0.7, top-p0.9,
top-k20 and seed123. Omit the first twenty output rounds. Timing ignores
EOS; separate natural-output checks terminate normally. Full-round means
are weighted equally by prompt; output-token costs pool all observations.

| Input | Off ms/round | On ms/round | Saving ms | Off tokens/round | On tokens/round | Off ms/output token | On ms/output token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 16.607 | 16.176 | 0.431 | 2.918 | 2.897 | 5.713 | 5.615 |
| 8K | 17.700 | 16.958 | 0.742 | 2.928 | 3.040 | 6.071 | 5.610 |

Draft acceptance is27.333→27.033% at1K and27.249→28.670% at8K; report
those changes separately from round latency. Every timed request records
1290/877MHz on all four ranks. Mean TTFT is367.618→367.194ms at1K and
2834.572→2835.586ms at8K. These are unprofiled model measurements.

## Correctness and dispatch

Eleven CPU dispatch checks and thirty-four installed-wheel policy/GPU
checks pass. They include a live page832 graph with the native symbol
explicitly absent, changed live lengths, interleaved K/V strides, page
indirection, zero-length queries, window masks and concurrent fallback.
The on-arm compilation contains split `part`/`merge` kernels while the off
arm does not. The earlier configuration comparison exercised the fallback
in both arms and is not evidence of split-kernel performance.

All128 common-context logit rows are retained: mean KL5.836e-6, max
KL3.996e-5 and top-1 agreement100%. Natural numerical and English prompts
finish normally with identical outputs. Four concurrent requests pass text
health checks; no C4 throughput claim follows. Artifact hashes, counts,
acceptance and the full-round comparison are recorded in
`data/sm70_dflash_page832_20261007.json`.
