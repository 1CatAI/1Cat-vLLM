# DFlash2 split windows on 832-token KV pages

The GGUF hybrid cache uses interleaved `[pages, 2, 832, 2, 128]` FP16
storage. After splitting K/V, the page stride is 425984 elements. The
existing split-window implementation accepts this page size and stride,
but both adapter and backend admission previously required 1024/2048.
The fallback uses a small paged grid instead of splitting the live window.

Admit B1/Q8/H8/KV2/D128, window `(2047, 2047)`, FP16 KV and SM70 to the
packaged FP32 split path. Keep the native concurrent BMHD page ABI limited
to 1024/2048. Page832 concurrent requests, unsupported layouts and missing
split imports retain general paged attention. The per-engine
`speculative_config.sm70_dflash2.draft_window_split` setting defaults on,
provides the page832 control arm, and participates in the graph hash.
There is no new environment variable or kernel implementation.

## Validation

A normal wheel from source `4658e8aeb031ef7b95db0fea9c6b53cb4d79d36c`
contains latest integration changes and this adapter/backend/configuration
change. All installed Python and sixteen native hashes match the artifact;
the wheel SHA256 is `508348421cd6b6c545de72a831bd1a9940c54fa936e4d661066c10d9208914c0`.

Eight dispatch checks, fifteen policy checks and twelve live-graph
FP64-reference cases pass in the installed wheel. Six focused paged batch
checks also pass, including concurrent fallback behavior. The GPU tests
use interleaved K/V strides, changed live lengths, indirection, zero padding
and window-edge masks.

A preceding research comparison on five separate KV banks at the actual
832-token page/stride measures 94.403 to 55.785 us/call at 1K, 205.604 to
60.738 at 8K, and 203.123 to 60.763 at 32K. FP64-relative L2 error is about
2.0e-4 for the split path. These are isolated graph operator measurements;
multiplying by five gives estimates of 0.193/0.724 ms per round at 1K/8K.
They are not end-to-end savings. The matched model comparison below
does not show those operator savings, so this path is not admitted yet.

## Same-wheel model comparison

Both arms include projection planes and collective/norm fusion. Only the
page832 split policy changes. TP4 FP16 KV, FP32 SSM, FULL_AND_PIECEWISE,
maximum context 262144 and seven probabilistic draft tokens are fixed.
Sixteen matched prompts generate 600 tokens each, with temperature 0.7,
top-p 0.9, top-k 20 and seed 123. Exclude the first twenty output rounds.
Timing ignores EOS; natural-output checks terminate normally.

| Input | Off ms/round | On ms/round | Off tokens/round | On tokens/round | Off ms/output token | On ms/output token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 16.608 | 16.636 | 3.049 | 2.964 | 5.491 | 5.640 |
| 8K | 17.714 | 17.728 | 2.983 | 2.952 | 5.970 | 6.044 |

Every timed request records 1290/877 MHz on all four ranks. Common-context
logits retain all 128 probe rows: mean KL 6.803e-6, max KL 1.159e-4 and
top-1 agreement 100%. Natural answers are identical and terminate normally.
Four concurrent requests pass text-health checks; no C4 speed claim follows.

There is no measured round saving. Keep integration pending while a single
post-batch node trace records the actual C1 query layout and dispatch. A
compile-only recorder in the on-arm benchmark harness reads unmatched
collective boundaries; it does not mutate graphs or run during timing.
Raw counts, acceptance and artifact hashes are in
`data/sm70_dflash_page832_20261007.json`.
