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
They are not end-to-end savings. The matched same-wheel sixteen-prompt
model comparison is still running.
