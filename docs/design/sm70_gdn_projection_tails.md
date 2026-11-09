# SM70 GDN projection tail copies

The opaque input-projection core reuses the existing split-copy kernel for
M5/M20 FP16 QKVZ `[M,4096]` and b/a `[M,24]`. It writes z directly into the
caller output and materializes b/a together. QKV keeps its original view,
row stride and existing contiguous policy; recurrent dispatch and arithmetic
are unchanged. Interleaved layouts, replicated b/a projections, other rows,
dtypes and unsupported storage retain the original path.

Four installed V100 GPU tests pass: all FP16 payload bits, QKV aliasing,
changed-input graph replay and the opaque model boundary against original
recurrent inputs. No projection or recurrence numerical operation changes.

Python 3.12.3, Torch 2.10.0+cu128, CUDA 12.8.93, V100-SXM2-32GB.
The benchmark captures 36 distinct synthetic projection outputs with the real
local geometry; five samples of 100 warmed graph replays are measured.

| Rows | Separate copies (us) | Fused tails (us) | Saved (us) |
| --- | ---: | ---: | ---: |
| 5 | 274.145 | 48.763 | 225.382 |
| 20 | 274.432 | 72.663 | 201.769 |

This is a copy-chain measurement, not complete-model round improvement.
Run `benchmarks/kernels/benchmark_sm70_gdn_projection_tails.py --output result.json`.
Measured operator source: `53f6258389`; model-boundary tests use the ordinary
source-containing `1.5.2.dev734+g73a8aa7e2` package. No private kernel DSO or
preload is needed. Whole-model testing is reserved for combined changes.

## Default forward integration

The input/core boundary is disabled by default, so its projection-tail
integration alone does not reach the ordinary MTP full-forward path. The
same copy-only helper is now called from that path for the existing M5/M20
non-interleaved geometry. Input/core and recurrent-state strategy selection
remain unchanged; QKV retains its producer view and row stride.

The ordinary package `1.5.2.dev757+g05be58756` passes both default-entry GPU
cases and all 213 dependency checks. Z/QKV/b/a payloads and QKV aliases
match the separate-copy path bitwise across three changing inputs and CUDA
graph replay, with the input/core boundary disabled.

A 36-call default-forward graph screen includes the original compile-graph
index gathers, separate copies and unchanged zero-initialized core buffer.
Ten alternating-order samples of 100 replays measure:

| Tokens | Separate chain µs | Tail-copy chain µs | Local saving ms |
| ---: | ---: | ---: | ---: |
| 5 | 581.965 | 86.441 | 0.496 |
| 20 | 595.640 | 94.223 | 0.501 |

Inputs have the actual projection dimensions; the copy-only comparison uses
synthetic payloads and omits GEMM and recurrence. These graph timings do not
establish complete-round savings.

Whole wheel SHA256:
`aede363ae1a6233b11a0ea18c98c5d63715ef41b6f888ba6afa184657199d623`.
Loaded core SHA256:
`b82a3ef00ddf7abcd80284548474becb2c06cbaa15e3791881db30344db5e1df`.
