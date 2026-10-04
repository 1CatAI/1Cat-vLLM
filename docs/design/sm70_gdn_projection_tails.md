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
