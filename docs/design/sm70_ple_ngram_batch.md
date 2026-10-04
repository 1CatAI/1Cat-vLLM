# Small-batch PLE n-gram preprocessing

The existing GPU path packs request histories, computes shifted tensors and
hashes each n-gram order through many small Torch kernels. For SM70 batches
of 2–32 tokens, one Triton launch reads the current token and its two history
tokens directly. Each token has one CTA. It retains signed 64-bit wrapping
arithmetic, non-negative remainder, EOS resets and request boundaries.

The capability requires trigram geometry with eight heads per order,
contiguous integer inputs on one SM70 device, two history tokens per request
and at most 32 requests. Unsupported descriptors report a fallback reason.
The existing M=1 and CPU offload paths remain available. Disk-row gathering
is a separate operation.

## Correctness

The ordinary source-containing package passes 13 checks: twelve SM70 cases
and one CPU capability check. Integer indices match the existing GPU method
exactly for int32/int64 input, M=2/5/10/20/32, EOS, empty requests, duplicate
boundaries and graph padding. CUDA graph replay remains exact after changing
tokens, history and request lengths. No floating-point arithmetic changes.

## Operator timing

Environment: V100 SXM2 32 GB, driver 580.173.02, Torch 2.10 CUDA 12.8,
Python 3.12.3. Source `2dd2b311ab`, ordinary package `1.5.2.dev527`.
The packaged core SHA256 is
`efc4cb077ae69c45848c187a048b09683f71a3a007c8fdc2623114f0834941c7`.
Fresh import and package dependency checks pass without private library
overrides. The implementation ships as Triton source in the ordinary package.

Synthetic integer inputs use the 16-head model geometry. Each graph contains
eight complete method calls. Times are the median of five samples, each with
100 graph replays, normalized per method call.

| Tokens | Requests | Torch path (µs) | Fused path (µs) | Saving (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 1 | 104.988 | 1.610 | 103.378 |
| 10 | 2 | 101.780 | 1.612 | 100.169 |
| 15 | 3 | 101.852 | 1.610 | 100.242 |
| 20 | 4 | 104.549 | 1.635 | 102.915 |
| 32 | 7 | 110.673 | 1.746 | 108.927 |

For one target PLE invocation per verification round, the M=5 result projects
0.103 ms less n-gram service. This is an operator estimate, not a measured
round or emitted-token throughput improvement.

```bash
python -m pytest tests/kernels/test_sm70_ple_ngram_batch.py
python benchmarks/kernels/benchmark_sm70_ple_ngram_batch.py --output RESULT.json
```
