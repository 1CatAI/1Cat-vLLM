# SM70 QPN2 code/scale layout screen

The research benchmark keeps the original native M8 grids (136 gate/up, 160
down), splits, arithmetic, logical weight bytes and activation addresses.
Each N32/K16 group stores 256 code bytes followed by 32 E4M3 scale bytes.
Both control and candidate are rebuilt with CUDA 12.8 and identical flags.
These private benchmark extensions are not serving artifacts.

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm70_qpn2_bundled_scale.py \
  --source-root . --model /path/to/model --out /tmp/bundle-screen --iters 200
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  benchmarks/kernels/benchmark_sm70_qpn2_bundle_layer.py \
  --model /path/to/model --out /tmp/bundle-layer \
  --control-extension /path/to/round12_qpn2_bundled_control.so \
  --candidate-extension /path/to/round12_qpn2_bundled_scale.so
```

The code-generating stage screen expects the baseline kernel signatures.
Freeze its source version before altering production templates. The layer
benchmark also accepts `--production-extension` instead of the two prototype
libraries, exercising complete production CUDA source registered in the
private `_qpn2_candidate` namespace.

## Matched prototype results

V100-SXM2-32GB, Torch 2.10/cu128, CUDA 12.8, cold L2, randomized paired graph
order, 200 projection and 150 TP4 layer samples:

| Interval | Control us | Bundled us |
| --- | ---: | ---: |
| Gate/up read skeleton | 41.733 | 43.648 |
| Gate/up decode skeleton | 49.551 | 46.997 |
| Gate/up full | 47.334 | 47.416 |
| Down read skeleton | 24.105 | 25.006 |
| Down decode skeleton | 26.056 | 25.528 |
| Down full | 26.870 | 26.066 |
| Complete MLP graph | 73.851 | 68.659 |
| Complete TP4 GDN layer, critical rank | 168.858 | 160.850 |

The full MLP saving is 5.192 us (95% interval 4.961–5.448); the GDN-layer
saving is 8.007 us (7.441–8.547). The graph retains ten timed compute kernels
on both arms; eviction and state resets are outside the timed interval.
The read skeleton regresses, so this is not evidence of a higher read-only
roofline. Different skeleton register pressure also prevents interpreting
read/decode/full differences as pure operation cost.

The initial comparison reused an older control toolchain. Its artifacts are
retained but excluded from these matched results.

## Complete production-source screen

The complete CUDA source with both contiguous and bundled dispatch templates
passes 16 integer bit-pattern comparisons across M1/7/8/9/16/24/32/64 and
repeated graphs. M8 MLP measures 73.144→69.842 us (saving 3.302,
95% interval 3.072–3.548). M32 measures 133.268→133.914 us (saving -0.645,
interval -0.865–-0.389). Complete TP4 GDN-layer critical-rank means are
169.643→163.772 us (saving 5.871, interval 4.943–6.813). Layer output,
residual, FP32 rollback state and convolution-history bits match at four
activation amplitudes. Both graphs retain ten timed compute kernels.

The smaller production gain and negative M32 delta require complete-artifact
same-service C1/C4 admission. No model latency or DRAM-bandwidth gain is
claimed by this research PR. Production integration is reviewed separately
in PR #1008.

## Serving admission and supported-clock control

Production PR #1008 is merged as `deca1e0da775`. The corrected standard wheel
uses the pinned Flash-V100 dependency and passes 23 targeted GPU tests, clean
artifact loading, natural EOS, and same-service C1/C4 admission. It requires
neither a private kernel DSO nor a serving library overlay. Target LM-head and
draft precision are unchanged by the bundled-layout change.

The matched serving workload uses four full-NV2 V100-SXM2-32GB GPUs, Torch
2.10/cu128, CUDA 12.8, TP4, 262144 maximum model length, four maximum sequences,
FP8 E4M3 KV, block size 2048 and memory utilization 0.8. C1 uses eight prompts
with 600 generated tokens, seed 123, temperature 0.7, top-p 0.9, top-k 20 and
thinking disabled. EOS completion is checked separately. Both layouts run in
one service process with all affected graphs recaptured between arms.

| Application clock, 300 W | Prompt | Separate layout ms/round | Bundled ms/round |
| --- | ---: | ---: | ---: |
| 1290 MHz | 1K | 14.82269 | 14.38960 |
| 1290 MHz | 8K | 15.25806 | 14.84206 |
| 1530 MHz, highest supported | 1K | 14.04183 | 13.62042 |
| 1530 MHz, highest supported | 8K | 14.44284 | 14.01472 |

At 1290 MHz, tokens/round and compact/reference counts match within each
layout comparison. Tokens/round are 2.97980 at 1K, 95% interval
2.79042–3.18596, and 2.94764 at 8K, interval 2.87709–3.03002. Reference-path
shares are 1.3497% and 1.8337%. At 1530 MHz, both layouts produce 3.09965
at 1K, interval 2.93036–3.26894, and 2.91995 at 8K, interval
2.84053–3.01017. These are acceptance statistics, not sampled-sequence gates.

A warmed ABAB C4 follow-up at 1290 MHz gives means 403.122→406.681 tokens/s;
visit variance overlaps, so it supports parity rather than a throughput-gain
claim. At 1530 MHz C4 measures 451.140→454.450 tokens/s. The M32 target graph
GPU envelope at 1290 MHz still increases by 0.052 ms; retain that limitation.
The M8 target envelope improves 11.47256→11.04126 ms, consistent with the
0.43309 ms C1 improvement. Target/draft graph kernel counts remain 606/142.
The small standalone estimate underpredicted the complete-model gain; use
0.43 ms as the calibrated software contribution.

The 1530 MHz result includes a hardware clock change. It is not the default
1290 MHz latency and is not a software-only gain. Clocks were restored to
1290 MHz after testing. No result establishes a 13.0 ms endpoint or sustained
700 GB/s for the full projections.

## Activation reuse and finer-grid rejection screens

Additional independent-extension screens retain the same production weight
layout, original eight logical K slices, FP16 arithmetic and FP32 reduction
order. M1/7/8 intermediate and final output bits match at amplitudes
0.01, 0.125, 1 and 4. Timings below are cold-L2 paired CUDA graphs, 200 samples,
GPU0 on the same full-NV2 machine, 1290 MHz and 300 W. They are research-only;
none is selected by serving dispatch.

| Candidate | Control us | Candidate us | Paired saving us, 95% interval |
| --- | ---: | ---: | ---: |
| N16, warp-coalesced activation, MLP | 68.500 | 73.712 | -5.212, -5.596–-4.848 |
| N32, warp-coalesced activation, MLP | 68.802 | 67.200 | 1.602, 1.234–1.996 |
| Whole-CTA 80 KiB input cache, MLP | 68.357 | 80.364 | -12.006, -12.273–-11.750 |
| 16 KiB chunked input cache, MLP | 68.388 | 74.220 | -5.832, -6.093–-5.581 |

The N16 plain-activation geometry also regresses in a separate screening run
(68.500→77.573 us). Finer geometry alone is rejected. The N32 coalesced gain
has an optimistic 56-layer estimate of only 0.090 ms, below the threshold for
a new complete-artifact serving test; do not count it in the endpoint baseline.
The full input cache retains the original 16 KiB partial scratch by reusing
storage after a CTA barrier, but permits only one resident CTA. The chunked
variant uses 32 KiB total shared memory and 55 registers, with two resident
CTAs, but incurs repeated staging barriers. Both variants fail the speed gate.

Direct-launch Nsight Compute 2025.1.1 counters compare bundled N32 control
with warp-coalesced activation. Cache control is `all`, clock control is
`none`, and counters are collected outside graphs:

| Counter | Control | Warp-coalesced |
| --- | ---: | ---: |
| DRAM read bytes | 25.15936 MB | 25.15987 MB |
| L1 global-load request sectors, 32 B | 1,975,226 | 1,275,578 |
| L2 read sectors, 32 B | 990,700 | 997,181 |
| Global load instructions | 348,160 | 261,120 |
| Registers/thread | 50 | 62 |
| Eligible warps/scheduler/cycle | 1.011 | 1.543 |
| Long-scoreboard stalled-warps/issued-instruction ratio | 7.805 | 5.586 |
| LG-throttle stalled-warps/issued-instruction ratio | 1.681 | 0.179 |

Activation coalescing reduces L1 request traffic by 35.4% and load
instructions by 25%, but does not reduce L2 or DRAM reads. Eight shuffle
instructions and increased register pressure offset much of that improvement.
The broad section capture reports 574.265→554.398 GB/s and 44.576→46.176 us;
a separate four-pass byte-counter capture reports 46.080→46.464 us. Keep
these instrumented results distinct from the paired graph saving. The
counter evidence does not establish a higher bandwidth roofline.

NCU reports two resident CTAs per SM and 0.85 capacity waves for the original
136-CTA, 512-thread kernel. Treating 136/80 as 1.7 waves ignores that resident
capacity and is not a sufficient diagnosis. The N16 experiment is the direct
geometry control and fails its full-kernel gate.

The same warp-coalescing method also fails on FP8 projections, preserving
output bits but regressing gdn_in 35.569→39.235 us, gdn_out 17.546→19.200 us,
full_qkv 31.964→37.555 us and down 37.929→38.252 us. Reject it before complete
layer or endpoint tests. These FP8 projection screens fix split=16/nacc=2;
they do not replace production shape tuning or establish a C4 result.
