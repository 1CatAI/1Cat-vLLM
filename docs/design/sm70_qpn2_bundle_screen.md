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
