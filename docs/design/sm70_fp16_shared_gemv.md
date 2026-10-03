# FP16 shared expert GEMV scheduling

The refreshed TP4 no-MTP graph has 99 cuBLAS `gemv2T` nodes. Of these,
48 compute shared expert gate/up and 48 compute shared expert down. Their
summed service times are 503.589 and 392.530 us per token. The other three
are one PLE value projection and final HC down/up. Shared and routed expert
work overlaps, so these service times are not additive endpoint savings.

## Operator change

Reuse `Sm70Fp16GemvSiluKernel` in the existing linear kernel directory. Short
rows process four output rows per CTA when there is sufficient independent
work to fill the device. Long rows keep the established K/warp schedule.
The predicate uses row length, work and SM count, not model names or TP size.
The flattened grid handles partial row tiles and prefix/suffix padding.

M is a template parameter. Any positive batch width follows the same compute
implementation; the former M<=16 capability restriction is removed. There
is no new batch-width fallback, environment variable or dense quantization.
Weights, inputs and outputs stay FP16. Products and reductions remain FP32,
and prefix SiLU retains its FP16 projection boundary and FP32 arithmetic.

## C1 operator measurements

The benchmark rotates 16 checkpoint layers to exceed L2 capacity. It measures
each of the four TP4 rank mappings on one V100 SXM2 32-GB GPU, using 256 nodes
per graph and five alternating repeats. CUDA is 12.8 and Torch is 2.10. FP16
reduced-precision reduction and accumulation are disabled in both arms.
These are operator results, not a four-GPU model run.

| Rank mapping | Projection | cuBLAS us | Candidate us |
|---:|---|---:|---:|
| 0 | Gate/up 320x2560 | 9.364 | 4.192 |
| 1 | Gate/up 320x2560 | 8.756 | 3.988 |
| 2 | Gate/up 320x2560 | 8.388 | 3.832 |
| 3 | Gate/up 320x2560 | 8.348 | 3.832 |
| 0 | Down 2560x160 | 4.280 | 2.984 |
| 1 | Down 2560x160 | 4.292 | 2.964 |
| 2 | Down 2560x160 | 4.104 | 2.808 |
| 3 | Down 2560x160 | 4.100 | 2.864 |

Median across rank mappings is 8.572 to 3.910 us for gate/up and 4.192 to
2.914 us for down. The original one-row generic kernel made down slower,
measuring 5.292–6.220 us versus cuBLAS 4.112–4.568 us. Four-row scheduling
resolves that local regression. It is not evidence of a full-model delta.

Independent FP64 references pass for real weights at M1/5/7/17/33 and three
input scales. Maximum absolute error is 0.001953125 across both arms. Eleven
shape cases pass eight changed-input captured replays, checking row ranges,
padding, SiLU boundaries and poisoned output, including widths above 16.
FP16 subnormal and invalid-layout/range checks also pass. These operator
checks are separate from the C1 teacher-forcing distribution and task gates.

One-pass Nsight Compute 2022.4 counters on real layer-zero weights provide
actual DRAM reads. These use isolated kernel replay with the profiler's cold
cache, rather than the rotating graph timing above. They must not replace
whole-model graph durations or fill unmeasured layer-wide byte columns.

| Projection and arm | Actual read bytes | NCU us | Read floor us at 750 GB/s | Remaining us |
|---|---:|---:|---:|---:|
| Gate/up cuBLAS | 1651200 | 8.000 | 2.202 | 5.798 |
| Gate/up candidate | 1646720 | 5.536 | 2.196 | 3.340 |
| Down cuBLAS | 824896 | 6.272 | 1.100 | 5.172 |
| Down candidate | 822208 | 4.768 | 1.096 | 3.672 |

Measured DRAM writes are 32 bytes for cuBLAS gate/up and zero for the other
three kernels. Output data can remain in L2 at kernel completion; zero DRAM
writes does not mean no output stores occurred. Candidate read bandwidth is
297.457 and 172.443 GB/s respectively. The remaining gap still argues for
reducing dependent stages rather than claiming these small projections have
reached the bandwidth target.

The measurements explicitly load candidate Python for a standalone operator
diagnostic. No installed model source is overlaid. Normal installed-wheel
validation omits `--kernel-source`:

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm70_fp16_shared_gemv.py \
  --model MODEL --out shared-gemv.json
```

## Integration status

This change extends the existing operator; it does not yet move the shared
expert linears from their cuBLAS loading/dispatch adapter. It does not claim
96 launches removed, a measured new per-layer budget or a model speedup.
Default routing needs capability-based selection and startup qualification
in the normal framework, then the installed source-complete artifact's C1
distribution/task gates and one short C4 end-to-end smoke. The 6-ms/token
target remains outstanding. The model's actual DRAM-byte and pure peer-wait
columns also remain unmeasured; logical weight sizes do not fill them.
