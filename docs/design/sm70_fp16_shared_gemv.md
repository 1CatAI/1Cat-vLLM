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

The benchmark rotates all 48 checkpoint layers before reusing weights. It
measures each of the four TP4 rank mappings on one V100 SXM2 32-GB GPU,
using 768 nodes per plain projection graph and five alternating repeats.
The fused baseline includes cuBLAS gate/up and native SiLU-and-multiply.
CUDA is 12.8 and Torch is 2.10. FP16 reduced-precision reduction and
accumulation are disabled in both arms. These are operator results, not a
four-GPU model run.

| Projection | Median across four rank mappings, baseline us | Candidate us |
|---|---:|---:|
| Gate/up 320x2560 | 8.320 | 4.131 |
| Gate/up + SiLU-and-multiply | 10.713 | 5.123 |
| Down 2560x160 | 4.646 | 3.241 |

The earlier 16-layer rotating result was gate/up 8.572 to 3.910 us and down
4.192 to 2.914 us. The 48-layer result above replaces it for the revised
cold-L2 criterion. The original one-row generic kernel made down slower;
four-row scheduling resolves that local regression. Neither result proves
an endpoint delta.

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

## Framework integration and pending acceptance

The unpacked FP16 adapter uses the existing `MPLinearKernel` registry and
post-load lifecycle. It admits SM70 FP16 input/weight matrices with the
existing FP32 accumulation policy, without model or TP predicates. Methods
with their own `apply` retain their established provider. Startup checks
compare the candidate with the vendor path and measure captured operations
with a 16-MiB L2 flush before each operation. Flush cost is included equally
in both arms and is not reported as projection latency. The existing startup
kernel report exposes accepted graph widths, numerical failures and timings.

Plain projection and fused gate/up use functional opaque operations with
fake implementations for compilation. M is a positive template parameter;
widths without a successful measurement use the vendor route. The decision
is per loaded matrix and measured width, rather than a batch-size cutoff.
Deterministic startup inputs do not consume model or sampler RNG state.

Gate/up accumulates both projections in FP32, materializes each as FP16,
computes SiLU in FP32, materializes it as FP16, then multiplies by the FP16 up
projection. It replaces gate/up plus activation with one kernel. Down uses
one projection kernel. No dense quantization or new environment variable is
introduced. The existing disabled-kernel control can select the A/B baseline.

Installed source-complete wheel route selection, C1 teacher-forcing and task
gates, full-model timing, and one short C4 smoke remain required before merge.
The operator result is not a claim of 96 launches removed or an endpoint
speedup. First-phase 7.5-ms/token and second-phase 6-ms/token targets remain
outstanding.

## Installed C1 endpoint and quality result

The ordinary source-complete artifact at 735099f240 includes the mapped PLE
transport qualified by #831. Both arms use that same artifact; the control
selects `Sm70Fp16LinearKernel` through the existing disabled-kernel control.
The kernel/adapter uses FP16 input, weights and output with FP32 accumulation;
the final capability guard explicitly rejects FP32 output configurations.

| C1 result, six samples | Disabled | Enabled |
|---|---:|---:|
| Median ms/token | 11.11511 | 10.98625 |
| Range ms/token | 11.02277–11.15845 | 10.77369–11.04027 |
| MBPP | 12/12 | 12/12 |
| GSM8K | 12/12 | 12/12 |
| Chinese QA | 8/8 | 8/8 |
| Needle retrieval | 4/4 | 4/4 |
| Natural EOS and nonempty final answers | 36/36 | 36/36 |
| Replacement characters and token-cap failures | 0 | 0 |

The median difference is 0.12886 ms/token (1.16% lower TPOT). Sample ranges
overlap. This is much smaller than the isolated projection gain; shared and
routed work overlaps, so service-time savings cannot be added to the endpoint.
A final graph-node trace is queued to verify actual kernel selection/counts
and attribute this small difference. Do not infer all layer routes from the
last admission record in a grouped startup report.

The installed GPU loader smoke selected M1 plain/fused gate-up and plain down,
reported their numerical/cold admission through the existing framework, and
passed changed-input graph replay against independent FP64 references.
C1 teacher-forcing distribution and the short C4 smoke remain pending.
