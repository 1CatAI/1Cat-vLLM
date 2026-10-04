# FP16 SiLU/multiply and down fusion on SM70

The shared-expert activation and down projection currently launch separately.
The candidate consumes the FP16 gate/up projection and emits the FP16 down
result in one kernel. SiLU is rounded to FP16 before multiplication; that
product is also rounded to FP16 before the FP32 dot product. FP32 accumulation
uses eight lanes with two values per lane. Accumulation order may differ from
the vendor projection; admission uses the existing
[distribution and task-quality contract](sm70_qwen38_distribution_acceptance.md).

The provider uses the mixed-precision kernel registry with an explicit unpacked
FP16 configuration. A SiLU/multiply consumer declares its input semantics;
admission checks actual tensors, precision, layout and resources. There is no
model, TP or batch-size whitelist. All positive batch widths are implemented,
including M=5. Each loaded matrix measures its graph capture widths with a
16-MiB read/write L2 flush in both arms. The candidate must pass its numerical
self-check and have an upper quartile below the vendor lower quartile.
Unmeasured or rejected widths keep the original visible activation/projection
path. The unified `fused_fp16_silu_down` kernel policy defaults to enabled;
quantized, biased and unsupported providers retain their existing operations.

## Operator evidence

Research control: ordinary installed dev340+g77a31212a, Torch 2.10/cu128,
CUDA 12.8, V100 SXM2 32 GB. The benchmark rotates 48 real TP4 rank-zero down
matrices, 39,321,600 bytes in total, through CUDA graphs. Gate/up outputs use
retained real model inputs. Nine alternating event measurements compare native
SiLU/multiply plus vendor down against the fused operation. Both FP16 reduction
settings remain disabled. These are isolated operator measurements, not TPOT.

| Implementation | C1 grid for down/fusion | Kernels/operator | Median us |
| --- | ---: | ---: | ---: |
| Native activation + vendor down | 1 + 320 | 2 | 7.7733 |
| Fused, four rows/CTA | 640 | 1 | 7.9200 |
| Fused, eight rows/CTA | 320 | 1 | 7.1600 |
| Fused, sixteen rows/CTA | 160 | 1 | 6.5920 |

The sixteen-row candidate reduces isolated time by 15.2%. Each operator reads
819,200 weight bytes: its weight-only floor is 1.0923 us at 750 GB/s, leaving
5.4997 us of measured excess. No actual DRAM-bandwidth claim is made here.
M=1/5/17/33 and non-aligned K=17/N=97 passed materialized-FP16 FP64-oracle
checks. Across 48 real cases the retained sixteen-row candidate differed from
vendor at 35 FP16 outputs, with maximum error 3.0518e-5. This is an operator
diagnostic, not proof of model distribution or quality equivalence.

The committed operator benchmark uses the installed provider:

```bash
python benchmarks/kernels/benchmark_sm70_fp16_silu_down.py \
  --model MODEL --out RESULT.json --tensor-parallel-size 4 --rank 0 \
  --inputs SNAPSHOT_DIR
```

Omit `--inputs` to use deterministic inputs. It rotates checkpoint matrices;
if the rotating down weights fit within 16 MiB, both graph arms instead flush
16 MiB before every operation and explicitly report that flush-inclusive time.

A whole-tile prefetch variant was also measured with the same cold protocol.
Its best result was 6.6427 us, so it does not replace the 6.5920-us version.
Standalone router/shared-gate fusion and two-stage parallel router top-k also
remain outside default admission: consumer multiplication erased most router
fusion savings, while parallel top-k added service time.

## Qualification status

Eight CPU admission/hash tests and scoped static checks pass. The normal source
artifact contains 16 newly built native modules; packaged Python matches the
qualified source and loader paths exclude build directories. Ten installed
GPU tests pass, including replay with changed inputs and all 63,488 finite
FP16 gate values against native activation. The installed wheel SHA256 is
`bd774a8c6b25bf8da90efdd9f5f426954ddd5f83cd8a878826bea2b935118be8`.

Matched C1 distribution and natural task-quality gates, endpoint per-step
timing, a short C4 health check and the graph-node trace remain pending.
No model speedup or phase-target claim is accepted yet. In-projection,
out-projection, LM head, MTP, dense quantization and communication algorithms
are outside this change.
