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

The same-artifact C1 focus compared 32 valid-vocabulary positions. Disabled
admissions were zero on all four ranks; enabled M=1 admissions were
36/38/35/34 matrices. Top-1 agreement was 100%, but the initial distribution
gate failed:

| Stratum | Mean KL | p99 KL | Max logit error |
| --- | ---: | ---: | ---: |
| All | 0.0011553 | 0.0135292 | 0.8359375 |
| English | 0.0022305 | 0.0153994 | 0.8359375 |
| Chinese | 0.0000800 | 0.0010405 | 0.5693359 |

Serial and pair-first product folding, and alternate lane reduction trees,
were checked against retained real inputs without finding a better match.
Further layout trials are stopped. The complete natural quality comparison
also rejects this candidate; distribution thresholds remain unchanged.

The matched installed-artifact endpoint uses TP4, 262144 startup capacity,
8192 input / 513 output tokens, FP16 dense/activations/KV, FP32 state, CUDA
graphs, disk-mapped ngrams and no MTP. Six timing samples follow one warmup.
GPU 0–3 are V100-SXM2-32GBs with full NVLink connectivity, a measured 300-W
power limit and 877-MHz memory clocks; no GPU settings were changed.

| Arm | C1 median ms/token | Sample range ms/token | M=1 admitted matrices/rank | Task scores | Natural EOS |
| --- | ---: | ---: | --- | ---: | ---: |
| Disabled via existing kernel registry | 11.04115 | 10.92117–11.17972 | 0/0/0/0 | 36/36 | 36/36 |
| Fusion enabled | 11.06298 | 10.86482–11.24090 | 39/32/29/35 | 36/36 | 35/36 |

The candidate median is 0.20% higher, with overlapping sample ranges. This
does not establish an endpoint speedup despite the isolated 15.2% reduction.
With identical seeded natural sampling, MBPP-16 stops at 628 tokens in the
control, but reaches the 4096-token cap in the candidate and repeats a long
final-answer line eleven times. Passing the code assertions does not override
that output-health regression. Both arms retain all four needle contexts,
including 258K. The 768-position investigation, C4 smoke and graph-node trace
were skipped after the C1 quality failure; no default admission is claimed.

The quality runner now records output-health failures independently of task
scores and returns a nonzero status when health fails. The failed fusion is
retained as research evidence and is not merged. In-projection,
out-projection, LM head, MTP, dense quantization and communication algorithms
are outside this change.
