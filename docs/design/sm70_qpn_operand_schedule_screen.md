# SM70 QPN operand and collective scheduling screens

These are independent research extensions. No serving route or precision is
changed, and their timings are not end-to-end model results. The controls are
extracted from the owned source and checked against the installed operators.

## Measurement contract

V100-SXM2-32GB on a fully NV2-connected TP4 group, 300 W, 1290 MHz SM and
877 MHz memory; Torch 2.10/cu128, CUDA toolkit 12.8. Original unsloth target
weights, rank-local layer 0, M8 FP16 activations and FP32 residual/state.
CUDA Graph measurements evict 128 MiB before external timing events and retain
200 paired samples after warmup. All five GPU file locks are acquired before
the process check. Compiler caches and extensions are isolated from serving.

## Effective-scale rejection

The FP4 decoder retains its two FP16 multiplication boundaries. A 512-byte
lookup stores the exact effective scale for each E4M3 code; a second candidate
stores effective FP16 scales beside the unchanged codes. Both match the
installed MLP's output bits at amplitudes 0.01, 0.1, 1 and 4.

| Complete MLP | Control us | Candidate us | Saving us, 95% interval |
| --- | ---: | ---: | ---: |
| Effective-scale lookup | 68.500 | 72.986 | -4.485, -4.777 to -4.234 |
| Inline FP16 scales | 68.833 | 71.449 | -2.616, -2.939 to -2.278 |

Lookup adds a dependent load. Inline FP16 scales enlarge each bundled group
from 288 to 320 bytes. Neither clears the full-kernel speed gate. Keep the
existing effective-scale arithmetic and bundled layout.

## Paired gate/up warps

Each warp computes both projections, sharing the activation registers while
preserving the original eight K slices, per-projection FP32 accumulation order,
ordered partial reduction, FP16 gate/up rounding, SiLU and FP16 multiply.
Sequential and interleaved schedules use 256 threads rather than 512, 80
registers per thread and 16 KiB shared memory, without spills or extra weights.

| Complete MLP | Control us | Candidate us | Saving us, 95% interval |
| --- | ---: | ---: | ---: |
| Sequential projections | 68.337 | 65.971 | 2.365, 2.207 to 2.524 |
| Interleaved HMMA | 68.705 | 66.058 | 2.647, 2.309 to 2.970 |

The sequential version also passes a complete real-weight GDN layer graph.
Outputs, FP32 residual, rollback states and convolution history match bitwise
on all four ranks at amplitudes 0.01, 0.125, 1 and 4. Each graph retains 11
kernel nodes, including the eviction outside the timed range. The timed layer
contains ten compute nodes in both arms.

Taking the maximum rank event duration for each paired sample gives
159.134 to 157.696 us, saving 1.438 us with a 95% bootstrap interval of
0.997 to 1.884 us. This is smaller than the isolated MLP saving. Extrapolating
48 GDN layers gives about 0.069 ms; the other eight NVFP4 MLPs still require
attention-layer admission. Do not substitute the optimistic isolated estimate
for a measured full round or promote this alone through a serving rebuild.

## Collective rejection

A direct-pull candidate retains the existing forty-CTA geometry, five CUB128
partials per row and ordered norm reduction. System-visible start/end peer
handshakes protect input visibility and reuse. It reads independent CUDA IPC
inputs directly, without casting the installed custom-allreduce class across
an extension ABI boundary. A second candidate retains push but uses the local
input in registers, omitting its local payload store/load/reset.

Both match output and residual bits for four amplitudes. Single cold graph
calls include rank launch skew, so a second screen uses 64 consecutive
collectives per graph. Maximum-rank means per call are:

| Route | Time us |
| --- | ---: |
| Original push + norm | 8.110 |
| Direct pull + norm | 20.540 |
| Push with local operand retained | 8.598 |

Both candidates fail. The local-operand variant regresses by 0.488 us with
95% interval 0.448 to 0.529 us. The direct-pull protocol does not remove the
latency cost of its visibility and completion handshakes.

## GDN normalization-cache rejection

A separate candidate lets convolution own a complete K128 head and prepare
normalized Q/K in FP32 for the unchanged BV2 recurrence. It needs no producer
flags or grid synchronization. Complete-layer timing regresses by about
1.8 us; the extra cache and preprocessing offset the removed repeated work.
Convolution-history bits match, while state and output bits can differ because
the normalization reduction layout changes. This fails speed admission, so no
teacher-forcing or serving rollout is justified for this candidate.

## FP8 two-phase rejection

Another screen halves physical warp count while retaining the original logical
K slices, two FP32 accumulator chains and ordered partial reduction. Each warp
finishes two slices sequentially; there are no extra splits or weight padding.
Both projections match the installed operator bitwise at four amplitudes.
Cold-L2 qkvz regresses 35.635 to 44.165 us and GDN out regresses 17.224 to
20.849 us. The register/shared-resource reduction does not compensate for the
longer per-warp serial work. Reject before adding fused a/b or model tests.

## Reproduction and evidence

Benchmarks are under `benchmarks/kernels/`:

- `benchmark_sm70_qpn2_effective_scale.py`
- `benchmark_sm70_qpn2_paired_gate.py`
- `benchmark_sm70_qpn2_paired_layer.py`
- `benchmark_sm70_tp4_pull_norm.py`
- `benchmark_sm70_qpn8_two_phase.py`

The layer benchmark loads the extension generated by the paired-gate benchmark.
`--mode 1` selects sequential paired warps; `--mode 0 --prepared-qk` isolates
the rejected GDN normalization cache. The collective benchmark accepts
`--burst 64`. Extensions are built with `TORCH_CUDA_ARCH_LIST=7.0`, `-O3` and
`-lineinfo`; no task-cache DSO is installed into a serving runtime.

Retained JSON samples, compiler logs, source/extension hashes, commands and
GPU snapshots are in the campaign's `qwen38-qpn2-effective-scale-20261008`
artifact directory. Both positive microbenchmarks and rejected candidates
remain outside production dispatch pending complete-artifact admission.
