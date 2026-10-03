# SM70 expert dataflow research

## Scope and admission

This is a research benchmark, with no production dispatch or default enablement.
It tests W13, SiLU, W2 and weighted reduction before adding router, shared-expert
work or TP communication. All selected experts are included. Activations and
dense weights stay FP16, matrix accumulation stays FP32, and existing NVFP4
expert storage is decoded through the native FP16 weight boundary.

Kernel templates cover M=1, M=5 and dynamic M with the same algorithm. There is
no TP-size or model-name admission condition. Layout requires positive H divisible
by 32 and I divisible by 16; dtype, contiguity, device, workspace, shared-memory
capacity and cooperative residency are checked. FP16 and NVFP4 readers share
the schedule. FP16 prefetches a bounded W2 prefix; NVFP4 prefetches the full W2
output tile. GGUF/IQ3 readers are not implemented here.

## Scheduling and generations

Producer CTAs compute a 16-element intermediate slice, retain the existing
FP16 projection/SiLU/product boundaries, and publish that small vector followed
by a per-slice generation flag. Consumer CTAs prefetch W2 weights before waiting
for activation slices, accumulate in FP32, retain the per-expert FP16 output
boundary, and compute the complete weighted sum. They do not write the large
FP32 per-slice output matrix used in the rejected first prototype.

The kernel uses cooperative launch to guarantee resident workers, without
`grid.sync`. Each CTA owns producer or consumer tasks statically. Residency is
the minimum occupancy of the M=1, M=5 and dynamic variants for the same weight
layout; the grid remains constant when M changes. Empty workers still retire
their generations. This permits one initialized workspace to serve changing
batch widths on a serialized stream. Layout changes require separate workspace.
The workspace must be zero-initialized once; generations subsequently advance
on the GPU across graph replays.

Writers fence activation stores before release publication. Consumers use a
volatile readiness poll with bounded sleep, then acquire publication before
dependent loads. This preserves the memory contract while testing whether busy
polling delays producers. There is no floating-point atomic reduction.

## Correctness evidence

FP16 and NVFP4 readers passed independent CPU FP64 oracles at H/I shapes
32/16, 96/48, 64/32 and 2560/160, with M=1, 5 and 7. Six changing/poisoned
input graph replays per case cover duplicate and invalid expert IDs. Maximum
absolute oracle differences at the largest shape were 0.000244 for FP16 and
0.001953 for NVFP4; FP32 reassociation and FP16 materialization account for
small differences. These operator tolerances do not replace the C1 distribution
and model-quality gates.

Both readers also passed eight graph replays with widths 1, 5, 7, 5, 1, 7, 1, 5
sharing one control/scratch workspace, changing inputs and route IDs. Inactive
output rows remain poisoned and the generation advances once per replay.
Compilation reports no local-memory spill stores or loads.

## Measured C1 screens

The checkpoint screen rotates real TP0 shards from layers 0, 15, 31 and 47,
using ten selected experts, synthetic FP16 activations and complete ten-route
weighting. The existing installed native W13/SiLU and W2/weighted-reduce pair
is the control. Seven graph-timing samples alternate arms. This is an operator
screen, not a model speed or quality result.

| Prototype | Native pair median us | Candidate median us | Decision |
| --- | ---: | ---: | --- |
| Global queue, per-slice FP32 output matrix | 19.100 | 41.592 | Reject |
| Static slices, per-tile flags, full weight tiles | 18.828 | 43.770 | Reject |
| W2 prefetch followed by small activation-vector wait | 18.684 | 31.140 | Reject |
| Same pipeline with bounded polling backoff | 19.505 | 32.656 | Reject |

Do not rerun unchanged variants or spend a model startup qualifying a slower
operator. The latest kernel is retained for correctness and dependency research;
it has no demonstrated C1 gain. Broader fusion requires a different work schedule.

Actual cold-cache NCU counters for the first prototype:

| Kernel | Read MB | Write MB | NCU us | Registers/thread | Shared KiB/block |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native W13/SiLU | 5.137184 | 0.019488 | 15.264 | 60 | 2.000 |
| Native W2/reduce | 2.577024 | 0.000128 | 12.256 | 43 | 0.625 |
| First fused prototype | 7.716672 | 1.627360 | 54.720 | 121 | 77.047 |

The first prototype reads almost the same bytes, but adds output-partial traffic
and limits residency to one CTA per SM. Its read floor at 750 GB/s is 10.289 us;
the remaining 44.431 us cannot be explained by the extra writes alone. NCU times
use cold-cache replay and are separate from graph-timing samples. The later
pipeline compiles with 71 NVFP4 registers and 78 FP16 registers, but remains
slower; occupancy improvements alone did not qualify it.

Reproduce the synthetic oracle and checkpoint screen with the companion
benchmark scripts, `--build` pointing to an owned extension cache, and `--model`
pointing to the checkpoint. The scripts acquire the shared GPU lock. The
checkpoint benchmark explicitly uses the retained TP4 workload; that is not an
operator admission restriction. New work optimizes C1 only; correctness includes
other M values, and merge needs one short C4 execution smoke.
