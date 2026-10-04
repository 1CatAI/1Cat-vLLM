# Batched FP32 HC on SM70 with partial NVLink connectivity

The existing TP4 batched HyperConnection kernel requires fully connected
NVLink IPC. On systems with only selected direct NVLink edges, the dense
fallback remains active even when the weights and token batch qualify for
the FP32 MMA implementation.

Two alternatives reuse the existing Volta MMA templates and preserve FP32
accumulation. Neither changes the FP16 materialization before SiLU or the
gated output mix.

## Local projections with admitted TP transport

Each rank computes its local HC rows. Two ordinary TP reductions transport
disjoint FP16 rows: one after the down projection and one after the up
projection. Each transported element has only one contributing rank, so
transport does not narrow a sum of live contributions.

This removes the full-connectivity requirement, but adds two collectives per
HC pair. The measured result does not justify selecting this alternative
for M=10.

## Replicated projections without transport

Each rank already retains the original complete HC matrices. Lossless batch
packing now also supports complete matrices, reusing the same down and up
MMA implementation. The down projection writes 20 FP32 partials, followed by
ordered FP32 reduction and the existing FP16/SiLU boundary. The up projection
then writes the complete gated output. No TP collective is needed.

The complete pack costs approximately one additional GiB per rank for the
96 HC pairs, compared with local packs. Preparation occurs before KV-cache
profiling. The route requires SM70, TP4 admission, FP16 dense weights, FP32
partial policy and M=2..16. The legacy fully connected IPC route retains its
existing arithmetic contract.

## Local-path operator screen

Source `6f093ea36e`, ordinary packaged runtime
`1.5.2.dev487+g6f093ea36.precompiled`; Torch 2.10.0 with CUDA 12.8,
driver 580.173.02, Python 3.12.3 and four V100 SXM2 32 GB GPUs. The TP4
topology has direct NVLink edges 0–1, 0–2, 1–3 and 2–3; it is not fully
connected.

The benchmark reads eight actual Flash-Next GSQ-RCO IQ3_S HC pairs: attention
and FFN pairs from layers 0, 12, 24 and 47. BF16 checkpoint weights undergo
the ordinary FP16 conversion. Activations are synthetic FP16. Both arms run
as CUDA graphs with FP32 accumulation. The working set exceeds L2. Maximum
rank median chain time is reported, rather than summing rank service times.

| M | Dense chain, eight pairs (µs) | Local chain, eight pairs (µs) | Estimated saving for 96 pairs (ms) |
| --- | ---: | ---: | ---: |
| 5 | 250.593 | 239.022 | 0.139 |
| 10 | 247.480 | 360.704 | -1.359 |

Outputs match the dense reference exactly for all eight pairs, both M values
and every rank. Separate M=2/5/10/16 tests check FP32 reference arithmetic,
disjoint transport rows and graph replay. These results are operator-chain
measurements; they do not establish a model-round speedup.

Reproduce with `benchmarks/kernels/benchmark_sm70_hc_gguf_tp4.py MODEL.gguf`
under a four-rank launcher. Add `--replicated` to screen the complete-pack
alternative.

## Replicated-path operator screen

Source `58124264ec`, ordinary packaged runtime
`1.5.2.dev488+g58124264e.precompiled`, with the same hardware, weights,
activation generation and timing method as the local screen. The installed
extension advertises both local and replicated operators. No private kernel
DSO or preload is used.

| M | Dense chain, eight pairs (µs) | Replicated chain, eight pairs (µs) | Estimated saving for 96 pairs (ms) |
| --- | ---: | ---: | ---: |
| 5 | 254.874 | 200.745 | 0.650 |
| 10 | 253.409 | 204.186 | 0.591 |

Outputs again match the dense reference exactly on all four ranks. The
M=2/5/10/16 GPU tests also exercise the replicated FP32 reference and graph
replay. The complete-pack alternative improves both measured verifier
batch sizes, so it qualifies for the combined dense-route C1/C4 model check.
That model comparison remains pending; the table is not a model-round gain.
