# HC-to-input-projection readiness screen on SM70

The hypothesis was to start the next GGUF dense projection when a group of
four HC CTAs had materialized their output columns, rather than waiting for
full HC completion. Twenty K-slice partials preserve canonical weight storage
and FP32 accumulation. A subsequent deterministic reduction produces FP16
projection outputs. No production dispatch changes are enabled.

The implementation reuses the existing GGUF reconstruction functions. It
includes Q4_K, Q5_K, Q6_K and IQ4_XS input segments, with no weight or activation
precision change. The execution idea is similar to the chunk readiness in
[mKernel](https://github.com/uccl-project/mKernel) and
[FLUX](https://github.com/bytedance/flux/blob/main/docs/design.md); their
Hopper/Ampere implementations are not used as SM70 kernels.

## Isolated numerical and chain gate

Four V100-SXM2-32GB GPUs, actual mixed NV1/NV2 topology, TP4, CUDA 12.8,
Torch 2.10.0+cu128. M5, eight actual HC weight pairs and eight actual dense
input groups, sixteen repetitions in a CUDA graph, same-process ABBA.
Rank0 projection shards are mirrored across cards in this isolated test;
HC uses each card's actual TP4 partition. This is not a model measurement.

Attention has two KV heads replicated under TP4. The original extracted
K/V quarters were 128 rows, so the screen reads full 256-row KV heads from
the GGUF before packing. The first unsupported-format input check failed
before numerical checks or timing; it is not a performance result.

HC residual, block and injection outputs match the packaged control exactly,
including two changed-input graph checks. Projection reconstruction is checked
against canonical FP16 weights with FP32 matmul and official FP32 GGUF
dequantization. The maximum relative numerical errors are recorded by tensor
and rank in the benchmark JSON. Worst relative maximum errors are 3.97e-4
against canonical FP32 matmul, 3.50e-4 against the packaged projection and
8.18e-4 against official FP32 dequantization. The packaged projection also
has 3.97e-4 error against canonical FP32 matmul.

| Chain | Maximum rank median us |
| --- | ---: |
| Packaged HC plus packaged dense | 49.380 |
| Copied HC control plus packaged dense | 48.936 |
| HC with chunk-ready dense partials and reduction | 58.224 |

The candidate loses 8.84 us per chain. The copied control rules out a large
baseline-copy difference. The consumer kernel has a 112-byte local stack
frame and an extra split-K partial/reduction chain; these are possible causes,
not independently measured attributions. The numerical gate passes, but the
chain gate fails. It is rejected without a model restart or an endpoint claim.
M1/M8 and C4 promotion checks are unnecessary for this rejected version.

Both timing arms share one canonical set of dense codes and coefficients.
All partials, counters and ready flags have fixed addresses before graph
capture. The research DSO is built separately for this screen and is not part
of the production wheel; its results cannot qualify a production route.

## Reproduce

Build with `benchmarks/kernels/build_sm70_hcx_consumer_pipeline.py --cache-dir`
and an explicit directory. Run
`benchmarks/kernels/benchmark_sm70_hcx_consumer_pipeline.py` under torchrun
with four workers and all GPU ownership locks. Supply `--hc-weights`,
`--projection-shards`, both GGUF paths through `--gguf`, the built `--library`
and `--output`. The JSON records the native core and candidate DSO hashes,
per-rank samples and numerical errors. Model weights and generated binaries
are not included in the repository.
