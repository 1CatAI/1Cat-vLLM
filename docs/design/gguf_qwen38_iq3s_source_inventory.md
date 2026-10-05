# Qwen3.8-27B IQ3_S source inventory and activation reuse

This CPU-only inventory reads the actual `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`:
64 layers, 851 tensors, 11,760,551,936 tensor-payload bytes. The file contains
11,771,546,784 bytes; headers and alignment account for 10,994,848 bytes.
Percentages below use tensor-payload bytes, not file size or FP16-expanded
storage. No weights are decoded and no CUDA runtime is used.

## Source formats and positions

| GGUF type | Tensors | Source MiB | Payload % | Largest positions (tensor counts) |
| --- | ---: | ---: | ---: | --- |
| IQ3_S | 144 | 3474.02 | 30.975 | ffn_up (25), ffn_down (22), ffn_gate (15), attn_qkv (22) |
| IQ4_XS | 96 | 2882.03 | 25.696 | ffn_down (21), ffn_gate (15), ffn_up (10), ssm_out (16) |
| IQ3_XXS | 78 | 1979.14 | 17.646 | ffn_gate (21), ffn_up (18), attn_qkv (13), ffn_down (7) |
| Q4_K | 39 | 1508.91 | 13.453 | output (1), ffn_down (6), ffn_up (3), ssm_out (6) |
| IQ2_S | 17 | 763.14 | 6.804 | token_embd (1), ffn_up (5), ffn_down (4), ffn_gate (4) |
| Q2_K | 13 | 229.69 | 2.048 | attn_q (6), attn_gate (4), ffn_down (1), ffn_gate (1) |
| IQ2_XS | 9 | 211.02 | 1.881 | ffn_gate (4), ffn_down (3), ffn_up (1), attn_qkv (1) |
| IQ2_XXS | 5 | 94.10 | 0.839 | ffn_up (2), ffn_gate (1), attn_q (1), attn_qkv (1) |
| BF16 | 96 | 45.00 | 0.401 | ssm_alpha (48), ssm_beta (48) |
| IQ1_M | 1 | 18.59 | 0.166 | ffn_gate (1) |
| F32 | 353 | 10.09 | 0.090 | ssm_conv1d (48), attn_norm (64), post_attention_norm (64), ssm_norm (48) |

The lattice family occupies **58.311%**, LUT4 **25.696%**, affine formats
**15.501%**, and floating tensors **0.491%**. This file has no ternary, MXFP4,
NVFP4, or Q2_0 tensors. IQ3_S alone occupies 30.975%; the filename does not
mean that every weight uses that format. The three FFN projections account
for **62.903%** of the payload.

Complete per-type/per-position/per-layer records are emitted in
`by_type_role.csv`, `by_layer_type.csv` and `tensors.csv`, including logical N/K shapes, original
source bytes, and GGUF data offsets. The position table above shows only the
four largest byte contributors for each type.

## Mixed gate/up coverage order

All 64 gate/up pairs have logical N17408/K5120, or N4352/K5120 on each TP4
rank. Their combined original payload is 4,800,430,080 bytes. Forty pairs
use different gate/up types: **2,989,998,080 bytes**, or **62.286%** of all
pair bytes. The eight IQ3_S/IQ3_S pairs occur in layers
`6, 23, 24, 25, 46, 51, 55, 56`; their 612,761,600 bytes cover only
12.765% of gate/up payload.

The following priority is by real source-byte coverage. Format sets combine
both orientations for planning; `mixed_pair_priority.csv` and
`mixed_type_set_priority.csv` preserve the exact gate/up orientation and
layer lists. This is a coverage ranking, not an unmeasured speed forecast.

| Priority | Format set | Layers | Mixed bytes % | Cumulative layers | Cumulative mixed bytes % |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | IQ3_S + IQ4_XS | 11 | 31.509 | 11 | 31.509 |
| 2 | IQ3_S + IQ3_XXS | 7 | 16.954 | 18 | 48.463 |
| 3 | IQ4_XS + Q4_K | 3 | 9.781 | 21 | 58.244 |
| 4 | IQ3_XXS + IQ4_XS | 3 | 8.174 | 24 | 66.418 |
| 5 | IQ2_S + IQ3_S | 3 | 6.707 | 27 | 73.125 |
| 6 | IQ2_S + IQ3_XXS | 3 | 6.288 | 30 | 79.413 |
| 7 | IQ3_S + Q4_K | 2 | 5.915 | 32 | 85.328 |
| 8 | IQ2_XS + IQ3_XXS | 2 | 4.006 | 34 | 89.334 |
| 9 | IQ2_XS + IQ2_XXS | 2 | 3.260 | 36 | 92.594 |
| 10 | IQ3_S + Q2_K | 1 | 2.259 | 37 | 94.853 |
| 11 | IQ2_S + IQ2_XS | 1 | 1.816 | 38 | 96.670 |
| 12 | IQ2_S + IQ2_XXS | 1 | 1.723 | 39 | 98.393 |
| 13 | IQ1_M + IQ2_S | 1 | 1.607 | 40 | 100.000 |

The first two format sets cover 18/40 mixed layers and **48.463%** of mixed
pair bytes. The first six cover 30/40 layers and **79.413%**. IQ4_XS/IQ3_S
has nine layers and its reverse has two; both orientations need the correct
individual reader and the same gate/up output order.

Generate the reusable JSON and CSV records with:

```bash
python benchmarks/benchmark_gguf_quantization_inventory.py MODEL.gguf \
  --output-dir inventory --tensor-parallel-size 4
```

The script reconciles tensor/type/role byte totals, pair shape and TP row
alignment, and ordered-pair coverage. `inventory.json` is the complete record;
CSV tables include source-type totals, families, roles, type-role positions,
all 64 layers with each present type’s byte share and projection roles,
every tensor, every gate/up pair, and both priority views. TP source bytes
for FFN gate/up are exact output-row shards; other tensor types are not
assumed to divide uniformly across ranks.

## Existing activation-supply interfaces

Source inspection is pinned to main
`0e359c87d315931b89b7d5a774927f212d57f3b1`. No operator is added here.

| Existing interface | Source | Reuse boundary |
| --- | --- | --- |
| `MainloopSm70` A staging | `kernels/gemm/mainloop_sm70.h` | `GmemIterA::Fetch/Store` fills `storage.A`; `SmemCopyA` loads MMA fragments consumed across the CTA N tiles |
| `Operand_A_BatchPadded<half>` | `kernels/gemm/arch/operand_sm70_s884.h` | A-only policy keeps FP16 values, 16-byte global loads and a K+8 shared row stride; M48 has surplus-warp predicates |
| `Config_QuantizedBatch<Weight,Order,Transform>` | `kernels/gemm/arch/config_sm70_s884.h` | Uses the padded A policy with existing FP4/FP8 B transforms; its V type is uint16 and cannot be assumed correct for all GGUF formats |
| `DenseBatchSupplyKernelImpl` | `kernels/gemm/batch_kernel_sm70.h` | Already admitted for dense M33..64, N>=2048, K>=1536; full/tail tile checks remain necessary |
| `Operand_A_Swizzle_8x64<half>` | `kernels/gemm/arch/operand_sm70_s884.h` | Existing exact M8/K64 shared A swizzle; current registrations have specific M/N/K gates |
| `qpn_pair_m16_kernel<Reader,Split,Gated>` | `ops/qpn_pair_sm70.cuh` | Loads each A fragment once into registers and feeds both projections; shared storage holds FP32 partials, not activations |
| `nvfp4_qpn2_m32_twophase_sm70_kernel` | `ops/nvfp4_qpn2_sm70.cu` | One decoded weight serves four row tiles; A is read from packed global input and shared stores reductions |

Paths in the table are relative to
`csrc/sm70_turbomind/lmdeploy/src/turbomind` for `kernels/`, and to
`csrc/sm70_turbomind` for `ops/`.

The actual CTA shared-activation implementation is TurboMind's
`storage.A -> SmemCopyA -> TransformA` path. The existing QPN pair and M32
kernels provide register reuse of A or decoded B; their shared reductions
must not be described as shared activation staging. `pack_k16_input` packs
`[M,K]` to `[K/16,M,16]` in a separate global buffer.

For GGUF, reuse the A operand policy, thread map, `MainloopSm70`, and
scheduler while retaining each GGUF B reader, V layout and scale transform.
`Config_GgufLattice` currently has format-dependent V widths; substituting
`Config_QuantizedBatch` wholesale would lose that contract. The QPN pair
reader contract is a constructor plus `load(group, half2*)`, with the
FP4-specific epilogue rounding tag. Its two readers currently share one
Reader type and one codes/scales layout. Mixed formats require preserving
separate typed readers and their FP16 intermediate rounding; this document
proposes no new operator or dispatch.

The qkvz prototypes remain research-only. The canonical path remains selected;
this CPU inventory does not connect them to the model or run GPU tests.
