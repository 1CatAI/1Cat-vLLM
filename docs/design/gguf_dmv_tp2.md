# Shared GGUF projection planes at TP2 and TP4

The M8 projection-plane implementation now accepts the measured 27B TP2
shards as well as TP4 shards. Gate/up uses N8704 per matrix at TP2, down uses
K8704, GDN output uses K3072, and attention inputs use widths 6144/512/512.
These dimensions remain aligned to the GGUF blocks. No new decoder or
quantization precision is introduced.

GDN's three value heads per key head require eight local key-head groups at
TP2 and four at TP4. The shared activation reader selects the corresponding
head permutation. An explicit native capability query prevents older wheels
from selecting the new GDN layout. Outside M8, the same resident plane bank
restores canonical coefficients and head order. Restoration scratch grows
during loading to fit the largest coalesced TP2 gate/up, before graph capture.

The existing kernel configuration controls admission. Three-format attention
inputs and the measured TP2 IQ2/IQ3 qkvz combinations use the same kernel
family; combinations requiring different scale precision retain fallback.
Unsupported source formats, including Q2_K and IQ1_M plane banks, keep their
existing fallback. This does not claim that every projection format has been
ported to plane storage.

## Operator measurements

Real rank-0 TP2 weights from Qwen3.8-27B-GSQ-RCO-IQ3_S are compared with
canonical projections under cold rotation and graph replay. Compatible
canonical source shards are coalesced exactly as in the loader; measuring
each original q/k/v shard separately would inflate the control cost.

Representative measurements, in microseconds:

| Projection | Sources | Plane kernel | Coalesced canonical |
| --- | --- | ---: | ---: |
| down | IQ3_S | 33.14 | 54.77 |
| GDN output, raw head order | IQ3_S | 14.21 | 25.54 |
| gate/up with SiLU | IQ3_XXS / IQ3_XXS | 60.10 | 93.88 |
| gate/up with SiLU | IQ3_XXS / IQ3_S | 65.56 | 111.80 |
| qkvz | IQ3_S / IQ3_XXS | 35.13 | 65.37 |
| qkvz | IQ4_XS / IQ4_XS | 35.94 | 41.82 |

The qkvz operator fixture excludes floating a/b weights; model integration
adds them to the same launch. GDN microbenchmarks use raw head order; the
native TP2 permutation is independently checked against official weights.
The maximum relative L2 error across the 68 corrected operator points is
0.000939. Effective weight bandwidth counts resident codes/scales, not all
DRAM traffic, and is not an end-to-end estimate.

A long coalesced-control fixture encountered an illegal CUDA address with
the existing qualified control wheel. The remaining six points pass after
starting a fresh process. The original failure is retained, and its cause
is unresolved; it is not presented as a repaired production bug. The normal
installed model comparison below completes both full workloads without that
failure.

## Installed-artifact validation

The source-complete wheel contains the rebuilt standard `_C` extension.
Fresh-process resolved Torch/CUDA libraries come from the declared runtime;
no private kernel DSO or library override is required. Runtime source is
`753ae2bca8`, with wheel SHA256
`fc90f428b80244c7777d20e088eee5801aca39fcb8cd391d5c1fa7273f1edf3e`.
Later documentation changes do not alter runtime code.

All 136 installed quantization, graph and admission tests pass. This includes
both GDN head layouts, official-dequantization comparisons, changed-input
graphs, exact restored coefficients, and the N17408/K5120 bank's bitwise
canonical fallback at M1 and M16.

## TP2 model comparison, 2026-10-07

Hardware: V100-SXM2-16GB GPUs 0/1, pairwise NV2, driver 580.178.04,
CUDA 12.8, Torch 2.10.0+cu128, Python 3.12.3. All six shared GPU locks are
held during each run. GPUs 2/3 stay unused. Sampled SM clocks during steady
requests are 1530 MHz for both configurations; memory clocks are 877 MHz.

Target: Qwen3.8-27B-GSQ-RCO-IQ3_S, 10.963 GiB on disk. Draft:
Qwen3.8-27B-DFlash2-Q8_0, 1.915 GiB. Both configurations use the same wheel,
the same 16 prompts, 600 output tokens, seed 123, temperature 0.7, top-p 0.9,
top-k 20, max length 32768, max sequences 4, batch budget 1024, memory
utilization 0.88, CUDA Graph, target E4M3 KV, draft FP16 KV, and FP32 SSM.
Only `kernel_config.sm70_gguf.projection_planes` changes. EOS is ignored for
the fixed-length timing fixture and respected for natural text checks.
Twenty full rounds per prompt are discarded, and prompts are averaged
equally. Timing is the complete engine output interval, including draft,
target, sampling, communication and scheduling; TTFT is excluded.

| Input | Plane off, ms/round | Plane on, ms/round | Saving | Tokens/round, off / on | Pooled ms/output token, off / on |
| --- | ---: | ---: | ---: | --- | --- |
| 1K | 31.112 | 26.134 | 4.978 ms (16.0%) | 2.989 / 3.022 | 10.423 / 8.677 |
| 8K | 31.830 | 26.848 | 4.982 ms (15.7%) | 2.947 / 2.970 | 10.854 / 9.112 |

Each rank admits 240 projection layers with planes enabled, versus zero
when disabled. These results establish an actual model saving of roughly
5 ms/round, not the sum of isolated projection improvements. There is only
one sequential matched pair; this is not an independent-run confidence
interval.

All 16 first-verifier logit probes complete. On 126 rows sharing the same
token prefix and positions, mean KL(control || candidate) is 0.00007336,
maximum KL is 0.0009929, and top-1 agreement is 100%. These are logits from
the same model weights with different execution paths, not a llama.cpp or
full quality-set comparison. Both natural prompts produce identical healthy
text and terminate normally. The C4 execution check produces four nonempty
96-token outputs in 6.604 seconds with planes enabled; first-use compilation
is included, so this is not a warmed throughput claim.

Peak GPU usage is 15565/15607 MiB (15.20/15.24 GiB) with planes enabled and
15359/15359 MiB with them disabled. Target unique tensor storage drops from
7.559 to 7.232 GiB/rank, while the hybrid cache budget rises from 3.41 to
3.68 GiB/rank. The memory utilization setting therefore assigns freed weight
memory to cache instead of lowering the overall reservation. The configured
maximum is 32K and actual C1 inputs are 1K/8K; the allocator's 66,446-token
capacity report does not establish a longer-context run.

## TP4 regression

The same wheel and kernels run on all four 16GB GPUs. With the original
128-token C1 execution fixture, max sequences 1 and the same KV/SSM settings,
the control measures 14.272/14.647 ms at 1K/8K; the candidate measures
14.308/14.645 ms. Tokens per round are identical at 4.182/3.083, and both
natural prompts terminate with identical text. The short sample supports an
execution/regression check, not a new 16-prompt TP4 performance baseline.

## Full operator point table

These use the qualified control-native kernel before the TP2 head-map
change. Model integration and the new native mapping are validated above.
MB means decimal megabytes. Bandwidth is effective resident-weight bandwidth.
KW/TN/split identify the best measured configuration.

| Role | Types | N segments | K | Resident MB | KW/TN/split | Plane µs | Canonical µs | GB/s | Relative L2 |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| attn_gate | IQ4_XS | 3072 | 5120 | 8.847 | 4/2/1 | 17.48 | 27.84 | 506.0 | 0.000387 |
| ffn_down | IQ2_S | 5120 | 8704 | 25.068 | 8/2/1 | 36.66 | 54.68 | 683.8 | 0.000312 |
| ssm_out | IQ4_XS | 5120 | 3072 | 8.847 | 4/2/1 | 15.08 | 23.62 | 586.8 | 0.000396 |
| attn_gate | IQ3_XXS | 3072 | 5120 | 6.390 | 4/2/1 | 18.70 | 28.70 | 341.7 | 0.000398 |
| ffn_down | IQ3_S | 5120 | 8704 | 19.497 | 4/2/1 | 33.14 | 54.77 | 588.3 | 0.000394 |
| ssm_out | Q4_K | 5120 | 3072 | 9.830 | 4/2/1 | 15.75 | 20.44 | 624.0 | 0.000793 |
| attn_gate | IQ3_S | 3072 | 5120 | 6.881 | 4/2/1 | 19.51 | 29.60 | 352.7 | 0.000396 |
| attn_k | IQ3_S | 512 | 5120 | 1.147 | 4/2/2 | 12.00 | 23.26 | 95.5 | 0.000403 |
| attn_output | IQ3_S | 5120 | 3072 | 6.881 | 4/2/1 | 14.16 | 25.59 | 485.9 | 0.000396 |
| attn_v | IQ3_S | 512 | 5120 | 1.147 | 4/2/2 | 11.98 | 23.38 | 95.7 | 0.000391 |
| ffn_down | IQ3_XXS | 5120 | 8704 | 18.104 | 4/2/2 | 31.36 | 52.29 | 577.2 | 0.000392 |
| ssm_out | IQ3_S | 5120 | 3072 | 6.881 | 4/2/1 | 14.21 | 25.54 | 484.3 | 0.000400 |
| ffn_down | IQ4_XS | 5120 | 8704 | 25.068 | 4/2/1 | 34.99 | 48.48 | 716.4 | 0.000373 |
| attn_k | IQ4_XS | 512 | 5120 | 1.475 | 4/2/2 | 10.97 | 21.81 | 134.4 | 0.000377 |
| attn_output | IQ3_XXS | 5120 | 3072 | 6.390 | 4/2/1 | 13.59 | 24.96 | 470.1 | 0.000391 |
| ssm_out | IQ3_XXS | 5120 | 3072 | 6.390 | 4/2/1 | 13.63 | 24.99 | 468.8 | 0.000397 |
| attn_v | IQ3_XXS | 512 | 5120 | 1.065 | 4/2/2 | 11.67 | 22.96 | 91.2 | 0.000397 |
| ffn_down | IQ2_XS | 5120 | 8704 | 25.068 | 8/2/1 | 37.08 | 52.63 | 676.0 | 0.000320 |
| attn_k | Q4_K | 512 | 5120 | 1.638 | 4/2/2 | 9.93 | 17.18 | 165.0 | 0.000754 |
| attn_k | IQ3_XXS | 512 | 5120 | 1.065 | 4/2/2 | 11.65 | 22.98 | 91.4 | 0.000398 |
| attn_gate | Q4_K | 3072 | 5120 | 9.830 | 4/2/1 | 15.86 | 23.26 | 619.8 | 0.000758 |
| attn_q | IQ3_S | 6144 | 5120 | 13.763 | 4/2/2 | 29.95 | 39.33 | 459.5 | 0.000400 |
| attn_v | Q4_K | 512 | 5120 | 1.638 | 4/2/2 | 9.93 | 17.15 | 165.0 | 0.000786 |
| attn_q | IQ4_XS | 6144 | 5120 | 17.695 | 4/2/1 | 32.61 | 35.94 | 542.5 | 0.000382 |
| ffn_down | Q4_K | 5120 | 8704 | 27.853 | 4/2/1 | 36.95 | 44.02 | 753.9 | 0.000737 |
| attn_output | IQ4_XS | 5120 | 3072 | 8.847 | 4/2/1 | 15.33 | 23.76 | 577.1 | 0.000387 |
| attn_output | Q4_K | 5120 | 3072 | 9.830 | 4/2/1 | 15.79 | 20.52 | 622.5 | 0.000767 |
| attn_v | IQ4_XS | 512 | 5120 | 1.475 | 4/2/2 | 10.96 | 21.72 | 134.5 | 0.000382 |
| attn_q | IQ3_XXS | 6144 | 5120 | 12.780 | 4/2/2 | 28.88 | 38.52 | 442.5 | 0.000402 |
| attn_q | Q4_K | 6144 | 5120 | 19.661 | 4/2/1 | 31.65 | 33.02 | 621.3 | 0.000765 |
| gate_up | IQ2_XS / IQ2_XXS | 8704 / 8704 | 5120 | 50.135 | 8/2/1 | 71.30 | 107.72 | 703.2 | 0.000492 |
| gate_up | IQ3_XXS / IQ3_S | 8704 / 8704 | 5120 | 37.601 | 4/4/1 | 65.56 | 111.80 | 573.5 | 0.000675 |
| gate_up | IQ3_XXS / IQ3_XXS | 8704 / 8704 | 5120 | 36.209 | 4/4/1 | 60.10 | 93.88 | 602.5 | 0.000676 |
| gate_up | IQ3_S / IQ2_S | 8704 / 8704 | 5120 | 44.564 | 8/2/1 | 80.65 | 112.56 | 552.6 | 0.000524 |
| gate_up | IQ3_S / IQ3_S | 8704 / 8704 | 5120 | 38.994 | 4/4/1 | 64.75 | 94.37 | 602.2 | 0.000669 |
| gate_up | IQ2_S / IQ3_XXS | 8704 / 8704 | 5120 | 43.172 | 8/2/1 | 77.54 | 111.81 | 556.8 | 0.000508 |
| gate_up | IQ2_S / IQ3_S | 8704 / 8704 | 5120 | 44.564 | 8/2/1 | 81.03 | 112.56 | 550.0 | 0.000498 |
| gate_up | IQ2_XS / IQ3_XXS | 8704 / 8704 | 5120 | 43.172 | 8/2/1 | 77.56 | 109.41 | 556.7 | 0.000495 |
| gate_up | IQ2_S / IQ2_XS | 8704 / 8704 | 5120 | 50.135 | 8/2/1 | 71.62 | 113.19 | 700.0 | 0.000506 |
| gate_up | IQ2_XXS / IQ2_S | 8704 / 8704 | 5120 | 50.135 | 8/2/1 | 71.75 | 110.28 | 698.7 | 0.000496 |
| gate_up | IQ3_XXS / IQ2_S | 8704 / 8704 | 5120 | 43.172 | 8/2/1 | 78.01 | 111.49 | 553.4 | 0.000514 |
| gate_up | IQ3_XXS / IQ4_XS | 8704 / 8704 | 5120 | 43.172 | 4/4/1 | 63.84 | 105.20 | 676.3 | 0.000670 |
| gate_up | IQ3_S / Q4_K | 8704 / 8704 | 5120 | 47.350 | 4/4/1 | 65.95 | 102.85 | 717.9 | 0.000908 |
| gate_up | Q4_K / IQ4_XS | 8704 / 8704 | 5120 | 52.920 | 4/4/1 | 69.02 | 95.24 | 766.8 | 0.000929 |
| gate_up | IQ3_S / IQ3_XXS | 8704 / 8704 | 5120 | 37.601 | 4/4/1 | 64.72 | 110.51 | 581.0 | 0.000683 |
| gate_up | IQ4_XS / IQ3_S | 8704 / 8704 | 5120 | 44.564 | 4/4/1 | 65.29 | 106.05 | 682.6 | 0.000659 |
| gate_up | IQ4_XS / IQ4_XS | 8704 / 8704 | 5120 | 50.135 | 4/4/1 | 66.30 | 80.59 | 756.2 | 0.000656 |
| gate_up | IQ3_S / IQ4_XS | 8704 / 8704 | 5120 | 44.564 | 4/4/1 | 65.98 | 105.10 | 675.4 | 0.000670 |
| gate_up | Q4_K / IQ3_S | 8704 / 8704 | 5120 | 47.350 | 4/4/1 | 65.68 | 101.85 | 720.9 | 0.000939 |
| gate_up | IQ4_XS / Q4_K | 8704 / 8704 | 5120 | 52.920 | 4/4/1 | 69.05 | 94.51 | 766.4 | 0.000904 |
| qkvz | IQ4_XS / IQ4_XS / IQ4_XS / IQ4_XS | 1024 / 1024 / 3072 / 3072 | 5120 | 23.593 | 4/2/1 | 35.94 | 41.82 | 656.5 | 0.000391 |
| qkvz | IQ3_S / IQ3_S / IQ3_S / IQ3_XXS | 1024 / 1024 / 3072 / 3072 | 5120 | 17.859 | 4/2/1 | 35.13 | 65.37 | 508.4 | 0.000400 |
| qkvz | IQ3_XXS / IQ3_XXS / IQ3_XXS / IQ3_S | 1024 / 1024 / 3072 / 3072 | 5120 | 17.531 | 4/2/1 | 35.16 | 65.05 | 498.6 | 0.000403 |
| qkvz | IQ3_S / IQ3_S / IQ3_S / IQ3_S | 1024 / 1024 / 3072 / 3072 | 5120 | 18.350 | 4/2/1 | 35.26 | 46.54 | 520.4 | 0.000403 |
| qkvz | IQ2_XS / IQ2_XS / IQ2_XS / IQ3_XXS | 1024 / 1024 / 3072 / 3072 | 5120 | 21.135 | 8/2/1 | 39.50 | 66.20 | 535.1 | 0.000319 |
| qkvz | IQ3_XXS / IQ3_XXS / IQ3_XXS / IQ4_XS | 1024 / 1024 / 3072 / 3072 | 5120 | 19.497 | 4/2/1 | 33.07 | 63.60 | 589.5 | 0.000395 |
| qkvz | IQ3_S / IQ3_S / IQ3_S / IQ2_S | 1024 / 1024 / 3072 / 3072 | 5120 | 20.316 | 8/2/1 | 39.18 | 67.92 | 518.6 | 0.000320 |
| qkvz | IQ3_XXS / IQ3_XXS / IQ3_XXS / Q4_K | 1024 / 1024 / 3072 / 3072 | 5120 | 20.480 | 4/2/1 | 31.91 | 59.05 | 641.8 | 0.000544 |
| qkvz | IQ3_S / IQ3_S / IQ3_S / IQ4_XS | 1024 / 1024 / 3072 / 3072 | 5120 | 20.316 | 4/2/1 | 33.89 | 64.32 | 599.4 | 0.000399 |
| qkvz | IQ3_XXS / IQ3_XXS / IQ3_XXS / IQ3_XXS | 1024 / 1024 / 3072 / 3072 | 5120 | 17.039 | 4/2/1 | 33.75 | 46.31 | 504.9 | 0.000396 |
| qkvz | IQ4_XS / IQ4_XS / IQ4_XS / IQ3_S | 1024 / 1024 / 3072 / 3072 | 5120 | 21.627 | 4/2/1 | 35.52 | 64.33 | 608.9 | 0.000394 |
| qkv | IQ3_S / IQ4_XS / Q4_K | 6144 / 512 / 512 | 5120 | 16.876 | 4/2/2 | 35.01 | 79.73 | 482.0 | 0.000450 |
| qkvz | IQ3_S / IQ3_S / IQ3_S / Q4_K | 1024 / 1024 / 3072 / 3072 | 5120 | 21.299 | 4/2/1 | 33.16 | 59.65 | 642.4 | 0.000564 |
| qkvz | IQ4_XS / IQ4_XS / IQ4_XS / IQ3_XXS | 1024 / 1024 / 3072 / 3072 | 5120 | 21.135 | 4/2/1 | 34.60 | 64.09 | 610.8 | 0.000393 |
| qkv | IQ3_XXS / IQ4_XS / Q4_K | 6144 / 512 / 512 | 5120 | 15.892 | 4/2/1 | 33.26 | 79.26 | 477.8 | 0.000464 |
| qkvz | IQ3_XXS / IQ3_XXS / IQ3_XXS / IQ2_S | 1024 / 1024 / 3072 / 3072 | 5120 | 19.497 | 8/2/1 | 38.16 | 67.30 | 510.9 | 0.000318 |
| qkvz | Q4_K / Q4_K / Q4_K / IQ3_S | 1024 / 1024 / 3072 / 3072 | 5120 | 23.265 | 4/2/1 | 36.27 | 60.12 | 641.5 | 0.000673 |
| qkv | Q4_K / IQ4_XS / IQ3_S | 6144 / 512 / 512 | 5120 | 22.282 | 4/2/1 | 35.18 | 80.73 | 633.4 | 0.000695 |
