# Shared-activation GGUF output projections

TP4 GDN output and full-attention output share M8/N5120/K1536. Reuse the
measured single-matrix QPN body and existing lattice/IQ4 readers. Same-round
screening compares eighty N64 CTAs with one hundred sixty CTAs and two K
partitions. The admitted one-partition version writes ordered FP32 intra-CTA
reduction directly, without a separate reduction launch.

GDN weights retain GGUF's three-head tiled K order. Each 128-wide activation
segment is mapped to its vLLM head order while filling shared A. This needs
no activation permutation kernel, decoded weight cache or new coefficient
rounding. Attention output uses ordinary input order. FP16 operands/output,
original FP32 scale products and FP32 accumulation remain unchanged.

## Admission and fallback

Default admission is SM70, FP16 activations, M8/N5120/K1536 and original
IQ3_XXS, IQ3_S or IQ4_XS records. Prefix and storage/head-layout guards require
a GDN out projection with the adapter's 3x128 head layout, or an attention
o_proj with no head layout. This covers 42 GDN and 13 attention layers.
Q4_K and unmeasured shapes/types remain canonical, with an explicit reason.

The selection is inside an opaque runtime-M operator. Other M follows the
same canonical operator and, when canonical preparation did not restore
GDN weights, the same input head permutation. A prefill-first compiled call
therefore cannot freeze M8 onto the prefill path. The one-partition route
never writes reduction storage; its outer operator treats that unused storage
as read-only to avoid functionalization copies during replay.

## Same-machine operator results

Real rank0 TP4 weights, V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128,
Python 3.12.14, 300W, 1290MHz SM and 877MHz memory. Each graph call follows
16MiB L2 eviction; median timing uses the common cold-graph fixture.
Canonical/candidate/candidate/canonical arms are measured together. The table
uses the installed, prefill-first compiled model-wiring call, not only its
native entry. Read bytes are the original record stream, not NCU DRAM traffic.

The counterpart is channel FP8 in the NVFP4 checkpoint, measured on the same
machine/clock/shape. Its GDN output is 18.432us; attention output is
18.432–19.456us. No speedup is attributed to the difference in model weights.

| Role | GGUF type | Layers | Canonical us | QPN us | Read bytes/rank | Effective GB/s | Estimated saving us/round |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| gdn_out | IQ4_XS | 16 | 26.624–26.624 | 21.504–21.504 | 4177920 | 194.3 | 81.920–81.920 |
| attention_o | IQ3_S | 10 | 25.600–25.600 | 20.480–20.480 | 3379200 | 165.0 | 51.200–51.200 |
| gdn_out | IQ3_S | 22 | 28.672–28.672 | 21.504–21.504 | 3379200 | 157.1 | 157.696–157.696 |
| attention_o | IQ3_XXS | 1 | 24.576–24.576 | 19.456–19.456 | 3010560 | 154.7 | 5.120–5.120 |
| gdn_out | IQ3_XXS | 4 | 28.672–28.672 | 20.480–20.480 | 3010560 | 147.0 | 32.768–32.768 |
| attention_o | IQ4_XS | 2 | 22.528–23.552 | 21.504–21.504 | 4177920 | 194.3 | 2.048–4.096 |

Layer-weighted savings are **0.330752–0.332800ms per round**, an
operator estimate rather than an end-to-end result. The original 8us target
is not met. These measured shapes are still faster than canonical; additional
reader/layout work can be measured separately without delaying their admission.

Q4_K is screened through the existing canonical u4/group32 stream, avoiding
nested source-scale decoding. It loses: GDN 21.504 versus 20.480us, attention
20.480 versus 19.456us. Those six GDN and three attention layers stay on the
original canonical route. No original Q4_K reader tuning is introduced.

## Correctness and package checks

All eight prototype role/type cases pass official FP32 GGUF reconstruction
with three seeded inputs, one thousand bitwise-stable graph replays and reset
counters; maximum relative L2 is 0.0008324, including the rejected Q4_K cases.
The six installed-wiring cases pass the same oracle with relative L2 below
0.0003406 and finite outputs. Runtime rows 512/8/1/5/16/20/32/8 cover compiled
prefill-first selection, exact other-M canonical output and graph/eager equality.
Eight CPU tests cover shape/M/precision admission, fallback GDN order,
serialized band handling and one opaque dynamic export boundary.

The complete `dev56+g5255e0d1b6` wheel matches the owned source Python modules
and all shipped native libraries. Every native source/build input is identical
to the previously compiled complete `dev54+gab9eb0c084` wheel, so the ordinary
precompiled-wheel build reuses those unchanged artifacts. No private native
library or source overlay is needed. The twenty-eight native variants use
56–64 registers, 33793B shared memory and no stack/local spills.

The [retained operator record](data/gguf_qpn_small_outputs_20261006.json)
contains every arm, oracle and runtime-M check. Model-level measurement follows
integration of the projection batch; no model speedup is claimed here.
