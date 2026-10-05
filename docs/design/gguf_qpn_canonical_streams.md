# Canonical streams on the small-M QPN skeleton

The measured raw Q4_K and pure IQ4_XS candidates lose to canonical execution.
Screen the existing group32 U4/IQ4 code and scale streams on the same shared-A
QPN skeleton before adding a model route. This avoids nested source-scale
unpacking without changing coefficient storage, FP32 accumulation or the
canonical IQ lookup transform. Coalesced stats retain their full row stride.

## Same-machine screening

Real rank0 TP4 projections, M8, V100-SXM2-32GB, CUDA 12.8,
Torch 2.10.0+cu128, Python 3.12.14, 300W, 1290MHz SM and 877MHz memory.
Each graph call follows 16MiB L2 eviction. Canonical/candidate/candidate/
canonical arms are measured together. The pair baseline includes the model's
single SiLU/multiply launch, rather than two eager pointwise calls.
Read bytes are canonical operand-stream sizes, not NCU DRAM traffic.

| Role | Type | N/K | Canonical us | Candidate us | Read bytes/rank | Effective GB/s | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| down | IQ4_XS | 5120/4352 | 38.912–39.936 | 40.960–40.960 | 12533760 | 306.0 | Keep canonical |
| down | Q4_K | 5120/4352 | 34.816–34.816 | 38.912–38.912 | 13926400 | 357.9 | Keep canonical |
| pair | IQ4_XS | 4352/5120 | 60.416–60.928 | 63.488–63.488 | 25067520 | 394.8 | Keep canonical |

The matching NVFP4-checkpoint down call takes 26.624us, and its gated pair
50.176–53.248us on the same machine and clock. Reading canonical streams does
not close that skeleton gap by itself. These candidates add no model benefit:
all twenty-one IQ4_XS down layers, six Q4_K down layers and four pure IQ4_XS
pairs retain their existing canonical execution. No original-superblock
shape tuning, precision reduction or new default route is introduced.

## Correctness and retained evidence

All three candidates pass official GGUF FP32 reconstruction with three seeded
inputs, finite outputs, one thousand bitwise-stable CUDA graph replays and
zero reduction counters. The complete `dev58+g9da4442bca` prototype wheel
passes installed-native hash auditing. This is isolated operator evidence;
the unsuccessful operators are excluded from the final source change.

The [retained screening record](data/gguf_qpn_canonical_streams_20261006.json)
contains all timing arms, error norms and bytes. Prototype source and the
matched-epilogue benchmark remain reproducible from their recorded commits.

## Optional lossless lattice expansion

A projection-only U4 stream with per-group FP32 coefficients would occupy
2,621,767,680B per rank across the recorded lattice weights. Replacing all
source-sized projection records would add 1,009,152,000B per rank; retaining
those records alongside the new stream would add the full 2,621,767,680B.
IQ2_XS/S and IQ1_M need group16, while IQ2_XXS and IQ3_XXS/S use group32.
Codebooks are global format constants; no coefficient is narrowed to FP16.
This is a storage budget, with no claimed throughput or allocation result.

The previous TP4 capture already allocated 30,005,748,224B per rank. An
expanded route therefore needs both a measured exact-shape winner and a
fresh max-length/KV-capacity check before committing memory. No expanded
weight allocation is selected for this projection batch. Existing canonical
routes preserve the current memory and numerical contract.
