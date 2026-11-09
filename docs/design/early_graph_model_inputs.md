# Prepare independent model inputs before FULL graph metadata

Default model-state position preparation depends on staged request/token
buffers, not attention metadata. Declare this capability explicitly and submit
it before attention metadata construction. Preserve current-stream ordering;
model states without the capability retain the existing order. PLE submission
remains independent of whether positions were prepared early.

## Installed checks

Seven CPU cases cover FULL/eager modes, capability rejection and PLE submission
ownership. Two V100 cases compare position values and changed-input graph
replay for one/four requests. All nine pass. A later main integration wheel
passes 29 related CPU checks; the two GPU cases are skipped in that CPU run.
FP16 operands and FP32 accumulation/state are unchanged.

## Same-engine Flash-Next phase comparison

Flash-Next IQ3_S GGUF, FP16 MTP4, TP4 V100-SXM2-32GB, CUDA 12.8,
Torch 2.10.0+cu128, Python 3.12.3, FULL graphs, FP16 KV/activation, FP32
recurrent state/accumulation, max length 9216, batch 512, four sequences,
memory utilization 0.95, and prefix caching disabled. The ordinary installed
wheel is source `173fc0b97a`; its native core is
`793eaf9751a9a9dd10e1d0708164ce22c471c60d2957b1acdf21ea69883eeafd`.

After warmup, one engine runs late/early observed C1 arms, early/late unobserved
C1 arms, and late/early C4 arms. C1 uses I8192/O256; C4 uses I128/O600.
These synthetic performance probes ignore EOS. All four C1 arms have identical
complete token IDs. Both C4 arms have identical IDs in every stream.

| Probe | Late round mean | Early round mean | Difference |
| --- | ---: | ---: | ---: |
| C1, 35 middle intervals / 171 tokens | 24.3621 ms | 23.9364 ms | −0.4257 ms |
| C4, 232 middle intervals / 2171 tokens | 49.2177 ms | 48.9204 ms | −0.2972 ms |

CPU-only observation timestamps the actual replay call, after manager waits;
it introduces no GPU fence. Across 38 matched middle C1 rounds, median
inter-rank entry spread falls from 547.455 µs to 202.568 µs. The observed
arms contain scheduling outliers and do not replace the unobserved speed
comparison. Nested CPU stage walls include GPU waits and are not additive.

After reordering, the post-attention tail is 45.5–50.9 µs across ranks. The
remaining 202 µs spread already exists when attention preparation returns.
Rank zero also handles asynchronous output materialization/serialization;
its measured host launch durations differ from the other ranks. This result
does not meet the 100 µs entry-spread objective. Further diagnosis belongs to
the attention-wait/output boundary, rather than another position-launch move
or an artificial synchronization barrier.
