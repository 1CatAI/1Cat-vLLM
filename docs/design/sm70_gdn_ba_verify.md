# SM70 verifier input projections

The SM70 Qwen H5120 verifier combines the channel FP8 QKVZ projection with
the FP16 b/a projections and writes contiguous QKV, Z, b and a outputs.
The local tensor geometry controls admission. The model adapter owns
qualification; a TP degree or one speculative query width does not select
a different provider. Weights, scales and KV storage retain their precision.

The b/a reduction order requires numerical validation against dense FP32 or
FP64 evaluation. QKVZ retains the QPN8 accumulation schedule for each row.
The native projection accepts continuous small-row batches and tiles both
the token rows and b/a columns. This includes the second b/a column tile
required by a wider local shard.

## Shared TP2 and TP4 paths

For the qualified 27B model, sharding changes these local sizes:

| Tensor | TP2 | TP4 |
| --- | ---: | ---: |
| Attention Q heads / KV heads | 12 / 2 | 6 / 1 |
| GDN key heads / value heads | 8 / 24 | 4 / 12 |
| GDN QKV width | 5120 | 2560 |
| GDN Z width | 3072 | 1536 |
| GDN QKVZ width | 8192 | 4096 |
| GDN b/a width together | 48 | 24 |
| FFN down-projection K | 8704 | 4352 |
| LM-head vocabulary rows | 124160 | 62080 |

The same owners handle these changes:

- QPN2/QPN8 preparation and dispatch consume local N/K. Bundled FFN weights,
  scale rebias and gated projections use the same implementation for wider
  shards; measured constant-width schedules remain internal tuning choices.
  Shared-weight/compact-scale down projections retain one TurboMind packing.
  Their native decode schedule consumes that packing directly, avoiding a
  second persistent weight copy and retaining the original prefill projection
  and its accumulation order.
- GDN projection splitting, convolution/gating/output initialization and
  per-head N128 normalization derive their widths from the local head layout.
  Live lengths, invalid slots and accepted-history offsets retain the existing
  state contract. LoRA forwards retain their adapter-aware boundary.
- E4M3 grouped attention runs one grid over requests and KV heads. Every
  six-query-head group retains its compensated FP32 reduction and owns a
  workspace panel. Older extensions retain serial grouping and its query/output
  copies. There is no KV conversion between the two schedules.
- Refined draft and bounded target heads use the full vocabulary shard.
  Candidate and dense-fallback ranking use wider IDs when a shard exceeds
  65536 rows. The target still certifies against every discarded bound and
  otherwise scores the complete original FP16 head. Old head extensions decline
  wider shards. Public dense projection APIs keep their original fallback.
  A wider multirow cuBLAS reference selects two K partitions even with
  FP32-only reductions. Native scoring retains that FP32 partition schedule;
  single-row and narrower projections retain one chain. Qualified TP2/TP4
  scores are checked bitwise against the original dense projection, including
  every M1–8 width and the dense fallback above the fast-row capacity.
- Collective admission uses communicator capabilities. The two-peer pull
  reduction and four-peer push transport remain topology-specific internals;
  both provide fused all-reduce/Gemma RMSNorm and can publish the existing K16
  activation layout for downstream projections. Their historical op names stay
  compatible. TP4 row admission follows the native capacity rather than a list
  of speculative batch sizes.
- Local argmax and compact token all-gather already consume the communicator's
  TP size. They retain their shared selection and stable token-ID tie breaking.

Native QPN2 decode reports its row capacity instead of imposing a separate
Python M32 limit. Older extensions retain M32 admission; rows beyond the loaded
operator's capacity use the existing bounded prefill workspace. Gated N128
normalization similarly reports its native indexing capacity, with the older
extension's limit retained when the capability query is absent.

This is the H5120 dense model's MTP path. Other models, positive-temperature
DFlash sampler graphs and its separately qualified single-layout loading policy
retain their existing qualification. Tensor support for another TP degree is
not an end-to-end quality, memory-fit or performance claim for that degree.
The optional TP4 long-prefill reduce-scatter/norm/all-gather experiment also
retains its qualification; it is disabled in the MTP measurements and is not
advertised as part of this shared path.
