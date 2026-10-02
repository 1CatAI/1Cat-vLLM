# PLE overflow cascade

FP8 PLE tables can span device memory, pinned host memory and file-backed
checkpoint storage. Each rank places rows in that order, filling its measured
device budget, then its configured host share, and reading the remainder from
mapped safetensors shards. Every row belongs to exactly one tier.

The compute rank gathers its resident rows and waits for the offload worker's
raw FP8 bytes inside its graph. Both use the same E4M3 dequantization and scale
before the tensor-parallel reduction. The worker's disk segments are disjoint
in the global vocabulary and each rank merges only its own slots.

## Configuration and capabilities

`KernelConfig.ple_disk_cascade` defaults to true. Admission requires raw E4M3
PLE storage, FP16 embedding output, CUDA compute ranks, file-backed safetensors
loading and a supported local worker topology. PLE layers must be on the first
pipeline stage. Pipeline parallelism is supported; context-parallel groups,
DBO, non-local workers and weight transfer currently use their existing paths.
The startup report records the admission result and rejection reason.

Existing explicit hybrid, disk-only or whole-table offload placement retains
precedence. The cascade resolves per engine and starts its worker without
changing process environment. An ineligible model retains its existing placement.

The existing `VLLM_QWEN4EXP_PLE_HOST_GIB` sets the pinned share per rank;
zero requests no host tier. Device and host reserve settings keep their existing
meaning. A fitting table pins only the rows it needs. If the gather contract
cannot retain its required resident row, startup reports the memory requirement.

Use `--kernel-config '{"ple_disk_cascade":false}'` to disable the cascade.
`ple_disk_release_pages` defaults to false. Setting it true releases mappings
of the file-backed pages read after each gather, reducing resident RAM while
leaving the checkpoint and page-cache data intact. Anonymous memory is never
released this way. No new environment variable is introduced.

## Lifetime and loading

Only ranks that own PLE layers register a connector. Other pipeline stages
participate in their normal model execution. Resident device and host tables
remain in the compute ranks; the offload worker keeps the remainder mapped
from the checkpoint. Direct shard copies avoid a separate device staging copy.

The worker returns bytes in its request-bound output buffer. The compute rank
waits on that request's event before merging rows, preserving graph replay and
concurrent-request ownership. Page release occurs after the gather has copied
its selected rows into owned storage.

## Validation

CPU tests cover tier capacities and boundaries, rank-local disk segments,
checkpoint mapping, repeated reads after page release, configuration isolation,
worker registration and pipeline admission. The configuration tests verify
that activation does not write process environment.

GPU controls exercise resident/remote row merging and graph replay. Default
placement changes also require paired model quality and target speed validation.
