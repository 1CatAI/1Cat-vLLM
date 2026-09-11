# Grouped CPU offload and filesystem compatibility

Equal-block-size hybrid attention/Mamba caches use group-local CPU pools to
avoid charging each offload key for every group's backing tensors. The native
connector retains original KV group IDs, including empty positions for scratch
groups. Group-local slot IDs may repeat because the CPU tensors are disjoint.

## Public interface boundary

The generic KV connector base classes, connector factory, LMCache connectors,
and LMCache adapters are unchanged by this fork patch. The offloading metadata
classes, request context, manager interface, and worker interface also retain
their existing contracts. Native offloading block-size validation now ignores
non-prefix-cacheable scratch groups while preserving full group arrays.

The group wrapper delegates allocation, eviction, reference counting, and I/O
completion to existing managers. It forwards request lifecycle and shutdown
hooks to every child, and combines their request-level store preferences.
The regular CPU manager and transfer implementation are not duplicated.

This is source-level interface compatibility. It does not certify any
particular LMCache release with this fork, model, or GPU stack.

## CPU filesystem composition test

`tests/v1/kv_offload/cpu/test_grouped_tiering.py` composes one existing
`TieringOffloadingManager` per cacheable group behind the group wrapper. Each
child uses the existing `CPUPrimaryTierOffloadingManager`, `SharedOffloadRegion`,
`SecondaryTierFactory`, and `FileSystemTierManager`. No filesystem backend,
serializer, on-disk envelope, or transfer state machine is introduced.

The test uses real mmap backing and filesystem I/O. A writer process exits
before a fresh spawned reader process starts. It covers:

- Noncontiguous groups 0 and 2 with the same content hash and slot ID.
- Different group page sizes, each containing two distinct worker slices.
- Existing group-aware FileMapper paths and exact byte preservation.
- A new reader with empty RAM and different destination slot IDs.
- LRU eviction and reuse after the restored cache exceeds capacity.
- A truncated file becoming a miss rather than usable corrupt state, while
  the other group still restores correctly.
- Request policy propagation, request completion, and child resource cleanup.

The standard FS backend obtains row size from its own primary memory view.
Separate instances therefore handle different group row sizes without changing
its I/O interface. Slot IDs are transient locations; the existing OffloadKey
and FileMapper identify persisted data independently of those locations.

## Serving gate and remaining work

This test is a scheduler/storage composition proof. It is not an implementation
of grouped tiering in `TieringOffloadingSpec`, which still requires a single
group. The GPU worker's grouped CPU path still uses private tensors and rejects
mmap backing. Keep these guards until the worker integration is implemented and
validated.

Before enabling grouped tiering in serving:

1. Bind the grouped worker tensors to the same per-group shared regions used
   by the scheduler, preserving TP rank slices, completion fences, and the total
   CPU byte budget. Verify cleanup after partial initialization failures.
2. Define an explicit persistent layout namespace. Existing group tags prevent
   cross-group collisions, but do not by themselves distinguish legacy full-row
   files from new group-row files or every physical-layout change. Do not read
   old incompatible bytes just because their content hash matches. Reuse the
   backend format with a distinct, validated configuration namespace.
3. Validate GPU -> RAM -> filesystem -> fresh process -> RAM -> GPU with real
   model output checks, Mamba boundaries, MTP, and eviction pressure. CPU byte
   equality alone cannot establish inference correctness.
4. Define storage capacity/cleanup and crash-durability requirements separately.
   Process-restart reuse is not a power-loss durability guarantee, and a real
   temporary filesystem test is not an SSD throughput benchmark.

LMCache is an alternative connector path with its own cache objects and
backends. Keeping the public connector contract compatible enables evaluating
that path; it does not imply that LMCache can read native FS files or attach
directly to the native group pools.
