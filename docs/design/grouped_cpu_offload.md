# Grouped CPU offload and filesystem compatibility

Equal-block-size hybrid attention/Mamba caches use group-local CPU pools to
avoid charging each offload key for every group's backing tensors. The native
connector retains original KV group IDs, including empty positions for scratch
groups. Group-local slot IDs may repeat because the CPU tensors are disjoint.

## Mamba checkpoint stride

Flash-Next's `align` Mamba mode materializes one recurrent-state snapshot per
784-token block boundary during prefill. The GPU keeps only the latest state
per request, but the offload tier received every boundary for each of the four
Mamba groups. On the measured TP4 layout that is 29.7 MB of state per token
block against 10.24 MB of attention/PLE data, so 74% of the RAM budget held
state snapshots and 16 GiB fit about 1.3 contexts of 64K tokens.

`mamba_checkpoint_stride` (kv_connector_extra_config, default `1`) keeps only
every stride-th boundary plus the final prefill boundary of each request. The
final boundary mirrors the core scheduler's last cache position, including the
one-block EAGLE/MTP shift. Loads already fall back to the nearest stored state:
the Mamba group lookup scans backwards from the attention hit and the outer
lookup shrinks the hit window, so full-prefix hits stay exact and partial-prefix
hits recompute at most `stride - 1` blocks. Intermediate states are never
loaded; only the boundary state at the hit position is transferred.

Group pools are sized by role. Token groups receive `N` slots and each Mamba
group receives `cdiv(N * (2 * stride - 1), stride**2)` slots, the number of
`cdiv(B, stride)` checkpoints that requests of at least `stride` blocks can
occupy across `N` token blocks. `N` is the largest value whose token pages plus
state slots fit `cpu_bytes_to_use`. With stride 1 this reproduces the equal
107-slot layout. On the same 16 GiB budget:

| stride | token slots | 64K contexts | state slots per group |
|-------:|------------:|-------------:|----------------------:|
| 1      | 107         | 1.3          | 107                   |
| 4      | 184         | 2.2          | 81                    |
| 8      | 248         | 3.0          | 59                    |
| 16     | 309         | 3.7          | 38                    |

Shared regions and mmap files carry the per-group slot count, so the worker
views, the scheduler memoryviews and the FS tier rows use the same geometry.
Requests shorter than `stride` blocks may occupy proportionally more state
slots than the bound assumes; their Mamba states age out first and such
prefixes then miss rather than restore a wrong state.

### V100 validation of stride 8

Flash-Next AWQ, TP4, FP16 KV, CUDA Graphs, 16 GiB CPU budget, 1.19 GiB GPU
KV per rank, the production image `b8aa829785` with the eight PR Python files
overlaid and hash-verified at start. GPU prefix state was reset before every
restore, so all hits below are external (local hits were zero throughout) and
output token IDs were compared against the cold run of the same prompt.

| mode | scenario | result |
|---|---|---|
| MTP0 | 64K cold, 20K pressure, restore, rehit | 63,504 external tokens (81 blocks, the final prefill boundary); IDs identical |
| MTP0 | partial prefix: 40K prompt sharing 39,983 tokens (50 blocks) with a stored prompt, offload disabled for the probe | 37,632 external tokens = 48 blocks, rounded down to the stride; IDs identical to its cold run |
| MTP0 | three distinct 64K contexts stored, then each restored | 63,504 external tokens for all three |
| MTP3 | 64K cold, pressure, restore, rehit | 62,400 external tokens = 78 x 800-token blocks, the MTP-shifted final boundary; six drafted/accepted; IDs identical |
| MTP3 | three distinct 60K contexts stored, then each restored | 58,400 external tokens for all three |
| MTP0, stride 1 | same three 64K contexts (107 slots per group) | all three restores missed: each miss re-stores 81 blocks and evicts the others |

Pool geometry at stride 8: MTP0 248 token slots and 59 state slots per Mamba
group (3.996 GiB pinned per rank); MTP3 232 token slots and 55 state slots
because its speculative padding makes 800-token blocks. Restores of 60K-64K
took 1.1-1.4 s against 27-35 s cold. Evidence:
`/mnt/llm_hfs/builds/qsa-stride-validation-20260912` (diag JSONL, raw
requests, result JSON, server logs, acceptance summary).

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

### External LMCache metadata validation

`tests/v1/kv_connector/unit/test_lmcache_group_metadata.py` optionally loads
the real LMCache group converter and native extension, using this fork's
`KVCacheConfig` and spec classes with small synthetic CPU tensors. It checks
engine IDs, layer mapping, recurrent windows, DCP token spans, and block-ID
routing. It skips when the optional LMCache dependency is absent.

Against LMCache `b5d109ea99a89b4d8a670ee4fc2e8cb76411ee5c`, four dense/hybrid
metadata cases pass. The QSA scratch-exclusion case is a **known failure**,
recorded with strict xfail: LMCache returns engine groups `[0, 1, 2]` for
attention/scratch/Mamba, rather than excluding scratch while retaining IDs
`[0, 2]`. Its converter does not honor `prefix_cacheable=False`. The converter
file is unchanged in LMCache dev `fcb67c0ab1db2a4bad78e085b3a2df33da003b7c`.
An unexpected pass deliberately fails the test so this limitation can be
reassessed after a third-party update.

Run in an environment with the compiled LMCache dependency installed:

```bash
.venv/bin/python -m pytest --noconftest -q -rx \
  tests/v1/kv_connector/unit/test_lmcache_group_metadata.py
```

This is metadata compatibility evidence, not GPU transfer, real-model
inference, LMCache eviction, or restart acceptance. The native offloading
scratch filter does not change the separate LMCache connector path; this PR
does not claim QSA serving support through LMCache.

### Unmerged LMCache PR compatibility checks

LMCache [#5042](https://github.com/LMCache/LMCache/pull/5042), tested at
`322fecf84a6cac2d126fb3de3ea91fa5ac177945`, fixes the scratch metadata case:
all five downstream metadata tests pass with `--runxfail`. Its native
transfer/shape tests pass on V100 (81 cases), and its native filesystem tests
pass (two cases). Relevant upstream CPU tests report 119 passed and two
GLM-specific failures because this fork lacks the newer `tokens_per_state`
constructor argument.

Real Flash-Next AWQ TP4/FP16-KV/MTP0 initialization nevertheless fails during
LMCache registration. The new MLA view rule rejects the compressed QSA NHD
shape `(124, 196, 1, 128)`. This fork represents compression with
`compress_ratio=4` and `storage_block_size=196`; the rule's legacy fallback
uses the logical block size 784 instead. A minimal CPU reproducer fails for
both NHD and HND, while the PR-base group-edits module passes both cases.
The [upstream test report](https://github.com/LMCache/LMCache/pull/5042#issuecomment-5639817810)
includes the reproducer. No real-model store/retrieve or MTP acceptance was
reached. The separate tracker relocation issue addressed by LMCache #5004
also remains on that head.

LMCache [#5059](https://github.com/LMCache/LMCache/pull/5059), tested at
`616484d156f2ae97f77ec9fadac27c8e5bbbaa60`, deliberately rejects scratch
groups. Its 24 related upstream CPU tests and six additional real-fork
validation cases pass, including direct/wrapped scratch rejection at both
validation and registration, with attention/Mamba positive controls. The
[validation report](https://github.com/LMCache/LMCache/pull/5059#issuecomment-5639752221)
records the scope. Its rejection policy conflicts with #5042's support
policy; the two heads were tested independently, not combined.

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

The composition test now enters `TieringOffloadingSpec.get_manager()`. The spec
creates one primary/secondary manager per group and binds worker tensors to
matching shared regions. Construction unwinds partially created tiers and
mappings on failure. The grouped path requires single-node TP and an explicitly
selected attention backend; other layouts retain their existing path.

FileMapper accepts optional persistent layout metadata. Grouped tiering supplies
a version tag, the existing vLLM configuration hash, attention backend, model
revision, group page sizes, and physical tensor order/sharing information.
Legacy paths are unchanged when no layout metadata is supplied. This separates
group-row files from old full-row files without changing the FS byte format.

## Validation scope

CPU tests cover scheduler/worker shared views, independent group rows,
noncontiguous group IDs, different destination slots after restart, capacity
pressure, truncated-file isolation, and partial construction cleanup.
The related connector, CPU, shared-region, tiering, FileMapper and FS regression
suite passed 165 tests with two skips. Legacy FS tests used buffered I/O because
their ordinary Torch buffers were not O_DIRECT-aligned; the grouped mmap/spawn
cases retained real O_DIRECT.

On four V100 GPUs with Flash-Next AWQ, TP4 and CUDA Graphs, fresh-process
restoration of a 16,000-token prompt restored 15,680 tokens with MTP disabled
and 15,200 with MTP3. Local prefix hits were zero. Both restored outputs matched
the writer's eight output token IDs exactly. MTP3 reported six drafted and six
accepted tokens. Additional distinct approximately 16K prompts exceeded the
4 GiB CPU budget; after resetting GPU prefix state, the original prompt still
restored with the same external-hit count and identical output token IDs.
MTP3 writer and reader shutdowns removed all five group mmap files.

The GPU runs used the Python implementation at `2b98bec016` over a verified
`b8aa829785` image, rather than a full rebuilt image. File storage was NFS.
These tests establish the tested restart/pressure path, not SSD throughput,
power-loss durability, arbitrary backend/layout interoperability, or maximum
production context acceptance. Persistent disk quota/garbage collection remains
a separate backend policy; the bounded LRU/ARC budget applies to RAM. Cache
files must be isolated when model weights or their physical representation
change, including replacing weights in place under the same model path.

### Complete-image regression

A clean native build at `0f0139b23d` was packaged with the matching Python
source; all 1,954 packaged Python/native files were hash-verified before each
server start. The image used CUDA 12.8, Torch 2.10.0+cu128, and V100/SM70.
No runtime source overlay was used.

On Flash-Next AWQ, TP4, FP16 KV, a 4 GiB CPU budget and CUDA Graphs, MTP0
and MTP3 each passed six RAM requests with 16,000-token contexts. Clearing GPU
prefix state before each request exposed 15,680 external tokens with MTP0 and
15,200 with MTP3 on a repeat. After distinct
B/C contexts exceeded RAM capacity, C still hit while the older A had zero
external hits. Cold, restored and recomputed output token IDs matched exactly, including
across MTP0/MTP3. MTP3 drafted and accepted tokens during these requests.

The same complete image passed MTP0 filesystem restart and capacity pressure:
a fresh reader restored 15,680 tokens with zero local hits and the writer's
identical eight output token IDs. After two more approximately 16K contexts,
A restored with the same hit count and output IDs. Writer and reader both
exited zero and removed all five group mmap files.

After the host-registration error handling follow-up at `9c47de86a6`, a new
complete image with unchanged native sources passed MTP3 filesystem restart
and pressure. Both restored A requests had 15,200 external tokens, zero local
hits and the writer's identical eight output token IDs; each reported six
drafted and six accepted tokens. The writer and reader both exited zero and removed all five
mmap files. These images contain the complete matching Python source and
verified native artifacts; no runtime source overlay was used.

The ordinary single-group regression used Qwen3-0.6B on one V100 with FP16,
CUDA Graphs and a 256 MiB CPU budget. Three distinct 1,597-token prompts served
with offload disabled established the baseline. With native CPU offload enabled,
a repeat restored 1,584 tokens; after B/C pressure, recent C still hit and old A
missed. All 24 generated token IDs matched the corresponding disabled-offload
baseline. GPU prefix state was reset between requests.

LMCache is an alternative connector path with its own cache objects and
backends. Keeping the public connector contract compatible enables evaluating
that path; it does not imply that LMCache can read native FS files or attach
directly to the native group pools.

## Host registration failures

Shared mmap regions must be successfully registered with `cudaHostRegister`
before native batch KV transfers can use them. An MTP3 filesystem startup
exposed a registration failure: the previous warning-and-continue branch left
a CUDA error pending and the next unrelated kernel failed. A separate GPU
probe also confirmed that the native batch transfer rejected unregistered host
memory. Registration failure now raises immediately with rank, path, size and
error code, using the existing construction cleanup path, consistent with the
other native CPU offload path. It does not introduce a pageable-memory transfer
fallback or guarantee that host memory registration always succeeds.

The follow-up CPU regression passed 167 tests with two skips, including
registration error codes 1/2 and partial-construction cleanup. Four-GPU probes
also passed 100 registrations of five shared regions per process over five
iterations; this does not establish the cause of the one serving-time failure.

## Reproducible block keys across restarts

Set a fixed `PYTHONHASHSEED` (for example, `PYTHONHASHSEED=0`) before starting
both the writer and reader engines, and retain the same prefix hashing
algorithm. vLLM initializes the prefix chain's first hash from random bytes when
this variable is absent. Matching file layout metadata alone therefore cannot
produce restart hits: the same prompt will have different block keys. Use the
existing seed configuration; do not replace vLLM's hashing algorithm.

## Shutdown and resource lifetime

For file-backed serving, allow normal engine shutdown to finish, for example
with `--shutdown-timeout 60`, and give the container runtime a longer stop grace
period. A zero engine shutdown timeout can terminate the process before its
scheduler cleanup runs. The scheduler performs final unlink even when a worker
created a shared region first; already open mappings remain valid until closed.

The TP4/MTP3 GPU test verified exit code zero and removal of all five group mmap
files with a 60-second engine timeout and a 90-second container stop grace.
SIGKILL, host failure, or an insufficient stop grace can still leave files in
`/dev/shm`; this is not a crash-recovery mechanism. Never delete a live instance's
shared regions while treating them as stale cache files.
