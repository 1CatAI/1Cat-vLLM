# Grouped CPU offload and filesystem compatibility

Equal-block-size hybrid attention/Mamba caches use group-local CPU pools to
avoid charging each offload key for every group's backing tensors. The native
connector retains original KV group IDs, including empty positions for scratch
groups. Group-local slot IDs may repeat because the CPU tensors are disjoint.

## Sparse Mamba checkpoint retention

Flash-Next's `align` Mamba mode materializes one recurrent-state snapshot per
chunk end during prefill. The fork used to end a chunk at every 784-token
block boundary and the offload tier stored every boundary for all four Mamba
groups. On the measured TP4 layout that is 29.7 MB of state per token block
against 10.24 MB of attention/PLE data, so 74% of the RAM budget held state
snapshots and 16 GiB fit about 1.3 contexts of 64K tokens, while the GPU keeps
a single state per request.

This branch ports upstream vLLM's retention policy instead of adding an
offload-only stride (upstream #43447, #45845, #37898, #52216, #54713, and the
store-side wiring of #51886/#54362):

- `--prefix-cache-retention-interval` (`CacheConfig`, default `0`) decides
  which sliding-window tails and Mamba states stay in the prefix cache.
  `0` keeps only semantic checkpoints: the replay boundary of each prompt
  (with the EAGLE/MTP shift) and detected shared-prefix junctions. `N > 0`
  additionally keeps one checkpoint per `N` tokens (a multiple of the cache-hit
  alignment). `None` keeps every boundary, the previous behavior.
- `Request.shared_prefix_boundary` is the Marconi-style junction: when a
  request's attention groups hit a longer prefix than its Mamba groups, the
  hybrid coordinator reports the uncached common prefix, the scheduler records
  it on the request, ends a prefill chunk there so the state materializes, and
  the mask keeps it. The third request sharing that prefix hits it.
- `SingleTypeKVCacheManager.reachable_block_mask` is the single source of
  truth: the coordinator passes it to `cache_full_blocks` on the GPU and the
  offloading scheduler applies the same mask when choosing which blocks to
  store, so host retention follows GPU retention exactly. Mamba `align`
  boundary hand-offs only carry hashed states, which the mask already filtered.
- `_mamba_block_aligned_split` ends chunks only where a state must exist:
  every block boundary under dense retention, every interval boundary under
  periodic retention, and always the replay boundary and the junction.
  With the default policy a 64K prefill runs in 8K chunks instead of 784-token
  steps.
- Unhashed blocks (running state, masked-out checkpoints, SWA scratch) are
  returned to the front of the free queue so a request's own churn does not
  flush older cached prefixes.

### Group pool sizing

Group pools are sized from the same mask. Token groups receive `N` slots. Each
Mamba group receives the number of states the retention mask keeps for a
request of `mamba_state_slots_reference_tokens` tokens (default
`max_model_len`), plus one junction allowance per request, multiplied by the
number of such requests the token pool holds; `N` is the largest value whose
token pages and state slots fit `cpu_bytes_to_use`. Dense retention keeps one
state slot per token slot, reproducing the previous equal layout. Workloads
dominated by prompts shorter than the reference should lower the reference so
the state pools do not run out before the token pools.

On the measured 16 GiB TP4 layout (784-token blocks, 64K reference):

| retention | token slots | 64K contexts | state slots per group |
|---|---:|---:|---:|
| None (dense) | 107 | 1.3 | 107 |
| 8 blocks (6272) | 280 | 3.3 | 48 |
| 0 (semantic, default) | 390 | 4.6 | 10 |
| 0 with a 16K reference | 326 | 3.9 | 32 |

Requests still pay for their state only where the mask keeps it: a 64K
request stores one Mamba state per group under the default policy instead of
84, and a partial-prefix reuse costs at most one recompute of the shared
prefix before its junction state exists.

### Validation history

The stride prototype (superseded by this retention port) was validated on
four V100s on 2026-09-13: with 8-block sparsity MTP0 and MTP3 restored 64K
prompts at their final boundary with identical token IDs, partial prefixes
rounded down to the stride, and three distinct 60K-64K contexts restored from
RAM, while dense retention thrashed the 107-slot pools. Evidence:
`/mnt/llm_hfs/builds/qsa-stride-validation-20260912`. The retention port
replaces the stride with the upstream policy; its own V100 validation is
recorded below when available.

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
