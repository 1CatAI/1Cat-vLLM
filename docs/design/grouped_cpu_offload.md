# Grouped CPU offload and filesystem compatibility

Equal-block-size hybrid attention/Mamba caches use group-local CPU pools to
avoid charging each offload key for every group's backing tensors. The native
connector retains original KV group IDs, including empty positions for scratch
groups. Group-local slot IDs may repeat because the CPU tensors are disjoint.

## Sparse Mamba checkpoint retention

Flash-Next's `align` Mamba mode materializes one recurrent-state snapshot per
block boundary during prefill. Before sparse retention, the offload tier
stored every boundary for all four Mamba groups: on the measured TP4 layout
that is 29.7 MB of state per token block against 10.24 MB of attention/PLE
data, so 74% of the RAM budget held state snapshots and 16 GiB fit about 1.3
contexts of 64K tokens, while the GPU keeps a single state per request.

The retention policy itself lives in the core prefix cache
(`--prefix-cache-retention-interval`, ported from upstream vLLM in 1Cat
PR #617: `0` keeps the replay boundaries and detected shared-prefix junctions,
`N > 0` adds periodic checkpoints, `None` keeps every boundary). This PR only
consumes it on the offload side:

- Mamba `align` boundary hand-offs carry hashed states only, so the host tier
  stores exactly the states the GPU mask admits; no offload-specific policy.
- Group pools are sized from the same mask (see below) instead of one state
  slot per token slot.
- Junctions discovered in the external tier are propagated: when the host tier
  holds a longer full-attention prefix than a sparse group can serve, the
  connector records `Request.shared_prefix_boundary` (like the GPU prefix
  cache does), so the scheduler ends a chunk there, the mask keeps the state
  and the hand-off offloads it; the next sibling hits after a restart or GPU
  eviction.

### Group pool sizing

Token groups receive `N` slots. Each Mamba group receives the number of states
the retention mask keeps for a request of `mamba_state_slots_reference_tokens`
tokens (default `max_model_len`), plus one junction allowance per request,
multiplied by the number of such requests the token pool holds; `N` is the
largest value whose token pages and state slots fit `cpu_bytes_to_use`. Dense
retention keeps one state slot per token slot, reproducing the previous equal
layout. Workloads dominated by prompts shorter than the reference should lower
the reference so the state pools do not run out before the token pools.

On the measured 16 GiB TP4 layout (784-token blocks, 64K reference):

| retention | token slots | 64K contexts | state slots per group |
|---|---:|---:|---:|
| None (dense) | 107 | 1.3 | 107 |
| 8 blocks (6272) | 280 | 3.3 | 48 |
| 0 (semantic, default) | 390 | 4.6 | 10 |
| 0 with a 16K reference | 326 | 3.9 | 32 |

### Validation history

An earlier revision of this branch carried its own core port of the retention
policy and an offload-only checkpoint stride; both were validated on four
V100s on 2026-09-13 (64K restores at the replay boundary with identical token
IDs, GPU-side and externally discovered junctions served to a third request
from RAM, four distinct 60K-64K contexts restored, dense retention thrashing
the 107-slot pools). Evidence: `/mnt/llm_hfs/builds/qsa-stride-validation-20260912`
and `/mnt/llm_hfs/builds/qsa-retention-validation-20260913`. That core port was
dropped in favor of PR #617. The converged implementation at `eef147cafd`
was revalidated on 2026-09-13 with #617's `a9ab97a755` core.

| Check | MTP0 | MTP3 |
|---|---|---|
| Token slots / state slots per group | 390 / 10 | 352 / 10 |
| 63,999-token prompt: first RAM restore | 63,504 tokens, 1.044 s | 62,400 tokens, 1.402 s |
| GPU-discovered junction restored from RAM | 39,200 tokens | 39,200 tokens |
| Externally discovered junction restored from RAM | 39,200 tokens | 39,200 tokens |
| Four distinct contexts restored | 4 x 63,999; each hits 63,504 | 4 x 59,999; each hits 58,400 |
| Four-context restore request time | 1.103-1.542 s | 1.435-1.448 s |

All restore checks reset GPU prefix state first and report zero local hits.
Restored output token IDs match their cold controls. MTP3 draft/accept counters
are nonzero. Each worker allocates approximately 3.99 GiB of pinned cache;
these are preallocated pools, not one-context minimum RAM. Independent CPU
regression on the converged source: **300 passed, 2 skipped**.

One old probe assertion required MTP3's second restore to match its first hit
exactly. The first restore instead materializes the longer Attention boundary,
so the second hits 63,200 rather than 62,400 tokens. All four newly stored Mamba
keys were confirmed as hits on that second request, with identical output IDs.
The original failed assertion and evidence-backed recheck are both retained.
MTP3 junction probes use distinct long tails so the earlier replay checkpoint
cannot substitute for a missing shared-prefix state. An initial tail fixture
misplaced the output instruction; its uncached control failed before any
restore. Corrected inputs passed both junction scenarios.

This is a joint-source regression: 18 hash-verified Python files over the
verified `b8aa829785` image, plus two diagnostic wrappers. It is not a clean
complete-image acceptance of `eef147cafd`. No engine/CUDA/OOM errors occurred
in either test run. Sparse-retention filesystem restart, positive-interval GPU
coverage, and a complete rebuilt image remain outside this round's scope.
Evidence: `/mnt/llm_hfs/builds/pr598-on617-validation-20260913`.

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

### Concurrent registration and long-term page migration

A subsequent TP4/MTP3 serving reproduction with a 32 GiB total RAM tier
identified a concrete failure mechanism. Kernel tracing showed
`pin_user_pages()` returning `-ENOMEM` after `MR_LONGTERM_PIN` migration
failed (0 succeeded / 71 failed pages on one worker, then 40 / 31 on another).
Other workers subsequently migrated the remaining pages and registered the
same shared region successfully. No new cgroup OOM kill occurred in this run;
this `ENOMEM` was a migration failure, not evidence that RAM was exhausted.

`pin_mmap_region` now holds an exclusive `flock` on the region's independently
opened file descriptor while calling `cudaHostRegister`. This serializes
registration of the same backing file across workers, allowing page migration
to finish before another worker pins it. The lock is released in `finally`,
including when registration raises. KV transfers and inference do not hold
the lock. Registration errors remain fatal; there is no retry or pageable
fallback. Tests use independent file descriptors to verify contention during
registration and release on success and exceptions.

The updated stacked PR branch passed 302 CPU tests with two skips using the
existing CPU-admission and legacy unaligned-FS fixtures, including
`tests/v1/core/test_mamba_sparse_retention.py`. The grouped mmap/spawn tests
retain real direct I/O. This CPU run is distinct from the GPU integration
results below.

Two diagnostic starts changing only registration order and two starts of the
formal image with the equivalent source fix passed registration and CUDA
Graph capture on four V100s. The formal image was built from integration
`3fcc73b208` (upstream `7217bb5d4f` plus #624/#617/#598 and this fix), with
matching Python hashes and previously rebuilt, verified native extensions.
This is integration evidence, not a separate native rebuild of every stacked
PR revision.

| Check (TP4, MTP3, 32 GiB total RAM, FS, FP16 KV) | Result |
|---|---|
| Diagnostic cold prompt, 63,999 tokens | 28.823 s; correct marker |
| GPU prefix reset, same-process RAM restore | 62,400 external tokens, zero local; 1.359 s |
| Two concurrent cold prompts, 170,000 tokens each | Both correct; 233.145 / 233.103 s |
| Diagnostic fresh-process FS restore | 63,200 external tokens, zero local; 2.602 s |
| Formal image, same-version fresh-process FS restore | 62,400 external tokens, zero local; 2.967 s |

Restored output token IDs matched the relevant cold controls. There were no
preemptions or new cgroup OOM kills. The long-prompt timings are dominated by
prefill and are not decode throughput measurements. Positive retention
intervals and arbitrary model/backend combinations are not covered by this
follow-up.

## Reproducible block keys across restarts

Set a fixed `PYTHONHASHSEED` (for example, `PYTHONHASHSEED=0`) before starting
both the writer and reader engines, and retain the same prefix hashing
algorithm. vLLM initializes the prefix chain's first hash from random bytes when
this variable is absent. Matching file layout metadata alone therefore cannot
produce restart hits: the same prompt will have different block keys. Use the
existing seed configuration; do not replace vLLM's hashing algorithm.

The physical layout's configuration hash also incorporates the vLLM version.
Changing the engine version therefore selects a separate FS namespace. In the
above validation, the formal image correctly cold-computed a prompt stored by
the older diagnostic version; its own subsequent same-version restart hit.
Do not remove this compatibility boundary or rename directories to force
cross-version reuse.

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

A later serving test also encountered an independent RAM OOM caused by
approximately 93 GiB of stale host-IPC mappings accumulated across stopped
instances. Docker's per-container `OOMKilled=false` did not rule out a worker
being killed in the enclosing LXC cgroup. Use parent-cgroup memory events and
kernel logs when diagnosing this situation.

For a single-container deployment, private IPC with sufficient shared-memory
capacity isolates these mappings to the container lifetime. The 32 GiB RAM
offload test used a 128 GiB private `/dev/shm` ceiling to accommodate additional
model host allocations; the ceiling is not a preallocation or a universal
recommendation. Repeated exits reclaimed the private mappings. This deployment
mitigation is separate from the registration code fix and does not make
host-IPC mappings crash-safe.
