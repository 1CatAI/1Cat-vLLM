# Flash-Next abliterated deployment on 9683

This deployment reuses the source and model contract of the
[qualified local deployment](flashnext_abliterated_local_20261014.md) on the
9683 host. It combines integration `6f3557c973`, decode `46f5c8c197` and the
qualified prefill implementation `3faaa58d08`. The copied source package is
verified against 2862 source/native file hashes. Native components are the
normal CMake build described in the local report, loaded from the deployed
package through their normal loaders; no private kernel overlay or preload is
required. The shared remote Python environment is not modified.

## Runtime and model contract

- Ubuntu 24.04, Python 3.12.3, Torch 2.10.0+cu128, CUDA toolkit 12.8.
- Four V100-SXM2-16GB GPUs with NVLink, physical GPUs 0–3, TP4, 46 GiB RAM.
- SC117 Flash-Next GSQ-RCO abliterated IQ3_S, revision
  `eec4e10a2c29b440a2ae85591fa11569ce752963`.
- 262144 total input/output-token capacity, 8192-token chunks, one active
  sequence; thinking, prefix caching and Qwen3-Coder tools enabled.
- T0.6, top-p0.95, top-k20; matching stock BF16 MTP with the existing Q8_0
  preparation route, greedy MTP4 and FULL M5 target graphs.
- Experts retain 80% GPU / 20% host placement. PLE remains on disk, target
  history remains host E4M3 with encoded HOT32768, draft FP16 HOT16384.
- `QSA_HOST_INDEXER_HISTORY=true` also places compressed indexer history on
  host, freeing about 230 MiB device memory. The launcher default remains
  `false` for the separately measured local deployment.
- Private runtime caches, GPU ownership locks and a process-group memory guard
  are retained. The guard requires 5 GiB host memory available, matching the
  existing host's memory-protection threshold.

Correctness support and fallbacks remain those of the existing implementation.
This qualification does not cover concurrent execution, other TP sizes or
quantization tiers, or every context length within the configured capacity.

## Download and public-service transition

The new first shard is downloaded directly on the destination from
`https://hf-mirror.com`, with real DNS and sockets bound to the physical NIC.
The host also has a transparent proxy; clearing proxy environment variables
alone is insufficient. The physical socket endpoints are retained as evidence.
The attempted machine-to-machine weight transfer is stopped at the user's
request. The final first-shard checksum is
`40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c`.

The second shard is byte-identical to the existing stock shard, verified as
`316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.
It is retained through an owned hard link without another download or disk
allocation. The new first shard uses the second disk because the root disk is
nearly full. Existing model weights and source remain available for rollback.
The matching compact BF16 MTP checksum remains
`d50d273a87f51d9b7de3142a3f93f2c2823cd36a6fbf0f53f08303ce33029c17`.

The old Strata unit owns this host's public API and GPU locks. It is stopped
only after the replacement model has downloaded and passed its checksum. The
new model uses the existing public tunnel and an authenticated proxy, with the
same credential supplied to the backend. Credentials remain outside Git.
Two duplicate tunnel clients are reduced to one, eliminating their repeated
control-channel disconnects.

## Verification

All shipped native components load through their normal package loaders.
Twenty-one focused GPU cases pass on the destination, covering encoded hot
storage, changed graph inputs, missing-op fallback and misaligned rejection
(22 unaffected cases deselected). The existing local qualification retains its
81-case evidence separately.

The first configuration passes arithmetic and tools but runs out of device
memory on its first long prefill: PLE convolution needs a 144 MiB output
allocation with about 92 MiB free. The existing host-indexer setting addresses
this without reducing context capacity, expert placement, MTP depth or the
8192-token chunk. Three focused cache-budget tests pass, including the existing
host-indexer pool case. The first cold startup also exceeds the controller's
20-minute readiness timeout; its model process continues compiling normally
and completes without being restarted by that timeout. These failed attempts
and their raw logs are retained.

With host indexer history enabled, arithmetic, automatic tool calls and replies
pass with natural EOS. Cold-prefix prefill uses the same 32000-word/seed-9683
contract, two warmups and three measured trials, early unique nonces and no
profiler. Actual measured inputs are 34688, 34606 and 34631 tokens. Median TTFT
is **6.944 s / 4987.2 input tokens/s**. After applying the existing host's CPU
isolation pattern only to owned API/worker descendants, a matching repeat gives
**6.945 s / 4982.5 tokens/s**: no measurable prefill gain is claimed. These
measurements are close to, but do not establish the strict 5000-tokens/s target.

The third identical 32774-input prefix request proves 31824 cached tokens;
total request time falls from 11.476/10.886 s to 4.621 s. Retrieval/arithmetic
tasks at 32761, 131070 and 260977 input tokens pass answer, EOS and repetition
checks (196, 199 and 210 output tokens). The largest completes with 261187
total tokens in 310.776 s, without OOM or a memory-guard intervention under
the final configuration. These are task/capacity checks, not cold throughput
baselines at each length.

Public authentication rejects an unauthenticated model query; authenticated
discovery and streamed arithmetic succeed with the correct 1517 and EOS.
The initial external generation queues behind the long-context check and
takes 49.689 s to its first chunk; after the owned test queue is empty, the
same short check reaches its first streamed chunk in **1.050 s**, completing
in 1.341 s. First HTTP/SSE chunks, first generated tokens and first visible
answer text remain separate concepts. Single active-sequence capacity means
simultaneous Agent requests can still queue.

The new model and proxy units are enabled for startup; the old Strata unit is
disabled with its definition and prior enabled state retained for rollback.
The CPU isolation helper is retained in the deployment wrapper for subsequent
starts. Production development cache-reset endpoints remain disabled.

The external `flashnext-abliterated-9683-20261014` report retains checksums,
source and native identity, failed/abandoned attempts, GPU tests, runtime logs
and request responses. Use its handoff for the live service paths and endpoint.
