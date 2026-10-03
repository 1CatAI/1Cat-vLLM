# Advisory PLE lookup after sampling

The sampled token completes the raw ngram key before the next model forward.
Publish that key into a small registered host mailbox immediately after
sampling, before output copying and request-state postprocessing. The existing
CPU offload worker performs the lookup while the scheduler and early layers
advance. Ngram weights remain disk-backed mmap.

The kernel accepts integer history, request mapping, sequence length, sampled
ID and accepted count. Incomplete prefill and counts other than one publish no
valid row. The caller trims graph padding to actual requests. Capability
admission requires a local sampler, immutable independent lookup rows, CUDA
SM70 or later, mapped host storage and system atomics. Speculative and hybrid
local lookup retain their existing paths. There is no model size or TP-count
predicate.

`KernelConfig.ple_sample_prefetch` controls the advisory path. The existing
per-layer transport report records the producer binding and registered key
bytes. Only TP root publishes, while the result transport still fans out to
all consumers. This code does not change floating-point model arithmetic.

## Coherent publication and exact reuse

The single producer publishes an odd generation, atomically writes each
payload word, then publishes an even generation with system release ordering.
The CPU reads words atomically between two generation checks. Changed or odd
generations are discarded. The GPU never waits for acknowledgement; skipped
publications use normal lookup.

Each layer and DP rank owns a bounded cache keyed by all raw ngram tokens.
Lookup and cache fill run on the same CPU worker thread. Cached rows own their
bytes, so staging-buffer reuse cannot mutate them. Reordered/recycled requests
reuse an entry only if its complete key matches. A different result for an
existing key is a byte-integrity failure. Cache misses perform the original
lookup; no row is approximated or omitted. Capacity is twice the engine's
maximum concurrent requests, not the embedding table size.

## Validation status

Three targeted CPU tests cover torn snapshots, skipped generations, byte
ownership, eviction, request reordering, wrong-byte detection, cache hits and
normal-lookup fallback. Fifty related CPU tests passed. The SM70 publication
kernel compiles without allocating GPU work and emitted PTX includes system
release stores, system fences and CTA barriers.

The initial four-GPU oracle published 200 changing-width updates per rank.
The CUDA prototype and repository Triton version both had zero wrong keys;
the latter observed 197 coherent snapshots per rank. Lost snapshots are allowed
because publication is advisory. Capture streams are explicit per device; the
first multi-GPU test reused a default stream and captured empty graphs after
rank zero. That was corrected before taking these results. The expanded M1–16
oracle also passed, with 200 publications and 199 coherent snapshots per rank,
zero wrong keys.

A normal source-complete wheel has been built. Installed-runtime integration,
teacher-forcing/task quality and matched C1/C4/C8/C16 timing remain pending.
No end-to-end speedup is claimed yet. This change depends on the mapped result
transport in #831. Dense 8-bit and MTP are separate work.

The load/store distinction follows the
[NVIDIA CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html):
naturally aligned scalar loads/stores in mapped memory do not require host
native read/modify/write atomics. This kernel uses no host-memory RMW operation.
