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

The reader owns both shared tensors for its entire lifetime. Keeping only a
flag address is insufficient: registration messages are released after startup,
which otherwise unmaps the flag while the reader still polls it.

Each layer and DP rank owns a bounded cache keyed by all raw ngram tokens.
Lookup and cache fill run on the same CPU worker thread. Cached rows own their
bytes, so staging-buffer reuse cannot mutate them. Reordered/recycled requests
reuse an entry only if its complete key matches. A different result for an
existing key is a byte-integrity failure. Cache misses perform the original
lookup; no row is approximated or omitted. Capacity is twice the engine's
maximum concurrent requests, not the embedding table size.

## Validation status

Four targeted CPU tests cover mailbox lifetime, torn snapshots, skipped generations, byte
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

A normal source-complete wheel has been built. The C1 control with prefetch
disabled passed all 36 task cases, including the 258K needle, with natural EOS
and no replacement characters. Its six-run median decode time was 11.075 ms.
The first enabled run exposed a mailbox flag lifetime bug: the CPU lookup
process exited with SIGSEGV and GPU workers waited during startup. A regression
test fails before the ownership fix and passes after it. This failed run is not
a speed or quality result.

The rebuilt installed wheel keeps all 16 native library hashes unchanged and
passes startup. With sampled prefetch enabled, six-run median C1 decode time
is 10.895 ms versus 11.075 ms disabled: 0.180 ms, or 1.62%, lower. This is an
isolated prefetch delta; dense precision remains unchanged.

The enabled task run passes 35/36 cases. MBPP-9 reaches the 4096-token cap while
still thinking and repeats several lines; its final code is missing. The other
cases, including the 258K needle, pass. The disabled control passes 36/36 with
natural EOS. The enabled path therefore fails the task/health gate and is not
ready to merge. Different sampled continuations alone are not rejection grounds.
A focused C1 teacher-forcing comparison checks MBPP-9, a 32K window and English
before deciding whether byte transport, distribution or sampling needs further
investigation. It is diagnostic, not a replacement for the full C1 gate.

That focused diagnostic completes 144 teacher-forcing positions (three repeats
of MBPP-9, a 32K needle window and English). Both forward/reverse KL and maximum
raw/centered logit error are zero; top-1 agreement is 100%. Default-repeat noise
is also zero. These particular prefixes show no distribution change.

Isolated MBPP-9 repeats retain its original seed 4207. Both arms pass three
times, stop naturally at 758 output tokens and produce identical continuations.
This does not erase the full-suite failure. A prefix reproduction now includes
the original 8K timing requests and preceding MBPP cases, then repeats MBPP-9.
The diagnostic modes preserve original case indices/seeds and label their
reports as diagnostic-only; ordinary suite execution is unchanged.

Teacher-forcing/task quality and matched timing are accepted at C1 only.
One short C4 end-to-end smoke is required before merging.
No end-to-end speedup is claimed yet. This change depends on the mapped result
transport in #831. Dense 8-bit and MTP are separate work.

The load/store distinction follows the
[NVIDIA CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html):
naturally aligned scalar loads/stores in mapped memory do not require host
native read/modify/write atomics. This kernel uses no host-memory RMW operation.
