# TP4 HC with direct NVLink LL forwarding

HC down and up use one kernel each and keep one quarter of each batch-packed
weight per rank. Each rank publishes Half values with a 16-bit generation tag
in the same 32-bit word. Only the two verified direct NVLink neighbors are
accessed; one neighbor forwards the diagonal peer data. Unsupported topologies
retain the existing projection route and report the rejection reason.

## Same-machine operator measurements

On four concurrently active V100s, Flash-Next real HC weights, Torch 2.10,
CUDA 12.8, M5 and CUDA graphs, the supplied route improves 26.22 to 18.84 us
per HC. Eight real weight pairs repeated three times on all ranks give lora
relative maximum error 0.0000540, injection zero and mixed output 0.000419.
The machine has two SYS diagonals; the actual protocol uses direct NVLink
neighbors and therefore does not require a full NVLink mesh.

Sequential eight-token groups are rejected at M20: 54.50 us versus 32.05 us
for replicated HC, a 2.20 ms penalty across 98 operations. Parallel groups
instead share each launch, with independent token rows and one common tag
advance after every CTA completes. This candidate gives M5 19.62 versus
26.22 us and M20 28.58 versus 31.93 us. M20 lora error is 0.000205, injection
zero and mix error 0.000419. Both shapes retain two kernel launches per HC.
These are research JIT measurements. Installed-wheel route and model checks
are required before promotion.

Two alternating receive generations protect the previous operation while a
faster rank begins its next down phase. Data and tag stay in one word; no
additional transport kernel is added.

Buffers and counters are allocated in the communicator before model graph
capture. Peer handles follow the existing shared-buffer ownership and close
protocol. Rank order is derived from verified physical NVLink peers, not
CUDA ordinal assumptions. Runtime capability reports include the rank order,
peer matrix and failure reason. Batch weights are packed during loading;
the replicated batch pack is not retained for admitted layers.

The supplied shard3 implementation provides the arithmetic and LL protocol.
The parallel token-group extension preserves each group's FP32 reduction
and FP16 materialization boundaries. The supplied hc_one single-kernel
combine/norm variant is excluded.

## Tag reuse across changing batches

Alternating pages alone do not protect rows that have been inactive for a
complete tag cycle. A reproduced M20, long M5 interval, then M20 case with
a 5ms rank-0 delay reads old values whose 16-bit tags match the new operation.
Skipping complete modulo cycles preserves the same inactive receive state
without executing thousands of redundant operations. Mix relative maximum
errors reach 0.976–1.415 while injection remains exact.

Both existing kernels now invalidate inactive rows of their local current
receive page. They leave the next page untouched because a faster peer may
already be writing it. No extra launch or system fence is introduced. All
ranks use the same batch shape, so active peer writes cannot race these
inactive-row stores. Completion counters publish only after those stores
finish. M20 has no inactive served rows to clear.

The same regression passes in the complete installed extension with mix
relative maximum 0.000213 on all four ranks and exact injection. Standard
M1/M5/M8/M20 changed-generation and delayed-rank checks also pass; maximum
mix difference is below 0.000430. Amortized four-process graph times on two
real HC pairs are 17.11/17.41/18.93/30.82us respectively. These use a different
contract from the eight-pair research timings above; they are not subtracted
from those earlier results.

The joint dense/HC whole-model comparison in
[the segment design](gguf_dense_segments.md) reports the earlier extension
explicitly. Guarded HC performance is checked against the packaged replicated
operator in the same four-process ABBA benchmark before promotion.

## Packaged same-process ABBA and model gate

The guarded extension is measured against its packaged replicated operator
on all four GPUs, with two real pairs and 16 operations per graph. M5 is
23.72–24.02us versus 17.40–17.44us; M8 is 24.22–24.61us versus
18.88–18.89us. M20 is 29.76–30.19us versus 30.82–30.83us: this shape is a
0.6–1.1us operator regression, not an operator speedup. An existing four-warp
up variant gives 30.55–30.60us at M20 and remains slower than the control;
the five-warp default is retained. The admitted path keeps a single sharded
batch pack, saves replicated weight storage, and avoids rebuilding it during
M20 graph execution. This is a memory/performance tradeoff at M20.

The joint dense/HC installed-wheel comparison reports C1 pooled mean
23.301→22.107ms and C4 mean 55.110→51.008ms. C4 median is
54.581→50.830ms. It uses the extension before the inactive-row guard, which
has separately passed the numerical and ABBA checks above. These are combined
route results, not an HC-only end-to-end delta. Teacher forcing has 63/64 top-1
agreement and mean KL 0.001010. Eight natural prompts show acceptance
44.320→44.802%, paired difference confidence interval [-0.724,+1.634]
percentage points. Two bounded natural prompts terminate identically.

A new guarded-extension trace confirms 96 down and 96 up kernels per target
verification, 1,263 target kernels total, and direct-peer exchanges on the
partially connected topology. GPU target entry skew remains p50 0.586ms and
p90 0.663ms; the host synchronization target is still open. Trace timings
are diagnostic and are not substituted for the unprofiled model result.
