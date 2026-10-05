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
