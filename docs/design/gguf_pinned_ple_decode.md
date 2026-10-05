# Packed GGUF PLE decode from pinned memory

The Flash-Next GGUF PLE table uses IQ4_NL rows with width 160. A complete
320,001,536-row table occupies 28,800,138,240 bytes. TP4 stores a quarter of
these original packed bytes on each worker: 7,200,034,560 bytes per rank.
No floating-point table copy or second packed bank is retained.

## Placement and dispatch

The loader resolves metadata and host capacity before constructing the model.
All local TP workers agree on admission through their CPU process group.
Admission requires SM70, FP16 embedding output, local TP4, one IQ4_NL table
with complete 160-value rows, and at most four scheduled requests. Available
host memory minus the reserve is divided among local workers. An explicit
host budget is also respected. Existing disk-cascade placement takes
precedence.

The route additionally requires Model Runner V2, dual compilation and FULL
decode CUDA graphs. During decode the prepared PLE layer reads its rank-local
pinned table and the connector submits no CPU lookup request. Dynamic prefill
continues to consume the offload worker's result. Both decisions belong to the
prepared layer; changing the current configuration for the MTP model does not
change that ownership. Dummy capture requests retain the existing semaphore
protocol.

Pinned allocation and checkpoint copying temporarily restrict the loading
thread to allowed CPUs on the GPU's nearest NUMA node. The original affinity
is restored afterwards, including on an allocation or loading exception.
Unknown or restricted topology retains the existing placement.

`KernelConfig.ple_pinned_decode` defaults to enabled. The acceleration report
includes the source family, calibrated M interval, packed bytes per rank and
an explicit rejection reason. The resolved active state participates in the
compilation hash because it changes graph topology; diagnostic status does not.

## Row decoder and correctness

One Triton launch gathers and decodes the requested rows. The original FP16
block scale is reconstructed from its bytes. Each 4-bit index selects the
IQ4_NL integer codebook value; FP32 multiplication followed by FP16 storage
matches official GGUF dequantization followed by FP16 conversion. There is no
activation quantization or extra numerical approximation.

The codebook and pinned mappings are prepared before graph capture. Startup
checks the first, middle and final physical row against official dequantization.
M=1/5/20 tests compare every output value exactly, including negative scales,
zeros and FP16 subnormal scales, then replay graphs with changed indices.
The installed artifact passes 17 admission, ownership and affinity checks,
and three GPU row/graph checks.

## NUMA operator measurements

Measurements use four concurrent V100 workers, CUDA 12.8, Torch 2.10.0 and
FP16 output. Two complete pinned copies permit a direct comparison: all
workers read the first NUMA node, or each GPU pair reads its local node.
Physical page placement is recorded from `numa_maps`. Production retains
only the four quarter-table shards.

The changing-index test uses n-gram rows from eight real generated sequences.
Each captured graph traverses thousands of different lookups rather than
repeating one tiny cached row set. Capture is serial per device, followed by
concurrent replay and an ABBA comparison.

| Tokens | Single-node lookup range | Local-node lookup range |
| --- | --- | --- |
| 1 | 5.219–5.628 µs | 5.199–5.243 µs |
| 5 | 10.335–12.723 µs | 10.294–10.510 µs |
| 20 | 25.362–25.959 µs | 25.369–25.835 µs |

These are lookup service times, not round latency. One early M=5 single-node
sample is slower than the remaining samples. NUMA locality has a small effect
in this measurement; removing CPU request submission and waiting is the main
model-level hypothesis. A fixed-index warm test reached approximately 4 µs
at M=5, but that number is not a cold-row or end-to-end claim.

`benchmarks/kernels/benchmark_gguf_pinned_ple_numa.py` accepts the packed-table
GGUF, metadata GGUF, saved acceptance report and output filename. Run it under
the shared GPU ownership locks. It allocates two full table copies and needs
at least twice the packed table size in available host memory.

## Model measurements

A same-wheel comparison with only `ple_pinned_decode` changed is in progress.
It uses Flash-Next IQ3_S, FP16 MTP4, TP4, FP16 KV cache, FP32 SSM state,
FULL target graphs, 8K C1 input and a four-request C4 cohort. Teacher-forcing
prefixes are fixed and saved explicitly. Operator service times do not justify
promotion before the round, completion and acceptance checks finish.
